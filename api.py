import os
import uuid
import json
import datetime
import logging
import random
from typing import Optional, List

from dotenv import load_dotenv

_env_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_env_dir, ".env"))
load_dotenv(os.path.join(_env_dir, ".env.local"), override=True)

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from database import db
from auth_utils import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    get_optional_user,
)
from ai_utils import AiServiceError, describe_product_from_image
from stripe_utils import (
    StripeNotConfiguredError,
    create_checkout_session,
    construct_webhook_event,
    is_stripe_configured,
    build_mock_payment_url,
)
from order_fulfillment import fulfill_paid_order

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CebolaAPI")

# Initialize FastAPI App
app = FastAPI(title="Cebola Backend API")

_cors_raw = os.environ.get("CEBOLA_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
_allow_credentials = "*" not in _cors_origins
if "*" in _cors_origins:
    _cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
MAX_UPLOAD_SIZE = 5 * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/api/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# --- REQUEST SCHEMAS ---

class LoginPayload(BaseModel):
    email: str
    password: str

class RegisterPayload(BaseModel):
    email: str
    password: str
    name: str

class UpdateProfilePayload(BaseModel):
    name: str

class ChangePasswordPayload(BaseModel):
    currentPassword: str
    newPassword: str

class ShopFormValues(BaseModel):
    name: str
    description: str
    logoUrl: Optional[str] = None
    isActive: bool = True

class ProductFormValues(BaseModel):
    title: str
    description: str
    price: float
    quantity: int
    photoUrl: Optional[str] = None
    isAvailable: bool = True

class OrderStatusUpdate(BaseModel):
    status: str

class CustomerOrderItemPayload(BaseModel):
    productId: str
    quantity: int

class CustomerOrderPayload(BaseModel):
    shopId: str
    items: List[CustomerOrderItemPayload]

class SubscribePayload(BaseModel):
    planId: str

class RatingPayload(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None


def verify_shop_ownership(shop_id: str, user_id: str) -> dict:
    shop = db.execute_one("SELECT * FROM shops WHERE id = %s", (shop_id,))
    if not shop:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shop not found")
    if shop["owner_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")
    return shop

def get_shop_rating_stats(shop_id: str) -> dict:
    row = db.execute_one(
        "SELECT COUNT(*) as count, AVG(rating) as avg_rating FROM shop_ratings WHERE shop_id = %s",
        (shop_id,)
    )
    count = int(row.get("count", 0) or 0) if row else 0
    avg = row.get("avg_rating") if row else None
    return {
        "avgRating": round(float(avg), 2) if avg is not None else None,
        "ratingCount": count
    }

# --- RESPONSE FORMATTING HELPERS ---

def format_user(row: dict) -> Optional[dict]:
    if not row:
        return None
    order_ids = []
    if row.get("order_ids"):
        try:
            order_ids = json.loads(row["order_ids"])
        except json.JSONDecodeError:
            order_ids = []
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "orderIds": order_ids,
        "createdAt": row["created_at"]
    }

def format_shop(row: dict, include_rating: bool = False) -> Optional[dict]:
    if not row:
        return None
    result = {
        "id": row["id"],
        "ownerId": row["owner_id"],
        "name": row["name"],
        "description": row["description"],
        "logoUrl": row["logo_url"],
        "isActive": bool(row["is_active"]),
        "createdAt": row["created_at"]
    }
    if include_rating:
        result.update(get_shop_rating_stats(row["id"]))
    return result

def format_product(row: dict) -> Optional[dict]:
    if not row:
        return None
    return {
        "id": row["id"],
        "shopId": row["shop_id"],
        "title": row["title"],
        "description": row["description"],
        "price": float(row["price"]),
        "quantity": int(row["quantity"]),
        "photoUrl": row["photo_url"],
        "isAvailable": bool(row["is_available"]),
        "createdAt": row["created_at"]
    }

def format_order(row: dict, items: Optional[List[dict]] = None) -> Optional[dict]:
    if not row:
        return None
    return {
        "id": row["id"],
        "shopId": row["shop_id"],
        "userId": row.get("user_id"),
        "guestOrderId": row.get("guest_order_id"),
        "customerName": row.get("customer_name"),
        "customerEmail": row.get("customer_email"),
        "customerPhone": row.get("customer_phone"),
        "totalAmount": float(row["total_amount"]),
        "totalQuantity": int(row["total_quantity"]),
        "status": row["status"],
        "qrCodeData": row.get("qr_code_data"),
        "createdAt": row["created_at"],
        "items": items or []
    }

def format_order_item(row: dict) -> Optional[dict]:
    if not row:
        return None
    return {
        "id": row["id"],
        "orderId": row["order_id"],
        "productId": row.get("product_id"),
        "productTitle": row["product_title"],
        "productLogo": row.get("product_logo"),
        "quantity": int(row["quantity"]),
        "priceAtTime": float(row["price_at_time"])
    }

def format_customer_shop(row: dict) -> Optional[dict]:
    if not row:
        return None
    result = {
        "id": row["id"],
        "title": row["name"],
        "description": row["description"],
        "logoUrl": row.get("logo_url"),
        "isAvailable": bool(row["is_active"]),
    }
    result.update(get_shop_rating_stats(row["id"]))
    return result

def format_customer_product(row: dict, shop_name: str) -> Optional[dict]:
    if not row:
        return None
    quantity = int(row["quantity"])
    is_available = bool(row["is_available"]) and quantity > 0
    return {
        "id": row["id"],
        "shopId": row["shop_id"],
        "shopTitle": shop_name,
        "title": row["title"],
        "description": row["description"],
        "price": float(row["price"]),
        "quantity": quantity,
        "photoUrl": row.get("photo_url"),
        "isAvailable": is_available,
    }

def format_customer_order_response(order_id: str, guest_order_id: str, payment_url: Optional[str] = None) -> dict:
    response = {
        "id": order_id,
        "guestOrderId": guest_order_id,
    }
    if payment_url:
        response["paymentUrl"] = payment_url
    return response

def _generate_guest_order_id() -> str:
    return f"{random.randint(10000000, 99999999)}"

def format_subscription_plan(row: dict) -> Optional[dict]:
    if not row:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "price": float(row["price"]),
        "interval": row["interval"],
        "maxProducts": row.get("max_products"),
        "isActive": bool(row["is_active"]),
        "createdAt": row["created_at"]
    }

def format_subscription(row: dict, plan: Optional[dict] = None) -> Optional[dict]:
    if not row:
        return None
    result = {
        "id": row["id"],
        "userId": row["user_id"],
        "planId": row["plan_id"],
        "status": row["status"],
        "startedAt": row["started_at"],
        "expiresAt": row.get("expires_at"),
        "cancelledAt": row.get("cancelled_at"),
        "createdAt": row["created_at"]
    }
    if plan:
        result["plan"] = format_subscription_plan(plan)
    return result

def format_rating(row: dict, user_name: Optional[str] = None) -> Optional[dict]:
    if not row:
        return None
    return {
        "id": row["id"],
        "shopId": row["shop_id"],
        "userId": row["user_id"],
        "userName": user_name,
        "rating": int(row["rating"]),
        "comment": row.get("comment"),
        "createdAt": row["created_at"]
    }

def _subscription_duration_days(interval: str) -> int:
    if interval == "yearly":
        return 365
    return 30

# --- USER AUTHENTICATION ENDPOINTS ---

@app.post("/api/auth/register")
def register(payload: RegisterPayload):
    # Check if email is already taken
    existing_user = db.execute_one("SELECT id FROM users WHERE email = %s", (payload.email.lower(),))
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists"
        )
    
    # Create User
    user_id = f"usr-{uuid.uuid4().hex[:12]}"
    pwd_hash = hash_password(payload.password)
    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    
    db.execute_write(
        "INSERT INTO users (id, email, name, password_hash, created_at) VALUES (%s, %s, %s, %s, %s)",
        (user_id, payload.email.lower(), payload.name, pwd_hash, created_at)
    )
    
    # Retrieve inserted user details
    user = db.execute_one("SELECT id, email, name, created_at FROM users WHERE id = %s", (user_id,))
    token = create_access_token({"sub": user_id})
    
    return {
        "token": token,
        "user": format_user(user)
    }

@app.post("/api/auth/login")
def login(payload: LoginPayload):
    user = db.execute_one("SELECT * FROM users WHERE email = %s", (payload.email.lower(),))
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    token = create_access_token({"sub": user["id"]})
    return {
        "token": token,
        "user": format_user(user)
    }

@app.get("/api/auth/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return format_user(current_user)

@app.put("/api/auth/me")
def update_profile(payload: UpdateProfilePayload, current_user: dict = Depends(get_current_user)):
    db.execute_write(
        "UPDATE users SET name = %s WHERE id = %s",
        (payload.name, current_user["id"])
    )
    updated = db.execute_one("SELECT id, email, name, created_at FROM users WHERE id = %s", (current_user["id"],))
    return format_user(updated)

@app.post("/api/auth/change-password")
def change_password(payload: ChangePasswordPayload, current_user: dict = Depends(get_current_user)):
    full_user = db.execute_one("SELECT password_hash FROM users WHERE id = %s", (current_user["id"],))
    if not full_user or not verify_password(payload.currentPassword, full_user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The current password you entered is incorrect"
        )
    
    new_hash = hash_password(payload.newPassword)
    db.execute_write("UPDATE users SET password_hash = %s WHERE id = %s", (new_hash, current_user["id"]))
    return {"success": True}


# --- SHOP MANAGEMENT ENDPOINTS ---

@app.get("/api/merchant/shops")
def get_shops(
    page: int = 1,
    limit: int = 10,
    q: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    offset = (page - 1) * limit
    owner_id = current_user["id"]
    if q:
        search_pattern = f"%{q.lower()}%"
        count_row = db.execute_one(
            "SELECT COUNT(*) as count FROM shops WHERE owner_id = %s "
            "AND (LOWER(name) LIKE %s OR LOWER(description) LIKE %s)",
            (owner_id, search_pattern, search_pattern),
        )
        total = count_row.get("count", 0) if count_row else 0
        
        rows = db.execute_query(
            "SELECT * FROM shops WHERE owner_id = %s "
            "AND (LOWER(name) LIKE %s OR LOWER(description) LIKE %s) "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (owner_id, search_pattern, search_pattern, limit, offset),
        )
    else:
        count_row = db.execute_one(
            "SELECT COUNT(*) as count FROM shops WHERE owner_id = %s",
            (owner_id,),
        )
        total = count_row.get("count", 0) if count_row else 0
        
        rows = db.execute_query(
            "SELECT * FROM shops WHERE owner_id = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (owner_id, limit, offset),
        )
        
    return {
        "items": [format_shop(row) for row in rows],
        "total": total,
        "page": page,
        "limit": limit
    }

@app.get("/api/merchant/shop")
def get_merchant_default_shop(current_user: dict = Depends(get_current_user)):
    """Returns the first shop owned by the authenticated merchant, or null if none exists."""
    row = db.execute_one(
        "SELECT * FROM shops WHERE owner_id = %s ORDER BY created_at ASC LIMIT 1",
        (current_user["id"],)
    )
    return format_shop(row)

@app.get("/api/merchant/shops/{shopId}")
def get_shop_by_id(shopId: str):
    row = db.execute_one("SELECT * FROM shops WHERE id = %s", (shopId,))
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Shop not found"
        )
    return format_shop(row, include_rating=True)

@app.post("/api/merchant/shop")
def create_shop(payload: ShopFormValues, current_user: dict = Depends(get_current_user)):
    shop_id = f"shop-{uuid.uuid4().hex[:8]}"
    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    
    db.execute_write(
        "INSERT INTO shops (id, owner_id, name, description, logo_url, is_active, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (shop_id, current_user["id"], payload.name, payload.description, payload.logoUrl, payload.isActive, created_at)
    )
    
    row = db.execute_one("SELECT * FROM shops WHERE id = %s", (shop_id,))
    return format_shop(row)

@app.put("/api/merchant/shops/{shopId}")
def update_shop(shopId: str, payload: ShopFormValues, current_user: dict = Depends(get_current_user)):
    # Verify ownership
    shop = db.execute_one("SELECT owner_id FROM shops WHERE id = %s", (shopId,))
    if not shop:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shop not found")
    
    if shop["owner_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to modify this shop"
        )
        
    db.execute_write(
        "UPDATE shops SET name = %s, description = %s, logo_url = %s, is_active = %s WHERE id = %s",
        (payload.name, payload.description, payload.logoUrl, payload.isActive, shopId)
    )
    
    row = db.execute_one("SELECT * FROM shops WHERE id = %s", (shopId,))
    return format_shop(row)


# --- PRODUCT CATALOGUE ENDPOINTS ---

@app.get("/api/merchant/products")
def get_all_merchant_products(current_user: dict = Depends(get_current_user)):
    """Retrieves all products from all shops owned by the merchant."""
    rows = db.execute_query(
        "SELECT p.* FROM products p JOIN shops s ON p.shop_id = s.id "
        "WHERE s.owner_id = %s ORDER BY p.created_at DESC",
        (current_user["id"],)
    )
    return [format_product(row) for row in rows]

@app.get("/api/merchant/shops/{shopId}/products")
def get_shop_products(shopId: str, page: int = 1, limit: int = 100, q: Optional[str] = None):
    offset = (page - 1) * limit
    if q:
        search_pattern = f"%{q.lower()}%"
        count_row = db.execute_one(
            "SELECT COUNT(*) as count FROM products WHERE shop_id = %s AND (LOWER(title) LIKE %s OR LOWER(description) LIKE %s)",
            (shopId, search_pattern, search_pattern)
        )
        total = count_row.get("count", 0) if count_row else 0
        
        rows = db.execute_query(
            "SELECT * FROM products WHERE shop_id = %s AND (LOWER(title) LIKE %s OR LOWER(description) LIKE %s) "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (shopId, search_pattern, search_pattern, limit, offset)
        )
    else:
        count_row = db.execute_one("SELECT COUNT(*) as count FROM products WHERE shop_id = %s", (shopId,))
        total = count_row.get("count", 0) if count_row else 0
        
        rows = db.execute_query(
            "SELECT * FROM products WHERE shop_id = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (shopId, limit, offset)
        )
        
    return {
        "items": [format_product(row) for row in rows],
        "total": total,
        "page": page,
        "limit": limit
    }

@app.get("/api/merchant/shops/{shopId}/products/{productId}")
def get_product_by_id(shopId: str, productId: str):
    row = db.execute_one("SELECT * FROM products WHERE id = %s AND shop_id = %s", (productId, shopId))
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    return format_product(row)

# Dual-route posting: with or without specific shop ID
@app.post("/api/merchant/products")
@app.post("/api/merchant/shops/{shopId}/products")
def create_product(
    payload: ProductFormValues,
    shopId: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    # Resolve target shop
    if not shopId:
        first_shop = db.execute_one(
            "SELECT id FROM shops WHERE owner_id = %s ORDER BY created_at ASC LIMIT 1",
            (current_user["id"],)
        )
        if not first_shop:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please create a merchant shop before creating products"
            )
        target_shop_id = first_shop["id"]
    else:
        # Verify ownership
        shop = db.execute_one("SELECT owner_id FROM shops WHERE id = %s", (shopId,))
        if not shop:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shop not found")
        if shop["owner_id"] != current_user["id"]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")
        target_shop_id = shopId

    product_id = f"prod-{uuid.uuid4().hex[:8]}"
    created_at = datetime.datetime.utcnow().isoformat() + "Z"

    db.execute_write(
        "INSERT INTO products (id, shop_id, title, description, price, quantity, photo_url, is_available, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (product_id, target_shop_id, payload.title, payload.description, payload.price, payload.quantity, payload.photoUrl, payload.isAvailable, created_at)
    )

    row = db.execute_one("SELECT * FROM products WHERE id = %s", (product_id,))
    return format_product(row)

# Dual-route updating: with or without specific shop ID
@app.put("/api/merchant/products/{productId}")
@app.put("/api/merchant/shops/{shopId}/products/{productId}")
def update_product(
    productId: str,
    payload: ProductFormValues,
    shopId: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    # Verify product and ownership
    product = db.execute_one("SELECT shop_id FROM products WHERE id = %s", (productId,))
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    
    shop = db.execute_one("SELECT owner_id FROM shops WHERE id = %s", (product["shop_id"],))
    if not shop or shop["owner_id"] != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")

    db.execute_write(
        "UPDATE products SET title = %s, description = %s, price = %s, quantity = %s, photo_url = %s, is_available = %s WHERE id = %s",
        (payload.title, payload.description, payload.price, payload.quantity, payload.photoUrl, payload.isAvailable, productId)
    )

    row = db.execute_one("SELECT * FROM products WHERE id = %s", (productId,))
    return format_product(row)

# Dual-route deletion: with or without specific shop ID
@app.delete("/api/merchant/products/{productId}")
@app.delete("/api/merchant/shops/{shopId}/products/{productId}")
def delete_product(
    productId: str,
    shopId: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    # Verify product and ownership
    product = db.execute_one("SELECT shop_id FROM products WHERE id = %s", (productId,))
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    
    shop = db.execute_one("SELECT owner_id FROM shops WHERE id = %s", (product["shop_id"],))
    if not shop or shop["owner_id"] != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")

    db.execute_write("DELETE FROM products WHERE id = %s", (productId,))
    return {"success": True}


# --- ORDER MANAGEMENT ENDPOINTS ---

@app.get("/api/merchant/orders")
def get_all_merchant_orders(
    page: int = 1,
    limit: int = 20,
    q: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Retrieves all orders from all shops owned by the merchant, with details."""
    offset = (page - 1) * limit
    
    # Base query finding all shops owned by user
    if q:
        search_pattern = f"%{q.lower()}%"
        count_row = db.execute_one(
            "SELECT COUNT(*) as count FROM orders o JOIN shops s ON o.shop_id = s.id "
            "WHERE s.owner_id = %s AND (LOWER(o.id) LIKE %s OR LOWER(o.guest_order_id) LIKE %s OR LOWER(o.status) LIKE %s)",
            (current_user["id"], search_pattern, search_pattern, search_pattern)
        )
        total = count_row.get("count", 0) if count_row else 0
        
        order_rows = db.execute_query(
            "SELECT o.* FROM orders o JOIN shops s ON o.shop_id = s.id "
            "WHERE s.owner_id = %s AND (LOWER(o.id) LIKE %s OR LOWER(o.guest_order_id) LIKE %s OR LOWER(o.status) LIKE %s) "
            "ORDER BY o.created_at DESC LIMIT %s OFFSET %s",
            (current_user["id"], search_pattern, search_pattern, search_pattern, limit, offset)
        )
    else:
        count_row = db.execute_one(
            "SELECT COUNT(*) as count FROM orders o JOIN shops s ON o.shop_id = s.id WHERE s.owner_id = %s",
            (current_user["id"],)
        )
        total = count_row.get("count", 0) if count_row else 0
        
        order_rows = db.execute_query(
            "SELECT o.* FROM orders o JOIN shops s ON o.shop_id = s.id "
            "WHERE s.owner_id = %s ORDER BY o.created_at DESC LIMIT %s OFFSET %s",
            (current_user["id"], limit, offset)
        )

    # Hydrate each order with its items
    hydrated_orders = []
    for order_row in order_rows:
        item_rows = db.execute_query("SELECT * FROM order_items WHERE order_id = %s", (order_row["id"],))
        items = [format_order_item(item) for item in item_rows]
        hydrated_orders.append(format_order(order_row, items))

    return {
        "items": hydrated_orders,
        "total": total,
        "page": page,
        "limit": limit
    }

@app.get("/api/merchant/shops/{shopId}/orders")
def get_shop_orders(
    shopId: str,
    page: int = 1,
    limit: int = 20,
    q: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    verify_shop_ownership(shopId, current_user["id"])
    offset = (page - 1) * limit
    
    if q:
        search_pattern = f"%{q.lower()}%"
        count_row = db.execute_one(
            "SELECT COUNT(*) as count FROM orders WHERE shop_id = %s "
            "AND (LOWER(id) LIKE %s OR LOWER(guest_order_id) LIKE %s OR LOWER(status) LIKE %s)",
            (shopId, search_pattern, search_pattern, search_pattern)
        )
        total = count_row.get("count", 0) if count_row else 0
        
        order_rows = db.execute_query(
            "SELECT * FROM orders WHERE shop_id = %s "
            "AND (LOWER(id) LIKE %s OR LOWER(guest_order_id) LIKE %s OR LOWER(status) LIKE %s) "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (shopId, search_pattern, search_pattern, search_pattern, limit, offset)
        )
    else:
        count_row = db.execute_one("SELECT COUNT(*) as count FROM orders WHERE shop_id = %s", (shopId,))
        total = count_row.get("count", 0) if count_row else 0
        
        order_rows = db.execute_query(
            "SELECT * FROM orders WHERE shop_id = %s ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (shopId, limit, offset)
        )

    # Hydrate each order with its items
    hydrated_orders = []
    for order_row in order_rows:
        item_rows = db.execute_query("SELECT * FROM order_items WHERE order_id = %s", (order_row["id"],))
        items = [format_order_item(item) for item in item_rows]
        hydrated_orders.append(format_order(order_row, items))

    return {
        "items": hydrated_orders,
        "total": total,
        "page": page,
        "limit": limit
    }

@app.get("/api/merchant/shops/{shopId}/orders/{orderId}")
def get_order_by_id(
    shopId: str,
    orderId: str,
    current_user: dict = Depends(get_current_user),
):
    verify_shop_ownership(shopId, current_user["id"])
    order_row = db.execute_one("SELECT * FROM orders WHERE id = %s AND shop_id = %s", (orderId, shopId))
    if not order_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
        
    item_rows = db.execute_query("SELECT * FROM order_items WHERE order_id = %s", (orderId,))
    items = [format_order_item(item) for item in item_rows]
    return format_order(order_row, items)

@app.put("/api/merchant/shops/{shopId}/orders/{orderId}")
def update_order_status(
    shopId: str,
    orderId: str,
    payload: OrderStatusUpdate,
    current_user: dict = Depends(get_current_user)
):
    # Verify order ownership
    shop = db.execute_one("SELECT owner_id FROM shops WHERE id = %s", (shopId,))
    if not shop or shop["owner_id"] != current_user["id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")

    order_row = db.execute_one("SELECT * FROM orders WHERE id = %s AND shop_id = %s", (orderId, shopId))
    if not order_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")

    db.execute_write(
        "UPDATE orders SET status = %s WHERE id = %s AND shop_id = %s",
        (payload.status, orderId, shopId)
    )

    # Return refreshed and hydrated order
    refreshed_order = db.execute_one("SELECT * FROM orders WHERE id = %s AND shop_id = %s", (orderId, shopId))
    item_rows = db.execute_query("SELECT * FROM order_items WHERE order_id = %s", (orderId,))
    items = [format_order_item(item) for item in item_rows]
    return format_order(refreshed_order, items)


# --- CUSTOMER STOREFRONT ENDPOINTS (PUBLIC) ---

@app.get("/api/shops")
def list_public_shops(page: int = 1, limit: int = 10, q: Optional[str] = None):
    offset = (page - 1) * limit
    if q:
        search_pattern = f"%{q.lower()}%"
        count_row = db.execute_one(
            "SELECT COUNT(*) as count FROM shops WHERE is_active = TRUE "
            "AND (LOWER(name) LIKE %s OR LOWER(description) LIKE %s)",
            (search_pattern, search_pattern),
        )
        rows = db.execute_query(
            "SELECT * FROM shops WHERE is_active = TRUE "
            "AND (LOWER(name) LIKE %s OR LOWER(description) LIKE %s) "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (search_pattern, search_pattern, limit, offset),
        )
    else:
        count_row = db.execute_one("SELECT COUNT(*) as count FROM shops WHERE is_active = TRUE")
        rows = db.execute_query(
            "SELECT * FROM shops WHERE is_active = TRUE ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )

    total = count_row.get("count", 0) if count_row else 0
    return {
        "items": [format_customer_shop(row) for row in rows],
        "total": total,
    }


@app.get("/api/shops/{shopId}")
def get_public_shop(shopId: str):
    row = db.execute_one(
        "SELECT * FROM shops WHERE id = %s AND is_active = TRUE",
        (shopId,),
    )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shop not found")
    return format_customer_shop(row)


@app.get("/api/shops/{shopId}/products")
def list_public_shop_products(
    shopId: str,
    page: int = 1,
    limit: int = 100,
    q: Optional[str] = None,
):
    shop = db.execute_one(
        "SELECT * FROM shops WHERE id = %s AND is_active = TRUE",
        (shopId,),
    )
    if not shop:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shop not found")

    offset = (page - 1) * limit
    if q:
        search_pattern = f"%{q.lower()}%"
        count_row = db.execute_one(
            "SELECT COUNT(*) as count FROM products WHERE shop_id = %s AND is_available = TRUE AND quantity > 0 "
            "AND (LOWER(title) LIKE %s OR LOWER(description) LIKE %s)",
            (shopId, search_pattern, search_pattern),
        )
        rows = db.execute_query(
            "SELECT * FROM products WHERE shop_id = %s AND is_available = TRUE AND quantity > 0 "
            "AND (LOWER(title) LIKE %s OR LOWER(description) LIKE %s) "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (shopId, search_pattern, search_pattern, limit, offset),
        )
    else:
        count_row = db.execute_one(
            "SELECT COUNT(*) as count FROM products WHERE shop_id = %s AND is_available = TRUE AND quantity > 0",
            (shopId,),
        )
        rows = db.execute_query(
            "SELECT * FROM products WHERE shop_id = %s AND is_available = TRUE AND quantity > 0 "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (shopId, limit, offset),
        )

    total = count_row.get("count", 0) if count_row else 0
    shop_name = shop["name"]
    return {
        "items": [format_customer_product(row, shop_name) for row in rows],
        "total": total,
    }


@app.get("/api/shops/{shopId}/products/{productId}")
def get_public_product(shopId: str, productId: str):
    shop = db.execute_one(
        "SELECT * FROM shops WHERE id = %s AND is_active = TRUE",
        (shopId,),
    )
    if not shop:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shop not found")

    row = db.execute_one(
        "SELECT * FROM products WHERE id = %s AND shop_id = %s AND is_available = TRUE AND quantity > 0",
        (productId, shopId),
    )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    return format_customer_product(row, shop["name"])


@app.post("/api/orders")
def create_customer_order(
    payload: CustomerOrderPayload,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    if not payload.items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Order must contain at least one item",
        )

    shop = db.execute_one(
        "SELECT * FROM shops WHERE id = %s AND is_active = TRUE",
        (payload.shopId,),
    )
    if not shop:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shop not found")

    line_items = []
    stripe_items = []
    total_amount = 0.0
    total_quantity = 0

    for item in payload.items:
        if item.quantity < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Each item quantity must be at least 1",
            )

        product = db.execute_one(
            "SELECT * FROM products WHERE id = %s AND shop_id = %s",
            (item.productId, payload.shopId),
        )
        if not product:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Product not found: {item.productId}",
            )
        if not product["is_available"] or int(product["quantity"]) < item.quantity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient stock for product: {product['title']}",
            )

        line_total = float(product["price"]) * item.quantity
        total_amount += line_total
        total_quantity += item.quantity
        line_items.append((product, item.quantity, line_total))
        stripe_items.append({
            "title": product["title"],
            "price": float(product["price"]),
            "quantity": item.quantity,
            "product_id": product["id"],
        })

    order_id = str(uuid.uuid4())
    guest_order_id = _generate_guest_order_id()
    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    user_id = current_user["id"] if current_user else None
    customer_email = current_user["email"] if current_user else None

    db.execute_write(
        "INSERT INTO orders (id, shop_id, user_id, guest_order_id, customer_name, customer_email, "
        "customer_phone, total_amount, total_quantity, status, qr_code_data, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            order_id,
            payload.shopId,
            user_id,
            guest_order_id,
            None,
            customer_email,
            None,
            total_amount,
            total_quantity,
            "pending",
            f"order:{order_id}",
            created_at,
        ),
    )

    for product, quantity, _line_total in line_items:
        item_id = f"item-{uuid.uuid4().hex[:8]}"
        price_at_time = float(product["price"])
        db.execute_write(
            "INSERT INTO order_items (id, order_id, product_id, product_title, product_logo, quantity, price_at_time) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (
                item_id,
                order_id,
                product["id"],
                product["title"],
                product.get("photo_url"),
                quantity,
                price_at_time,
            ),
        )

    try:
        checkout_result = create_checkout_session(
            order_id,
            guest_order_id,
            shop["name"],
            stripe_items,
            customer_email=customer_email,
        )
    except (StripeNotConfiguredError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    payment_url = checkout_result
    if isinstance(checkout_result, tuple):
        payment_url, session_id = checkout_result
        db.execute_write(
            "UPDATE orders SET stripe_session_id = %s WHERE id = %s",
            (session_id, order_id),
        )

    return format_customer_order_response(order_id, guest_order_id, payment_url)


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    try:
        event = construct_webhook_event(payload, signature)
    except StripeNotConfiguredError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    except Exception as exc:
        logger.warning("Stripe webhook verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook signature",
        ) from exc

    event_type = event["type"]
    if event_type == "checkout.session.completed":
        session = event["data"]["object"]
        order_id = session.get("client_reference_id") or session.get("metadata", {}).get("order_id")
        if order_id:
            payment_intent = session.get("payment_intent")
            session_id = session.get("id")
            if payment_intent or session_id:
                db.execute_write(
                    "UPDATE orders SET stripe_payment_intent_id = %s, stripe_session_id = %s WHERE id = %s",
                    (payment_intent, session_id, order_id),
                )
            customer_email = (
                (session.get("customer_details") or {}).get("email")
                or session.get("customer_email")
            )
            fulfill_paid_order(order_id, customer_email=customer_email)
    elif event_type == "checkout.session.expired":
        session = event["data"]["object"]
        order_id = session.get("client_reference_id") or session.get("metadata", {}).get("order_id")
        if order_id:
            order = db.execute_one("SELECT status FROM orders WHERE id = %s", (order_id,))
            if order and order["status"] == "pending":
                db.execute_write(
                    "UPDATE orders SET status = %s WHERE id = %s",
                    ("cancelled", order_id),
                )

    return {"received": True}


@app.api_route("/api/orders/{orderId}/mock-pay", methods=["GET", "POST"])
def mock_pay_order(orderId: str, request: Request):
    if os.environ.get("CEBOLA_ENV") != "development" or is_stripe_configured():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    order = db.execute_one("SELECT * FROM orders WHERE id = %s", (orderId,))
    if not order:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")

    fulfill_paid_order(orderId, customer_email=order.get("customer_email"))

    if request.method == "GET":
        success_url = os.environ.get(
            "STRIPE_SUCCESS_URL",
            "http://localhost:5173/checkout/success?session_id={CHECKOUT_SESSION_ID}",
        ).replace("{CHECKOUT_SESSION_ID}", "mock")
        return RedirectResponse(success_url, status_code=303)

    return {"success": True, "orderId": orderId}


@app.get("/api/orders/{orderId}")
def get_customer_order(orderId: str):
    order_row = db.execute_one(
        "SELECT * FROM orders WHERE id = %s OR guest_order_id = %s",
        (orderId, orderId),
    )
    if not order_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")

    guest_order_id = order_row.get("guest_order_id") or ""
    payment_url = None
    if order_row.get("status") == "pending":
        payment_url = build_mock_payment_url(order_row["id"], guest_order_id)
    return format_customer_order_response(order_row["id"], guest_order_id, payment_url)


# --- UPLOAD & ARTIFICIAL INTELLIGENCE ENDPOINTS ---

@app.post("/api/upload")
def upload_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File type not allowed. Use jpg, jpeg, png, webp, or gif.",
        )

    content = file.file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File exceeds maximum size of 5 MB",
        )

    unique_filename = f"{uuid.uuid4().hex[:16]}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(dest_path, "wb") as buffer:
            buffer.write(content)
        return {"url": f"/api/uploads/{unique_filename}"}
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded image"
        )

@app.post("/api/ai")
def describe_product(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Runs Google Vision OCR, then MiniCPM5 or OCR-based fallback for title/description."""
    try:
        image_bytes = file.file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image is empty",
            )
        return describe_product_from_image(image_bytes)
    except AiServiceError as exc:
        logger.error("AI product description failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected AI failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate product description",
        )


# --- SUBSCRIPTION ENDPOINTS ---

@app.get("/api/subscription/plans")
def list_subscription_plans():
    rows = db.execute_query(
        "SELECT * FROM subscription_plans WHERE is_active = TRUE ORDER BY price ASC"
    )
    return [format_subscription_plan(row) for row in rows]

@app.get("/api/subscription/me")
def get_my_subscription(current_user: dict = Depends(get_current_user)):
    sub = db.execute_one(
        "SELECT * FROM subscriptions WHERE user_id = %s AND status = 'active' "
        "ORDER BY created_at DESC LIMIT 1",
        (current_user["id"],)
    )
    if not sub:
        return None
    plan = db.execute_one("SELECT * FROM subscription_plans WHERE id = %s", (sub["plan_id"],))
    return format_subscription(sub, plan)

@app.post("/api/subscription/subscribe")
def subscribe(payload: SubscribePayload, current_user: dict = Depends(get_current_user)):
    plan = db.execute_one(
        "SELECT * FROM subscription_plans WHERE id = %s AND is_active = TRUE",
        (payload.planId,)
    )
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subscription plan not found")

    now = datetime.datetime.utcnow()
    now_iso = now.isoformat() + "Z"
    expires = now + datetime.timedelta(days=_subscription_duration_days(plan["interval"]))
    expires_iso = expires.isoformat() + "Z"

    existing = db.execute_one(
        "SELECT * FROM subscriptions WHERE user_id = %s AND status = 'active'",
        (current_user["id"],)
    )
    if existing:
        db.execute_write(
            "UPDATE subscriptions SET plan_id = %s, started_at = %s, expires_at = %s, cancelled_at = NULL "
            "WHERE id = %s",
            (payload.planId, now_iso, expires_iso, existing["id"])
        )
        sub_id = existing["id"]
    else:
        sub_id = f"sub-{uuid.uuid4().hex[:12]}"
        db.execute_write(
            "INSERT INTO subscriptions (id, user_id, plan_id, status, started_at, expires_at, cancelled_at, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (sub_id, current_user["id"], payload.planId, "active", now_iso, expires_iso, None, now_iso)
        )

    sub = db.execute_one("SELECT * FROM subscriptions WHERE id = %s", (sub_id,))
    return format_subscription(sub, plan)

@app.post("/api/subscription/cancel")
def cancel_subscription(current_user: dict = Depends(get_current_user)):
    sub = db.execute_one(
        "SELECT * FROM subscriptions WHERE user_id = %s AND status = 'active'",
        (current_user["id"],)
    )
    if not sub:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active subscription found")

    now_iso = datetime.datetime.utcnow().isoformat() + "Z"
    db.execute_write(
        "UPDATE subscriptions SET status = 'cancelled', cancelled_at = %s WHERE id = %s",
        (now_iso, sub["id"])
    )
    updated = db.execute_one("SELECT * FROM subscriptions WHERE id = %s", (sub["id"],))
    plan = db.execute_one("SELECT * FROM subscription_plans WHERE id = %s", (updated["plan_id"],))
    return format_subscription(updated, plan)


# --- SHOP RATING ENDPOINTS ---

@app.get("/api/shops/{shopId}/ratings")
def list_shop_ratings(shopId: str, page: int = 1, limit: int = 20):
    shop = db.execute_one("SELECT id FROM shops WHERE id = %s AND is_active = TRUE", (shopId,))
    if not shop:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shop not found")

    offset = (page - 1) * limit
    count_row = db.execute_one(
        "SELECT COUNT(*) as count FROM shop_ratings WHERE shop_id = %s",
        (shopId,)
    )
    total = count_row.get("count", 0) if count_row else 0

    rows = db.execute_query(
        "SELECT r.*, u.name as user_name FROM shop_ratings r "
        "JOIN users u ON r.user_id = u.id "
        "WHERE r.shop_id = %s ORDER BY r.created_at DESC LIMIT %s OFFSET %s",
        (shopId, limit, offset)
    )
    stats = get_shop_rating_stats(shopId)
    return {
        "items": [format_rating(row, row.get("user_name")) for row in rows],
        "total": total,
        "page": page,
        "limit": limit,
        "avgRating": stats["avgRating"],
        "ratingCount": stats["ratingCount"],
    }

@app.post("/api/shops/{shopId}/ratings")
def submit_shop_rating(
    shopId: str,
    payload: RatingPayload,
    current_user: dict = Depends(get_current_user),
):
    shop = db.execute_one("SELECT * FROM shops WHERE id = %s AND is_active = TRUE", (shopId,))
    if not shop:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shop not found")
    if shop["owner_id"] == current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot rate your own shop"
        )

    now_iso = datetime.datetime.utcnow().isoformat() + "Z"
    existing = db.execute_one(
        "SELECT id FROM shop_ratings WHERE shop_id = %s AND user_id = %s",
        (shopId, current_user["id"])
    )
    if existing:
        db.execute_write(
            "UPDATE shop_ratings SET rating = %s, comment = %s, created_at = %s WHERE id = %s",
            (payload.rating, payload.comment, now_iso, existing["id"])
        )
        rating_id = existing["id"]
    else:
        rating_id = f"rating-{uuid.uuid4().hex[:12]}"
        db.execute_write(
            "INSERT INTO shop_ratings (id, shop_id, user_id, rating, comment, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (rating_id, shopId, current_user["id"], payload.rating, payload.comment, now_iso)
        )

    row = db.execute_one("SELECT * FROM shop_ratings WHERE id = %s", (rating_id,))
    return format_rating(row, current_user["name"])

@app.delete("/api/shops/{shopId}/ratings/{ratingId}")
def delete_shop_rating(
    shopId: str,
    ratingId: str,
    current_user: dict = Depends(get_current_user),
):
    rating = db.execute_one(
        "SELECT * FROM shop_ratings WHERE id = %s AND shop_id = %s",
        (ratingId, shopId)
    )
    if not rating:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rating not found")

    shop = db.execute_one("SELECT owner_id FROM shops WHERE id = %s", (shopId,))
    is_owner = shop and shop["owner_id"] == current_user["id"]
    is_rater = rating["user_id"] == current_user["id"]
    if not is_owner and not is_rater:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unauthorized")

    db.execute_write("DELETE FROM shop_ratings WHERE id = %s", (ratingId,))
    return {"success": True}

@app.get("/api/merchant/shops/{shopId}/ratings")
def list_merchant_shop_ratings(
    shopId: str,
    page: int = 1,
    limit: int = 20,
    current_user: dict = Depends(get_current_user),
):
    verify_shop_ownership(shopId, current_user["id"])
    offset = (page - 1) * limit
    count_row = db.execute_one(
        "SELECT COUNT(*) as count FROM shop_ratings WHERE shop_id = %s",
        (shopId,)
    )
    total = count_row.get("count", 0) if count_row else 0
    rows = db.execute_query(
        "SELECT r.*, u.name as user_name FROM shop_ratings r "
        "JOIN users u ON r.user_id = u.id "
        "WHERE r.shop_id = %s ORDER BY r.created_at DESC LIMIT %s OFFSET %s",
        (shopId, limit, offset)
    )
    stats = get_shop_rating_stats(shopId)
    return {
        "items": [format_rating(row, row.get("user_name")) for row in rows],
        "total": total,
        "page": page,
        "limit": limit,
        "avgRating": stats["avgRating"],
        "ratingCount": stats["ratingCount"],
    }

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
