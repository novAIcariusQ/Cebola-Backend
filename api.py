import os
import uuid
import datetime
import shutil
import logging
import random
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException, status, Header, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db
from auth_utils import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user
)
from ai_utils import AiServiceError, describe_product_from_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CebolaAPI")

# Initialize FastAPI App
app = FastAPI(title="Cebola Backend API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files serving for uploaded product images
# Serves at /api/uploads/ to map perfectly with Vite's /api proxy
UPLOAD_DIR = "uploads"
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

# --- RESPONSE FORMATTING HELPERS ---

def format_user(row: dict) -> Optional[dict]:
    if not row:
        return None
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "createdAt": row["created_at"]
    }

def format_shop(row: dict) -> Optional[dict]:
    if not row:
        return None
    return {
        "id": row["id"],
        "ownerId": row["owner_id"],
        "name": row["name"],
        "description": row["description"],
        "logoUrl": row["logo_url"],
        "isActive": bool(row["is_active"]),
        "createdAt": row["created_at"]
    }

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
    return {
        "id": row["id"],
        "title": row["name"],
        "description": row["description"],
        "logoUrl": row.get("logo_url"),
        "isAvailable": bool(row["is_active"]),
    }

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

def format_customer_order_response(order_id: str, guest_order_id: str) -> dict:
    payment_template = os.getenv(
        "PAYMENT_URL_TEMPLATE",
        "https://checkout.stripe.com/c/pay/{order_id}",
    )
    payment_url = payment_template.format(order_id=order_id, guest_order_id=guest_order_id)
    return {
        "id": order_id,
        "guestOrderId": guest_order_id,
        "paymentUrl": payment_url,
    }

def _generate_guest_order_id() -> str:
    return f"{random.randint(10000000, 99999999)}"

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
    return format_shop(row)

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
    q: Optional[str] = None
):
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
def get_order_by_id(shopId: str, orderId: str):
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
def create_customer_order(payload: CustomerOrderPayload):
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

    order_id = str(uuid.uuid4())
    guest_order_id = _generate_guest_order_id()
    created_at = datetime.datetime.utcnow().isoformat() + "Z"

    db.execute_write(
        "INSERT INTO orders (id, shop_id, user_id, guest_order_id, customer_name, customer_email, "
        "customer_phone, total_amount, total_quantity, status, qr_code_data, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            order_id,
            payload.shopId,
            None,
            guest_order_id,
            None,
            None,
            None,
            total_amount,
            total_quantity,
            "pending",
            f"order:{order_id}",
            created_at,
        ),
    )

    for product, quantity, line_total in line_items:
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

        new_quantity = int(product["quantity"]) - quantity
        is_available = new_quantity > 0 and bool(product["is_available"])
        db.execute_write(
            "UPDATE products SET quantity = %s, is_available = %s WHERE id = %s",
            (new_quantity, is_available, product["id"]),
        )

    return format_customer_order_response(order_id, guest_order_id)


@app.get("/api/orders/{orderId}")
def get_customer_order(orderId: str):
    order_row = db.execute_one(
        "SELECT * FROM orders WHERE id = %s OR guest_order_id = %s",
        (orderId, orderId),
    )
    if not order_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")

    guest_order_id = order_row.get("guest_order_id") or ""
    return format_customer_order_response(order_row["id"], guest_order_id)


# --- UPLOAD & ARTIFICIAL INTELLIGENCE ENDPOINTS ---

@app.post("/api/upload")
def upload_image(file: UploadFile = File(...)):
    """Receives and stores product or shop logo images in the local uploads/ folder."""
    # Generate unique filename to avoid collision
    ext = os.path.splitext(file.filename)[1]
    # Default to .png if no extension found
    if not ext:
        ext = ".png"
    unique_filename = f"{uuid.uuid4().hex[:16]}{ext}"
    dest_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Public URL served at /api/uploads/
        public_url = f"/api/uploads/{unique_filename}"
        return {"url": public_url}
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded image"
        )

@app.post("/api/ai")
def describe_product(file: UploadFile = File(...)):
    """Runs Google Vision OCR on the image, then generates title and description with MiniCPM5."""
    try:
        image_bytes = file.file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image is empty"
            )
        return describe_product_from_image(image_bytes)
    except AiServiceError as exc:
        logger.error("AI product description failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected AI failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate product description"
        )
