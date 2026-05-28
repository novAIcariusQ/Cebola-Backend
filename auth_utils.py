import hashlib
import os
import base64
import jwt
import logging
from datetime import datetime, timedelta, timezone
from fastapi import Header, HTTPException, status
from database import db

logger = logging.getLogger("CebolaAuth")

SECRET_KEY = os.environ.get("CEBOLA_SECRET_KEY", "super-secret-cebola-key-change-in-production-123456789")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 360  # 6 hours

def hash_password(password: str) -> str:
    """Hashes a password using PBKDF2-HMAC-SHA256 with a unique salt."""
    salt = os.urandom(16)
    db_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    salt_b64 = base64.b64encode(salt).decode('utf-8')
    hash_b64 = base64.b64encode(db_hash).decode('utf-8')
    return f"{salt_b64}:{hash_b64}"

def verify_password(password: str, hashed_password: str) -> bool:
    """Verifies a password against its hash."""
    try:
        salt_b64, hash_b64 = hashed_password.split(':')
        salt = base64.b64decode(salt_b64)
        stored_hash = base64.b64decode(hash_b64)
        new_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return new_hash == stored_hash
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False

def create_access_token(data: dict) -> str:
    """Creates a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    """Decodes and verifies a JWT token. Handles optional Bearer prefix."""
    try:
        if token.startswith("Bearer "):
            token = token[7:]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None

async def get_current_user(authorization: str = Header(None)) -> dict:
    """
    FastAPI dependency to retrieve the current authenticated user from the request header.
    Throws HTTP 401 if missing, invalid, or expired.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Header"
        )
    
    payload = verify_token(authorization)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token"
        )
    
    user = db.execute_one(
        "SELECT id, email, name, created_at FROM users WHERE id = %s",
        (payload["sub"],)
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user
