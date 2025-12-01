"""
Authentication module for Passport Pro.
Handles user registration, login, and session management.
"""
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Optional
import secrets

# Try to import secret_manager, fall back to environment variable
try:
    from backend.secret_manager import get_secret_key
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    SECRET_KEY = get_secret_key(project_id) or secrets.token_hex(32)
except ImportError:
    try:
        from secret_manager import get_secret_key
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        SECRET_KEY = get_secret_key(project_id) or secrets.token_hex(32)
    except ImportError:
        # Fallback to environment variable or generate new
        SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))

# Simple file-based user storage (in production, use a proper database)
USERS_FILE = "users.json"

# In-memory session storage (in production, use Redis or database)
sessions = {}


def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password) == hashed


def load_users() -> dict:
    """Load users from file."""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_users(users: dict):
    """Save users to file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


def register_user(username: str, password: str, email: str = "") -> dict:
    """
    Register a new user.
    Returns: {"success": bool, "message": str}
    """
    users = load_users()
    
    # Check if username already exists
    if username in users:
        return {"success": False, "message": "Username already exists"}
    
    # Validate username
    if len(username) < 3:
        return {"success": False, "message": "Username must be at least 3 characters"}
    
    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters"}
    
    # Create user
    users[username] = {
        "password_hash": hash_password(password),
        "email": email,
        "created_at": datetime.now().isoformat(),
        "photos_processed": 0
    }
    
    save_users(users)
    return {"success": True, "message": "User registered successfully"}


def authenticate_user(username: str, password: str) -> Optional[str]:
    """
    Authenticate a user.
    Returns: session_token if successful, None otherwise
    """
    users = load_users()
    
    if username not in users:
        return None
    
    user = users[username]
    if not verify_password(password, user["password_hash"]):
        return None
    
    # Create session token
    session_token = secrets.token_urlsafe(32)
    sessions[session_token] = {
        "username": username,
        "created_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(hours=24)
    }
    
    return session_token


def verify_session(session_token: str) -> Optional[str]:
    """
    Verify a session token.
    Returns: username if valid, None otherwise
    """
    if session_token not in sessions:
        return None
    
    session = sessions[session_token]
    
    # Check if session expired
    if datetime.now() > session["expires_at"]:
        del sessions[session_token]
        return None
    
    return session["username"]


def logout_user(session_token: str):
    """Logout a user by removing their session."""
    if session_token in sessions:
        del sessions[session_token]


def get_user_stats(username: str) -> dict:
    """Get user statistics."""
    users = load_users()
    if username in users:
        return {
            "username": username,
            "email": users[username].get("email", ""),
            "photos_processed": users[username].get("photos_processed", 0),
            "created_at": users[username].get("created_at", "")
        }
    return {}

