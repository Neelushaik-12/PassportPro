from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Cookie, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
import io
import os
import sys
from typing import Optional
import base64
import uvicorn
from dotenv import load_dotenv

# Handle imports for both local development and Docker deployment
try:
    # Try absolute import (works when PYTHONPATH includes /app)
    from backend.vertex_ai_service import VertexAIService
    from backend.passport_specs import get_passport_specs, get_all_countries
    from backend.auth import register_user, authenticate_user, verify_session, logout_user, get_user_stats, load_users, save_users
    from backend.secret_manager import get_google_api_key
except ImportError:
    # Fallback to relative imports (works in local development)
    from vertex_ai_service import VertexAIService
    from passport_specs import get_passport_specs, get_all_countries
    from auth import register_user, authenticate_user, verify_session, logout_user, get_user_stats, load_users, save_users
    from secret_manager import get_google_api_key

# Load environment variables from .env file FIRST
load_dotenv()

# Now we can safely access environment variables
from google import genai
from google.genai import types

# Get project ID for Secret Manager
project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")

# Initialize Google GenAI client only if API key is available
# Try Secret Manager first, then fall back to environment variable
client = None
google_api_key = get_google_api_key(project_id)
if google_api_key:
    try:
        client = genai.Client(api_key=google_api_key)
        print("✓ Google GenAI client initialized")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize Google GenAI client: {e}")
        client = None
else:
    print("⚠ GOOGLE_API_KEY not set. Google GenAI client not initialized.")

app = FastAPI(title="Passport Pro API")

security = HTTPBearer(auto_error=False)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Vertex AI service
vertex_ai_service = VertexAIService()


# Authentication dependency
async def get_current_user(session_token: Optional[str] = Cookie(None)):
    """Get current user from session token."""
    if not session_token:
        return None
    username = verify_session(session_token)
    return username


@app.post("/api/auth/register")
async def register(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form("")
):
    """Register a new user."""
    result = register_user(username, password, email)
    if result["success"]:
        return JSONResponse({"success": True, "message": result["message"]})
    else:
        raise HTTPException(status_code=400, detail=result["message"])


@app.post("/api/auth/login")
async def login(
    username: str = Form(...),
    password: str = Form(...)
):
    """Login a user and return session token."""
    session_token = authenticate_user(username, password)
    if session_token:
        response = JSONResponse({
            "success": True,
            "message": "Login successful",
            "username": username
        })
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            max_age=86400,  # 24 hours
            samesite="lax"
        )
        return response
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")


@app.post("/api/auth/logout")
async def logout(session_token: Optional[str] = Cookie(None)):
    """Logout the current user."""
    if session_token:
        logout_user(session_token)
    response = JSONResponse({"success": True, "message": "Logged out successfully"})
    response.delete_cookie(key="session_token")
    return response


@app.get("/api/auth/me")
async def get_current_user_info(current_user: Optional[str] = Depends(get_current_user)):
    """Get current user information."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    stats = get_user_stats(current_user)
    return JSONResponse({"success": True, "user": stats})


@app.get("/api/auth/check")
async def check_auth(current_user: Optional[str] = Depends(get_current_user)):
    """Check if user is authenticated."""
    return JSONResponse({
        "authenticated": current_user is not None,
        "username": current_user
    })


# Mount static files (frontend) - only if frontend directory exists
import pathlib

# Determine frontend path - works both locally and in Docker
# In Docker: /app/frontend
# Locally: ../frontend from backend/
if os.path.exists("/app/frontend"):
    frontend_path = pathlib.Path("/app/frontend")
elif os.path.exists(str(pathlib.Path(__file__).parent.parent / "frontend")):
    frontend_path = pathlib.Path(__file__).parent.parent / "frontend"
else:
    frontend_path = None

if frontend_path and frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.get("/")
async def root():
    # Try to serve index.html if frontend exists, otherwise return API info
    if frontend_path and (frontend_path / "index.html").exists():
        return FileResponse(str(frontend_path / "index.html"))
    return {
        "message": "Passport Pro API is running",
        "endpoints": {
            "process_photo": "POST /api/process-passport-photo",
            "countries": "GET /api/countries",
            "health": "GET /"
        }
    }

@app.get("/index.html")
async def index_page():
    """Serve main application page"""
    if frontend_path and (frontend_path / "index.html").exists():
        return FileResponse(str(frontend_path / "index.html"))
    raise HTTPException(status_code=404, detail="Index page not found")

@app.get("/login.html")
async def login_page():
    """Serve login page"""
    if frontend_path and (frontend_path / "login.html").exists():
        return FileResponse(str(frontend_path / "login.html"))
    raise HTTPException(status_code=404, detail="Login page not found")


@app.get("/api/countries")
async def get_countries():
    """Get list of all available countries with their passport specifications."""
    countries = get_all_countries()
    return {
        "success": True,
        "countries": countries,
        "total": len(countries)
    }


@app.get("/api/countries/{country_code}/specs")
async def get_country_specs(country_code: str):
    """Get passport specifications for a specific country."""
    specs = get_passport_specs(country_code)
    return {
        "success": True,
        "country_code": country_code.upper(),
        "country_name": specs["name"],
        "specs": {
            "width_mm": specs["width_mm"],
            "height_mm": specs["height_mm"],
            "width_px": specs["width"],
            "height_px": specs["height"],
            "dpi": specs["dpi"],
            "background_color": specs["background_color"]
        }
    }


@app.get("/favicon.ico")
async def favicon():
    """Return empty response for favicon requests to avoid 404 errors."""
    from fastapi.responses import Response
    return Response(status_code=204)  # No Content


@app.get("/api/process-passport-photo")
async def process_passport_photo_get():
    """GET endpoint to show API usage information."""
    return {
        "error": "This endpoint requires POST method",
        "usage": "Send a POST request with a file field containing an image",
        "example": "POST /api/process-passport-photo with multipart/form-data"
    }


@app.post("/api/process-passport-photo")
async def process_passport_photo(
    file: UploadFile = File(...),
    country: str = Form(default="US"),
    current_user: Optional[str] = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Process an uploaded photo to make it passport-ready:
    1. Remove/replace background using Vertex AI
    2. Resize to passport dimensions based on selected country
    3. Ensure appropriate background color
    4. Optimize for passport photo requirements
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Normalize country code
        country = country.upper().strip() if country else "US"
        
        # Get passport specifications for the selected country
        specs = get_passport_specs(country)
        print(f"✓ Processing photo for country: {country} ({specs['name']})")
        print(f"  Dimensions: {specs['width_mm']}mm x {specs['height_mm']}mm ({specs['width']}x{specs['height']}px at {specs['dpi']} DPI)")
        bg_color = specs['background_color']
        color_name = 'Royal Blue' if bg_color == (0, 35, 102) else 'White' if bg_color == (255, 255, 255) else 'Light Grey' if bg_color == (240, 240, 240) else 'Custom'
        print(f"  Background color: RGB{bg_color} ({color_name})")
        print(f"  Face height: {specs.get('face_height_percent', 55)}%, Position: {specs.get('face_position_percent', 45)}%")
        
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Process with Vertex AI for background removal
        # Pass target background color - if Imagen 3 is available, it will use it for direct replacement
        processed_image = await vertex_ai_service.remove_background(image, target_background_color=specs['background_color'])
        print(f"  Background removal complete. Image mode: {processed_image.mode}")
        
        # If Imagen 3 was used, it may have already replaced the background
        # But we still do a cleanup pass to ensure all background is replaced
        print(f"  Final background replacement with color RGB{specs['background_color']}...")
        # Pass the original image to help detect original background colors
        processed_image = _replace_background_with_color(processed_image, specs['background_color'], original_image=image)
        print(f"  Background replacement complete. Final mode: {processed_image.mode}")
        
        # Resize and format for passport photo based on country
        passport_image = format_passport_photo(processed_image, specs)
        
        # Update user stats if authenticated
        user_email = None
        if current_user:
            users = load_users()
            if current_user in users:
                users[current_user]["photos_processed"] = users[current_user].get("photos_processed", 0) + 1
                user_email = users[current_user].get("email", None)
                save_users(users)
        
        # Send email notification to admin (non-blocking background task)
        def send_notification():
            try:
                from backend.email_notifier import send_photo_processed_notification
            except ImportError:
                try:
                    from email_notifier import send_photo_processed_notification
                except ImportError:
                    return
            
            try:
                send_photo_processed_notification(
                    username=current_user,
                    user_email=user_email,
                    country=country,
                    country_name=specs['name'],
                    project_id=project_id
                )
            except Exception as email_error:
                print(f"  ⚠ Email notification failed: {email_error}")
        
        # Add to background tasks (non-blocking)
        background_tasks.add_task(send_notification)
        
        # Convert to base64 for response
        img_buffer = io.BytesIO()
        passport_image.save(img_buffer, format="PNG", dpi=(specs["dpi"], specs["dpi"]))
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        
        return JSONResponse({
            "success": True,
            "image": f"data:image/png;base64,{img_base64}",
            "country": specs["name"],
            "country_code": country.upper(),
            "dimensions": {
                "width_mm": specs["width_mm"],
                "height_mm": specs["height_mm"],
                "width_px": specs["width"],
                "height_px": specs["height"],
                "dpi": specs["dpi"]
            },
            "message": f"Photo processed for {specs['name']} passport requirements"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


def _replace_background_with_color(image: Image.Image, target_color: tuple, original_image: Image.Image = None) -> Image.Image:
    """
    Replace background pixels in an RGBA image with the target color.
    Uses alpha channel AND original image colors to detect background.
    """
    import numpy as np
    import cv2
    
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    alpha = img_array[:, :, 3]
    rgb = img_array[:, :, :3]
    
    # Step 0: Create a HARD mask from alpha (ignore feathered edges to prevent halo)
    # Use a threshold to convert soft alpha edges to hard edges
    # This prevents the halo effect from feathered edges
    alpha_hard = (alpha > 200).astype(np.uint8) * 255  # Hard threshold - no soft edges
    
    # Step 1: Start with edges as background (they're ALWAYS background)
    # Don't rely on alpha channel - it might be wrong if background removal failed
    edge_size = max(min(width, height) // 10, 20)
    background_mask = np.zeros((height, width), dtype=bool)
    background_mask[:edge_size, :] = True
    background_mask[-edge_size:, :] = True
    background_mask[:, :edge_size] = True
    background_mask[:, -edge_size:] = True
    
    # Also mark corners
    corner_size = max(min(width, height) // 8, 25)
    background_mask[:corner_size, :corner_size] = True
    background_mask[:corner_size, -corner_size:] = True
    background_mask[-corner_size:, :corner_size] = True
    background_mask[-corner_size:, -corner_size:] = True
    
    # Step 1b: Use the hard alpha mask to detect background
    # Background = low alpha in the hard mask (ignores feathered edges)
    low_alpha_mask = alpha_hard < 255  # Use hard mask, not feathered alpha
    background_mask = background_mask | low_alpha_mask
    
    # Debug: Check if we have any low-alpha pixels
    low_alpha_count = np.sum(alpha < 200)
    print(f"    Debug: Found {low_alpha_count} pixels with alpha < 200")
    
    # Initialize subject protection mask early (will be populated in Step 6)
    subject_protection_mask = np.zeros((height, width), dtype=bool)
    
    # Step 2: If original image provided, detect original background colors
    # This catches background pixels that weren't properly removed (still have high alpha)
    if original_image is not None:
        original_array = np.array(original_image.convert("RGB"))
        original_rgb = original_array.astype(np.float32)
        
        # Sample background colors from edges of original image
        edge_size = max(min(width, height) // 10, 20)
        edge_pixels = np.concatenate([
            original_rgb[:edge_size, :].reshape(-1, 3),
            original_rgb[-edge_size:, :].reshape(-1, 3),
            original_rgb[:, :edge_size].reshape(-1, 3),
            original_rgb[:, -edge_size:].reshape(-1, 3)
        ])
        
        if len(edge_pixels) > 0:
            # Calculate background color statistics
            bg_mean = np.mean(edge_pixels, axis=0)
            bg_std = np.std(edge_pixels, axis=0)
            
            # Find pixels similar to original background color
            # But ONLY if they're not in the subject area (protected by alpha)
            rgb_float = rgb.astype(np.float32)
            color_diff = np.linalg.norm(rgb_float - bg_mean, axis=2)
            
            # Threshold: pixels similar to background
            # Use a more lenient threshold for background detection
            # Calculate threshold based on std, but make it more aggressive
            threshold = max(np.mean(bg_std) * 3.0 + 50, 60)  # More lenient
            similar_to_bg = color_diff < threshold
            
            # Mark as background if:
            # 1. Similar to background color AND
            # 2. Either low alpha OR not in the center (subject area)
            center_mask = np.zeros((height, width), dtype=bool)
            center_x, center_y = width // 2, height // 2
            center_radius = min(width, height) // 2.5  # Larger center area
            y_coords, x_coords = np.ogrid[:height, :width]
            center_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) < center_radius**2
            
            # Background if: similar color AND (low alpha OR outside center)
            # BUT exclude high-alpha pixels in center (protects white clothing)
            # Only mark as background if alpha is low OR it's outside center AND not in subject protection area
            bg_from_color = similar_to_bg & ((alpha < 200) | ((~center_mask) & (alpha < 240)))
            
            # Exclude subject protection mask from background
            bg_from_color = bg_from_color & (~subject_protection_mask)
            
            # Combine with existing background mask
            background_mask = background_mask | bg_from_color
    
    # Step 3: Edge mask for protection (already added in Step 1, but keep for reference)
    edge_size = max(min(width, height) // 10, 20)
    edge_mask = np.zeros((height, width), dtype=bool)
    edge_mask[:edge_size, :] = True
    edge_mask[-edge_size:, :] = True
    edge_mask[:, :edge_size] = True
    edge_mask[:, -edge_size:] = True
    
    # Step 4: Use flood fill from edges to catch connected background regions
    # Make flood fill more aggressive - allow it to spread through background-colored areas
    mask_refined = background_mask.astype(np.uint8) * 255
    
    # Flood fill from corners and edges
    h, w = mask_refined.shape
    
    # Create a mask for flood fill - OpenCV requires mask to be 2 pixels larger
    # Use HARD alpha mask to prevent halo effects
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    
    # If we have original image, use color similarity for flood fill
    if original_image is not None:
        # Allow flood fill through areas similar to background color
        original_array = np.array(original_image.convert("RGB"))
        original_rgb = original_array.astype(np.float32)
        edge_size = max(min(width, height) // 10, 20)
        edge_pixels = np.concatenate([
            original_rgb[:edge_size, :].reshape(-1, 3),
            original_rgb[-edge_size:, :].reshape(-1, 3),
            original_rgb[:, :edge_size].reshape(-1, 3),
            original_rgb[:, -edge_size:].reshape(-1, 3)
        ])
        if len(edge_pixels) > 0:
            bg_mean = np.mean(edge_pixels, axis=0)
            rgb_float = rgb.astype(np.float32)
            color_diff = np.linalg.norm(rgb_float - bg_mean, axis=2)
            # Use hard alpha mask OR background-colored areas (prevents halo)
            fill_allowed = (alpha_hard < 255) | (color_diff < 50)  # Use hard mask
            fill_mask[1:-1, 1:-1] = fill_allowed.astype(np.uint8) * 255
        else:
            # Fallback: use hard alpha mask
            fill_mask[1:-1, 1:-1] = (alpha_hard < 255).astype(np.uint8) * 255
    else:
        # No original image - use hard alpha mask
        fill_mask[1:-1, 1:-1] = (alpha_hard < 255).astype(np.uint8) * 255
    
    # Flood fill from corners - adjust coordinates for the larger mask
    cv2.floodFill(mask_refined, fill_mask, (0, 0), 255)
    cv2.floodFill(mask_refined, fill_mask, (w-1, 0), 255)
    cv2.floodFill(mask_refined, fill_mask, (0, h-1), 255)
    cv2.floodFill(mask_refined, fill_mask, (w-1, h-1), 255)
    
    # Also flood fill from edge centers
    cv2.floodFill(mask_refined, fill_mask, (w//2, 0), 255)
    cv2.floodFill(mask_refined, fill_mask, (w//2, h-1), 255)
    cv2.floodFill(mask_refined, fill_mask, (0, h//2), 255)
    cv2.floodFill(mask_refined, fill_mask, (w-1, h//2), 255)
    
    background_mask = mask_refined > 128
    
    # Step 5: Protect foreground areas using HARD alpha mask (not feathered)
    # Use the hard mask to avoid halo effects
    # Only protect areas that are:
    # 1. High alpha in hard mask (255) AND
    # 2. NOT on edges AND  
    # 3. In the center area (likely subject)
    center_mask = np.zeros((height, width), dtype=bool)
    center_x, center_y = width // 2, height // 2
    center_radius = min(width, height) // 3
    y_coords, x_coords = np.ogrid[:height, :width]
    center_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) < center_radius**2
    
    # Use hard alpha mask to protect foreground (avoids halo)
    foreground_mask = (alpha_hard == 255) & center_mask & (~edge_mask)
    background_mask = background_mask & (~foreground_mask)  # Remove foreground from background mask
    
    # Step 6: Additional protection - detect and protect face/body area
    # This is a safety measure in case alpha channel isn't perfect
    img_bgr = cv2.cvtColor(img_array[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Create a comprehensive subject protection mask
    subject_protection_mask = np.zeros((height, width), dtype=bool)
    
    if len(faces) > 0:
        # Protect face and entire body area with more aggressive expansion
        for (x, y, w, h) in faces:
            # Much more aggressive expansion to include entire head, neck, shoulders, and shirt
            expand_x = int(w * 1.5)  # Wider expansion for shoulders
            expand_y_up = int(h * 0.5)  # Expand upward for hair
            expand_y_down = int(h * 4.0)  # Expand downward to include full torso and shirt
            x1 = max(0, x - expand_x)
            y1 = max(0, y - expand_y_up)
            x2 = min(width, x + w + expand_x)
            y2 = min(height, y + h + expand_y_down)
            # Mark this entire area as protected
            subject_protection_mask[y1:y2, x1:x2] = True
    else:
        # If no face detected, protect a large center area
        center_x, center_y = width // 2, height // 2
        protect_radius = min(width, height) // 2.0  # Large protection radius
        y_coords, x_coords = np.ogrid[:height, :width]
        subject_protection_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) < protect_radius**2
    
    # Step 6b: Protect all high-alpha pixels in the center region (especially important for white clothing)
    # This ensures that any pixel with high alpha (likely foreground) in the subject area is protected
    center_x, center_y = width // 2, height // 2
    center_radius = min(width, height) // 1.8  # Large center area
    y_coords, x_coords = np.ogrid[:height, :width]
    center_area = ((x_coords - center_x)**2 + (y_coords - center_y)**2) < center_radius**2
    
    # Protect all high-alpha pixels in the center area (this protects white clothing)
    high_alpha_in_center = (alpha > 180) & center_area
    subject_protection_mask = subject_protection_mask | high_alpha_in_center
    
    # Remove protected subject area from background mask
    background_mask = background_mask & (~subject_protection_mask)
    
    # Step 7: Final fallback - if we still haven't detected enough background,
    # force background detection from original image edges (background removal might have completely failed)
    num_bg_pixels = np.sum(background_mask)
    total_pixels = height * width
    bg_percentage = (num_bg_pixels / total_pixels) * 100
    
    # If we detected less than 10% background, background removal likely failed completely
    # Force background detection from original image using VERY aggressive method
    if bg_percentage < 10.0 and original_image is not None:
        print(f"    ⚠ Low background detection ({bg_percentage:.1f}%) - forcing AGGRESSIVE detection from original image...")
        original_array = np.array(original_image.convert("RGB"))
        original_rgb = original_array.astype(np.float32)
        
        # Sample background from edges (larger sample area)
        edge_size = max(min(width, height) // 6, 40)
        edge_pixels = np.concatenate([
            original_rgb[:edge_size, :].reshape(-1, 3),
            original_rgb[-edge_size:, :].reshape(-1, 3),
            original_rgb[:, :edge_size].reshape(-1, 3),
            original_rgb[:, -edge_size:].reshape(-1, 3)
        ])
        
        if len(edge_pixels) > 0:
            bg_mean = np.mean(edge_pixels, axis=0)
            rgb_float = rgb.astype(np.float32)
            color_diff = np.linalg.norm(rgb_float - bg_mean, axis=2)
            
            # VERY aggressive: mark anything similar to background as background
            # Use a large threshold to catch all background-like colors
            similar_to_bg = color_diff < 80  # Very lenient threshold
            
            # Only protect the center area (subject) - everything else can be background
            center_mask = np.zeros((height, width), dtype=bool)
            center_x, center_y = width // 2, height // 2
            center_radius = min(width, height) // 2.2  # Protect center area
            y_coords, x_coords = np.ogrid[:height, :width]
            center_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) < center_radius**2
            
            # Mark as background if: similar to background color AND outside protected center
            # BUT exclude the subject protection mask (face/body/shirt area)
            aggressive_bg = similar_to_bg & (~center_mask) & (~subject_protection_mask)
            
            # Also force edges to be background
            aggressive_bg = aggressive_bg | edge_mask
            
            # Combine with existing mask, but respect subject protection
            background_mask = (background_mask | aggressive_bg) & (~subject_protection_mask)
            
            # Re-count
            num_bg_pixels = np.sum(background_mask)
            bg_percentage = (num_bg_pixels / total_pixels) * 100
            print(f"    After aggressive fallback: Detected {num_bg_pixels}/{total_pixels} background pixels ({bg_percentage:.1f}%)")
    
    print(f"    Detected {num_bg_pixels}/{total_pixels} background pixels ({bg_percentage:.1f}%)")
    print(f"    Protected foreground pixels: {np.sum(alpha > 150)} ({np.sum(alpha > 150)/total_pixels*100:.1f}%)")
    
    img_array[background_mask, 0] = target_color[0]  # R
    img_array[background_mask, 1] = target_color[1]  # G
    img_array[background_mask, 2] = target_color[2]  # B
    img_array[background_mask, 3] = 255  # Make fully opaque
    
    return Image.fromarray(img_array, 'RGBA')


def format_passport_photo(image: Image.Image, specs: dict) -> Image.Image:
    """
    Format image to passport photo specifications based on country:
    - Resize to passport dimensions
    - Add appropriate background color
    - Center the subject with proper positioning
    """
    import cv2
    import numpy as np
    
    # Get specifications
    PASSPORT_WIDTH = specs["width"]
    PASSPORT_HEIGHT = specs["height"]
    PASSPORT_DPI = specs["dpi"]
    BACKGROUND_COLOR = specs["background_color"]
    FACE_HEIGHT_PERCENT = specs.get("face_height_percent", 55) / 100
    FACE_POSITION_PERCENT = specs.get("face_position_percent", 45) / 100
    
    # Convert to numpy for processing
    if image.mode == "RGBA":
        img_array = np.array(image)
        alpha = img_array[:, :, 3]
        img_rgb = img_array[:, :, :3]
    else:
        img_array = np.array(image.convert("RGB"))
        img_rgb = img_array
        alpha = None
    
    img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    height, width = img_cv.shape[:2]
    
    # Detect face to better position the subject
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Calculate optimal size and position
    if len(faces) > 0:
        # Use the largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        fx, fy, fw, fh = face
        
        # For passport photos, face should be specified percentage of image height
        target_face_height = PASSPORT_HEIGHT * FACE_HEIGHT_PERCENT
        scale_factor = target_face_height / fh
        
        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Ensure it fits within passport dimensions
        if new_width > PASSPORT_WIDTH:
            scale_factor = PASSPORT_WIDTH / width
            new_width = PASSPORT_WIDTH
            new_height = int(height * scale_factor)
        
        if new_height > PASSPORT_HEIGHT:
            scale_factor = PASSPORT_HEIGHT / height
            new_height = PASSPORT_HEIGHT
            new_width = int(width * scale_factor)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Calculate position: face should be at specified position from top
        face_y_in_resized = int(fy * scale_factor)
        target_face_y = int(PASSPORT_HEIGHT * FACE_POSITION_PERCENT)
        y_offset = target_face_y - face_y_in_resized
        
        # Center horizontally
        x_offset = (PASSPORT_WIDTH - new_width) // 2
        
        # Ensure offsets are non-negative
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)
        
    else:
        # No face detected, use standard centering
        img_aspect = width / height
        target_aspect = PASSPORT_WIDTH / PASSPORT_HEIGHT
        
        if img_aspect > target_aspect:
            new_height = PASSPORT_HEIGHT
            new_width = int(PASSPORT_HEIGHT * img_aspect)
        else:
            new_width = PASSPORT_WIDTH
            new_height = int(PASSPORT_WIDTH / img_aspect)
        
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        x_offset = (PASSPORT_WIDTH - new_width) // 2
        y_offset = (PASSPORT_HEIGHT - new_height) // 2
    
    # Create background with specified color (exact passport dimensions)
    passport_img = Image.new("RGB", (PASSPORT_WIDTH, PASSPORT_HEIGHT), BACKGROUND_COLOR)
    
    # Ensure resized image has alpha channel for proper background replacement
    if resized.mode != "RGBA":
        # Convert to RGBA if not already
        resized = resized.convert("RGBA")
    
    # Ensure the resized image fits within passport dimensions
    # Crop if necessary to ensure it doesn't exceed boundaries
    if new_width > PASSPORT_WIDTH:
        # Crop horizontally if needed
        crop_x = (new_width - PASSPORT_WIDTH) // 2
        resized = resized.crop((crop_x, 0, crop_x + PASSPORT_WIDTH, new_height))
        new_width = PASSPORT_WIDTH
        x_offset = 0
    
    if new_height > PASSPORT_HEIGHT:
        # Crop vertically if needed
        crop_y = (new_height - PASSPORT_HEIGHT) // 2
        resized = resized.crop((0, crop_y, new_width, crop_y + PASSPORT_HEIGHT))
        new_height = PASSPORT_HEIGHT
        y_offset = 0
    
    # Ensure offsets are within bounds
    x_offset = max(0, min(x_offset, PASSPORT_WIDTH - new_width))
    y_offset = max(0, min(y_offset, PASSPORT_HEIGHT - new_height))
    
    # CRITICAL: Ensure the resized image fills the entire passport dimensions
    # If the resized image is smaller than passport dimensions, we need to scale it up to fill
    if new_width < PASSPORT_WIDTH or new_height < PASSPORT_HEIGHT:
        # Calculate scale to fill passport dimensions while maintaining aspect ratio
        scale_x = PASSPORT_WIDTH / new_width
        scale_y = PASSPORT_HEIGHT / new_height
        # Use the larger scale to ensure we fill the entire space (may crop slightly)
        fill_scale = max(scale_x, scale_y)
        
        # Resize to fill
        fill_width = int(new_width * fill_scale)
        fill_height = int(new_height * fill_scale)
        resized = resized.resize((fill_width, fill_height), Image.Resampling.LANCZOS)
        
        # Crop to exact passport dimensions (center crop)
        if fill_width > PASSPORT_WIDTH:
            crop_x = (fill_width - PASSPORT_WIDTH) // 2
            resized = resized.crop((crop_x, 0, crop_x + PASSPORT_WIDTH, fill_height))
            fill_width = PASSPORT_WIDTH
            x_offset = 0
        
        if fill_height > PASSPORT_HEIGHT:
            crop_y = (fill_height - PASSPORT_HEIGHT) // 2
            resized = resized.crop((0, crop_y, fill_width, crop_y + PASSPORT_HEIGHT))
            fill_height = PASSPORT_HEIGHT
            y_offset = 0
        
        # Update dimensions
        new_width = fill_width
        new_height = fill_height
    
    # Paste the resized image - it should now fill the entire canvas
    # If it's exactly the passport size, paste at (0, 0)
    if new_width == PASSPORT_WIDTH and new_height == PASSPORT_HEIGHT:
        passport_img.paste(resized, (0, 0), resized if resized.mode == "RGBA" else None)
    else:
        # Should not happen after the fill logic above, but handle it
        passport_img.paste(resized, (x_offset, y_offset), resized if resized.mode == "RGBA" else None)
    
    # Final verification: ensure no white space by cropping/resizing to exact dimensions
    if passport_img.size != (PASSPORT_WIDTH, PASSPORT_HEIGHT):
        # Crop if larger, or resize if smaller (shouldn't happen)
        if passport_img.size[0] > PASSPORT_WIDTH or passport_img.size[1] > PASSPORT_HEIGHT:
            # Center crop to exact dimensions
            w, h = passport_img.size
            left = (w - PASSPORT_WIDTH) // 2
            top = (h - PASSPORT_HEIGHT) // 2
            passport_img = passport_img.crop((left, top, left + PASSPORT_WIDTH, top + PASSPORT_HEIGHT))
        else:
            # Shouldn't happen, but resize if needed
            passport_img = passport_img.resize((PASSPORT_WIDTH, PASSPORT_HEIGHT), Image.Resampling.LANCZOS)
    
    # Verify final dimensions
    final_width, final_height = passport_img.size
    width_mm = specs.get("width_mm", 0)
    height_mm = specs.get("height_mm", 0)
    print(f"  Final image size: {final_width}x{final_height}px ({width_mm}mm × {height_mm}mm at {PASSPORT_DPI} DPI)")
    
    # Verify dimensions match exactly
    assert final_width == PASSPORT_WIDTH and final_height == PASSPORT_HEIGHT, \
        f"Final dimensions {final_width}x{final_height} don't match required {PASSPORT_WIDTH}x{PASSPORT_HEIGHT}"
    
    return passport_img


if __name__ == "__main__":
    # Use PORT environment variable for Cloud Run, default to 8001 for local
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

