# Passport Pro 📸

A professional passport photo generator that uses Google Vertex AI (Gemini) and advanced computer vision to automatically remove backgrounds and format photos to meet country-specific passport requirements.

## Features

- 🤖 **AI-Powered Background Removal**: Uses Vertex AI Gemini for intelligent image analysis and Rembg (U2-Net) for perfect segmentation
- 📐 **Country-Specific Formatting**: Automatically formats images to meet passport requirements for 50+ countries
- 🎨 **Smart Background Replacement**: Ensures correct background colors (white, light grey, or royal blue) based on country requirements
- 📏 **Precise Dimensions**: Converts measurements from millimeters to pixels at correct DPI (typically 300 DPI)
- 💻 **Modern Web Interface**: Beautiful, responsive UI with file upload and camera capture capabilities
- 🔐 **User Authentication**: Secure registration and login system with session management
- 📥 **Download Ready**: Download your processed passport photo instantly
- ☁️ **Cloud Ready**: Deploy to Google Cloud Run with Secret Manager integration

## Process Flow

### 1. User Upload/Capture
- User selects a country from the dropdown menu
- User uploads a photo via file upload or captures one using the camera
- Frontend validates the image and displays preview

### 2. Photo Processing Request
- Frontend sends POST request to `/api/process-passport-photo` with:
  - Image file (multipart/form-data)
  - Country code (e.g., "US", "UK", "AU")
  - Session token (if authenticated)

### 3. Backend Processing Pipeline

#### Step 3.1: Country Specification Retrieval
- Backend retrieves passport specifications for the selected country:
  - Dimensions (width_mm × height_mm)
  - Pixel dimensions (calculated at specified DPI)
  - Background color (RGB tuple)
  - Face height percentage
  - Face position percentage

#### Step 3.2: Image Preprocessing
- Image is loaded and converted to RGB format
- Original image is preserved for background color detection

#### Step 3.3: Background Removal (Multi-Stage Process)

**Stage 1: Gemini Analysis (Optional)**
- Vertex AI Gemini model analyzes the image
- Provides guidance on:
  - Subject location
  - Background characteristics
  - Color differences between subject and background
  - Shadows and complex elements

**Stage 2: Initial Background Removal**
- Uses OpenCV with advanced techniques:
  - Face detection using Haar cascades
  - GrabCut algorithm for segmentation
  - Edge and corner detection
  - Color similarity analysis

**Stage 3: Rembg Cleanup**
- Applies Rembg (U2-Net) model for perfect segmentation
- Removes any remaining background artifacts
- Produces clean alpha channel mask

**Stage 4: Background Color Replacement**
- Detects background pixels using:
  - Alpha channel from previous stages
  - Edge and corner analysis
  - Color similarity to original background
  - Flood fill algorithm to catch connected regions
- Replaces background with country-specific color:
  - White `(255, 255, 255)` for most countries
  - Light Grey `(240, 240, 240)` for UK, Australia, Switzerland
  - Royal Blue `(0, 35, 102)` for Philippines
- Protects foreground (face, body, clothing) from replacement

#### Step 3.4: Passport Photo Formatting
- Resizes image to exact passport dimensions:
  - Calculates pixel dimensions: `pixels = (mm × DPI) / 25.4`
  - Maintains aspect ratio while fitting to dimensions
  - Crops if necessary to match exact size
- Positions face according to specifications:
  - Face height: Percentage of image height (e.g., 50-75%)
  - Face position: Vertical position from top (typically 50%)
- Applies correct DPI metadata (typically 300 DPI)

#### Step 3.5: User Statistics Update (If Authenticated)
- Increments user's `photos_processed` counter
- Saves updated user data

#### Step 3.6: Response Generation
- Converts processed image to base64-encoded PNG
- Returns JSON response with:
  - Success status
  - Base64-encoded image data
  - Country information
  - Dimensions (mm and pixels)
  - DPI information

### 4. Frontend Display
- Frontend receives processed image
- Displays "Passport Ready" photo in preview section
- Shows country name and dimensions
- Enables download button

### 5. Download
- User clicks "Download Passport Photo"
- Browser downloads the processed image as PNG
- Image is ready for passport application

## Architecture

```
┌─────────────┐
│   Browser   │
│  (Frontend) │
└──────┬──────┘
       │ HTTP POST
       │ /api/process-passport-photo
       ▼
┌─────────────────────────────────────┐
│      FastAPI Backend (main.py)     │
│  ┌───────────────────────────────┐ │
│  │ 1. Get Country Specs          │ │
│  │    (passport_specs.py)        │ │
│  └──────────────┬────────────────┘ │
│                 ▼                   │
│  ┌───────────────────────────────┐ │
│  │ 2. Background Removal         │ │
│  │    (vertex_ai_service.py)     │ │
│  │    ├─ Gemini Analysis         │ │
│  │    ├─ OpenCV Processing       │ │
│  │    ├─ Rembg Cleanup          │ │
│  │    └─ Color Replacement      │ │
│  └──────────────┬────────────────┘ │
│                 ▼                   │
│  ┌───────────────────────────────┐ │
│  │ 3. Format to Passport Size   │ │
│  │    (format_passport_photo)    │ │
│  └──────────────┬────────────────┘ │
│                 ▼                   │
│  ┌───────────────────────────────┐ │
│  │ 4. Return Base64 Image        │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
       │
       │ Uses
       ▼
┌─────────────────────────────────────┐
│      Google Cloud Services          │
│  ├─ Vertex AI (Gemini)              │
│  ├─ Secret Manager (API Keys)       │
│  └─ Cloud Run (Deployment)          │
└─────────────────────────────────────┘
```

## Prerequisites

- Python 3.8 or higher
- Google Cloud Project with:
  - Vertex AI API enabled
  - Secret Manager API enabled (for production)
  - Cloud Run API enabled (for deployment)
- Google Cloud credentials configured:
  ```bash
  gcloud auth application-default login
  ```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Google Cloud

1. Create a Google Cloud Project (if you don't have one)
2. Enable the required APIs:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable secretmanager.googleapis.com
   gcloud services enable run.googleapis.com
   ```
3. Set up authentication:
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

### 3. Configure Environment Variables

Create a `.env` file in the `backend` directory:

```bash
cd backend
# Copy the template
cat config_template.txt > .env
# Then edit .env with your project details
```

Edit `.env` and add your project details:
```
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
MODEL_NAME=gemini-1.5-pro  # Optional: gemini-1.5-flash, gemini-pro
```

**For Production (Cloud Run):**
- Store secrets in Google Cloud Secret Manager:
  - `google-api-key`: Your Google API key
  - `secret-key`: Secret key for session encryption
  - `admin-email`: Admin email for photo processing notifications
  - `smtp-server`, `smtp-port`, `smtp-username`, `smtp-password`: SMTP configuration for email notifications
- The application automatically retrieves secrets from Secret Manager

**Email Notifications:**
- Admin receives email notifications when photos are processed
- Configure using `./setup_email.sh` or set environment variables:
  - `ADMIN_EMAIL`: Admin email address
  - `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`: SMTP settings
- For Gmail, use an App Password (not your regular password)

### 4. Run the Application

**Option 1: Using the startup script (Recommended)**
```bash
./start.sh
```

**Option 2: Manual setup**
```bash
# Install dependencies
pip install -r requirements.txt

# Start the backend server
cd backend
python3 main.py
```

The API will be available at `http://localhost:8001`

### 5. Open the Frontend

The backend serves the frontend automatically. Simply navigate to:
- **Main App**: `http://localhost:8001/` or `http://localhost:8001/index.html`
- **Login Page**: `http://localhost:8001/login.html`

## Deployment to Google Cloud Run

### Quick Deployment

```bash
# Make script executable
chmod +x deploy.sh

# Deploy to Cloud Run
./deploy.sh us-central1
```

### Manual Deployment

1. **Set up secrets in Secret Manager:**
   ```bash
   # Create secrets
   echo -n "your-api-key" | gcloud secrets create google-api-key --data-file=-
   echo -n "your-secret-key" | gcloud secrets create secret-key --data-file=-
   
   # Grant Cloud Run access
   gcloud secrets add-iam-policy-binding google-api-key \
     --member="serviceAccount:YOUR_SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   ```

2. **Build and deploy:**
   ```bash
   # Build Docker image
   docker build -t gcr.io/YOUR_PROJECT_ID/passport-pro:latest .
   
   # Push to Container Registry
   docker push gcr.io/YOUR_PROJECT_ID/passport-pro:latest
   
   # Deploy to Cloud Run
   gcloud run deploy passport-pro \
     --image gcr.io/YOUR_PROJECT_ID/passport-pro:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2 \
     --set-env-vars "GOOGLE_CLOUD_PROJECT_ID=YOUR_PROJECT_ID,GOOGLE_CLOUD_LOCATION=us-central1"
   ```

See `QUICK_DEPLOY.md` for detailed deployment instructions.

## API Endpoints

### POST `/api/process-passport-photo`

Processes an uploaded image to make it passport-ready.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: Image file
  - `country`: Country code (e.g., "US", "UK", "AU") - optional, defaults to "US"

**Response:**
```json
{
  "success": true,
  "image": "data:image/png;base64,...",
  "country": "United States",
  "country_code": "US",
  "dimensions": {
    "width_mm": 50.8,
    "height_mm": 50.8,
    "width_px": 600,
    "height_px": 600,
    "dpi": 300
  },
  "message": "Photo processed for United States passport requirements"
}
```

### GET `/api/countries`

Returns list of all supported countries.

**Response:**
```json
{
  "success": true,
  "countries": [
    {"code": "US", "name": "United States"},
    {"code": "UK", "name": "United Kingdom"},
    ...
  ]
}
```

### GET `/api/countries/{country_code}/specs`

Returns passport specifications for a specific country.

**Response:**
```json
{
  "success": true,
  "country_code": "US",
  "country_name": "United States",
  "specs": {
    "width_mm": 50.8,
    "height_mm": 50.8,
    "width_px": 600,
    "height_px": 600,
    "dpi": 300,
    "background_color": [255, 255, 255]
  }
}
```

### POST `/api/auth/register`

Register a new user.

**Request:**
- Content-Type: `application/x-www-form-urlencoded`
- Body:
  - `username`: Username
  - `password`: Password
  - `email`: Email (optional)

### POST `/api/auth/login`

Login and get session token.

**Request:**
- Content-Type: `application/x-www-form-urlencoded`
- Body:
  - `username`: Username
  - `password`: Password

**Response:**
```json
{
  "success": true,
  "message": "Login successful",
  "session_token": "..."
}
```

### GET `/api/auth/stats`

Get user statistics (requires authentication).

**Response:**
```json
{
  "success": true,
  "user": {
    "username": "user123",
    "photos_processed": 5
  }
}
```

## Project Structure

```
PassportPro/
├── backend/
│   ├── main.py                  # FastAPI application and endpoints
│   ├── vertex_ai_service.py     # Vertex AI integration (Gemini + Rembg)
│   ├── passport_specs.py        # Country-specific passport specifications
│   ├── auth.py                  # User authentication and session management
│   ├── secret_manager.py        # Google Cloud Secret Manager integration
│   ├── background_editor.py     # Reference for Imagen 3 (optional)
│   ├── config_template.txt      # Environment variables template
│   └── .env                     # Environment variables (create from template)
├── frontend/
│   ├── index.html               # Main application interface
│   └── login.html               # Login and registration page
├── requirements.txt              # Python dependencies
├── Dockerfile                   # Docker image definition
├── start_server.sh              # Server startup script
├── start.sh                     # Local development startup script
├── deploy.sh                    # Cloud Run deployment script
├── README.md                    # This file
└── QUICK_START.md               # Quick start guide
```

## Supported Countries

The application supports 50+ countries with accurate passport photo specifications, including:

- **Americas**: US, Canada, Mexico, Brazil, Argentina
- **Europe**: UK, France, Germany, Italy, Spain, Switzerland, Turkey
- **Asia-Pacific**: Australia, India, China, Japan, Philippines, Singapore
- **Middle East**: UAE, Saudi Arabia
- And many more...

Each country has specific requirements for:
- Dimensions (width × height in mm)
- Background color (white, light grey, or royal blue)
- Face height percentage
- Face position percentage
- DPI (typically 300)

## Technology Stack

- **Backend**: FastAPI (Python)
- **AI/ML**: 
  - Google Vertex AI (Gemini) for image analysis
  - Rembg (U2-Net) for background removal
  - OpenCV for image processing
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Image Processing**: PIL/Pillow, NumPy
- **Deployment**: Google Cloud Run, Docker
- **Security**: Google Cloud Secret Manager, bcrypt

## Notes

- The application uses a hybrid approach for background removal:
  1. **Gemini** analyzes the image and provides guidance
  2. **OpenCV** performs initial segmentation
  3. **Rembg** (U2-Net) provides perfect segmentation cleanup
  4. **Custom algorithm** replaces background with country-specific color
- Passport photo dimensions are stored in millimeters and converted to pixels based on DPI
- Background colors are country-specific and enforced during processing
- The application works without Vertex AI using fallback methods, but results are better with Vertex AI enabled
- Make sure your Google Cloud project has billing enabled and appropriate API quotas

## License

MIT License
