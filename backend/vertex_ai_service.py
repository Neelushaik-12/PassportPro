import os
import io
from PIL import Image
import numpy as np
import base64
import cv2
import urllib.request
from rembg import remove
from PIL import Image

# Try to import Vertex AI packages (optional)
try:
    from google.cloud import aiplatform
    from vertexai.preview.vision_models import ImageGenerationModel, Image as VertexImage
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    from google.cloud import storage
    VERTEX_AI_AVAILABLE = True
    GCS_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    GCS_AVAILABLE = False
    VertexImage = None
    print("Note: Vertex AI packages not available. Using fallback background removal.")

# Download face detection model if not available
FACE_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_WEIGHTS_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


class VertexAIService:
    """
    Service for interacting with Vertex AI to remove backgrounds from images.
    Uses Vertex AI's image generation/editing capabilities and computer vision.
    """
    
    def __init__(self):
        # Initialize Vertex AI
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        # Initialize GCS client for Imagen 3
        self.gcs_client = None
        self.gcs_bucket_name = None
        if project_id and GCS_AVAILABLE:
            try:
                self.gcs_client = storage.Client(project=project_id)
                # Use a bucket name based on project ID (will create if needed)
                self.gcs_bucket_name = f"{project_id}-imagen-temp"
                print(f"✓ GCS client initialized for bucket: {self.gcs_bucket_name}")
            except Exception as gcs_error:
                print(f"Warning: GCS client initialization failed: {gcs_error}")
                self.gcs_client = None
        
        if not project_id or not VERTEX_AI_AVAILABLE:
            # Allow fallback mode if project ID not set or Vertex AI not available
            if not project_id:
                print("Warning: GOOGLE_CLOUD_PROJECT_ID not set. Using fallback background removal.")
            self.use_vertex_ai = False
            self.model = None
            self.imagen_model = None
            self.use_imagen = False
            self.gemini_model = None
        else:
            try:
                # Initialize Vertex AI with explicit credentials handling
                # This works both locally (with ADC) and in Cloud Run (with service account)
                import google.auth
                from google.auth import default as get_default_credentials
                
                # Get default credentials (works with ADC or service account)
                credentials, _ = get_default_credentials()
                
                aiplatform.init(project=project_id, location=location, credentials=credentials)
                vertexai.init(project=project_id, location=location, credentials=credentials)
                
                # Initialize Gemini model for image analysis
                # Check for model name in env, default to gemini-1.5-pro (supports vision)
                # Vertex AI model names: gemini-1.5-pro, gemini-1.5-flash, gemini-pro
                model_name = os.getenv("MODEL_NAME", "gemini-1.5-pro")
                try:
                    self.gemini_model = GenerativeModel(model_name)
                    print(f"✓ Vertex AI Gemini model initialized: {model_name}")
                    self.use_vertex_ai = True
                except Exception as gemini_error:
                    print(f"Warning: Could not initialize Gemini model: {gemini_error}")
                    # Try fallback models
                    fallback_models = ["gemini-1.5-flash", "gemini-pro"]
                    self.gemini_model = None
                    for fallback in fallback_models:
                        try:
                            self.gemini_model = GenerativeModel(fallback)
                            print(f"✓ Using fallback Gemini model: {fallback}")
                            self.use_vertex_ai = True
                            break
                        except:
                            continue
                    if not self.gemini_model:
                        print(f"  → All Gemini models failed. Using OpenCV fallback.")
                        self.use_vertex_ai = False
                
                # Initialize Imagen 3 model for background removal/replacement
                try:
                    # Try Imagen 3 first (more advanced)
                    imagen_model_name = os.getenv("IMAGEN_MODEL", "imagen-3.0-capability-001")
                    try:
                        self.imagen_model = ImageGenerationModel.from_pretrained(imagen_model_name)
                        print(f"✓ Imagen 3 model initialized: {imagen_model_name}")
                        self.use_imagen = True
                    except Exception as imagen_error:
                        print(f"Note: Imagen 3 model ({imagen_model_name}) not available: {imagen_error}")
                        # Fallback to older version
                        try:
                            self.imagen_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
                            print(f"✓ Using fallback image generation model: imagegeneration@006")
                            self.use_imagen = True
                        except Exception as fallback_error:
                            print(f"Note: Image generation model not available: {fallback_error}")
                            self.imagen_model = None
                            self.use_imagen = False
                except Exception as img_error:
                    print(f"Note: Image generation model not available: {img_error}")
                    self.imagen_model = None
                    self.use_imagen = False
                
                # Keep old model reference for compatibility
                self.model = self.imagen_model
                    
            except Exception as e:
                print(f"Warning: Could not initialize Vertex AI: {e}. Using fallback method.")
                self.use_vertex_ai = False
                self.model = None
                self.imagen_model = None
                self.use_imagen = False
                self.gemini_model = None
        
        # Initialize face detector (using Haar Cascade as fallback)
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.face_detector_available = True
        except:
            self.face_detector_available = False
            print("Note: Face detector not available. Using center-based detection.")
    
    def _upload_to_gcs(self, image_bytes: bytes, filename: str) -> str:
        """
        Upload image bytes to GCS and return the GCS URI.
        Creates bucket if it doesn't exist.
        """
        if not self.gcs_client or not self.gcs_bucket_name:
            raise Exception("GCS client not initialized")
        
        try:
            # Get or create bucket
            try:
                bucket = self.gcs_client.bucket(self.gcs_bucket_name)
                # Check if bucket exists
                if not bucket.exists():
                    # Create bucket
                    bucket = self.gcs_client.create_bucket(self.gcs_bucket_name, location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"))
                    print(f"  ✓ Created GCS bucket: {self.gcs_bucket_name}")
            except Exception as bucket_error:
                # Bucket might already exist or other error
                bucket = self.gcs_client.bucket(self.gcs_bucket_name)
            
            # Upload file
            blob = bucket.blob(filename)
            blob.upload_from_string(image_bytes, content_type='image/png')
            
            # Return GCS URI
            gcs_uri = f"gs://{self.gcs_bucket_name}/{filename}"
            print(f"  ✓ Uploaded to GCS: {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            raise Exception(f"Failed to upload to GCS: {e}")
    
    def _delete_from_gcs(self, filename: str):
        """Delete a file from GCS."""
        if not self.gcs_client or not self.gcs_bucket_name:
            return
        
        try:
            bucket = self.gcs_client.bucket(self.gcs_bucket_name)
            blob = bucket.blob(filename)
            blob.delete()
        except Exception as e:
            # Don't fail if deletion fails
            print(f"  ⚠ Could not delete GCS file {filename}: {e}")
    
    async def remove_background(self, image: Image.Image, target_background_color: tuple = None) -> Image.Image:
        """
        Remove background from image using Vertex AI or advanced image processing.
        If Imagen 3 is available and target_background_color is provided, uses Imagen for direct replacement.
        """
        try:
            # Try Imagen 3 first if available and target color is provided
            if self.use_imagen and self.imagen_model and target_background_color and self.gcs_client:
                try:
                    return await self._remove_background_with_imagen(image, target_background_color)
                except Exception as imagen_error:
                    print(f"  ⚠ Imagen 3 background replacement failed: {imagen_error}")
                    print(f"  → Falling back to Gemini + OpenCV method...")
            
            # Use Gemini + OpenCV method (working well)
            if self.use_vertex_ai:
                return await self._remove_background_with_vertex_ai(image, target_background_color)
            else:
                # Apply background color if provided
                processed_image = self._remove_background_advanced(image)
                if target_background_color:
                    processed_image = processed_image.convert("RGBA")
                    bg = Image.new("RGB", processed_image.size, target_background_color)
                    bg.paste(processed_image, mask=processed_image.split()[3])
                    return bg
                return processed_image
        except Exception as e:
            print(f"Error in background removal: {e}. Using fallback method.")
            return self._remove_background_advanced(image)
    
    async def _remove_background_with_vertex_ai(self, image: Image.Image, target_background_color: tuple = None) -> Image.Image:
        """
        Hybrid background removal using:
        1. Gemini (optional) for analysis
        2. Vertex AI background removal
        3. Rembg (U2-Net) for perfect segmentation cleanup
        4. Background color replacement
        """
        if not self.gemini_model:
            # Fallback if Gemini not available
            processed_image = self._remove_background_advanced(image)
        else:
            try:
                # Convert image to bytes for Gemini
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                image_bytes = img_buffer.read()
                
                # Use Gemini to analyze the image and get segmentation guidance
                prompt = """Analyze this image and provide information about:
1. Where is the person/subject located in the image?
2. What is the background like?
3. Are there any shadows or complex elements?
4. What are the main colors of the subject vs background?

Provide a brief analysis that will help with background removal."""
                
                # Create image part for Gemini
                image_part = Part.from_data(image_bytes, mime_type="image/png")
                
                # Get analysis from Gemini
                response = self.gemini_model.generate_content([prompt, image_part])
                
                # Extract text from response - handle Vertex AI GenerationResponse structure
                analysis = ""
                try:
                    # Vertex AI GenerationResponse structure: response.candidates[0].content.parts[0].text
                    if hasattr(response, 'candidates') and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content'):
                            if hasattr(candidate.content, 'parts') and len(candidate.content.parts) > 0:
                                # Extract text from all parts
                                text_parts = []
                                for part in candidate.content.parts:
                                    if hasattr(part, 'text') and part.text:
                                        text_parts.append(part.text)
                                analysis = " ".join(text_parts)
                            elif hasattr(candidate.content, 'text'):
                                analysis = candidate.content.text
                        elif hasattr(candidate, 'text'):
                            analysis = candidate.text
                    
                    # Try direct text access (some response formats)
                    if not analysis and hasattr(response, 'text'):
                        analysis = response.text
                    
                    # Try parts structure directly on response
                    if not analysis and hasattr(response, 'parts'):
                        text_parts = []
                        for part in response.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                        analysis = " ".join(text_parts)
                    
                    # Try to get text using getattr (in case it's a property)
                    if not analysis:
                        try:
                            text_attr = getattr(response, 'text', None)
                            if text_attr and callable(text_attr):
                                analysis = text_attr()
                            elif text_attr:
                                analysis = str(text_attr)
                        except:
                            pass
                    
                    # Try converting response to string (last resort)
                    if not analysis:
                        try:
                            response_str = str(response)
                            if response_str and len(response_str) > 10:  # Meaningful content
                                analysis = response_str
                        except:
                            pass
                    
                    # Debug if still no text found
                    if not analysis:
                        print(f"  ⚠ Response structure debug:")
                        print(f"    Type: {type(response)}")
                        if hasattr(response, 'candidates'):
                            print(f"    Has candidates: {len(response.candidates) if response.candidates else 0}")
                            if response.candidates and len(response.candidates) > 0:
                                candidate = response.candidates[0]
                                print(f"    Candidate type: {type(candidate)}")
                                print(f"    Candidate attributes: {[attr for attr in dir(candidate) if not attr.startswith('_')][:10]}")
                                if hasattr(candidate, 'content'):
                                    print(f"    Content type: {type(candidate.content)}")
                                    if hasattr(candidate.content, 'parts'):
                                        print(f"    Parts count: {len(candidate.content.parts) if candidate.content.parts else 0}")
                                        if candidate.content.parts and len(candidate.content.parts) > 0:
                                            print(f"    First part type: {type(candidate.content.parts[0])}")
                                            print(f"    First part attributes: {[attr for attr in dir(candidate.content.parts[0]) if not attr.startswith('_')][:10]}")
                        
                except Exception as extract_error:
                    import traceback
                    print(f"  ⚠ Error extracting text from Gemini response: {extract_error}")
                    print(f"  Traceback: {traceback.format_exc()[-300:]}")
                
                if analysis:
                    print(f"  ✓ Gemini Analysis received: {analysis[:200]}...")  # Log first 200 chars
                    # Use Gemini's understanding to improve the OpenCV processing
                    processed_image = self._remove_background_with_gemini_guidance(image, analysis)
                else:
                    print(f"  ⚠ Gemini response received but no text found. Response type: {type(response)}")
                    print(f"  → Falling back to standard OpenCV method...")
                    processed_image = self._remove_background_advanced(image)
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Error using Vertex AI Gemini: {e}")
                print(f"Error details: {error_details[-500:]}")  # Last 500 chars of traceback
                print(f"  → Falling back to standard OpenCV method...")
                processed_image = self._remove_background_advanced(image)
        
        # Step 2: Apply Rembg cleanup for perfect segmentation
        print("  Vertex AI background removal complete.")
        print("  Running Rembg cleanup...")
        try:
            # Convert processed image to bytes for rembg
            img_buffer = io.BytesIO()
            processed_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            clean_bytes = remove(img_buffer.read())
            clean_image = Image.open(io.BytesIO(clean_bytes)).convert("RGBA")
            print("  Rembg cleanup complete.")
        except Exception as rembg_error:
            print(f"  ⚠ Rembg cleanup failed: {rembg_error}")
            print("  → Using processed image without Rembg cleanup...")
            clean_image = processed_image.convert("RGBA")
        
        # Step 4: Apply background color if provided
        if target_background_color:
            print(f"  Applying flat solid background: RGB{target_background_color}")
            bg = Image.new("RGB", clean_image.size, target_background_color)
            bg.paste(clean_image, mask=clean_image.split()[3])  # use alpha channel
            processed_image = bg
            print("  Final background replaced.")
        else:
            processed_image = clean_image
        
        return processed_image
    
    async def _remove_background_with_imagen(self, image: Image.Image, target_background_color: tuple) -> Image.Image:
        """
        Use Imagen 3 to directly replace background with target color.
        Uses google.genai API with automatic background detection (no mask needed!).
        """
        try:
            # Import google.genai API (different from vertexai.preview.vision_models)
            try:
                from google import genai
                from google.genai import types as genai_types
                from google.genai.types import (
                    RawReferenceImage,
                    MaskReferenceConfig,
                    MaskReferenceImage,
                    EditImageConfig
                )
            except ImportError as import_err:
                raise Exception(f"google.genai API not available. Install google-genai package. Error: {import_err}")
            
            # Get the genai client from main.py (it's already initialized there)
            # We need to initialize it here if not available
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            
            # Set environment for Vertex AI
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
            os.environ["GOOGLE_CLOUD_LOCATION"] = location
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
            
            # Initialize genai client
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                # Try to get from secret manager
                try:
                    from backend.secret_manager import get_google_api_key
                    google_api_key = get_google_api_key(project_id)
                except:
                    pass
            
            if google_api_key:
                genai_client = genai.Client(api_key=google_api_key)
            else:
                # Try without API key (uses ADC)
                genai_client = genai.Client()
            
            # Convert target color to description
            color_name = "white" if target_background_color == (255, 255, 255) else \
                        "royal blue" if target_background_color == (0, 35, 102) else \
                        "light grey" if target_background_color == (240, 240, 240) else \
                        f"RGB{target_background_color}"
            
            print(f"  → Using Imagen 3 (google.genai API) for background replacement to {color_name}...")
            
            # Save image to temp file and upload to genai files API
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image.save(tmp_file, format="PNG")
                tmp_path = tmp_file.name
            
            # Upload file to genai files API (since genai.Image doesn't exist)
            uploaded_file = None
            try:
                print(f"  → Uploading image to genai files API...")
                # Check what methods are available on files
                files_methods = [m for m in dir(genai_client.files) if not m.startswith('_')]
                print(f"  Debug: files API methods: {files_methods}")
                
                # Try different upload methods
                if hasattr(genai_client.files, 'upload'):
                    # Try with path parameter
                    try:
                        uploaded_file = genai_client.files.upload(path=tmp_path)
                        print(f"  ✓ File uploaded using path parameter")
                    except Exception as path_error:
                        print(f"  ⚠ Error with path parameter: {path_error}")
                        # Try with file parameter
                        try:
                            with open(tmp_path, 'rb') as f:
                                uploaded_file = genai_client.files.upload(file=f)
                            print(f"  ✓ File uploaded using file parameter")
                        except Exception as file_error:
                            print(f"  ⚠ Error with file parameter: {file_error}")
                            # Try with data/bytes
                            with open(tmp_path, 'rb') as f:
                                image_bytes = f.read()
                            try:
                                uploaded_file = genai_client.files.upload(data=image_bytes, mime_type='image/png')
                                print(f"  ✓ File uploaded using data parameter")
                            except Exception as data_error:
                                # Last resort: try just the path as positional arg
                                uploaded_file = genai_client.files.upload(tmp_path)
                                print(f"  ✓ File uploaded using positional argument")
                else:
                    raise Exception("files.upload method not found")
                
                # Log uploaded file info
                if uploaded_file:
                    file_attrs = [a for a in dir(uploaded_file) if not a.startswith('_')]
                    print(f"  Debug: Uploaded file attributes: {file_attrs[:10]}")
                    if hasattr(uploaded_file, 'name'):
                        print(f"  ✓ File uploaded: {uploaded_file.name}")
                    elif hasattr(uploaded_file, 'uri'):
                        print(f"  ✓ File uploaded: {uploaded_file.uri}")
                    else:
                        print(f"  ✓ File uploaded (type: {type(uploaded_file)})")
                        
            except Exception as upload_error:
                print(f"  ⚠ Error uploading file: {upload_error}")
                raise Exception(f"Cannot upload file to genai: {upload_error}")
            
            # Create RawReferenceImage using uploaded file
            # Note: Parameters are camelCase: referenceImage, referenceId, referenceType
            raw_ref = None
            try:
                # Try with camelCase parameter names
                raw_ref = RawReferenceImage(
                    referenceImage=uploaded_file,  # Use uploaded file object
                    referenceId=0,  # camelCase
                )
                print(f"  ✓ Created RawReferenceImage with uploaded file")
            except Exception as ref_error:
                print(f"  ⚠ Error with camelCase parameters: {ref_error}")
                # Try with snake_case (in case the API accepts both)
                try:
                    raw_ref = RawReferenceImage(
                        reference_image=uploaded_file,
                        reference_id=0,
                    )
                    print(f"  ✓ Created RawReferenceImage with snake_case")
                except Exception as ref_error2:
                    # Try inspecting what parameters are actually needed
                    try:
                        import inspect
                        sig = inspect.signature(RawReferenceImage)
                        params = list(sig.parameters.keys())
                        print(f"  Debug: RawReferenceImage parameters: {params}")
                        print(f"  Debug: Uploaded file type: {type(uploaded_file)}")
                        print(f"  Debug: Uploaded file attributes: {[a for a in dir(uploaded_file) if not a.startswith('_')][:10]}")
                    except:
                        pass
                    raise Exception(f"Cannot create RawReferenceImage: {ref_error2}")
            
            # Clean up temp file after API call (we'll do this in finally block)
            
            # MaskReferenceImage - automatic background detection (NO MASK NEEDED!)
            # Try camelCase first, then snake_case
            mask_ref = None
            try:
                mask_ref = MaskReferenceImage(
                    referenceId=1,  # camelCase
                    referenceImage=None,  # No mask needed - Imagen 3 auto-detects background!
                    config=MaskReferenceConfig(
                        maskMode="MASK_MODE_BACKGROUND"  # KEY SETTING - auto-detects background
                    )
                )
                print(f"  ✓ Created MaskReferenceImage with camelCase")
            except Exception as mask_error:
                print(f"  ⚠ Error with camelCase MaskReferenceImage: {mask_error}")
                # Try snake_case
                try:
                    mask_ref = MaskReferenceImage(
                        reference_id=1,
                        reference_image=None,
                        config=MaskReferenceConfig(
                            mask_mode="MASK_MODE_BACKGROUND"
                        )
                    )
                    print(f"  ✓ Created MaskReferenceImage with snake_case")
                except Exception as mask_error2:
                    # Try inspecting parameters
                    try:
                        import inspect
                        sig = inspect.signature(MaskReferenceImage)
                        print(f"  Debug: MaskReferenceImage parameters: {list(sig.parameters.keys())}")
                        sig2 = inspect.signature(MaskReferenceConfig)
                        print(f"  Debug: MaskReferenceConfig parameters: {list(sig2.parameters.keys())}")
                    except:
                        pass
                    raise Exception(f"Cannot create MaskReferenceImage: {mask_error2}")
            
            # Create prompt for background replacement
            prompt = f"plain uniform {color_name} background, passport photo style, no shadows, no textures, professional headshot"
            
            print(f"  → Calling Imagen 3 edit_image with automatic background detection...")
            
            try:
                # Call Imagen 3 using google.genai API
                # Try camelCase for config first
                edit_config = None
                try:
                    edit_config = EditImageConfig(
                        editMode="EDIT_MODE_BGSWAP",  # camelCase
                    )
                except Exception as config_error:
                    print(f"  ⚠ Error with camelCase EditImageConfig: {config_error}")
                    # Try snake_case
                    edit_config = EditImageConfig(
                        edit_mode="EDIT_MODE_BGSWAP",
                    )
                
                # Try camelCase first, then snake_case
                try:
                    response = genai_client.models.edit_image(
                        model="imagen-3.0-capability-001",
                        prompt=prompt,
                        referenceImages=[raw_ref, mask_ref],  # camelCase
                        config=edit_config,
                    )
                    print(f"  ✓ Called edit_image with camelCase parameters")
                except Exception as api_call_error:
                    print(f"  ⚠ Error with camelCase referenceImages: {api_call_error}")
                    # Try snake_case
                    response = genai_client.models.edit_image(
                        model="imagen-3.0-capability-001",
                        prompt=prompt,
                        reference_images=[raw_ref, mask_ref],  # snake_case
                        config=edit_config,
                    )
                    print(f"  ✓ Called edit_image with snake_case parameters")
                
                # Get the result
                if response and hasattr(response, 'generated_images') and len(response.generated_images) > 0:
                    output_image = response.generated_images[0].image
                    
                    # Convert to PIL Image
                    if hasattr(output_image, 'save'):
                        # If it's a genai Image, we need to get the bytes
                        img_buffer = io.BytesIO()
                        output_image.save(img_buffer)
                        img_buffer.seek(0)
                        final_image = Image.open(img_buffer)
                    elif isinstance(output_image, Image.Image):
                        final_image = output_image
                    else:
                        # Try to convert
                        final_image = Image.fromarray(np.array(output_image))
                    
                    print(f"  ✓ Imagen 3 background replacement successful")
                    return final_image
                else:
                    raise Exception("Imagen 3 did not return a valid image")
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except:
                    pass
                
        except Exception as api_error:
            # Clean up temp file on error too
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass
            print(f"  ⚠ Imagen 3 API error: {api_error}")
            raise api_error
            
        except Exception as e:
            print(f"  ⚠ Imagen 3 background replacement error: {e}")
            raise e
    
    def _remove_background_with_gemini_guidance(self, image: Image.Image, gemini_analysis: str) -> Image.Image:
        """
        Use Gemini's analysis to improve background removal.
        The analysis helps identify subject location and background characteristics.
        """
        # Convert PIL Image to OpenCV format
        img_array = np.array(image.convert("RGB"))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        
        # Detect face
        face_rect = self._detect_face(img_cv)
        
        # Use Gemini analysis to adjust parameters
        analysis_lower = gemini_analysis.lower()
        
        # Adjust GrabCut parameters based on Gemini's analysis
        iterations = 20
        if "complex" in analysis_lower or "busy" in analysis_lower:
            iterations = 25
        
        # Create improved mask with Gemini guidance
        mask = self._create_mask_with_gemini_guidance(img_cv, face_rect, iterations)
        
        # Ensure face is included
        if face_rect is not None:
            mask = self._ensure_face_included(mask, face_rect)
        
        # Remove shadows
        mask = self._remove_shadows(img_cv, mask)
        
        # Refine mask edges
        mask = self._refine_mask_edges(mask, img_cv)
        
        # Final check
        if face_rect is not None:
            mask = self._ensure_face_included(mask, face_rect)
        
        # Apply mask with feathering
        result = self._apply_mask_with_feathering(img_array, mask)
        
        return Image.fromarray(result, 'RGBA')
    
    def _create_mask_with_gemini_guidance(self, img_cv, face_rect, iterations):
        """Create mask using GrabCut with Gemini-guided parameters."""
        height, width = img_cv.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        
        if face_rect:
            x1, y1, x2, y2 = face_rect
            rect = (x1, y1, x2 - x1, y2 - y1)
        else:
            rect = (int(width * 0.15), int(height * 0.1), 
                   int(width * 0.7), int(height * 0.8))
        
        # Mark background areas
        mask[:] = cv2.GC_PR_BGD
        
        # Mark corners and edges as sure background
        corner_size = min(width, height) // 5
        edge_margin = min(width, height) // 10
        
        mask[:corner_size, :corner_size] = cv2.GC_BGD
        mask[:corner_size, -corner_size:] = cv2.GC_BGD
        mask[-corner_size:, :corner_size] = cv2.GC_BGD
        mask[-corner_size:, -corner_size:] = cv2.GC_BGD
        
        mask[:edge_margin, :] = cv2.GC_BGD
        mask[-edge_margin:, :] = cv2.GC_BGD
        mask[:, :edge_margin] = cv2.GC_BGD
        mask[:, -edge_margin:] = cv2.GC_BGD
        
        # Mark sure foreground in face area
        if face_rect:
            x1, y1, x2, y2 = face_rect
            x1 = max(0, x1 - 30)
            y1 = max(0, y1 - 30)
            x2 = min(width, x2 + 30)
            y2 = min(height, y2 + 100)
            mask[y1:y2, x1:x2] = cv2.GC_FGD
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Use Gemini-guided iteration count
            cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
            cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_EVAL)
            
            mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            
            # Post-process
            kernel = np.ones((5, 5), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=2)
            
        except Exception as e:
            print(f"GrabCut failed: {e}. Using color-based segmentation.")
            mask2 = self._color_based_segmentation(img_cv, face_rect)
        
        return mask2
    
    def _remove_background_advanced(self, image: Image.Image) -> Image.Image:
        """
        Advanced background removal using computer vision techniques.
        Uses face detection, improved GrabCut, shadow removal, and better segmentation.
        """
        # Convert PIL Image to OpenCV format
        img_array = np.array(image.convert("RGB"))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        original = img_cv.copy()
        
        height, width = img_cv.shape[:2]
        
        # Step 1: Detect face to better identify the subject
        face_rect = self._detect_face(img_cv)
        
        # Step 2: Create initial mask using improved method
        mask = self._create_improved_mask(img_cv, face_rect)
        
        # Step 3: Ensure face and upper body are always included (do this early)
        if face_rect is not None:
            mask = self._ensure_face_included(mask, face_rect)
        
        # Step 4: Remove shadows from the background mask
        mask = self._remove_shadows(img_cv, mask)
        
        # Step 5: Additional pass to remove shadows specifically around head/face
        if face_rect is not None:
            mask = self._remove_head_shadows(img_cv, mask, face_rect)
        
        # Step 6: Refine mask edges
        mask = self._refine_mask_edges(mask, img_cv)
        
        # Step 7: Final check - ensure face is still included after refinements
        if face_rect is not None:
            mask = self._ensure_face_included(mask, face_rect)
        
        # Step 8: Apply mask with feathering for smooth edges
        result = self._apply_mask_with_feathering(img_array, mask)
        
        return Image.fromarray(result, 'RGBA')
    
    def _detect_face(self, img_cv):
        """Detect face in the image to better identify the subject."""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        if self.face_detector_available:
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            if len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                # Expand the rectangle to include more of the body
                height, width = img_cv.shape[:2]
                return (
                    max(0, int(x - w * 0.3)),
                    max(0, int(y - h * 0.5)),
                    min(width, int(x + w * 1.6)),
                    min(height, int(y + h * 2.5))
                )
        
        # Fallback: assume center of image
        height, width = img_cv.shape[:2]
        return (int(width * 0.2), int(height * 0.1), int(width * 0.8), int(height * 0.8))
    
    def _create_improved_mask(self, img_cv, face_rect):
        """Create an improved mask using multiple techniques with aggressive background removal."""
        height, width = img_cv.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        
        # Initialize GrabCut with face rectangle
        if face_rect:
            x1, y1, x2, y2 = face_rect
            rect = (x1, y1, x2 - x1, y2 - y1)
        else:
            rect = (int(width * 0.15), int(height * 0.1), 
                   int(width * 0.7), int(height * 0.8))
        
        # More aggressive background marking - mark more areas as sure background
        mask[:] = cv2.GC_PR_BGD  # Probable background
        
        # Mark larger corner areas as sure background
        corner_size = min(width, height) // 5  # Larger corners
        edge_margin = min(width, height) // 10
        
        # All four corners
        mask[:corner_size, :corner_size] = cv2.GC_BGD
        mask[:corner_size, -corner_size:] = cv2.GC_BGD
        mask[-corner_size:, :corner_size] = cv2.GC_BGD
        mask[-corner_size:, -corner_size:] = cv2.GC_BGD
        
        # Mark edges as background
        mask[:edge_margin, :] = cv2.GC_BGD  # Top edge
        mask[-edge_margin:, :] = cv2.GC_BGD  # Bottom edge
        mask[:, :edge_margin] = cv2.GC_BGD  # Left edge
        mask[:, -edge_margin:] = cv2.GC_BGD  # Right edge
        
        # Mark sure foreground in face area (more conservatively)
        if face_rect:
            x1, y1, x2, y2 = face_rect
            # Expand significantly to include upper body
            x1 = max(0, x1 - 30)
            y1 = max(0, y1 - 30)
            x2 = min(width, x2 + 30)
            y2 = min(height, y2 + 100)  # Include more of the body
            mask[y1:y2, x1:x2] = cv2.GC_FGD  # Sure foreground
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Run GrabCut with mask initialization - more iterations
            cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 15, cv2.GC_INIT_WITH_MASK)
            
            # Run additional iterations to refine
            cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_EVAL)
            
            # Create final mask (sure foreground + probable foreground)
            mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            
            # Post-process: remove small background regions inside foreground
            kernel = np.ones((5, 5), np.uint8)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=2)
            
        except Exception as e:
            print(f"GrabCut failed: {e}. Using color-based segmentation.")
            mask2 = self._color_based_segmentation(img_cv, face_rect)
        
        return mask2
    
    def _color_based_segmentation(self, img_cv, face_rect):
        """Color-based segmentation as fallback."""
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Sample colors from face area if available
        if face_rect:
            x1, y1, x2, y2 = face_rect
            face_region = lab[y1:y2, x1:x2]
            if face_region.size > 0:
                # Get median color from face region
                face_color = np.median(face_region.reshape(-1, 3), axis=0)
            else:
                face_color = np.array([100, 128, 128])  # Default
        else:
            # Sample from center
            h, w = lab.shape[:2]
            center_region = lab[h//3:2*h//3, w//3:2*w//3]
            face_color = np.median(center_region.reshape(-1, 3), axis=0)
        
        # Calculate distance from face color
        diff = cv2.absdiff(lab, face_color.astype(np.uint8))
        distance = np.sqrt(np.sum(diff**2, axis=2))
        
        # Threshold to create mask
        threshold = np.percentile(distance, 60)  # Adjust based on distribution
        mask = (distance < threshold).astype(np.uint8) * 255
        
        return mask
    
    def _remove_shadows(self, img_cv, mask):
        """Aggressively remove shadows, especially behind head and around subject."""
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Identify background areas
        background_mask = 255 - mask
        
        # Method 1: Enhanced shadow detection using multiple techniques
        if np.any(background_mask > 0):
            # Use multiple blur levels to catch different shadow types
            blurred_small = cv2.GaussianBlur(gray, (5, 5), 0)
            blurred_large = cv2.GaussianBlur(gray, (15, 15), 0)
            
            # Find dark regions in background (shadows)
            bg_pixels = blurred_large[background_mask > 0]
            if len(bg_pixels) > 0:
                # More aggressive threshold to catch shadows
                dark_threshold = np.percentile(bg_pixels, 40)  # Increased from 30
                dark_regions = (blurred_large < dark_threshold).astype(np.uint8) * 255
                
                # Also check for gradual shadows (areas that are darker than surrounding)
                # Use Laplacian to detect shadow edges
                laplacian = cv2.Laplacian(blurred_small, cv2.CV_64F)
                laplacian_abs = np.abs(laplacian)
                shadow_edges = (laplacian_abs < np.percentile(laplacian_abs, 20)).astype(np.uint8) * 255
                
                # Combine dark regions and shadow edges
                dark_background = cv2.bitwise_and(dark_regions, background_mask)
                shadow_background = cv2.bitwise_and(shadow_edges, background_mask)
                
                # Combine both shadow detection methods
                all_shadows = cv2.bitwise_or(dark_background, shadow_background)
                
                # Expand shadow regions to catch shadow edges
                kernel = np.ones((5, 5), np.uint8)
                all_shadows = cv2.dilate(all_shadows, kernel, iterations=3)
                
                # Add shadows to foreground mask (removing from background)
                mask = cv2.bitwise_or(mask, all_shadows)
        
        # Method 2: Remove shadows specifically around head/face area
        # Find the head region (usually in upper portion of foreground)
        foreground_coords = np.where(mask > 0)
        if len(foreground_coords[0]) > 0:
            # Get bounding box of foreground
            min_y = np.min(foreground_coords[0])
            max_y = np.max(foreground_coords[0])
            min_x = np.min(foreground_coords[1])
            max_x = np.max(foreground_coords[1])
            
            # Head region is typically in upper 40% of foreground
            head_region_top = min_y
            head_region_bottom = min_y + int((max_y - min_y) * 0.4)
            
            # Create mask for head region
            head_region_mask = np.zeros((height, width), np.uint8)
            head_region_mask[head_region_top:head_region_bottom, min_x:max_x] = 255
            
            # Find dark areas around head (likely shadows)
            head_area_gray = gray[head_region_top:head_region_bottom, min_x:max_x]
            if head_area_gray.size > 0:
                # Get brightness of head area
                head_brightness = np.percentile(head_area_gray, 75)
                
                # Find areas around head that are significantly darker (shadows)
                head_region_bg = cv2.bitwise_and(background_mask, head_region_mask)
                if np.any(head_region_bg > 0):
                    head_bg_pixels = gray[head_region_bg > 0]
                    if len(head_bg_pixels) > 0:
                        # If background near head is much darker, it's likely a shadow
                        shadow_threshold = head_brightness * 0.6  # 40% darker
                        head_shadows = (gray < shadow_threshold).astype(np.uint8) * 255
                        head_shadows = cv2.bitwise_and(head_shadows, head_region_bg)
                        
                        # Expand and add to mask
                        kernel = np.ones((7, 7), np.uint8)
                        head_shadows = cv2.dilate(head_shadows, kernel, iterations=2)
                        mask = cv2.bitwise_or(mask, head_shadows)
        
        # Method 3: Use edge detection to find subject boundaries
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        
        # Method 4: Use watershed-like approach for better segmentation
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Combine with edges
        mask = cv2.bitwise_or(mask, sure_fg)
        mask = cv2.bitwise_or(mask, edges)
        
        # Method 5: Remove isolated background pixels near foreground using color similarity
        kernel = np.ones((9, 9), np.uint8)  # Larger kernel for better coverage
        mask_dilated = cv2.dilate(mask, kernel, iterations=3)
        
        # Find background pixels that are very close to foreground
        nearby_bg = cv2.bitwise_and(255 - mask, mask_dilated)
        
        # Use color similarity in LAB color space
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        
        # Sample foreground colors (especially from face/head area)
        fg_pixels = lab[mask > 0]
        if len(fg_pixels) > 0:
            fg_mean_color = np.mean(fg_pixels, axis=0)
            
            # Check nearby background pixels (vectorized)
            nearby_bg_coords = np.where(nearby_bg > 0)
            if len(nearby_bg_coords[0]) > 0:
                nearby_bg_colors = lab[nearby_bg_coords[0], nearby_bg_coords[1]]
                
                # Calculate color differences (vectorized)
                color_diffs = np.linalg.norm(nearby_bg_colors - fg_mean_color, axis=1)
                
                # More lenient threshold for shadows (they might be darker but similar hue)
                similar_mask = color_diffs < 50  # Increased from 40
                similar_coords = (
                    nearby_bg_coords[0][similar_mask],
                    nearby_bg_coords[1][similar_mask]
                )
                
                # Add similar pixels to foreground
                if len(similar_coords[0]) > 0:
                    mask[similar_coords] = 255
        
        # Method 6: Additional cleanup - remove small background holes and smooth
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    
    def _remove_head_shadows(self, img_cv, mask, face_rect):
        """Specifically remove shadows behind and around the head/face area."""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        x1, y1, x2, y2 = face_rect
        
        # Expand face rectangle significantly to include head area and shadow zones
        head_expand_x = int((x2 - x1) * 0.6)  # More horizontal expansion
        head_expand_y = int((y2 - y1) * 1.2)  # Much more expansion upward for head/shadow
        
        head_x1 = max(0, x1 - head_expand_x)
        head_y1 = max(0, y1 - head_expand_y)
        head_x2 = min(width, x2 + head_expand_x)
        head_y2 = min(height, y2 + int((y2 - y1) * 0.4))  # Some expansion downward
        
        # Create head region mask
        head_region = np.zeros((height, width), np.uint8)
        head_region[head_y1:head_y2, head_x1:head_x2] = 255
        
        # Get background in head region
        background_mask = 255 - mask
        head_bg = cv2.bitwise_and(background_mask, head_region)
        
        if np.any(head_bg > 0):
            # Sample face/head brightness (use higher percentile to avoid dark hair)
            face_region_gray = gray[y1:y2, x1:x2]
            if face_region_gray.size > 0:
                face_brightness = np.percentile(face_region_gray, 80)  # Use 80th percentile
                
                # Sample background brightness in head area
                head_bg_pixels = gray[head_bg > 0]
                if len(head_bg_pixels) > 0:
                    bg_brightness = np.percentile(head_bg_pixels, 40)
                    
                    # More aggressive: if background is darker than 75% of face brightness, it's a shadow
                    shadow_threshold = face_brightness * 0.75
                    
                    # Find all dark areas in head region background
                    head_shadows = (gray < shadow_threshold).astype(np.uint8) * 255
                    head_shadows = cv2.bitwise_and(head_shadows, head_bg)
                    
                    # Also check areas that are darker relative to face
                    relative_shadows = (gray < bg_brightness * 1.2).astype(np.uint8) * 255
                    relative_shadows = cv2.bitwise_and(relative_shadows, head_bg)
                    
                    # Combine both shadow detection methods
                    all_head_shadows = cv2.bitwise_or(head_shadows, relative_shadows)
                    
                    # Expand shadows more aggressively to catch all shadow edges
                    kernel = np.ones((11, 11), np.uint8)
                    all_head_shadows = cv2.dilate(all_head_shadows, kernel, iterations=3)
                    
                    # Add to foreground mask
                    mask = cv2.bitwise_or(mask, all_head_shadows)
        
        # Use gradient detection for gradual shadows
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Areas with low gradient (smooth) and low brightness are likely shadows
        gradient_threshold = np.percentile(gradient_magnitude, 25)
        brightness_threshold = np.percentile(gray, 30)
        
        low_gradient = (gradient_magnitude < gradient_threshold).astype(np.uint8) * 255
        low_brightness = (gray < brightness_threshold).astype(np.uint8) * 255
        
        # Combine conditions for shadow detection in head region
        shadow_candidates = cv2.bitwise_and(low_gradient, low_brightness)
        shadow_candidates = cv2.bitwise_and(shadow_candidates, head_bg)
        
        # Expand and add to mask
        if np.any(shadow_candidates > 0):
            kernel = np.ones((9, 9), np.uint8)
            shadow_candidates = cv2.dilate(shadow_candidates, kernel, iterations=3)
            mask = cv2.bitwise_or(mask, shadow_candidates)
        
        # Final pass: Use color similarity in head region
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        
        # Sample colors from face area
        face_colors = lab[y1:y2, x1:x2]
        if face_colors.size > 0:
            face_mean_color = np.mean(face_colors.reshape(-1, 3), axis=0)
            
            # Check background pixels in head region
            head_bg_coords = np.where(head_bg > 0)
            if len(head_bg_coords[0]) > 0:
                head_bg_colors = lab[head_bg_coords[0], head_bg_coords[1]]
                
                # Calculate color differences
                color_diffs = np.linalg.norm(head_bg_colors - face_mean_color, axis=1)
                
                # If color is similar but darker, it might be a shadow on subject
                # Use more lenient threshold for head area
                similar_shadows = color_diffs < 60
                similar_coords = (
                    head_bg_coords[0][similar_shadows],
                    head_bg_coords[1][similar_shadows]
                )
                
                # Also check if these pixels are darker than face
                similar_pixels_gray = gray[similar_coords]
                if len(similar_pixels_gray) > 0:
                    face_avg_brightness = np.mean(face_region_gray)
                    darker_mask = similar_pixels_gray < face_avg_brightness * 0.85
                    darker_coords = (
                        similar_coords[0][darker_mask],
                        similar_coords[1][darker_mask]
                    )
                    
                    if len(darker_coords[0]) > 0:
                        mask[darker_coords] = 255
        
        return mask
    
    def _refine_mask_edges(self, mask, img_cv):
        """Refine mask edges using morphological operations for clean, sharp edges."""
        # Fill holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Use connected components to remove small isolated background regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - mask, connectivity=8)
        
        # Remove small background components
        min_component_size = mask.shape[0] * mask.shape[1] * 0.005  # 0.5% of image (smaller threshold)
        for i in range(1, num_labels):  # Skip background label 0
            if stats[i, cv2.CC_STAT_AREA] < min_component_size:
                mask[labels == i] = 255  # Fill small holes
        
        # Create sharp, clean edges - use erosion/dilation to smooth but keep edges sharp
        # Don't use bilateral filter as it can create halos
        # Instead, use a small median filter to remove noise while keeping edges sharp
        mask = cv2.medianBlur(mask, 3)
        
        # Ensure binary mask (0 or 255)
        mask = (mask > 127).astype(np.uint8) * 255
        
        return mask
    
    def _ensure_face_included(self, mask, face_rect):
        """Ensure the face area is always included in the mask."""
        x1, y1, x2, y2 = face_rect
        # Expand face rectangle slightly
        x1 = max(0, x1 - 30)
        y1 = max(0, y1 - 30)
        x2 = min(mask.shape[1], x2 + 30)
        y2 = min(mask.shape[0], y2 + 30)
        
        # Force face area to be foreground
        mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def _apply_mask_with_feathering(self, img_array, mask):
        """Apply mask with minimal feathering for clean edges."""
        # First, clean up the mask to remove small holes and smooth edges
        # Use morphological operations to clean the mask
        kernel = np.ones((3, 3), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Convert mask to alpha channel with MINIMAL feathering (reduced from 15 to 5)
        mask_float = mask_cleaned.astype(np.float32) / 255.0
        
        # Apply very light Gaussian blur for minimal feathering (reduced blur)
        mask_feathered = cv2.GaussianBlur(mask_float, (5, 5), 0)  # Reduced from (15, 15)
        
        # Make edges sharper by thresholding the feathered mask
        # This prevents the halo effect
        mask_feathered = np.where(mask_feathered > 0.3, mask_feathered, 0)  # Remove very low alpha
        mask_feathered = np.where(mask_feathered < 0.7, mask_feathered, 1.0)  # Make high alpha fully opaque
        
        # Create RGBA image
        result = np.zeros((img_array.shape[0], img_array.shape[1], 4), dtype=np.uint8)
        result[:, :, :3] = img_array
        result[:, :, 3] = (mask_feathered * 255).astype(np.uint8)
        
        return result
    

