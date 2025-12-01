"""
Google Cloud Secret Manager integration for secure API key management.
Falls back to environment variables for local development.
"""
import os
from typing import Optional

try:
    from google.cloud import secretmanager
    SECRET_MANAGER_AVAILABLE = True
except ImportError:
    SECRET_MANAGER_AVAILABLE = False
    # Only print warning in development, and only once
    import sys
    if 'pytest' not in sys.modules and not hasattr(sys, '_secret_manager_warned'):
        print("⚠ google-cloud-secret-manager not installed. Using environment variables only.")
        sys._secret_manager_warned = True


def get_secret(secret_id: str, project_id: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve a secret from Google Cloud Secret Manager or environment variable.
    
    Args:
        secret_id: The name of the secret (e.g., 'google-api-key')
        project_id: Google Cloud project ID (optional, will try to get from env)
        default: Default value if secret is not found
    
    Returns:
        The secret value or None if not found
    """
    # First, try environment variable (for local development)
    env_value = os.getenv(secret_id.upper().replace("-", "_"))
    if env_value:
        return env_value
    
    # Try Secret Manager if available and project_id is provided
    if SECRET_MANAGER_AVAILABLE and project_id:
        try:
            client = secretmanager.SecretManagerServiceClient()
            secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
            
            response = client.access_secret_version(request={"name": secret_name})
            secret_value = response.payload.data.decode("UTF-8")
            print(f"✓ Retrieved secret '{secret_id}' from Secret Manager")
            return secret_value
        except Exception as e:
            print(f"⚠ Could not retrieve secret '{secret_id}' from Secret Manager: {e}")
            print(f"   Falling back to environment variable or default")
    
    # Return default if provided
    return default


def get_google_api_key(project_id: Optional[str] = None) -> Optional[str]:
    """Get Google API key from Secret Manager or environment variable."""
    # Try lowercase hyphenated name first (preferred), then uppercase underscore (legacy)
    return (get_secret("google-api-key", project_id) or 
            get_secret("GOOGLE_API_KEY", project_id) or 
            os.getenv("GOOGLE_API_KEY"))


def get_secret_key(project_id: Optional[str] = None) -> Optional[str]:
    """Get secret key for authentication from Secret Manager or environment variable."""
    return get_secret("secret-key", project_id) or os.getenv("SECRET_KEY")


def get_vertex_ai_config(project_id: Optional[str] = None) -> dict:
    """Get Vertex AI configuration from Secret Manager or environment variables."""
    return {
        "project_id": project_id or os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
        "location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        "model_name": os.getenv("MODEL_NAME", "gemini-1.5-pro"),
        "imagen_model": os.getenv("IMAGEN_MODEL", "imagen-3.0-capability-001"),
    }

