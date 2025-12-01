"""
Email notification module for Passport Pro.
Sends email notifications to admin when photos are processed.
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional
import logging

# Try to import secret_manager for email credentials
try:
    from backend.secret_manager import get_secret
except ImportError:
    try:
        from secret_manager import get_secret
    except ImportError:
        get_secret = None

logger = logging.getLogger(__name__)


def get_admin_email(project_id: Optional[str] = None) -> Optional[str]:
    """Get admin email from Secret Manager or environment variable."""
    if get_secret:
        try:
            admin_email = get_secret("admin-email", project_id)
            if admin_email:
                return admin_email
        except:
            pass
    
    # Fallback to environment variable
    return os.getenv("ADMIN_EMAIL", "admin@passportpro.com")


def get_smtp_config(project_id: Optional[str] = None) -> dict:
    """Get SMTP configuration from Secret Manager or environment variables."""
    config = {}
    
    # Try to get from Secret Manager
    if get_secret:
        try:
            config["smtp_server"] = get_secret("smtp-server", project_id) or os.getenv("SMTP_SERVER", "smtp.gmail.com")
            config["smtp_port"] = int(get_secret("smtp-port", project_id) or os.getenv("SMTP_PORT", "587"))
            config["smtp_username"] = get_secret("smtp-username", project_id) or os.getenv("SMTP_USERNAME", "")
            config["smtp_password"] = get_secret("smtp-password", project_id) or os.getenv("SMTP_PASSWORD", "")
        except:
            pass
    
    # Fallback to environment variables
    if not config:
        config = {
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "smtp_username": os.getenv("SMTP_USERNAME", ""),
            "smtp_password": os.getenv("SMTP_PASSWORD", ""),
        }
    
    return config


def send_photo_processed_notification(
    username: Optional[str],
    user_email: Optional[str],
    country: str,
    country_name: str,
    project_id: Optional[str] = None
) -> bool:
    """
    Send email notification to admin when a photo is processed.
    
    Args:
        username: Username of the user who processed the photo
        user_email: Email of the user (if available)
        country: Country code (e.g., "US", "AU")
        country_name: Full country name (e.g., "United States", "Australia")
        project_id: Google Cloud project ID for Secret Manager
    
    Returns:
        True if email sent successfully, False otherwise
    """
    try:
        # Get admin email
        admin_email = get_admin_email(project_id)
        if not admin_email or admin_email == "admin@passportpro.com":
            logger.warning("Admin email not configured. Skipping email notification.")
            return False
        
        # Get SMTP configuration
        smtp_config = get_smtp_config(project_id)
        
        # Check if SMTP is configured
        if not smtp_config.get("smtp_username") or not smtp_config.get("smtp_password"):
            logger.warning("SMTP credentials not configured. Skipping email notification.")
            return False
        
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = smtp_config["smtp_username"]
        msg['To'] = admin_email
        msg['Subject'] = f"📸 Passport Photo Processed - {country_name}"
        
        # Create email body
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        body = f"""
Passport Photo Processing Notification

A new passport photo has been processed through Passport Pro.

User Details:
-----------
Username: {username if username else 'Anonymous/Guest User'}
Email: {user_email if user_email else 'Not provided'}
Timestamp: {timestamp}

Photo Details:
-----------
Country: {country_name} ({country})
Processed At: {timestamp}

---
This is an automated notification from Passport Pro.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        try:
            with smtplib.SMTP(smtp_config["smtp_server"], smtp_config["smtp_port"]) as server:
                server.starttls()
                server.login(smtp_config["smtp_username"], smtp_config["smtp_password"])
                server.send_message(msg)
            
            logger.info(f"Email notification sent to {admin_email} for photo processed by {username}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error in send_photo_processed_notification: {e}")
        return False

