"""
Passport photo specifications for different countries.
Dimensions are stored in millimeters (mm) and converted to pixels based on DPI.
Standard conversion: pixels = (mm * DPI) / 25.4
"""

def mm_to_pixels(mm: float, dpi: int) -> int:
    """Convert millimeters to pixels based on DPI."""
    return int((mm * dpi) / 25.4)

PASSPORT_SPECS = {
    "US": {
        "name": "United States",
        "width_mm": 50.8,  # 2 inches
        "height_mm": 50.8,  # 2 inches
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 50,  # Face should be 50-69% of image height
        "face_position_percent": 50,  # Face center at 50% from top
    },
    "UK": {
        "name": "United Kingdom",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (240, 240, 240),  # Light grey (off-white) - UK accepts light grey
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "CA": {
        "name": "Canada",
        "width_mm": 50,  # Canada standard: 50mm x 70mm
        "height_mm": 70,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "AU": {
        "name": "Australia",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (240, 240, 240),  # Light grey (off-white) - Australia accepts light grey
        "face_height_percent": 75,  # Face height: 32-36mm (75% of 45mm = 33.75mm, within range)
        "face_position_percent": 50,
    },
    "IN": {
        "name": "India",
        "width_mm": 51,
        "height_mm": 51,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "DE": {
        "name": "Germany",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (240, 240, 240),  # Light grey (off-white) - Germany accepts light grey
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "FR": {
        "name": "France",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (240, 240, 240),  # Light grey (off-white) - France accepts light grey
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "IT": {
        "name": "Italy",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "ES": {
        "name": "Spain",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "JP": {
        "name": "Japan",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "CN": {
        "name": "China",
        "width_mm": 33,  # China standard: 33mm x 48mm
        "height_mm": 48,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "BR": {
        "name": "Brazil",
        "width_mm": 30,  # Brazil standard: 30mm x 40mm
        "height_mm": 40,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "MX": {
        "name": "Mexico",
        "width_mm": 50.8,  # Mexico standard: 2x2 inches (51mm x 51mm)
        "height_mm": 50.8,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "RU": {
        "name": "Russia",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "KR": {
        "name": "South Korea",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "SG": {
        "name": "Singapore",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "NZ": {
        "name": "New Zealand",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "NL": {
        "name": "Netherlands",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "SE": {
        "name": "Sweden",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "CH": {
        "name": "Switzerland",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (240, 240, 240),  # Light grey - Switzerland requires light grey, not white
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "AE": {
        "name": "United Arab Emirates",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "SA": {
        "name": "Saudi Arabia",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "ZA": {
        "name": "South Africa",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "AR": {
        "name": "Argentina",
        "width_mm": 40,  # Argentina standard: 40mm x 40mm (square)
        "height_mm": 40,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "TR": {
        "name": "Turkey",
        "width_mm": 50,  # 5cm x 6cm (Turkey standard)
        "height_mm": 60,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "TH": {
        "name": "Thailand",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "MY": {
        "name": "Malaysia",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "PH": {
        "name": "Philippines",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # WHITE background (official)
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "ID": {
        "name": "Indonesia",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "VN": {
        "name": "Vietnam",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "PK": {
        "name": "Pakistan",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "BD": {
        "name": "Bangladesh",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "EG": {
        "name": "Egypt",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "NG": {
        "name": "Nigeria",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
    "KE": {
        "name": "Kenya",
        "width_mm": 35,
        "height_mm": 45,
        "dpi": 300,
        "background_color": (255, 255, 255),  # White
        "face_height_percent": 70,
        "face_position_percent": 50,
    },
}

def get_passport_specs(country_code: str):
    """
    Get passport specifications for a country code.
    Returns specs with both mm and pixel dimensions.
    """
    country_code = country_code.upper().strip()
    specs = PASSPORT_SPECS.get(country_code, PASSPORT_SPECS["US"]).copy()
    
    # Convert mm to pixels
    dpi = specs["dpi"]
    specs["width"] = mm_to_pixels(specs["width_mm"], dpi)
    specs["height"] = mm_to_pixels(specs["height_mm"], dpi)
    
    return specs

def get_all_countries():
    """Get list of all available countries."""
    return [
        {"code": code, "name": specs["name"]}
        for code, specs in sorted(PASSPORT_SPECS.items(), key=lambda x: x[1]["name"])
    ]

