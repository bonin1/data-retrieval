import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ScraperConfig:
    BASE_URL = "https://gjirafa50.com"
    GJIRAFAMALL_URL = "https://gjirafamall.com"
    
    SPECIAL_OFFERS_API = "/Catalog/GetSpecialOfferProducts"
    RECOMMENDED_API = "/Catalog/GetRecommendedProductsByAI"
    SEARCH_ENDPOINT = "/search"
    SITEMAP_URL = "/sitemap.xml"
    CATEGORY_URLS = [
        "/kompjuter-laptop-monitor",
        "/kompjuter",
        "/gaming-kompjuter", 
        "/all-in-one-aio",
        "/mini-pc",
        "/laptop",
        "/monitor",
        "/gaming",
        "/tv-projektor",
        "/tv",
        "/projektor",
        "/audio-kufje",
        "/audio",
        "/kufje",
        "/telefon-tablet",
        "/telefon",
        "/tablet",
        "/smart-orë-aksesore",
        "/kamera-dron",
        "/kamera",
        "/dron"
    ]
    
    REQUEST_DELAY = 1.0
    MAX_RETRIES = 3
    TIMEOUT = 30
    
    HEADLESS = True
    IMPLICIT_WAIT = 10
    PAGE_LOAD_TIMEOUT = 30

    OUTPUT_DIR = "scraped_data"
    MAX_PRODUCTS_PER_FILE = 1000
    MAX_PRODUCTS_PER_CATEGORY = 100 
    
    MIN_PRICE = 0.01
    MAX_PRICE = 999999.99
    MIN_TITLE_LENGTH = 3
    MAX_TITLE_LENGTH = 500

SELECTORS = {
    "product_links": [
        ".product-item a",  
        "a[href*='/laptop-']", 
        "a[href*='/kompjuter-']", 
        "a[href*='/telefon-']",
        "a[href*='/tv-']", 
        "a[href*='/monitor-']",
        "a[href*='/kufje-']", 
        ".product-card a",
        "[data-testid='product-link']"
    ],
    
    "title": [
        "h1.product-title",
        ".product-name h1",
        "[data-testid='product-title']",
        ".product-header h1",
        "h1"
    ],
      "price": [
        ".product-price",
        ".prices",
        "[class*='price-value']",
        ".current-price",
        ".price-current",
        ".price-now",
        ".price"
    ],
      "original_price": [
        ".non-discounted-price",
        ".line-through",
        "[class*='price-value']",
        ".price-original",
        ".old-price",
        ".price-before",
        ".original-price"
    ],
    
    "description": [
        ".product-description",
        ".product-details",
        "[data-testid='product-description']",
        ".description",
        ".product-info"
    ],
      "images": [
        ".product-details img",
        ".details img", 
        "img[src*='iqq6kf0xmf.gjirafa.net']",
        "img[src*='gjirafa']",
        ".product-gallery img",
        ".product-images img",
        "[data-testid='product-image']",
        ".gallery img",
        ".product-photo img",
        "img[alt='']" 
    ],

    "specifications": [
        "[class*='spec']",
        ".product-details",
        ".details",
        ".specifications table",
        ".product-specs",
        ".spec-table",
        ".attributes",
        ".product-attributes"
    ],
    
    "availability": [
        ".stock-status",
        ".availability",
        "[data-testid='stock-status']",
        ".in-stock",
        ".stock-info"
    ],
    
    "brand": [
        ".product-brand",
        ".brand-name",
        "[data-testid='brand']",
        ".manufacturer"
    ],
    
    "category": [
        ".breadcrumb",
        ".category-path",
        ".product-category",
        "[data-testid='breadcrumb']"
    ],
      "rating": [
        ".ratingsAndReviews",
        "[class*='rating']",
        "[class*='star']",
        ".product-rating",
        "[data-testid='rating']",
        ".stars"
    ],
      "reviews_count": [
        ".product-reviews-overview",
        ".product-no-reviews",
        "[class*='ratingsAndReviews']",
        ".reviews-count",
        ".review-count",
        "[data-testid='reviews-count']"
    ]
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

CATEGORY_URLS = [
    "/kategoria/teknologji",
    "/kategoria/mode",
    "/kategoria/shtepi-kopesht",
    "/kategoria/sport-outdoor",
    "/kategoria/bukuri-shendeti",
    "/kategoria/foshnja-femije",
    "/kategoria/automjete",
    "/kategoria/liber-muzike"
]

def get_env_var(name: str, default: str = "") -> str:
    """Get environment variable with default value"""
    return os.getenv(name, default)

CHROME_OPTIONS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-web-security",
    "--disable-features=VizDisplayCompositor",
    "--window-size=1920,1080",
    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
]
