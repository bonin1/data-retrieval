"""
Gjirafa50.com Scraper Configuration
===================================
Central configuration file for the web scraper.
Contains selectors, headers, and scraping parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ScraperConfig:
    """Main configuration class for the scraper"""
    
    # Base URLs
    BASE_URL: str = "https://gjirafa50.com"
    GJIRAFAMALL_URL: str = "https://gjirafamall.com"
    
    # API Endpoints
    SPECIAL_OFFERS_API: str = "/Catalog/GetSpecialOfferProducts"
    RECOMMENDED_API: str = "/Catalog/GetRecommendedProductsByAI"
    SEARCH_ENDPOINT: str = "/search"
    SITEMAP_URL: str = "/sitemap.xml"
    
    # Predefined category URLs for fallback
    CATEGORY_URLS: List[str] = field(default_factory=lambda: [
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
        "/televizor",
        "/projektor",
        "/audio-kufje",
        "/audio",
        "/kufje",
        "/telefon-tablet",
        "/telefon",
        "/celular",
        "/tablet",
        "/smart-orë-aksesore",
        "/kamera-dron",
        "/kamera",
        "/dron"
    ])
    
    # Request settings
    REQUEST_DELAY: float = 1.5  # Increased for politeness
    MAX_RETRIES: int = 3
    TIMEOUT: int = 30
    
    # Selenium settings
    HEADLESS: bool = True
    IMPLICIT_WAIT: int = 10
    PAGE_LOAD_TIMEOUT: int = 45  # Increased for slow pages
    
    # Output settings
    OUTPUT_DIR: str = "scraped_data"
    MAX_PRODUCTS_PER_FILE: int = 1000
    MAX_PRODUCTS_PER_CATEGORY: int = 100
    
    # Validation settings
    MIN_PRICE: float = 0.01
    MAX_PRICE: float = 999999.99
    MIN_TITLE_LENGTH: int = 3
    MAX_TITLE_LENGTH: int = 500

# =============================================================================
# CSS Selectors for Data Extraction
# =============================================================================
# These selectors are ordered by specificity - most specific first
# The scraper will try each selector in order until one matches

SELECTORS = {
    # Product links on category pages
    "product_links": [
        ".product-item a[href]",
        ".product-card a[href]",
        ".product-box a[href]",
        "[data-product-id] a[href]",
        "a[href*='/laptop-']",
        "a[href*='/kompjuter-']",
        "a[href*='/telefon-']",
        "a[href*='/televizor-']",
        "a[href*='/tv-']",
        "a[href*='/monitor-']",
        "a[href*='/kufje-']",
        "a[href*='/tablet-']",
        "a[href*='/gaming-']",
        "a[href*='/audio-']",
    ],
    
    # Product title selectors
    "title": [
        "h1.product-title",
        ".product-name h1",
        ".product-header h1",
        "h1[itemprop='name']",
        "[data-testid='product-title']",
        "h1",
    ],
    
    # Current/sale price selectors
    "price": [
        ".prices .text-green-600",          # Discounted price (green color)
        ".product-price .current-price",
        ".product-price .sale-price",
        ".prices [class*='current']",
        ".price-current",
        ".price-now",
        ".product-price",
        ".prices",
        "[itemprop='price']",
    ],
    
    # Original/crossed-out price selectors
    "original_price": [
        ".prices .line-through",            # Crossed out original price
        ".non-discounted-price",
        "[class*='line-through']",
        ".price-original",
        ".old-price",
        ".was-price",
        ".price-before",
    ],
    
    # Product description selectors
    "description": [
        ".product-description",
        ".product-details .description",
        "[itemprop='description']",
        ".description",
        ".product-info",
        ".product-content",
    ],
    
    # Product images - only target actual product images, not site assets
    "images": [
        ".product-gallery img[src*='iqq6kf0xmf.gjirafa.net/images']",
        ".product-images img[src*='iqq6kf0xmf.gjirafa.net/images']",
        ".product-slider img[src*='iqq6kf0xmf.gjirafa.net/images']",
        ".product-details img[src*='iqq6kf0xmf.gjirafa.net/images']",
        "[class*='gallery'] img[src*='iqq6kf0xmf.gjirafa.net/images']",
        "img[src*='iqq6kf0xmf.gjirafa.net/images']",
    ],
    
    # Product specifications table
    "specifications": [
        ".specifications table",
        ".product-specs table",
        ".spec-table",
        "[class*='specification'] table",
        ".product-details table",
        ".details table",
        ".attributes table",
        "dl.specifications",
    ],
    
    # Stock/availability status
    "availability": [
        ".stock-status",
        ".availability",
        "[class*='stock']",
        ".in-stock",
        ".out-of-stock",
        "[data-testid='stock-status']",
    ],
    
    # Brand name
    "brand": [
        ".product-brand",
        ".brand-name",
        "[itemprop='brand']",
        ".manufacturer",
        "[data-testid='brand']",
    ],
    
    # Category/breadcrumb
    "category": [
        ".breadcrumb",
        ".breadcrumbs",
        "[class*='breadcrumb']",
        "nav[aria-label='breadcrumb']",
        ".category-path",
    ],
    
    # Rating and reviews
    "rating": [
        ".ratingsAndReviews",
        "[itemprop='ratingValue']",
        ".product-rating",
        "[class*='rating']",
        "[class*='star']",
    ],
    
    # Review count
    "reviews_count": [
        ".product-reviews-overview",
        ".product-no-reviews",
        "[itemprop='reviewCount']",
        "[class*='review-count']",
        ".reviews-count",
    ],
    
    # Load more button for pagination
    "load_more": [
        "button.load-more-products-btn",
        "button[data-page-infinite]",
        "button[class*='load-more']",
        "[onclick*='loadProductsAjax']",
    ],
}

# =============================================================================
# HTTP Headers for Requests
# =============================================================================
# Modern browser headers to avoid detection

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,sq;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0',
}

# Alternative category URLs (for gjirafamall.com if needed)
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


# =============================================================================
# Chrome WebDriver Options
# =============================================================================
# Options for Selenium Chrome driver

CHROME_OPTIONS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-extensions",
    "--disable-infobars",
    "--disable-notifications",
    "--disable-popup-blocking",
    "--window-size=1920,1080",
    "--start-maximized",
    "--disable-blink-features=AutomationControlled",
]


# =============================================================================
# Image Filtering Configuration
# =============================================================================
# Patterns for filtering product images from site assets

IMAGE_EXCLUDE_PATTERNS = [
    r'gjProductsSVG',
    r'/Content/images/',
    r'logo',
    r'\.svg$',
    r'flags/',
    r'calculator\.png',
    r'teb\.svg',
    r'rbko',
    r'kepPayment',
    r'gjirafaAds',
    r'gjirafamall',
    r'gjirafatravel',
    r'gjirafavideo',
    r'gjirafapikbiz',
]

# Known placeholder/banner image UUIDs to exclude
EXCLUDED_IMAGE_UUIDS = [
    'b6db299a-d4e4-4afe-8226-ab5c21a7fd53',
    'ce80ffe8-09e5-4635-8c44-45dced59df48',
    'e6fd61b7-0171-43c1-aaf9-8b49c445a0c1',
    'a4721784-e748-4ad7-9b85-d8011377962d',
    '28acc4e8-dff5-4ba7-8bbd-9a446acb5dc7',
    'e664607f-8727-4844-aa4f-dfaf6a220143',
    '8f204494-19fd-4e2d-8651-fcd76188184a',
]

# Product image CDN pattern
PRODUCT_IMAGE_CDN = 'iqq6kf0xmf.gjirafa.net/images/'


# =============================================================================
# Known Brands for Extraction
# =============================================================================

KNOWN_BRANDS = [
    # Computer brands
    'Lenovo', 'HP', 'Dell', 'ASUS', 'Acer', 'MSI', 'Apple', 'Alienware', 
    'Razer', 'Gigabyte', 'Microsoft', 'Huawei', 'Toshiba', 'Fujitsu',
    # Phone brands
    'Samsung', 'iPhone', 'Xiaomi', 'OnePlus', 'Google', 'Sony', 'Nokia', 
    'Oppo', 'Vivo', 'Realme', 'Motorola', 'Honor', 'ZTE', 'Nothing',
    # TV/Monitor brands
    'LG', 'TCL', 'Hisense', 'Philips', 'Panasonic', 'Sharp',
    'AOC', 'BenQ', 'ViewSonic', 'iiyama', 'GoGEN', 'Vivax', 'Grundig',
    'Vox', 'Tesla', 'Thomson', 'Blaupunkt', 'Haier', 'Skyworth',
    # Gaming brands
    'Nintendo', 'PlayStation', 'Xbox', 'Steam', 'Valve', 'Corsair', 
    'Logitech', 'SteelSeries', 'HyperX', 'Roccat', 'Turtle Beach',
    # Audio brands
    'Bose', 'JBL', 'Sennheiser', 'Audio-Technica', 'Beyerdynamic', 
    'AKG', 'Shure', 'Marshall', 'Beats', 'Skullcandy', 'Jabra', 'Anker',
    'Harman Kardon', 'Bang & Olufsen', 'Sonos', 'Ultimate Ears',
    # Peripherals
    'LogiLink', 'Baseus', 'Ugreen', 'Belkin', 'SanDisk', 'Kingston', 
    'Crucial', 'Western Digital', 'Seagate', 'TP-Link', 'Netgear',
    # Gaming furniture
    'Cougar', 'Genesis', 'Trust', 'Marvo', 'Redragon', 'Havit', 'Hama',
]
