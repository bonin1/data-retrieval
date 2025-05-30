"""
Configuration settings for Gjirafa50.com scraper
"""
import os
from typing import Dict, List, Tuple

# Base configuration
BASE_URL = "https://gjirafa50.com"
API_BASE_URL = "https://api.gjirafa50.com"

# Scraping configuration
SCRAPING_CONFIG = {
    # Request settings
    'timeout': 30,
    'max_retries': 5,
    'retry_delay': 2,
    'delay_range': (1, 3),  # Random delay between requests (min, max) seconds
    
    # Threading settings
    'max_workers': 8,
    'semaphore_limit': 10,
    
    # Anti-detection
    'rotate_user_agents': True,
    'use_proxies': False,  # Set to True if you have proxy list
    'proxy_file': 'proxies.txt',
    
    # Data collection
    'include_images': True,
    'include_reviews': True,
    'include_specifications': True,
    'max_pages_per_category': 100,
    'products_per_page': 24,
}

# Export configuration
EXPORT_CONFIG = {
    'output_dir': 'output',
    'filename_prefix': 'gjirafa_products',
    'timestamp_suffix': True,
    'formats': ['json', 'csv', 'excel'],
    'chunk_size': 1000,  # For large datasets
}

# Data validation
VALIDATION_CONFIG = {
    'required_fields': ['id', 'name', 'price', 'url'],
    'price_currency': 'EUR',
    'min_name_length': 3,
    'max_name_length': 200,
    'validate_urls': True,
}

# Categories to scrape (will be auto-discovered if empty)
TARGET_CATEGORIES = [
    # Electronics
    'telefona-dhe-aksesore',
    'kompjuter-dhe-laptop',
    'audio-video',
    'elektroshtepiak',
    
    # Fashion
    'veshje-femra',
    'veshje-meshkuj',
    'kepuce',
    'aksesore-mode',
    
    # Home & Garden
    'mobilje',
    'dekorime-shtepie',
    'kopesht-dhe-ballkon',
    
    # Sports & Recreation
    'sport-dhe-argëtim',
    'lodra',
    
    # Beauty & Health
    'bukuri-dhe-shendeti',
    'parfume',
]

# Selectors for web scraping (CSS/XPath selectors)
SELECTORS = {
    'product_grid': '.product-grid .product-item',
    'product_links': '.product-item a[href*="/product/"]',
    'pagination': '.pagination .page-link',
    'next_page': '.pagination .next',
    
    # Product detail selectors
    'product_name': 'h1.product-title, .product-name h1',
    'product_price': '.price, .product-price .current-price',
    'product_old_price': '.old-price, .product-price .old-price',
    'product_description': '.product-description, .description',
    'product_images': '.product-images img, .gallery img',
    'product_rating': '.rating .stars, .review-rating',
    'product_reviews_count': '.reviews-count, .rating-count',
    'product_availability': '.availability, .stock-status',
    'product_category': '.breadcrumb a, .category-path',
    'product_specifications': '.specifications table, .product-specs',
    'seller_info': '.seller-info, .vendor-info',
}

# Headers for requests
DEFAULT_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5,sq;q=0.3',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Cache-Control': 'max-age=0',
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'scraper.log',
    'console': True,
}

# Environment variables
def load_env_config():
    """Load configuration from environment variables"""
    return {
        'proxy_username': os.getenv('PROXY_USERNAME'),
        'proxy_password': os.getenv('PROXY_PASSWORD'),
        'api_key': os.getenv('GJIRAFA_API_KEY'),
        'database_url': os.getenv('DATABASE_URL'),
    }
