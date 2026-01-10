"""
Gjirafa50.com Web Scraper - Production-Ready Implementation
===========================================================
A robust, efficient scraper for extracting product data from gjirafa50.com
with best practices: proper error handling, rate limiting, retry logic,
and accurate data extraction.

Author: Data Retrieval Project
Version: 2.0.0
"""

import requests
import time
import json
import logging
import re
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False

try:
    from fake_useragent import UserAgent
    HAS_FAKE_UA = True
except ImportError:
    HAS_FAKE_UA = False

from tqdm import tqdm

from config import ScraperConfig, SELECTORS, HEADERS, CHROME_OPTIONS
from utils import DataValidator, DataExporter, URLHelper

# Configure logging with both console and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ScrapedProduct:
    """Data class for product information with validation"""
    url: str
    title: str = ""
    price: Optional[float] = None
    original_price: Optional[float] = None
    discount_percentage: Optional[float] = None
    description: str = ""
    brand: str = ""
    category: str = ""
    availability: str = ""
    images: List[str] = field(default_factory=list)
    specifications: Dict[str, str] = field(default_factory=dict)
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    sku: str = ""
    scraped_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            'url': self.url,
            'title': self.title,
            'price': self.price,
            'original_price': self.original_price,
            'discount_percentage': self.discount_percentage,
            'description': self.description,
            'brand': self.brand,
            'category': self.category,
            'availability': self.availability,
            'images': self.images,
            'image_count': len(self.images),
            'main_image': self.images[0] if self.images else None,
            'specifications': self.specifications,
            'rating': self.rating,
            'reviews_count': self.reviews_count,
            'sku': self.sku,
            'scraped_at': self.scraped_at or datetime.now().isoformat()
        }


class RateLimiter:
    """Rate limiter to prevent overwhelming the server"""
    
    def __init__(self, requests_per_second: float = 1.0, burst_limit: int = 5):
        self.requests_per_second = requests_per_second
        self.burst_limit = burst_limit
        self.request_times: List[float] = []
    
    def acquire(self):
        """Synchronous rate limiting with jitter"""
        now = time.time()
        # Remove old request times (older than 1 second)
        self.request_times = [t for t in self.request_times if now - t < 1.0]
        
        if len(self.request_times) >= self.burst_limit:
            sleep_time = 1.0 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.request_times = self.request_times[1:]
        
        self.request_times.append(time.time())
        
        # Add random jitter to appear more human-like
        time.sleep(random.uniform(0.1, 0.3))


class RetryHandler:
    """Handles retry logic with exponential backoff"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = random.uniform(0, delay * 0.1)
        return delay + jitter
    
    def execute(self, func, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.get_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
        
        raise last_exception


class ImageFilter:
    """Filter and validate product images - excludes site assets, logos, icons"""
    
    # Patterns for images to EXCLUDE (site assets, logos, icons, etc.)
    EXCLUDE_PATTERNS = [
        r'gjProductsSVG',           # SVG product icons
        r'/Content/images/',        # Site content images
        r'logo',                    # Any logo images
        r'\.svg$',                  # All SVG files
        r'flags/',                  # Country flags
        r'calculator\.png',         # Calculator icon
        r'teb\.svg',                # Bank logo
        r'rbko',                    # Bank logo
        r'kepPayment',              # Payment logo
        r'gjirafaAds',              # Ads logo
        r'gjirafamall',             # Mall logo
        r'gjirafatravel',           # Travel logo
        r'gjirafavideo',            # Video logo
        r'gjirafapikbiz',           # Business logo
        r'gjirafa50\.svg',          # Main site logo
        r'gjirafa\.svg',            # Site logo
        # Known placeholder/banner UUIDs
        r'b6db299a-d4e4-4afe-8226-ab5c21a7fd53',
        r'ce80ffe8-09e5-4635-8c44-45dced59df48',
        r'e6fd61b7-0171-43c1-aaf9-8b49c445a0c1',
        r'a4721784-e748-4ad7-9b85-d8011377962d',
        r'28acc4e8-dff5-4ba7-8bbd-9a446acb5dc7',
        r'e664607f-8727-4844-aa4f-dfaf6a220143',
        r'8f204494-19fd-4e2d-8651-fcd76188184a',
    ]
    
    # Minimum dimensions for product images (exclude tiny thumbnails)
    MIN_IMAGE_WIDTH = 150
    
    @classmethod
    def is_product_image(cls, url: str) -> bool:
        """Check if URL is likely a product image"""
        if not url:
            return False
        
        url_lower = url.lower()
        
        # Check exclude patterns
        for pattern in cls.EXCLUDE_PATTERNS:
            if re.search(pattern, url_lower, re.IGNORECASE):
                return False
        
        # Must be from the product CDN and be an actual image
        if 'iqq6kf0xmf.gjirafa.net/images/' in url_lower:
            # Check it's an image format
            if any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                # Check for small thumbnail indicators
                width_match = re.search(r'\?w=(\d+)', url_lower)
                if width_match:
                    width = int(width_match.group(1))
                    if width < cls.MIN_IMAGE_WIDTH:
                        return False
                return True
        
        return False
    
    @classmethod
    def filter_product_images(cls, urls: List[str]) -> List[str]:
        """Filter list of URLs to only include product images"""
        seen = set()
        filtered = []
        
        for url in urls:
            # Normalize URL
            normalized = url.split('?')[0] if '?' in url else url  # Remove query params for dedup
            
            if cls.is_product_image(url) and normalized not in seen:
                seen.add(normalized)
                filtered.append(url)
        
        return filtered
    
    @classmethod
    def get_main_product_image(cls, urls: List[str]) -> Optional[str]:
        """Get the best main product image from a list"""
        product_images = cls.filter_product_images(urls)
        
        if not product_images:
            return None
        
        # Prefer images without query parameters (usually higher quality)
        for img in product_images:
            if '?' not in img:
                return img
        
        return product_images[0]


class GjirafaScraper:
    """
    Production-ready scraper for Gjirafa50.com
    
    Features:
    - Robust error handling with retry logic
    - Rate limiting to prevent server overload
    - Smart image filtering (excludes logos, icons, banners)
    - Accurate data extraction with multiple selector fallbacks
    - Duplicate detection across sessions
    - Progress tracking with tqdm
    - Comprehensive logging
    
    Usage:
        with GjirafaScraper() as scraper:
            products = scraper.discover_product_urls("https://gjirafa50.com/tv")
            for url in products:
                data = scraper.extract_product_data(url)
                if data:
                    scraper.products.append(data)
            scraper.export_data(formats=['json', 'csv'])
    """
    
    # Known brand patterns for extraction
    KNOWN_BRANDS = [
        # Computer brands
        'Lenovo', 'HP', 'Dell', 'ASUS', 'Acer', 'MSI', 'Apple', 'Alienware', 
        'Razer', 'Gigabyte', 'Microsoft', 'Huawei', 'Toshiba', 'Fujitsu',
        # Phone brands
        'Samsung', 'iPhone', 'Xiaomi', 'OnePlus', 'Google', 'Sony', 'Nokia', 
        'Oppo', 'Vivo', 'Realme', 'Motorola', 'Honor', 'ZTE', 'Nothing',
        # TV/Monitor brands
        'LG', 'TCL', 'Hisense', 'Philips', 'Panasonic', 'Sharp', 'Toshiba', 
        'AOC', 'BenQ', 'ViewSonic', 'iiyama', 'GoGEN', 'Vivax', 'Grundig',
        'Vox', 'Tesla', 'Thomson', 'Blaupunkt', 'Haier', 'Skyworth',
        # Gaming brands
        'Nintendo', 'PlayStation', 'Xbox', 'Steam', 'Valve', 'Corsair', 
        'Logitech', 'SteelSeries', 'HyperX', 'Roccat', 'Turtle Beach',
        # Audio brands
        'Bose', 'JBL', 'Sennheiser', 'Audio-Technica', 'Beyerdynamic', 
        'AKG', 'Shure', 'Marshall', 'Beats', 'Skullcandy', 'Jabra', 'Anker',
        'Harman Kardon', 'Bang & Olufsen', 'Sonos', 'Ultimate Ears',
        # Appliance/Camera brands
        'Bosch', 'Siemens', 'Canon', 'Nikon', 'GoPro', 'DJI', 'Fitbit', 
        'Garmin', 'Dyson', 'Braun', 'Oral-B', 'Remington', 'Fujifilm',
        # Peripherals
        'LogiLink', 'Baseus', 'Ugreen', 'Belkin', 'SanDisk', 'Kingston', 
        'Crucial', 'Western Digital', 'Seagate', 'TP-Link', 'Netgear',
        # Gaming furniture
        'Cougar', 'Genesis', 'Trust', 'Marvo', 'Redragon', 'Havit', 'Hama',
    ]
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self.session = None
        self.driver = None
        self.scraped_urls: Set[str] = set()
        self.products: List[Dict[str, Any]] = []
        self.base_url = self.config.BASE_URL
        
        # Initialize helpers
        self.rate_limiter = RateLimiter(
            requests_per_second=1.0 / max(self.config.REQUEST_DELAY, 0.5),
            burst_limit=3
        )
        self.retry_handler = RetryHandler(
            max_retries=self.config.MAX_RETRIES,
            base_delay=1.0
        )
        
        # User agent rotation
        if HAS_FAKE_UA:
            try:
                self.ua = UserAgent()
            except Exception:
                self.ua = None
        else:
            self.ua = None
        
        self._init_session()
        self._load_existing_data()
        
        logger.info(f"GjirafaScraper initialized - Base URL: {self.base_url}")
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent string"""
        if self.ua:
            try:
                return self.ua.random
            except Exception:
                pass
        
        # Fallback user agents (recent Chrome versions)
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        ]
        return random.choice(user_agents)
    
    def _init_session(self) -> None:
        """Initialize HTTP session with optimal settings"""
        try:
            if HAS_CLOUDSCRAPER:
                self.session = cloudscraper.create_scraper(
                    browser={
                        'browser': 'chrome',
                        'platform': 'windows',
                        'desktop': True
                    }
                )
                logger.info("Initialized CloudScraper session")
            else:
                self.session = requests.Session()
                logger.info("Initialized standard requests session")
            
            # Set comprehensive headers
            self.session.headers.update({
                'User-Agent': self._get_random_user_agent(),
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
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
            self.session = requests.Session()
    
    def _init_driver(self) -> None:
        """Initialize Selenium WebDriver with optimal anti-detection settings"""
        if self.driver:
            return
        
        try:
            chrome_options = Options()
            
            # Essential options for stability
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-infobars")
            chrome_options.add_argument("--disable-notifications")
            chrome_options.add_argument("--disable-popup-blocking")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--start-maximized")
            chrome_options.add_argument(f"--user-agent={self._get_random_user_agent()}")
            
            # Anti-detection measures
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Optional: Disable image loading for faster scraping
            # prefs = {"profile.managed_default_content_settings.images": 2}
            # chrome_options.add_experimental_option("prefs", prefs)
            
            if self.config.HEADLESS:
                chrome_options.add_argument("--headless=new")
            
            # Get driver path
            driver_path = self._get_driver_path()
            
            service = Service(driver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Set timeouts
            self.driver.implicitly_wait(self.config.IMPLICIT_WAIT)
            self.driver.set_page_load_timeout(self.config.PAGE_LOAD_TIMEOUT)
            
            # Execute CDP commands to further reduce detection
            self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": self._get_random_user_agent()
            })
            
            # Remove webdriver property
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            
            logger.info("Selenium WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def _get_driver_path(self) -> str:
        """Get the correct ChromeDriver path with fallback logic"""
        import os
        import glob
        
        driver_path = ChromeDriverManager().install()
        
        # Handle Windows-specific path issues
        if not driver_path.endswith('.exe') or not os.path.exists(driver_path):
            driver_dir = os.path.dirname(driver_path)
            
            possible_paths = [
                os.path.join(driver_dir, 'chromedriver.exe'),
                os.path.join(driver_dir, 'chromedriver-win32', 'chromedriver.exe'),
            ]
            
            exe_files = glob.glob(os.path.join(driver_dir, '**', 'chromedriver.exe'), recursive=True)
            possible_paths.extend(exe_files)
            
            for path in possible_paths:
                if os.path.exists(path) and path.endswith('.exe'):
                    return path
        
        return driver_path
    
    def get_page_content(self, url: str, use_selenium: bool = False) -> Optional[BeautifulSoup]:
        """
        Get page content with automatic fallback to Selenium if needed
        
        Args:
            url: URL to fetch
            use_selenium: Force Selenium usage
            
        Returns:
            BeautifulSoup object or None if failed
        """
        self.rate_limiter.acquire()
        
        try:
            if use_selenium:
                return self._get_content_selenium(url)
            else:
                soup = self._get_content_requests(url)
                if soup is None:
                    logger.info(f"Falling back to Selenium for {url}")
                    return self._get_content_selenium(url)
                return soup
        except Exception as e:
            logger.error(f"Error getting page content for {url}: {e}")
            return None
    
    def _get_content_requests(self, url: str) -> Optional[BeautifulSoup]:
        """Get content using requests/cloudscraper"""
        try:
            headers = {
                'User-Agent': self._get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': self.base_url,
            }
            
            response = self.session.get(
                url,
                headers=headers,
                timeout=self.config.TIMEOUT,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Check for blocking indicators
            if response.status_code == 403 or "blocked" in response.text.lower():
                logger.warning(f"Possible blocking detected for {url}")
                return None
            
            return BeautifulSoup(response.content, 'lxml')
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def _get_content_selenium(self, url: str) -> Optional[BeautifulSoup]:
        """Get content using Selenium WebDriver"""
        try:
            if not self.driver:
                self._init_driver()
            
            self.driver.get(url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for dynamic content
            time.sleep(2)
            
            # Scroll to load lazy content
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
            time.sleep(1)
            
            return BeautifulSoup(self.driver.page_source, 'lxml')
            
        except TimeoutException:
            logger.error(f"Timeout loading page: {url}")
            return None
        except WebDriverException as e:
            logger.error(f"WebDriver error for {url}: {e}")
            return None
    
    def discover_category_urls(self, base_url: str = None) -> List[str]:
        """Discover all category URLs from the homepage"""
        if not base_url:
            base_url = self.base_url
        
        category_urls = []
        
        try:
            soup = self.get_page_content(base_url)
            if not soup:
                logger.error("Failed to get homepage content")
                # Return predefined categories as fallback
                return [URLHelper.normalize_url(cat, base_url) for cat in self.config.CATEGORY_URLS]
            
            # Look for category links
            selectors = [
                ".header-menu a[href]",
                "nav a[href]",
                ".main-menu a[href]",
                ".category-menu a[href]",
                "[class*='menu'] a[href]",
            ]
            
            seen = set()
            exclude_patterns = [
                '/account', '/login', '/register', '/cart', '/wishlist',
                '/faq', '/contact', '/about', '/terms', '/privacy',
                '/cdn-cgi/', 'mailto:', 'tel:', '#', '/outlet',
                '/gift-cards', '/cfare-ka-te-re', '/search', 'javascript:',
            ]
            
            for selector in selectors:
                for link in soup.select(selector):
                    href = link.get('href', '')
                    if not href or href in seen:
                        continue
                    
                    if any(p in href.lower() for p in exclude_patterns):
                        continue
                    
                    if href.startswith('/') and len(href) > 1:
                        full_url = URLHelper.normalize_url(href, base_url)
                        if full_url not in seen:
                            seen.add(full_url)
                            category_urls.append(full_url)
            
            logger.info(f"Discovered {len(category_urls)} category URLs")
            
        except Exception as e:
            logger.error(f"Error discovering categories: {e}")
            return [URLHelper.normalize_url(cat, base_url) for cat in self.config.CATEGORY_URLS]
        
        return category_urls
    
    def discover_product_urls(self, category_url: str, max_products: Optional[int] = None, 
                              max_pages: int = 50, wait_for_all: bool = True) -> List[str]:
        """
        Discover product URLs from a category page
        
        Args:
            category_url: URL of the category to scrape
            max_products: Maximum number of products to discover (None for unlimited - gets ALL products)
            max_pages: Maximum number of pagination attempts (default 50 to handle large categories)
            wait_for_all: If True, keep loading until no more products (default True)
            
        Returns:
            List of product URLs
        """
        product_urls = []
        
        try:
            self._init_driver()
            
            logger.info(f"Loading category: {category_url}")
            if max_products:
                logger.info(f"Target: {max_products} products")
            
            self.driver.get(category_url)
            time.sleep(3)
            
            # Extract initial products
            initial_products = self._extract_products_from_page()
            product_urls.extend(initial_products)
            logger.info(f"Found {len(initial_products)} products on initial load")
            
            if max_products and len(product_urls) >= max_products:
                return product_urls[:max_products]
            
            # Handle "Load More" pagination
            load_more_attempts = 0
            consecutive_no_new = 0  # Track consecutive attempts with no new products
            
            while load_more_attempts < max_pages:
                if max_products and len(product_urls) >= max_products:
                    logger.info(f"Reached target of {max_products} products")
                    break
                
                try:
                    load_more_button = self._find_load_more_button()
                    
                    if not load_more_button:
                        logger.info("No more 'Load More' button found - all products loaded")
                        break
                    
                    products_before = len(product_urls)
                    
                    # Scroll and click
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});", 
                        load_more_button
                    )
                    time.sleep(0.5)
                    self.driver.execute_script("arguments[0].click();", load_more_button)
                    time.sleep(3)
                    
                    # Extract new products
                    current_products = self._extract_products_from_page()
                    new_products = [url for url in current_products if url not in product_urls]
                    product_urls.extend(new_products)
                    
                    new_count = len(product_urls) - products_before
                    logger.info(f"Loaded {new_count} new products (total: {len(product_urls)})")
                    
                    if new_count == 0:
                        consecutive_no_new += 1
                        if consecutive_no_new >= 3:
                            logger.info("No new products after 3 attempts - all products loaded")
                            break
                    else:
                        consecutive_no_new = 0
                    
                    load_more_attempts += 1
                    
                except Exception as e:
                    logger.warning(f"Error during pagination: {e}")
                    break
            
            # Try traditional pagination as fallback (only if few products found)
            if len(product_urls) < 30:
                traditional = self._try_traditional_pagination(category_url, max_pages)
                new_traditional = [url for url in traditional if url not in product_urls]
                if max_products:
                    remaining = max_products - len(product_urls)
                    new_traditional = new_traditional[:remaining]
                product_urls.extend(new_traditional)
            
        except Exception as e:
            logger.error(f"Error discovering products: {e}")
        
        # Apply limit
        if max_products and len(product_urls) > max_products:
            product_urls = product_urls[:max_products]
        
        logger.info(f"Total discovered: {len(product_urls)} product URLs")
        return product_urls
    
    def _find_load_more_button(self):
        """Find the load more button on the page"""
        selectors = [
            'button.load-more-products-btn',
            'button[data-page-infinite]',
            'button[class*="load-more"]',
            '[onclick*="loadProductsAjax"]',
        ]
        
        for selector in selectors:
            try:
                buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for button in buttons:
                    if button.is_displayed() and button.is_enabled():
                        return button
            except Exception:
                continue
        
        # Try XPath for Albanian text
        try:
            xpath_patterns = [
                "//button[contains(text(), 'SHFAQ MË SHUMË')]",
                "//button[contains(text(), 'Shfaq më shumë')]",
                "//button[contains(translate(text(), 'SHFAQMËHUMË', 'shfaqmëhumë'), 'shfaq më shumë')]",
            ]
            for xpath in xpath_patterns:
                buttons = self.driver.find_elements(By.XPATH, xpath)
                for button in buttons:
                    if button.is_displayed() and button.is_enabled():
                        return button
        except Exception:
            pass
        
        return None
    
    def _extract_products_from_page(self) -> List[str]:
        """Extract product URLs from current page"""
        product_urls = []
        
        soup = BeautifulSoup(self.driver.page_source, 'lxml')
        
        # Product link selectors
        selectors = [
            ".product-item a[href]",
            ".product-card a[href]",
            ".product-box a[href]",
            "[data-product-id] a[href]",
            "a[href*='/laptop-']",
            "a[href*='/kompjuter-']",
            "a[href*='/telefon-']",
            "a[href*='/televizor-']",
            "a[href*='/monitor-']",
            "a[href*='/kufje-']",
            "a[href*='/tablet-']",
            "a[href*='/tv-']",
            "a[href*='/audio-']",
        ]
        
        seen = set()
        for selector in selectors:
            for link in soup.select(selector):
                href = link.get('href', '')
                if href and self._is_valid_product_url(href) and href not in seen:
                    seen.add(href)
                    full_url = URLHelper.normalize_url(href, self.base_url)
                    product_urls.append(full_url)
        
        return product_urls
    
    def _is_valid_product_url(self, href: str) -> bool:
        """Check if URL is a valid product page"""
        if not href or not href.startswith('/'):
            return False
        
        if href.startswith('//'):
            return False
        
        # Exclude non-product URLs
        exclude_patterns = [
            '/account', '/login', '/register', '/cart', '/wishlist',
            '/faq', '/contact', '/about', '/terms', '/privacy',
            '/cdn-cgi/', 'mailto:', 'tel:', '#', '/outlet', '/search',
            '/categories', '/brands', '/offers', '/sale', '/new-arrivals',
        ]
        
        href_lower = href.lower()
        if any(p in href_lower for p in exclude_patterns):
            return False
        
        # Must be reasonably long and contain hyphens
        if len(href) < 15 or '-' not in href:
            return False
        
        # Look for product-like patterns
        product_indicators = [
            r'\d+gb', r'\d+tb', r'i\d+', r'rtx\d+', r'gtx\d+',
            r'\d+inch', r'\d+hz', r'\d+mp', r'\d+"',
            r'pro\b', r'max\b', r'mini\b', r'plus\b', r'ultra\b',
        ]
        
        for pattern in product_indicators:
            if re.search(pattern, href_lower):
                return True
        
        # Category-specific product patterns
        product_prefixes = [
            '/laptop-', '/kompjuter-', '/telefon-', '/televizor-',
            '/monitor-', '/kufje-', '/tablet-', '/tv-', '/audio-',
            '/gaming-', '/smart-', '/kamera-', '/mbajtese-',
        ]
        
        if any(href_lower.startswith(p) for p in product_prefixes):
            return True
        
        # Long descriptive URLs are usually products
        if len(href) > 50 and href.count('-') > 5:
            return True
        
        return False
    
    def _try_traditional_pagination(self, category_url: str, max_pages: int) -> List[str]:
        """Try traditional pagination (page numbers)"""
        product_urls = []
        
        for page in range(1, max_pages + 1):
            page_products = []
            
            for pattern in [f"?page={page}", f"/page/{page}", f"?p={page}"]:
                page_url = category_url + pattern
                
                try:
                    soup = self.get_page_content(page_url)
                    if not soup:
                        continue
                    
                    for selector in SELECTORS.get("product_links", []):
                        for link in soup.select(selector):
                            href = link.get('href', '')
                            if href and self._is_valid_product_url(href):
                                full_url = URLHelper.normalize_url(href, self.base_url)
                                if full_url not in product_urls:
                                    product_urls.append(full_url)
                                    page_products.append(full_url)
                    
                    if page_products:
                        break
                except Exception as e:
                    logger.debug(f"Error on page {page}: {e}")
            
            if not page_products:
                break
            
            time.sleep(self.config.REQUEST_DELAY)
        
        return product_urls
    
    def extract_product_data(self, product_url: str) -> Optional[Dict[str, Any]]:
        """
        Extract comprehensive product data from a product page
        
        Args:
            product_url: URL of the product page
            
        Returns:
            Dictionary with product data or None if extraction failed
        """
        try:
            soup = self.get_page_content(product_url, use_selenium=True)
            if not soup:
                return None
            
            product = ScrapedProduct(url=product_url)
            product.scraped_at = datetime.now().isoformat()
            
            # Extract title first
            product.title = self._extract_title(soup)
            
            if not product.title:
                logger.warning(f"No title found for {product_url}")
                return None
            
            # Check for duplicates
            if self._is_duplicate(product_url, product.title):
                logger.info(f"Skipping duplicate: {product.title[:50]}...")
                return None
            
            # Extract all data fields
            product.price, product.original_price = self._extract_prices(soup)
            
            # Calculate discount percentage
            if product.price and product.original_price and product.original_price > product.price:
                product.discount_percentage = round(
                    ((product.original_price - product.price) / product.original_price) * 100, 2
                )
            
            product.description = self._extract_description(soup)
            product.brand = self._extract_brand(soup, product.title)
            product.category = self._extract_category(soup)
            product.availability = self._extract_availability(soup)
            product.images = self._extract_images(soup)
            product.specifications = self._extract_specifications(soup)
            product.rating, product.reviews_count = self._extract_rating_reviews(soup)
            product.sku = self._extract_sku(soup, product_url)
            
            result = product.to_dict()
            
            # Track for duplicate detection
            self._track_product(product_url, product.title)
            
            logger.info(f"Extracted: {product.title[:60]}... | €{product.price or 'N/A'}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting data from {product_url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract product title"""
        selectors = [
            "h1.product-title",
            ".product-name h1",
            ".product-header h1",
            "h1[itemprop='name']",
            "[data-testid='product-title']",
            "h1",
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title and len(title) > 3:
                    return DataValidator.clean_text(title)
        
        return ""
    
    def _extract_prices(self, soup: BeautifulSoup) -> Tuple[Optional[float], Optional[float]]:
        """Extract current and original prices"""
        current_price = None
        original_price = None
        
        # Look for price container
        price_container = soup.select_one(".product-price, .prices, .price-box")
        
        if price_container:
            price_text = price_container.get_text()
            
            # Find all prices in the container
            prices = re.findall(r'(\d+(?:[.,]\d{1,2})?)\s*€', price_text)
            
            if len(prices) >= 2:
                # Multiple prices: first is usually original, second is current (discounted)
                try:
                    original_price = float(prices[0].replace(',', '.'))
                    current_price = float(prices[1].replace(',', '.'))
                except ValueError:
                    pass
            elif len(prices) == 1:
                try:
                    current_price = float(prices[0].replace(',', '.'))
                except ValueError:
                    pass
        
        # Try specific selectors for discounted price
        if not current_price:
            discount_selectors = [
                ".text-green-600",
                ".current-price",
                ".sale-price",
                "[class*='price-current']",
                ".prices .text-green-600",
            ]
            
            for selector in discount_selectors:
                element = soup.select_one(selector)
                if element:
                    price_text = element.get_text()
                    match = re.search(r'(\d+(?:[.,]\d{1,2})?)\s*€', price_text)
                    if match:
                        try:
                            current_price = float(match.group(1).replace(',', '.'))
                            break
                        except ValueError:
                            pass
        
        # Try specific selectors for original price
        if not original_price:
            original_selectors = [
                ".line-through",
                ".non-discounted-price",
                ".old-price",
                ".was-price",
                "[class*='line-through']",
            ]
            
            for selector in original_selectors:
                element = soup.select_one(selector)
                if element:
                    price_text = element.get_text()
                    match = re.search(r'(\d+(?:[.,]\d{1,2})?)\s*€', price_text)
                    if match:
                        try:
                            original_price = float(match.group(1).replace(',', '.'))
                            break
                        except ValueError:
                            pass
        
        return current_price, original_price
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract product description"""
        selectors = [
            ".product-description",
            ".product-details .description",
            "[itemprop='description']",
            ".description",
            ".product-info",
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                desc = element.get_text(strip=True)
                if desc and len(desc) > 20:
                    return DataValidator.clean_text(desc)
        
        return ""
    
    def _extract_brand(self, soup: BeautifulSoup, title: str) -> str:
        """Extract brand name using multiple strategies"""
        # First try specific brand selectors
        brand_selectors = [
            ".product-brand",
            ".brand-name",
            "[itemprop='brand']",
            ".manufacturer",
            "[data-testid='brand']",
        ]
        
        for selector in brand_selectors:
            element = soup.select_one(selector)
            if element:
                brand = element.get_text(strip=True)
                # Validate it's actually a brand, not a product type
                if brand and len(brand) > 1 and brand.lower() not in [
                    'televizor', 'laptop', 'kompjuter', 'telefon', 'monitor', 
                    'tv', 'kufje', 'tablet', 'mbajtëse', 'smart'
                ]:
                    return brand
        
        # Extract from title using known brands (case-insensitive)
        title_lower = title.lower()
        for brand in self.KNOWN_BRANDS:
            if brand.lower() in title_lower:
                return brand
        
        # Try to extract first capitalized word that looks like a brand
        words = title.split()
        for word in words[:3]:  # Check first 3 words
            clean_word = re.sub(r'[^\w]', '', word)
            if (clean_word and len(clean_word) >= 2 and 
                clean_word[0].isupper() and
                clean_word.lower() not in [
                    'televizor', 'laptop', 'kompjuter', 'telefon', 'monitor',
                    'tv', 'kufje', 'tablet', 'mbajtëse', 'smart', 'gaming'
                ]):
                return clean_word
        
        return ""
    
    def _extract_category(self, soup: BeautifulSoup) -> str:
        """Extract product category from breadcrumbs"""
        breadcrumb_selectors = [
            ".breadcrumb",
            ".breadcrumbs",
            "[class*='breadcrumb']",
            "nav[aria-label='breadcrumb']",
            ".category-path",
        ]
        
        for selector in breadcrumb_selectors:
            element = soup.select_one(selector)
            if element:
                links = element.select('a')
                if links:
                    categories = [DataValidator.clean_text(link.get_text()) for link in links]
                    # Filter out empty and "Home" entries
                    categories = [c for c in categories if c and c.lower() not in ['home', 'kryefaqja', '']]
                    if categories:
                        return " > ".join(categories)
        
        return ""
    
    def _extract_availability(self, soup: BeautifulSoup) -> str:
        """Extract product availability/stock status"""
        selectors = [
            ".stock-status",
            ".availability",
            ".in-stock",
            ".stock-info",
            "[class*='stock']",
            "[data-testid='stock-status']",
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                if text:
                    return DataValidator.clean_text(text)
        
        return ""
    
    def _extract_images(self, soup: BeautifulSoup) -> List[str]:
        """Extract product images with smart filtering"""
        all_images = []
        
        # Look for product gallery images
        gallery_selectors = [
            ".product-gallery img",
            ".product-images img",
            ".product-slider img",
            ".product-details img",
            ".details img",
            "[class*='gallery'] img",
            "[class*='slider'] img",
        ]
        
        for selector in gallery_selectors:
            for img in soup.select(selector):
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src:
                    # Normalize URL
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = self.base_url + src
                    all_images.append(src)
        
        # Also look for CDN images specifically
        for img in soup.find_all('img'):
            src = img.get('src') or img.get('data-src')
            if src and 'iqq6kf0xmf.gjirafa.net/images/' in src:
                if src.startswith('//'):
                    src = 'https:' + src
                all_images.append(src)
        
        # Filter to only include actual product images
        filtered_images = ImageFilter.filter_product_images(list(set(all_images)))
        
        return filtered_images
    
    def _extract_specifications(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract product specifications"""
        specs = {}
        
        # Look for specification tables
        spec_selectors = [
            ".specifications table",
            ".product-specs table",
            ".spec-table",
            "[class*='spec'] table",
            ".product-details table",
            ".details table",
            ".attributes table",
        ]
        
        for selector in spec_selectors:
            table = soup.select_one(selector)
            if table:
                rows = table.select('tr')
                for row in rows:
                    cells = row.select('td, th')
                    if len(cells) >= 2:
                        key = DataValidator.clean_text(cells[0].get_text())
                        value = DataValidator.clean_text(cells[1].get_text())
                        if key and value and key != value:
                            specs[key] = value
        
        # Try definition lists
        if not specs:
            for dl in soup.select('dl'):
                dts = dl.select('dt')
                dds = dl.select('dd')
                for dt, dd in zip(dts, dds):
                    key = DataValidator.clean_text(dt.get_text())
                    value = DataValidator.clean_text(dd.get_text())
                    if key and value:
                        specs[key] = value
        
        # Try key-value divs with specific class patterns
        if not specs:
            for container in soup.select('[class*="spec"], [class*="attribute"], [class*="detail"]'):
                # Look for label/value pairs
                label = container.select_one('[class*="label"], [class*="key"], [class*="name"]')
                value_el = container.select_one('[class*="value"], [class*="data"]')
                
                if label and value_el:
                    key = DataValidator.clean_text(label.get_text())
                    value = DataValidator.clean_text(value_el.get_text())
                    if key and value and len(key) < 50:
                        specs[key] = value
                else:
                    # Try parsing colon-separated text
                    text = container.get_text(strip=True)
                    if ':' in text:
                        parts = text.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            # Skip numeric keys or ratio patterns (e.g., "16:9", "5000:1")
                            if (key and value and len(key) < 50 and 
                                not re.match(r'^\d+$', key) and
                                not re.match(r'^\d+\s*:\s*\d+$', text)):
                                specs[key] = value
        
        return specs
    
    def _extract_rating_reviews(self, soup: BeautifulSoup) -> Tuple[Optional[float], Optional[int]]:
        """Extract rating and review count"""
        rating = None
        reviews_count = None
        
        # Look for rating elements
        rating_selectors = [
            ".ratingsAndReviews",
            ".product-rating",
            "[class*='rating']",
            "[class*='star']",
            "[itemprop='ratingValue']",
        ]
        
        for selector in rating_selectors:
            for element in soup.select(selector):
                text = element.get_text(strip=True)
                
                # Try itemprop first
                if element.get('content'):
                    try:
                        r = float(element.get('content'))
                        if 0 <= r <= 5:
                            rating = r
                            continue
                    except ValueError:
                        pass
                
                # Extract rating from text
                rating_patterns = [
                    r'(\d+(?:[.,]\d+)?)\s*(?:/\s*5|out\s*of\s*5)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:★|star)',
                    r'(\d+(?:[.,]\d+)?)\s*(?:vlerësi|rating)',
                ]
                
                for pattern in rating_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        try:
                            r = float(match.group(1).replace(',', '.'))
                            if 0 <= r <= 5:
                                rating = r
                                break
                        except ValueError:
                            pass
                
                # Extract review count
                review_patterns = [
                    r'(\d+)\s*(?:reviews?|vlerësi|vlerësime|opinione?|komente?)',
                    r'(\d+)\s*(?:recensione?)',
                ]
                
                for pattern in review_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        try:
                            reviews_count = int(match.group(1))
                            break
                        except ValueError:
                            pass
        
        return rating, reviews_count
    
    def _extract_sku(self, soup: BeautifulSoup, url: str) -> str:
        """Extract product SKU/ID"""
        # Try specific selectors
        sku_selectors = [
            "[itemprop='sku']",
            ".product-sku",
            ".sku",
            "[data-product-id]",
            "[data-sku]",
        ]
        
        for selector in sku_selectors:
            element = soup.select_one(selector)
            if element:
                sku = (element.get_text(strip=True) or 
                       element.get('content', '') or 
                       element.get('data-product-id', '') or
                       element.get('data-sku', ''))
                if sku:
                    return sku
        
        # Extract from URL
        return URLHelper.extract_product_id(url) or ""
    
    def _is_duplicate(self, url: str, title: str) -> bool:
        """Check if product is a duplicate"""
        if not hasattr(self, 'existing_products'):
            self.existing_products = set()
        
        key = f"{url}|||{title}"
        
        if key in self.existing_products:
            return True
        
        # Check current session
        for p in self.products:
            if p.get('url') == url or p.get('title') == title:
                return True
        
        return False
    
    def _track_product(self, url: str, title: str):
        """Track product for duplicate detection"""
        if not hasattr(self, 'existing_products'):
            self.existing_products = set()
        
        key = f"{url}|||{title}"
        self.existing_products.add(key)
    
    def _load_existing_data(self) -> None:
        """Load existing scraped data for duplicate detection"""
        try:
            data_dir = Path("scraped_data")
            if not data_dir.exists():
                self.existing_products = set()
                return
            
            existing_products = set()
            
            for json_file in data_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            if 'url' in item and 'title' in item:
                                key = f"{item['url']}|||{item['title']}"
                                existing_products.add(key)
                except Exception as e:
                    logger.debug(f"Error loading {json_file}: {e}")
            
            self.existing_products = existing_products
            logger.info(f"Loaded {len(existing_products)} existing products for duplicate detection")
            
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            self.existing_products = set()
    
    def export_data(self, data: List[Dict[str, Any]] = None, output_dir: str = "scraped_data",
                   formats: List[str] = None, filename_prefix: str = None) -> Dict[str, str]:
        """
        Export scraped data to various formats
        
        Args:
            data: List of product dictionaries (uses self.products if None)
            output_dir: Directory to save files
            formats: List of formats ('json', 'csv', 'excel')
            filename_prefix: Prefix for output filenames
            
        Returns:
            Dictionary mapping format to filepath
        """
        if formats is None:
            formats = ['json', 'csv']
        
        products = data if data is not None else self.products
        
        if not products:
            logger.warning("No products to export")
            return {}
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}" if filename_prefix else f"gjirafa_products_{timestamp}"
        
        exported_files = {}
        
        if 'json' in formats:
            json_file = Path(output_dir) / f"{base_filename}.json"
            if DataExporter.export_to_json(products, str(json_file)):
                exported_files['json'] = str(json_file)
                logger.info(f"Exported to JSON: {json_file}")
        
        if 'csv' in formats:
            csv_file = Path(output_dir) / f"{base_filename}.csv"
            if DataExporter.export_to_csv(products, str(csv_file)):
                exported_files['csv'] = str(csv_file)
                logger.info(f"Exported to CSV: {csv_file}")
        
        if 'excel' in formats or 'xlsx' in formats:
            excel_file = Path(output_dir) / f"{base_filename}.xlsx"
            if DataExporter.export_to_excel(products, str(excel_file)):
                exported_files['excel'] = str(excel_file)
                logger.info(f"Exported to Excel: {excel_file}")
        
        return exported_files
    
    def close(self) -> None:
        """Clean up resources"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
            if self.session:
                self.session.close()
                self.session = None
            logger.info("Scraper resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Convenience Functions
# =============================================================================

def scrape_category(category_url: str, max_products: Optional[int] = None, 
                   export_formats: List[str] = None) -> List[Dict[str, Any]]:
    """
    Quick function to scrape a category
    
    Args:
        category_url: Full URL of the category to scrape
        max_products: Maximum number of products to scrape (None = ALL products in category)
        export_formats: List of formats to export ('json', 'csv', 'excel')
    
    Returns:
        List of scraped product dictionaries
        
    Examples:
        # Scrape ALL products in category:
        products = scrape_category("https://gjirafa50.com/tv")
        
        # Scrape limited products:
        products = scrape_category("https://gjirafa50.com/tv", max_products=20)
    """
    export_formats = export_formats or ['json', 'csv']
    
    with GjirafaScraper() as scraper:
        # Discover products
        product_urls = scraper.discover_product_urls(category_url, max_products=max_products)
        
        # Scrape each product
        products = []
        for url in tqdm(product_urls, desc="Scraping products"):
            product_data = scraper.extract_product_data(url)
            if product_data:
                products.append(product_data)
                scraper.products.append(product_data)
        
        # Export
        if products:
            category_name = category_url.split('/')[-1].replace('-', '_')
            scraper.export_data(formats=export_formats, filename_prefix=category_name)
        
        return products


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        category = sys.argv[1]
        # Default to None (ALL products) if not specified
        max_prod = int(sys.argv[2]) if len(sys.argv) > 2 else None
        products = scrape_category(category, max_products=max_prod)
        print(f"\nScraped {len(products)} products")
    else:
        print("Usage: python gjirafa_scraper.py <category_url> [max_products]")
        print("Example (all products): python gjirafa_scraper.py https://gjirafa50.com/tv")
        print("Example (limited):      python gjirafa_scraper.py https://gjirafa50.com/tv 20")
