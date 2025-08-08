import requests
import time
import json
import logging
from typing import Dict, List, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
import cloudscraper
from tqdm import tqdm

from config import ScraperConfig, SELECTORS, HEADERS, CHROME_OPTIONS
from utils import DataValidator, DataExporter, URLHelper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GjirafaScraper:
    
    def __init__(self, config: Optional[ScraperConfig] = None):
        self.config = config or ScraperConfig()
        self.session = None
        self.driver = None
        self.scraped_urls: Set[str] = set()
        self.products: List[Dict[str, Any]] = []
        self.ua = UserAgent()
        self.base_url = self.config.BASE_URL
        
        self._init_session()
        self._load_existing_data()
    
    def _init_session(self) -> None:
        try:
            self.session = cloudscraper.create_scraper()
            self.session.headers.update(HEADERS)
            logger.info("Initialized CloudScraper session")
        except Exception as e:
            logger.warning(f"CloudScraper failed, using regular requests: {e}")
            self.session = requests.Session()
            self.session.headers.update(HEADERS)
    
    def _init_driver(self) -> None:
        if self.driver:
            return
        
        try:
            chrome_options = Options()
            
            for option in CHROME_OPTIONS:
                chrome_options.add_argument(option)
            
            if self.config.HEADLESS:
                chrome_options.add_argument("--headless")
            
            driver_path = ChromeDriverManager().install()
            
            import os
            import glob
            
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
                        driver_path = path
                        break
            
            self.driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(driver_path),
                options=chrome_options
            )
            
            self.driver.implicitly_wait(self.config.IMPLICIT_WAIT)
            self.driver.set_page_load_timeout(self.config.PAGE_LOAD_TIMEOUT)
            
            logger.info("Selenium WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def get_page_content(self, url: str, use_selenium: bool = False) -> Optional[BeautifulSoup]:
        try:
            if use_selenium:
                return self._get_content_selenium(url)
            else:
                return self._get_content_requests(url)
        except Exception as e:
            logger.error(f"Error getting page content for {url}: {e}")
            return None
    
    def _get_content_requests(self, url: str) -> Optional[BeautifulSoup]:
        try:
            headers = HEADERS.copy()
            headers['User-Agent'] = self.ua.random
            
            response = self.session.get(
                url, 
                headers=headers, 
                timeout=self.config.TIMEOUT,
                allow_redirects=True
            )
            response.raise_for_status()
            
            if "blocked" in response.text.lower() or response.status_code == 403:
                logger.warning(f"Possible blocking detected for {url}, switching to Selenium")
                return self._get_content_selenium(url)
            
            return BeautifulSoup(response.content, 'lxml')
            
        except Exception as e:
            logger.error(f"Requests failed for {url}: {e}")
            return None
    
    def _get_content_selenium(self, url: str) -> Optional[BeautifulSoup]:
        try:
            if not self.driver:
                self._init_driver()
            
            self.driver.get(url)
            
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(2)
            
            html = self.driver.page_source
            return BeautifulSoup(html, 'lxml')
            
        except TimeoutException:
            logger.error(f"Timeout loading page: {url}")
            return None
        except Exception as e:
            logger.error(f"Selenium failed for {url}: {e}")
            return None
    
    def discover_category_urls(self, base_url: str = None) -> List[str]:
        if not base_url:
            base_url = self.base_url
        
        category_urls = []
        
        try:
            soup = self.get_page_content(base_url)
            if not soup:
                logger.error("Failed to get homepage content")
                return category_urls
            
            category_selectors = [
                ".header-menu a", 
                "nav a",
                ".nav a",
                ".menu a",
                "[class*='grid'] a" 
            ]
            
            for selector in category_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href and href.startswith('/') and not href.startswith('//'):
                        exclude_patterns = [
                            '/account/', '/login', '/register', '/cart', '/wishlist',
                            '/faq', '/contact', '/about', '/terms', '/privacy',
                            '/cdn-cgi/', 'mailto:', 'tel:', '#', '/outlet',
                            '/gift-cards', '/cfare-ka-te-re'
                        ]
                        
                        if not any(pattern in href.lower() for pattern in exclude_patterns):
                            full_url = URLHelper.normalize_url(href, base_url)
                            if full_url not in category_urls:
                                category_urls.append(full_url)
            
            logger.info(f"Discovered {len(category_urls)} category URLs")
            
        except Exception as e:
            logger.error(f"Error discovering categories: {e}")
        
        return category_urls
    def discover_product_urls(self, category_url: str, max_products: Optional[int] = None, max_pages: int = 10) -> List[str]:
        product_urls = []
        
        try:
            self._init_driver()
            
            logger.info(f"Loading category page: {category_url}")
            if max_products:
                logger.info(f"Target: {max_products} products")
            self.driver.get(category_url)
            time.sleep(3) 
            
            initial_products = self._extract_products_from_page()
            product_urls.extend(initial_products)
            logger.info(f"Found {len(initial_products)} products on initial page load")
            
            if max_products and len(product_urls) >= max_products:
                logger.info(f"Target reached with initial load: {len(product_urls)} products")
                return product_urls[:max_products]
            
            load_more_attempts = 0
            max_load_more_attempts = max_pages
            while load_more_attempts < max_load_more_attempts:
                if max_products and len(product_urls) >= max_products:
                    logger.info(f"Target reached: {len(product_urls)} products (target: {max_products})")
                    break
                    
                try:
                    load_more_selectors = [
                        'button.load-more-products-btn',
                        'button[data-page-infinite]',
                        'button[class*="load-more"]',
                        'button:contains("SHFAQ MË SHUMË")',
                        '[onclick*="loadProductsAjax"]'
                    ]
                    
                    load_more_button = None
                    for selector in load_more_selectors:
                        try:
                            if 'contains' in selector:
                                buttons = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'SHFAQ MË SHUMË')]")
                            else:
                                buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            
                            if buttons:
                                for button in buttons:
                                    if button.is_displayed() and button.is_enabled():
                                        load_more_button = button
                                        break
                                if load_more_button:
                                    break
                        except Exception:
                            continue
                    
                    if not load_more_button:
                        logger.info("No more 'Load More' button found, stopping pagination")
                        break
                    
                    products_before = len(product_urls)
                    
                    self.driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
                    time.sleep(1)
                    
                    logger.info(f"Clicking 'Load More' button (attempt {load_more_attempts + 1})")
                    if max_products:
                        remaining = max_products - len(product_urls)
                        logger.info(f"Need {remaining} more products to reach target")
                    self.driver.execute_script("arguments[0].click();", load_more_button)
                    
                    time.sleep(3)
                    
                    current_products = self._extract_products_from_page()
                    
                    new_products = [url for url in current_products if url not in product_urls]
                    product_urls.extend(new_products)
                    
                    products_after = len(product_urls)
                    new_count = products_after - products_before
                    
                    logger.info(f"Loaded {new_count} new products (total: {products_after})")
                    
                    if new_count == 0:
                        logger.info("No new products loaded, stopping pagination")
                        break
                    
                    load_more_attempts += 1
                    
                    time.sleep(self.config.REQUEST_DELAY)
                    
                except Exception as e:
                    logger.warning(f"Error during load more attempt {load_more_attempts + 1}: {e}")
                    break
            if (max_products is None or len(product_urls) < max_products) and len(product_urls) < 50:
                logger.info("Trying traditional pagination as fallback...")
                remaining_needed = max_products - len(product_urls) if max_products else None
                traditional_products = self._try_traditional_pagination(category_url, max_pages)
                
                new_traditional = [url for url in traditional_products if url not in product_urls]
                if max_products and remaining_needed:
                    new_traditional = new_traditional[:remaining_needed]
                product_urls.extend(new_traditional)
                logger.info(f"Added {len(new_traditional)} products from traditional pagination")
            
        except Exception as e:
            logger.error(f"Error discovering products from {category_url}: {e}")
        
        if max_products and len(product_urls) > max_products:
            product_urls = product_urls[:max_products]
            logger.info(f"Limited results to {max_products} products as requested")
        
        logger.info(f"Total discovered products: {len(product_urls)} from {category_url}")
        return product_urls
    def _extract_products_from_page(self) -> List[str]:
        product_urls = []
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        for selector in SELECTORS["product_links"]:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href and self._is_valid_product_url(href):
                    full_url = URLHelper.normalize_url(href, self.base_url)
                    if full_url not in product_urls:
                        product_urls.append(full_url)
        
        return product_urls
    
    def _is_valid_product_url(self, href: str) -> bool:
        """Improved validation to distinguish between product and category pages"""
        
        # Basic checks
        if not href.startswith('/') or href.startswith('//'):
            return False
        
        # Exclude obvious non-product URLs
        exclude_patterns = [
            '/account/', '/login', '/register', '/cart', '/wishlist',
            '/faq', '/contact', '/about', '/terms', '/privacy',
            '/cdn-cgi/', 'mailto:', 'tel:', '#', '/outlet',
            '/gift-cards', '/cfare-ka-te-re', '/search', '/categories',
            '/brands', '/offers', '/sale', '/new-arrivals'
        ]
        
        if any(pattern in href.lower() for pattern in exclude_patterns):
            return False
        
        # URL must be reasonably long (products have descriptive URLs)
        if len(href) < 15:
            return False
        
        # Check if URL contains product keywords
        product_keywords = [
            'laptop', 'kompjuter', 'telefon', 'tv', 'monitor', 
            'kufje', 'tablet', 'kamera', 'audio', 'gaming'
        ]
        
        if not any(keyword in href.lower() for keyword in product_keywords):
            return False
        
        # More sophisticated filtering: product URLs typically have multiple path segments
        # and contain descriptive text, while category URLs are shorter
        url_parts = href.strip('/').split('/')
        
        # Category pages typically have 1-2 segments, product pages have longer descriptive URLs
        if len(url_parts) <= 2 and not any(len(part) > 20 for part in url_parts):
            return False
        
        # Product URLs typically contain hyphens (word separators) and model numbers/specs
        if '-' not in href:
            return False
        
        # Look for signs this is a product URL (model numbers, specs, brands)
        import re
        
        # Check for product-like patterns in URL
        product_patterns = [
            r'\b\d+gb\b',           # RAM/storage sizes
            r'\b\d+tb\b',           # Storage sizes  
            r'\bi\d+\b',            # Intel processors
            r'\brtx\d+\b',          # Graphics cards
            r'\bgtx\d+\b',          # Graphics cards
            r'\b\d+inch\b',         # Screen sizes
            r'\b\d+"?\b',           # Screen sizes with quotes
            r'\b\d+hz\b',           # Refresh rates
            r'\b\d+mp\b',           # Camera megapixels
            r'\bwindows\b',         # Operating system
            r'\bmac\b',             # Apple products
            r'\bpro\b',             # Product lines
            r'\bmax\b',             # Product lines
            r'\bmini\b',            # Product lines
            r'\bcore\b',            # Processors
            r'\bamd\b',             # Processors
            r'\bintel\b',           # Processors
            r'\bnvidia\b',          # Graphics
        ]
        
        href_lower = href.lower()
        pattern_matches = sum(1 for pattern in product_patterns if re.search(pattern, href_lower))
        
        # If URL has product-like patterns, it's likely a product
        if pattern_matches >= 2:
            return True
        
        # Additional check: very long URLs with many hyphens are usually products
        if len(href) > 50 and href.count('-') > 5:
            return True
        
        # If URL has at least one product pattern and is reasonably long, consider it a product
        if pattern_matches >= 1 and len(href) > 30:
            return True
        
        return False
    
    def _try_traditional_pagination(self, category_url: str, max_pages: int) -> List[str]:
        product_urls = []
        
        try:
            for page in range(1, max_pages + 1):
                page_urls = [
                    f"{category_url}?page={page}",
                    f"{category_url}/page/{page}",
                    f"{category_url}?p={page}"
                ]
                for page_url in page_urls:
                    soup = self.get_page_content(page_url)
                    if not soup:
                        continue
                    
                    page_products = []
                    for selector in SELECTORS["product_links"]:
                        links = soup.select(selector)
                        for link in links:
                            href = link.get('href')
                            if href and self._is_valid_product_url(href):
                                full_url = URLHelper.normalize_url(href, self.base_url)
                                if full_url not in product_urls:
                                    product_urls.append(full_url)
                                    page_products.append(full_url)
                    
                    if page_products:
                        break
                
                if not page_products:
                    break
                
                time.sleep(self.config.REQUEST_DELAY)
        
        except Exception as e:
            logger.error(f"Error in traditional pagination: {e}")
        
        return product_urls
    def extract_product_data(self, product_url: str) -> Optional[Dict[str, Any]]:
        try:
            soup = self.get_page_content(product_url, use_selenium=True)
            if not soup:
                return None
            
            product_data = {
                'url': product_url,
                'product_id': URLHelper.extract_product_id(product_url),
                'base_url': self.base_url
            }
            
            product_data['title'] = self._extract_by_selectors(soup, SELECTORS["title"])
            
            if self._is_duplicate(product_data):
                logger.info(f"Skipping duplicate product: {product_data.get('title', 'Unknown')[:50]}...")
                return None
            
            product_data['price'] = self._extract_price_smart(soup)
            product_data['original_price'] = self._extract_original_price_smart(soup)
            
            product_data['description'] = self._extract_by_selectors(soup, SELECTORS["description"])
            
            product_data['images'] = self._extract_images(soup)
            
            product_data['specifications'] = self._extract_specifications(soup)
            
            product_data['brand'] = self._extract_brand_smart(soup)
            product_data['category'] = self._extract_category(soup)
            product_data['availability'] = self._extract_by_selectors(soup, SELECTORS["availability"])
            product_data['rating'] = self._extract_rating_smart(soup)
            product_data['reviews_count'] = self._extract_reviews_count_smart(soup)
            validated_data = DataValidator.validate_product_data(product_data)
            
            # Add to existing products set to prevent future duplicates in this session
            if hasattr(self, 'existing_products') and 'url' in validated_data and 'title' in validated_data:
                key = f"{validated_data['url']}|||{validated_data['title']}"
                self.existing_products.add(key)
            
            logger.info(f"Extracted data for: {validated_data.get('title', 'Unknown')}")
            return validated_data
            
        except Exception as e:
            logger.error(f"Error extracting product data from {product_url}: {e}")
            return None
    
    def _extract_by_selectors(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text(strip=True)
                    if text:
                        return DataValidator.clean_text(text)
            except Exception:
                continue
        return ""
    
    def _extract_price_smart(self, soup: BeautifulSoup) -> Optional[str]:
        """Smart price extraction that handles complex price structures"""
        
        # First try to find specific discounted price
        discounted_selectors = [
            ".prices .text-green-600",
            ".prices [class*='green']", 
            ".current-price",
            ".sale-price"
        ]
        
        for selector in discounted_selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    price_text = element.get_text(strip=True)
                    # Extract just the price number with €
                    import re
                    price_match = re.search(r'(\d+(?:\.\d{2})?)\s*€', price_text)
                    if price_match:
                        return price_match.group(0)
            except Exception:
                continue
        
        # Try to extract from the main price container
        price_containers = [
            ".product-price",
            ".prices"
        ]
        
        for container_selector in price_containers:
            try:
                container = soup.select_one(container_selector)
                if container:
                    # Look for price patterns within the container
                    price_text = container.get_text(strip=True)
                    
                    # Extract multiple prices and identify the current one
                    import re
                    prices = re.findall(r'(\d+(?:\.\d{2})?)\s*€', price_text)
                    
                    if len(prices) >= 2:
                        # If multiple prices, the second one is usually the discounted price
                        return f"{prices[1]} €"
                    elif len(prices) == 1:
                        # Single price
                        return f"{prices[0]} €"
            except Exception:
                continue
        
        # Fallback to original method
        return self._extract_by_selectors(soup, SELECTORS["price"])

    def _extract_original_price_smart(self, soup: BeautifulSoup) -> Optional[str]:
        """Smart original price extraction"""
        
        # Look for crossed-out prices
        original_selectors = [
            ".line-through",
            ".non-discounted-price", 
            "[class*='line-through']",
            ".old-price",
            ".was-price"
        ]
        
        for selector in original_selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    price_text = element.get_text(strip=True)
                    import re
                    price_match = re.search(r'(\d+(?:\.\d{2})?)\s*€', price_text)
                    if price_match:
                        return price_match.group(0)
            except Exception:
                continue
        
        # Try to extract from price container
        price_containers = [
            ".product-price",
            ".prices"
        ]
        
        for container_selector in price_containers:
            try:
                container = soup.select_one(container_selector)
                if container:
                    price_text = container.get_text(strip=True)
                    
                    # Extract multiple prices - first one is usually original
                    import re
                    prices = re.findall(r'(\d+(?:\.\d{2})?)\s*€', price_text)
                    
                    if len(prices) >= 2:
                        # If multiple prices, first one is usually original
                        return f"{prices[0]} €"
            except Exception:
                continue
        
        return self._extract_by_selectors(soup, SELECTORS["original_price"])

    def _extract_rating_smart(self, soup: BeautifulSoup) -> Optional[str]:
        """Smart rating extraction"""
        
        # Look for star ratings or numeric ratings
        rating_selectors = [
            ".ratingsAndReviews",
            "[class*='rating']",
            "[class*='star']"
        ]
        
        for selector in rating_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    
                    # Look for rating patterns
                    import re
                    # Match patterns like "4.5", "4/5", "4 out of 5", etc.
                    rating_patterns = [
                        r'(\d+(?:\.\d+)?)\s*(?:/\s*5|\s*out\s*of\s*5)',  # 4.5/5 or 4.5 out of 5
                        r'(\d+(?:\.\d+)?)\s*(?:stars?|★)',               # 4.5 stars or 4.5★
                        r'(\d+(?:\.\d+)?)\s*(?:vlerësime?)',             # "4.5 vlerësime" (Albanian)
                        r'(\d+(?:\.\d+)?)'                               # Just numbers
                    ]
                    
                    for pattern in rating_patterns:
                        match = re.search(pattern, text)
                        if match:
                            rating = float(match.group(1))
                            if 0 <= rating <= 5:
                                return str(rating)
                    
                    # Handle "00 Vlerësime" case - extract the rating part
                    if "vlerësime" in text.lower():
                        rating_match = re.search(r'(\d+(?:\.\d+)?)', text)
                        if rating_match:
                            rating = float(rating_match.group(1))
                            if 0 <= rating <= 5:
                                return str(rating)
            except Exception:
                continue
        
        return self._extract_by_selectors(soup, SELECTORS["rating"])

    def _extract_reviews_count_smart(self, soup: BeautifulSoup) -> Optional[str]:
        """Smart reviews count extraction"""
        
        review_selectors = [
            ".product-reviews-overview",
            ".product-no-reviews", 
            ".ratingsAndReviews",
            "[class*='review']"
        ]
        
        for selector in review_selectors:
            try:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    
                    # Look for review count patterns
                    import re
                    patterns = [
                        r'(\d+)\s*(?:reviews?|vlerësi|vlerësime)',  # "5 reviews" or "5 vlerësime"
                        r'(\d+)\s*(?:opinione?|komente?)',          # "5 opinione" 
                        r'(\d+)\s*(?:recensione?)',                 # "5 recensioni"
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            return match.group(1)
                    
                    # Handle "00 Vlerësime" case specifically 
                    if "vlerësime" in text.lower():
                        # Extract the number before "Vlerësime"
                        count_match = re.search(r'(\d+)\s*vlerësime', text, re.IGNORECASE)
                        if count_match:
                            return count_match.group(1)
                    
                    # Look for just numbers in review contexts
                    if any(word in text.lower() for word in ['vlerësi', 'review', 'opinione', 'koment']):
                        number_match = re.search(r'(\d+)', text)
                        if number_match:
                            return number_match.group(1)
            except Exception:
                continue
        
        return self._extract_by_selectors(soup, SELECTORS["reviews_count"])
    
    def _extract_images(self, soup: BeautifulSoup) -> List[str]:
        """Extract product images from the page"""
        images = []
        
        for selector in SELECTORS["images"]:
            try:
                img_elements = soup.select(selector)
                for img in img_elements:
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src:
                        images.append(src)
            except Exception:
                continue
        
        return DataValidator.clean_image_urls(images, self.base_url)
    
    def _extract_specifications(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract product specifications from the page"""
        for selector in SELECTORS["specifications"]:
            try:
                spec_element = soup.select_one(selector)
                if spec_element:
                    return DataValidator.extract_specifications(spec_element)
            except Exception:
                continue
        return {}
    
    def _extract_category(self, soup: BeautifulSoup) -> str:
        """Extract product category from breadcrumbs or page structure"""
        for selector in SELECTORS["category"]:
            try:
                element = soup.select_one(selector)
                if element:
                    breadcrumb_links = element.select('a')
                    if breadcrumb_links:
                        categories = [DataValidator.clean_text(link.get_text()) for link in breadcrumb_links]
                        return " > ".join(categories)
                    else:
                        return DataValidator.clean_text(element.get_text())
            except Exception:
                continue
        return ""
    
    def close(self) -> None:
        try:
            if self.driver:
                self.driver.quit()
            if self.session:
                self.session.close()
            logger.info("Scraper resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    def export_data(self, data: List[Dict[str, Any]] = None, output_dir: str = "scraped_data", 
                   formats: List[str] = None, filename_prefix: str = None) -> Dict[str, str]:
        """Export scraped product data to various formats"""
        if formats is None:
            formats = ['json', 'csv']
        
        # Use provided data or fallback to stored products
        products = data if data is not None else self.products
        
        if not products:
            logger.warning("No products to export")
            return {}
        
        from pathlib import Path
        import os
        from datetime import datetime
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename_prefix:
            base_filename = f"{filename_prefix}_{timestamp}"
        else:
            base_filename = f"gjirafa_products_{timestamp}"
        
        exported_files = {}
        
        # Export to JSON
        if 'json' in formats:
            json_file = os.path.join(output_dir, f"{base_filename}.json")
            if DataExporter.export_to_json(products, json_file):
                exported_files['json'] = json_file
                logger.info(f"Exported to JSON: {json_file}")
        
        # Export to CSV
        if 'csv' in formats:
            csv_file = os.path.join(output_dir, f"{base_filename}.csv")
            if DataExporter.export_to_csv(products, csv_file):
                exported_files['csv'] = csv_file
                logger.info(f"Exported to CSV: {csv_file}")
        
        # Export to Excel
        if 'excel' in formats or 'xlsx' in formats:
            excel_file = os.path.join(output_dir, f"{base_filename}.xlsx")
            if DataExporter.export_to_excel(products, excel_file):
                exported_files['excel'] = excel_file
                logger.info(f"Exported to Excel: {excel_file}")
        
        return exported_files

    def _load_existing_data(self) -> None:
        """Load existing scraped data to avoid duplicates"""
        try:
            from pathlib import Path
            data_dir = Path("scraped_data")
            
            if not data_dir.exists():
                logger.info("No existing data directory found")
                return
            
            json_files = list(data_dir.glob("*.json"))
            existing_products = set()
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            if 'url' in item and 'title' in item:
                                # Create composite key for duplicate detection
                                key = f"{item['url']}|||{item['title']}"
                                existing_products.add(key)
                
                except Exception as e:
                    logger.warning(f"Error loading existing data from {json_file}: {e}")
                    continue
            
            self.existing_products = existing_products
            logger.info(f"Loaded {len(existing_products)} existing products for duplicate detection")
            
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            self.existing_products = set()
    
    def _is_duplicate(self, product_data: Dict[str, Any]) -> bool:
        """Check if a product is a duplicate based on URL and title"""
        try:
            if not hasattr(self, 'existing_products'):
                self.existing_products = set()
            
            if 'url' not in product_data or 'title' not in product_data:
                return False
            
            # Create composite key
            key = f"{product_data['url']}|||{product_data['title']}"
            
            # Check against existing products
            if key in self.existing_products:
                return True
            
            # Check against products scraped in this session
            if key in {f"{p.get('url', '')}|||{p.get('title', '')}" for p in self.products}:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False
    
    def _extract_brand_smart(self, soup: BeautifulSoup) -> str:
        """Smart brand extraction that looks in title and specific selectors"""
        
        # First try the standard selectors
        brand = self._extract_by_selectors(soup, SELECTORS["brand"])
        if brand and len(brand) > 1:
            return brand
        
        # If no brand found, extract from title
        title_element = soup.select_one('h1')
        if not title_element:
            title_element = soup.select_one('.product-title')
        if not title_element:
            title_element = soup.select_one('.product-name')
        
        if title_element:
            title = title_element.get_text(strip=True)
            
            # Common brands found on Gjirafa (case-insensitive)
            brand_patterns = [
                # Major computer brands
                r'\b(Lenovo|HP|Dell|ASUS|Acer|MSI|Apple|Alienware|Razer|Origin|CyberPowerPC)\b',
                # Phone brands
                r'\b(Samsung|iPhone|Apple|Xiaomi|Huawei|OnePlus|Google|Sony|Nokia|Oppo|Vivo)\b',
                # TV/Monitor brands
                r'\b(LG|Samsung|Sony|TCL|Hisense|Philips|Panasonic|Sharp|Toshiba|AOC|ASUS|Acer|BenQ)\b',
                # Gaming brands
                r'\b(Nintendo|PlayStation|Xbox|Steam|Valve|Corsair|Logitech|SteelSeries)\b',
                # Audio brands
                r'\b(Sony|Bose|JBL|Sennheiser|Audio-Technica|Beyerdynamic|AKG|Shure|Marshall)\b',
                # General electronics
                r'\b(Panasonic|Bosch|Siemens|Canon|Nikon|GoPro|DJI|Fitbit|Garmin)\b'
            ]
            
            for pattern in brand_patterns:
                import re
                match = re.search(pattern, title, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            # Fallback: extract first word that looks like a brand (capitalized, 2+ chars)
            words = title.split()
            for word in words[:3]:  # Check first 3 words
                clean_word = re.sub(r'[^\w]', '', word)
                if (clean_word and 
                    len(clean_word) >= 2 and 
                    clean_word[0].isupper() and 
                    not clean_word.lower() in ['kompjuter', 'laptop', 'telefon', 'monitor', 'tv', 'kufje']):
                    return clean_word
        
        return ""
