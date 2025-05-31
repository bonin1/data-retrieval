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
                if href:
                    if (href.startswith('/') and 
                        not href.startswith('//') and
                        any(product_type in href.lower() for product_type in [
                            'laptop', 'kompjuter', 'telefon', 'tv', 'monitor', 
                            'kufje', 'tablet', 'kamera', 'audio', 'gaming'
                        ]) and
                        len(href) > 10):
                        
                        full_url = URLHelper.normalize_url(href, self.base_url)
                        if full_url not in product_urls:
                            product_urls.append(full_url)
        
        return product_urls
    
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
                            if href:
                                if (href.startswith('/') and 
                                    not href.startswith('//') and
                                    any(product_type in href.lower() for product_type in [
                                        'laptop', 'kompjuter', 'telefon', 'tv', 'monitor', 
                                        'kufje', 'tablet', 'kamera', 'audio', 'gaming'
                                    ]) and
                                    len(href) > 10): 
                                    
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
            
            product_data['price'] = self._extract_by_selectors(soup, SELECTORS["price"])
            product_data['original_price'] = self._extract_by_selectors(soup, SELECTORS["original_price"])
            
            product_data['description'] = self._extract_by_selectors(soup, SELECTORS["description"])
            
            product_data['images'] = self._extract_images(soup)
            
            product_data['specifications'] = self._extract_specifications(soup)
            
            product_data['brand'] = self._extract_by_selectors(soup, SELECTORS["brand"])
            product_data['category'] = self._extract_category(soup)
            product_data['availability'] = self._extract_by_selectors(soup, SELECTORS["availability"])
            product_data['rating'] = self._extract_by_selectors(soup, SELECTORS["rating"])
            product_data['reviews_count'] = self._extract_by_selectors(soup, SELECTORS["reviews_count"])
            
            validated_data = DataValidator.validate_product_data(product_data)
            
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
    
    def _extract_images(self, soup: BeautifulSoup) -> List[str]:
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
        for selector in SELECTORS["specifications"]:
            try:
                spec_element = soup.select_one(selector)
                if spec_element:
                    return DataValidator.extract_specifications(spec_element)
            except Exception:
                continue
        return {}
    
    def _extract_category(self, soup: BeautifulSoup) -> str:
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
    
    def scrape_products(self, 
                       urls: List[str] = None, 
                       max_products: int = None,
                       save_interval: int = 100) -> List[Dict[str, Any]]:
        
        if not urls:
            logger.info("No URLs provided, discovering products automatically...")
            urls = self._discover_all_products()
        
        if max_products:
            urls = urls[:max_products]
        
        logger.info(f"Starting to scrape {len(urls)} products")
        
        scraped_products = []
        failed_urls = []
        
        with tqdm(total=len(urls), desc="Scraping products") as pbar:
            for i, url in enumerate(urls):
                if url in self.scraped_urls:
                    pbar.update(1)
                    continue
                
                try:
                    product_data = self.extract_product_data(url)
                    if product_data:
                        scraped_products.append(product_data)
                        self.scraped_urls.add(url)
                    else:
                        failed_urls.append(url)
                    
                    if (i + 1) % save_interval == 0:
                        self._save_progress(scraped_products, f"progress_{i+1}")
                    
                    time.sleep(self.config.REQUEST_DELAY)
                    
                except Exception as e:
                    logger.error(f"Error scraping {url}: {e}")
                    failed_urls.append(url)
                
                pbar.update(1)
        
        logger.info(f"Scraping completed. Success: {len(scraped_products)}, Failed: {len(failed_urls)}")
        
        if failed_urls:
            logger.info(f"Failed URLs: {failed_urls[:10]}...") 
        
        self.products.extend(scraped_products)
        return scraped_products
    def _discover_all_products(self) -> List[str]:
        all_product_urls = []
        
        category_urls = self.discover_category_urls()
        
        if not category_urls:
            from config import CATEGORY_URLS
            category_urls = [self.base_url + cat for cat in CATEGORY_URLS]
            logger.info(f"Using predefined categories: {len(category_urls)}")
        
        logger.info(f"Discovering products from {len(category_urls)} categories")
        
        for category_url in category_urls:
            try:
                products = self.discover_product_urls(category_url, max_pages=5)
                all_product_urls.extend(products)
                logger.info(f"Found {len(products)} products in {category_url}")
                
                time.sleep(self.config.REQUEST_DELAY)
                
            except Exception as e:
                logger.error(f"Error discovering products from {category_url}: {e}")
        
        all_product_urls = list(set(all_product_urls))
        logger.info(f"Total unique products discovered: {len(all_product_urls)}")
        
        return all_product_urls
    
    def _save_progress(self, data: List[Dict], filename_prefix: str) -> None:
        try:
            filename = f"{self.config.OUTPUT_DIR}/{filename_prefix}_{int(time.time())}.json"
            DataExporter.export_to_json(data, filename)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def export_data(self, 
                   data: List[Dict] = None, 
                   formats: List[str] = None,
                   filename_prefix: str = "gjirafa_products") -> Dict[str, str]:
        
        if data is None:
            data = self.products
        
        if not data:
            logger.warning("No data to export")
            return {}
        
        if formats is None:
            formats = ['json', 'csv']
        
        exported_files = {}
        timestamp = int(time.time())
        
        for format_type in formats:
            try:
                filename = f"{self.config.OUTPUT_DIR}/{filename_prefix}_{timestamp}.{format_type}"
                
                if format_type == 'json':
                    success = DataExporter.export_to_json(data, filename)
                elif format_type == 'csv':
                    success = DataExporter.export_to_csv(data, filename)
                elif format_type == 'excel':
                    filename = filename.replace('.excel', '.xlsx')
                    success = DataExporter.export_to_excel(data, filename)
                else:
                    logger.warning(f"Unsupported format: {format_type}")
                    continue
                
                if success:
                    exported_files[format_type] = filename
                
            except Exception as e:
                logger.error(f"Error exporting to {format_type}: {e}")
        
        try:
            summary = DataExporter.generate_summary_report(data)
            summary_file = f"{self.config.OUTPUT_DIR}/{filename_prefix}_summary_{timestamp}.json"
            DataExporter.export_to_json([summary], summary_file)
            exported_files['summary'] = summary_file
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
        
        return exported_files
    
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
