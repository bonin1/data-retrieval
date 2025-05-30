"""
Advanced Gjirafa50.com Web Scraper
"""
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import List, Dict, Optional, Set, Tuple, Any
from urllib.parse import urljoin, urlparse
import concurrent.futures
from tqdm import tqdm
import random
import json
from datetime import datetime
import uuid
import os

# Local imports
from config import (
    BASE_URL, SCRAPING_CONFIG, SELECTORS, DEFAULT_HEADERS, 
    TARGET_CATEGORIES, VALIDATION_CONFIG
)
from data_models import (
    Product, ProductImage, ProductReview, ProductSpecification, 
    SellerInfo, PriceInfo, ScrapingSession, ProductValidator
)
from anti_detection import AntiDetection, ProxyManager
from exporters import DataExporter
from utils import (
    setup_logging, clean_text, extract_price, extract_rating, 
    extract_product_id, normalize_url, extract_image_urls,
    parse_specifications, retry_on_failure, chunk_list,
    calculate_progress_stats, format_duration
)


class GjirafaScraper:
    """Advanced scraper for Gjirafa50.com"""
    
    def __init__(self, config: Optional[Dict] = None):
        # Initialize configuration
        self.config = {**SCRAPING_CONFIG, **(config or {})}
        
        # Initialize components
        self.logger = setup_logging(
            level='INFO',
            log_file='gjirafa_scraper.log',
            console=True
        )
        
        self.anti_detection = AntiDetection()
        self.proxy_manager = ProxyManager(self.config.get('proxy_file'))
        self.data_exporter = DataExporter()
        self.validator = ProductValidator()
        
        # Session state
        self.session = None
        self.scraped_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.products: List[Product] = []
        
        # Statistics
        self.stats = {
            'start_time': 0,
            'requests_made': 0,
            'products_scraped': 0,
            'errors': 0,
            'rate_limits': 0
        }
        
        self.logger.info("Gjirafa50 Scraper initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()
    
    async def _create_session(self):
        """Create aiohttp session"""
        connector = aiohttp.TCPConnector(
            limit=self.config['max_workers'],
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self.anti_detection.get_random_headers()
        )
    
    async def _close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    @retry_on_failure(max_retries=3)
    async def _fetch_page(self, url: str, retries: int = 0) -> Optional[str]:
        """Fetch a single page with error handling"""
        try:
            # Apply anti-detection measures
            headers = self.anti_detection.get_random_headers(url)
            
            # Add proxy if configured
            proxy = None
            if self.config.get('use_proxies') and self.proxy_manager.get_proxy_count() > 0:
                proxy_dict = self.proxy_manager.get_next_proxy()
                if proxy_dict:
                    proxy = proxy_dict.get('http')
            
            # Make request
            async with self.session.get(url, headers=headers, proxy=proxy) as response:
                self.stats['requests_made'] += 1
                
                # Check for rate limiting
                if response.status == 429 or response.status >= 500:
                    self.stats['rate_limits'] += 1
                    self.anti_detection.handle_rate_limiting(retries + 1)
                    if retries < self.config['max_retries']:
                        return await self._fetch_page(url, retries + 1)
                    return None
                
                if response.status == 200:
                    content = await response.text()
                    
                    # Check for rate limiting in content
                    if self.anti_detection.detect_rate_limiting(content, response.status):
                        self.stats['rate_limits'] += 1
                        self.anti_detection.handle_rate_limiting(retries + 1)
                        if retries < self.config['max_retries']:
                            return await self._fetch_page(url, retries + 1)
                        return None
                    
                    return content
                
                self.logger.warning(f"HTTP {response.status} for {url}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            self.stats['errors'] += 1
            if retries < self.config['max_retries']:
                await asyncio.sleep(2 ** retries)  # Exponential backoff
                return await self._fetch_page(url, retries + 1)
            return None
    
    def _parse_product_page(self, html: str, url: str) -> Optional[Product]:
        """Parse individual product page"""
        try:
            soup = BeautifulSoup(html, 'lxml')
            
            # Extract basic information
            product_id = extract_product_id(url)
            name = self._extract_product_name(soup)
            description = self._extract_product_description(soup)
            
            if not name:
                self.logger.warning(f"No product name found for {url}")
                return None
            
            # Extract price information
            price_info = self._extract_price_info(soup)
            
            # Extract availability
            availability = self._extract_availability(soup)
            
            # Extract categories
            categories = self._extract_categories(soup)
            
            # Extract brand and model
            brand, model = self._extract_brand_model(soup)
            
            # Extract images
            images = self._extract_images(soup, url)
            
            # Extract rating and reviews
            rating, reviews_count = self._extract_rating_reviews(soup)
            
            # Extract specifications
            specifications = self._extract_specifications(soup)
            
            # Extract seller information
            seller = self._extract_seller_info(soup)
            
            # Create product object
            product = Product(
                id=product_id,
                name=name,
                url=url,
                description=description,
                price_info=price_info,
                availability=availability,
                categories=categories,
                brand=brand,
                model=model,
                images=images,
                rating=rating,
                reviews_count=reviews_count,
                specifications=specifications,
                seller=seller
            )
            
            # Validate product
            if self.validator.is_valid_product(product):
                return product
            else:
                errors = self.validator.validate_product(product)
                self.logger.warning(f"Product validation failed for {url}: {errors}")
                return product  # Return anyway with warnings
                
        except Exception as e:
            self.logger.error(f"Error parsing product page {url}: {e}")
            return None
    
    def _extract_product_name(self, soup: BeautifulSoup) -> str:
        """Extract product name"""
        selectors = [
            'h1.product-title',
            '.product-name h1',
            'h1[data-testid="product-title"]',
            '.product-info h1',
            'h1'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return clean_text(element.get_text())
        
        return ""
    
    def _extract_product_description(self, soup: BeautifulSoup) -> str:
        """Extract product description"""
        selectors = [
            '.product-description',
            '.description',
            '.product-details',
            '[data-testid="product-description"]',
            '.product-info .description'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return clean_text(element.get_text())
        
        return ""
    
    def _extract_price_info(self, soup: BeautifulSoup) -> Optional[PriceInfo]:
        """Extract price information"""
        current_price = None
        old_price = None
        currency = "EUR"
        
        # Current price selectors
        current_price_selectors = [
            '.price .current',
            '.product-price .current-price',
            '.price-current',
            '[data-testid="current-price"]',
            '.price'
        ]
        
        for selector in current_price_selectors:
            element = soup.select_one(selector)
            if element:
                price_text = element.get_text()
                current_price = extract_price(price_text)
                if current_price:
                    break
        
        # Old price selectors
        old_price_selectors = [
            '.price .old',
            '.product-price .old-price',
            '.price-old',
            '[data-testid="old-price"]',
            '.original-price'
        ]
        
        for selector in old_price_selectors:
            element = soup.select_one(selector)
            if element:
                price_text = element.get_text()
                old_price = extract_price(price_text)
                if old_price:
                    break
        
        if current_price is not None:
            return PriceInfo(
                current_price=current_price,
                old_price=old_price,
                currency=currency
            )
        
        return None
    
    def _extract_availability(self, soup: BeautifulSoup) -> str:
        """Extract availability status"""
        selectors = [
            '.availability',
            '.stock-status',
            '[data-testid="availability"]',
            '.product-availability'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return clean_text(element.get_text())
        
        return "unknown"
    
    def _extract_categories(self, soup: BeautifulSoup) -> List[str]:
        """Extract product categories"""
        categories = []
        
        # Breadcrumb selectors
        breadcrumb_selectors = [
            '.breadcrumb a',
            '.breadcrumbs a',
            '.category-path a',
            '[data-testid="breadcrumb"] a'
        ]
        
        for selector in breadcrumb_selectors:
            elements = soup.select(selector)
            if elements:
                categories = [clean_text(el.get_text()) for el in elements[1:]]  # Skip home
                break
        
        # Category tag selectors
        if not categories:
            category_selectors = [
                '.product-categories .category',
                '.product-tags .tag',
                '.categories a'
            ]
            
            for selector in category_selectors:
                elements = soup.select(selector)
                if elements:
                    categories = [clean_text(el.get_text()) for el in elements]
                    break
        
        return [cat for cat in categories if cat]
    
    def _extract_brand_model(self, soup: BeautifulSoup) -> Tuple[str, str]:
        """Extract brand and model"""
        brand = ""
        model = ""
        
        # Brand selectors
        brand_selectors = [
            '.product-brand',
            '.brand',
            '[data-testid="brand"]',
            '.manufacturer'
        ]
        
        for selector in brand_selectors:
            element = soup.select_one(selector)
            if element:
                brand = clean_text(element.get_text())
                break
        
        # Model selectors
        model_selectors = [
            '.product-model',
            '.model',
            '[data-testid="model"]'
        ]
        
        for selector in model_selectors:
            element = soup.select_one(selector)
            if element:
                model = clean_text(element.get_text())
                break
        
        return brand, model
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[ProductImage]:
        """Extract product images"""
        images = []
        
        # Image selectors
        image_selectors = [
            '.product-images img',
            '.gallery img',
            '.product-gallery img',
            '[data-testid="product-image"]'
        ]
        
        for selector in image_selectors:
            img_elements = soup.select(selector)
            if img_elements:
                urls = extract_image_urls(img_elements, base_url)
                for i, url in enumerate(urls):
                    images.append(ProductImage(
                        url=url,
                        is_primary=(i == 0)
                    ))
                break
        
        return images
    
    def _extract_rating_reviews(self, soup: BeautifulSoup) -> Tuple[Optional[float], Optional[int]]:
        """Extract rating and review count"""
        rating = None
        reviews_count = None
        
        # Rating selectors
        rating_selectors = [
            '.rating .stars',
            '.review-rating',
            '[data-testid="rating"]',
            '.product-rating'
        ]
        
        for selector in rating_selectors:
            element = soup.select_one(selector)
            if element:
                rating_text = element.get_text() or element.get('title', '')
                rating = extract_rating(rating_text)
                if rating:
                    break
        
        # Reviews count selectors
        reviews_selectors = [
            '.reviews-count',
            '.rating-count',
            '[data-testid="reviews-count"]',
            '.review-summary .count'
        ]
        
        for selector in reviews_selectors:
            element = soup.select_one(selector)
            if element:
                count_text = element.get_text()
                # Extract number from text like "123 reviews"
                import re
                match = re.search(r'(\d+)', count_text)
                if match:
                    reviews_count = int(match.group(1))
                    break
        
        return rating, reviews_count
    
    def _extract_specifications(self, soup: BeautifulSoup) -> List[ProductSpecification]:
        """Extract product specifications"""
        specifications = []
        
        # Specification table selectors
        spec_selectors = [
            '.specifications table',
            '.product-specs table',
            '.details-table',
            '[data-testid="specifications"]'
        ]
        
        for selector in spec_selectors:
            element = soup.select_one(selector)
            if element:
                specs_dict = {}
                
                # Extract from table rows
                rows = element.select('tr')
                for row in rows:
                    cells = row.select('td, th')
                    if len(cells) >= 2:
                        key = clean_text(cells[0].get_text())
                        value = clean_text(cells[1].get_text())
                        if key and value:
                            specs_dict[key] = value
                
                if specs_dict:
                    specifications.append(ProductSpecification(
                        category="General",
                        specifications=specs_dict
                    ))
                break
        
        return specifications
    
    def _extract_seller_info(self, soup: BeautifulSoup) -> Optional[SellerInfo]:
        """Extract seller information"""
        seller_selectors = [
            '.seller-info',
            '.vendor-info',
            '[data-testid="seller"]',
            '.product-seller'
        ]
        
        for selector in seller_selectors:
            element = soup.select_one(selector)
            if element:
                name = clean_text(element.get_text())
                if name:
                    return SellerInfo(name=name)
        
        return None
    
    async def discover_categories(self) -> List[str]:
        """Discover available product categories"""
        self.logger.info("Discovering product categories...")
        
        try:
            # Fetch main page
            main_page = await self._fetch_page(BASE_URL)
            if not main_page:
                self.logger.warning("Could not fetch main page, using predefined categories")
                return TARGET_CATEGORIES
            
            soup = BeautifulSoup(main_page, 'lxml')
            categories = []
            
            # Category menu selectors
            category_selectors = [
                '.main-menu a[href*="/category/"]',
                '.categories a[href*="/c/"]',
                '.nav-categories a',
                '[data-testid="category-link"]'
            ]
            
            for selector in category_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        href = element.get('href', '')
                        if href:
                            # Extract category slug from URL
                            category_slug = href.split('/')[-1]
                            if category_slug and category_slug not in categories:
                                categories.append(category_slug)
            
            if categories:
                self.logger.info(f"Discovered {len(categories)} categories")
                return categories
            else:
                self.logger.warning("No categories found, using predefined list")
                return TARGET_CATEGORIES
                
        except Exception as e:
            self.logger.error(f"Error discovering categories: {e}")
            return TARGET_CATEGORIES
    
    async def get_category_product_urls(self, category: str, max_pages: int = None) -> List[str]:
        """Get all product URLs from a category"""
        if max_pages is None:
            max_pages = self.config['max_pages_per_category']
        
        self.logger.info(f"Scraping category: {category}")
        product_urls = []
        page = 1
        
        while page <= max_pages:
            # Construct category URL
            category_url = f"{BASE_URL}/category/{category}?page={page}"
            
            # Add delay between requests
            if page > 1:
                self.anti_detection.wait_with_jitter(
                    self.config['delay_range'][0],
                    self.config['delay_range'][1]
                )
            
            # Fetch category page
            html = await self._fetch_page(category_url)
            if not html:
                self.logger.warning(f"Could not fetch category page: {category_url}")
                break
            
            soup = BeautifulSoup(html, 'lxml')
            
            # Extract product links
            page_product_urls = self._extract_product_links(soup)
            
            if not page_product_urls:
                self.logger.info(f"No more products found on page {page} for category {category}")
                break
            
            product_urls.extend(page_product_urls)
            self.logger.info(f"Found {len(page_product_urls)} products on page {page} of {category}")
            
            # Check for next page
            if not self._has_next_page(soup):
                break
            
            page += 1
        
        self.logger.info(f"Found {len(product_urls)} total products in category {category}")
        return product_urls
    
    def _extract_product_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract product links from category page"""
        links = []
        
        # Product link selectors
        link_selectors = [
            '.product-item a[href*="/product/"]',
            '.product-grid a[href*="/p/"]',
            '.product-card a',
            '[data-testid="product-link"]'
        ]
        
        for selector in link_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    href = element.get('href')
                    if href:
                        full_url = normalize_url(href, BASE_URL)
                        if full_url not in links:
                            links.append(full_url)
                break
        
        return links
    
    def _has_next_page(self, soup: BeautifulSoup) -> bool:
        """Check if there's a next page"""
        next_selectors = [
            '.pagination .next:not(.disabled)',
            '.pagination a[aria-label="Next"]',
            '.pager .next',
            '[data-testid="next-page"]:not(.disabled)'
        ]
        
        for selector in next_selectors:
            element = soup.select_one(selector)
            if element:
                return True
        
        return False
    
    async def scrape_products_async(self, product_urls: List[str]) -> List[Product]:
        """Scrape products asynchronously"""
        self.logger.info(f"Starting async scraping of {len(product_urls)} products")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config['semaphore_limit'])
        
        # Progress tracking
        progress_bar = tqdm(total=len(product_urls), desc="Scraping products")
        
        async def scrape_single_product(url: str) -> Optional[Product]:
            async with semaphore:
                try:
                    # Add delay
                    self.anti_detection.wait_with_jitter(
                        self.config['delay_range'][0],
                        self.config['delay_range'][1]
                    )
                    
                    # Fetch and parse
                    html = await self._fetch_page(url)
                    if html:
                        product = self._parse_product_page(html, url)
                        if product:
                            self.stats['products_scraped'] += 1
                            progress_bar.update(1)
                            return product
                    
                    self.failed_urls.add(url)
                    progress_bar.update(1)
                    return None
                    
                except Exception as e:
                    self.logger.error(f"Error scraping product {url}: {e}")
                    self.failed_urls.add(url)
                    progress_bar.update(1)
                    return None
        
        # Execute all tasks
        tasks = [scrape_single_product(url) for url in product_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        progress_bar.close()
        
        # Filter successful results
        products = [result for result in results if isinstance(result, Product)]
        
        self.logger.info(f"Successfully scraped {len(products)} products")
        return products
    
    async def scrape_all_products(self, categories: Optional[List[str]] = None) -> List[Product]:
        """Main method to scrape all products"""
        self.stats['start_time'] = time.time()
        
        # Create scraping session
        session = ScrapingSession(
            session_id=str(uuid.uuid4()),
            start_time=datetime.now()
        )
        
        try:
            # Discover categories if not provided
            if not categories:
                categories = await self.discover_categories()
            
            self.logger.info(f"Starting scraping session for {len(categories)} categories")
            
            all_product_urls = []
            
            # Collect URLs from all categories
            for category in categories:
                try:
                    category_urls = await self.get_category_product_urls(category)
                    all_product_urls.extend(category_urls)
                    session.categories_scraped.append(category)
                    
                    # Add delay between categories
                    if category != categories[-1]:
                        await asyncio.sleep(random.uniform(2, 5))
                        
                except Exception as e:
                    error_msg = f"Error scraping category {category}: {e}"
                    self.logger.error(error_msg)
                    session.errors.append(error_msg)
            
            # Remove duplicates
            unique_urls = list(set(all_product_urls))
            self.logger.info(f"Found {len(unique_urls)} unique product URLs")
            
            session.total_products = len(unique_urls)
            
            # Scrape all products
            products = await self.scrape_products_async(unique_urls)
            
            session.successful_products = len(products)
            session.failed_products = len(unique_urls) - len(products)
            session.end_time = datetime.now()
            
            # Log session summary
            self.logger.info(f"Scraping session completed:")
            self.logger.info(f"- Duration: {format_duration(session.get_duration())}")
            self.logger.info(f"- Success rate: {session.get_success_rate()}%")
            self.logger.info(f"- Products scraped: {session.successful_products}")
            self.logger.info(f"- Failed: {session.failed_products}")
            self.logger.info(f"- Requests made: {self.stats['requests_made']}")
            self.logger.info(f"- Rate limits encountered: {self.stats['rate_limits']}")
            
            # Export session report
            self.data_exporter.export_session_report(session, products)
            
            return products
            
        except Exception as e:
            self.logger.error(f"Critical error in scraping session: {e}")
            session.errors.append(str(e))
            session.end_time = datetime.now()
            raise
    
    def export_products(
        self, 
        products: List[Product], 
        formats: List[str] = None,
        filename_prefix: str = "gjirafa_products"
    ) -> Dict[str, str]:
        """Export products to various formats"""
        if not formats:
            formats = ['json', 'csv', 'excel']
        
        exported_files = {}
        
        for format_type in formats:
            try:
                if format_type.lower() == 'json':
                    filepath = self.data_exporter.export_to_json(products, f"{filename_prefix}.json")
                    exported_files['json'] = filepath
                    
                elif format_type.lower() == 'csv':
                    filepath = self.data_exporter.export_to_csv(products, f"{filename_prefix}.csv")
                    exported_files['csv'] = filepath
                    
                elif format_type.lower() in ['excel', 'xlsx']:
                    filepath = self.data_exporter.export_to_excel(products, f"{filename_prefix}.xlsx")
                    exported_files['excel'] = filepath
                    
                self.logger.info(f"Exported {len(products)} products to {format_type}: {filepath}")
                
            except Exception as e:
                self.logger.error(f"Error exporting to {format_type}: {e}")
        
        return exported_files
    
    async def run_scraper(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main entry point for running the scraper"""
        self.logger.info("Starting Gjirafa50 scraper...")
        
        try:
            # Scrape products
            products = await self.scrape_all_products(categories)
            
            # Export products
            exported_files = self.export_products(products)
            
            return {
                'success': True,
                'products_count': len(products),
                'exported_files': exported_files,
                'stats': self.stats
            }
            
        except Exception as e:
            self.logger.error(f"Scraper failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }
