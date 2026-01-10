"""
Utility Functions for Gjirafa50 Scraper
=======================================
Data validation, cleaning, and export utilities.
"""

import re
import json
import csv
import pandas as pd
import validators
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Image Filtering Constants
# =============================================================================

# Patterns for images to EXCLUDE (site assets, logos, icons, etc.)
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
    r'gjirafa50\.svg',
    r'gjirafa\.svg',
    # Known placeholder/banner UUIDs
    r'b6db299a-d4e4-4afe-8226-ab5c21a7fd53',
    r'ce80ffe8-09e5-4635-8c44-45dced59df48',
    r'e6fd61b7-0171-43c1-aaf9-8b49c445a0c1',
    r'a4721784-e748-4ad7-9b85-d8011377962d',
    r'28acc4e8-dff5-4ba7-8bbd-9a446acb5dc7',
    r'e664607f-8727-4844-aa4f-dfaf6a220143',
    r'8f204494-19fd-4e2d-8651-fcd76188184a',
]

# Minimum width for product images
MIN_IMAGE_WIDTH = 150

class DataValidator:
    """Data validation and cleaning utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text - remove extra whitespace and control characters"""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Remove excessive special characters
        text = re.sub(r'[•·▪▸►]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_price(price_text: str) -> Optional[float]:
        """Extract numeric price from text, handling various formats"""
        if not price_text:
            return None
        
        # Remove currency symbols and extra whitespace
        price_text = re.sub(r'[€$£¥]', '', price_text)
        price_text = re.sub(r'\s+', '', price_text)
        
        # Handle European format (comma as decimal separator)
        # e.g., "1.234,56" -> "1234.56" or "234,50" -> "234.50"
        if ',' in price_text and '.' in price_text:
            # Both present - assume European: 1.234,56
            price_text = price_text.replace('.', '').replace(',', '.')
        elif ',' in price_text:
            # Only comma - assume it's decimal separator
            price_text = price_text.replace(',', '.')
        
        # Extract the number
        match = re.search(r'(\d+(?:\.\d{1,2})?)', price_text)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                return None
        
        return None
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        if not url:
            return False
        return bool(validators.url(url))
    
    @staticmethod
    def is_product_image(url: str) -> bool:
        """Check if URL is a valid product image (not a site asset)"""
        if not url:
            return False
        
        url_lower = url.lower()
        
        # Check exclude patterns
        for pattern in IMAGE_EXCLUDE_PATTERNS:
            if re.search(pattern, url_lower, re.IGNORECASE):
                return False
        
        # Must be from the product CDN and be an actual image
        if 'iqq6kf0xmf.gjirafa.net/images/' in url_lower:
            if any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                # Check for small thumbnail indicators
                width_match = re.search(r'\?w=(\d+)', url_lower)
                if width_match:
                    width = int(width_match.group(1))
                    if width < MIN_IMAGE_WIDTH:
                        return False
                return True
        
        return False
    
    @staticmethod
    def clean_image_urls(urls: List[str], base_url: str = "") -> List[str]:
        """Clean and filter image URLs to only include product images"""
        cleaned_urls = []
        seen = set()
        
        for url in urls:
            if not url:
                continue
            
            # Normalize URL
            if url.startswith('//'):
                url = 'https:' + url
            elif url.startswith('/'):
                url = base_url.rstrip('/') + url
            elif not url.startswith(('http://', 'https://')):
                url = base_url.rstrip('/') + '/' + url.lstrip('/')
            
            # Remove query params for deduplication
            normalized = url.split('?')[0]
            
            # Only add if it's a product image and not seen
            if DataValidator.is_product_image(url) and normalized not in seen:
                seen.add(normalized)
                cleaned_urls.append(url)
        
        return cleaned_urls
    
    @staticmethod
    def extract_specifications(spec_element) -> Dict[str, str]:
        """Extract specifications from various HTML structures"""
        specs = {}
        
        if not spec_element:
            return specs
        
        try:
            # Try table rows first
            rows = spec_element.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    key = DataValidator.clean_text(cells[0].get_text())
                    value = DataValidator.clean_text(cells[1].get_text())
                    if key and value and key != value and len(key) < 100:
                        specs[key] = value
            
            # Try definition lists
            if not specs:
                dt_elements = spec_element.find_all('dt')
                dd_elements = spec_element.find_all('dd')
                
                for dt, dd in zip(dt_elements, dd_elements):
                    key = DataValidator.clean_text(dt.get_text())
                    value = DataValidator.clean_text(dd.get_text())
                    if key and value and len(key) < 100:
                        specs[key] = value
            
            # Try key-value pairs in divs/spans
            if not specs:
                # Look for label/value patterns
                for item in spec_element.find_all(['div', 'li', 'span']):
                    # Try finding label and value children
                    label = item.find(class_=re.compile(r'label|key|name|spec-name', re.I))
                    value_el = item.find(class_=re.compile(r'value|data|spec-value', re.I))
                    
                    if label and value_el:
                        key = DataValidator.clean_text(label.get_text())
                        value = DataValidator.clean_text(value_el.get_text())
                        if key and value and len(key) < 100:
                            specs[key] = value
                    else:
                        # Try colon-separated text
                        text = DataValidator.clean_text(item.get_text())
                        if ':' in text and text.count(':') == 1:
                            parts = text.split(':', 1)
                            if len(parts) == 2:
                                key = parts[0].strip()
                                value = parts[1].strip()
                                if key and value and len(key) < 50:
                                    specs[key] = value
                                    
        except Exception as e:
            logger.error(f"Error extracting specifications: {e}")
        
        return specs
    
    @staticmethod
    def validate_product_data(product: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean product data dictionary"""
        validated = {}
        
        # Basic fields
        validated['title'] = DataValidator.clean_text(product.get('title', ''))
        validated['url'] = product.get('url', '')
        validated['scraped_at'] = datetime.now().isoformat()
        
        # Price handling
        price = product.get('price')
        if isinstance(price, (int, float)):
            validated['price'] = float(price) if price > 0 else None
        else:
            validated['price'] = DataValidator.extract_price(str(price)) if price else None
        
        original_price = product.get('original_price')
        if isinstance(original_price, (int, float)):
            validated['original_price'] = float(original_price) if original_price > 0 else None
        else:
            validated['original_price'] = DataValidator.extract_price(str(original_price)) if original_price else None
        
        # Calculate discount percentage
        if validated['price'] and validated['original_price']:
            if validated['original_price'] > validated['price']:
                discount = ((validated['original_price'] - validated['price']) / validated['original_price']) * 100
                validated['discount_percentage'] = round(discount, 2)
            else:
                validated['discount_percentage'] = None
        else:
            validated['discount_percentage'] = None
        
        # Text fields
        validated['description'] = DataValidator.clean_text(product.get('description', ''))
        validated['brand'] = DataValidator.clean_text(product.get('brand', ''))
        validated['category'] = DataValidator.clean_text(product.get('category', ''))
        validated['availability'] = DataValidator.clean_text(product.get('availability', ''))
        
        # Images - clean and filter
        base_url = product.get('base_url', 'https://gjirafa50.com')
        images = product.get('images', [])
        if isinstance(images, str):
            images = [images]
        validated['images'] = DataValidator.clean_image_urls(images, base_url)
        validated['image_count'] = len(validated['images'])
        validated['main_image'] = validated['images'][0] if validated['images'] else None
        
        # Specifications
        specs = product.get('specifications', {})
        if isinstance(specs, dict):
            validated['specifications'] = specs
        else:
            validated['specifications'] = {}
        
        # Rating - validate range
        try:
            rating = product.get('rating')
            if rating is not None:
                rating = float(rating)
                validated['rating'] = rating if 0 <= rating <= 5 else None
            else:
                validated['rating'] = None
        except (ValueError, TypeError):
            validated['rating'] = None
        
        # Reviews count - validate non-negative
        try:
            reviews_count = product.get('reviews_count')
            if reviews_count is not None:
                reviews_count = int(reviews_count)
                validated['reviews_count'] = reviews_count if reviews_count >= 0 else None
            else:
                validated['reviews_count'] = None
        except (ValueError, TypeError):
            validated['reviews_count'] = None
        
        # SKU
        validated['sku'] = product.get('sku', '')
        
        return validated

class DataExporter:
    
    @staticmethod
    def ensure_directory(filepath: str) -> None:
        """Ensure directory exists for filepath"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def export_to_json(data: List[Dict], filepath: str, indent: int = 2) -> bool:
        try:
            DataExporter.ensure_directory(filepath)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            
            logger.info(f"Exported {len(data)} products to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False
    
    @staticmethod
    def export_to_csv(data: List[Dict], filepath: str) -> bool:
        try:
            if not data:
                logger.warning("No data to export")
                return False
            
            DataExporter.ensure_directory(filepath)
            
            flattened_data = []
            for item in data:
                flattened = DataExporter.flatten_dict(item)
                flattened_data.append(flattened)
            
            df = pd.DataFrame(flattened_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            logger.info(f"Exported {len(data)} products to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    @staticmethod
    def export_to_excel(data: List[Dict], filepath: str) -> bool:
        try:
            if not data:
                logger.warning("No data to export")
                return False
            
            DataExporter.ensure_directory(filepath)
            
            flattened_data = []
            for item in data:
                flattened = DataExporter.flatten_dict(item)
                flattened_data.append(flattened)
            
            df = pd.DataFrame(flattened_data)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Products', index=False)
            
            logger.info(f"Exported {len(data)} products to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return False
    
    @staticmethod
    def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(DataExporter.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if v and isinstance(v[0], str):
                    items.append((new_key, ', '.join(v)))
                else:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    @staticmethod
    def generate_summary_report(data: List[Dict]) -> Dict[str, Any]:
        if not data:
            return {}
        
        summary = {
            'total_products': len(data),
            'scraped_at': datetime.now().isoformat(),
            'price_statistics': {},
            'category_distribution': {},
            'brand_distribution': {},
            'availability_statistics': {},
            'data_quality': {}
        }
        
        prices = [item.get('price') for item in data if item.get('price')]
        if prices:
            summary['price_statistics'] = {
                'min_price': min(prices),
                'max_price': max(prices),
                'avg_price': sum(prices) / len(prices),
                'products_with_price': len(prices)
            }
        
        categories = [item.get('category') for item in data if item.get('category')]
        summary['category_distribution'] = pd.Series(categories).value_counts().to_dict()
        
        brands = [item.get('brand') for item in data if item.get('brand')]
        summary['brand_distribution'] = pd.Series(brands).value_counts().head(10).to_dict()
        
        summary['data_quality'] = {
            'products_with_title': len([d for d in data if d.get('title')]),
            'products_with_price': len([d for d in data if d.get('price')]),
            'products_with_description': len([d for d in data if d.get('description')]),
            'products_with_images': len([d for d in data if d.get('images')]),
            'products_with_specifications': len([d for d in data if d.get('specifications')]),
        }
        
        return summary

class URLHelper:
    
    @staticmethod
    def normalize_url(url: str, base_url: str = "https://gjirafa50.com") -> str:
        if not url:
            return ""
        
        if url.startswith('//'):
            return 'https:' + url
        elif url.startswith('/'):
            return base_url.rstrip('/') + url
        elif url.startswith(('http://', 'https://')):
            return url
        else:
            return base_url.rstrip('/') + '/' + url.lstrip('/')
    
    @staticmethod
    def extract_product_id(url: str) -> Optional[str]:
        patterns = [
            r'/product/(\d+)',
            r'/p/(\d+)',
            r'product-(\d+)',
            r'id=(\d+)',
            r'/(\d+)/?$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
