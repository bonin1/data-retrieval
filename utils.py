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

class DataValidator:
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text.strip())
        
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text
    
    @staticmethod
    def extract_price(price_text: str) -> Optional[float]:
        """Extract numeric price from text"""
        if not price_text:
            return None
        
        price_text = re.sub(r'[^\d.,]', '', price_text)
        
        if ',' in price_text and '.' in price_text:
            price_text = price_text.replace(',', '')
        elif ',' in price_text:
            price_text = price_text.replace(',', '.')
        
        try:
            return float(price_text)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def validate_url(url: str) -> bool:
        return validators.url(url) if url else False
    
    @staticmethod
    def clean_image_urls(urls: List[str], base_url: str = "") -> List[str]:
        cleaned_urls = []
        
        for url in urls:
            if not url:
                continue
                
            if url.startswith('//'):
                url = 'https:' + url
            elif url.startswith('/'):
                url = base_url.rstrip('/') + url
            elif not url.startswith(('http://', 'https://')):
                url = base_url.rstrip('/') + '/' + url.lstrip('/')
            
            if DataValidator.validate_url(url):
                cleaned_urls.append(url)
        
        return list(set(cleaned_urls))
    
    @staticmethod
    def extract_specifications(spec_element) -> Dict[str, str]:
        specs = {}
        
        if not spec_element:
            return specs
        
        try:
            rows = spec_element.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    key = DataValidator.clean_text(cells[0].get_text())
                    value = DataValidator.clean_text(cells[1].get_text())
                    if key and value:
                        specs[key] = value
            
            if not specs:
                dt_elements = spec_element.find_all('dt')
                dd_elements = spec_element.find_all('dd')
                
                for dt, dd in zip(dt_elements, dd_elements):
                    key = DataValidator.clean_text(dt.get_text())
                    value = DataValidator.clean_text(dd.get_text())
                    if key and value:
                        specs[key] = value
            
            if not specs:
                items = spec_element.find_all(['div', 'li'])
                for item in items:
                    text = DataValidator.clean_text(item.get_text())
                    if ':' in text:
                        parts = text.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key and value:
                                specs[key] = value
        except Exception as e:
            logger.error(f"Error extracting specifications: {e}")
        
        return specs
    
    @staticmethod
    def validate_product_data(product: Dict[str, Any]) -> Dict[str, Any]:
        validated = {}
        
        validated['title'] = DataValidator.clean_text(product.get('title', ''))
        validated['url'] = product.get('url', '')
        validated['scraped_at'] = datetime.now().isoformat()
        
        price = DataValidator.extract_price(str(product.get('price', '')))
        validated['price'] = price if price and price > 0 else None
        
        original_price = DataValidator.extract_price(str(product.get('original_price', '')))
        validated['original_price'] = original_price if original_price and original_price > 0 else None
        
        if validated['price'] and validated['original_price']:
            discount = ((validated['original_price'] - validated['price']) / validated['original_price']) * 100
            validated['discount_percentage'] = round(discount, 2)
        else:
            validated['discount_percentage'] = None
        
        validated['description'] = DataValidator.clean_text(product.get('description', ''))
        validated['brand'] = DataValidator.clean_text(product.get('brand', ''))
        validated['category'] = DataValidator.clean_text(product.get('category', ''))
        validated['availability'] = DataValidator.clean_text(product.get('availability', ''))
        
        base_url = product.get('base_url', 'https://gjirafa50.com')
        images = product.get('images', [])
        if isinstance(images, str):
            images = [images]
        validated['images'] = DataValidator.clean_image_urls(images, base_url)
        validated['image_count'] = len(validated['images'])
        validated['main_image'] = validated['images'][0] if validated['images'] else None
        
        specs = product.get('specifications', {})
        if isinstance(specs, dict):
            validated['specifications'] = specs
        else:
            validated['specifications'] = {}
        
        try:
            rating = float(product.get('rating', 0)) if product.get('rating') else None
            validated['rating'] = rating if rating is not None and 0 <= rating <= 5 else None
        except (ValueError, TypeError):
            validated['rating'] = None
        
        try:
            reviews_count = int(product.get('reviews_count', 0)) if product.get('reviews_count') else None
            validated['reviews_count'] = reviews_count if reviews_count is not None and reviews_count >= 0 else None
        except (ValueError, TypeError):
            validated['reviews_count'] = None
        
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
