"""
Utility functions for the Gjirafa50.com scraper
"""
import re
import logging
import os
import json
import time
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime
import hashlib


def setup_logging(level: str = 'INFO', log_file: str = 'scraper.log', console: bool = True) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('gjirafa_scraper')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove extra whitespace and line breaks
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common unwanted characters
    text = text.replace('\u00a0', ' ')  # Non-breaking space
    text = text.replace('\u200b', '')   # Zero-width space
    text = text.replace('\u2028', ' ')  # Line separator
    text = text.replace('\u2029', ' ')  # Paragraph separator
    
    return text.strip()


def extract_price(price_text: str) -> Optional[float]:
    """Extract price from text"""
    if not price_text:
        return None
    
    # Remove currency symbols and extra text
    price_text = re.sub(r'[^\d.,\s]', '', price_text)
    
    # Find price patterns
    price_patterns = [
        r'(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})',  # 1,234.56 or 1.234,56
        r'(\d{1,3}(?:[.,]\d{3})*)',           # 1,234 or 1.234
        r'(\d+[.,]\d{2})',                    # 123.45
        r'(\d+)',                             # 123
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, price_text)
        if match:
            price_str = match.group(1)
            # Normalize decimal separator
            if ',' in price_str and '.' in price_str:
                # Determine which is decimal separator
                if price_str.rindex(',') > price_str.rindex('.'):
                    price_str = price_str.replace('.', '').replace(',', '.')
                else:
                    price_str = price_str.replace(',', '')
            elif ',' in price_str:
                # Check if comma is likely decimal separator
                if len(price_str.split(',')[-1]) == 2:
                    price_str = price_str.replace(',', '.')
                else:
                    price_str = price_str.replace(',', '')
            
            try:
                return float(price_str)
            except ValueError:
                continue
    
    return None


def extract_rating(rating_text: str) -> Optional[float]:
    """Extract rating from text"""
    if not rating_text:
        return None
    
    # Look for rating patterns
    rating_patterns = [
        r'(\d[.,]\d)', # 4.5 or 4,5
        r'(\d)/5',     # 4/5
        r'(\d)\s*stars?', # 4 stars
        r'(\d)',       # Just a number
    ]
    
    for pattern in rating_patterns:
        match = re.search(pattern, rating_text)
        if match:
            rating_str = match.group(1).replace(',', '.')
            try:
                rating = float(rating_str)
                return min(5.0, max(0.0, rating))  # Clamp between 0-5
            except ValueError:
                continue
    
    return None


def extract_product_id(url: str) -> Optional[str]:
    """Extract product ID from URL"""
    # Common product ID patterns
    patterns = [
        r'/product/(\d+)',
        r'/p/(\d+)',
        r'id=(\d+)',
        r'product-(\d+)',
        r'/(\d+)/?$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Generate ID from URL if no pattern matches
    return hashlib.md5(url.encode()).hexdigest()[:10]


def normalize_url(url: str, base_url: str = "https://gjirafa50.com") -> str:
    """Normalize and complete URLs"""
    if not url:
        return ""
    
    # Remove fragments and clean up
    url = url.split('#')[0].strip()
    
    # Make absolute URL
    if url.startswith('//'):
        url = 'https:' + url
    elif url.startswith('/'):
        url = urljoin(base_url, url)
    elif not url.startswith('http'):
        url = urljoin(base_url, url)
    
    return url


def extract_image_urls(image_elements: List, base_url: str = "https://gjirafa50.com") -> List[str]:
    """Extract and normalize image URLs"""
    urls = []
    
    for img in image_elements:
        # Try different attributes
        src = (
            img.get('src') or 
            img.get('data-src') or 
            img.get('data-lazy-src') or
            img.get('data-original')
        )
        
        if src:
            normalized_url = normalize_url(src, base_url)
            if normalized_url and normalized_url not in urls:
                urls.append(normalized_url)
    
    return urls


def parse_specifications(spec_text: str) -> Dict[str, str]:
    """Parse product specifications from text"""
    specs = {}
    
    if not spec_text:
        return specs
    
    # Split by common separators
    lines = re.split(r'[\n\r|;]', spec_text)
    
    for line in lines:
        line = clean_text(line)
        if not line:
            continue
        
        # Look for key-value patterns
        patterns = [
            r'^([^:]+):\s*(.+)$',       # Key: Value
            r'^([^-]+)-\s*(.+)$',       # Key - Value
            r'^([^=]+)=\s*(.+)$',       # Key = Value
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                key = clean_text(match.group(1))
                value = clean_text(match.group(2))
                if key and value:
                    specs[key] = value
                break
    
    return specs


def create_output_directory(output_dir: str) -> str:
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def generate_filename(prefix: str, extension: str, timestamp: bool = True) -> str:
    """Generate filename with optional timestamp"""
    if timestamp:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{timestamp_str}.{extension}"
    return f"{prefix}.{extension}"


def save_json(data: Any, filepath: str, indent: int = 2) -> bool:
    """Save data to JSON file"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON to {filepath}: {e}")
        return False


def load_json(filepath: str) -> Optional[Any]:
    """Load data from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON from {filepath}: {e}")
        return None


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator for retrying functions on failure"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logging.error(f"All {max_retries + 1} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator


def validate_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def extract_domain(url: str) -> str:
    """Extract domain from URL"""
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_progress_stats(completed: int, total: int, start_time: float) -> Dict[str, Any]:
    """Calculate progress statistics"""
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    if completed == 0:
        return {
            'progress_percent': 0.0,
            'elapsed_time': elapsed_time,
            'estimated_total_time': 0.0,
            'eta': 0.0,
            'rate': 0.0
        }
    
    progress_percent = (completed / total) * 100 if total > 0 else 0.0
    rate = completed / elapsed_time if elapsed_time > 0 else 0.0
    estimated_total_time = total / rate if rate > 0 else 0.0
    eta = estimated_total_time - elapsed_time if estimated_total_time > elapsed_time else 0.0
    
    return {
        'progress_percent': round(progress_percent, 2),
        'elapsed_time': elapsed_time,
        'estimated_total_time': estimated_total_time,
        'eta': eta,
        'rate': round(rate, 2)
    }


def clean_filename(filename: str) -> str:
    """Clean filename removing invalid characters"""
    # Remove invalid characters for filenames
    invalid_chars = r'<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove extra spaces and dots
    filename = re.sub(r'\.+', '.', filename)
    filename = re.sub(r'\s+', ' ', filename).strip()
    
    return filename


def is_valid_image_url(url: str) -> bool:
    """Check if URL points to a valid image"""
    if not validate_url(url):
        return False
    
    # Check for image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp'}
    path = urlparse(url).path.lower()
    
    return any(path.endswith(ext) for ext in image_extensions)


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    try:
        if isinstance(value, str):
            # Remove non-numeric characters except minus
            value = re.sub(r'[^\d-]', '', value)
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        if isinstance(value, str):
            # Remove non-numeric characters except decimal point and minus
            value = re.sub(r'[^\d.-]', '', value)
        return float(value)
    except (ValueError, TypeError):
        return default
