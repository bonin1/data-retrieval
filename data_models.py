"""
Data models and validation for Gjirafa50.com products
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import re
from urllib.parse import urljoin, urlparse


@dataclass
class ProductImage:
    """Product image data model"""
    url: str
    alt_text: str = ""
    is_primary: bool = False
    
    def __post_init__(self):
        if not self.url.startswith('http'):
            raise ValueError(f"Invalid image URL: {self.url}")


@dataclass
class ProductReview:
    """Product review data model"""
    rating: float
    title: str = ""
    content: str = ""
    author: str = ""
    date: Optional[datetime] = None
    verified_purchase: bool = False


@dataclass
class ProductSpecification:
    """Product specification data model"""
    category: str
    specifications: Dict[str, str] = field(default_factory=dict)


@dataclass
class SellerInfo:
    """Seller information data model"""
    name: str
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    url: Optional[str] = None
    location: Optional[str] = None


@dataclass
class PriceInfo:
    """Price information data model"""
    current_price: float
    currency: str = "EUR"
    old_price: Optional[float] = None
    discount_percentage: Optional[float] = None
    
    def __post_init__(self):
        if self.old_price and self.current_price > 0:
            self.discount_percentage = round(
                ((self.old_price - self.current_price) / self.old_price) * 100, 2
            )


@dataclass
class Product:
    """Main product data model"""
    # Basic information
    id: str
    name: str
    url: str
    description: str = ""
    
    # Price and availability
    price_info: Optional[PriceInfo] = None
    availability: str = "unknown"
    stock_count: Optional[int] = None
    
    # Categories and classification
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    brand: str = ""
    model: str = ""
    
    # Media
    images: List[ProductImage] = field(default_factory=list)
    
    # Reviews and ratings
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    reviews: List[ProductReview] = field(default_factory=list)
    
    # Technical details
    specifications: List[ProductSpecification] = field(default_factory=list)
    
    # Seller information
    seller: Optional[SellerInfo] = None
    
    # Metadata
    scraped_at: datetime = field(default_factory=datetime.now)
    last_updated: Optional[datetime] = None
    
    # Additional data
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and clean data after initialization"""
        self._validate_basic_fields()
        self._clean_data()
    
    def _validate_basic_fields(self):
        """Validate required fields"""
        if not self.id:
            raise ValueError("Product ID is required")
        if not self.name or len(self.name.strip()) < 3:
            raise ValueError("Product name must be at least 3 characters")
        if not self.url or not self._is_valid_url(self.url):
            raise ValueError(f"Invalid product URL: {self.url}")
    
    def _clean_data(self):
        """Clean and normalize data"""
        self.name = self.name.strip()
        self.description = self.description.strip()
        self.brand = self.brand.strip()
        self.model = self.model.strip()
        
        # Clean categories and tags
        self.categories = [cat.strip() for cat in self.categories if cat.strip()]
        self.tags = [tag.strip() for tag in self.tags if tag.strip()]
        
        # Normalize availability status
        self.availability = self._normalize_availability(self.availability)
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _normalize_availability(self, availability: str) -> str:
        """Normalize availability status"""
        availability = availability.lower().strip()
        if any(word in availability for word in ['në stok', 'available', 'në dispozicion']):
            return 'in_stock'
        elif any(word in availability for word in ['pa stok', 'out of stock', 'mbaruar']):
            return 'out_of_stock'
        elif any(word in availability for word in ['limited', 'i kufizuar']):
            return 'limited_stock'
        else:
            return 'unknown'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert product to dictionary for export"""
        return {
            'id': self.id,
            'name': self.name,
            'url': self.url,
            'description': self.description,
            'price': self.price_info.current_price if self.price_info else None,
            'old_price': self.price_info.old_price if self.price_info else None,
            'currency': self.price_info.currency if self.price_info else None,
            'discount_percentage': self.price_info.discount_percentage if self.price_info else None,
            'availability': self.availability,
            'stock_count': self.stock_count,
            'categories': self.categories,
            'tags': self.tags,
            'brand': self.brand,
            'model': self.model,
            'rating': self.rating,
            'reviews_count': self.reviews_count,
            'seller_name': self.seller.name if self.seller else None,
            'seller_rating': self.seller.rating if self.seller else None,
            'image_urls': [img.url for img in self.images],
            'specifications': {
                spec.category: spec.specifications 
                for spec in self.specifications
            },
            'scraped_at': self.scraped_at.isoformat(),
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            **self.additional_data
        }
    
    def get_primary_image(self) -> Optional[ProductImage]:
        """Get the primary product image"""
        primary_images = [img for img in self.images if img.is_primary]
        if primary_images:
            return primary_images[0]
        elif self.images:
            return self.images[0]
        return None
    
    def get_average_rating(self) -> Optional[float]:
        """Calculate average rating from reviews"""
        if self.reviews:
            ratings = [review.rating for review in self.reviews if review.rating > 0]
            if ratings:
                return round(sum(ratings) / len(ratings), 2)
        return self.rating


class ProductValidator:
    """Validator class for product data"""
    
    @staticmethod
    def validate_product(product: Product) -> List[str]:
        """Validate a product and return list of errors"""
        errors = []
        
        # Basic validation
        if not product.name or len(product.name) < 3:
            errors.append("Product name is too short")
        
        if not product.id:
            errors.append("Product ID is missing")
        
        if not product.url:
            errors.append("Product URL is missing")
        
        # Price validation
        if product.price_info:
            if product.price_info.current_price < 0:
                errors.append("Price cannot be negative")
            if product.price_info.old_price and product.price_info.old_price < product.price_info.current_price:
                errors.append("Old price cannot be less than current price")
        
        # Rating validation
        if product.rating is not None:
            if not (0 <= product.rating <= 5):
                errors.append("Rating must be between 0 and 5")
        
        # Image validation
        for i, image in enumerate(product.images):
            if not image.url.startswith('http'):
                errors.append(f"Invalid image URL at index {i}")
        
        return errors
    
    @staticmethod
    def is_valid_product(product: Product) -> bool:
        """Check if product is valid"""
        return len(ProductValidator.validate_product(product)) == 0


@dataclass
class ScrapingSession:
    """Session information for scraping run"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_products: int = 0
    successful_products: int = 0
    failed_products: int = 0
    categories_scraped: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_products == 0:
            return 0.0
        return round((self.successful_products / self.total_products) * 100, 2)
    
    def get_duration(self) -> Optional[float]:
        """Get session duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
