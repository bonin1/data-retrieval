"""
Data export functionality for different formats
"""
import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
from data_models import Product, ScrapingSession
from utils import create_output_directory, generate_filename, clean_filename


class DataExporter:
    """Handle data export to various formats"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = create_output_directory(output_dir)
    
    def export_to_json(
        self, 
        products: List[Product], 
        filename: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """Export products to JSON format"""
        if not filename:
            filename = generate_filename("products", "json")
        
        filepath = os.path.join(self.output_dir, clean_filename(filename))
        
        # Convert products to dictionaries
        products_data = [product.to_dict() for product in products]
        
        # Prepare export data
        export_data = {
            "products": products_data,
            "total_count": len(products),
            "exported_at": datetime.now().isoformat(),
        }
        
        if include_metadata:
            export_data["metadata"] = {
                "format_version": "1.0",
                "source": "gjirafa50.com",
                "exporter": "Gjirafa50 Advanced Scraper",
                "fields_included": list(products_data[0].keys()) if products_data else [],
                "export_settings": {
                    "include_images": any("image_urls" in p for p in products_data),
                    "include_reviews": any("reviews_count" in p for p in products_data),
                    "include_specifications": any("specifications" in p for p in products_data),
                }
            }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        return filepath
    
    def export_to_csv(
        self, 
        products: List[Product], 
        filename: Optional[str] = None,
        include_images: bool = True,
        max_images: int = 5
    ) -> str:
        """Export products to CSV format"""
        if not filename:
            filename = generate_filename("products", "csv")
        
        filepath = os.path.join(self.output_dir, clean_filename(filename))
        
        if not products:
            # Create empty CSV with headers
            headers = ["id", "name", "url", "price", "currency", "availability"]
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            return filepath
        
        # Prepare data rows
        rows = []
        headers = set()
        
        for product in products:
            row = product.to_dict()
            
            # Handle complex fields
            row["categories"] = "; ".join(row.get("categories", []))
            row["tags"] = "; ".join(row.get("tags", []))
            
            # Handle images
            if include_images and "image_urls" in row:
                image_urls = row["image_urls"][:max_images]
                for i, url in enumerate(image_urls):
                    row[f"image_url_{i+1}"] = url
                # Fill remaining image columns
                for i in range(len(image_urls), max_images):
                    row[f"image_url_{i+1}"] = ""
                del row["image_urls"]
            
            # Handle specifications
            if "specifications" in row:
                specs = row["specifications"]
                if isinstance(specs, dict):
                    for category, spec_dict in specs.items():
                        if isinstance(spec_dict, dict):
                            for key, value in spec_dict.items():
                                row[f"spec_{category}_{key}"] = value
                del row["specifications"]
            
            # Flatten any remaining nested structures
            flattened_row = self._flatten_dict(row)
            rows.append(flattened_row)
            headers.update(flattened_row.keys())
        
        # Sort headers for consistent column order
        ordered_headers = self._get_ordered_headers(headers)
        
        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        
        return filepath
    
    def export_to_excel(
        self, 
        products: List[Product], 
        filename: Optional[str] = None,
        include_summary: bool = True
    ) -> str:
        """Export products to Excel format with multiple sheets"""
        if not filename:
            filename = generate_filename("products", "xlsx")
        
        filepath = os.path.join(self.output_dir, clean_filename(filename))
        
        # Convert to DataFrame
        products_data = [product.to_dict() for product in products]
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main products sheet
            if products_data:
                df = pd.json_normalize(products_data)
                df.to_excel(writer, sheet_name='Products', index=False)
                
                # Categories sheet
                categories_data = []
                for product in products:
                    for category in product.categories:
                        categories_data.append({
                            'product_id': product.id,
                            'product_name': product.name,
                            'category': category
                        })
                
                if categories_data:
                    categories_df = pd.DataFrame(categories_data)
                    categories_df.to_excel(writer, sheet_name='Categories', index=False)
                
                # Images sheet
                images_data = []
                for product in products:
                    for i, image in enumerate(product.images):
                        images_data.append({
                            'product_id': product.id,
                            'product_name': product.name,
                            'image_index': i + 1,
                            'image_url': image.url,
                            'alt_text': image.alt_text,
                            'is_primary': image.is_primary
                        })
                
                if images_data:
                    images_df = pd.DataFrame(images_data)
                    images_df.to_excel(writer, sheet_name='Images', index=False)
                
                # Summary sheet
                if include_summary:
                    summary_data = self._create_summary_data(products)
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            else:
                # Empty sheet with headers
                empty_df = pd.DataFrame(columns=['id', 'name', 'url', 'price'])
                empty_df.to_excel(writer, sheet_name='Products', index=False)
        
        return filepath
    
    def export_session_report(
        self, 
        session: ScrapingSession, 
        products: List[Product],
        filename: Optional[str] = None
    ) -> str:
        """Export scraping session report"""
        if not filename:
            filename = generate_filename("session_report", "json")
        
        filepath = os.path.join(self.output_dir, clean_filename(filename))
        
        # Prepare report data
        report_data = {
            "session_info": {
                "session_id": session.session_id,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "duration_seconds": session.get_duration(),
                "success_rate": session.get_success_rate(),
                "total_products": session.total_products,
                "successful_products": session.successful_products,
                "failed_products": session.failed_products,
                "categories_scraped": session.categories_scraped,
                "errors": session.errors
            },
            "products_summary": {
                "total_exported": len(products),
                "price_range": self._get_price_range(products),
                "categories": self._get_category_distribution(products),
                "availability_status": self._get_availability_distribution(products),
                "top_brands": self._get_top_brands(products),
            },
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "format": "JSON Report",
                "file_path": filepath
            }
        }
        
        # Save report
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
        
        return filepath
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                if v and isinstance(v[0], dict):
                    # Skip complex nested lists for CSV
                    items.append((new_key, str(v)))
                else:
                    items.append((new_key, "; ".join(map(str, v))))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _get_ordered_headers(self, headers: set) -> List[str]:
        """Get ordered headers for CSV export"""
        priority_headers = [
            'id', 'name', 'url', 'description', 'price', 'old_price', 
            'currency', 'discount_percentage', 'availability', 'brand', 
            'model', 'categories', 'tags', 'rating', 'reviews_count'
        ]
        
        ordered = []
        for header in priority_headers:
            if header in headers:
                ordered.append(header)
                headers.remove(header)
        
        # Add remaining headers alphabetically
        ordered.extend(sorted(headers))
        
        return ordered
    
    def _create_summary_data(self, products: List[Product]) -> List[Dict[str, Any]]:
        """Create summary statistics"""
        if not products:
            return []
        
        # Price statistics
        prices = [p.price_info.current_price for p in products if p.price_info]
        
        # Category distribution
        categories = {}
        for product in products:
            for category in product.categories:
                categories[category] = categories.get(category, 0) + 1
        
        # Brand distribution
        brands = {}
        for product in products:
            if product.brand:
                brands[product.brand] = brands.get(product.brand, 0) + 1
        
        summary = [
            {"Metric": "Total Products", "Value": len(products)},
            {"Metric": "Products with Prices", "Value": len(prices)},
            {"Metric": "Average Price", "Value": sum(prices) / len(prices) if prices else 0},
            {"Metric": "Min Price", "Value": min(prices) if prices else 0},
            {"Metric": "Max Price", "Value": max(prices) if prices else 0},
            {"Metric": "Unique Categories", "Value": len(categories)},
            {"Metric": "Unique Brands", "Value": len(brands)},
            {"Metric": "Products with Images", "Value": len([p for p in products if p.images])},
            {"Metric": "Products with Reviews", "Value": len([p for p in products if p.reviews_count])},
        ]
        
        return summary
    
    def _get_price_range(self, products: List[Product]) -> Dict[str, Any]:
        """Get price range statistics"""
        prices = [p.price_info.current_price for p in products if p.price_info]
        if not prices:
            return {"min": 0, "max": 0, "average": 0, "count": 0}
        
        return {
            "min": min(prices),
            "max": max(prices),
            "average": round(sum(prices) / len(prices), 2),
            "count": len(prices)
        }
    
    def _get_category_distribution(self, products: List[Product]) -> Dict[str, int]:
        """Get category distribution"""
        categories = {}
        for product in products:
            for category in product.categories:
                categories[category] = categories.get(category, 0) + 1
        return dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))
    
    def _get_availability_distribution(self, products: List[Product]) -> Dict[str, int]:
        """Get availability status distribution"""
        availability = {}
        for product in products:
            status = product.availability
            availability[status] = availability.get(status, 0) + 1
        return availability
    
    def _get_top_brands(self, products: List[Product], limit: int = 10) -> Dict[str, int]:
        """Get top brands by product count"""
        brands = {}
        for product in products:
            if product.brand:
                brands[product.brand] = brands.get(product.brand, 0) + 1
        
        sorted_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_brands[:limit])
