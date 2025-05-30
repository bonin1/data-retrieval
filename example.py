"""
Simple example script for using the Gjirafa50.com scraper
"""
import asyncio
import json
from gjirafa_scraper import GjirafaScraper


async def simple_scraping_example():
    """Simple example of how to use the scraper"""
    print("🚀 Starting Gjirafa50.com scraper...")
    
    # Create scraper with custom configuration
    config = {
        'max_workers': 5,           # Reduce workers for gentler scraping
        'delay_range': (2, 4),      # Wait 2-4 seconds between requests
        'max_pages_per_category': 5, # Limit to first 5 pages per category
        'timeout': 30,              # 30 second timeout
    }
    
    try:
        async with GjirafaScraper(config=config) as scraper:
            # Option 1: Scrape specific categories
            categories = ['telefona-dhe-aksesore', 'kompjuter-dhe-laptop']
            print(f"📋 Scraping categories: {categories}")
            
            # Run the scraper
            products = await scraper.scrape_all_products(categories=categories)
            
            print(f"✅ Successfully scraped {len(products)} products!")
            
            # Export to different formats
            exported_files = scraper.export_products(
                products, 
                formats=['json', 'csv'],
                filename_prefix='gjirafa_sample'
            )
            
            print("📄 Exported files:")
            for format_type, filepath in exported_files.items():
                print(f"   {format_type.upper()}: {filepath}")
            
            # Show some sample products
            if products:
                print("\n📦 Sample products:")
                for i, product in enumerate(products[:3]):
                    print(f"{i+1}. {product.name}")
                    print(f"   Price: {product.price_info.current_price if product.price_info else 'N/A'} EUR")
                    print(f"   URL: {product.url}")
                    print()
            
            return products
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return []


async def category_discovery_example():
    """Example of discovering available categories"""
    print("🔍 Discovering available categories...")
    
    try:
        async with GjirafaScraper() as scraper:
            categories = await scraper.discover_categories()
            
            print(f"Found {len(categories)} categories:")
            for i, category in enumerate(categories[:10], 1):  # Show first 10
                print(f"{i:2d}. {category}")
            
            if len(categories) > 10:
                print(f"... and {len(categories) - 10} more")
            
            return categories
            
    except Exception as e:
        print(f"❌ Error discovering categories: {e}")
        return []


async def single_category_example():
    """Example of scraping a single category in detail"""
    print("🎯 Scraping single category in detail...")
    
    config = {
        'max_workers': 3,
        'delay_range': (1, 2),
        'max_pages_per_category': 2,
    }
    
    try:
        async with GjirafaScraper(config=config) as scraper:
            # Scrape just electronics category
            products = await scraper.scrape_all_products(categories=['telefona-dhe-aksesore'])
            
            if products:
                # Analyze the data
                print(f"\n📊 Analysis of {len(products)} products:")
                
                # Price analysis
                prices = [p.price_info.current_price for p in products if p.price_info]
                if prices:
                    print(f"💰 Price range: {min(prices):.2f} - {max(prices):.2f} EUR")
                    print(f"💰 Average price: {sum(prices)/len(prices):.2f} EUR")
                
                # Brand analysis
                brands = {}
                for product in products:
                    if product.brand:
                        brands[product.brand] = brands.get(product.brand, 0) + 1
                
                if brands:
                    top_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)[:5]
                    print("🏭 Top brands:")
                    for brand, count in top_brands:
                        print(f"   {brand}: {count} products")
                
                # Export detailed data
                exported_files = scraper.export_products(
                    products,
                    formats=['json', 'excel'],
                    filename_prefix='electronics_detailed'
                )
                
                print(f"\n📁 Detailed export:")
                for format_type, filepath in exported_files.items():
                    print(f"   {filepath}")
            
            return products
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def main():
    """Main function to run examples"""
    print("=" * 60)
    print("🛒 GJIRAFA50.COM SCRAPER EXAMPLES")
    print("=" * 60)
    
    print("\nChoose an example to run:")
    print("1. Simple scraping (2 categories, quick)")
    print("2. Discover available categories")
    print("3. Single category detailed analysis")
    print("4. Run all examples")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                asyncio.run(simple_scraping_example())
                break
            elif choice == '2':
                asyncio.run(category_discovery_example())
                break
            elif choice == '3':
                asyncio.run(single_category_example())
                break
            elif choice == '4':
                print("\n" + "="*40)
                print("RUNNING ALL EXAMPLES")
                print("="*40)
                
                print("\n1️⃣ CATEGORY DISCOVERY")
                asyncio.run(category_discovery_example())
                
                print("\n2️⃣ SIMPLE SCRAPING")
                asyncio.run(simple_scraping_example())
                
                print("\n3️⃣ DETAILED ANALYSIS")
                asyncio.run(single_category_example())
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    print("\n✨ Done! Check the 'output' folder for exported files.")


if __name__ == "__main__":
    main()
