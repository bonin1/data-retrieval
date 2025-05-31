import sys
import os
import json
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gjirafa_scraper import GjirafaScraper
from config import ScraperConfig

def display_categories(categories: List[str]) -> None:
    """Display categories in a nice format"""
    print("\n" + "="*60)
    print("📂 AVAILABLE CATEGORIES")
    print("="*60)
    
    for i, category in enumerate(categories, 1):
        category_name = category.split('/')[-1].replace('-', ' ').title()
        if not category_name:
            category_name = category.split('/')[-2].replace('-', ' ').title()
        
        print(f"{i:3d}. {category_name}")
        print(f"     URL: {category}")
        if i % 5 == 0: 
            print()
    
    print("="*60)

def get_user_choice(max_choice: int, prompt: str = "Enter your choice") -> int:
    while True:
        try:
            choice = int(input(f"{prompt} (1-{max_choice}): "))
            if 1 <= choice <= max_choice:
                return choice - 1
            else:
                print(f"❌ Please enter a number between 1 and {max_choice}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def get_product_count() -> int:
    while True:
        try:
            count = int(input("🔢 How many products do you want to scrape? (1-1000): "))
            if 1 <= count <= 1000:
                return count
            else:
                print("❌ Please enter a number between 1 and 1000")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def get_export_formats() -> List[str]:
    print("\n📤 Export formats:")
    print("1. JSON only")
    print("2. CSV only") 
    print("3. Both JSON and CSV")
    
    while True:
        try:
            choice = int(input("Choose export format (1-3): "))
            if choice == 1:
                return ['json']
            elif choice == 2:
                return ['csv']
            elif choice == 3:
                return ['json', 'csv']
            else:
                print("❌ Please enter 1, 2, or 3")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def main():
    print("🚀 Welcome to Gjirafa50.com Interactive Scraper!")
    print("=" * 50)
    
    print("\n🔧 Initializing scraper...")
    config = ScraperConfig()
    
    headless_choice = input("🖥️  Run in headless mode? (faster, no browser window) [Y/n]: ").lower()
    if headless_choice in ['', 'y', 'yes']:
        config.HEADLESS = True
        print("✅ Running in headless mode")
    else:
        config.HEADLESS = False
        print("✅ Running with visible browser")
    
    scraper = GjirafaScraper(config)
    
    try:
        print("\n🔍 Discovering available categories...")
        categories = scraper.discover_category_urls()
        
        if not categories:
            print("❌ No categories found! Please check your internet connection.")
            return 1
        
        print(f"✅ Found {len(categories)} categories")
        
        display_categories(categories)
        
        chosen_category_idx = get_user_choice(
            len(categories), 
            "🎯 Select a category"
        )
        
        chosen_category = categories[chosen_category_idx]
        category_name = chosen_category.split('/')[-1].replace('-', ' ').title()
        if not category_name:
            category_name = chosen_category.split('/')[-2].replace('-', ' ').title()
        
        print(f"\n✅ Selected category: {category_name}")
        print(f"🔗 URL: {chosen_category}")
        
        product_count = get_product_count()
        print(f"✅ Will scrape {product_count} products")
        
        export_formats = get_export_formats()
        print(f"✅ Will export as: {', '.join(export_formats).upper()}")
        
        print(f"\n📋 SCRAPING SUMMARY:")
        print(f"   Category: {category_name}")
        print(f"   Products: {product_count}")
        print(f"   Export: {', '.join(export_formats).upper()}")
        
        confirm = input("\n🚀 Start scraping? [Y/n]: ").lower()
        if confirm not in ['', 'y', 'yes']:
            print("❌ Scraping cancelled")
            return 0
        
        print(f"\n🔍 Discovering product URLs in {category_name}...")
        product_urls = scraper.discover_product_urls(chosen_category)
        
        if not product_urls:
            print(f"❌ No products found in category: {category_name}")
            return 1
        
        print(f"✅ Found {len(product_urls)} product URLs")
        
        if len(product_urls) > product_count:
            product_urls = product_urls[:product_count]
            print(f"📝 Limited to first {product_count} products")
        
        print(f"\n⏳ Scraping {len(product_urls)} products...")
        scraped_count = 0
        
        for i, url in enumerate(product_urls, 1):
            print(f"\n📦 Scraping product {i}/{len(product_urls)}: {url}")
            
            product_data = scraper.extract_product_data(url)
            
            if product_data:
                scraper.products.append(product_data)
                scraped_count += 1
                
                print(f"   ✅ Title: {product_data.get('title', 'N/A')}")
                print(f"   💰 Price: {product_data.get('price', 'N/A')}")
                print(f"   🖼️  Images: {len(product_data.get('images', []))}")
                print(f"   📊 Specs: {len(product_data.get('specifications', {}))}")
            else:
                print(f"   ❌ Failed to extract data")
            
            progress = (i / len(product_urls)) * 100
            print(f"   📈 Progress: {progress:.1f}% ({scraped_count} successful)")
        
        if scraper.products:
            print(f"\n📤 Exporting {len(scraper.products)} products...")
            
            filename_prefix = f"{category_name.lower().replace(' ', '_')}_scrape"
            exported_files = scraper.export_data(
                formats=export_formats,
                filename_prefix=filename_prefix
            )
            
            print(f"\n🎉 SCRAPING COMPLETED!")
            print(f"📊 Total products scraped: {len(scraper.products)}")
            print(f"📁 Exported files:")
            
            for format_type, filepath in exported_files.items():
                print(f"   - {format_type.upper()}: {filepath}")
                
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    size_mb = size / (1024 * 1024)
                    print(f"     Size: {size_mb:.2f} MB")
            
            if scraper.products:
                print(f"\n📋 Sample product data:")
                sample = scraper.products[0]
                print(f"   Title: {sample.get('title', 'N/A')}")
                print(f"   Price: {sample.get('price', 'N/A')}")
                print(f"   URL: {sample.get('url', 'N/A')}")
                
        else:
            print("❌ No products were scraped successfully")
            return 1
            
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user")
        if scraper.products:
            print(f"💾 Saving {len(scraper.products)} products scraped so far...")
            exported_files = scraper.export_data(
                formats=['json'],
                filename_prefix='interrupted_scrape'
            )
            print(f"✅ Saved to: {exported_files.get('json', 'unknown')}")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        scraper.close()
        print("\n👋 Thanks for using Gjirafa50 scraper!")

if __name__ == "__main__":
    sys.exit(main())
