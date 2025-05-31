import argparse
import sys
import logging
from pathlib import Path

from gjirafa_scraper import GjirafaScraper
from config import ScraperConfig
from utils import DataExporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scraper.log')
    ]
)
logger = logging.getLogger(__name__)

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Advanced web scraper for Gjirafa50.com e-commerce site',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape 100 products and export to JSON and CSV
  python cli.py --max-products 100 --formats json csv
  
  # Scrape from specific URLs
  python cli.py --urls https://gjirafa50.com/product/123 https://gjirafa50.com/product/456
  
  # Scrape specific categories
  python cli.py --categories teknologji mode --max-products 200
  
  # Use Selenium for all requests (slower but more reliable)
  python cli.py --use-selenium --max-products 50
  
  # Custom output directory and filename
  python cli.py --output-dir ./my_data --filename-prefix custom_scrape
        """
    )
    
    input_group = parser.add_argument_group('Input options')
    input_group.add_argument(
        '--urls', 
        nargs='+', 
        help='Specific product URLs to scrape'
    )
    input_group.add_argument(
        '--categories',
        nargs='+',
        help='Category names to scrape (e.g., teknologji mode)'
    )
    input_group.add_argument(
        '--discover-all',
        action='store_true',
        help='Automatically discover and scrape all products'
    )
    
    scraping_group = parser.add_argument_group('Scraping options')
    scraping_group.add_argument(
        '--max-products',
        type=int,
        help='Maximum number of products to scrape'
    )
    scraping_group.add_argument(
        '--max-pages',
        type=int,
        default=10,
        help='Maximum pages to scrape per category (default: 10)'
    )
    scraping_group.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )
    scraping_group.add_argument(
        '--use-selenium',
        action='store_true',
        help='Use Selenium for all requests (slower but more reliable)'
    )
    scraping_group.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run browser in headless mode (default: True)'
    )
    
    output_group = parser.add_argument_group('Output options')
    output_group.add_argument(
        '--output-dir',
        default='scraped_data',
        help='Output directory (default: scraped_data)'
    )
    output_group.add_argument(
        '--filename-prefix',
        default='gjirafa_products',
        help='Output filename prefix (default: gjirafa_products)'
    )
    output_group.add_argument(
        '--formats',
        nargs='+',
        choices=['json', 'csv', 'excel'],
        default=['json', 'csv'],
        help='Export formats (default: json csv)'
    )
    output_group.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='Save progress every N products (default: 100)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--base-url',
        default='https://gjirafa50.com',
        help='Base URL to scrape (default: https://gjirafa50.com)'
    )
    
    return parser

def setup_config(args):
    config = ScraperConfig()
    
    if args.delay:
        config.REQUEST_DELAY = args.delay
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.headless is not None:
        config.HEADLESS = args.headless
        
    return config

def get_urls_from_categories(scraper, categories, max_pages):
    urls = []
    
    for category in categories:
        category_patterns = [
            f"{scraper.base_url}/kategoria/{category}",
            f"{scraper.base_url}/category/{category}",
            f"{scraper.base_url}/c/{category}",
            f"{scraper.base_url}/{category}"
        ]
        
        for category_url in category_patterns:
            try:
                category_products = scraper.discover_product_urls(category_url, max_pages)
                if category_products:
                    urls.extend(category_products)
                    logger.info(f"Found {len(category_products)} products in category: {category}")
                    break
            except Exception as e:
                logger.error(f"Error scraping category {category_url}: {e}")
                continue
    
    return list(set(urls))

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    config = setup_config(args)
    
    logger.info("Starting Gjirafa50.com scraper...")
    logger.info(f"Configuration: {config}")
    
    try:
        with GjirafaScraper(config) as scraper:
            scraper.base_url = args.base_url
            
            urls_to_scrape = []
            
            if args.urls:
                urls_to_scrape = args.urls
                logger.info(f"Using provided URLs: {len(urls_to_scrape)} products")
                
            elif args.categories:
                urls_to_scrape = get_urls_from_categories(scraper, args.categories, args.max_pages)
                logger.info(f"Discovered {len(urls_to_scrape)} products from categories")
                
            elif args.discover_all:
                logger.info("Discovering all products automatically...")
                urls_to_scrape = None 
                
            else:
                logger.info("No specific input provided, discovering products automatically...")
                urls_to_scrape = None
            
            products = scraper.scrape_products(
                urls=urls_to_scrape,
                max_products=args.max_products,
                save_interval=args.save_interval
            )
            
            if not products:
                logger.error("No products were scraped successfully")
                return 1
            
            logger.info(f"Exporting {len(products)} products...")
            exported_files = scraper.export_data(
                data=products,
                formats=args.formats,
                filename_prefix=args.filename_prefix
            )
            
            print(f"\n✅ Scraping completed successfully!")
            print(f"📊 Total products scraped: {len(products)}")
            print(f"📁 Exported files:")
            
            for format_type, filepath in exported_files.items():
                print(f"  - {format_type.upper()}: {filepath}")
            
            if products:
                sample_product = products[0]
                print(f"\n📋 Sample product data:")
                print(f"  - Title: {sample_product.get('title', 'N/A')}")
                print(f"  - Price: {sample_product.get('price', 'N/A')}")
                print(f"  - URL: {sample_product.get('url', 'N/A')}")
            
            return 0
            
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
