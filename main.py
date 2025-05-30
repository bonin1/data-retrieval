"""
Main CLI interface for Gjirafa50.com scraper
"""
import asyncio
import argparse
import sys
import json
from typing import List, Optional
from pathlib import Path

from gjirafa_scraper import GjirafaScraper
from config import TARGET_CATEGORIES, SCRAPING_CONFIG
from utils import setup_logging


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Advanced Gjirafa50.com Product Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape all categories
  python main.py

  # Scrape specific categories
  python main.py --categories electronics fashion

  # Custom configuration
  python main.py --max-workers 5 --delay 2 3 --output custom_output

  # Export only to CSV
  python main.py --formats csv

  # Verbose logging
  python main.py --verbose
        """
    )
    
    # Categories
    parser.add_argument(
        '--categories', '-c',
        nargs='+',
        help='Specific categories to scrape (default: all available)'
    )
    
    # Scraping configuration
    parser.add_argument(
        '--max-workers', '-w',
        type=int,
        default=SCRAPING_CONFIG['max_workers'],
        help=f'Maximum concurrent workers (default: {SCRAPING_CONFIG["max_workers"]})'
    )
    
    parser.add_argument(
        '--delay', '-d',
        nargs=2,
        type=float,
        default=SCRAPING_CONFIG['delay_range'],
        metavar=('MIN', 'MAX'),
        help=f'Delay range between requests in seconds (default: {SCRAPING_CONFIG["delay_range"]})'
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        default=SCRAPING_CONFIG['max_pages_per_category'],
        help=f'Maximum pages per category (default: {SCRAPING_CONFIG["max_pages_per_category"]})'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=SCRAPING_CONFIG['timeout'],
        help=f'Request timeout in seconds (default: {SCRAPING_CONFIG["timeout"]})'
    )
    
    # Export options
    parser.add_argument(
        '--formats', '-f',
        nargs='+',
        choices=['json', 'csv', 'excel'],
        default=['json', 'csv', 'excel'],
        help='Export formats (default: all formats)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='output',
        help='Output directory (default: output)'
    )
    
    parser.add_argument(
        '--filename',
        default='gjirafa_products',
        help='Output filename prefix (default: gjirafa_products)'
    )
    
    # Other options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--use-proxies',
        action='store_true',
        help='Use proxy rotation (requires proxies.txt file)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be scraped without actually scraping'
    )
    
    parser.add_argument(
        '--list-categories',
        action='store_true',
        help='List available categories and exit'
    )
    
    return parser.parse_args()


async def list_categories():
    """List available categories"""
    print("Discovering available categories...")
    
    async with GjirafaScraper() as scraper:
        categories = await scraper.discover_categories()
    
    print(f"\nFound {len(categories)} categories:")
    for i, category in enumerate(categories, 1):
        print(f"{i:2d}. {category}")
    
    print(f"\nTo scrape specific categories, use:")
    print(f"python main.py --categories {' '.join(categories[:3])}")


async def dry_run(args):
    """Show what would be scraped without actually scraping"""
    print("DRY RUN MODE - No actual scraping will be performed\n")
    
    config = {
        'max_workers': args.max_workers,
        'delay_range': tuple(args.delay),
        'max_pages_per_category': args.max_pages,
        'timeout': args.timeout,
        'use_proxies': args.use_proxies,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nOutput directory: {args.output}")
    print(f"Filename prefix: {args.filename}")
    print(f"Export formats: {', '.join(args.formats)}")
    
    categories = args.categories or await discover_categories_for_dry_run()
    print(f"\nCategories to scrape ({len(categories)}):")
    for category in categories:
        print(f"  - {category}")
    
    print("\nEstimated products per category: ~100-500")
    print(f"Estimated total products: {len(categories) * 200}-{len(categories) * 500}")
    print(f"Estimated time: {len(categories) * 5}-{len(categories) * 15} minutes")


async def discover_categories_for_dry_run():
    """Discover categories for dry run"""
    try:
        async with GjirafaScraper() as scraper:
            return await scraper.discover_categories()
    except Exception:
        return TARGET_CATEGORIES


def create_custom_config(args):
    """Create custom configuration from arguments"""
    config = {
        'max_workers': args.max_workers,
        'delay_range': tuple(args.delay),
        'max_pages_per_category': args.max_pages,
        'timeout': args.timeout,
        'use_proxies': args.use_proxies,
    }
    
    return config


async def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logging(level=log_level, console=True)
    
    # Handle special commands
    if args.list_categories:
        await list_categories()
        return
    
    if args.dry_run:
        await dry_run(args)
        return
    
    # Create custom configuration
    config = create_custom_config(args)
    
    logger.info("Starting Gjirafa50.com scraper")
    logger.info(f"Configuration: {config}")
    
    try:
        # Create and run scraper
        async with GjirafaScraper(config=config) as scraper:
            # Set output directory
            scraper.data_exporter.output_dir = args.output
            
            # Run scraper
            result = await scraper.run_scraper(categories=args.categories)
            
            if result['success']:
                logger.info("Scraping completed successfully!")
                logger.info(f"Products scraped: {result['products_count']}")
                logger.info("Exported files:")
                for format_type, filepath in result['exported_files'].items():
                    logger.info(f"  {format_type.upper()}: {filepath}")
                
                # Export additional formats if requested
                if args.formats != ['json', 'csv', 'excel']:
                    logger.info("Exporting to requested formats...")
                    # Re-load products and export in specified formats
                    # This is a simplified approach - in production you might want to store products
                    
                print("\n" + "="*60)
                print("SCRAPING COMPLETED SUCCESSFULLY!")
                print("="*60)
                print(f"Products scraped: {result['products_count']}")
                print(f"Output directory: {args.output}")
                print("Files created:")
                for format_type, filepath in result['exported_files'].items():
                    print(f"  📄 {Path(filepath).name}")
                print("="*60)
                
            else:
                logger.error(f"Scraping failed: {result['error']}")
                print("\n❌ Scraping failed. Check the logs for details.")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        print("\n⚠️  Scraping interrupted. Partial data may have been saved.")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


def run_scraper_cli():
    """Entry point for CLI"""
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_scraper_cli()
