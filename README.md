# Gjirafa50.com Data Retrieval Tool

An advanced web scraper for extracting product data from Gjirafa50.com e-commerce platform.

## Features

- 🚀 **High-Performance Scraping**: Multi-threaded async scraping with rate limiting
- 🛡️ **Anti-Detection**: Rotating user agents, headers, and request patterns
- 📊 **Multiple Export Formats**: JSON, CSV, and Excel outputs
- 🔄 **Robust Error Handling**: Automatic retries and error recovery
- 📈 **Progress Tracking**: Real-time progress bars and logging
- ⚙️ **Configurable**: Extensive configuration options
- 🧹 **Data Cleaning**: Automatic data validation and cleaning

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd data-retrieval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure settings in `config.py`

## Usage

### Basic Usage

```python
from gjirafa_scraper import GjirafaScraper

# Initialize scraper
scraper = GjirafaScraper()

# Scrape all products
products = scraper.scrape_all_products()

# Export to JSON
scraper.export_to_json(products, 'products.json')

# Export to CSV
scraper.export_to_csv(products, 'products.csv')
```

### Advanced Usage

```python
# Custom configuration
config = {
    'max_workers': 10,
    'delay_range': (1, 3),
    'max_retries': 5,
    'timeout': 30
}

scraper = GjirafaScraper(config=config)

# Scrape specific categories
categories = ['electronics', 'fashion', 'home']
products = scraper.scrape_categories(categories)

# Filter and export
filtered_products = scraper.filter_products(products, min_price=10, max_price=1000)
scraper.export_to_excel(filtered_products, 'filtered_products.xlsx')
```

## Project Structure

```
data-retrieval/
├── gjirafa_scraper.py      # Main scraper class
├── config.py               # Configuration settings
├── data_models.py          # Data structures and validation
├── utils.py                # Utility functions
├── exporters.py            # Data export functionality
├── anti_detection.py       # Anti-detection measures
├── main.py                 # CLI interface
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Data Structure

Each product contains:
- ID and URL
- Name and description
- Price information
- Images
- Categories and tags
- Seller information
- Reviews and ratings
- Availability status
- Technical specifications

## Configuration

Edit `config.py` to customize:
- Request delays and timeouts
- Worker thread counts
- Export formats
- Filtering options
- Anti-detection settings

## Legal Notice

This tool is for educational and research purposes. Please respect the website's robots.txt and terms of service. Use responsibly and consider the impact on the target website's servers.

## License

MIT License - see LICENSE file for details.
