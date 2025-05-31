# 🛒 Gjirafa50.com Advanced Web Scraper

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg" alt="Status">
</div>

A powerful, intelligent web scraper designed specifically for extracting product data from **Gjirafa50.com**, Albania's leading e-commerce platform. This tool features advanced pagination handling, GUI interface, anti-bot protection, and comprehensive data export capabilities.

## 🌟 Key Features

### 🚀 **Smart Scraping Technology**
- **🧠 Intelligent Pagination**: Automatically handles AJAX "Load More" buttons with Albanian text support
- **🎯 Precise Product Targeting**: Stop scraping exactly when you reach your desired product count
- **🔄 Adaptive Methods**: Seamlessly switches between CloudScraper and Selenium based on site behavior
- **🛡️ Anti-Bot Protection**: Built-in evasion techniques with user agent rotation and request throttling

### 🖥️ **Multiple Interfaces**
- **🎮 GUI Application**: User-friendly graphical interface for non-technical users
- **💻 Command Line Interface**: Powerful CLI with extensive options for automation
- **🐍 Python API**: Full programmatic control for developers and integration

### 📊 **Comprehensive Data Extraction**
- **Product Information**: Titles, descriptions, prices, brands, categories
- **Visual Content**: High-quality images with duplicate removal
- **Technical Specs**: Detailed product specifications and attributes
- **Social Proof**: Ratings, review counts, and availability status
- **Metadata**: URLs, product IDs, timestamps, and quality metrics

### 💾 **Flexible Export Options**
- **Multiple Formats**: JSON, CSV, Excel with customizable schemas
- **Summary Reports**: Automatic analytics and data quality assessments
- **Progress Saving**: Periodic saves prevent data loss during long operations
- **Data Validation**: Comprehensive cleaning and validation of extracted data

## 📋 System Requirements

### **Minimum Requirements**
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: Version 3.8 or higher
- **RAM**: 2GB minimum (4GB+ recommended for large scraping operations)
- **Storage**: 500MB free space for installation and data
- **Internet**: Stable broadband connection

### **Recommended Setup**
- **Python**: 3.9+ for optimal performance
- **RAM**: 8GB+ for processing large datasets
- **Storage**: 2GB+ for extensive scraping projects
- **Browser**: Latest Google Chrome (automatically managed)

## 🚀 Quick Start Guide

### **Step 1: Installation**

```bash
# Clone the repository
git clone https://github.com/bonin1/gjirafa-scraper.git
cd gjirafa-scraper

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: First Scrape**

#### **Option A: GUI Interface (Easiest)**
```bash
python gui_scraper.py
```
1. Select a category from the dropdown
2. Enter number of products (e.g., 50)
3. Choose export format (JSON/CSV/Excel)
4. Click "Start Scraping"
5. Watch real-time progress

#### **Option B: Command Line**
```bash
# Scrape 50 laptops and export to JSON
python cli.py --max-products 50 --formats json

# Interactive setup with guided options
python interactive_cli.py
```

#### **Option C: Python API**
```python
from gjirafa_scraper import GjirafaScraper

# Initialize scraper
scraper = GjirafaScraper()

# Scrape 100 products automatically
products = scraper.scrape_products(max_products=100)

# Export results
scraper.export_data(formats=['json', 'csv'], filename_prefix='my_scrape')

# Clean up
scraper.close()

print(f"Successfully scraped {len(products)} products!")
```

## 📖 Detailed Usage Guide

### 🖥️ **GUI Interface (Recommended for Beginners)**

The graphical interface provides the easiest way to use the scraper:

```bash
python gui_scraper.py
```

**GUI Features:**
- **📂 Category Selection**: Choose from auto-discovered categories
- **🎯 Product Limits**: Set exact number of products to scrape
- **📊 Real-time Progress**: Live progress bars and status updates
- **📁 Export Options**: Multiple format selection (JSON/CSV/Excel)
- **🔍 Results Preview**: Preview scraped data before saving
- **❌ Error Handling**: Clear error messages and troubleshooting tips

**Step-by-Step GUI Workflow:**
1. **Launch**: Run `python gui_scraper.py`
2. **Select Category**: Choose target category (e.g., "Kompjuter-Laptop-Monitor")
3. **Set Limit**: Enter desired product count (recommended: 50-500)
4. **Choose Format**: Select JSON for full data, CSV for spreadsheet analysis
5. **Start Scraping**: Click the big "Start Scraping" button
6. **Monitor Progress**: Watch the progress bar and log messages
7. **Review Results**: Check the summary statistics
8. **Access Files**: Files are saved in `scraped_data/` folder

### 💻 **Command Line Interface (For Power Users)**

The CLI provides extensive options for automation and scripting:

#### **Basic Commands**
```bash
# Scrape specific number of products
python cli.py --max-products 100 --formats json csv

# Target specific category
python cli.py --category "kompjuter-laptop-monitor" --max-products 200

# Use custom output directory
python cli.py --max-products 50 --output-dir "./my_results"
```

#### **Advanced Options**
```bash
# Enable Selenium for all requests (more reliable, slower)
python cli.py --use-selenium --max-products 50 --headless

# Custom delays and timeouts
python cli.py --delay 2.0 --timeout 60 --max-retries 5

# Verbose logging for debugging
python cli.py --verbose --max-products 10 --log-level DEBUG

# Batch processing with progress saves
python cli.py --max-products 1000 --save-interval 100 --formats json csv excel
```

#### **Complete CLI Options**
```bash
python cli.py [OPTIONS]

Options:
  --max-products INT     Maximum number of products to scrape
  --category TEXT        Specific category to scrape
  --urls TEXT            Comma-separated list of specific URLs
  --formats TEXT         Export formats: json,csv,excel
  --output-dir PATH      Output directory for files
  --filename-prefix TEXT Prefix for output files
  --use-selenium         Force Selenium for all requests
  --headless             Run browser in headless mode
  --delay FLOAT          Delay between requests (seconds)
  --timeout INT          Request timeout (seconds)
  --max-retries INT      Maximum retry attempts
  --save-interval INT    Save progress every N products
  --verbose              Enable verbose logging
  --log-level TEXT       Logging level (DEBUG, INFO, WARNING, ERROR)
  --help                 Show help message
```

### 🐍 **Python API (For Developers)**

The Python API provides full programmatic control:

#### **Basic API Usage**
```python
from gjirafa_scraper import GjirafaScraper
from config import ScraperConfig

# Initialize with default settings
scraper = GjirafaScraper()

# Method 1: Auto-discovery (easiest)
products = scraper.scrape_products(max_products=100)

# Method 2: Target specific category
category_url = "https://gjirafa50.com/kategoria/kompjuter-laptop-monitor"
product_urls = scraper.discover_product_urls(category_url, max_products=50)
products = scraper.scrape_products(urls=product_urls)

# Export results
exported_files = scraper.export_data(
    formats=['json', 'csv'],
    filename_prefix='my_scrape'
)

print(f"Scraped {len(products)} products")
print(f"Files: {exported_files}")

# Always clean up
scraper.close()
```

#### **Advanced Configuration**
```python
from gjirafa_scraper import GjirafaScraper
from config import ScraperConfig

# Custom configuration
config = ScraperConfig()
config.REQUEST_DELAY = 2.0           # 2 seconds between requests
config.MAX_RETRIES = 5               # Retry failed requests 5 times
config.HEADLESS = False              # Show browser window for debugging
config.TIMEOUT = 60                  # 60 second timeout
config.OUTPUT_DIR = "./my_data"      # Custom output directory
config.MAX_PRODUCTS_PER_CATEGORY = 500  # Default limit per category

# Initialize with custom config
scraper = GjirafaScraper(config)

# Discover all available categories
categories = scraper.discover_category_urls()
print(f"Found {len(categories)} categories:")
for cat in categories[:10]:  # Show first 10
    print(f"  - {cat}")

# Scrape specific category with precise control
laptop_category = "https://gjirafa50.com/kategoria/kompjuter-laptop-monitor"
laptop_urls = scraper.discover_product_urls(
    category_url=laptop_category,
    max_products=200,    # Stop at exactly 200 products
    max_pages=20         # Maximum "Load More" clicks as fallback
)

print(f"Discovered {len(laptop_urls)} laptop URLs")

# Extract detailed product data with progress tracking
detailed_products = []
for i, url in enumerate(laptop_urls):
    print(f"Processing product {i+1}/{len(laptop_urls)}")
    
    product_data = scraper.extract_product_data(url)
    if product_data:
        detailed_products.append(product_data)
    
    # Save progress every 50 products
    if (i + 1) % 50 == 0:
        scraper.export_data(
            data=detailed_products,
            formats=['json'],
            filename_prefix=f'progress_batch_{i//50 + 1}'
        )

# Final export with all formats
final_export = scraper.export_data(
    data=detailed_products,
    formats=['json', 'csv', 'excel'],
    filename_prefix='laptops_complete'
)

print(f"Final export: {final_export}")
scraper.close()
```

### 🔄 **Advanced Pagination System**

One of the most powerful features is the intelligent pagination handling:

#### **How It Works**
1. **Initial Load**: Scraper loads category page and extracts visible products
2. **Smart Detection**: Automatically detects "Load More" buttons using multiple strategies:
   - Albanian text: "SHFAQ MË SHUMË PRODUKTE"
   - CSS selectors: `.load-more-products-btn`, `[data-page-infinite]`
   - JavaScript functions: `loadProductsAjax()`
3. **Efficient Loading**: Clicks "Load More" only until target product count is reached
4. **Fallback System**: Falls back to traditional pagination (`?page=1`, `/page/1`) if needed

#### **Pagination Configuration**
```python
# Fine-tune pagination behavior
product_urls = scraper.discover_product_urls(
    category_url="https://gjirafa50.com/kategoria/teknologji",
    max_products=100,     # Stop at exactly 100 products (primary control)
    max_pages=10          # Maximum "Load More" clicks (safety limit)
)

# The scraper will:
# 1. Load initial page (gets ~24 products)
# 2. Click "Load More" 4-5 times to reach 100 products
# 3. Stop immediately when target is reached
# 4. NOT click "Load More" unnecessarily
```

#### **Benefits of Smart Pagination**
- **⚡ Efficiency**: No wasted requests - stops when target is reached
- **🚀 Speed**: Faster than loading all pages then filtering
- **🔄 Reliability**: Multiple fallback mechanisms handle different page layouts
- **💚 Resource-Friendly**: Reduces server load and memory usage
- **🎯 Precision**: Gets exactly the number of products you want