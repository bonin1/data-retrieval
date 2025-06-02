# 🛒 E-commerce Data Retrieval & Analytics Platform

A comprehensive Python-based platform for scraping, analyzing, and visualizing e-commerce data with advanced machine learning capabilities and real-time dashboard.

## 🚀 Features

### 📊 Core Capabilities
- **Web Scraping**: Automated data extraction from Gjirafa50.com
- **Real-time Dashboard**: Interactive Streamlit dashboard with live analytics
- **Machine Learning**: Price prediction, demand forecasting, and product recommendations
- **Advanced Analytics**: Statistical analysis, anomaly detection, and market insights
- **Data Pipeline**: End-to-end ETL processing with quality assessment
- **Visualization**: Comprehensive charts, plots, and interactive visualizations

### 🤖 Machine Learning & Analytics
- **Price Prediction Models**: Random Forest, XGBoost, LightGBM, Neural Networks
- **Recommendation Engine**: Content-based and collaborative filtering
- **Customer Segmentation**: K-Means, DBSCAN, Hierarchical clustering
- **Anomaly Detection**: Isolation Forest, One-Class SVM
- **Time Series Forecasting**: Prophet, ARIMA models
- **Hyperparameter Optimization**: Optuna-based automated tuning

### 📈 Visualization & Reporting
- **Interactive Charts**: Plotly, Bokeh, Matplotlib visualizations
- **Market Analysis**: Price distributions, category performance, brand analysis
- **Quality Metrics**: Data completeness, outlier detection
- **Word Clouds**: Product description and title analysis
- **Export Options**: JSON, CSV, Excel, HTML reports

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Chrome/Chromium browser (for Selenium)
- Git

### 1. Clone Repository
```bash
git clone https://github.com/bonin1/data-retrieval.git
cd data-retrieval
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Chrome WebDriver
The system automatically downloads ChromeDriver using `webdriver-manager`, but ensure Chrome is installed:
- [Download Chrome](https://www.google.com/chrome/)

### 4. Verify Installation
```bash
python -c "import streamlit, selenium, pandas, plotly; print('✅ All dependencies installed successfully')"
```

## 🚀 Quick Start

### 1. Interactive Scraping
Launch the interactive CLI scraper:
```bash
python interactive_cli.py
```

Follow the prompts to:
- Select product categories
- Configure scraping parameters
- Choose export formats

### 2. Start Real-time Dashboard
```bash
python restart_dashboard.py
```
Or directly:
```bash
streamlit run real_time_dashboard.py
```

Access dashboard at: http://localhost:8501

### 3. Run Complete Data Pipeline
```python
from data_pipeline import DataPipeline

# Initialize pipeline with your data
pipeline = DataPipeline("path/to/data.json")

# Generate comprehensive report
report = pipeline.generate_comprehensive_report()

# Export results
exported_files = pipeline.export_results(['json', 'csv', 'excel'])
```

## 📋 Usage Guide

### Web Scraping

#### Basic Scraping
```python
from gjirafa_scraper import GjirafaScraper
from config import ScraperConfig

# Initialize scraper
config = ScraperConfig()
scraper = GjirafaScraper(config)

# Discover categories
categories = scraper.discover_category_urls()

# Scrape products from a category
product_urls = scraper.discover_product_urls(categories[0], max_products=100)

# Extract product data
for url in product_urls:
    product_data = scraper.extract_product_data(url)
    if product_data:
        scraper.products.append(product_data)

# Export data
scraper.export_data(formats=['json', 'csv'])
```

#### GUI Scraping
```python
from gui_scraper import launch_gui
launch_gui()
```

### Advanced Analytics

#### Statistical Analysis
```python
from advanced_analytics import AdvancedAnalytics
import pandas as pd

# Load data
df = pd.read_json("scraped_data/products.json")

# Initialize analytics
analytics = AdvancedAnalytics(data=df)
analytics.preprocess_data()

# Run analysis
stats_results = analytics.descriptive_statistics()
outliers = analytics.detect_outliers()
correlations = analytics.correlation_analysis()
```

#### Machine Learning
```python
from ml_analytics import MLAnalytics

# Initialize ML analytics
ml_analytics = MLAnalytics(df, target_column='price')

# Price prediction
price_models = ml_analytics.price_prediction_models()

# Demand forecasting
forecasting = ml_analytics.demand_forecasting()

# Customer segmentation
clustering = ml_analytics.customer_segmentation_ml()

# Generate comprehensive ML report
ml_report = ml_analytics.generate_ml_report()
```

#### Recommendation System
```python
from recommendation_engine import RecommendationEngine

# Initialize recommendation engine
rec_engine = RecommendationEngine(df)

# Build models
rec_engine.build_hybrid_model()

# Get similar products
similar_products = rec_engine.get_similar_products(
    product_id=123, 
    n_recommendations=10
)

# Category recommendations
category_recs = rec_engine.get_category_recommendations(
    category="Electronics", 
    n_recommendations=15
)

# Price-based recommendations
price_recs = rec_engine.get_price_based_recommendations(
    target_price=500.0, 
    price_tolerance=0.2
)
```

### Visualization

#### Advanced Visualizations
```python
from advanced_visualizer import AdvancedVisualizer

# Initialize visualizer
visualizer = AdvancedVisualizer(df, output_dir="visualizations")

# Generate market analysis plots
market_plots = visualizer.create_market_analysis_plots()

# Price distribution analysis
price_plots = visualizer.create_price_distribution_plots()

# Quality analysis
quality_plots = visualizer.create_quality_analysis_plots()

# Word cloud
wordcloud_plots = visualizer.create_wordcloud_visualization()
```

## ⚙️ Configuration

### Scraper Configuration
```python
from config import ScraperConfig

config = ScraperConfig()
config.HEADLESS = True  # Run browser in headless mode
config.TIMEOUT = 30     # Request timeout
config.MAX_RETRIES = 3  # Maximum retry attempts
config.IMPLICIT_WAIT = 10  # Selenium wait time
```

### Dashboard Configuration
- **Data Directory**: Configure in `real_time_dashboard.py`
- **Auto-refresh**: Set refresh intervals
- **Export Formats**: Choose output formats
- **Cache Duration**: Adjust data caching

### ML Model Configuration
- **Target Variables**: Price, rating, reviews_count
- **Model Types**: Regression, Classification, Clustering
- **Hyperparameters**: Automated tuning with Optuna
- **Feature Engineering**: Automated feature creation

## 📊 Dashboard Features

### Main Dashboard Sections

#### 📈 Basic Analytics
- **KPI Cards**: Key metrics overview
- **Price Distribution**: Histograms and box plots
- **Category Analysis**: Product distribution and performance
- **Brand Analysis**: Top brands and positioning
- **Temporal Analysis**: Time-based patterns
- **Quality Metrics**: Data completeness scores

#### 🧠 Advanced Analytics
- **Statistical Analysis**: Descriptive statistics, correlations, outliers
- **Machine Learning**: Model training, evaluation, predictions
- **Visualizations**: Advanced charts and plots
- **Recommendations**: Product recommendation engine
- **Data Pipeline**: End-to-end processing workflow
- **Export Tools**: Multiple format exports

#### ⚙️ System Status
- **Performance Metrics**: Memory usage, processing speed
- **Module Status**: Component health monitoring
- **File System**: Directory and file management
- **Debug Information**: System diagnostics

### Interactive Features
- **Real-time Updates**: Auto-refresh capabilities
- **Filtering**: Dynamic data filtering
- **Export Options**: CSV, Excel, JSON, HTML reports
- **Cache Management**: Memory optimization
- **Model Persistence**: Save/load trained models

## 📁 Project Structure

```
data-retrieval/
├── 📊 Core Modules
│   ├── gjirafa_scraper.py       # Main web scraper
│   ├── advanced_analytics.py    # Statistical analysis
│   ├── ml_analytics.py         # Machine learning models
│   ├── recommendation_engine.py # Recommendation system
│   ├── advanced_visualizer.py  # Visualization engine
│   └── data_pipeline.py        # ETL pipeline
│
├── 🖥️ User Interfaces
│   ├── interactive_cli.py       # Command-line interface
│   ├── gui_scraper.py          # Graphical interface
│   └── real_time_dashboard.py  # Streamlit dashboard
│
├── ⚙️ Configuration & Utilities
│   ├── config.py               # Configuration settings
│   ├── utils.py                # Utility functions
│   └── restart_dashboard.py    # Dashboard launcher
│
├── 📋 Documentation & Setup
│   ├── README.md               # This file
│   ├── requirements.txt        # Dependencies
│   └── setup.py               # Package setup
│
└── 📁 Output Directories
    ├── scraped_data/           # Raw scraped data
    ├── dashboard_output/       # Dashboard exports
    ├── visualizations/         # Generated plots
    ├── ml_models/             # Trained ML models
    └── pipeline_output/       # Pipeline results
```

## 🔧 Advanced Usage

### Custom Data Pipeline
```python
from data_pipeline import DataPipeline

# Create custom pipeline
pipeline = DataPipeline(
    data_source="custom_data.csv",
    output_dir="custom_output"
)

# Run specific analyses
quality_report = pipeline.data_quality_assessment()
processed_data = pipeline.data_preprocessing()
statistical_results = pipeline.run_statistical_analysis()
ml_results = pipeline.run_ml_analysis()
visualizations = pipeline.run_visualizations()

# Generate report
comprehensive_report = pipeline.generate_comprehensive_report()
```

### Model Deployment
```python
# Save trained models
ml_analytics.save_models_and_results("production_models")

# Load and use models
import joblib
model = joblib.load("production_models/trained_models/price_prediction_xgboost.joblib")

# Make predictions
predictions = model['model'].predict(new_data)
```

### API Integration
```python
# Example FastAPI integration
from fastapi import FastAPI
from recommendation_engine import RecommendationEngine

app = FastAPI()
rec_engine = RecommendationEngine(df)

@app.get("/recommendations/{product_id}")
def get_recommendations(product_id: int, limit: int = 10):
    return rec_engine.get_similar_products(product_id, limit)
```

## 🐛 Troubleshooting

### Common Issues

#### ChromeDriver Issues
```bash
# If ChromeDriver fails to download automatically
pip install --upgrade webdriver-manager
```

#### Memory Issues
```python
# For large datasets, use chunking
chunk_size = 1000
for chunk in pd.read_json("large_file.json", chunksize=chunk_size):
    process_chunk(chunk)
```

#### Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

#### Dashboard Not Loading
```bash
# Check Streamlit installation
streamlit --version

# Clear Streamlit cache
streamlit cache clear
```

### Performance Optimization

#### Scraping Performance
- Use `HEADLESS=True` for faster scraping
- Adjust `TIMEOUT` and `IMPLICIT_WAIT` for your network
- Implement request delays to avoid blocking

#### ML Performance
- Use `n_jobs=-1` for parallel processing
- Implement early stopping in models
- Use feature selection for large datasets

#### Dashboard Performance
- Enable caching with `@st.cache_data`
- Limit data display size
- Use data sampling for large datasets

## 📈 Monitoring & Logging

### Logging Configuration
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
```

### Performance Monitoring
- Monitor scraping success rates
- Track model accuracy metrics
- Analyze dashboard usage patterns
- Monitor system resource usage

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `pytest tests/`
5. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all functions
- Include unit tests for new features

## 🙏 Acknowledgments

- **Selenium**: Web automation framework
- **Streamlit**: Dashboard framework
- **Scikit-learn**: Machine learning library
- **Plotly**: Interactive visualizations
- **Prophet**: Time series forecasting
- **BeautifulSoup**: HTML parsing

## 📞 Support

For support and questions:
- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/bonin1/data-retrieval/issues)
---

**Made with ❤️ for e-commerce data analysis**