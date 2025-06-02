"""
Comprehensive Data Pipeline
End-to-end data processing, analysis, and ML pipeline
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from advanced_analytics import AdvancedAnalytics
from advanced_visualizer import AdvancedVisualizer
from ml_analytics import MLAnalytics
from utils import DataValidator, DataExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipeline:
    """Comprehensive data processing and analytics pipeline"""
    
    def __init__(self, data_source: Union[str, pd.DataFrame], output_dir: str = "pipeline_output"):
        """
        Initialize the data pipeline
        
        Args:
            data_source: Path to data file or pandas DataFrame
            output_dir: Directory for outputs
        """
        self.data_source = data_source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Pipeline components
        self.raw_data = None
        self.processed_data = None
        self.analytics = None
        self.visualizer = None
        self.ml_analytics = None
        
        # Results storage
        self.results = {
            'data_quality': {},
            'statistical_analysis': {},
            'visualizations': {},
            'ml_results': {},
            'pipeline_metadata': {}
        }
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load data from source"""
        try:
            if isinstance(self.data_source, pd.DataFrame):
                self.raw_data = self.data_source.copy()
                logger.info(f"Loaded DataFrame with {len(self.raw_data)} records")
            
            elif isinstance(self.data_source, str):
                if self.data_source.endswith('.json'):
                    with open(self.data_source, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.raw_data = pd.DataFrame(data)
                elif self.data_source.endswith('.csv'):
                    self.raw_data = pd.read_csv(self.data_source)
                else:
                    raise ValueError("Unsupported file format")
                
                logger.info(f"Loaded {len(self.raw_data)} records from {self.data_source}")
            
            else:
                raise ValueError("Data source must be DataFrame or file path")
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def data_quality_assessment(self) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        logger.info("Performing data quality assessment...")
        
        if self.raw_data is None:
            raise ValueError("No data loaded")
        
        df = self.raw_data
        quality_report = {}
        
        # Basic data info
        quality_report['basic_info'] = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'data_types': df.dtypes.to_dict()
        }
        
        # Missing values analysis
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df) * 100).round(2)
        
        quality_report['missing_values'] = {
            'columns_with_missing': missing_data[missing_data > 0].to_dict(),
            'missing_percentages': missing_percentage[missing_percentage > 0].to_dict(),
            'total_missing_values': int(missing_data.sum()),
            'columns_mostly_missing': missing_percentage[missing_percentage > 50].index.tolist()
        }
        
        # Data completeness score
        completeness_scores = {}
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            completeness_scores[col] = (non_null_count / len(df)) * 100
        
        quality_report['completeness_scores'] = completeness_scores
        quality_report['overall_completeness'] = np.mean(list(completeness_scores.values()))
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        quality_report['duplicates'] = {
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': (duplicate_rows / len(df)) * 100
        }
        
        # Data type consistency
        inconsistent_types = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column should be numeric
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
                    if numeric_count > len(df) * 0.8:  # 80% numeric
                        inconsistent_types[col] = "Should be numeric"
                except:
                    pass
        
        quality_report['type_inconsistencies'] = inconsistent_types
        
        # Value range analysis for numerical columns
        numerical_analysis = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                numerical_analysis[col] = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'zeros_count': int((col_data == 0).sum()),
                    'negative_count': int((col_data < 0).sum()),
                    'outliers_iqr': self._detect_outliers_iqr(col_data)
                }
        
        quality_report['numerical_analysis'] = numerical_analysis
        
        # Text field analysis
        text_analysis = {}
        text_columns = ['title', 'description', 'brand', 'category']
        
        for col in text_columns:
            if col in df.columns:
                col_data = df[col].dropna().astype(str)
                if len(col_data) > 0:
                    text_analysis[col] = {
                        'avg_length': float(col_data.str.len().mean()),
                        'min_length': int(col_data.str.len().min()),
                        'max_length': int(col_data.str.len().max()),
                        'empty_strings': int((col_data == '').sum()),
                        'unique_values': int(col_data.nunique()),
                        'most_common': col_data.value_counts().head(5).to_dict()
                    }
        
        quality_report['text_analysis'] = text_analysis
        
        # Data quality score calculation
        quality_score = 0
        max_score = 100
        
        # Completeness (40 points)
        quality_score += (quality_report['overall_completeness'] / 100) * 40
        
        # No duplicates (20 points)
        if duplicate_rows == 0:
            quality_score += 20
        else:
            quality_score += max(0, 20 - (duplicate_rows / len(df)) * 100)
        
        # Type consistency (20 points)
        if not inconsistent_types:
            quality_score += 20
        else:
            quality_score += max(0, 20 - len(inconsistent_types) * 5)
        
        # Reasonable value ranges (20 points)
        outlier_penalty = 0
        for col, analysis in numerical_analysis.items():
            outlier_pct = analysis['outliers_iqr']['percentage']
            if outlier_pct > 10:  # More than 10% outliers
                outlier_penalty += 5
        
        quality_score += max(0, 20 - outlier_penalty)
        
        quality_report['overall_quality_score'] = min(100, quality_score)
        
        # Quality recommendations
        recommendations = []
        
        if quality_report['overall_completeness'] < 80:
            recommendations.append("Data has significant missing values - consider imputation strategies")
        
        if duplicate_rows > 0:
            recommendations.append(f"Remove {duplicate_rows} duplicate rows")
        
        if inconsistent_types:
            recommendations.append("Fix data type inconsistencies")
        
        for col, analysis in numerical_analysis.items():
            if analysis['outliers_iqr']['percentage'] > 15:
                recommendations.append(f"High outlier percentage in {col} - investigate data collection")
        
        quality_report['recommendations'] = recommendations
        
        self.results['data_quality'] = quality_report
        logger.info(f"Data quality score: {quality_score:.1f}/100")
        
        return quality_report
    
    def _detect_outliers_iqr(self, data: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
    
    def data_preprocessing(self) -> pd.DataFrame:
        """Advanced data preprocessing and cleaning"""
        logger.info("Starting data preprocessing...")
        
        if self.raw_data is None:
            raise ValueError("No data loaded")
        
        df = self.raw_data.copy()
        preprocessing_log = []
        
        # Remove completely empty rows
        initial_rows = len(df)
        df = df.dropna(how='all')
        if len(df) < initial_rows:
            removed = initial_rows - len(df)
            preprocessing_log.append(f"Removed {removed} completely empty rows")
        
        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            removed = initial_rows - len(df)
            preprocessing_log.append(f"Removed {removed} duplicate rows")
        
        # Clean and validate data using utils
        for idx, row in df.iterrows():
            # Convert row to dict and validate
            product_dict = row.to_dict()
            validated_product = DataValidator.validate_product_data(product_dict)
            
            # Update row with validated data
            for key, value in validated_product.items():
                if key in df.columns:
                    df.at[idx, key] = value
        
        # Handle missing values intelligently
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                # For numerical columns, use median for missing values
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    preprocessing_log.append(f"Filled {df[col].isnull().sum()} missing values in {col} with median")
            
            elif df[col].dtype == 'object':
                # For categorical columns, use mode or 'Unknown'
                if df[col].isnull().sum() > 0:
                    if col in ['brand', 'category']:
                        df[col] = df[col].fillna('Unknown')
                        preprocessing_log.append(f"Filled missing values in {col} with 'Unknown'")
                    elif col in ['title', 'description']:
                        df[col] = df[col].fillna('')
                        preprocessing_log.append(f"Filled missing values in {col} with empty string")
        
        # Feature engineering
        # Price features
        if 'price' in df.columns:
            df['price_log'] = np.log1p(df['price'].fillna(0))
            df['price_category'] = pd.cut(
                df['price'], 
                bins=5, 
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                include_lowest=True
            )
        
        if 'original_price' in df.columns and 'price' in df.columns:
            df['has_discount'] = (df['original_price'] > df['price']).astype(int)
            df['discount_amount'] = df['original_price'] - df['price']
            df['discount_percentage'] = (df['discount_amount'] / df['original_price'] * 100).fillna(0)
        
        # Text features
        if 'title' in df.columns:
            df['title_length'] = df['title'].str.len()
            df['word_count'] = df['title'].str.split().str.len()
            df['title_cleaned'] = df['title'].str.lower().str.strip()
        
        if 'description' in df.columns:
            df['description_length'] = df['description'].str.len()
            df['has_description'] = (df['description'].str.len() > 0).astype(int)
        
        # Rating features
        if 'rating' in df.columns:
            df['rating_category'] = pd.cut(
                df['rating'], 
                bins=[0, 3, 4, 4.5, 5], 
                labels=['Poor', 'Good', 'Very Good', 'Excellent'],
                include_lowest=True
            )
            df['high_rating'] = (df['rating'] >= 4.0).astype(int)
        
        # Reviews features
        if 'reviews_count' in df.columns:
            df['reviews_log'] = np.log1p(df['reviews_count'].fillna(0))
            df['has_reviews'] = (df['reviews_count'] > 0).astype(int)
            df['review_popularity'] = pd.cut(
                df['reviews_count'].fillna(0),
                bins=[0, 10, 50, 100, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High'],
                include_lowest=True
            )
        
        # Brand and category encoding
        if 'brand' in df.columns:
            brand_counts = df['brand'].value_counts()
            # Group rare brands
            rare_brands = brand_counts[brand_counts < 5].index
            df['brand_grouped'] = df['brand'].copy()
            df.loc[df['brand'].isin(rare_brands), 'brand_grouped'] = 'Other'
            
            # Brand popularity
            df['brand_popularity'] = df['brand'].map(brand_counts).fillna(0)
            df['is_popular_brand'] = (df['brand_popularity'] >= brand_counts.median()).astype(int)
        
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            df['category_popularity'] = df['category'].map(category_counts).fillna(0)
        
        # Time-based features
        if 'scraped_at' in df.columns:
            df['scraped_at'] = pd.to_datetime(df['scraped_at'])
            df['scraped_date'] = df['scraped_at'].dt.date
            df['scraped_hour'] = df['scraped_at'].dt.hour
            df['scraped_day_of_week'] = df['scraped_at'].dt.dayofweek
            df['scraped_month'] = df['scraped_at'].dt.month
            df['is_weekend'] = (df['scraped_day_of_week'] >= 5).astype(int)
        
        # Data quality score
        quality_factors = ['price', 'title', 'brand', 'category', 'rating', 'description']
        df['data_completeness_score'] = 0
        
        for factor in quality_factors:
            if factor in df.columns:
                df['data_completeness_score'] += (~df[factor].isna()).astype(int)
        
        df['data_completeness_score'] = df['data_completeness_score'] / len(quality_factors)
        
        # Store preprocessing log
        self.results['pipeline_metadata']['preprocessing_log'] = preprocessing_log
        self.processed_data = df
        
        logger.info(f"Preprocessing completed. Final dataset: {df.shape}")
        logger.info(f"Preprocessing steps: {len(preprocessing_log)}")
        
        return df
    
    def run_statistical_analysis(self) -> Dict[str, Any]:
        """Run comprehensive statistical analysis"""
        logger.info("Running statistical analysis...")
        
        if self.processed_data is None:
            self.data_preprocessing()
        
        # Initialize analytics module
        self.analytics = AdvancedAnalytics(data=self.processed_data)
        
        # Run all statistical analyses
        results = self.analytics.generate_insights()
        
        self.results['statistical_analysis'] = results
        
        # Save analytics results
        analytics_output_dir = self.output_dir / "analytics"
        self.analytics.save_results(str(analytics_output_dir))
        
        return results
    
    def run_visualizations(self) -> Dict[str, Any]:
        """Generate comprehensive visualizations"""
        logger.info("Generating visualizations...")
        
        if self.processed_data is None:
            self.data_preprocessing()
        
        # Initialize visualizer
        viz_output_dir = self.output_dir / "visualizations"
        self.visualizer = AdvancedVisualizer(self.processed_data, str(viz_output_dir))
        
        # Generate all visualizations
        viz_results = self.visualizer.save_all_visualizations()
        
        self.results['visualizations'] = viz_results
        
        return viz_results
    
    def run_ml_analysis(self) -> Dict[str, Any]:
        """Run machine learning analysis"""
        logger.info("Running machine learning analysis...")
        
        if self.processed_data is None:
            self.data_preprocessing()
        
        # Initialize ML analytics
        target_column = 'price' if 'price' in self.processed_data.columns else None
        
        if target_column is None:
            logger.warning("No suitable target column found for ML analysis")
            return {}
        
        self.ml_analytics = MLAnalytics(self.processed_data, target_column=target_column)
        
        # Run comprehensive ML analysis
        ml_results = self.ml_analytics.generate_ml_report()
        
        self.results['ml_results'] = ml_results
        
        # Save ML models and results
        ml_output_dir = self.output_dir / "ml_models"
        self.ml_analytics.save_models_and_results(str(ml_output_dir))
        
        return ml_results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate final comprehensive analytics report"""
        logger.info("Generating comprehensive report...")
        
        # Ensure all analyses are run
        quality_report = self.data_quality_assessment()
        self.data_preprocessing()
        statistical_results = self.run_statistical_analysis()
        visualization_results = self.run_visualizations()
        ml_results = self.run_ml_analysis()
        
        # Generate executive summary
        summary = {
            'pipeline_execution': {
                'execution_timestamp': datetime.now().isoformat(),
                'data_source': str(self.data_source),
                'output_directory': str(self.output_dir),
                'total_processing_time': 'N/A'  # Would need timing implementation
            },
            'data_overview': {
                'total_records': len(self.processed_data),
                'total_features': len(self.processed_data.columns),
                'data_quality_score': quality_report.get('overall_quality_score', 0),
                'completeness_score': quality_report.get('overall_completeness', 0)
            },
            'key_insights': [],
            'model_performance': {},
            'recommendations': []
        }
        
        # Extract key insights
        if 'insights' in statistical_results:
            summary['key_insights'].extend(statistical_results['insights'].get('key_findings', []))
        
        if quality_report.get('recommendations'):
            summary['recommendations'].extend(quality_report['recommendations'])
        
        # Model performance summary
        if ml_results and 'results' in ml_results:
            if 'price_prediction' in ml_results['results']:
                models = ml_results['results']['price_prediction']
                if models:
                    best_model = max(models.items(), key=lambda x: x[1].get('r2', 0))
                    summary['model_performance'] = {
                        'best_model_name': best_model[0],
                        'best_r2_score': best_model[1].get('r2', 0),
                        'best_rmse': best_model[1].get('rmse', 0),
                        'total_models_trained': len(models)
                    }
        
        # Business recommendations
        if quality_report.get('overall_quality_score', 0) > 80:
            summary['recommendations'].append("High data quality - suitable for production ML models")
        elif quality_report.get('overall_quality_score', 0) > 60:
            summary['recommendations'].append("Moderate data quality - consider data cleaning improvements")
        else:
            summary['recommendations'].append("Low data quality - significant data cleaning required")
        
        if 'clustering' in ml_results.get('results', {}):
            clustering = ml_results['results']['clustering']
            if 'kmeans' in clustering:
                optimal_k = clustering['kmeans'].get('optimal_k', 0)
                summary['recommendations'].append(f"Products naturally segment into {optimal_k} distinct groups")
        
        # Compile final report
        comprehensive_report = {
            'executive_summary': summary,
            'data_quality_assessment': quality_report,
            'statistical_analysis': statistical_results,
            'visualization_summary': {
                'total_visualizations': sum(len(plots) for plots in visualization_results.values()),
                'dashboard_location': str(self.output_dir / "visualizations" / "analytics_dashboard.html"),
                'categories': list(visualization_results.keys())
            },
            'machine_learning_analysis': ml_results,
            'pipeline_metadata': self.results['pipeline_metadata']
        }
          # Save comprehensive report
        report_file = self.output_dir / "comprehensive_analytics_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            # Convert numpy types and tuple keys before JSON serialization
            json_compatible_report = self._convert_numpy_types(comprehensive_report)
            json.dump(json_compatible_report, f, indent=2, ensure_ascii=False, default=str)
        
        # Generate HTML summary report
        html_report = self._generate_html_report(comprehensive_report)
        html_file = self.output_dir / "analytics_summary_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"Comprehensive report saved to {report_file}")
        logger.info(f"HTML summary report saved to {html_file}")
        
        return comprehensive_report
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML summary report"""
        summary = report['executive_summary']
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Analytics Pipeline Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}
                .content {{
                    padding: 30px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .metric-card {{
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 2.5em;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .metric-label {{
                    font-size: 1.1em;
                    opacity: 0.9;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 25px;
                    background: #f8f9fa;
                    border-radius: 10px;
                    border-left: 5px solid #667eea;
                }}
                .section h2 {{
                    color: #333;
                    margin-top: 0;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }}
                .insights-list {{
                    list-style: none;
                    padding: 0;
                }}
                .insights-list li {{
                    background: white;
                    margin: 10px 0;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #28a745;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .recommendations-list {{
                    list-style: none;
                    padding: 0;
                }}
                .recommendations-list li {{
                    background: white;
                    margin: 10px 0;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #ffc107;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .model-performance {{
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    color: #666;
                    border-top: 1px solid #eee;
                    margin-top: 40px;
                }}
                .quality-score {{
                    font-size: 1.2em;
                    font-weight: bold;
                    color: {'#28a745' if summary['data_overview']['data_quality_score'] > 80 else '#ffc107' if summary['data_overview']['data_quality_score'] > 60 else '#dc3545'};
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Analytics Pipeline Report</h1>
                    <p>Comprehensive Data Analysis & Machine Learning Insights</p>
                </div>
                
                <div class="content">
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{summary['data_overview']['total_records']:,}</div>
                            <div class="metric-label">Total Records</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary['data_overview']['total_features']}</div>
                            <div class="metric-label">Features Analyzed</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary['data_overview']['data_quality_score']:.1f}%</div>
                            <div class="metric-label">Data Quality Score</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{summary['model_performance'].get('total_models_trained', 0)}</div>
                            <div class="metric-label">ML Models Trained</div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>📊 Data Quality Assessment</h2>
                        <p>Overall data quality score: <span class="quality-score">{summary['data_overview']['data_quality_score']:.1f}%</span></p>
                        <p>Data completeness: <span class="quality-score">{summary['data_overview']['completeness_score']:.1f}%</span></p>
                    </div>
                    
                    {f'''
                    <div class="section">
                        <h2>🤖 Model Performance</h2>
                        <div class="model-performance">
                            <h3>Best Performing Model: {summary['model_performance']['best_model_name']}</h3>
                            <p>R² Score: <strong>{summary['model_performance']['best_r2_score']:.4f}</strong></p>
                            <p>RMSE: <strong>{summary['model_performance']['best_rmse']:.2f}</strong></p>
                        </div>
                    </div>
                    ''' if summary['model_performance'] else ''}
                    
                    {f'''
                    <div class="section">
                        <h2>💡 Key Insights</h2>
                        <ul class="insights-list">
                            {"".join(f"<li>{insight}</li>" for insight in summary['key_insights'])}
                        </ul>
                    </div>
                    ''' if summary['key_insights'] else ''}
                    
                    {f'''
                    <div class="section">
                        <h2>📋 Recommendations</h2>
                        <ul class="recommendations-list">
                            {"".join(f"<li>{rec}</li>" for rec in summary['recommendations'])}
                        </ul>
                    </div>
                    ''' if summary['recommendations'] else ''}
                    
                    <div class="section">
                        <h2>📁 Output Files</h2>
                        <ul>
                            <li><strong>Comprehensive Report:</strong> comprehensive_analytics_report.json</li>
                            <li><strong>Visualizations Dashboard:</strong> visualizations/analytics_dashboard.html</li>
                            <li><strong>ML Models:</strong> ml_models/ directory</li>
                            <li><strong>Statistical Analysis:</strong> analytics/ directory</li>
                        </ul>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Generated on {summary['pipeline_execution']['execution_timestamp']}</p>
                    <p>Pipeline completed successfully ✅</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def export_results(self, formats: List[str] = ['json', 'csv', 'excel']) -> Dict[str, str]:
        """Export processed data and results in multiple formats"""
        logger.info("Exporting results...")
        
        if self.processed_data is None:
            self.data_preprocessing()
        
        export_dir = self.output_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        
        exported_files = {}
        
        # Export processed data
        if 'csv' in formats:
            csv_file = export_dir / "processed_data.csv"
            self.processed_data.to_csv(csv_file, index=False)
            exported_files['processed_data_csv'] = str(csv_file)
        
        if 'excel' in formats:
            excel_file = export_dir / "processed_data.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                self.processed_data.to_excel(writer, sheet_name='Processed Data', index=False)
                
                # Add summary statistics
                summary_stats = self.processed_data.describe()
                summary_stats.to_excel(writer, sheet_name='Summary Statistics')
            
            exported_files['processed_data_excel'] = str(excel_file)
        
        if 'json' in formats:
            json_file = export_dir / "processed_data.json"
            self.processed_data.to_json(json_file, orient='records', indent=2)
            exported_files['processed_data_json'] = str(json_file)        # Export results summary
        if 'json' in formats:
            results_file = export_dir / "analysis_results_summary.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                # Convert numpy types and tuple keys before JSON serialization
                json_compatible_results = self._convert_numpy_types(self.results)
                json.dump(json_compatible_results, f, indent=2, ensure_ascii=False, default=str)
            exported_files['results_summary'] = str(results_file)
        
        logger.info(f"Exported {len(exported_files)} files to {export_dir}")
        return exported_files
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            # Convert tuple keys to strings for JSON compatibility
            return {str(key): self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)  # Convert tuples to lists
        else:
            return obj

def main():
    """Example usage of the data pipeline"""
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'title': [f'Product {i}' for i in range(500)],
        'price': np.random.lognormal(4, 1, 500),
        'original_price': np.random.lognormal(4.2, 1, 500),
        'rating': np.random.normal(4, 0.5, 500),
        'reviews_count': np.random.poisson(50, 500),
        'brand': np.random.choice(['Brand A', 'Brand B', 'Brand C', 'Brand D'], 500),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 500),
        'description': [f'Description for product {i}' for i in range(500)],
        'scraped_at': pd.date_range('2024-01-01', periods=500, freq='2H')
    })
    
    # Initialize and run pipeline
    pipeline = DataPipeline(sample_data, "sample_pipeline_output")
    
    # Generate comprehensive report
    report = pipeline.generate_comprehensive_report()
    
    # Export results
    exported_files = pipeline.export_results()
    
    print("Pipeline completed successfully!")
    print(f"Output directory: {pipeline.output_dir}")
    print(f"Data quality score: {report['executive_summary']['data_overview']['data_quality_score']:.1f}%")
    print(f"Total features analyzed: {report['executive_summary']['data_overview']['total_features']}")

if __name__ == "__main__":
    main()
