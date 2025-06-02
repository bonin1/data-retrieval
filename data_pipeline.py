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
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        
        # Load and clean data
        self.raw_data = self._load_data(data_source)
        self.processed_data = None
        self.analytics = None
        self.ml_analytics = None
        self.visualizer = None
        
        logger.info(f"Loaded DataFrame with {len(self.raw_data)} records")
    
    def _load_data(self, data_source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Load data from various sources"""
        try:
            if isinstance(data_source, pd.DataFrame):
                df = data_source.copy()
            elif isinstance(data_source, str):
                if data_source.endswith('.json'):
                    with open(data_source, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                elif data_source.endswith('.csv'):
                    df = pd.read_csv(data_source)
                else:
                    raise ValueError(f"Unsupported file format: {data_source}")
            else:
                raise ValueError("Data source must be DataFrame or file path")
            
            # Clean data immediately after loading
            df = self._clean_raw_data(df)
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data to prevent encoding and Arrow issues"""
        try:
            # Handle timestamp columns
            if 'scraped_at' in df.columns:
                df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
                df['scraped_at_str'] = df['scraped_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Clean all object columns
            for col in df.select_dtypes(include=['object']).columns:
                if col in df.columns:
                    # Handle lists, tuples, dicts
                    df[col] = df[col].apply(self._convert_complex_to_string)
                    # Ensure all values are strings
                    df[col] = df[col].astype(str).replace(['nan', 'None', 'null'], '')
            
            # Clean numeric columns
            numeric_candidates = ['price', 'rating', 'reviews_count', 'discount_percentage']
            for col in numeric_candidates:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning raw data: {e}")
            return df
    
    def _convert_complex_to_string(self, value):
        """Convert complex data types to strings"""
        if pd.isna(value):
            return ''
        elif isinstance(value, (list, tuple)):
            return ', '.join(map(str, value))
        elif isinstance(value, dict):
            return str(value)
        else:
            return str(value)
    
    def data_quality_assessment(self) -> Dict[str, Any]:
        """Assess data quality with error handling"""
        logger.info("Performing data quality assessment...")
        
        try:
            df = self.raw_data
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            completeness = ((total_cells - missing_cells) / total_cells) * 100
            
            # Quality metrics
            quality_metrics = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'missing_cells': int(missing_cells),
                'completeness_percentage': round(completeness, 2),
                'overall_score': min(100, max(0, completeness)),
                'missing_percentage': round((missing_cells / total_cells) * 100, 2)
            }
            
            # Column-specific quality
            column_quality = {}
            for col in df.columns:
                if col in df.columns:
                    missing_pct = (df[col].isnull().sum() / len(df)) * 100
                    column_quality[col] = {
                        'missing_percentage': round(missing_pct, 2),
                        'data_type': str(df[col].dtype),
                        'unique_values': int(df[col].nunique()) if df[col].dtype != 'object' else min(df[col].nunique(), 100)
                    }
            
            quality_metrics['column_quality'] = column_quality
            
            logger.info(f"Data quality score: {quality_metrics['overall_score']}/100")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return {'error': str(e), 'overall_score': 0}
    
    def data_preprocessing(self) -> pd.DataFrame:
        """Preprocess data with robust error handling"""
        logger.info("Starting data preprocessing...")
        
        try:
            df = self.raw_data.copy()
            processing_steps = []
            
            # Remove duplicates
            before_dedup = len(df)
            df = df.drop_duplicates()
            after_dedup = len(df)
            
            if before_dedup != after_dedup:
                processing_steps.append(f"Removed {before_dedup - after_dedup} duplicates")
            
            # Handle missing values
            for col in df.columns:
                if df[col].dtype in ['object']:
                    df[col] = df[col].fillna('')
                elif df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
            
            processing_steps.append("Filled missing values")
            
            self.processed_data = df
            
            logger.info(f"Preprocessing completed. Final dataset: {df.shape}")
            logger.info(f"Preprocessing steps: {len(processing_steps)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return self.raw_data.copy()
    
    def run_statistical_analysis(self) -> Dict[str, Any]:
        """Run statistical analysis"""
        logger.info("Running statistical analysis...")
        
        try:
            if self.processed_data is None:
                self.data_preprocessing()
            
            # Initialize analytics
            self.analytics = AdvancedAnalytics(data=self.processed_data)
            self.analytics.preprocess_data()
            
            # Run analyses
            results = {}
            
            # Descriptive statistics
            desc_stats = self.analytics.descriptive_statistics()
            results['descriptive_statistics'] = desc_stats
            
            # Correlation analysis
            corr_analysis = self.analytics.correlation_analysis()
            results['correlation_analysis'] = corr_analysis
            
            # Outlier detection
            outliers = self.analytics.detect_outliers()
            results['outlier_detection'] = outliers
            
            self.results['statistical_analysis'] = results
            return results
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {'error': str(e)}
    
    def run_ml_analysis(self) -> Dict[str, Any]:
        """Run ML analysis with proper error handling"""
        logger.info("Running machine learning analysis...")
        
        try:
            # Ensure we have processed data
            if self.processed_data is None:
                self.data_preprocessing()
            
            # Initialize ML analytics with cleaned data
            self.ml_analytics = MLAnalytics(self.processed_data.copy(), target_column='price')
            
            # Run ML models with error handling
            ml_results = {}
            
            try:
                price_results = self.ml_analytics.price_prediction_models()
                ml_results['price_prediction'] = price_results
            except Exception as e:
                logger.warning(f"Price prediction failed: {e}")
                ml_results['price_prediction'] = {'error': str(e)}
            
            try:
                clustering_results = self.ml_analytics.customer_segmentation_ml()
                ml_results['clustering'] = clustering_results
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")
                ml_results['clustering'] = {'error': str(e)}
            
            self.results['ml_analysis'] = ml_results
            return ml_results
            
        except Exception as e:
            logger.error(f"ML analysis failed: {e}")
            return {'error': str(e)}

    def run_visualizations(self) -> Dict[str, Any]:
        """Run visualization generation"""
        logger.info("Running visualization generation...")
        
        try:
            if self.processed_data is None:
                self.data_preprocessing()
            
            # Initialize visualizer
            viz_output_dir = self.output_dir / "visualizations"
            viz_output_dir.mkdir(exist_ok=True)
            self.visualizer = AdvancedVisualizer(self.processed_data, str(viz_output_dir))
            
            # Generate visualizations
            viz_results = {}
            
            try:
                market_plots = self.visualizer.create_market_analysis_plots()
                viz_results['market_analysis'] = market_plots
            except Exception as e:
                logger.warning(f"Market analysis plots failed: {e}")
                viz_results['market_analysis'] = {'error': str(e)}
            
            try:
                price_plots = self.visualizer.create_price_distribution_plots()
                viz_results['price_analysis'] = price_plots
            except Exception as e:
                logger.warning(f"Price analysis plots failed: {e}")
                viz_results['price_analysis'] = {'error': str(e)}
            
            self.results['visualizations'] = viz_results
            return viz_results
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        logger.info("Generating comprehensive report...")
        
        try:
            # Ensure all analyses are run
            if 'statistical_analysis' not in self.results:
                self.run_statistical_analysis()
            
            if 'ml_analysis' not in self.results:
                self.run_ml_analysis()
            
            if 'visualizations' not in self.results:
                self.run_visualizations()
            
            # Generate report
            report = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_records': len(self.raw_data),
                    'processed_records': len(self.processed_data) if self.processed_data is not None else 0,
                    'analysis_modules': list(self.results.keys())
                },
                'data_quality': self.data_quality_assessment(),
                'results': self.results,
                'summary': {
                    'key_insights': [],
                    'recommendations': []
                }
            }
            
            # Add key insights
            if 'statistical_analysis' in self.results:
                report['summary']['key_insights'].append("Statistical analysis completed successfully")
            
            if 'ml_analysis' in self.results:
                ml_results = self.results['ml_analysis']
                if 'price_prediction' in ml_results and isinstance(ml_results['price_prediction'], dict):
                    best_model = max(ml_results['price_prediction'].items(), 
                                   key=lambda x: x[1].get('r2', 0) if isinstance(x[1], dict) else 0)
                    if len(best_model) == 2 and isinstance(best_model[1], dict):
                        r2_score = best_model[1].get('r2', 0)
                        report['summary']['key_insights'].append(f"Best ML model achieved R² score of {r2_score:.3f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def export_results(self, formats: List[str]) -> Dict[str, str]:
        """Export results in specified formats"""
        export_paths = {}
        
        try:
            # Generate report first
            report = self.generate_comprehensive_report()
            
            for format_type in formats:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    if format_type == 'json':
                        file_path = self.output_dir / f"pipeline_report_{timestamp}.json"
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(report, f, indent=2, default=str)
                        export_paths['json'] = str(file_path)
                    
                    elif format_type == 'csv' and self.processed_data is not None:
                        file_path = self.output_dir / f"processed_data_{timestamp}.csv"
                        self.processed_data.to_csv(file_path, index=False)
                        export_paths['csv'] = str(file_path)
                    
                    elif format_type == 'html report':
                        file_path = self.output_dir / f"analytics_report_{timestamp}.html"
                        html_content = self._generate_html_report(report)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        export_paths['html'] = str(file_path)
                        
                except Exception as e:
                    logger.error(f"Failed to export {format_type}: {e}")
                    continue
            
            return export_paths
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {}
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report from results"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Pipeline Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; color: #2E86AB; }}
                    .section {{ margin: 30px 0; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                             background: #f0f0f0; border-radius: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #2E86AB; color: white; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>📊 Data Pipeline Analysis Report</h1>
                    <p>Generated: {report.get('metadata', {}).get('generated_at', 'Unknown')}</p>
                </div>
                
                <div class="section">
                    <h2>📈 Summary</h2>
                    <div class="metric">
                        <strong>Total Records:</strong> {report.get('metadata', {}).get('total_records', 0):,}
                    </div>
                    <div class="metric">
                        <strong>Data Quality:</strong> {report.get('data_quality', {}).get('overall_score', 0):.1f}%
                    </div>
                </div>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"HTML report generation failed: {e}")
            return f"<html><body><h1>Report Generation Error</h1><p>{str(e)}</p></body></html>"

def main():
    """Example usage"""
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'title': [f'Product {i}' for i in range(100)],
        'price': np.random.lognormal(4, 1, 100),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home'], 100),
        'scraped_at': pd.date_range('2024-01-01', periods=100, freq='1H')
    })
    
    # Initialize pipeline
    pipeline = DataPipeline(sample_data)
    
    # Run complete analysis
    report = pipeline.generate_comprehensive_report()
    
    print("Pipeline completed successfully!")
    print(f"Data quality score: {report.get('data_quality', {}).get('overall_score', 0):.1f}%")

if __name__ == "__main__":
    main()
