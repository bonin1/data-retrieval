"""
Advanced Analytics Module
Statistical analysis and data insights for e-commerce data
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import warnings
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    """Advanced statistical analysis for e-commerce data"""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with processed data"""
        self.data = data.copy()
        self.processed_data = None
        self.analysis_results = {}
        
    def preprocess_data(self):
        """Preprocess data for analysis"""
        logger.info("Preprocessing data for advanced analytics...")
        
        try:
            df = self.data.copy()
            
            # Handle numeric columns
            numeric_cols = ['price', 'rating', 'reviews_count', 'original_price', 'discount_percentage']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create derived features
            if 'price' in df.columns and 'original_price' in df.columns:
                df['discount_amount'] = df['original_price'] - df['price']
                df['has_discount'] = (df['discount_amount'] > 0).astype(int)
            
            if 'rating' in df.columns:
                df['rating_category'] = pd.cut(df['rating'], 
                                             bins=[0, 2, 3, 4, 5], 
                                             labels=['Poor', 'Fair', 'Good', 'Excellent'])
            
            # Text length features
            text_cols = ['title', 'description']
            for col in text_cols:
                if col in df.columns:
                    df[f'{col}_length'] = df[col].astype(str).str.len()
                    df[f'{col}_word_count'] = df[col].astype(str).str.split().str.len()
            
            self.processed_data = df
            logger.info(f"Preprocessed data with {len(df.columns)} features")
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            self.processed_data = self.data.copy()
    
    def descriptive_statistics(self) -> Dict[str, Any]:
        """Comprehensive descriptive statistics"""
        if self.processed_data is None:
            self.preprocess_data()
        
        results = {}
        df = self.processed_data
        
        # Numeric column analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns and not df[col].isna().all():
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    results[f'{col}_analysis'] = {
                        'count': len(col_data),
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'skewness': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis()),
                        'percentile_25': float(col_data.quantile(0.25)),
                        'percentile_75': float(col_data.quantile(0.75)),
                        'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25))
                    }
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col in df.columns and not df[col].isna().all():
                value_counts = df[col].value_counts()
                
                results[f'{col}_distribution'] = {
                    'unique_values': len(value_counts),
                    'most_common': value_counts.head(5).to_dict(),
                    'null_count': int(df[col].isna().sum()),
                    'null_percentage': float((df[col].isna().sum() / len(df)) * 100)
                }
        
        return results
    
    def correlation_analysis(self) -> Dict[str, Any]:
        """Correlation analysis between variables"""
        if self.processed_data is None:
            self.preprocess_data()
        
        df = self.processed_data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'error': 'Not enough numeric columns for correlation analysis'}
        
        # Calculate correlation matrices
        pearson_corr = df[numeric_cols].corr(method='pearson')
        spearman_corr = df[numeric_cols].corr(method='spearman')
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                col1, col2 = pearson_corr.columns[i], pearson_corr.columns[j]
                pearson_val = pearson_corr.iloc[i, j]
                spearman_val = spearman_corr.iloc[i, j]
                
                if abs(pearson_val) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'pearson_correlation': float(pearson_val),
                        'spearman_correlation': float(spearman_val)
                    })
        
        return {
            'pearson_correlation_matrix': pearson_corr.to_dict(),
            'spearman_correlation_matrix': spearman_corr.to_dict(),
            'strong_correlations': strong_correlations
        }
    
    def detect_outliers(self) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        if self.processed_data is None:
            self.preprocess_data()
        
        df = self.processed_data
        outlier_results = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns and not df[col].isna().all():
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    # IQR method
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                    # Z-score method
                    z_scores = np.abs(stats.zscore(col_data))
                    zscore_outliers = col_data[z_scores > 3]
                    
                    outlier_results[f'{col}_outliers'] = {
                        'iqr_method': {
                            'count': len(iqr_outliers),
                            'percentage': (len(iqr_outliers) / len(col_data)) * 100,
                            'values': iqr_outliers.tolist()[:10]  # First 10 outliers
                        },
                        'zscore_method': {
                            'count': len(zscore_outliers),
                            'percentage': (len(zscore_outliers) / len(col_data)) * 100,
                            'values': zscore_outliers.tolist()[:10]
                        },
                        'bounds': {
                            'iqr_lower': float(lower_bound),
                            'iqr_upper': float(upper_bound)
                        }
                    }
        
        return outlier_results
    
    def price_analysis(self) -> Dict[str, Any]:
        """Detailed price analysis"""
        if self.processed_data is None:
            self.preprocess_data()
        
        df = self.processed_data
        
        if 'price' not in df.columns:
            return {'error': 'Price column not found'}
        
        price_data = df['price'].dropna()
        
        if len(price_data) == 0:
            return {'error': 'No valid price data'}
        
        results = {
            'basic_stats': {
                'mean': float(price_data.mean()),
                'median': float(price_data.median()),
                'std': float(price_data.std()),
                'min': float(price_data.min()),
                'max': float(price_data.max())
            },
            'price_ranges': {
                'under_50': len(price_data[price_data < 50]),
                '50_to_100': len(price_data[(price_data >= 50) & (price_data < 100)]),
                '100_to_500': len(price_data[(price_data >= 100) & (price_data < 500)]),
                'over_500': len(price_data[price_data >= 500])
            }
        }
        
        # Price by category analysis
        if 'category' in df.columns:
            category_prices = df.groupby('category')['price'].agg(['mean', 'median', 'count']).round(2)
            results['price_by_category'] = category_prices.to_dict()
        
        # Price by brand analysis
        if 'brand' in df.columns:
            brand_prices = df.groupby('brand')['price'].agg(['mean', 'median', 'count']).round(2)
            # Get top 10 brands by count
            top_brands = brand_prices.nlargest(10, 'count')
            results['price_by_brand'] = top_brands.to_dict()
        
        return results
    
    def market_segmentation(self) -> Dict[str, Any]:
        """Market segmentation analysis"""
        if self.processed_data is None:
            self.preprocess_data()
        
        df = self.processed_data
        results = {}
        
        # Price segmentation
        if 'price' in df.columns:
            price_data = df['price'].dropna()
            price_segments = pd.qcut(price_data, q=4, labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
            
            results['price_segments'] = {
                'distribution': price_segments.value_counts().to_dict(),
                'segment_stats': df.groupby(price_segments)['price'].agg(['mean', 'count']).to_dict()
            }
        
        # Rating segmentation
        if 'rating' in df.columns:
            rating_data = df['rating'].dropna()
            if len(rating_data) > 0:
                rating_segments = pd.cut(rating_data, bins=[0, 2, 3, 4, 5], 
                                       labels=['Poor', 'Fair', 'Good', 'Excellent'])
                
                results['rating_segments'] = {
                    'distribution': rating_segments.value_counts().to_dict()
                }
        
        return results
    
    def trend_analysis(self) -> Dict[str, Any]:
        """Analyze trends over time"""
        if self.processed_data is None:
            self.preprocess_data()
        
        df = self.processed_data
        
        # Check for time columns
        time_cols = [col for col in df.columns if 'scraped' in col.lower() or 'date' in col.lower()]
        
        if not time_cols:
            return {'error': 'No time columns found for trend analysis'}
        
        results = {}
        
        for time_col in time_cols:
            try:
                df[f'{time_col}_dt'] = pd.to_datetime(df[time_col], errors='coerce')
                
                if df[f'{time_col}_dt'].notna().any():
                    # Daily trends
                    daily_counts = df.groupby(df[f'{time_col}_dt'].dt.date).size()
                    
                    results[f'{time_col}_trends'] = {
                        'daily_average': float(daily_counts.mean()),
                        'total_days': len(daily_counts),
                        'max_daily': int(daily_counts.max()),
                        'min_daily': int(daily_counts.min())
                    }
                    
                    # Price trends over time
                    if 'price' in df.columns:
                        daily_prices = df.groupby(df[f'{time_col}_dt'].dt.date)['price'].mean()
                        
                        results[f'{time_col}_price_trends'] = {
                            'average_price_trend': daily_prices.to_dict(),
                            'price_volatility': float(daily_prices.std())
                        }
                        
            except Exception as e:
                logger.warning(f"Could not analyze trends for {time_col}: {e}")
                continue
        
        return results
    
    def generate_insights(self) -> Dict[str, Any]:
        """Generate actionable insights from all analyses"""
        insights = {
            'data_quality_insights': [],
            'market_insights': [],
            'pricing_insights': [],
            'recommendations': []
        }
        
        try:
            # Run all analyses
            desc_stats = self.descriptive_statistics()
            correlation_results = self.correlation_analysis()
            outlier_results = self.detect_outliers()
            price_results = self.price_analysis()
            
            # Data quality insights
            total_records = len(self.processed_data)
            
            # Check data completeness
            for col in ['price', 'title', 'category']:
                if col in self.processed_data.columns:
                    missing_pct = (self.processed_data[col].isna().sum() / total_records) * 100
                    if missing_pct > 20:
                        insights['data_quality_insights'].append(
                            f"High missing data in {col}: {missing_pct:.1f}%"
                        )
            
            # Price insights
            if 'price_analysis' in locals() and 'basic_stats' in price_results:
                price_stats = price_results['basic_stats']
                price_cv = price_stats['std'] / price_stats['mean']  # Coefficient of variation
                
                if price_cv > 1:
                    insights['pricing_insights'].append("High price variability detected")
                
                insights['pricing_insights'].append(
                    f"Average price: €{price_stats['mean']:.2f}, Range: €{price_stats['min']:.2f} - €{price_stats['max']:.2f}"
                )
            
            # Market insights
            if 'category' in self.processed_data.columns:
                category_counts = self.processed_data['category'].value_counts()
                dominant_category = category_counts.index[0]
                market_share = (category_counts.iloc[0] / total_records) * 100
                
                insights['market_insights'].append(
                    f"Dominant category: {dominant_category} ({market_share:.1f}% market share)"
                )
            
            # Recommendations
            if 'strong_correlations' in correlation_results:
                strong_corrs = correlation_results['strong_correlations']
                if strong_corrs:
                    insights['recommendations'].append("Strong correlations found - consider feature engineering")
            
            # Outlier recommendations
            outlier_count = 0
            for col_outliers in outlier_results.values():
                if isinstance(col_outliers, dict) and 'iqr_method' in col_outliers:
                    outlier_count += col_outliers['iqr_method']['count']
            
            if outlier_count > total_records * 0.1:
                insights['recommendations'].append("High outlier rate - review data collection process")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights['error'] = str(e)
        
        return insights
