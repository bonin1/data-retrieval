import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import pingouin as pg

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
import lightgbm as lgb

from textblob import TextBlob
from wordcloud import WordCloud
import re
from collections import Counter

from prophet import Prophet

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAnalytics:
    def __init__(self, data_path: str = None, data: pd.DataFrame = None):
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.models = {}
        self.results = {}
        
        if data is not None:
            self.raw_data = data
        elif data_path:
            self.load_data(data_path)
    
    def load_data(self, path: str) -> pd.DataFrame:
        """Load data from file"""
        try:
            if path.endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.raw_data = pd.DataFrame(data)
            elif path.endswith('.csv'):
                self.raw_data = pd.read_csv(path)
            else:
                raise ValueError("Unsupported file format. Use JSON or CSV.")
            
            logger.info(f"Loaded {len(self.raw_data)} records from {path}")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        if self.raw_data is None:
            raise ValueError("No data loaded")
        
        df = self.raw_data.copy()
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        if 'original_price' in df.columns:
            df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce')
        else:
            df['original_price'] = df['price'] * np.random.uniform(1.0, 1.3, len(df))
        
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['reviews_count'] = pd.to_numeric(df['reviews_count'], errors='coerce')
        
        df['has_discount'] = (df['original_price'] > df['price']).astype(int)
        df['discount_amount'] = df['original_price'] - df['price']
        df['discount_percentage'] = (df['discount_amount'] / df['original_price']) * 100
        df['price_log'] = np.log1p(df['price'].fillna(0))
        
        df['title_length'] = df['title'].str.len()
        df['description_length'] = df['description'].str.len()
        df['word_count'] = df['title'].str.split().str.len()
        
        le_brand = LabelEncoder()
        le_category = LabelEncoder()
        
        df['brand_encoded'] = le_brand.fit_transform(df['brand'].fillna('Unknown'))
        df['category_encoded'] = le_category.fit_transform(df['category'].fillna('Unknown'))
        
        if 'scraped_at' in df.columns:
            df['scraped_at'] = pd.to_datetime(df['scraped_at'])
            df['hour'] = df['scraped_at'].dt.hour
            df['day_of_week'] = df['scraped_at'].dt.dayofweek
            df['month'] = df['scraped_at'].dt.month
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            if df['price'].notna().any() and df['price'].max() > df['price'].min():
                try:
                    df['price_category'] = pd.cut(df['price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                except (ValueError, TypeError):
                    df['price_category'] = pd.qcut(df['price'].rank(method='first'), q=4, 
                                                 labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            else:
                df['price_category'] = 'Medium'
        quality_factors = []
        
        possible_factors = ['price', 'description', 'rating', 'reviews_count', 'images', 'title', 'brand', 'category']
        for factor in possible_factors:
            if factor in df.columns:
                quality_factors.append(factor)
        
        if quality_factors:
            df['data_quality_score'] = 0
            for factor in quality_factors:
                df['data_quality_score'] += (~df[factor].isna()).astype(int)
            
            df['data_quality_score'] = df['data_quality_score'] / len(quality_factors)
        else:
            df['data_quality_score'] = 0.5
        
        self.processed_data = df
        logger.info(f"Preprocessed data with {df.shape[1]} features")
        return df
    
    def descriptive_statistics(self) -> Dict[str, Any]:
        if self.processed_data is None:
            self.preprocess_data()
        
        df = self.processed_data
        stats_results = {}
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        stats_results['numerical_summary'] = df[numerical_cols].describe()
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        stats_results['categorical_summary'] = {}
        
        for col in categorical_cols:
            if col in ['title', 'description', 'url']: 
                continue
            stats_results['categorical_summary'][col] = df[col].value_counts().head(10)
        
        if 'price' in df.columns:
            price_data = df['price'].dropna()
            stats_results['price_analysis'] = {
                'mean': price_data.mean(),
                'median': price_data.median(),
                'std': price_data.std(),
                'skewness': stats.skew(price_data),
                'kurtosis': stats.kurtosis(price_data),
                'quartiles': price_data.quantile([0.25, 0.5, 0.75]).to_dict()
            }
        
        missing_analysis = df.isnull().sum()
        stats_results['missing_values'] = missing_analysis[missing_analysis > 0].to_dict()
        
        correlation_matrix = df[numerical_cols].corr()
        stats_results['correlation_matrix'] = correlation_matrix
        
        self.results['descriptive_stats'] = stats_results
        return stats_results
    
    def price_analysis(self) -> Dict[str, Any]:
        df = self.processed_data
        price_results = {}
        
        price_data = df['price'].dropna()
        
        shapiro_stat, shapiro_p = stats.shapiro(price_data.sample(min(5000, len(price_data))))
        price_results['normality_test'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05
        }
        
        if 'category' in df.columns:
            category_prices = df.groupby('category')['price'].agg(['mean', 'median', 'std', 'count'])
            price_results['category_analysis'] = category_prices.to_dict()
            
            categories = df['category'].dropna().unique()
            if len(categories) > 1:
                category_groups = [df[df['category'] == cat]['price'].dropna() for cat in categories]
                category_groups = [group for group in category_groups if len(group) > 0]
                
                if len(category_groups) > 1:
                    f_stat, p_value = stats.f_oneway(*category_groups)
                    price_results['category_anova'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant_difference': p_value < 0.05
                    }
        
        price_features = ['brand_encoded', 'category_encoded', 'title_length', 'word_count', 'data_quality_score']
        available_features = [f for f in price_features if f in df.columns]
        
        if available_features and not df['price'].isna().all():
            feature_data = df[available_features + ['price']].dropna()
            
            if len(feature_data) > 10:
                X = feature_data[available_features]
                y = feature_data['price']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                models = {
                    'linear_regression': LinearRegression(),
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42)
                }
                
                model_results = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    model_results[name] = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                    }
                    
                    self.models[f'price_prediction_{name}'] = model
                
                price_results['prediction_models'] = model_results
        
        self.results['price_analysis'] = price_results
        return price_results
    
    def customer_segmentation(self) -> Dict[str, Any]:
        df = self.processed_data
        segmentation_results = {}
        
        feature_cols = ['price', 'rating', 'reviews_count', 'title_length', 'data_quality_score']
        available_cols = [col for col in feature_cols if col in df.columns and not df[col].isna().all()]
        
        if len(available_cols) >= 2:
            cluster_data = df[available_cols].dropna()
            
            if len(cluster_data) > 10:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                inertias = []
                silhouette_scores = []
                k_range = range(2, min(11, len(cluster_data)//2))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(scaled_data)
                    inertias.append(kmeans.inertia_)
                    silhouette_scores.append(silhouette_score(scaled_data, labels))
                
                optimal_k = k_range[np.argmax(silhouette_scores)]
                
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_data)
                
                df_clustered = cluster_data.copy()
                df_clustered['cluster'] = cluster_labels
                
                cluster_analysis = df_clustered.groupby('cluster').agg({
                    col: ['mean', 'median', 'std'] for col in available_cols
                }).round(2)
                
                segmentation_results['kmeans_clustering'] = {
                    'optimal_k': optimal_k,
                    'silhouette_score': silhouette_scores[optimal_k - 2],
                    'cluster_analysis': cluster_analysis.to_dict(),
                    'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict()
                }
                
                self.models['clustering_scaler'] = scaler
                self.models['clustering_kmeans'] = kmeans
                
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                dbscan_labels = dbscan.fit_predict(scaled_data)
                
                outliers = np.sum(dbscan_labels == -1)
                segmentation_results['outlier_detection'] = {
                    'outliers_count': int(outliers),
                    'outliers_percentage': (outliers / len(scaled_data)) * 100
                }
        
        self.results['segmentation'] = segmentation_results
        return segmentation_results
    
    def sentiment_analysis(self) -> Dict[str, Any]:
        df = self.processed_data
        sentiment_results = {}
        
        if 'title' in df.columns:
            titles = df['title'].dropna()
            
            sentiments = []
            polarities = []
            subjectivities = []
            
            for title in titles:
                blob = TextBlob(str(title))
                sentiment = blob.sentiment
                sentiments.append('positive' if sentiment.polarity > 0.1 
                                else 'negative' if sentiment.polarity < -0.1 
                                else 'neutral')
                polarities.append(sentiment.polarity)
                subjectivities.append(sentiment.subjectivity)
            
            sentiment_results['title_sentiment'] = {
                'sentiment_distribution': pd.Series(sentiments).value_counts().to_dict(),
                'average_polarity': np.mean(polarities),
                'average_subjectivity': np.mean(subjectivities)
            }
            
            all_words = ' '.join(titles).lower()
            all_words = re.sub(r'[^a-zA-Z\s]', '', all_words)
            words = all_words.split()
            
            stop_words = set(['dhe', 'per', 'me', 'nga', 'ne', 'te', 'se', 'nje', 'i', 'e', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            word_freq = Counter(filtered_words)
            sentiment_results['word_frequency'] = dict(word_freq.most_common(20))
        
        self.results['sentiment_analysis'] = sentiment_results
        return sentiment_results
    
    def anomaly_detection(self) -> Dict[str, Any]:
        df = self.processed_data
        anomaly_results = {}
        
        if 'price' in df.columns:
            price_data = df['price'].dropna()
            
            Q1 = price_data.quantile(0.25)
            Q3 = price_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            price_outliers = price_data[(price_data < lower_bound) | (price_data > upper_bound)]
            
            anomaly_results['price_outliers'] = {
                'count': len(price_outliers),
                'percentage': (len(price_outliers) / len(price_data)) * 100,
                'outlier_values': price_outliers.tolist()[:10] 
            }
            
            feature_cols = ['price', 'rating', 'reviews_count', 'title_length']
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if len(available_cols) >= 2:
                anomaly_data = df[available_cols].dropna()
                
                if len(anomaly_data) > 10:
                    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_labels = isolation_forest.fit_predict(anomaly_data)
                    
                    anomalies = anomaly_data[anomaly_labels == -1]
                    
                    anomaly_results['multivariate_anomalies'] = {
                        'count': len(anomalies),
                        'percentage': (len(anomalies) / len(anomaly_data)) * 100
                    }
        
        self.results['anomaly_detection'] = anomaly_results
        return anomaly_results
    
    def competitive_analysis(self) -> Dict[str, Any]:
        df = self.processed_data
        competitive_results = {}
        
        if 'brand' in df.columns and 'price' in df.columns:
            brand_analysis = df.groupby('brand').agg({
                'price': ['mean', 'median', 'min', 'max', 'count'],
                'rating': 'mean',
                'reviews_count': 'sum'
            }).round(2)
        
            brand_counts = df['brand'].value_counts()
            total_products = len(df)
            market_share = ((brand_counts / total_products) * 100).round(2)
            
            competitive_results['brand_analysis'] = brand_analysis.to_dict()
            competitive_results['market_share'] = market_share.to_dict()
            
            brand_price_stats = df.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
            overall_mean_price = df['price'].mean()
            
            brand_price_stats['price_position'] = brand_price_stats['mean'].apply(
                lambda x: 'Premium' if x > overall_mean_price * 1.2
                else 'Budget' if x < overall_mean_price * 0.8
                else 'Mid-range'
            )
            
            competitive_results['price_positioning'] = brand_price_stats.to_dict()
        
        self.results['competitive_analysis'] = competitive_results
        return competitive_results
    
    def generate_insights(self) -> Dict[str, Any]:
        insights = {}
        
        self.descriptive_statistics()
        self.price_analysis()
        self.customer_segmentation()
        self.sentiment_analysis()
        self.anomaly_detection()
        self.competitive_analysis()
        
        insights['key_findings'] = []
        
        if 'price_analysis' in self.results:
            price_data = self.results['price_analysis']
            if 'category_anova' in price_data and price_data['category_anova']['significant_difference']:
                insights['key_findings'].append("Significant price differences exist across product categories")
        
        if 'segmentation' in self.results:
            seg_data = self.results['segmentation']
            if 'kmeans_clustering' in seg_data:
                k = seg_data['kmeans_clustering']['optimal_k']
                insights['key_findings'].append(f"Products can be segmented into {k} distinct clusters")
        
        if 'sentiment_analysis' in self.results:
            sent_data = self.results['sentiment_analysis']
            if 'title_sentiment' in sent_data:
                avg_polarity = sent_data['title_sentiment']['average_polarity']
                if avg_polarity > 0.1:
                    insights['key_findings'].append("Product titles show generally positive sentiment")
                elif avg_polarity < -0.1:
                    insights['key_findings'].append("Product titles show generally negative sentiment")
        
        if 'anomaly_detection' in self.results:
            anom_data = self.results['anomaly_detection']
            if 'price_outliers' in anom_data:
                outlier_pct = anom_data['price_outliers']['percentage']
                if outlier_pct > 10:
                    insights['key_findings'].append(f"High percentage ({outlier_pct:.1f}%) of price outliers detected")
        
        if 'competitive_analysis' in self.results:
            comp_data = self.results['competitive_analysis']
            if 'market_share' in comp_data:
                top_brand = max(comp_data['market_share'], key=comp_data['market_share'].get)
                top_share = comp_data['market_share'][top_brand]
                insights['key_findings'].append(f"{top_brand} leads market share with {top_share}% of products")
        
        insights['analysis_summary'] = {
            'total_analyses_performed': len(self.results),
            'data_points_analyzed': len(self.processed_data) if self.processed_data is not None else 0,
            'models_trained': len(self.models),
            'generated_at': datetime.now().isoformat()
        }
        
        self.results['insights'] = insights
        return insights
    
    def save_results(self, output_dir: str = "analytics_output"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results_file = output_path / "advanced_analytics_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json_results = self._convert_numpy_types(self.results)
            json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
        
        if self.models:
            models_dir = output_path / "models"
            models_dir.mkdir(exist_ok=True)
            
            for model_name, model in self.models.items():
                model_file = models_dir / f"{model_name}.joblib"
                joblib.dump(model, model_file)
        
        if self.processed_data is not None:
            data_file = output_path / "processed_data.csv"
            self.processed_data.to_csv(data_file, index=False)
        
        logger.info(f"Results saved to {output_path}")
        return output_path
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(key): self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj) 
        else:
            return obj

def main():
    sample_data = {
        'title': ['Product A', 'Product B', 'Product C'],
        'price': [100, 200, 150],
        'brand': ['Brand1', 'Brand2', 'Brand1'],
        'category': ['Electronics', 'Clothing', 'Electronics'],
        'rating': [4.5, 3.8, 4.2],
        'reviews_count': [100, 50, 75]
    }
    
    df = pd.DataFrame(sample_data)
    
    analytics = AdvancedAnalytics(data=df)
    results = analytics.generate_insights()
    
    print("Advanced Analytics Results:")
    for finding in results['key_findings']:
        print(f"- {finding}")

if __name__ == "__main__":
    main()
