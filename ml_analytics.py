"""
Machine Learning Analytics Module
Advanced ML models for e-commerce predictive analytics
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier

# Time series forecasting
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Utilities
import joblib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLAnalytics:
    """Comprehensive machine learning analytics for e-commerce data"""
    
    def __init__(self, data: pd.DataFrame, target_column: str = 'price'):
        """
        Initialize ML Analytics
        
        Args:
            data: Processed pandas DataFrame
            target_column: Target variable for prediction
        """
        self.data = data.copy()
        self.target_column = target_column
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.predictions = {}
        
        # Prepare features
        self._prepare_features()
    
    def _prepare_features(self):
        """Prepare features for machine learning"""
        # Clean data first to avoid encoder issues
        self.data = self._clean_data_for_ml()
        
        # Encode categorical variables
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in ['title', 'description', 'url', 'scraped_at', 'scraped_at_str']:  # Skip text/url columns
                try:
                    # Ensure column contains only strings
                    self.data[col] = self.data[col].astype(str).fillna('Unknown')
                    
                    # Check if column has valid values for encoding
                    unique_values = self.data[col].unique()
                    if len(unique_values) > 1 and not all(val == '' for val in unique_values):
                        le = LabelEncoder()
                        self.data[f'{col}_encoded'] = le.fit_transform(self.data[col])
                    else:
                        # Skip columns with no meaningful variation
                        logger.warning(f"Skipping encoding for column {col} - no variation")
                        
                except Exception as e:
                    logger.warning(f"Could not encode column {col}: {e}")
                    continue
        
        # Create text-based features
        if 'title' in self.data.columns:
            try:
                self.data['title_length'] = self.data['title'].astype(str).str.len()
                self.data['word_count'] = self.data['title'].astype(str).str.split().str.len()
                self.data['has_brand_in_title'] = self.data.apply(
                    lambda x: 1 if pd.notna(x.get('brand')) and str(x.get('brand')).lower() in str(x.get('title', '')).lower() else 0,
                    axis=1
                )
            except Exception as e:
                logger.warning(f"Error creating text features: {e}")
        
        # Create price-based features
        if 'price' in self.data.columns:
            try:
                price_series = pd.to_numeric(self.data['price'], errors='coerce').fillna(0)
                self.data['price_log'] = np.log1p(price_series)
            except Exception as e:
                logger.warning(f"Error creating price features: {e}")
        
        if 'original_price' in self.data.columns and 'price' in self.data.columns:
            self.data['discount_amount'] = self.data['original_price'] - self.data['price']
            self.data['discount_percentage'] = (
                self.data['discount_amount'] / self.data['original_price'] * 100
            ).fillna(0)
            self.data['has_discount'] = (self.data['discount_amount'] > 0).astype(int)
          # Create rating-based features
        if 'rating' in self.data.columns:
            self.data['rating_category'] = pd.cut(
                self.data['rating'], 
                bins=[0, 3, 4, 4.5, 5], 
                labels=['Poor', 'Good', 'Very Good', 'Excellent'],
                include_lowest=True
            )
            # Handle missing values by adding 'Unknown' to categories first
            if self.data['rating_category'].isnull().any():
                self.data['rating_category'] = self.data['rating_category'].cat.add_categories(['Unknown'])
                self.data['rating_category'] = self.data['rating_category'].fillna('Unknown')
            
            self.data['rating_category_encoded'] = LabelEncoder().fit_transform(
                self.data['rating_category'].astype(str)
            )
        
        # Time-based features
        if 'scraped_at' in self.data.columns:
            self.data['scraped_at'] = pd.to_datetime(self.data['scraped_at'])
            self.data['hour'] = self.data['scraped_at'].dt.hour
            self.data['day_of_week'] = self.data['scraped_at'].dt.dayofweek
            self.data['month'] = self.data['scraped_at'].dt.month
            self.data['is_weekend'] = (self.data['day_of_week'] >= 5).astype(int)
        
        logger.info(f"Feature engineering completed. Dataset shape: {self.data.shape}")
    
    def _clean_data_for_ml(self) -> pd.DataFrame:
        """Clean data specifically for ML to avoid encoder issues"""
        df = self.data.copy()
        
        # Handle columns that might contain lists or complex objects
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for lists, tuples, or other non-string objects
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Convert lists/tuples to strings
                    if any(isinstance(x, (list, tuple)) for x in sample_values):
                        df[col] = df[col].apply(
                            lambda x: ', '.join(map(str, x)) if isinstance(x, (list, tuple))
                            else str(x) if pd.notna(x) else ''
                        )
                    # Convert dicts to strings
                    elif any(isinstance(x, dict) for x in sample_values):
                        df[col] = df[col].apply(
                            lambda x: str(x) if isinstance(x, dict)
                            else str(x) if pd.notna(x) else ''
                        )
                    # Ensure all values are strings
                    else:
                        df[col] = df[col].astype(str).fillna('')
        
        return df
    
    def price_prediction_models(self) -> Dict[str, Any]:
        """Build multiple price prediction models"""
        if self.target_column not in self.data.columns:
            logger.error(f"Target column '{self.target_column}' not found")
            return {}
        
        # Select features for modeling
        feature_columns = [
            col for col in self.data.columns 
            if col.endswith('_encoded') or col in [
                'title_length', 'word_count', 'rating', 'reviews_count',
                'hour', 'day_of_week', 'month', 'is_weekend',
                'discount_percentage', 'has_discount', 'has_brand_in_title'
            ]
        ]
        
        # Remove columns with all NaN or constant values
        feature_columns = [
            col for col in feature_columns 
            if col in self.data.columns and 
            not self.data[col].isna().all() and 
            self.data[col].nunique() > 1
        ]
        
        if not feature_columns:
            logger.warning("No suitable features found for modeling")
            return {}
        
        # Prepare data
        X = self.data[feature_columns].fillna(0)
        y = self.data[self.target_column].dropna()
        
        # Align X and y
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) < 10:
            logger.warning("Insufficient data for modeling")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'catboost': CatBoostRegressor(iterations=100, random_state=42, verbose=False)
        }
        
        model_results = {}
        
        for name, model in models.items():
            try:
                logger.info(f"Training {name} model...")
                
                # Train model
                if name in ['linear_regression', 'ridge', 'lasso', 'elastic_net']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Cross-validation
                if name in ['linear_regression', 'ridge', 'lasso', 'elastic_net']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                              scoring='r2', n_jobs=-1)
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                              scoring='r2', n_jobs=-1)
                
                model_results[name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'predictions': y_pred.tolist()
                }
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': feature_columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    self.feature_importance[name] = importance_df.to_dict()
                
                # Store model
                self.models[f'price_prediction_{name}'] = {
                    'model': model,
                    'scaler': scaler if name in ['linear_regression', 'ridge', 'lasso', 'elastic_net'] else None,
                    'features': feature_columns
                }
                
                logger.info(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.2f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        self.results['price_prediction'] = model_results
        return model_results
    
    def demand_forecasting(self) -> Dict[str, Any]:
        """Time series forecasting for product demand"""
        if 'scraped_at' not in self.data.columns:
            logger.warning("No temporal data available for forecasting")
            return {}
        
        # Prepare time series data
        self.data['scraped_date'] = pd.to_datetime(self.data['scraped_at']).dt.date
        
        # Daily product counts
        daily_counts = self.data.groupby('scraped_date').size().reset_index(name='count')
        daily_counts['ds'] = pd.to_datetime(daily_counts['scraped_date'])
        daily_counts['y'] = daily_counts['count']
        
        if len(daily_counts) < 10:
            logger.warning("Insufficient time series data for forecasting")
            return {}
        
        forecasting_results = {}
        
        try:
            # Prophet forecasting
            logger.info("Training Prophet forecasting model...")
            
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            model.fit(daily_counts[['ds', 'y']])
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=7)  # 7 days ahead
            forecast = model.predict(future)
            
            # Calculate metrics on training data
            train_predictions = forecast['yhat'][:len(daily_counts)]
            mape = mean_absolute_percentage_error(daily_counts['y'], train_predictions)
            mae = mean_absolute_error(daily_counts['y'], train_predictions)
            
            forecasting_results['prophet'] = {
                'mape': mape,
                'mae': mae,
                'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).to_dict(),
                'components': model.predict(future)[['ds', 'trend', 'weekly']].tail(14).to_dict()
            }
            
            self.models['demand_forecasting_prophet'] = model
            
            logger.info(f"Prophet forecasting - MAPE: {mape:.4f}, MAE: {mae:.2f}")
            
        except Exception as e:
            logger.error(f"Error in Prophet forecasting: {e}")
        
        # Category-specific forecasting
        if 'category' in self.data.columns:
            top_categories = self.data['category'].value_counts().head(3).index
            
            for category in top_categories:
                try:
                    category_data = self.data[self.data['category'] == category]
                    category_daily = category_data.groupby('scraped_date').size().reset_index(name='count')
                    
                    if len(category_daily) < 5:
                        continue
                    
                    category_daily['ds'] = pd.to_datetime(category_daily['scraped_date'])
                    category_daily['y'] = category_daily['count']
                    
                    category_model = Prophet(
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=False
                    )
                    
                    category_model.fit(category_daily[['ds', 'y']])
                    
                    future_cat = category_model.make_future_dataframe(periods=7)
                    forecast_cat = category_model.predict(future_cat)
                    
                    forecasting_results[f'category_{category}'] = {
                        'forecast': forecast_cat[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).to_dict()
                    }
                    
                    self.models[f'demand_forecasting_{category}'] = category_model
                    
                except Exception as e:
                    logger.error(f"Error forecasting category {category}: {e}")
                    continue
        
        self.results['demand_forecasting'] = forecasting_results
        return forecasting_results
    
    def customer_segmentation_ml(self) -> Dict[str, Any]:
        """Advanced ML-based customer/product segmentation"""
        # Select features for clustering
        clustering_features = []
        
        # Price-related features
        if 'price' in self.data.columns:
            clustering_features.append('price')
        
        # Rating and reviews
        if 'rating' in self.data.columns:
            clustering_features.append('rating')
        if 'reviews_count' in self.data.columns:
            clustering_features.append('reviews_count')
        
        # Text features
        if 'title_length' in self.data.columns:
            clustering_features.append('title_length')
        if 'word_count' in self.data.columns:
            clustering_features.append('word_count')
        
        # Discount features
        if 'discount_percentage' in self.data.columns:
            clustering_features.append('discount_percentage')
        
        if len(clustering_features) < 2:
            logger.warning("Insufficient features for clustering")
            return {}
        
        # Prepare clustering data
        cluster_data = self.data[clustering_features].dropna()
        
        if len(cluster_data) < 10:
            logger.warning("Insufficient data for clustering")
            return {}
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        clustering_results = {}
        
        # K-Means clustering with optimal k selection
        logger.info("Performing K-Means clustering...")
        
        k_range = range(2, min(11, len(cluster_data)//5))
        silhouette_scores = []
        calinski_scores = []
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            calinski_avg = calinski_harabasz_score(scaled_data, cluster_labels)
            
            silhouette_scores.append(silhouette_avg)
            calinski_scores.append(calinski_avg)
            inertias.append(kmeans.inertia_)
        
        # Select optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Final K-Means with optimal k
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(scaled_data)
        
        # Analyze clusters
        cluster_analysis = cluster_data.copy()
        cluster_analysis['cluster'] = cluster_labels
        
        cluster_summary = cluster_analysis.groupby('cluster').agg({
            col: ['mean', 'median', 'std', 'count'] for col in clustering_features
        }).round(2)
        
        clustering_results['kmeans'] = {
            'optimal_k': optimal_k,
            'silhouette_score': silhouette_scores[optimal_k - 2],
            'calinski_score': calinski_scores[optimal_k - 2],
            'cluster_summary': cluster_summary.to_dict(),
            'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict()
        }
        
        # DBSCAN for outlier detection
        logger.info("Performing DBSCAN clustering...")
        
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(scaled_data)
        
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        clustering_results['dbscan'] = {
            'n_clusters': n_clusters_dbscan,
            'n_outliers': n_noise,
            'outlier_percentage': (n_noise / len(dbscan_labels)) * 100
        }
        
        # Hierarchical clustering
        logger.info("Performing Hierarchical clustering...")
        
        hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
        hierarchical_labels = hierarchical.fit_predict(scaled_data)
        
        hierarchical_silhouette = silhouette_score(scaled_data, hierarchical_labels)
        
        clustering_results['hierarchical'] = {
            'n_clusters': optimal_k,
            'silhouette_score': hierarchical_silhouette,
            'cluster_sizes': pd.Series(hierarchical_labels).value_counts().to_dict()
        }
        
        # Store models and data
        self.models['clustering_kmeans'] = final_kmeans
        self.models['clustering_scaler'] = scaler
        self.models['clustering_dbscan'] = dbscan
        self.models['clustering_hierarchical'] = hierarchical
        
        # Add cluster labels to main data
        cluster_data_with_labels = cluster_data.copy()
        cluster_data_with_labels['kmeans_cluster'] = cluster_labels
        cluster_data_with_labels['dbscan_cluster'] = dbscan_labels
        cluster_data_with_labels['hierarchical_cluster'] = hierarchical_labels
        
        self.predictions['clustering'] = cluster_data_with_labels
        
        self.results['clustering'] = clustering_results
        return clustering_results
    
    def anomaly_detection_ml(self) -> Dict[str, Any]:
        """Advanced ML-based anomaly detection"""
        from sklearn.ensemble import IsolationForest
        from sklearn.covariance import EllipticEnvelope
        from sklearn.svm import OneClassSVM
        
        # Select numerical features
        numerical_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if it's in features
        if self.target_column in numerical_features:
            numerical_features.remove(self.target_column)
        
        if len(numerical_features) < 2:
            logger.warning("Insufficient numerical features for anomaly detection")
            return {}
        
        # Prepare data
        anomaly_data = self.data[numerical_features].dropna()
        
        if len(anomaly_data) < 10:
            logger.warning("Insufficient data for anomaly detection")
            return {}
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(anomaly_data)
        
        anomaly_results = {}
        
        # Isolation Forest
        logger.info("Running Isolation Forest anomaly detection...")
        
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        isolation_anomalies = isolation_forest.fit_predict(scaled_data)
        
        n_anomalies_if = (isolation_anomalies == -1).sum()
        anomaly_results['isolation_forest'] = {
            'n_anomalies': int(n_anomalies_if),
            'anomaly_percentage': (n_anomalies_if / len(scaled_data)) * 100,
            'anomaly_scores': isolation_forest.decision_function(scaled_data).tolist()
        }
        
        # Elliptic Envelope
        logger.info("Running Elliptic Envelope anomaly detection...")
        
        try:
            elliptic_envelope = EllipticEnvelope(contamination=0.1, random_state=42)
            elliptic_anomalies = elliptic_envelope.fit_predict(scaled_data)
            
            n_anomalies_ee = (elliptic_anomalies == -1).sum()
            anomaly_results['elliptic_envelope'] = {
                'n_anomalies': int(n_anomalies_ee),
                'anomaly_percentage': (n_anomalies_ee / len(scaled_data)) * 100
            }
        except Exception as e:
            logger.warning(f"Elliptic Envelope failed: {e}")
        
        # One-Class SVM
        logger.info("Running One-Class SVM anomaly detection...")
        
        try:
            one_class_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
            svm_anomalies = one_class_svm.fit_predict(scaled_data)
            
            n_anomalies_svm = (svm_anomalies == -1).sum()
            anomaly_results['one_class_svm'] = {
                'n_anomalies': int(n_anomalies_svm),
                'anomaly_percentage': (n_anomalies_svm / len(scaled_data)) * 100
            }
        except Exception as e:
            logger.warning(f"One-Class SVM failed: {e}")
        
        # Store models
        self.models['anomaly_isolation_forest'] = isolation_forest
        self.models['anomaly_scaler'] = scaler
        
        if 'elliptic_envelope' in anomaly_results:
            self.models['anomaly_elliptic_envelope'] = elliptic_envelope
        if 'one_class_svm' in anomaly_results:
            self.models['anomaly_one_class_svm'] = one_class_svm
        
        self.results['anomaly_detection'] = anomaly_results
        return anomaly_results
    
    def hyperparameter_optimization(self, model_type: str = 'xgboost', n_trials: int = 50) -> Dict[str, Any]:
        """Hyperparameter optimization using Optuna"""
        if self.target_column not in self.data.columns:
            logger.error(f"Target column '{self.target_column}' not found")
            return {}
        
        # Prepare data (same as in price_prediction_models)
        feature_columns = [
            col for col in self.data.columns 
            if col.endswith('_encoded') or col in [
                'title_length', 'word_count', 'rating', 'reviews_count',
                'hour', 'day_of_week', 'month', 'is_weekend',
                'discount_percentage', 'has_discount', 'has_brand_in_title'
            ]
        ]
        
        feature_columns = [
            col for col in feature_columns 
            if col in self.data.columns and 
            not self.data[col].isna().all() and 
            self.data[col].nunique() > 1
        ]
        
        if not feature_columns:
            logger.warning("No suitable features found for optimization")
            return {}
        
        X = self.data[feature_columns].fillna(0)
        y = self.data[self.target_column].dropna()
        
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) < 10:
            logger.warning("Insufficient data for optimization")
            return {}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**params)
                
            elif model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
            return cv_scores.mean()
        
        logger.info(f"Starting hyperparameter optimization for {model_type}...")
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Train best model
        best_params = study.best_params
        best_params['random_state'] = 42
        
        if model_type == 'xgboost':
            best_model = xgb.XGBRegressor(**best_params)
        elif model_type == 'lightgbm':
            best_params['verbose'] = -1
            best_model = lgb.LGBMRegressor(**best_params)
        elif model_type == 'random_forest':
            best_model = RandomForestRegressor(**best_params)
        
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        # Calculate final metrics
        final_metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        optimization_results = {
            'model_type': model_type,
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': n_trials,
            'final_metrics': final_metrics,
            'optimization_history': [trial.value for trial in study.trials if trial.value is not None]
        }
        
        # Store optimized model
        self.models[f'optimized_{model_type}'] = {
            'model': best_model,
            'features': feature_columns,
            'optimization_results': optimization_results
        }
        
        logger.info(f"Optimization completed. Best R²: {study.best_value:.4f}")
        
        self.results['hyperparameter_optimization'] = optimization_results
        return optimization_results
    
    def model_ensemble(self) -> Dict[str, Any]:
        """Create ensemble models for improved predictions"""
        if not self.models:
            logger.warning("No trained models available for ensemble")
            return {}
        
        # Get price prediction models
        price_models = {
            name: model_data for name, model_data in self.models.items()
            if name.startswith('price_prediction_') and 'model' in model_data
        }
        
        if len(price_models) < 2:
            logger.warning("Need at least 2 models for ensemble")
            return {}
        
        # Prepare data
        feature_columns = [
            col for col in self.data.columns 
            if col.endswith('_encoded') or col in [
                'title_length', 'word_count', 'rating', 'reviews_count',
                'hour', 'day_of_week', 'month', 'is_weekend',
                'discount_percentage', 'has_discount', 'has_brand_in_title'
            ]
        ]
        
        feature_columns = [
            col for col in feature_columns 
            if col in self.data.columns and 
            not self.data[col].isna().all() and 
            self.data[col].nunique() > 1
        ]
        
        X = self.data[feature_columns].fillna(0)
        y = self.data[self.target_column].dropna()
        
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create ensemble models
        ensemble_results = {}
        
        try:
            # Voting Regressor
            estimators = []
            for name, model_data in list(price_models.items())[:5]:  # Use top 5 models
                model_name = name.replace('price_prediction_', '')
                estimators.append((model_name, model_data['model']))
            
            voting_regressor = VotingRegressor(estimators=estimators)
            voting_regressor.fit(X_train, y_train)
            
            y_pred_voting = voting_regressor.predict(X_test)
            
            ensemble_results['voting_regressor'] = {
                'r2': r2_score(y_test, y_pred_voting),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_voting)),
                'mae': mean_absolute_error(y_test, y_pred_voting),
                'n_models': len(estimators)
            }
            
            self.models['ensemble_voting'] = voting_regressor
            
        except Exception as e:
            logger.error(f"Error creating voting ensemble: {e}")
        
        # Simple averaging ensemble
        try:
            predictions = []
            for name, model_data in price_models.items():
                model = model_data['model']
                if model_data.get('scaler'):
                    scaler = model_data['scaler']
                    X_test_scaled = scaler.transform(X_test)
                    pred = model.predict(X_test_scaled)
                else:
                    pred = model.predict(X_test)
                predictions.append(pred)
            
            # Average predictions
            y_pred_avg = np.mean(predictions, axis=0)
            
            ensemble_results['simple_average'] = {
                'r2': r2_score(y_test, y_pred_avg),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_avg)),
                'mae': mean_absolute_error(y_test, y_pred_avg),
                'n_models': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error creating simple average ensemble: {e}")
        
        self.results['ensemble'] = ensemble_results
        return ensemble_results
    
    def save_models_and_results(self, output_dir: str = "ml_models"):
        """Save all models and results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save results
        results_file = output_path / "ml_analytics_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self._convert_numpy_types(self.results), f, indent=2, default=str)
        
        # Save feature importance
        if self.feature_importance:
            importance_file = output_path / "feature_importance.json"
            with open(importance_file, 'w', encoding='utf-8') as f:
                json.dump(self.feature_importance, f, indent=2)
        
        # Save models
        models_dir = output_path / "trained_models"
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model_data in self.models.items():
            try:
                if isinstance(model_data, dict) and 'model' in model_data:
                    # Save model and metadata
                    model_file = models_dir / f"{model_name}.joblib"
                    joblib.dump(model_data, model_file)
                else:
                    # Save simple model
                    model_file = models_dir / f"{model_name}.joblib"
                    joblib.dump(model_data, model_file)
            except Exception as e:
                logger.error(f"Error saving model {model_name}: {e}")
        
        # Save predictions
        if self.predictions:
            predictions_dir = output_path / "predictions"
            predictions_dir.mkdir(exist_ok=True)
            
            for pred_name, pred_data in self.predictions.items():
                pred_file = predictions_dir / f"{pred_name}_predictions.csv"
                if isinstance(pred_data, pd.DataFrame):
                    pred_data.to_csv(pred_file, index=False)
        
        logger.info(f"Models and results saved to {output_path}")
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
            # Convert tuple keys to strings for JSON compatibility
            return {str(key): self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)  # Convert tuples to lists
        else:
            return obj
    
    def generate_ml_report(self) -> Dict[str, Any]:
        """Generate comprehensive ML analytics report"""
        logger.info("Running comprehensive ML analytics...")
        
        # Run all ML analyses
        price_results = self.price_prediction_models()
        forecasting_results = self.demand_forecasting()
        clustering_results = self.customer_segmentation_ml()
        anomaly_results = self.anomaly_detection_ml()
        
        # Try hyperparameter optimization
        try:
            optimization_results = self.hyperparameter_optimization(n_trials=20)
        except Exception as e:
            logger.warning(f"Hyperparameter optimization failed: {e}")
            optimization_results = {}
        
        # Try ensemble
        try:
            ensemble_results = self.model_ensemble()
        except Exception as e:
            logger.warning(f"Ensemble creation failed: {e}")
            ensemble_results = {}
        
        # Generate insights
        insights = {
            'model_performance': {},
            'data_insights': {},
            'recommendations': []
        }
        
        # Best model performance
        if price_results:
            best_model = max(price_results.items(), key=lambda x: x[1].get('r2', 0))
            insights['model_performance']['best_model'] = {
                'name': best_model[0],
                'r2_score': best_model[1]['r2'],
                'rmse': best_model[1]['rmse']
            }
        
        # Clustering insights
        if clustering_results:
            insights['data_insights']['optimal_clusters'] = clustering_results.get('kmeans', {}).get('optimal_k', 'N/A')
            insights['data_insights']['outlier_percentage'] = clustering_results.get('dbscan', {}).get('outlier_percentage', 'N/A')
        
        # Generate recommendations
        if price_results:
            avg_r2 = np.mean([result.get('r2', 0) for result in price_results.values()])
            if avg_r2 > 0.8:
                insights['recommendations'].append("High model accuracy achieved - suitable for price prediction")
            elif avg_r2 > 0.6:
                insights['recommendations'].append("Moderate model accuracy - consider feature engineering")
            else:
                insights['recommendations'].append("Low model accuracy - collect more data or different features")
        
        if anomaly_results:
            for method, result in anomaly_results.items():
                anomaly_pct = result.get('anomaly_percentage', 0)
                if anomaly_pct > 15:
                    insights['recommendations'].append(f"High anomaly rate ({anomaly_pct:.1f}%) detected - investigate data quality")
        
        report = {
            'summary': {
                'total_models_trained': len([k for k in self.models.keys() if 'model' in str(self.models[k])]),
                'data_points_analyzed': len(self.data),
                'features_engineered': len([col for col in self.data.columns if col.endswith('_encoded')]),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'results': self.results,
            'insights': insights,
            'model_inventory': list(self.models.keys())
        }
        
        return report

def main():
    """Example usage"""
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'title': [f'Product {i}' for i in range(1000)],
        'price': np.random.lognormal(4, 1, 1000),
        'rating': np.random.normal(4, 0.5, 1000),
        'reviews_count': np.random.poisson(50, 1000),
        'brand': np.random.choice(['Brand A', 'Brand B', 'Brand C'], 1000),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home'], 1000),
        'scraped_at': pd.date_range('2024-01-01', periods=1000, freq='1H')
    })
    
    # Initialize ML Analytics
    ml_analytics = MLAnalytics(sample_data, target_column='price')
    
    # Generate comprehensive report
    report = ml_analytics.generate_ml_report()
    
    print("ML Analytics Summary:")
    print(f"Models trained: {report['summary']['total_models_trained']}")
    print(f"Data points: {report['summary']['data_points_analyzed']}")
    
    if 'best_model' in report['insights']['model_performance']:
        best = report['insights']['model_performance']['best_model']
        print(f"Best model: {best['name']} (R²: {best['r2_score']:.4f})")

if __name__ == "__main__":
    main()
