"""
Advanced Recommendation System
Product recommendation engine using collaborative filtering and content-based methods
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
from pathlib import Path

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# NLP
import re
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Advanced recommendation system for e-commerce products"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize recommendation engine
        
        Args:
            data: Product data DataFrame
        """
        self.data = data.copy()
        self.models = {}
        self.similarity_matrices = {}
        self.user_item_matrix = None
        self.item_features = None
        self.recommendations_cache = {}
        
        # Preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess data for recommendations"""
        logger.info("Preprocessing data for recommendations...")
        
        # Create product ID if not exists
        if 'product_id' not in self.data.columns:
            self.data['product_id'] = range(len(self.data))
        
        # Clean text fields
        text_fields = ['title', 'description', 'brand', 'category']
        for field in text_fields:
            if field in self.data.columns:
                self.data[field] = self.data[field].fillna('').astype(str)
                self.data[f'{field}_clean'] = self.data[field].str.lower().str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        
        # Create combined text for content-based filtering
        text_cols = [f'{field}_clean' for field in text_fields if f'{field}_clean' in self.data.columns]
        if text_cols:
            self.data['combined_text'] = self.data[text_cols].apply(
                lambda x: ' '.join(x.values.astype(str)), axis=1
            )
        
        # Normalize numerical features
        numerical_features = ['price', 'rating', 'reviews_count']
        available_features = [f for f in numerical_features if f in self.data.columns]
        
        if available_features:
            scaler = MinMaxScaler()
            for feature in available_features:
                self.data[f'{feature}_normalized'] = scaler.fit_transform(
                    self.data[[feature]].fillna(self.data[feature].median())
                )
        
        logger.info(f"Preprocessed {len(self.data)} products")
    
    def build_content_based_model(self) -> Dict[str, Any]:
        """Build content-based recommendation model"""
        logger.info("Building content-based recommendation model...")
        
        if 'combined_text' not in self.data.columns:
            logger.warning("No text content available for content-based filtering")
            return {}
        
        try:
            # TF-IDF Vectorization
            tfidf = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = tfidf.fit_transform(self.data['combined_text'])
            
            # Compute cosine similarity
            content_similarity = cosine_similarity(tfidf_matrix)
            
            # Store model components
            self.models['content_tfidf'] = tfidf
            self.similarity_matrices['content_similarity'] = content_similarity
            
            # Feature importance analysis
            feature_names = tfidf.get_feature_names_out()
            tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
            feature_importance = dict(zip(feature_names, tfidf_scores))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:50]
            
            results = {
                'model_type': 'content_based',
                'similarity_matrix_shape': content_similarity.shape,
                'n_features': len(feature_names),
                'top_features': top_features,
                'sparsity': (tfidf_matrix != 0).mean()
            }
            
            logger.info(f"Content-based model built successfully with {len(feature_names)} features")
            return results
            
        except Exception as e:
            logger.error(f"Error building content-based model: {e}")
            return {}
    
    def build_collaborative_model(self) -> Dict[str, Any]:
        """Build collaborative filtering model using product similarities"""
        logger.info("Building collaborative filtering model...")
        
        try:
            # Create feature matrix for collaborative filtering
            feature_columns = []
            
            # Add numerical features
            numerical_features = ['price_normalized', 'rating_normalized', 'reviews_count_normalized']
            available_numerical = [f for f in numerical_features if f in self.data.columns]
            feature_columns.extend(available_numerical)
            
            # Add categorical features (one-hot encoded)
            categorical_features = ['brand', 'category']
            for feature in categorical_features:
                if feature in self.data.columns:
                    # Get top categories to avoid too many features
                    top_values = self.data[feature].value_counts().head(20).index
                    for value in top_values:
                        feature_name = f'{feature}_{value}'
                        self.data[feature_name] = (self.data[feature] == value).astype(int)
                        feature_columns.append(feature_name)
            
            if not feature_columns:
                logger.warning("No suitable features for collaborative filtering")
                return {}
            
            # Create feature matrix
            feature_matrix = self.data[feature_columns].fillna(0).values
            
            # Compute item-item similarity
            item_similarity = cosine_similarity(feature_matrix)
            
            # SVD for dimensionality reduction
            svd = TruncatedSVD(n_components=min(50, len(feature_columns)-1), random_state=42)
            reduced_features = svd.fit_transform(feature_matrix)
            
            # Store models
            self.models['collaborative_svd'] = svd
            self.similarity_matrices['item_similarity'] = item_similarity
            self.item_features = reduced_features
            
            results = {
                'model_type': 'collaborative_filtering',
                'similarity_matrix_shape': item_similarity.shape,
                'n_components': svd.n_components,
                'explained_variance_ratio': svd.explained_variance_ratio_.sum(),
                'feature_columns': feature_columns
            }
            
            logger.info(f"Collaborative model built with {len(feature_columns)} features")
            return results
            
        except Exception as e:
            logger.error(f"Error building collaborative model: {e}")
            return {}
    
    def build_hybrid_model(self) -> Dict[str, Any]:
        """Build hybrid recommendation model combining content and collaborative"""
        logger.info("Building hybrid recommendation model...")
        
        try:
            content_results = self.build_content_based_model()
            collaborative_results = self.build_collaborative_model()
            
            if not content_results and not collaborative_results:
                logger.error("Failed to build any recommendation models")
                return {}
            
            # Combine similarities if both models exist
            if ('content_similarity' in self.similarity_matrices and 
                'item_similarity' in self.similarity_matrices):
                
                content_sim = self.similarity_matrices['content_similarity']
                item_sim = self.similarity_matrices['item_similarity']
                
                # Weighted combination (can be tuned)
                content_weight = 0.6
                collaborative_weight = 0.4
                
                hybrid_similarity = (content_weight * content_sim + 
                                   collaborative_weight * item_sim)
                
                self.similarity_matrices['hybrid_similarity'] = hybrid_similarity
                
                results = {
                    'model_type': 'hybrid',
                    'content_weight': content_weight,
                    'collaborative_weight': collaborative_weight,
                    'content_results': content_results,
                    'collaborative_results': collaborative_results,
                    'hybrid_similarity_shape': hybrid_similarity.shape
                }
                
                logger.info("Hybrid model built successfully")
                return results
            
            else:
                # Use available model
                if content_results:
                    logger.info("Using content-based model only")
                    return content_results
                else:
                    logger.info("Using collaborative model only")
                    return collaborative_results
            
        except Exception as e:
            logger.error(f"Error building hybrid model: {e}")
            return {}
    
    def get_similar_products(self, product_id: int, n_recommendations: int = 10, 
                           method: str = 'hybrid') -> List[Dict[str, Any]]:
        """
        Get similar products for a given product
        
        Args:
            product_id: ID of the product to find similarities for
            n_recommendations: Number of recommendations to return
            method: 'content', 'collaborative', or 'hybrid'
            
        Returns:
            List of recommended products with similarity scores
        """
        if product_id not in self.data['product_id'].values:
            logger.error(f"Product ID {product_id} not found")
            return []
        
        try:
            # Get product index
            product_idx = self.data[self.data['product_id'] == product_id].index[0]
            
            # Select similarity matrix
            similarity_key = f'{method}_similarity'
            if similarity_key not in self.similarity_matrices:
                # Fallback to available method
                available_methods = [key.replace('_similarity', '') for key in self.similarity_matrices.keys()]
                if available_methods:
                    method = available_methods[0]
                    similarity_key = f'{method}_similarity'
                else:
                    logger.error("No similarity matrices available")
                    return []
            
            similarity_matrix = self.similarity_matrices[similarity_key]
            
            # Get similarity scores for the product
            sim_scores = list(enumerate(similarity_matrix[product_idx]))
            
            # Sort by similarity score (excluding the product itself)
            sim_scores = [(i, score) for i, score in sim_scores if i != product_idx]
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top recommendations
            top_indices = [i for i, _ in sim_scores[:n_recommendations]]
            
            recommendations = []
            for idx in top_indices:
                product_data = self.data.iloc[idx]
                similarity_score = sim_scores[top_indices.index(idx)][1]
                
                recommendation = {
                    'product_id': product_data['product_id'],
                    'title': product_data.get('title', 'Unknown'),
                    'brand': product_data.get('brand', 'Unknown'),
                    'category': product_data.get('category', 'Unknown'),
                    'price': product_data.get('price', 0),
                    'rating': product_data.get('rating', 0),
                    'similarity_score': float(similarity_score),
                    'method': method
                }
                
                # Add URL if available
                if 'url' in product_data and pd.notna(product_data['url']):
                    recommendation['url'] = product_data['url']
                
                recommendations.append(recommendation)
            
            # Cache recommendations
            cache_key = f"{product_id}_{method}_{n_recommendations}"
            self.recommendations_cache[cache_key] = {
                'recommendations': recommendations,
                'timestamp': datetime.now(),
                'method': method
            }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting similar products: {e}")
            return []
    
    def get_category_recommendations(self, category: str, n_recommendations: int = 10,
                                   sort_by: str = 'rating') -> List[Dict[str, Any]]:
        """
        Get top products in a specific category
        
        Args:
            category: Product category
            n_recommendations: Number of recommendations
            sort_by: 'rating', 'price', 'reviews_count'
            
        Returns:
            List of recommended products
        """
        if 'category' not in self.data.columns:
            logger.error("Category information not available")
            return []
        
        try:
            # Filter by category
            category_data = self.data[self.data['category'].str.contains(category, case=False, na=False)]
            
            if category_data.empty:
                logger.warning(f"No products found in category: {category}")
                return []
            
            # Sort by specified criteria
            if sort_by in category_data.columns:
                if sort_by == 'price':
                    # Sort by price ascending
                    sorted_data = category_data.sort_values(sort_by)
                else:
                    # Sort by rating/reviews descending
                    sorted_data = category_data.sort_values(sort_by, ascending=False)
            else:
                # Default sorting by product_id
                sorted_data = category_data.sort_values('product_id')
            
            # Get top recommendations
            top_products = sorted_data.head(n_recommendations)
            
            recommendations = []
            for _, product in top_products.iterrows():
                recommendation = {
                    'product_id': product['product_id'],
                    'title': product.get('title', 'Unknown'),
                    'brand': product.get('brand', 'Unknown'),
                    'category': product.get('category', 'Unknown'),
                    'price': product.get('price', 0),
                    'rating': product.get('rating', 0),
                    'reviews_count': product.get('reviews_count', 0),
                    'sort_criteria': sort_by,
                    'method': 'category_based'
                }
                
                if 'url' in product and pd.notna(product['url']):
                    recommendation['url'] = product['url']
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting category recommendations: {e}")
            return []
    
    def get_price_based_recommendations(self, target_price: float, price_tolerance: float = 0.2,
                                      n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        Get products within a specific price range
        
        Args:
            target_price: Target price
            price_tolerance: Price tolerance (0.2 = ±20%)
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended products
        """
        if 'price' not in self.data.columns:
            logger.error("Price information not available")
            return []
        
        try:
            # Calculate price range
            min_price = target_price * (1 - price_tolerance)
            max_price = target_price * (1 + price_tolerance)
            
            # Filter by price range
            price_filtered = self.data[
                (self.data['price'] >= min_price) & 
                (self.data['price'] <= max_price) &
                (self.data['price'].notna())
            ]
            
            if price_filtered.empty:
                logger.warning(f"No products found in price range: €{min_price:.2f} - €{max_price:.2f}")
                return []
            
            # Sort by rating (or other quality metric)
            if 'rating' in price_filtered.columns:
                sorted_data = price_filtered.sort_values('rating', ascending=False)
            else:
                sorted_data = price_filtered.sort_values('price')
            
            # Get top recommendations
            top_products = sorted_data.head(n_recommendations)
            
            recommendations = []
            for _, product in top_products.iterrows():
                price_diff = abs(product['price'] - target_price) / target_price * 100
                
                recommendation = {
                    'product_id': product['product_id'],
                    'title': product.get('title', 'Unknown'),
                    'brand': product.get('brand', 'Unknown'),
                    'category': product.get('category', 'Unknown'),
                    'price': product.get('price', 0),
                    'rating': product.get('rating', 0),
                    'price_difference_pct': round(price_diff, 2),
                    'target_price': target_price,
                    'method': 'price_based'
                }
                
                if 'url' in product and pd.notna(product['url']):
                    recommendation['url'] = product['url']
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting price-based recommendations: {e}")
            return []
    
    def get_trending_products(self, time_window_days: int = 7, 
                            n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending products based on recent activity
        
        Args:
            time_window_days: Number of days to look back
            n_recommendations: Number of recommendations
            
        Returns:
            List of trending products
        """
        if 'scraped_at' not in self.data.columns:
            logger.warning("Timestamp information not available, using all data")
            recent_data = self.data
        else:
            try:
                # Filter recent data
                self.data['scraped_at'] = pd.to_datetime(self.data['scraped_at'])
                cutoff_date = datetime.now() - pd.Timedelta(days=time_window_days)
                recent_data = self.data[self.data['scraped_at'] >= cutoff_date]
                
                if recent_data.empty:
                    logger.warning("No recent data found, using all data")
                    recent_data = self.data
            except:
                logger.warning("Error processing timestamps, using all data")
                recent_data = self.data
        
        try:
            # Calculate trending score based on multiple factors
            trending_data = recent_data.copy()
            
            # Factors for trending score
            score_factors = []
            
            # Reviews count (popularity)
            if 'reviews_count' in trending_data.columns:
                trending_data['reviews_score'] = trending_data['reviews_count'].fillna(0)
                score_factors.append('reviews_score')
            
            # Rating (quality)
            if 'rating' in trending_data.columns:
                trending_data['rating_score'] = trending_data['rating'].fillna(0) * 20  # Scale to 0-100
                score_factors.append('rating_score')
            
            # Inverse price factor (value for money)
            if 'price' in trending_data.columns:
                max_price = trending_data['price'].max()
                trending_data['value_score'] = (max_price - trending_data['price'].fillna(max_price)) / max_price * 100
                score_factors.append('value_score')
            
            # Recency factor
            if 'scraped_at' in trending_data.columns:
                latest_time = trending_data['scraped_at'].max()
                time_diff = (latest_time - trending_data['scraped_at']).dt.total_seconds() / 3600  # hours
                trending_data['recency_score'] = np.exp(-time_diff / 24) * 100  # Exponential decay
                score_factors.append('recency_score')
            
            # Calculate overall trending score
            if score_factors:
                trending_data['trending_score'] = trending_data[score_factors].mean(axis=1)
            else:
                # Fallback to random selection
                trending_data['trending_score'] = np.random.rand(len(trending_data)) * 100
            
            # Sort by trending score
            top_trending = trending_data.sort_values('trending_score', ascending=False).head(n_recommendations)
            
            recommendations = []
            for _, product in top_trending.iterrows():
                recommendation = {
                    'product_id': product['product_id'],
                    'title': product.get('title', 'Unknown'),
                    'brand': product.get('brand', 'Unknown'),
                    'category': product.get('category', 'Unknown'),
                    'price': product.get('price', 0),
                    'rating': product.get('rating', 0),
                    'reviews_count': product.get('reviews_count', 0),
                    'trending_score': round(product['trending_score'], 2),
                    'method': 'trending'
                }
                
                if 'url' in product and pd.notna(product['url']):
                    recommendation['url'] = product['url']
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting trending products: {e}")
            return []
    
    def generate_recommendation_report(self) -> Dict[str, Any]:
        """Generate comprehensive recommendation system report"""
        logger.info("Generating recommendation system report...")
        
        # Build models
        hybrid_results = self.build_hybrid_model()
        
        # Test recommendations with sample products
        sample_recommendations = {}
        
        if not self.data.empty:
            # Test with first few products
            sample_products = self.data['product_id'].head(5).tolist()
            
            for product_id in sample_products:
                try:
                    # Get recommendations using different methods
                    for method in ['hybrid', 'content', 'collaborative']:
                        if f'{method}_similarity' in self.similarity_matrices:
                            recs = self.get_similar_products(product_id, n_recommendations=5, method=method)
                            if recs:
                                sample_recommendations[f'{product_id}_{method}'] = recs[:3]  # Top 3
                            break
                except Exception as e:
                    logger.warning(f"Error testing recommendations for product {product_id}: {e}")
        
        # Get category statistics
        category_stats = {}
        if 'category' in self.data.columns:
            category_counts = self.data['category'].value_counts()
            category_stats = {
                'total_categories': len(category_counts),
                'top_categories': category_counts.head(10).to_dict(),
                'avg_products_per_category': category_counts.mean()
            }
        
        # Model performance metrics
        model_metrics = {
            'total_products': len(self.data),
            'feature_matrices': {
                name: matrix.shape for name, matrix in self.similarity_matrices.items()
            },
            'models_built': list(self.models.keys()),
            'cache_size': len(self.recommendations_cache)
        }
        
        report = {
            'system_overview': {
                'timestamp': datetime.now().isoformat(),
                'total_products': len(self.data),
                'available_methods': list(self.similarity_matrices.keys()),
                'models_trained': len(self.models)
            },
            'model_results': hybrid_results,
            'model_metrics': model_metrics,
            'category_analysis': category_stats,
            'sample_recommendations': sample_recommendations,
            'data_quality': {
                'text_coverage': (self.data['combined_text'].str.len() > 0).mean() if 'combined_text' in self.data.columns else 0,
                'price_coverage': self.data['price'].notna().mean() if 'price' in self.data.columns else 0,
                'rating_coverage': self.data['rating'].notna().mean() if 'rating' in self.data.columns else 0
            }
        }
        
        return report
    
    def save_models(self, output_dir: str = "recommendation_models"):
        """Save trained recommendation models"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Save models
            for model_name, model in self.models.items():
                model_file = output_path / f"{model_name}.joblib"
                joblib.dump(model, model_file)
            
            # Save similarity matrices (as compressed)
            for sim_name, sim_matrix in self.similarity_matrices.items():
                sim_file = output_path / f"{sim_name}.npz"
                np.savez_compressed(sim_file, similarity_matrix=sim_matrix)
            
            # Save metadata
            metadata = {
                'data_shape': self.data.shape,
                'models': list(self.models.keys()),
                'similarity_matrices': list(self.similarity_matrices.keys()),
                'saved_at': datetime.now().isoformat()
            }
            
            metadata_file = output_path / "recommendation_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Recommendation models saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return None

def main():
    """Example usage"""
    # Load sample data
    data_path = "scraped_data/kompjuter_laptop_monitor_scrape_1748731950.json"
    
    if Path(data_path).exists():
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Initialize recommendation engine
        rec_engine = RecommendationEngine(df)
        
        # Generate report
        report = rec_engine.generate_recommendation_report()
        print(json.dumps(report, indent=2, default=str))
        
        # Test recommendations
        if not df.empty:
            product_id = df['product_id'].iloc[0]
            recommendations = rec_engine.get_similar_products(product_id, n_recommendations=5)
            
            print(f"\nRecommendations for product {product_id}:")
            for rec in recommendations:
                print(f"- {rec['title']} (Score: {rec['similarity_score']:.3f})")
        
        # Save models
        rec_engine.save_models()
    
    else:
        print(f"Data file not found: {data_path}")

if __name__ == "__main__":
    main()
