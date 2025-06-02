"""
Enhanced Real-Time Analytics Dashboard
Comprehensive e-commerce analytics with ML integration and data pipeline execution
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import json
from datetime import datetime, timedelta
import time
from pathlib import Path
import asyncio
import threading
from typing import Dict, List, Any
import pickle
import os
import subprocess
import warnings
import logging
import base64
import io
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
try:
    from advanced_analytics import AdvancedAnalytics
    from ml_analytics import MLAnalytics
    from advanced_visualizer import AdvancedVisualizer
    from data_pipeline import DataPipeline
    from recommendation_engine import RecommendationEngine
    from utils import DataValidator, DataExporter
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

class RealTimeDashboard:
    """Real-time analytics dashboard using Streamlit"""
    
    def __init__(self, data_dir: str = "scraped_data"):
        self.data_dir = Path(data_dir)
        self.cache_duration = 300  # 5 minutes
        self.last_update = None
        self.cached_data = None
        self.analytics = None
        self.ml_analytics = None
        self.visualizer = None
        self.pipeline = None
        self.recommendation_engine = None
        self.pipeline_results = {}
        
        # Create output directories
        self.output_dir = Path("dashboard_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # ML model cache
        self.model_cache = {}

    def load_latest_data(self) -> pd.DataFrame:
        """Load the most recent scraped data"""
        try:
            # Find the latest JSON file
            json_files = list(self.data_dir.glob("*.json"))
            
            if not json_files:
                st.error("No data files found!")
                return pd.DataFrame()
            
            # Sort by modification time and get the latest
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            
            # Fix data types for Arrow compatibility
            df = self._fix_data_types(df)
            
            # Cache the data
            self.cached_data = df
            self.last_update = datetime.now()
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types for Arrow/Streamlit compatibility"""
        try:
            # Convert timestamp columns to proper datetime
            if 'scraped_at' in df.columns:
                df['scraped_at'] = pd.to_datetime(df['scraped_at'])
                
            # Handle list columns that cause ML issues
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains lists
                    if df[col].apply(lambda x: isinstance(x, list)).any():
                        # Convert lists to strings
                        df[col] = df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x) if pd.notna(x) else '')
                    
                    # Clean string columns
                    elif df[col].dtype == 'object':
                        df[col] = df[col].astype(str).replace('nan', '')
            
            # Ensure numeric columns are proper numeric types
            numeric_candidates = ['price', 'rating', 'reviews_count', 'discount_percentage']
            for col in numeric_candidates:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fixing data types: {e}")
            return df
    
    def get_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get data with caching"""
        if (force_refresh or 
            self.cached_data is None or 
            self.last_update is None or 
            (datetime.now() - self.last_update).seconds > self.cache_duration):
            
            return self.load_latest_data()
        
        return self.cached_data
    
    def create_kpi_cards(self, df: pd.DataFrame):
        """Create KPI cards at the top of dashboard"""
        if df.empty:
            return
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Total Products",
                value=f"{len(df):,}",
                delta=f"+{len(df.tail(100))}" if len(df) > 100 else None
            )
        
        with col2:
            if 'price' in df.columns:
                avg_price = df['price'].mean()
                st.metric(
                    label="Avg Price",
                    value=f"€{avg_price:.2f}" if pd.notna(avg_price) else "N/A"
                )
        
        with col3:
            if 'category' in df.columns:
                unique_categories = df['category'].nunique()
                st.metric(
                    label="Categories",
                    value=f"{unique_categories}"
                )
        
        with col4:
            if 'brand' in df.columns:
                unique_brands = df['brand'].nunique()
                st.metric(
                    label="Brands",
                    value=f"{unique_brands}"
                )
        
        with col5:
            if 'rating' in df.columns:
                avg_rating = df['rating'].mean()
                st.metric(
                    label="Avg Rating",
                    value=f"{avg_rating:.1f}⭐" if pd.notna(avg_rating) else "N/A"
                )
    
    def create_price_distribution(self, df: pd.DataFrame):
        """Create price distribution chart"""
        if 'price' not in df.columns or df['price'].isna().all():
            return
        
        st.subheader("💰 Price Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                df, 
                x='price',
                nbins=30,
                title="Price Distribution",
                labels={'price': 'Price (€)', 'count': 'Frequency'}
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot by category
            if 'category' in df.columns:
                # Get top 10 categories
                top_categories = df['category'].value_counts().head(10).index
                df_filtered = df[df['category'].isin(top_categories)]
                
                fig_box = px.box(
                    df_filtered,
                    x='category',
                    y='price',
                    title="Price by Category (Top 10)"
                )
                fig_box.update_xaxes(tickangle=45)
                fig_box.update_layout(height=400)
                st.plotly_chart(fig_box, use_container_width=True)
    
    def create_category_analysis(self, df: pd.DataFrame):
        """Create category analysis charts"""
        if 'category' not in df.columns:
            return
        
        st.subheader("📊 Category Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            category_counts = df['category'].value_counts().head(15)
            
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Product Distribution by Category"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Category performance (if price available)
            if 'price' in df.columns:
                category_stats = df.groupby('category').agg({
                    'price': ['mean', 'count']
                }).round(2)
                
                category_stats.columns = ['avg_price', 'product_count']
                category_stats = category_stats.sort_values('product_count', ascending=False).head(10)
                
                fig_scatter = px.scatter(
                    category_stats.reset_index(),
                    x='product_count',
                    y='avg_price',
                    hover_name='category',
                    title="Category Performance (Count vs Avg Price)",
                    labels={'product_count': 'Number of Products', 'avg_price': 'Average Price (€)'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
    
    def create_brand_analysis(self, df: pd.DataFrame):
        """Create brand analysis charts"""
        if 'brand' not in df.columns:
            return
        
        st.subheader("🏷️ Brand Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top brands by product count
            top_brands = df['brand'].value_counts().head(15)
            
            fig_bar = px.bar(
                x=top_brands.values,
                y=top_brands.index,
                orientation='h',
                title="Top Brands by Product Count",
                labels={'x': 'Number of Products', 'y': 'Brand'}
            )
            fig_bar.update_layout(height=500)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Brand price positioning
            if 'price' in df.columns:
                brand_stats = df.groupby('brand').agg({
                    'price': 'mean',
                    'brand': 'count'
                }).rename(columns={'brand': 'count'})
                
                brand_stats = brand_stats[brand_stats['count'] >= 5].sort_values('price', ascending=False).head(15)
                
                fig_brand_price = px.bar(
                    brand_stats.reset_index(),
                    x='brand',
                    y='price',
                    title="Average Price by Brand (Min 5 products)",
                    labels={'price': 'Average Price (€)', 'brand': 'Brand'}
                )
                fig_brand_price.update_xaxes(tickangle=45)
                fig_brand_price.update_layout(height=500)
                st.plotly_chart(fig_brand_price, use_container_width=True)
    
    def create_time_analysis(self, df: pd.DataFrame):
        """Create time-based analysis"""
        if 'scraped_at' not in df.columns:
            return
        
        st.subheader("⏰ Temporal Analysis")
        
        # Convert to datetime
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        df['date'] = df['scraped_at'].dt.date
        df['hour'] = df['scraped_at'].dt.hour
        df['day_of_week'] = df['scraped_at'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily scraping volume
            daily_counts = df.groupby('date').size().reset_index(name='count')
            
            fig_daily = px.line(
                daily_counts,
                x='date',
                y='count',
                title="Daily Scraping Volume",
                labels={'count': 'Products Scraped', 'date': 'Date'}
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        
        with col2:
            # Hourly pattern
            hourly_counts = df.groupby('hour').size()
            
            fig_hourly = px.bar(
                x=hourly_counts.index,
                y=hourly_counts.values,
                title="Scraping Activity by Hour",
                labels={'x': 'Hour of Day', 'y': 'Products Scraped'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
    
    def create_quality_metrics(self, df: pd.DataFrame):
        """Create data quality metrics"""
        st.subheader("✅ Data Quality Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Completeness by field
            completeness = {}
            important_fields = ['title', 'price', 'brand', 'category', 'description', 'rating']
            
            for field in important_fields:
                if field in df.columns:
                    completeness[field] = (df[field].notna().sum() / len(df)) * 100
            
            if completeness:
                fig_completeness = px.bar(
                    x=list(completeness.keys()),
                    y=list(completeness.values()),
                    title="Data Completeness by Field (%)",
                    labels={'x': 'Field', 'y': 'Completeness (%)'}
                )
                fig_completeness.update_layout(height=400)
                st.plotly_chart(fig_completeness, use_container_width=True)
        with col2:
            # Quality score distribution
            if 'data_quality_score' in df.columns:
                fig_quality = px.histogram(
                    df,
                    x='data_quality_score',
                    title="Data Quality Score Distribution",
                    labels={'data_quality_score': 'Quality Score', 'count': 'Frequency'}
                )
                st.plotly_chart(fig_quality, use_container_width=True)
            else:
                # Calculate simple quality score
                quality_scores = []
                for _, row in df.iterrows():
                    score = 0
                    total_fields = 0
                    for field in important_fields:
                        if field in df.columns:
                            total_fields += 1
                            if pd.notna(row[field]) and str(row[field]).strip() != '':
                                score += 1
                    quality_scores.append((score / total_fields) * 100 if total_fields > 0 else 0)
                
                fig_quality = px.histogram(
                    x=quality_scores,
                    title="Calculated Data Quality Score Distribution",
                    labels={'x': 'Quality Score (%)', 'y': 'Frequency'}
                )
                st.plotly_chart(fig_quality, use_container_width=True)
    
    def create_advanced_insights(self, df: pd.DataFrame):
        """Create advanced analytical insights with full ML and visualization integration"""
        st.subheader("🧠 Advanced Analytics & ML Insights")
        
        # Tabs for different analysis types
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Statistical Analysis", 
            "🤖 Machine Learning", 
            "📈 Visualizations",
            "🎯 Recommendations",
            "🔄 Data Pipeline",
            "📋 Export Results"
        ])
        
        with tab1:
            self._create_statistical_analysis_tab(df)
            
        with tab2:
            self._create_ml_analysis_tab(df)
            
        with tab3:
            self._create_visualization_tab(df)
            
        with tab4:
            self._create_recommendations_tab(df)
            
        with tab5:
            self._create_pipeline_tab(df)
            
        with tab6:
            self._create_export_tab(df)
    
    def _create_statistical_analysis_tab(self, df: pd.DataFrame):
        """Statistical analysis tab content"""
        st.subheader("📊 Statistical Analysis")
        
        if st.button("🔄 Run Statistical Analysis", key="stats_analysis"):
            with st.spinner("Running statistical analysis..."):
                try:
                    # Initialize analytics if not done
                    if self.analytics is None:
                        self.analytics = AdvancedAnalytics(data=df)
                        self.analytics.preprocess_data()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📈 Descriptive Statistics:**")
                        stats_results = self.analytics.descriptive_statistics()
                        
                        if 'price_analysis' in stats_results:
                            price_stats = stats_results['price_analysis']
                            st.metric("Average Price", f"€{price_stats.get('mean', 0):.2f}")
                            st.metric("Price Std Dev", f"€{price_stats.get('std', 0):.2f}")
                            st.metric("Price Skewness", f"{price_stats.get('skewness', 0):.2f}")
                            st.metric("Price Kurtosis", f"{price_stats.get('kurtosis', 0):.2f}")
                        
                        # Distribution analysis
                        st.write("**📊 Distribution Analysis:**")
                        if 'price' in df.columns:
                            fig_dist = px.histogram(df, x='price', nbins=50, 
                                                  title="Price Distribution with Normal Curve")
                            st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with col2:
                        st.write("**🔍 Advanced Insights:**")
                        
                        # Correlation analysis
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 1:
                            correlation_matrix = df[numeric_cols].corr()
                            
                            fig_corr = px.imshow(correlation_matrix, 
                                               text_auto=True, 
                                               aspect="auto",
                                               title="Correlation Matrix")
                            st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Outlier detection
                        if 'price' in df.columns:
                            outliers = self.analytics.detect_outliers()
                            if outliers and 'price_outliers' in outliers:
                                n_outliers = len(outliers['price_outliers'])
                                st.metric("Price Outliers", f"{n_outliers} ({n_outliers/len(df)*100:.1f}%)")
                
                except Exception as e:
                    st.error(f"Error in statistical analysis: {e}")
    
    def _create_ml_analysis_tab(self, df: pd.DataFrame):
        """Machine Learning analysis tab content"""
        st.subheader("🤖 Machine Learning Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**ML Configuration:**")
            
            # Target selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target_col = st.selectbox("Select Target Column", numeric_cols, 
                                        index=0 if 'price' in numeric_cols else 0)
                
                # ML task type
                ml_task = st.radio("ML Task", ["Regression", "Classification", "Clustering"])
                
                # Model selection
                if ml_task == "Regression":
                    models = ["Random Forest", "Gradient Boosting", "Linear Regression", 
                             "Ridge", "Lasso", "ElasticNet", "SVR"]
                elif ml_task == "Classification":
                    models = ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVC"]
                else:
                    models = ["KMeans", "DBSCAN", "Hierarchical"]
                
                selected_models = st.multiselect("Select Models", models, default=models[:3])
                
                # Feature selection
                feature_cols = [col for col in numeric_cols if col != target_col]
                if feature_cols:
                    selected_features = st.multiselect("Select Features", feature_cols, 
                                                     default=feature_cols[:5])
                
                if st.button("🚀 Run ML Analysis", key="ml_analysis"):
                    self._run_ml_analysis(df, target_col, ml_task, selected_models, selected_features)
        
        with col2:
            if hasattr(self, 'ml_results') and self.ml_results:
                self._display_ml_results()
    
    def _run_ml_analysis(self, df: pd.DataFrame, target_col: str, ml_task: str, 
                        selected_models: List[str], selected_features: List[str]):
        """Run ML analysis with selected parameters"""
        with st.spinner("Training ML models..."):
            try:
                # Initialize ML analytics
                if self.ml_analytics is None:
                    self.ml_analytics = MLAnalytics(df, target_column=target_col)
                
                # Prepare data
                ml_df = df[selected_features + [target_col]].dropna()
                
                if len(ml_df) < 10:
                    st.warning("Not enough data for ML analysis (minimum 10 rows required)")
                    return
                
                self.ml_results = {}
                
                if ml_task == "Regression":
                    # Run regression models
                    if "Random Forest" in selected_models:
                        rf_results = self.ml_analytics.price_prediction_models()
                        if rf_results and 'random_forest' in rf_results:
                            self.ml_results['Random Forest'] = rf_results['random_forest']
                    
                    # Additional models can be added here
                    self.ml_results['task_type'] = 'regression'
                
                elif ml_task == "Clustering":
                    # Run clustering
                    clustering_results = self.ml_analytics.customer_segmentation(ml_df)
                    self.ml_results = clustering_results
                    self.ml_results['task_type'] = 'clustering'
                
                st.success("ML analysis completed!")
                
            except Exception as e:
                st.error(f"Error in ML analysis: {e}")
                logger.error(f"ML analysis error: {e}")
    
    def _display_ml_results(self):
        """Display ML analysis results"""
        st.write("**🎯 ML Results:**")
        
        if self.ml_results.get('task_type') == 'regression':
            for model_name, results in self.ml_results.items():
                if model_name != 'task_type' and isinstance(results, dict):
                    st.write(f"**{model_name}:**")
                    if 'r2' in results:
                        st.metric(f"{model_name} R² Score", f"{results['r2']:.4f}")
                    if 'rmse' in results:
                        st.metric(f"{model_name} RMSE", f"{results['rmse']:.2f}")
                    if 'mae' in results:
                        st.metric(f"{model_name} MAE", f"{results['mae']:.2f}")
        
        elif self.ml_results.get('task_type') == 'clustering':
            if 'silhouette_score' in self.ml_results:
                st.metric("Silhouette Score", f"{self.ml_results['silhouette_score']:.4f}")
            if 'n_clusters' in self.ml_results:
                st.metric("Number of Clusters", f"{self.ml_results['n_clusters']}")
    
    def _create_visualization_tab(self, df: pd.DataFrame):
        """Advanced visualization tab content"""
        st.subheader("📈 Advanced Visualizations")
        
        if st.button("🎨 Generate Advanced Visualizations", key="advanced_viz"):
            with st.spinner("Creating advanced visualizations..."):
                try:
                    # Initialize visualizer
                    if self.visualizer is None:
                        viz_output_dir = self.output_dir / "visualizations"
                        viz_output_dir.mkdir(exist_ok=True)
                        self.visualizer = AdvancedVisualizer(df, str(viz_output_dir))
                    
                    # Generate visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📊 Market Analysis:**")
                        market_plots = self.visualizer.create_market_analysis_plots()
                        if market_plots:
                            for plot_path in market_plots[:2]:  # Show first 2 plots
                                if Path(plot_path).exists():
                                    st.image(plot_path, caption=Path(plot_path).stem)
                    
                    with col2:
                        st.write("**💰 Price Analysis:**")
                        price_plots = self.visualizer.create_price_distribution_plots()
                        if price_plots:
                            for plot_path in price_plots[:2]:  # Show first 2 plots
                                if Path(plot_path).exists():
                                    st.image(plot_path, caption=Path(plot_path).stem)
                    
                    # Quality analysis
                    st.write("**✅ Quality Analysis:**")
                    quality_plots = self.visualizer.create_quality_analysis_plots()
                    if quality_plots:
                        for plot_path in quality_plots[:3]:  # Show first 3 plots
                            if Path(plot_path).exists():
                                st.image(plot_path, caption=Path(plot_path).stem)
                    
                    # Word cloud
                    if 'description' in df.columns or 'title' in df.columns:
                        st.write("**☁️ Word Cloud:**")
                        wordcloud_plots = self.visualizer.create_wordcloud_visualization()
                        if wordcloud_plots:
                            for plot_path in wordcloud_plots:
                                if Path(plot_path).exists():
                                    st.image(plot_path, caption="Product Keywords")
                    
                    st.success("Advanced visualizations generated!")
                    
                except Exception as e:
                    st.error(f"Error generating visualizations: {e}")
                    logger.error(f"Visualization error: {e}")
    
    def _create_recommendations_tab(self, df: pd.DataFrame):
        """Product recommendations tab content"""
        st.subheader("🎯 Product Recommendations")
        
        if st.button("🔮 Build Recommendation System", key="recommendations"):
            with st.spinner("Building recommendation system..."):
                try:
                    # Initialize recommendation engine
                    if self.recommendation_engine is None:
                        self.recommendation_engine = RecommendationEngine(df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🛍️ Category-based Recommendations:**")
                        
                        if 'category' in df.columns:
                            categories = df['category'].dropna().unique()
                            selected_category = st.selectbox("Select Category", categories)
                            
                            if selected_category:
                                category_recs = self.recommendation_engine.get_category_recommendations(
                                    selected_category, n_recommendations=10
                                )
                                
                                if category_recs:
                                    rec_df = pd.DataFrame(category_recs)
                                    st.dataframe(rec_df[['title', 'price', 'rating']][:5])
                    
                    with col2:
                        st.write("**🔗 Similar Products:**")
                        
                        if len(df) > 0:
                            # Select a random product to show similar items
                            sample_product = df.sample(1).iloc[0]
                            st.write(f"**Similar to:** {sample_product.get('title', 'Unknown Product')}")
                            
                            if 'id' in df.columns or df.index.name:
                                product_id = sample_product.get('id', sample_product.name)
                                similar_products = self.recommendation_engine.get_similar_products(
                                    product_id, n_recommendations=5
                                )
                                
                                if similar_products:
                                    similar_df = pd.DataFrame(similar_products)
                                    st.dataframe(similar_df[['title', 'price', 'similarity_score']])
                    
                    st.success("Recommendation system ready!")
                    
                except Exception as e:
                    st.error(f"Error building recommendations: {e}")
                    logger.error(f"Recommendation error: {e}")
    
    def _create_pipeline_tab(self, df: pd.DataFrame):
        """Data pipeline execution tab"""
        st.subheader("🔄 Data Pipeline Execution")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Pipeline Configuration:**")
            
            # Pipeline options
            run_preprocessing = st.checkbox("Data Preprocessing", value=True)
            run_statistical = st.checkbox("Statistical Analysis", value=True)
            run_ml = st.checkbox("ML Analysis", value=True)
            run_visualizations = st.checkbox("Generate Visualizations", value=True)
            
            # Output format
            export_formats = st.multiselect("Export Formats", 
                                           ["JSON", "CSV", "Excel", "HTML Report"],
                                           default=["JSON", "HTML Report"])
            
            if st.button("🚀 Run Complete Pipeline", key="run_pipeline"):
                self._run_complete_pipeline(df, {
                    'preprocessing': run_preprocessing,
                    'statistical': run_statistical,
                    'ml': run_ml,
                    'visualizations': run_visualizations,
                    'export_formats': [fmt.lower() for fmt in export_formats]
                })
        
        with col2:
            if hasattr(self, 'pipeline_results') and self.pipeline_results:
                self._display_pipeline_results()
    
    def _run_complete_pipeline(self, df: pd.DataFrame, config: Dict[str, Any]):
        """Run the complete data pipeline"""
        with st.spinner("Running complete data pipeline..."):
            try:
                # Initialize pipeline
                if self.pipeline is None:
                    pipeline_output_dir = self.output_dir / "pipeline_results"
                    pipeline_output_dir.mkdir(exist_ok=True)
                    self.pipeline = DataPipeline(df, str(pipeline_output_dir))
                
                self.pipeline_results = {}
                
                # Run data quality assessment
                st.write("📊 Assessing data quality...")
                quality_results = self.pipeline.data_quality_assessment()
                self.pipeline_results['data_quality'] = quality_results
                
                if config['preprocessing']:
                    st.write("🔧 Preprocessing data...")
                    processed_df = self.pipeline.data_preprocessing()
                    self.pipeline_results['processed_rows'] = len(processed_df)
                
                if config['statistical']:
                    st.write("📈 Running statistical analysis...")
                    stats_results = self.pipeline.run_statistical_analysis()
                    self.pipeline_results['statistical'] = stats_results
                
                if config['ml']:
                    st.write("🤖 Running ML analysis...")
                    ml_results = self.pipeline.run_ml_analysis()
                    self.pipeline_results['ml'] = ml_results
                
                if config['visualizations']:
                    st.write("📊 Generating visualizations...")
                    viz_results = self.pipeline.run_visualizations()
                    self.pipeline_results['visualizations'] = viz_results
                
                # Generate comprehensive report
                st.write("📋 Generating comprehensive report...")
                report = self.pipeline.generate_comprehensive_report()
                self.pipeline_results['report'] = report
                
                # Export results
                if config['export_formats']:
                    st.write("💾 Exporting results...")
                    export_results = self.pipeline.export_results(config['export_formats'])
                    self.pipeline_results['exports'] = export_results
                
                st.success("✅ Pipeline execution completed!")
                
            except Exception as e:
                st.error(f"Pipeline execution error: {e}")
                logger.error(f"Pipeline error: {e}")
    
    def _display_pipeline_results(self):
        """Display pipeline execution results"""
        st.write("**📊 Pipeline Results:**")
        
        if 'data_quality' in self.pipeline_results:
            quality = self.pipeline_results['data_quality']
            st.metric("Data Quality Score", f"{quality.get('overall_score', 0):.2f}%")
            st.metric("Missing Values", f"{quality.get('missing_percentage', 0):.1f}%")
        
        if 'processed_rows' in self.pipeline_results:
            st.metric("Processed Rows", f"{self.pipeline_results['processed_rows']:,}")
        
        if 'exports' in self.pipeline_results:
            st.write("**📁 Generated Files:**")
            for format_type, file_path in self.pipeline_results['exports'].items():
                if file_path and Path(file_path).exists():
                    st.write(f"• {format_type.upper()}: `{Path(file_path).name}`")
                    
                    # Download button for files
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                        st.download_button(
                            label=f"📥 Download {format_type.upper()}",
                            data=file_data,
                            file_name=Path(file_path).name,
                            mime='application/octet-stream',
                            key=f"download_{format_type}"
                        )
    
    def _create_export_tab(self, df: pd.DataFrame):
        """Export and reporting tab"""
        st.subheader("📋 Export & Reporting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Quick Exports:**")
            
            # Data export options
            export_data = st.radio("Export Data", ["Filtered Data", "All Data", "Summary Statistics"])
            export_format = st.selectbox("Format", ["CSV", "Excel", "JSON"])
            
            if st.button("📥 Export Data", key="export_data"):
                self._export_data(df, export_data, export_format)
        
        with col2:
            st.write("**📈 Custom Reports:**")
            
            # Report options
            include_charts = st.checkbox("Include Charts", value=True)
            include_statistics = st.checkbox("Include Statistics", value=True)
            include_ml_results = st.checkbox("Include ML Results", value=False)
            
            if st.button("📄 Generate Report", key="generate_report"):
                self._generate_custom_report(df, {
                    'charts': include_charts,
                    'statistics': include_statistics,
                    'ml_results': include_ml_results
                })
    
    def _export_data(self, df: pd.DataFrame, data_type: str, format_type: str):
        """Export data in specified format"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_export_{data_type.lower().replace(' ', '_')}_{timestamp}"
            
            if data_type == "Summary Statistics":
                export_df = df.describe()
            else:
                export_df = df
            
            if format_type == "CSV":
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"{filename}.csv",
                    mime="text/csv"
                )
            
            elif format_type == "Excel":
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='Data', index=False)
                
                st.download_button(
                    label="📥 Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"{filename}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            elif format_type == "JSON":
                json_data = export_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"{filename}.json",
                    mime="application/json"
                )
            
            st.success(f"✅ {format_type} export ready for download!")
            
        except Exception as e:
            st.error(f"Export error: {e}")
    
    def _generate_custom_report(self, df: pd.DataFrame, options: Dict[str, bool]):
        """Generate custom HTML report"""
        try:
            # Basic report structure
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>E-commerce Analytics Report</title>
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
                    <h1>📊 E-commerce Analytics Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """
            
            # Add basic statistics
            html_content += f"""
                <div class="section">
                    <h2>📈 Key Metrics</h2>
                    <div class="metric">
                        <strong>Total Products:</strong> {len(df):,}
                    </div>
            """
            
            if 'price' in df.columns:
                html_content += f"""
                    <div class="metric">
                        <strong>Average Price:</strong> €{df['price'].mean():.2f}
                    </div>
                    <div class="metric">
                        <strong>Price Range:</strong> €{df['price'].min():.2f} - €{df['price'].max():.2f}
                    </div>
                """
            
            if 'category' in df.columns:
                html_content += f"""
                    <div class="metric">
                        <strong>Categories:</strong> {df['category'].nunique()}
                    </div>
                """
            
            html_content += "</div>"
            
            # Add statistics table if requested
            if options['statistics']:
                html_content += """
                    <div class="section">
                        <h2>📊 Statistical Summary</h2>
                        """ + df.describe().to_html() + """
                    </div>
                """
            
            html_content += "</body></html>"
            
            # Offer download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📄 Download HTML Report",
                data=html_content,
                file_name=f"analytics_report_{timestamp}.html",
                mime="text/html"
            )
            
            st.success("✅ Custom report generated!")
            
        except Exception as e:
            st.error(f"Report generation error: {e}")
    
    def create_filter_sidebar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create filters in sidebar"""
        st.sidebar.header("🔍 Filters")
        
        # Date range filter
        if 'scraped_at' in df.columns:
            df['scraped_at'] = pd.to_datetime(df['scraped_at'])
            min_date = df['scraped_at'].min().date()
            max_date = df['scraped_at'].max().date()
            
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                df = df[
                    (df['scraped_at'].dt.date >= start_date) &
                    (df['scraped_at'].dt.date <= end_date)
                ]
        
        # Category filter
        if 'category' in df.columns:
            categories = ['All'] + sorted(df['category'].dropna().unique().tolist())
            selected_category = st.sidebar.selectbox("Category", categories)
            
            if selected_category != 'All':
                df = df[df['category'] == selected_category]
        
        # Brand filter
        if 'brand' in df.columns:
            brands = ['All'] + sorted(df['brand'].dropna().unique().tolist())
            selected_brand = st.sidebar.selectbox("Brand", brands)
            
            if selected_brand != 'All':
                df = df[df['brand'] == selected_brand]
        
        # Price range filter
        if 'price' in df.columns and not df['price'].isna().all():
            min_price = float(df['price'].min())
            max_price = float(df['price'].max())
            
            price_range = st.sidebar.slider(
                "Price Range (€)",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price)
            )
            
            df = df[
                (df['price'] >= price_range[0]) &
                (df['price'] <= price_range[1])
            ]
        
        return df
    def run_dashboard(self):
        """Main dashboard runner"""
        st.set_page_config(
            page_title="E-commerce Analytics Dashboard",
            page_icon="🛒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🛒 Real-Time E-commerce Analytics Dashboard")
        st.markdown("---")
        
        # Sidebar navigation
        st.sidebar.title("🔧 Dashboard Controls")
        
        # Auto-refresh option
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 30)
        
        if auto_refresh:
            # Create a placeholder for the refresh
            time.sleep(refresh_interval)
            st.experimental_rerun()
        
        # Manual refresh button
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.cache_data.clear()
                df = self.get_data(force_refresh=True)
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                self.analytics = None
                self.ml_analytics = None
                self.visualizer = None
                self.pipeline = None
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        # Load data
        df = self.get_data()
        
        if df.empty:
            st.error("No data available. Please ensure scraped data exists in the data directory.")
            st.info("💡 **Tip:** Run the scraper first to collect data, then return to this dashboard.")
            
            # Show available files
            json_files = list(self.data_dir.glob("*.json"))
            if json_files:
                st.write("**Available data files:**")
                for file in json_files:
                    st.write(f"• {file.name} ({file.stat().st_size / 1024:.1f} KB)")
            else:
                st.write("No JSON data files found in the data directory.")
            return
        
        # Apply filters
        filtered_df = self.create_filter_sidebar(df)
        
        # Show data info in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📊 Data Information:**")
        st.sidebar.write(f"📁 Total records: {len(df):,}")
        st.sidebar.write(f"🔍 Filtered records: {len(filtered_df):,}")
        st.sidebar.write(f"🕐 Last updated: {self.last_update.strftime('%H:%M:%S') if self.last_update else 'Never'}")
        
        # Data freshness indicator
        if self.last_update:
            time_diff = datetime.now() - self.last_update
            if time_diff.seconds < 300:  # Less than 5 minutes
                st.sidebar.success("🟢 Data is fresh")
            elif time_diff.seconds < 1800:  # Less than 30 minutes
                st.sidebar.warning("🟡 Data is moderate")
            else:
                st.sidebar.error("🔴 Data is stale")
        
        # Quick actions
        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚡ Quick Actions:**")
        
        if st.sidebar.button("📊 Quick Summary", use_container_width=True):
            self._show_quick_summary(filtered_df)
        
        if st.sidebar.button("📈 Export to CSV", use_container_width=True):
            csv_data = filtered_df.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.sidebar.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"dashboard_export_{timestamp}.csv",
                mime="text/csv"
            )
        
        # Main dashboard content
        if not filtered_df.empty:
            # KPI Cards
            self.create_kpi_cards(filtered_df)
            st.markdown("---")
            
            # Main navigation tabs
            main_tab1, main_tab2, main_tab3 = st.tabs([
                "📊 Basic Analytics", 
                "🧠 Advanced Analytics", 
                "⚙️ System Status"
            ])
            
            with main_tab1:
                # Basic charts
                self.create_price_distribution(filtered_df)
                st.markdown("---")
                
                self.create_category_analysis(filtered_df)
                st.markdown("---")
                
                self.create_brand_analysis(filtered_df)
                st.markdown("---")
                
                self.create_time_analysis(filtered_df)
                st.markdown("---")
                
                self.create_quality_metrics(filtered_df)
                
                # Raw data table
                with st.expander("📋 Raw Data Sample"):
                    st.dataframe(filtered_df.head(100))
            
            with main_tab2:
                # Advanced analytics with full ML integration
                self.create_advanced_insights(filtered_df)
            
            with main_tab3:
                self._create_system_status_tab(filtered_df)
        
        else:
            st.warning("No data matches the current filters.")
            st.info("💡 **Tip:** Try adjusting the filters in the sidebar to see more data.")
    
    def _show_quick_summary(self, df: pd.DataFrame):
        """Show a quick summary popup"""
        with st.sidebar:
            st.markdown("**📊 Quick Summary:**")
            st.metric("Records", f"{len(df):,}")
            
            if 'price' in df.columns and not df['price'].isna().all():
                st.metric("Avg Price", f"€{df['price'].mean():.2f}")
                st.metric("Price Range", f"€{df['price'].min():.0f} - €{df['price'].max():.0f}")
            
            if 'category' in df.columns:
                st.metric("Categories", f"{df['category'].nunique()}")
                
                # Top categories
                top_cats = df['category'].value_counts().head(3)
                st.write("**Top Categories:**")
                for cat, count in top_cats.items():
                    st.write(f"• {cat}: {count}")
            
            if 'brand' in df.columns:
                st.metric("Brands", f"{df['brand'].nunique()}")
    
    def _create_system_status_tab(self, df: pd.DataFrame):
        """System status and performance tab"""
        st.subheader("⚙️ System Status & Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**💾 Memory Usage:**")
            
            # Calculate memory usage
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            st.metric("DataFrame Size", f"{memory_usage:.2f} MB")
            
            # Row and column info
            st.metric("Rows", f"{len(df):,}")
            st.metric("Columns", f"{len(df.columns)}")
        
        with col2:
            st.write("**🔧 Module Status:**")
            
            # Check module status
            modules_status = {
                "Advanced Analytics": self.analytics is not None,
                "ML Analytics": self.ml_analytics is not None,
                "Visualizer": self.visualizer is not None,
                "Pipeline": self.pipeline is not None,
                "Recommendations": self.recommendation_engine is not None
            }
            
            for module, status in modules_status.items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} {module}")
        
        with col3:
            st.write("**📁 File System:**")
            
            # Check output directories
            dirs_to_check = [
                ("Output Dir", self.output_dir),
                ("Data Dir", self.data_dir),
                ("Visualizations", self.output_dir / "visualizations"),
                ("Pipeline Results", self.output_dir / "pipeline_results")
            ]
            
            for dir_name, dir_path in dirs_to_check:
                exists = dir_path.exists()
                icon = "✅" if exists else "❌"
                st.write(f"{icon} {dir_name}")
                if exists and dir_path.is_dir():
                    file_count = len(list(dir_path.iterdir()))
                    st.write(f"   📁 {file_count} files")
        
        # Performance metrics
        st.markdown("---")
        st.write("**⚡ Performance Metrics:**")
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            # Data processing speed
            if hasattr(self, 'last_update') and self.last_update:
                processing_time = (datetime.now() - self.last_update).total_seconds()
                st.metric("Time Since Update", f"{processing_time:.0f}s")
            
            # Cache status
            cache_status = "Active" if self.cached_data is not None else "Empty"
            st.metric("Cache Status", cache_status)
        
        with perf_col2:
            # Data quality score
            if 'price' in df.columns:
                price_completeness = (df['price'].notna().sum() / len(df)) * 100
                st.metric("Price Data Quality", f"{price_completeness:.1f}%")
            
            # Missing data percentage
            missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_percentage:.1f}%")
        
        # System actions
        st.markdown("---")
        st.write("**🔧 System Actions:**")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("🧹 Clean Memory", use_container_width=True):
                # Clear caches and reset
                self.analytics = None
                self.ml_analytics = None
                self.visualizer = None
                st.cache_data.clear()
                st.success("Memory cleaned!")
        
        with action_col2:
            if st.button("📊 Reprocess Data", use_container_width=True):
                # Force data reload and reprocessing
                self.cached_data = None
                self.last_update = None
                df_new = self.get_data(force_refresh=True)
                st.success(f"Reprocessed {len(df_new)} records!")
        
        with action_col3:
            if st.button("📁 Create Backup", use_container_width=True):
                # Create backup of current data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = self.output_dir / f"data_backup_{timestamp}.json"
                df.to_json(backup_file, orient='records', indent=2)
                st.success(f"Backup created: {backup_file.name}")
        
        # Debug information (expandable)
        with st.expander("🔍 Debug Information"):
            st.write("**Environment Info:**")
            st.write(f"• Python version: {pd.__version__}")
            st.write(f"• Pandas version: {pd.__version__}")
            st.write(f"• Streamlit version: {st.__version__}")
            
            st.write("**Data Schema:**")
            st.write(df.dtypes.to_dict())
            
            if df.select_dtypes(include=[np.number]).columns.any():
                st.write("**Numeric Columns Summary:**")
                st.dataframe(df.describe())

def main():
    """Run the dashboard"""
    dashboard = RealTimeDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
