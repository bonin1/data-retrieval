import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import io
import base64

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedVisualizer:
    """Advanced visualization suite for e-commerce data analytics"""
    
    def __init__(self, data: pd.DataFrame, output_dir: str = "visualizations"):
        """
        Initialize visualizer with data
        
        Args:
            data: Processed pandas DataFrame
            output_dir: Directory to save visualizations
        """
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'palette': px.colors.qualitative.Set3
        }
        
        self.plots_created = []
    
    def create_price_distribution_plots(self) -> List[str]:
        plots = []
        
        if 'price' not in self.data.columns:
            logger.warning("Price column not found")
            return plots
        
        price_data = self.data['price'].dropna()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].hist(price_data, bins=50, alpha=0.7, color=self.colors['primary'], density=True)
        axes[0, 0].set_title('Price Distribution with Density')
        axes[0, 0].set_xlabel('Price')
        axes[0, 0].set_ylabel('Density')
        
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(price_data)
        x_range = np.linspace(price_data.min(), price_data.max(), 200)
        axes[0, 0].plot(x_range, kde(x_range), color=self.colors['danger'], linewidth=2, label='KDE')
        axes[0, 0].legend()
        
        axes[0, 1].boxplot(price_data, patch_artist=True)
        axes[0, 1].set_title('Price Box Plot')
        axes[0, 1].set_ylabel('Price')
        
        from scipy import stats
        stats.probplot(price_data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)')
        
        log_prices = np.log1p(price_data)
        axes[1, 1].hist(log_prices, bins=50, alpha=0.7, color=self.colors['success'])
        axes[1, 1].set_title('Log-Transformed Price Distribution')
        axes[1, 1].set_xlabel('Log(Price + 1)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        price_dist_file = self.output_dir / "price_distribution_analysis.png"
        plt.savefig(price_dist_file, dpi=300, bbox_inches='tight')
        plt.close()
        plots.append(str(price_dist_file))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Price Histogram', 'Price Box Plot', 'Price by Category', 'Price Trends'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Histogram(x=price_data, nbinsx=50, name='Price Distribution', 
                        marker_color=self.colors['primary'], opacity=0.7),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Box(y=price_data, name='Price', marker_color=self.colors['secondary']),
            row=1, col=2
        )
        
        if 'category' in self.data.columns:
            category_prices = self.data.groupby('category')['price'].median().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(x=category_prices.index, y=category_prices.values, 
                      name='Median Price by Category', marker_color=self.colors['success']),
                row=2, col=1
            )
        
        if 'scraped_at' in self.data.columns:
            self.data['scraped_date'] = pd.to_datetime(self.data['scraped_at']).dt.date
            daily_prices = self.data.groupby('scraped_date')['price'].mean()
            fig.add_trace(
                go.Scatter(x=daily_prices.index, y=daily_prices.values, 
                          mode='lines+markers', name='Daily Average Price',
                          line_color=self.colors['warning']),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Comprehensive Price Analysis")
        price_interactive_file = self.output_dir / "price_analysis_interactive.html"
        fig.write_html(price_interactive_file)
        plots.append(str(price_interactive_file))
        
        return plots
    
    def create_market_analysis_plots(self) -> List[str]:
        plots = []
        
        if 'brand' in self.data.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            brand_counts = self.data['brand'].value_counts().head(10)
            colors = plt.cm.Set3(np.linspace(0, 1, len(brand_counts)))
            
            wedges, texts, autotexts = axes[0, 0].pie(brand_counts.values, labels=brand_counts.index, 
                                                     autopct='%1.1f%%', colors=colors)
            axes[0, 0].set_title('Top 10 Brands Market Share')
            
            if 'price' in self.data.columns:
                brand_prices = self.data.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
                brand_prices = brand_prices[brand_prices['count'] >= 5].sort_values('mean', ascending=False)
                
                axes[0, 1].barh(brand_prices['brand'], brand_prices['mean'], color=self.colors['secondary'])
                axes[0, 1].set_title('Average Price by Brand')
                axes[0, 1].set_xlabel('Average Price')
                
                axes[1, 0].scatter(brand_prices['count'], brand_prices['mean'], 
                                  alpha=0.7, s=100, color=self.colors['primary'])
                axes[1, 0].set_xlabel('Number of Products')
                axes[1, 0].set_ylabel('Average Price')
                axes[1, 0].set_title('Price vs Market Presence')
                
                for idx, row in brand_prices.iterrows():
                    if row['count'] > brand_prices['count'].median() or row['mean'] > brand_prices['mean'].median():
                        axes[1, 0].annotate(row['brand'], (row['count'], row['mean']), 
                                          xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            if 'category' in self.data.columns:
                category_counts = self.data['category'].value_counts()
                axes[1, 1].bar(range(len(category_counts)), category_counts.values, 
                              color=self.colors['success'], alpha=0.7)
                axes[1, 1].set_xticks(range(len(category_counts)))
                axes[1, 1].set_xticklabels(category_counts.index, rotation=45, ha='right')
                axes[1, 1].set_title('Product Distribution by Category')
                axes[1, 1].set_ylabel('Number of Products')
            
            plt.tight_layout()
            market_file = self.output_dir / "market_analysis.png"
            plt.savefig(market_file, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(str(market_file))
        
        if 'category' in self.data.columns and 'brand' in self.data.columns:
            category_brand = self.data.groupby(['category', 'brand']).size().reset_index(name='count')
            category_brand = category_brand[category_brand['count'] >= 2]
            
            fig = px.treemap(category_brand, 
                           path=['category', 'brand'], 
                           values='count',
                           title='Market Structure: Categories and Brands',
                           color='count',
                           color_continuous_scale='Viridis')
            
            treemap_file = self.output_dir / "market_treemap.html"
            fig.write_html(treemap_file)
            plots.append(str(treemap_file))
        
        return plots
    
    def create_quality_analysis_plots(self) -> List[str]:
        plots = []
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        missing_data = self.data.isnull().sum()
        completeness = ((len(self.data) - missing_data) / len(self.data)) * 100
        
        heatmap_data = completeness.values.reshape(-1, 1)
        im = axes[0, 0].imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        axes[0, 0].set_yticks(range(len(completeness)))
        axes[0, 0].set_yticklabels(completeness.index, fontsize=8)
        axes[0, 0].set_title('Data Completeness by Field')
        axes[0, 0].set_xlabel('Completeness %')
        
        cbar = plt.colorbar(im, ax=axes[0, 0])
        cbar.set_label('Completeness %')
        
        if 'rating' in self.data.columns:
            rating_data = self.data['rating'].dropna()
            axes[0, 1].hist(rating_data, bins=20, alpha=0.7, color=self.colors['warning'], edgecolor='black')
            axes[0, 1].set_title('Rating Distribution')
            axes[0, 1].set_xlabel('Rating')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(rating_data.mean(), color='red', linestyle='--', 
                              label=f'Mean: {rating_data.mean():.2f}')
            axes[0, 1].legend()
        
        if 'reviews_count' in self.data.columns:
            reviews_data = self.data['reviews_count'].dropna()
            log_reviews = np.log1p(reviews_data)
            axes[0, 2].hist(log_reviews, bins=30, alpha=0.7, color=self.colors['info'])
            axes[0, 2].set_title('Reviews Count Distribution (Log Scale)')
            axes[0, 2].set_xlabel('Log(Reviews + 1)')
            axes[0, 2].set_ylabel('Frequency')
        
        if 'price' in self.data.columns and 'rating' in self.data.columns:
            price_rating_data = self.data[['price', 'rating']].dropna()
            axes[1, 0].scatter(price_rating_data['price'], price_rating_data['rating'], 
                             alpha=0.6, color=self.colors['primary'])
            axes[1, 0].set_xlabel('Price')
            axes[1, 0].set_ylabel('Rating')
            axes[1, 0].set_title('Price vs Rating Correlation')
            
            correlation = price_rating_data['price'].corr(price_rating_data['rating'])
            axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                           transform=axes[1, 0].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        if 'title' in self.data.columns:
            title_lengths = self.data['title'].str.len().dropna()
            axes[1, 1].hist(title_lengths, bins=30, alpha=0.7, color=self.colors['success'])
            axes[1, 1].set_title('Product Title Length Distribution')
            axes[1, 1].set_xlabel('Title Length (characters)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].axvline(title_lengths.mean(), color='red', linestyle='--', 
                              label=f'Mean: {title_lengths.mean():.1f}')
            axes[1, 1].legend()
        
        if 'data_quality_score' in self.data.columns:
            quality_scores = self.data['data_quality_score'].dropna()
            axes[1, 2].hist(quality_scores, bins=20, alpha=0.7, color=self.colors['danger'])
            axes[1, 2].set_title('Data Quality Score Distribution')
            axes[1, 2].set_xlabel('Quality Score')
            axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        quality_file = self.output_dir / "quality_analysis.png"
        plt.savefig(quality_file, dpi=300, bbox_inches='tight')
        plt.close()
        plots.append(str(quality_file))
        
        return plots
    
    def create_clustering_visualization(self, cluster_labels: np.ndarray = None) -> List[str]:
        plots = []
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in ['price', 'rating', 'reviews_count', 'title_length'] 
                       if col in numerical_cols]
        
        if len(feature_cols) < 2:
            logger.warning("Insufficient numerical features for clustering visualization")
            return plots
        
        cluster_data = self.data[feature_cols].dropna()
        
        if cluster_labels is None:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            kmeans = KMeans(n_clusters=4, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        scatter = axes[0, 0].scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], 
                                   c=cluster_labels, cmap='viridis', alpha=0.7)
        axes[0, 0].set_xlabel(feature_cols[0])
        axes[0, 0].set_ylabel(feature_cols[1])
        axes[0, 0].set_title('Clustering Visualization')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        axes[0, 1].bar(unique_labels, counts, color=self.colors['palette'][:len(unique_labels)])
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Number of Products')
        axes[0, 1].set_title('Cluster Size Distribution')
        
        if len(feature_cols) >= 3:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(cluster_data)
            
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(scaled_features)
            
            scatter_pca = axes[1, 0].scatter(pca_features[:, 0], pca_features[:, 1], 
                                           c=cluster_labels, cmap='viridis', alpha=0.7)
            axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            axes[1, 0].set_title('PCA Clustering Visualization')
            plt.colorbar(scatter_pca, ax=axes[1, 0])
            
            feature_importance = pd.DataFrame(
                pca.components_.T,
                columns=['PC1', 'PC2'],
                index=feature_cols
            )
            
            feature_importance.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Feature Importance in PCA')
            axes[1, 1].set_ylabel('Component Loading')
            axes[1, 1].legend()
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        clustering_file = self.output_dir / "clustering_analysis.png"
        plt.savefig(clustering_file, dpi=300, bbox_inches='tight')
        plt.close()
        plots.append(str(clustering_file))
        
        if len(feature_cols) >= 3:
            fig = go.Figure(data=[go.Scatter3d(
                x=cluster_data.iloc[:, 0],
                y=cluster_data.iloc[:, 1],
                z=cluster_data.iloc[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=cluster_labels,
                    colorscale='Viridis',
                    opacity=0.7,
                    colorbar=dict(title="Cluster")
                ),
                text=[f'Cluster: {label}' for label in cluster_labels],
                hovertemplate='<b>%{text}</b><br>' +
                             f'{feature_cols[0]}: %{{x}}<br>' +
                             f'{feature_cols[1]}: %{{y}}<br>' +
                             f'{feature_cols[2]}: %{{z}}<extra></extra>'
            )])
            
            fig.update_layout(
                title='3D Clustering Visualization',
                scene=dict(
                    xaxis_title=feature_cols[0],
                    yaxis_title=feature_cols[1],
                    zaxis_title=feature_cols[2]
                ),
                width=800,
                height=600
            )
            
            clustering_3d_file = self.output_dir / "clustering_3d.html"
            fig.write_html(clustering_3d_file)
            plots.append(str(clustering_3d_file))
        
        return plots
    
    def create_wordcloud_visualization(self) -> List[str]:
        plots = []
        
        if 'title' not in self.data.columns:
            logger.warning("Title column not found for word cloud")
            return plots
        
        titles = self.data['title'].dropna().astype(str)
        all_text = ' '.join(titles).lower()
        
        import re
        all_text = re.sub(r'[^a-zA-Z\s]', '', all_text)
        
        stop_words = set(['dhe', 'per', 'me', 'nga', 'ne', 'te', 'se', 'nje', 'i', 'e', 
                         'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        wordcloud = WordCloud(
            width=1200, 
            height=600, 
            background_color='white',
            stopwords=stop_words,
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
        
        # Plot word cloud
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Product Titles Word Cloud', fontsize=20, pad=20)
        
        wordcloud_file = self.output_dir / "wordcloud_titles.png"
        plt.savefig(wordcloud_file, dpi=300, bbox_inches='tight')
        plt.close()
        plots.append(str(wordcloud_file))
        
        if 'category' in self.data.columns:
            categories = self.data['category'].value_counts().head(4).index
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, category in enumerate(categories):
                if i >= 4:
                    break
                    
                category_titles = self.data[self.data['category'] == category]['title'].dropna()
                category_text = ' '.join(category_titles.astype(str)).lower()
                category_text = re.sub(r'[^a-zA-Z\s]', '', category_text)
                
                if len(category_text.strip()) > 0:
                    category_wordcloud = WordCloud(
                        width=600, 
                        height=400, 
                        background_color='white',
                        stopwords=stop_words,
                        max_words=50,
                        colormap='plasma'
                    ).generate(category_text)
                    
                    axes[i].imshow(category_wordcloud, interpolation='bilinear')
                    axes[i].axis('off')
                    axes[i].set_title(f'{category} Products', fontsize=14)
            
            plt.tight_layout()
            category_wordcloud_file = self.output_dir / "wordcloud_by_category.png"
            plt.savefig(category_wordcloud_file, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(str(category_wordcloud_file))
        
        return plots
    
    def create_correlation_heatmap(self) -> List[str]:
        plots = []
        
        numerical_data = self.data.select_dtypes(include=[np.number])
        
        if numerical_data.shape[1] < 2:
            logger.warning("Insufficient numerical columns for correlation analysis")
            return plots
        
        correlation_matrix = numerical_data.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
        plt.tight_layout()
        
        correlation_file = self.output_dir / "correlation_heatmap.png"
        plt.savefig(correlation_file, dpi=300, bbox_inches='tight')
        plt.close()
        plots.append(str(correlation_file))
        
        fig = px.imshow(correlation_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       color_continuous_scale='RdBu',
                       title='Interactive Correlation Heatmap')
        
        interactive_correlation_file = self.output_dir / "correlation_heatmap_interactive.html"
        fig.write_html(interactive_correlation_file)
        plots.append(str(interactive_correlation_file))
        
        return plots
    
    def create_time_series_plots(self) -> List[str]:
        plots = []
        
        if 'scraped_at' not in self.data.columns:
            logger.info("No temporal data found for time series analysis")
            return plots
        
        self.data['scraped_datetime'] = pd.to_datetime(self.data['scraped_at'])
        self.data['scraped_date'] = self.data['scraped_datetime'].dt.date
        self.data['scraped_hour'] = self.data['scraped_datetime'].dt.hour
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        daily_counts = self.data.groupby('scraped_date').size()
        axes[0, 0].plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2)
        axes[0, 0].set_title('Products Scraped Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Products')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        if 'price' in self.data.columns:
            daily_prices = self.data.groupby('scraped_date')['price'].mean()
            axes[0, 1].plot(daily_prices.index, daily_prices.values, 
                          marker='s', color=self.colors['secondary'], linewidth=2)
            axes[0, 1].set_title('Average Price Trend')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Average Price')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        hourly_counts = self.data.groupby('scraped_hour').size()
        axes[1, 0].bar(hourly_counts.index, hourly_counts.values, 
                      color=self.colors['success'], alpha=0.7)
        axes[1, 0].set_title('Scraping Activity by Hour')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Number of Products')
        
        if 'category' in self.data.columns:
            top_categories = self.data['category'].value_counts().head(5).index
            for i, category in enumerate(top_categories):
                category_daily = self.data[self.data['category'] == category].groupby('scraped_date').size()
                axes[1, 1].plot(category_daily.index, category_daily.values, 
                              marker='o', label=category, linewidth=2)
            
            axes[1, 1].set_title('Category Trends Over Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Number of Products')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        timeseries_file = self.output_dir / "time_series_analysis.png"
        plt.savefig(timeseries_file, dpi=300, bbox_inches='tight')
        plt.close()
        plots.append(str(timeseries_file))
        
        return plots
    
    def create_dashboard_summary(self) -> str:
        all_plots = []
        all_plots.extend(self.create_price_distribution_plots())
        all_plots.extend(self.create_market_analysis_plots())
        all_plots.extend(self.create_quality_analysis_plots())
        all_plots.extend(self.create_correlation_heatmap())
        all_plots.extend(self.create_wordcloud_visualization())
        all_plots.extend(self.create_time_series_plots())
        
        html_content = self._generate_dashboard_html(all_plots)
        
        dashboard_file = self.output_dir / "analytics_dashboard.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Dashboard created: {dashboard_file}")
        return str(dashboard_file)
    
    def _generate_dashboard_html(self, plots: List[str]) -> str:
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>E-commerce Analytics Dashboard</title>
            <style>
                body {{
                    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    padding: 30px;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                    border-bottom: 3px solid #007bff;
                    padding-bottom: 15px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 40px;
                    margin-bottom: 20px;
                    border-left: 4px solid #007bff;
                    padding-left: 15px;
                }}
                .plot-container {{
                    margin: 20px 0;
                    text-align: center;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 15px;
                    background-color: #fafafa;
                }}
                .plot-container img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .stat-label {{
                    font-size: 1.1em;
                    opacity: 0.9;
                }}
                .iframe-container {{
                    width: 100%;
                    height: 600px;
                    border: none;
                    margin: 10px 0;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                .timestamp {{
                    text-align: center;
                    color: #666;
                    font-style: italic;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 E-commerce Analytics Dashboard</h1>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{total_products}</div>
                        <div class="stat-label">Total Products</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{avg_price}</div>
                        <div class="stat-label">Average Price</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{unique_brands}</div>
                        <div class="stat-label">Unique Brands</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{unique_categories}</div>
                        <div class="stat-label">Categories</div>
                    </div>
                </div>
                
                {plot_sections}
                
                <div class="timestamp">
                    Generated on {timestamp}
                </div>
            </div>
        </body>
        </html>
        """
        
        total_products = len(self.data)
        avg_price = f"${self.data['price'].mean():.2f}" if 'price' in self.data.columns else "N/A"
        unique_brands = self.data['brand'].nunique() if 'brand' in self.data.columns else "N/A"
        unique_categories = self.data['category'].nunique() if 'category' in self.data.columns else "N/A"
        
        plot_sections = ""
        
        for plot_path in plots:
            plot_name = Path(plot_path).stem.replace('_', ' ').title()
            
            if plot_path.endswith('.html'):
                plot_sections += f"""
                <h2>{plot_name}</h2>
                <div class="plot-container">
                    <iframe src="{Path(plot_path).name}" class="iframe-container"></iframe>
                </div>
                """
            else:
                plot_sections += f"""
                <h2>{plot_name}</h2>
                <div class="plot-container">
                    <img src="{Path(plot_path).name}" alt="{plot_name}">
                </div>
                """
        
        return html_template.format(
            total_products=total_products,
            avg_price=avg_price,
            unique_brands=unique_brands,
            unique_categories=unique_categories,
            plot_sections=plot_sections,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def save_all_visualizations(self) -> Dict[str, List[str]]:
        logger.info("Creating comprehensive visualization suite...")
        
        visualization_results = {
            'price_analysis': self.create_price_distribution_plots(),
            'market_analysis': self.create_market_analysis_plots(),
            'quality_analysis': self.create_quality_analysis_plots(),
            'correlation_analysis': self.create_correlation_heatmap(),
            'text_analysis': self.create_wordcloud_visualization(),
            'time_series': self.create_time_series_plots(),
        }
        
        dashboard_path = self.create_dashboard_summary()
        visualization_results['dashboard'] = [dashboard_path]
        
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_plots': sum(len(plots) for plots in visualization_results.values()),
            'data_shape': self.data.shape,
            'output_directory': str(self.output_dir)
        }
        
        metadata_file = self.output_dir / "visualization_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created {metadata['total_plots']} visualizations in {self.output_dir}")
        return visualization_results

def main():
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'title': [f'Product {i}' for i in range(1000)],
        'price': np.random.lognormal(4, 1, 1000),
        'rating': np.random.normal(4, 0.5, 1000),
        'reviews_count': np.random.poisson(50, 1000),
        'brand': np.random.choice(['Brand A', 'Brand B', 'Brand C', 'Brand D'], 1000),
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 1000),
        'scraped_at': pd.date_range('2024-01-01', periods=1000, freq='1H')
    })
    
    visualizer = AdvancedVisualizer(sample_data, "sample_visualizations")
    
    results = visualizer.save_all_visualizations()
    
    print("Visualization Results:")
    for category, plots in results.items():
        print(f"{category}: {len(plots)} plots")

if __name__ == "__main__":
    main()
