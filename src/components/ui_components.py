import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Professional CSS Framework - Enhanced version
PROFESSIONAL_CSS = """
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styles */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Enhanced card components */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        border-color: #667eea;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px 0 0 4px;
    }
    
    /* Recommendation card styling */
    .recommendation-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #dee2e6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
        transform: translateY(-1px);
    }
    
    .recommendation-card.high-priority {
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
    }
    
    .recommendation-card.medium-priority {
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #fffbf0 0%, #feebc8 100%);
    }
    
    .recommendation-card.low-priority {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
    }
    
    /* Chart container styling */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
        position: relative;
    }
    
    .chart-container::before {
        content: '📊';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 1.2rem;
        opacity: 0.5;
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin-right: 1rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(180deg); }
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced button styling */
    .recommendation-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    .recommendation-button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transition: width 0.6s, height 0.6s, top 0.6s, left 0.6s;
    }
    
    .recommendation-button:active::before {
        width: 300px;
        height: 300px;
        top: -150px;
        left: -150px;
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
        box-shadow: 0 2px 8px rgba(86, 204, 242, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #FFB946 0%, #FF8C42 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
        box-shadow: 0 2px 8px rgba(255, 185, 70, 0.3);
    }
    
    .status-error {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF5252 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.3);
    }
    
    /* Progress bars */
    .progress-bar {
        background: #f0f2f5;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
        position: relative;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        transition: width 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .progress-fill::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer-progress 2s infinite;
    }
    
    @keyframes shimmer-progress {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .feature-card {
            background: #1e1e1e;
            border-color: #333;
            color: #e1e1e1;
        }
        
        .chart-container {
            background: #1e1e1e;
            border-color: #333;
        }
        
        .recommendation-card {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            border-color: #4a5568;
            color: #e1e1e1;
        }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .app-header {
            padding: 1rem;
        }
        
        .feature-card {
            padding: 1rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
</style>
"""

def render_professional_header(title: str, subtitle: str = "", icon: str = "🧠"):
    """Render a professional animated header"""
    st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)
    
    st.markdown(f'''
    <div class="app-header fade-in">
        <h1>{icon} {title}</h1>
        {f"<p style='margin-top: 1rem; font-size: 1.1rem; opacity: 0.9;'>{subtitle}</p>" if subtitle else ""}
    </div>
    ''', unsafe_allow_html=True)

def render_metric_cards(metrics: list):
    """Render professional metric cards with safe delta handling"""
    if not metrics:
        return
        
    cols = st.columns(len(metrics))
    
    for i, metric in enumerate(metrics):
        with cols[i]:
            # Safe delta handling - check if key exists
            delta_html = ""
            if metric.get('delta') is not None:
                delta_value = metric.get('delta')
                
                # Handle both numeric and string delta values
                if isinstance(delta_value, (int, float)):
                    delta_color = "green" if delta_value >= 0 else "red"
                    delta_symbol = "↗" if delta_value >= 0 else "↘"
                    delta_html = f"<div style='color: {delta_color}; font-size: 0.8rem; margin-top: 0.5rem;'>{delta_symbol} {delta_value}</div>"
                else:
                    # For string deltas, determine color based on presence of '+' or positive words
                    delta_str = str(delta_value)
                    if '+' in delta_str or any(word in delta_str.lower() for word in ['increase', 'up', 'growth', 'gain']):
                        delta_color = "green"
                        delta_symbol = "↗"
                    elif '-' in delta_str or any(word in delta_str.lower() for word in ['decrease', 'down', 'loss', 'drop']):
                        delta_color = "red"
                        delta_symbol = "↘"
                    else:
                        delta_color = "#666"
                        delta_symbol = "→"
                    delta_html = f"<div style='color: {delta_color}; font-size: 0.8rem; margin-top: 0.5rem;'>{delta_symbol} {delta_str}</div>"
            
            # Get title and value safely
            title = metric.get('title', 'Unknown')
            value = metric.get('value', '0')
            
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{title}</div>
                {delta_html}
            </div>
            ''', unsafe_allow_html=True)

def render_feature_card(title: str, content: str, icon: str = "📊", action_text: str = None, action_key: str = None):
    """Render a feature card with optional action button"""
    
    action_button = ""
    if action_text and action_key:
        action_button = f'<button onclick="alert(\'Feature coming soon!\')" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 0.5rem 1rem; border-radius: 6px; margin-top: 1rem; cursor: pointer;">{action_text}</button>'
    
    st.markdown(f'''
    <div class="feature-card">
        <h3>{icon} {title}</h3>
        <p>{content}</p>
        {action_button}
    </div>
    ''', unsafe_allow_html=True)

def render_insight_card(insight: str, confidence: float = None):
    """
    Render an AI insight card - FIXED to use markdown instead of HTML
    to avoid rendering issues with mixed content
    """
    
    # Clean the insight text from any problematic characters
    clean_insight = str(insight).strip()
    
    # Use streamlit's built-in info component instead of HTML
    st.info(f"💡 **AI Insights**\n\n{clean_insight}")
    
    # If confidence is provided, show it separately
    if confidence:
        confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
        st.caption(f"{confidence_color} Confidence: {confidence:.1%}")

def render_status_indicator(text: str, status: str = "success"):
    """Render a status indicator badge"""
    class_name = f"status-{status}"
    st.markdown(f'<span class="{class_name}">{text}</span>', unsafe_allow_html=True)

def render_animated_loading(text="Đang xử lý..."):
    """Render an animated loading indicator"""
    
    st.markdown(f'''
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <span style="color: #667eea; font-weight: 500;">{text}</span>
    </div>
    ''', unsafe_allow_html=True)

def render_chart_container(chart_content, title=""):
    """Render a professional chart container"""
    
    st.markdown(f'''
    <div class="chart-container">
        {f"<h4 style='margin-bottom: 1rem; color: #2c3e50;'>{title}</h4>" if title else ""}
        <div>{chart_content}</div>
    </div>
    ''', unsafe_allow_html=True)

def render_recommendation_card(title: str, description: str, action: str, icon: str = "💡", priority: str = "medium"):
    """Render an enhanced recommendation card"""
    
    priority_class = f"{priority}-priority"
    
    st.markdown(f'''
    <div class="recommendation-card {priority_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="flex: 1;">
                <h4 style="margin: 0 0 0.5rem 0; color: #2c3e50;">
                    <span style="margin-right: 0.5rem;">{icon}</span>
                    {title}
                </h4>
                <p style="margin: 0 0 1rem 0; color: #495057;">{description}</p>
                <button class="recommendation-button" onclick="this.innerHTML='🔄 Processing...'; this.disabled=true;">
                    {action}
                </button>
            </div>
            <div style="margin-left: 1rem;">
                <span style="
                    background: {'#dc3545' if priority == 'high' else '#ffc107' if priority == 'medium' else '#28a745'};
                    color: white;
                    padding: 0.25rem 0.5rem;
                    border-radius: 12px;
                    font-size: 0.7rem;
                    text-transform: uppercase;
                    font-weight: bold;
                ">
                    {priority}
                </span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

def create_professional_chart_layout():
    """Create a professional chart layout with modern styling"""
    
    layout = {
        'template': 'plotly_white',
        'font': dict(family="Inter, sans-serif", size=12),
        'title': dict(
            x=0.5,
            font=dict(size=16, color='#2c3e50'),
            pad=dict(t=20)
        ),
        'margin': dict(t=60, l=60, r=60, b=60),
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'xaxis': dict(
            gridcolor='#e1e5e9',
            gridwidth=1,
            zeroline=False
        ),
        'yaxis': dict(
            gridcolor='#e1e5e9',
            gridwidth=1,
            zeroline=False
        ),
        'colorway': ['#667eea', '#764ba2', '#56CCF2', '#2F80ED', '#FF6B6B', '#FF8E53', '#4ECDC4', '#45B7D1']
    }
    
    return layout

def render_progress_bar(progress: float, text: str = ""):
    """Render a progress bar with text"""
    progress_percent = min(100, max(0, progress * 100))
    
    st.markdown(f'''
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress_percent}%;"></div>
    </div>
    <small>{text} ({progress_percent:.1f}%)</small>
    ''', unsafe_allow_html=True)

def create_data_quality_indicator(df):
    """Create a comprehensive data quality indicator"""
    
    # Calculate quality metrics
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = 1 - (missing_cells / total_cells)
    
    # Uniqueness (average across columns)
    uniqueness_scores = []
    for col in df.columns:
        if df[col].dtype in ['object', 'string']:
            uniqueness = df[col].nunique() / len(df)
        else:
            uniqueness = 1.0  # Numeric columns assumed to have good uniqueness
        uniqueness_scores.append(uniqueness)
    
    avg_uniqueness = sum(uniqueness_scores) / len(uniqueness_scores)
    
    # Consistency (no mixed types in object columns)
    consistency = 1.0  # Simplified for this example
    
    # Overall quality score
    quality_score = (completeness + avg_uniqueness + consistency) / 3
    
    # Create visual indicator
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card" style="background: linear-gradient(135deg, {"#28a745" if completeness > 0.9 else "#ffc107" if completeness > 0.7 else "#dc3545"} 0%, {"#20c997" if completeness > 0.9 else "#fd7e14" if completeness > 0.7 else "#e55353"} 100%);">
            <div class="metric-value">{completeness:.1%}</div>
            <div class="metric-label">Completeness</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card" style="background: linear-gradient(135deg, {"#28a745" if avg_uniqueness > 0.8 else "#ffc107" if avg_uniqueness > 0.6 else "#dc3545"} 0%, {"#20c997" if avg_uniqueness > 0.8 else "#fd7e14" if avg_uniqueness > 0.6 else "#e55353"} 100%);">
            <div class="metric-value">{avg_uniqueness:.1%}</div>
            <div class="metric-label">Uniqueness</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card" style="background: linear-gradient(135deg, {"#28a745" if consistency > 0.9 else "#ffc107" if consistency > 0.7 else "#dc3545"} 0%, {"#20c997" if consistency > 0.9 else "#fd7e14" if consistency > 0.7 else "#e55353"} 100%);">
            <div class="metric-value">{consistency:.1%}</div>
            <div class="metric-label">Consistency</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card" style="background: linear-gradient(135deg, {"#28a745" if quality_score > 0.8 else "#ffc107" if quality_score > 0.6 else "#dc3545"} 0%, {"#20c997" if quality_score > 0.8 else "#fd7e14" if quality_score > 0.6 else "#e55353"} 100%);">
            <div class="metric-value">{quality_score:.1%}</div>
            <div class="metric-label">Overall Quality</div>
        </div>
        ''', unsafe_allow_html=True)
    
    return quality_score

def render_interactive_data_explorer(df):
    """Create an interactive data explorer widget"""
    
    st.markdown("### 🔍 Interactive Data Explorer")
    
    # Column selector
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_columns = st.multiselect(
            "Select columns to explore:",
            options=df.columns.tolist(),
            default=df.columns.tolist()[:5]
        )
        
        # Data type filter
        data_types = st.multiselect(
            "Filter by data type:",
            options=['numeric', 'categorical', 'datetime'],
            default=['numeric', 'categorical']
        )
        
        # Quick stats toggle
        show_stats = st.checkbox("Show statistics", value=True)
        show_missing = st.checkbox("Highlight missing values", value=True)
    
    with col2:
        if selected_columns:
            # Filter dataframe
            filtered_df = df[selected_columns]
            
            # Apply data type filters
            if data_types:
                type_cols = []
                if 'numeric' in data_types:
                    type_cols.extend(filtered_df.select_dtypes(include=['number']).columns.tolist())
                if 'categorical' in data_types:
                    type_cols.extend(filtered_df.select_dtypes(include=['object', 'category']).columns.tolist())
                if 'datetime' in data_types:
                    type_cols.extend(filtered_df.select_dtypes(include=['datetime']).columns.tolist())
                
                if type_cols:
                    filtered_df = filtered_df[type_cols]
            
            # Display data with styling
            if show_missing:
                styled_df = filtered_df.style.highlight_null(null_color='lightcoral')
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.dataframe(filtered_df, use_container_width=True)
            
            # Show statistics if requested
            if show_stats and not filtered_df.empty:
                st.markdown("#### 📈 Quick Statistics")
                
                numeric_cols = filtered_df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)

# Export the styling for use in other modules
__all__ = [
    'PROFESSIONAL_CSS',
    'render_professional_header',
    'render_metric_cards',
    'render_feature_card',
    'render_insight_card',
    'render_status_indicator',
    'create_professional_chart_layout',
    'render_progress_bar',
    'render_animated_loading',
    'render_chart_container',
    'render_recommendation_card',
    'create_data_quality_indicator',
    'render_interactive_data_explorer'
]