import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Professional CSS Framework
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
                    {"icon": "📂", "label": "Bảng Điều Khiển", "page": "pages/1_🧮_Bang_Dieu_Khien.py"},
            {"icon": "📊", "label": "Chi Tiết Bộ Dữ Liệu", "page": "pages/3_📂_Chi_Tiet_Bo_Du_Lieu.py"},
            {"icon": "📈", "label": "Biểu Đồ Thông Minh", "page": "pages/6_📈_Bieu_Do_Thong_Minh.py"},
            {"icon": "📋", "label": "Lịch Sử Biểu Đồ", "page": "pages/4_📊_Lich_Su_Bieu_Do.py"},
            {"icon": "🔗", "label": "Phân Tích Chéo", "page": "pages/7_🔗_Phan_Tich_Cheo_Du_Lieu.py"},
            {"icon": "📄", "label": "Báo Cáo EDA", "page": "pages/5_📋_Bao_Cao_EDA.py"},
            {"icon": "📖", "label": "Về Dự Án", "page": "pages/📖_Ve_Du_An.py"}und: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    
    /* Card components */
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
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
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
    }
    
    /* Insight cards */
    .insight-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 1px solid #667eea30;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        position: relative;
    }
    
    .insight-card::before {
        content: '💡';
        position: absolute;
        top: 1rem;
        left: 1rem;
        font-size: 1.2rem;
    }
    
    .insight-content {
        margin-left: 2rem;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
    }
    
    /* Navigation styling */
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
        border: 1px solid transparent;
    }
    
    .nav-item:hover {
        background: #667eea10;
        border-color: #667eea;
        transform: translateX(4px);
    }
    
    .nav-item.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Progress indicators */
    .progress-bar {
        background: #f0f2f5;
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    /* Custom selectbox styling */
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Custom button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Animation utilities */
    .fade-in {
        animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.8s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
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
        
        .stSelectbox > div > div {
            background-color: #2e2e2e;
            border-color: #444;
            color: #e1e1e1;
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
    """Render professional metric cards
    
    Args:
        metrics: List of dicts with keys: title, value, delta (optional)
    """
    cols = st.columns(len(metrics))
    
    for i, metric in enumerate(metrics):
        with cols[i]:
            delta_html = ""
            if metric['delta']:
                # Handle both numeric and string delta values
                if isinstance(metric['delta'], (int, float)):
                    delta_color = "green" if metric['delta'] >= 0 else "red"
                    delta_symbol = "↗" if metric['delta'] >= 0 else "↘"
                    delta_html = f"<div style='color: {delta_color}; font-size: 0.8rem; margin-top: 0.5rem;'>{delta_symbol} {metric['delta']}</div>"
                else:
                    # For string deltas, determine color based on presence of '+' or positive words
                    delta_str = str(metric['delta'])
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
            
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{metric['value']}</div>
                <div class="metric-label">{metric['title']}</div>
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
    """Render an AI insight card with confidence indicator"""
    
    confidence_html = ""
    if confidence:
        confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
        confidence_html = f'<div style="color: {confidence_color}; font-size: 0.8rem; margin-top: 0.5rem;">Confidence: {confidence:.1%}</div>'
    
    st.markdown(f'''
    <div class="insight-card">
        <div class="insight-content">
            {insight}
            {confidence_html}
        </div>
    </div>
    ''', unsafe_allow_html=True)

def render_status_indicator(text: str, status: str = "success"):
    """Render a status indicator badge"""
    class_name = f"status-{status}"
    st.markdown(f'<span class="{class_name}">{text}</span>', unsafe_allow_html=True)

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

def create_enhanced_dashboard_chart(data, chart_type="overview"):
    """Create enhanced dashboard charts with professional styling"""
    
    if chart_type == "overview":
        # Create a comprehensive overview chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Dataset Distribution', 'Upload Timeline', 'Data Quality', 'Usage Analytics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Apply professional layout
        fig.update_layout(create_professional_chart_layout())
        fig.update_layout(height=600, showlegend=False)
        
        return fig
    
    elif chart_type == "correlation":
        # Enhanced correlation heatmap
        fig = px.imshow(
            data,
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        
        fig.update_layout(create_professional_chart_layout())
        fig.update_layout(title="Correlation Analysis Heatmap")
        
        return fig
    
    elif chart_type == "distribution":
        # Enhanced distribution chart
        fig = px.histogram(
            data,
            marginal="box",
            hover_data=data.columns
        )
        
        fig.update_layout(create_professional_chart_layout())
        fig.update_layout(title="Data Distribution Analysis")
        
        return fig

def render_navigation_sidebar():
    """Render professional navigation sidebar"""
    
    with st.sidebar:
        st.markdown('''
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e1e5e9; margin-bottom: 1rem;">
            <h3 style="color: #667eea; margin: 0;">🧠 VizGenie-GPT</h3>
            <small style="color: #666;">Professional Analytics</small>
        </div>
        ''', unsafe_allow_html=True)
        
        # Navigation menu
        nav_items = [
            {"icon": "📂", "label": "Bảng Điều Khiển", "page": "pages/1_🧮_Bang_Dieu_Khien.py"},
            {"icon": "💬", "label": "Chat AI", "page": "main.py"},
            {"icon": "📊", "label": "Chi Tiết Bộ Dữ Liệu", "page": "pages/3_📂_Chi_Tiet_Bo_Du_Lieu.py"},
            {"icon": "📈", "label": "Biểu Đồ Thông Minh", "page": "pages/6_📈_Bieu_Do_Thong_Minh.py"},
            {"icon": "📋", "label": "Lịch Sử Biểu Đồ", "page": "pages/4_📊_Lich_Su_Bieu_Do.py"},
            {"icon": "🔗", "label": "Phân Tích Chéo", "page": "pages/7_🔗_Phan_Tich_Cheo_Du_Lieu.py"},
            {"icon": "📄", "label": "Báo Cáo EDA", "page": "pages/5_📋_Bao_Cao_EDA.py"},
            {"icon": "📖", "label": "Về Dự Án", "page": "pages/📖_Ve_Du_An.py"}
        ]
        
        for item in nav_items:
            st.markdown(f'''
            <div class="nav-item">
                {item["icon"]} {item["label"]}
            </div>
            ''', unsafe_allow_html=True)
            
            if st.button(f"{item['icon']} {item['label']}", key=f"nav_{item['label']}", use_container_width=True):
                st.switch_page(item["page"])
        
        # Quick stats section
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
        # This would be populated with actual data
        render_metric_cards([
            {"title": "Datasets", "value": "12", "delta": "+3"},
            {"title": "Charts", "value": "45", "delta": "+8"},
            {"title": "Insights", "value": "128", "delta": "+15"}
        ])

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

def create_ai_recommendation_panel(df, analysis_history=None):
    """Create an AI-powered recommendation panel"""
    
    st.markdown("### 🤖 AI Recommendations")
    
    # Analyze data characteristics
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    recommendations = []
    
    # Data structure recommendations
    if len(numeric_cols) >= 2:
        recommendations.append({
            "type": "analysis",
            "priority": "high",
            "title": "Correlation Analysis",
            "description": f"You have {len(numeric_cols)} numeric columns. Consider analyzing correlations between variables.",
            "action": "Create correlation heatmap",
            "icon": "🔥"
        })
    
    if categorical_cols and numeric_cols:
        recommendations.append({
            "type": "visualization",
            "priority": "medium",
            "title": "Group Comparisons",
            "description": f"Compare {numeric_cols[0]} across different {categorical_cols[0]} categories.",
            "action": "Create box plot analysis",
            "icon": "📊"
        })
    
    if datetime_cols:
        recommendations.append({
            "type": "trend",
            "priority": "high",
            "title": "Time Series Analysis",
            "description": f"Detected time-based data in {datetime_cols[0]}. Analyze trends over time.",
            "action": "Create time series chart",
            "icon": "📈"
        })
    
    # Data quality recommendations
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 5:
        recommendations.append({
            "type": "quality",
            "priority": "high",
            "title": "Data Cleaning Needed",
            "description": f"Dataset has {missing_pct:.1f}% missing values. Consider data cleaning.",
            "action": "Go to Dataset Details",
            "icon": "🧹"
        })
    
    # Display recommendations
    for rec in recommendations:
        priority_color = "#dc3545" if rec['priority'] == 'high' else "#ffc107" if rec['priority'] == 'medium' else "#28a745"
        
        st.markdown(f'''
        <div class="feature-card" style="border-left-color: {priority_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4>{rec['icon']} {rec['title']}</h4>
                    <p>{rec['description']}</p>
                </div>
                <span style="background: {priority_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.7rem; text-transform: uppercase;">
                    {rec['priority']}
                </span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button(rec['action'], key=f"rec_{rec['title']}"):
            st.info(f"Recommendation: {rec['action']} - This would trigger the appropriate action!")

def render_animated_loading(text="Processing..."):
    """Render an animated loading indicator"""
    
    st.markdown(f'''
    <div style="text-align: center; padding: 2rem;">
        <div style="display: inline-block; width: 40px; height: 40px; border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; animation: spin 1s linear infinite;"></div>
        <p style="margin-top: 1rem; color: #667eea; font-weight: 500;">{text}</p>
    </div>
    
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    ''', unsafe_allow_html=True)

def create_export_options_panel():
    """Create a professional export options panel"""
    
    st.markdown("### 📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div class="feature-card" style="text-align: center;">
            <h4>📊 Charts</h4>
            <p>Export as PNG, SVG, or PDF</p>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button("Export Charts", key="export_charts", use_container_width=True):
            st.success("Chart export functionality would be implemented here")
    
    with col2:
        st.markdown('''
        <div class="feature-card" style="text-align: center;">
            <h4>📋 Reports</h4>
            <p>Generate comprehensive PDF reports</p>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button("Generate Report", key="export_report", use_container_width=True):
            st.success("Report generation would be implemented here")
    
    with col3:
        st.markdown('''
        <div class="feature-card" style="text-align: center;">
            <h4>💾 Data</h4>
            <p>Export processed data as CSV/Excel</p>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button("Export Data", key="export_data", use_container_width=True):
            st.success("Data export functionality would be implemented here")

# Usage example function
def example_usage():
    """Example of how to use the professional UI components"""
    
    # Initialize professional styling
    render_professional_header(
        "VizGenie-GPT Professional Analytics", 
        "Advanced AI-powered data analysis with beautiful visualizations",
        "🧠"
    )
    
    # Sample data for demonstration
    import pandas as pd
    import numpy as np
    
    # Create sample dataframe
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'revenue': np.random.normal(100000, 20000, 1000),
        'customers': np.random.poisson(50, 1000),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'date': pd.date_range('2023-01-01', periods=1000, freq='D')
    })
    
    # Render metric cards
    render_metric_cards([
        {"title": "Total Revenue", "value": "$2.4M", "delta": "+12%"},
        {"title": "Active Users", "value": "45K", "delta": "+8%"},
        {"title": "Conversion Rate", "value": "3.2%", "delta": "-0.5%"},
        {"title": "Avg Order Value", "value": "$67", "delta": "+15%"}
    ])
    
    # Data quality indicator
    quality_score = create_data_quality_indicator(sample_df)
    
    # AI recommendations
    create_ai_recommendation_panel(sample_df)
    
    # Interactive explorer
    render_interactive_data_explorer(sample_df)
    
    # Export options
    create_export_options_panel()

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
    'create_enhanced_dashboard_chart',
    'render_navigation_sidebar',
    'create_data_quality_indicator',
    'render_interactive_data_explorer',
    'create_ai_recommendation_panel',
    'render_animated_loading',
    'create_export_options_panel'
]