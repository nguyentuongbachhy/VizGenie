import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.llms import load_llm
from src.utils import get_all_datasets, get_dataset, safe_read_csv, add_chart_card
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="📈 Enhanced Smart Charts", layout="wide")

# Professional styling with modern color schemes
st.markdown("""
<style>
    .chart-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .chart-option {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid transparent;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .chart-option:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
    }
    .color-palette {
        display: flex;
        gap: 10px;
        margin: 10px 0;
    }
    .color-box {
        width: 30px;
        height: 30px;
        border-radius: 5px;
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chart-header"><h1>📈 AI-Powered Smart Chart Builder</h1><p>Get intelligent chart recommendations and create stunning visualizations</p></div>', unsafe_allow_html=True)

llm = load_llm("gpt-3.5-turbo")

# Professional color palettes
COLOR_PALETTES = {
    "Professional Blue": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "Vibrant": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"],
    "Corporate": ["#2C3E50", "#3498DB", "#E74C3C", "#F39C12", "#27AE60", "#8E44AD", "#16A085", "#E67E22", "#34495E", "#1ABC9C"],
    "Sunset": ["#FF6B35", "#F7931E", "#FFD23F", "#06FFA5", "#118AB2", "#073B4C", "#E63946", "#F77F00", "#FCBF49", "#003566"],
    "Ocean": ["#0077BE", "#00A8CC", "#0FA3B1", "#B5E2FA", "#F9E784", "#F8AD9D", "#F4975A", "#E8871E", "#DA627D", "#A53860"]
}

# Load datasets
datasets = get_all_datasets()
if not datasets:
    st.warning("⚠️ Please upload a dataset from the Dashboard page.")
    st.stop()

dataset_options = {f"{d[0]} - {d[1]}": d[0] for d in datasets}
selected = st.selectbox("📂 Select dataset to analyze:", list(dataset_options.keys()))
dataset_id = dataset_options[selected]
dataset = get_dataset(dataset_id)
file_path = dataset[2]

@st.cache_data
def load_csv(file_path):
    for enc in ['utf-8', 'ISO-8859-1', 'utf-16', 'cp1252']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except:
            continue
    raise ValueError("❌ Cannot decode CSV file.")

df = load_csv(file_path)

st.markdown(f"**🧾 Dataset Info:** `{dataset[1]}` — {df.shape[0]:,} rows × {df.shape[1]} columns")

def get_chart_recommendations(df, user_intent=""):
    """AI-powered chart recommendations based on data characteristics"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    recommendations = []
    
    # Analysis based on data types
    if len(numeric_cols) >= 2:
        recommendations.append({
            "type": "Scatter Plot",
            "description": "Perfect for exploring relationships between two numeric variables",
            "confidence": 0.9,
            "suggested_x": numeric_cols[0],
            "suggested_y": numeric_cols[1],
            "icon": "🔵",
            "color_scheme": "Professional Blue"
        })
        
        recommendations.append({
            "type": "Correlation Heatmap",
            "description": "Shows correlations between all numeric variables",
            "confidence": 0.85,
            "suggested_x": "All numeric",
            "suggested_y": "All numeric",
            "icon": "🔥",
            "color_scheme": "Sunset"
        })
    
    if categorical_cols and numeric_cols:
        recommendations.append({
            "type": "Box Plot",
            "description": "Compare distributions of numeric data across categories",
            "confidence": 0.8,
            "suggested_x": categorical_cols[0],
            "suggested_y": numeric_cols[0],
            "icon": "📦",
            "color_scheme": "Vibrant"
        })
        
        recommendations.append({
            "type": "Bar Chart",
            "description": "Show averages or sums by category",
            "confidence": 0.75,
            "suggested_x": categorical_cols[0],
            "suggested_y": numeric_cols[0],
            "icon": "📊",
            "color_scheme": "Corporate"
        })
    
    if datetime_cols and numeric_cols:
        recommendations.append({
            "type": "Time Series",
            "description": "Track changes over time",
            "confidence": 0.95,
            "suggested_x": datetime_cols[0],
            "suggested_y": numeric_cols[0],
            "icon": "📈",
            "color_scheme": "Ocean"
        })
    
    if categorical_cols:
        recommendations.append({
            "type": "Pie Chart",
            "description": "Show proportions of categories",
            "confidence": 0.7,
            "suggested_x": categorical_cols[0],
            "suggested_y": "Count",
            "icon": "🥧",
            "color_scheme": "Sunset"
        })
    
    # Sort by confidence
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    
    return recommendations[:6]  # Return top 6 recommendations

def create_enhanced_chart(chart_type, df, x_col, y_col, color_col=None, palette="Professional Blue", custom_prompt=""):
    """Create enhanced charts with professional styling"""
    
    colors = COLOR_PALETTES[palette]
    
    fig = None
    code = ""
    
    if chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        color_discrete_sequence=colors,
                        title=f"Scatter Plot: {x_col} vs {y_col}",
                        template="plotly_white")
        
        code = f"""
import plotly.express as px

fig = px.scatter(df, x='{x_col}', y='{y_col}', 
                color='{color_col}' if '{color_col}' != 'None' else None,
                color_discrete_sequence={colors},
                title="Scatter Plot: {x_col} vs {y_col}",
                template="plotly_white")

fig.update_layout(
    font=dict(size=12),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
fig.show()
"""
    
    elif chart_type == "Bar Chart":
        if df[x_col].dtype == 'object':
            agg_df = df.groupby(x_col)[y_col].mean().reset_index()
        else:
            agg_df = df
            
        fig = px.bar(agg_df, x=x_col, y=y_col,
                    color_discrete_sequence=colors,
                    title=f"Bar Chart: {y_col} by {x_col}",
                    template="plotly_white")
        
        code = f"""
import plotly.express as px

# Aggregate data if needed
if df['{x_col}'].dtype == 'object':
    agg_df = df.groupby('{x_col}')['{y_col}'].mean().reset_index()
else:
    agg_df = df

fig = px.bar(agg_df, x='{x_col}', y='{y_col}',
            color_discrete_sequence={colors},
            title="Bar Chart: {y_col} by {x_col}",
            template="plotly_white")
fig.show()
"""
    
    elif chart_type == "Box Plot":
        fig = px.box(df, x=x_col, y=y_col, color=color_col,
                    color_discrete_sequence=colors,
                    title=f"Box Plot: {y_col} distribution by {x_col}",
                    template="plotly_white")
        
        code = f"""
import plotly.express as px

fig = px.box(df, x='{x_col}', y='{y_col}', 
            color='{color_col}' if '{color_col}' != 'None' else None,
            color_discrete_sequence={colors},
            title="Box Plot: {y_col} distribution by {x_col}",
            template="plotly_white")
fig.show()
"""
    
    elif chart_type == "Time Series":
        fig = px.line(df, x=x_col, y=y_col, color=color_col,
                     color_discrete_sequence=colors,
                     title=f"Time Series: {y_col} over {x_col}",
                     template="plotly_white")
        
        code = f"""
import plotly.express as px

# Convert to datetime if needed
df['{x_col}'] = pd.to_datetime(df['{x_col}'])

fig = px.line(df, x='{x_col}', y='{y_col}',
             color='{color_col}' if '{color_col}' != 'None' else None,
             color_discrete_sequence={colors},
             title="Time Series: {y_col} over {x_col}",
             template="plotly_white")
fig.show()
"""
    
    elif chart_type == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, 
                       color_continuous_scale="RdBu_r",
                       title="Correlation Heatmap",
                       template="plotly_white")
        
        code = f"""
import plotly.express as px

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

fig = px.imshow(corr_matrix, 
               color_continuous_scale="RdBu_r",
               title="Correlation Heatmap",
               template="plotly_white")
fig.show()
"""
    
    elif chart_type == "Pie Chart":
        value_counts = df[x_col].value_counts().head(10)
        fig = px.pie(values=value_counts.values, names=value_counts.index,
                    color_discrete_sequence=colors,
                    title=f"Distribution of {x_col}",
                    template="plotly_white")
        
        code = f"""
import plotly.express as px

value_counts = df['{x_col}'].value_counts().head(10)
fig = px.pie(values=value_counts.values, names=value_counts.index,
            color_discrete_sequence={colors},
            title="Distribution of {x_col}",
            template="plotly_white")
fig.show()
"""
    
    # Apply custom styling
    if fig:
        fig.update_layout(
            font=dict(size=12, family="Arial, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title=dict(x=0.5, font=dict(size=16, color='#2c3e50')),
            margin=dict(t=60, l=60, r=60, b=60)
        )
        
        # Add custom prompt modifications if provided
        if custom_prompt:
            prompt_modifications = f"""
            
# Custom modifications based on user request: "{custom_prompt}"
# Add any specific styling or modifications here
"""
            code += prompt_modifications
    
    return fig, code

# Main interface
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 AI Chart Recommendations")
    
    user_intent = st.text_input("💭 What story do you want to tell?", 
                               placeholder="e.g., show trends, compare categories, find outliers...")
    
    recommendations = get_chart_recommendations(df, user_intent)
    
    st.markdown("### 🤖 Suggested Charts")
    for i, rec in enumerate(recommendations):
        with st.container():
            st.markdown(f"""
            <div class="chart-option">
                <h4>{rec['icon']} {rec['type']}</h4>
                <p>{rec['description']}</p>
                <small>Confidence: {rec['confidence']:.0%} | Colors: {rec['color_scheme']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Use {rec['type']}", key=f"rec_{i}"):
                st.session_state.selected_chart = rec
                st.session_state.auto_x = rec['suggested_x']
                st.session_state.auto_y = rec['suggested_y']
                st.session_state.auto_palette = rec['color_scheme']

with col2:
    st.subheader("⚙️ Chart Configuration")
    
    # Chart type selection
    chart_types = ["Scatter Plot", "Bar Chart", "Box Plot", "Time Series", "Correlation Heatmap", "Pie Chart", "Violin Plot", "Histogram"]
    selected_chart_type = st.selectbox("📊 Chart Type:", chart_types, 
                                      index=chart_types.index(st.session_state.get('selected_chart', {}).get('type', 'Scatter Plot')))
    
    # Column selection
    col_a, col_b = st.columns(2)
    with col_a:
        x_axis = st.selectbox("X-axis:", df.columns.tolist(), 
                             index=df.columns.tolist().index(st.session_state.get('auto_x', df.columns[0])) if st.session_state.get('auto_x') in df.columns else 0)
    
    with col_b:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            y_axis = st.selectbox("Y-axis:", numeric_cols,
                                 index=numeric_cols.index(st.session_state.get('auto_y', numeric_cols[0])) if st.session_state.get('auto_y') in numeric_cols else 0)
        else:
            y_axis = st.selectbox("Y-axis:", df.columns.tolist())
    
    # Color and grouping
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    color_by = st.selectbox("🎨 Color By:", ["None"] + categorical_cols)
    
    # Color palette selection
    palette_name = st.selectbox("🎨 Color Palette:", list(COLOR_PALETTES.keys()),
                               index=list(COLOR_PALETTES.keys()).index(st.session_state.get('auto_palette', 'Professional Blue')))
    
    # Display color palette preview
    st.markdown("**Color Preview:**")
    palette_html = '<div class="color-palette">'
    for color in COLOR_PALETTES[palette_name][:8]:
        palette_html += f'<div class="color-box" style="background-color: {color}"></div>'
    palette_html += '</div>'
    st.markdown(palette_html, unsafe_allow_html=True)
    
    # Custom styling prompt
    custom_prompt = st.text_area("✨ Additional Styling Instructions:", 
                                placeholder="e.g., add trend line, use log scale, highlight outliers, add annotations...")
    
    # Generate chart
    if st.button("🚀 Generate Chart", type="primary"):
        with st.spinner("Creating your visualization..."):
            fig, code = create_enhanced_chart(
                selected_chart_type, df, x_axis, y_axis, 
                color_by if color_by != "None" else None,
                palette_name, custom_prompt
            )
            
            if fig:
                st.session_state.current_fig = fig
                st.session_state.current_code = code
                st.session_state.chart_generated = True

# Display generated chart
if st.session_state.get('chart_generated', False):
    st.subheader("📊 Generated Visualization")
    st.plotly_chart(st.session_state.current_fig, use_container_width=True)
    
    # AI-generated insights
    with st.spinner("Generating AI insights..."):
        insight_prompt = f"""
        Analyze this {selected_chart_type} chart showing {x_axis} vs {y_axis} from the dataset.
        
        Dataset info:
        - Shape: {df.shape}
        - Columns: {list(df.columns)}
        
        Provide 3-5 specific insights about:
        1. Key patterns or trends visible
        2. Outliers or interesting data points
        3. Business implications
        4. Suggested follow-up analyses
        
        Be specific and actionable. Include actual numbers where possible.
        """
        
        insights = llm.invoke(insight_prompt)
        
        st.markdown(f"""
        <div class="recommendation-card">
            <h3>🧠 AI-Generated Insights</h3>
            {insights}
        </div>
        """, unsafe_allow_html=True)
    
    # Code and export options
    with st.expander("📋 View Generated Code", expanded=False):
        st.code(st.session_state.current_code, language="python")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("💾 Save Chart"):
            add_chart_card(dataset_id, f"Chart: {selected_chart_type}", insights, st.session_state.current_code)
            st.success("✅ Chart saved to history!")
    
    with col_b:
        if st.button("📥 Download PNG"):
            st.info("PNG download would be implemented here")
    
    with col_c:
        if st.button("📊 Create Dashboard"):
            st.info("Dashboard creation would be implemented here")

# Chart gallery and history
st.subheader("🖼️ Chart Gallery & Inspiration")
with st.expander("View Chart Examples", expanded=False):
    example_charts = [
        {"name": "Sales Trend", "type": "Time Series", "description": "Monthly sales performance over time"},
        {"name": "Customer Segments", "type": "Pie Chart", "description": "Distribution of customer types"},
        {"name": "Performance Comparison", "type": "Box Plot", "description": "Compare metrics across departments"},
        {"name": "Correlation Analysis", "type": "Heatmap", "description": "Relationships between variables"}
    ]
    
    for chart in example_charts:
        st.markdown(f"**{chart['name']}** ({chart['type']}): {chart['description']}")

# Navigation hint
st.markdown("---")
st.info("💡 **Pro Tip:** Use the AI recommendations to get started quickly, then customize with your own styling preferences!")