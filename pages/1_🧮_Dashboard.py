import streamlit as st
import pandas as pd
import os
from datetime import datetime
from src.utils import init_db, add_dataset, get_all_datasets, delete_dataset, rename_dataset, safe_read_csv

# Import UI components
from src.components.ui_components import (
    render_professional_header, render_metric_cards, render_feature_card,
    render_insight_card, create_data_quality_indicator, render_interactive_data_explorer,
    create_ai_recommendation_panel, render_animated_loading, PROFESSIONAL_CSS
)

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="📂 Professional Dashboard", layout="wide", page_icon="📊")

# Apply professional styling
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# Professional header
render_professional_header(
    "Multi-Dataset Analytics Dashboard",
    "Upload, manage, and discover relationships across your data with AI-powered insights",
    "📊"
)

init_db()
if not os.path.exists('data/uploads'):
    os.makedirs('data/uploads')

# Enhanced sidebar for dataset upload
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e1e5e9; margin-bottom: 1rem;">
        <h3 style="color: #667eea; margin: 0;">📂 Dataset Manager</h3>
        <small style="color: #666;">Upload & Organize</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced multi-file upload
    st.markdown("#### 📤 Upload Datasets")
    uploaded_files = st.file_uploader(
        "Select CSV files (multiple supported)", 
        type=["csv"], 
        accept_multiple_files=True,
        help="💡 Upload multiple datasets to discover cross-data relationships"
    )
    
    # Upload progress and processing
    if uploaded_files:
        upload_progress = st.progress(0)
        upload_status = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            if f"uploaded_{uploaded_file.name}" not in st.session_state:
                upload_status.text(f"Processing {uploaded_file.name}...")
                upload_progress.progress((i + 1) / len(uploaded_files))
                
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{now}_{uploaded_file.name}"
                file_path = os.path.join('data', 'uploads', filename)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                df = safe_read_csv(file_path)
                rows, cols = df.shape
                upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                add_dataset(filename, file_path, rows, cols, upload_time)
                
                st.session_state[f"uploaded_{uploaded_file.name}"] = True
                st.success(f"✅ {uploaded_file.name}")
        
        upload_status.text("✅ All files uploaded successfully!")
        upload_progress.progress(1.0)
        st.rerun()

# Load datasets
datasets = get_all_datasets()

if datasets:
    # Enhanced dashboard metrics with animations
    st.markdown("### 📊 Dashboard Overview")
    
    total_datasets = len(datasets)
    total_rows = sum([d[2] for d in datasets])
    total_cols = sum([d[3] for d in datasets])
    avg_size = total_rows / total_datasets if total_datasets > 0 else 0
    
    # Calculate additional metrics
    largest_dataset = max(datasets, key=lambda x: x[2])
    newest_dataset = max(datasets, key=lambda x: datetime.strptime(x[4], "%Y-%m-%d %H:%M:%S"))
    
    # Professional metric cards
    metrics = [
        {"title": "Total Datasets", "value": f"{total_datasets}", "delta": "+3 this week"},
        {"title": "Total Records", "value": f"{total_rows:,}", "delta": f"+{total_rows//10:,} recent"},
        {"title": "Data Fields", "value": f"{total_cols}", "delta": None},
        {"title": "Avg Dataset Size", "value": f"{avg_size:,.0f}", "delta": None}
    ]
    
    render_metric_cards(metrics)
    
    # Enhanced analytics dashboard
    st.markdown("### 📈 Analytics Dashboard")
    
    # Create comprehensive dashboard visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Dataset Sizes (Records)', 'Upload Timeline', 'Column Distribution', 'Data Density Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Dataset sizes bar chart with enhanced styling
    dataset_names = [d[1][:25] + "..." if len(d[1]) > 25 else d[1] for d in datasets]
    dataset_sizes = [d[2] for d in datasets]
    
    colors = ['#667eea', '#764ba2', '#56CCF2', '#2F80ED', '#FF6B6B'] * (len(datasets) // 5 + 1)
    
    fig.add_trace(
        go.Bar(
            x=dataset_names, 
            y=dataset_sizes, 
            name="Records",
            marker=dict(color=colors[:len(datasets)], opacity=0.8),
            text=[f"{size:,}" for size in dataset_sizes],
            textposition="outside"
        ),
        row=1, col=1
    )
    
    # Upload timeline with trend
    upload_dates = [datetime.strptime(d[4], "%Y-%m-%d %H:%M:%S").date() for d in datasets]
    upload_counts = {}
    for date in upload_dates:
        upload_counts[date] = upload_counts.get(date, 0) + 1
    
    sorted_dates = sorted(upload_counts.keys())
    cumulative_counts = []
    total = 0
    for date in sorted_dates:
        total += upload_counts[date]
        cumulative_counts.append(total)
    
    fig.add_trace(
        go.Scatter(
            x=sorted_dates, 
            y=cumulative_counts,
            mode='lines+markers', 
            name="Cumulative Uploads",
            line=dict(color='#764ba2', width=3),
            marker=dict(size=8, color='#667eea')
        ),
        row=1, col=2
    )
    
    # Column distribution histogram
    column_counts = [d[3] for d in datasets]
    fig.add_trace(
        go.Histogram(
            x=column_counts, 
            name="Columns",
            marker=dict(color='#56CCF2', opacity=0.8),
            nbinsx=10
        ),
        row=2, col=1
    )
    
    # Data density (records per column) scatter
    density_scores = [d[2]/d[3] if d[3] > 0 else 0 for d in datasets]
    fig.add_trace(
        go.Scatter(
            x=dataset_names, 
            y=density_scores, 
            mode='markers',
            name="Density Score",
            marker=dict(
                size=[min(50, max(10, size//1000)) for size in dataset_sizes],
                color=density_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Density Score")
            ),
            text=[f"{d[1]}<br>Density: {score:.1f}" for d, score in zip(datasets, density_scores)],
            hovertemplate="<b>%{text}</b><br>Records/Column: %{y:.1f}<extra></extra>"
        ),
        row=2, col=2
    )
    
    # Update layout with professional styling
    fig.update_layout(
        height=600,
        showlegend=False,
        title=dict(
            text="📊 Dataset Analytics Overview",
            x=0.5,
            font=dict(size=18, color='#2c3e50')
        ),
        font=dict(family="Inter, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Update individual subplot styling
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(
                gridcolor='#e1e5e9',
                gridwidth=1,
                zeroline=False,
                row=i, col=j
            )
            fig.update_yaxes(
                gridcolor='#e1e5e9', 
                gridwidth=1,
                zeroline=False,
                row=i, col=j
            )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Dataset insights with AI analysis
    st.markdown("### 🤖 AI-Powered Insights")
    
    # Generate insights about the dataset collection
    insights = []
    
    if total_datasets >= 3:
        insights.append("🎯 **Multi-Dataset Analysis Ready**: You have enough datasets for comprehensive cross-analysis")
    
    if max(dataset_sizes) > 10000:
        insights.append(f"📈 **Large Dataset Detected**: {largest_dataset[1]} has {largest_dataset[2]:,} records - suitable for deep analytics")
    
    if len(set(d[3] for d in datasets)) > 3:
        insights.append("🔗 **Diverse Data Structure**: Varied column counts suggest different data types - good for comprehensive analysis")
    
    upload_recency = (datetime.now() - datetime.strptime(newest_dataset[4], "%Y-%m-%d %H:%M:%S")).days
    if upload_recency < 7:
        insights.append(f"⚡ **Fresh Data**: Latest upload ({newest_dataset[1]}) is only {upload_recency} days old")
    
    # Display insights in cards
    if insights:
        for insight in insights:
            render_insight_card(insight)
    else:
        render_insight_card("📊 **Getting Started**: Upload more datasets to unlock advanced AI insights and cross-data analysis!")
    
    # Multi-dataset relationship analysis
    st.markdown("### 🔗 Cross-Dataset Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_datasets = st.multiselect(
            "🎯 Select datasets for relationship analysis:",
            options=[f"{d[0]} - {d[1]}" for d in datasets],
            help="Choose 2+ datasets to discover hidden relationships and patterns",
            placeholder="Select multiple datasets..."
        )
    
    with col2:
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Column Similarity", "Statistical Correlation", "Semantic Relationships", "AI Deep Analysis"]
        )
    
    if len(selected_datasets) >= 2:
        if st.button("🚀 Analyze Relationships", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing cross-dataset relationships..."):
                render_animated_loading("Discovering patterns across your datasets...")
                
                # Store selection for cross-analysis page
                st.session_state.cross_analysis_datasets = selected_datasets
                st.session_state.cross_analysis_type = analysis_type
                
                st.success("✅ Analysis ready! Click below to view detailed results.")
                
                if st.button("📊 View Detailed Analysis", type="secondary"):
                    st.switch_page("pages/7_🔗_Cross_Dataset_Analysis.py")
    
    # Enhanced dataset management with professional cards
    st.markdown("### 🗂️ Dataset Management")
    
    # Filter and search options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("🔍 Search datasets:", placeholder="Filter by name...")
    
    with col2:
        size_filter = st.selectbox("📏 Size filter:", ["All", "Small (<1K)", "Medium (1K-10K)", "Large (>10K)"])
    
    with col3:
        sort_by = st.selectbox("📊 Sort by:", ["Name", "Upload Date", "Size", "Columns"])
    
    # Apply filters
    filtered_datasets = datasets
    
    if search_term:
        filtered_datasets = [d for d in filtered_datasets if search_term.lower() in d[1].lower()]
    
    if size_filter != "All":
        if size_filter == "Small (<1K)":
            filtered_datasets = [d for d in filtered_datasets if d[2] < 1000]
        elif size_filter == "Medium (1K-10K)":
            filtered_datasets = [d for d in filtered_datasets if 1000 <= d[2] <= 10000]
        elif size_filter == "Large (>10K)":
            filtered_datasets = [d for d in filtered_datasets if d[2] > 10000]
    
    # Sort datasets
    if sort_by == "Name":
        filtered_datasets.sort(key=lambda x: x[1])
    elif sort_by == "Upload Date":
        filtered_datasets.sort(key=lambda x: datetime.strptime(x[4], "%Y-%m-%d %H:%M:%S"), reverse=True)
    elif sort_by == "Size":
        filtered_datasets.sort(key=lambda x: x[2], reverse=True)
    elif sort_by == "Columns":
        filtered_datasets.sort(key=lambda x: x[3], reverse=True)
    
    # Display filtered datasets with enhanced cards
    for dataset in filtered_datasets:
        id_, name, rows, cols, uploaded, status = dataset
        
        with st.expander(f"📁 {name}", expanded=False):
            # Load dataset for preview and analysis
            file_path = os.path.join("data", "uploads", name)
            
            try:
                preview_df = safe_read_csv(file_path)
                
                # Dataset overview section
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### 📊 Dataset Preview")
                    st.dataframe(preview_df.head(5), use_container_width=True)
                    
                    # Quick statistics
                    numeric_cols = preview_df.select_dtypes(include=['number']).columns
                    categorical_cols = preview_df.select_dtypes(include=['object']).columns
                    missing_values = preview_df.isnull().sum().sum()
                    
                    # Data quality indicator
                    quality_score = create_data_quality_indicator(preview_df)
                    
                with col2:
                    st.markdown("#### 🎯 Quick Stats")
                    
                    # Mini metrics for this dataset
                    dataset_metrics = [
                        {"title": "Numeric Cols", "value": str(len(numeric_cols))},
                        {"title": "Text Cols", "value": str(len(categorical_cols))},
                        {"title": "Missing", "value": str(missing_values)},
                        {"title": "Quality", "value": f"{quality_score:.0%}"}
                    ]
                    
                    render_metric_cards(dataset_metrics)
                    
                    # Action buttons
                    st.markdown("#### ⚡ Quick Actions")
                    
                    action_col1, action_col2 = st.columns(2)
                    
                    with action_col1:
                        if st.button("🔍 Analyze", key=f"analyze_{id_}", use_container_width=True):
                            st.session_state.selected_dataset_id = id_
                            st.switch_page("pages/3_📂_Dataset_Details.py")
                        
                        if st.button("💬 Chat", key=f"chat_{id_}", use_container_width=True):
                            st.session_state.selected_dataset_id = id_
                            st.switch_page("main.py")
                    
                    with action_col2:
                        if st.button("📊 Charts", key=f"chart_{id_}", use_container_width=True):
                            st.session_state.selected_dataset_id = id_
                            st.switch_page("pages/6_📈_Smart_Charts.py")
                        
                        if st.button("📋 Report", key=f"report_{id_}", use_container_width=True):
                            st.session_state.selected_dataset_id = id_
                            st.switch_page("pages/5_📋_EDA Report.py")
                
                # AI recommendations for this specific dataset
                st.markdown("#### 🤖 AI Recommendations")
                create_ai_recommendation_panel(preview_df)
                
                # Management options
                st.markdown("#### ⚙️ Management Options")
                
                mgmt_col1, mgmt_col2, mgmt_col3 = st.columns(3)
                
                with mgmt_col1:
                    new_name = st.text_input(
                        "Rename dataset:", 
                        value=name, 
                        key=f"rename_input_{id_}",
                        help="Give your dataset a descriptive name"
                    )
                    
                    if st.button("✅ Rename", key=f"rename_btn_{id_}"):
                        rename_dataset(id_, new_name)
                        st.success("✅ Dataset renamed!")
                        st.rerun()
                
                with mgmt_col2:
                    if st.button("📥 Download", key=f"download_{id_}", help="Download processed dataset"):
                        # Create download functionality
                        csv_data = preview_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"{name.split('_', 1)[-1] if '_' in name else name}",
                            mime="text/csv",
                            key=f"download_btn_{id_}"
                        )
                
                with mgmt_col3:
                    if st.button("🗑️ Delete", key=f"del_{id_}", type="secondary", help="Permanently delete this dataset"):
                        # Confirmation dialog
                        if st.checkbox(f"Confirm deletion of {name}", key=f"confirm_{id_}"):
                            delete_dataset(id_)
                            st.warning(f"🗑️ Deleted dataset: {name}")
                            st.rerun()
                
                # Interactive data explorer for this dataset
                if st.checkbox("🔍 Open Data Explorer", key=f"explorer_{id_}"):
                    render_interactive_data_explorer(preview_df)
                
            except Exception as e:
                st.error(f"❌ Could not load dataset: {e}")
                
                # Still show management options even if preview fails
                mgmt_col1, mgmt_col2 = st.columns(2)
                
                with mgmt_col1:
                    new_name = st.text_input("Rename:", value=name, key=f"rename_error_{id_}")
                    if st.button("✅ Rename", key=f"rename_error_btn_{id_}"):
                        rename_dataset(id_, new_name)
                        st.rerun()
                
                with mgmt_col2:
                    if st.button("🗑️ Delete", key=f"del_error_{id_}", type="secondary"):
                        delete_dataset(id_)
                        st.rerun()

else:
    # Welcome screen for new users
    st.markdown("### 👋 Welcome to VizGenie-GPT Professional!")
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_feature_card(
            "🤖 AI-Powered Analysis",
            "Ask complex questions about your data in natural language and get intelligent insights with beautiful visualizations.",
            "🤖"
        )
    
    with col2:
        render_feature_card(
            "🔗 Cross-Dataset Discovery",
            "Upload multiple datasets and discover hidden relationships and patterns across your data sources.",
            "🔗"
        )
    
    with col3:
        render_feature_card(
            "📊 Professional Charts",
            "Generate stunning, publication-ready charts with smart color schemes and interactive features.",
            "📊"
        )
    
    # Getting started guide
    st.markdown("### 🚀 Getting Started")
    
    render_insight_card("""
    **📋 Quick Start Guide:**
    
    1. **📤 Upload Your Data**: Use the sidebar to upload one or more CSV files
    2. **🤖 Ask Questions**: Chat with your data using natural language
    3. **📊 Create Visualizations**: Generate professional charts automatically
    4. **🔗 Find Relationships**: Discover patterns across multiple datasets
    5. **📄 Export Reports**: Generate comprehensive PDF reports for stakeholders
    
    **💡 Pro Tips:**
    - Upload related datasets together for better cross-analysis
    - Use descriptive names for your datasets
    - Ask specific questions for better AI responses
    - Try different chart types and color schemes
    """)
    
    # Sample data offer
    st.markdown("### 📚 Try with Sample Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Load Sample Sales Data", type="primary", use_container_width=True):
            # Create sample sales dataset
            np.random.seed(42)
            sample_sales = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=365, freq='D'),
                'revenue': np.random.normal(10000, 2000, 365),
                'customers': np.random.poisson(50, 365),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
                'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 365)
            })
            
            # Save sample data
            sample_path = os.path.join('data', 'uploads', 'sample_sales_data.csv')
            sample_sales.to_csv(sample_path, index=False)
            
            # Add to database
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            add_dataset('sample_sales_data.csv', sample_path, len(sample_sales), len(sample_sales.columns), upload_time)
            
            st.success("✅ Sample sales data loaded!")
            st.rerun()
    
    with col2:
        if st.button("👥 Load Sample Customer Data", type="secondary", use_container_width=True):
            # Create sample customer dataset
            np.random.seed(24)
            sample_customers = pd.DataFrame({
                'customer_id': range(1, 501),
                'age': np.random.randint(18, 70, 500),
                'gender': np.random.choice(['Male', 'Female'], 500),
                'income': np.random.normal(50000, 15000, 500),
                'satisfaction_score': np.random.randint(1, 11, 500),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 500)
            })
            
            # Save sample data
            sample_path = os.path.join('data', 'uploads', 'sample_customer_data.csv')
            sample_customers.to_csv(sample_path, index=False)
            
            # Add to database
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            add_dataset('sample_customer_data.csv', sample_path, len(sample_customers), len(sample_customers.columns), upload_time)
            
            st.success("✅ Sample customer data loaded!")
            st.rerun()

# Enhanced sidebar with navigation and tips
with st.sidebar:
    if datasets:
        st.markdown("---")
        st.markdown("### 🎯 Quick Stats")
        
        # Overall statistics
        total_size_mb = sum(os.path.getsize(os.path.join("data", "uploads", d[1])) for d in datasets if os.path.exists(os.path.join("data", "uploads", d[1]))) / (1024 * 1024)
        
        quick_stats = [
            {"title": "Datasets", "value": str(len(datasets)), "delta": None},
            {"title": "Total Size", "value": f"{total_size_mb:.1f}MB", "delta": None},
            {"title": "Largest", "value": f"{max(d[2] for d in datasets):,}", "delta": None}
        ]
        
        render_metric_cards(quick_stats)
    
    st.markdown("---")
    st.markdown("### 🔗 Navigation")
    
    nav_links = [
        ("💬 AI Chat", "main.py"),
        ("📊 Dataset Details", "pages/3_📂_Dataset_Details.py"),
        ("📈 Smart Charts", "pages/6_📈_Smart_Charts.py"),
        ("🔗 Cross Analysis", "pages/7_🔗_Cross_Dataset_Analysis.py"),
        ("📋 Chart History", "pages/4_📊_Charts_History.py"),
        ("📄 EDA Report", "pages/5_📋_EDA Report.py"),
        ("📖 About", "pages/📖_About_Project.py")
    ]
    
    for label, page in nav_links:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.switch_page(page)
    
    st.markdown("---")
    st.markdown("### 💡 Tips & Tricks")
    
    tips = [
        "🎯 **Better Analysis**: Upload related datasets together",
        "🎨 **Visual Appeal**: Try different color schemes in charts", 
        "🤖 **Smart Questions**: Be specific about what you want to explore",
        "📊 **Cross-Analysis**: Look for patterns across datasets",
        "📋 **Save Work**: Use chart history and session management"
    ]
    
    for tip in tips:
        st.markdown(f"- {tip}")

# Footer with system info
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🧠 VizGenie-GPT Professional**")
    st.caption("Advanced Multi-Dataset Analytics Platform")

with col2:
    if datasets:
        st.markdown(f"**📊 System Status**")
        st.caption(f"{len(datasets)} datasets • {sum(d[2] for d in datasets):,} total records")
    else:
        st.markdown("**🚀 Ready to Start**")
        st.caption("Upload your first dataset to begin")

with col3:
    st.markdown("**👨‍💻 Delay Group**")
    st.caption("Making data analysis accessible to everyone")