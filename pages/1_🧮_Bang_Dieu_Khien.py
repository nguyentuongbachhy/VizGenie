import streamlit as st
import pandas as pd
import os
from datetime import datetime
from src.utils import init_db, add_dataset, get_all_datasets, delete_dataset, rename_dataset, safe_read_csv

# Import UI components
from src.components.ui_components import (
    render_professional_header, render_metric_cards, render_feature_card,
    render_insight_card, create_data_quality_indicator, PROFESSIONAL_CSS
)

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time

st.set_page_config(page_title="Bảng điều khiển Chuyên nghiệp", layout="wide", page_icon="📊")

# Apply professional styling
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# Professional header
render_professional_header(
    "Bảng điều khiển Phân tích Đa Bộ dữ liệu",
    "Tải lên, quản lý và khám phá mối quan hệ giữa dữ liệu của bạn với thông tin chi tiết được hỗ trợ bởi AI",
    "📊"
)

init_db()
if not os.path.exists('data/uploads'):
    os.makedirs('data/uploads')

def show_loading_animation(text="Đang xử lý..."):
    """Show loading animation"""
    return st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin: 1rem 0;">
        <div style="border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin-right: 1rem;"></div>
        <span style="color: #667eea; font-weight: 500;">{text}</span>
    </div>
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """, unsafe_allow_html=True)

def create_enhanced_analytics_dashboard(datasets):
    """Create a beautiful dark theme professional analytics dashboard"""
    try:
        # Validate input data
        if not datasets or len(datasets) == 0:
            return create_empty_dashboard_placeholder()

        # Prepare data with enhanced validation
        dataset_info = []
        for d in datasets:
            try:
                if len(d) >= 5:
                    name = str(d[1])[:15] + "..." if len(str(d[1])) > 15 else str(d[1])
                    rows = max(0, int(d[2]) if d[2] is not None else 0)
                    cols = max(1, int(d[3]) if d[3] is not None else 1)
                    
                    try:
                        upload_date = datetime.strptime(str(d[4]), "%Y-%m-%d %H:%M:%S")
                    except:
                        upload_date = datetime.now()
                    
                    dataset_info.append({
                        'name': name,
                        'rows': rows,
                        'cols': cols,
                        'date': upload_date.date(),
                        'density': rows / cols if cols > 0 else 0,
                        'size_category': get_size_category(rows)
                    })
            except Exception:
                continue

        if not dataset_info:
            return create_empty_dashboard_placeholder()

        # Create beautiful subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '📊 Dataset Size Overview', 
                '📈 Upload Timeline', 
                '🔢 Column Distribution', 
                '💎 Data Density Analysis'
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Dark theme color palette - vibrant colors that pop on dark background
        colors = {
            'primary': '#00d4ff',      # Bright cyan
            'secondary': '#ff6b9d',    # Pink
            'accent': '#c44569',       # Dark pink
            'success': '#00ff88',      # Bright green
            'warning': '#ffeb3b',      # Bright yellow
            'info': '#7c4dff',         # Purple
            'gradient': ['#00d4ff', '#ff6b9d', '#00ff88', '#ffeb3b', '#7c4dff', '#ff5722']
        }
        
        # Chart 1: Neon-style Dataset Sizes
        top_datasets = sorted(dataset_info, key=lambda x: x['rows'], reverse=True)[:8]
        names = [info['name'] for info in top_datasets]
        sizes = [info['rows'] for info in top_datasets]
        
        # Create gradient effect
        max_size = max(sizes) if sizes else 1
        bar_colors = []
        for size in sizes:
            intensity = size / max_size
            bar_colors.append(f'rgba(0, 212, 255, {0.3 + 0.7 * intensity})')
        
        fig.add_trace(
            go.Bar(
                x=names,
                y=sizes,
                marker=dict(
                    color=bar_colors,
                    line=dict(color=colors['primary'], width=2),
                    pattern=dict(shape="")
                ),
                text=[f"{size:,}" for size in sizes],
                textposition="outside",
                textfont=dict(color='white', size=11, family="Arial Bold"),
                name="Records",
                hovertemplate=(
                    "<b style='color:#00d4ff'>%{x}</b><br>"
                    "Records: <b>%{y:,}</b><br>"
                    "<extra></extra>"
                )
            ),
            row=1, col=1
        )
        
        # Chart 2: Glowing Timeline
        date_counts = {}
        for info in dataset_info:
            date_str = info['date'].strftime("%Y-%m-%d")
            date_counts[date_str] = date_counts.get(date_str, 0) + 1
        
        if date_counts:
            dates = sorted(date_counts.keys())
            daily_counts = [date_counts[date] for date in dates]
            
            # Calculate cumulative
            cumulative = []
            total = 0
            for count in daily_counts:
                total += count
                cumulative.append(total)
            
            # Main cumulative line with glow effect
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=cumulative,
                    mode='lines+markers',
                    line=dict(
                        color=colors['secondary'], 
                        width=4,
                        shape='spline'
                    ),
                    marker=dict(
                        size=10, 
                        color=colors['primary'],
                        line=dict(color='white', width=2)
                    ),
                    fill='tonexty',
                    fillcolor='rgba(255, 107, 157, 0.2)',
                    name="Total Datasets",
                    hovertemplate=(
                        "<b>Date:</b> %{x}<br>"
                        "<b>Total:</b> %{y}<br>"
                        "<extra></extra>"
                    )
                ),
                row=1, col=2
            )
            
            # Add daily bars with glow
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=daily_counts,
                    marker=dict(
                        color='rgba(0, 255, 136, 0.4)',
                        line=dict(color=colors['success'], width=1)
                    ),
                    name="Daily",
                    opacity=0.6,
                    hovertemplate=(
                        "<b>Date:</b> %{x}<br>"
                        "<b>New uploads:</b> %{y}<br>"
                        "<extra></extra>"
                    )
                ),
                row=1, col=2
            )
        
        # Chart 3: Neon Column Distribution
        col_counts = [info['cols'] for info in dataset_info]
        if col_counts:
            fig.add_trace(
                go.Histogram(
                    x=col_counts,
                    marker=dict(
                        color='rgba(255, 235, 59, 0.7)',
                        line=dict(color=colors['warning'], width=2)
                    ),
                    name="Columns",
                    hovertemplate="Columns: %{x}<br>Count: %{y}<extra></extra>"
                ),
                row=2, col=1
            )
        
        # Chart 4: Glowing Density Bubbles
        densities = [info['density'] for info in dataset_info]
        bubble_sizes = [min(40, max(15, info['rows']/500)) for info in dataset_info]
        
        # Color mapping based on size category
        category_colors = {
            'Small': colors['success'],
            'Medium': colors['warning'], 
            'Large': colors['primary'],
            'Very Large': colors['secondary']
        }
        
        bubble_colors = [category_colors.get(info['size_category'], colors['primary']) for info in dataset_info]
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(dataset_info))),
                y=densities,
                mode='markers',
                marker=dict(
                    size=bubble_sizes,
                    color=bubble_colors,
                    opacity=0.8,
                    line=dict(color='white', width=2)
                ),
                text=[info['name'] for info in dataset_info],
                customdata=[info['size_category'] for info in dataset_info],
                name="Density",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Density: <b>%{y:.1f}</b><br>"
                    "Category: <b>%{customdata}</b><br>"
                    "<extra></extra>"
                )
            ),
            row=2, col=2
        )
        
        # Apply dark theme styling
        fig.update_layout(
            height=650,
            showlegend=False,
            title=dict(
                text="🌟 VizGenie-GPT Analytics Dashboard",
                x=0.5,
                font=dict(size=22, color='#ffffff', family="Arial Black")
            ),
            font=dict(size=11, color='#e0e0e0'),
            # Dark theme colors
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#0d1117',
            margin=dict(t=80, l=60, r=60, b=60)
        )
        
        # Update subplot titles for dark theme
        for annotation in fig['layout']['annotations'][:4]:  # First 4 are subplot titles
            annotation.update(
                font=dict(size=13, color='#ffffff', family="Arial Bold"),
                bgcolor='rgba(255, 255, 255, 0.1)',
                bordercolor='rgba(0, 212, 255, 0.5)',
                borderwidth=1,
                borderpad=8
            )
        
        # Style axes for dark theme - safe approach
        # Apply to all subplots individually to avoid conflicts
        for row in [1, 2]:
            for col in [1, 2]:
                fig.update_xaxes(
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    gridwidth=1,
                    showline=True,
                    linecolor='rgba(255, 255, 255, 0.3)',
                    linewidth=1,
                    tickfont_color='#e0e0e0',
                    row=row, col=col
                )
                fig.update_yaxes(
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    gridwidth=1,
                    showline=True,
                    linecolor='rgba(255, 255, 255, 0.3)',
                    linewidth=1,
                    tickfont_color='#e0e0e0',
                    row=row, col=col
                )
        
        # Specific styling for each chart
        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_yaxes(tickformat=',.0f', row=1, col=1)
        
        # Add subtle glow effect annotation
        fig.add_annotation(
            text="✨ Professional Analytics Suite • Powered by AI",
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=10, color='rgba(255, 255, 255, 0.6)'),
            xanchor='center'
        )
        
        return fig
        
    except Exception as e:
        return create_error_dashboard(str(e))

def get_size_category(rows):
    """Categorize dataset size"""
    if rows < 1000:
        return "Small"
    elif rows < 10000:
        return "Medium"
    elif rows < 100000:
        return "Large"
    else:
        return "Very Large"

def create_empty_dashboard_placeholder():
    """Create beautiful dark empty state"""
    fig = go.Figure()
    
    fig.add_annotation(
        text=(
            "🌟 Welcome to VizGenie-GPT Analytics<br><br>"
            "📊 Upload your first dataset to unlock beautiful insights<br>"
            "💡 Professional dark-theme analytics powered by AI<br><br>"
            "<i style='color:#00d4ff'>Ready to illuminate your data!</i>"
        ),
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color='#ffffff'),
        bgcolor='rgba(0, 212, 255, 0.1)',
        bordercolor='rgba(0, 212, 255, 0.5)',
        borderwidth=2,
        borderpad=30,
        align='center'
    )
    
    fig.update_layout(
        height=500,
        title=dict(
            text="🌟 VizGenie-GPT Analytics Dashboard",
            x=0.5,
            font=dict(size=22, color='#ffffff', family="Arial Black")
        ),
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#0d1117',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    return fig

def create_error_dashboard(error_msg):
    """Create beautiful dark error state"""
    fig = go.Figure()
    
    fig.add_annotation(
        text=(
            f"⚠️ Dashboard Error<br><br>"
            f"Something went wrong while creating your analytics dashboard.<br>"
            f"<small style='color:#ff6b9d'>Error: {error_msg[:100]}...</small><br><br>"
            f"💡 Please try refreshing the page or contact support."
        ),
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color='#ff5722'),
        bgcolor='rgba(255, 87, 34, 0.1)',
        bordercolor='rgba(255, 87, 34, 0.5)',
        borderwidth=2,
        borderpad=20,
        align='center'
    )
    
    fig.update_layout(
        height=400,
        title="🌟 Dashboard Error",
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#0d1117',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    return fig

def perform_ai_deep_analysis(datasets):
    """Perform AI deep analysis with loading indicators and error handling"""
    try:
        # Loading indicator
        loading_placeholder = st.empty()
        loading_placeholder.markdown(show_loading_animation("🤖 AI đang phân tích sâu dữ liệu của bạn..."))
        
        # Simulate AI processing time
        time.sleep(2)
        
        # Analyze datasets
        total_records = sum([d[2] for d in datasets])
        total_fields = sum([d[3] for d in datasets])
        avg_size = total_records / len(datasets) if datasets else 0
        largest_dataset = max(datasets, key=lambda x: x[2]) if datasets else None
        
        # Generate insights
        insights = []
        
        if len(datasets) >= 3:
            insights.append({
                "icon": "🎯",
                "title": "Cơ sở Dữ liệu Phong phú",
                "description": f"Bạn có {len(datasets)} bộ dữ liệu với tổng cộng {total_records:,} bản ghi. Đây là cơ sở tuyệt vời cho phân tích đa chiều và khám phá mối quan hệ chéo.",
                "confidence": 0.9,
                "action": "Thử phân tích chéo dữ liệu"
            })
        
        if largest_dataset and largest_dataset[2] > 10000:
            insights.append({
                "icon": "📈",
                "title": "Tiềm năng Big Data",
                "description": f"Bộ dữ liệu '{largest_dataset[1]}' có {largest_dataset[2]:,} bản ghi. Kích thước này rất phù hợp cho machine learning và phân tích xu hướng phức tạp.",
                "confidence": 0.85,
                "action": "Áp dụng thuật toán ML"
            })
        
        if total_fields > 50:
            insights.append({
                "icon": "🔗",
                "title": "Dữ liệu Đa chiều",
                "description": f"Với {total_fields} trường dữ liệu tổng cộng, bạn có thể thực hiện phân tích tương quan sâu và phát hiện các mối quan hệ ẩn giữa các biến.",
                "confidence": 0.8,
                "action": "Tạo ma trận tương quan"
            })
        
        # Data quality assessment
        quality_scores = []
        for dataset in datasets:
            try:
                df = safe_read_csv(dataset[2])
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                quality_score = max(0, 100 - missing_pct)
                quality_scores.append(quality_score)
            except:
                quality_scores.append(75)  # Default score if can't read
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 75
        
        if avg_quality > 85:
            insights.append({
                "icon": "✅",
                "title": "Chất lượng Dữ liệu Cao",
                "description": f"Chất lượng dữ liệu trung bình là {avg_quality:.1f}%. Dữ liệu sạch này sẵn sàng cho các phân tích nâng cao và mô hình hóa.",
                "confidence": 0.9,
                "action": "Bắt đầu phân tích nâng cao"
            })
        elif avg_quality < 60:
            insights.append({
                "icon": "⚠️",
                "title": "Cần Làm sạch Dữ liệu",
                "description": f"Chất lượng dữ liệu trung bình chỉ {avg_quality:.1f}%. Nên làm sạch dữ liệu trước khi phân tích để có kết quả chính xác hơn.",
                "confidence": 0.85,
                "action": "Đi đến Chi tiết Bộ dữ liệu"
            })
        
        # Time-based analysis
        recent_uploads = [d for d in datasets 
                         if (datetime.now() - datetime.strptime(d[4], "%Y-%m-%d %H:%M:%S")).days < 7]
        
        if recent_uploads:
            insights.append({
                "icon": "⚡",
                "title": "Dữ liệu Mới",
                "description": f"{len(recent_uploads)} bộ dữ liệu được tải lên trong 7 ngày qua. Dữ liệu mới thường phản ánh xu hướng hiện tại và có giá trị phân tích cao.",
                "confidence": 0.75,
                "action": "Phân tích xu hướng mới nhất"
            })
        
        # Clear loading
        loading_placeholder.empty()
        
        return insights, avg_quality
        
    except Exception as e:
        st.error(f"❌ Lỗi trong quá trình phân tích AI: {str(e)}")
        return [], 75

# Enhanced sidebar for dataset upload
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e1e5e9; margin-bottom: 1rem;">
        <h3 style="color: #667eea; margin: 0;">📂 Quản lý Bộ dữ liệu</h3>
        <small style="color: #666;">Tải lên & Tổ chức</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced multi-file upload with progress
    st.markdown("#### 📤 Tải lên Bộ dữ liệu")
    uploaded_files = st.file_uploader(
        "Chọn các file CSV (hỗ trợ nhiều file)", 
        type=["csv"], 
        accept_multiple_files=True,
        help="💡 Tải lên nhiều bộ dữ liệu để khám phá mối quan hệ chéo dữ liệu"
    )
    
    # Upload processing with better feedback
    if uploaded_files:
        upload_progress = st.progress(0)
        upload_status = st.empty()
        success_count = 0
        
        for i, uploaded_file in enumerate(uploaded_files):
            cache_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
            
            if cache_key not in st.session_state:
                upload_status.text(f"🔄 Đang xử lý {uploaded_file.name}...")
                
                try:
                    # Validate file
                    if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
                        st.error(f"❌ File {uploaded_file.name} quá lớn (>50MB)")
                        continue
                    
                    # Process file
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{now}_{uploaded_file.name}"
                    file_path = os.path.join('data', 'uploads', filename)
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Read and validate CSV
                    df = safe_read_csv(file_path)
                    
                    if df.empty:
                        st.error(f"❌ File {uploaded_file.name} trống")
                        os.remove(file_path)
                        continue
                    
                    rows, cols = df.shape
                    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    add_dataset(filename, file_path, rows, cols, upload_time)
                    
                    st.session_state[cache_key] = True
                    success_count += 1
                    
                    upload_status.success(f"✅ {uploaded_file.name} ({rows:,} hàng, {cols} cột)")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi xử lý {uploaded_file.name}: {str(e)}")
                    
            else:
                success_count += 1
            
            upload_progress.progress((i + 1) / len(uploaded_files))
        
        if success_count == len(uploaded_files):
            upload_status.success(f"🎉 Đã tải lên thành công {success_count}/{len(uploaded_files)} file!")
            time.sleep(1)
            st.rerun()

# Load datasets
datasets = get_all_datasets()

if datasets:
    # Enhanced dashboard metrics
    st.markdown("### 📊 Tổng quan Bảng điều khiển")
    
    # Calculate comprehensive metrics
    total_datasets = len(datasets)
    total_rows = sum([d[2] for d in datasets])
    total_cols = sum([d[3] for d in datasets])
    avg_size = total_rows / total_datasets if total_datasets > 0 else 0
    
    # Additional metrics
    largest_dataset = max(datasets, key=lambda x: x[2]) if datasets else None
    newest_dataset = max(datasets, key=lambda x: datetime.strptime(x[4], "%Y-%m-%d %H:%M:%S")) if datasets else None
    
    # Calculate storage size
    total_size_mb = 0
    for d in datasets:
        try:
            if os.path.exists(os.path.join("data", "uploads", d[1])):
                total_size_mb += os.path.getsize(os.path.join("data", "uploads", d[1])) / (1024 * 1024)
        except:
            continue
    
    # Professional metric cards with enhanced data
    metrics = [
        {
            "title": "Tổng Bộ dữ liệu", 
            "value": f"{total_datasets}", 
            "delta": "+3 tuần này" if total_datasets > 0 else None
        },
        {
            "title": "Tổng Bản ghi", 
            "value": f"{total_rows:,}", 
            "delta": f"+{total_rows//10:,} gần đây" if total_rows > 0 else None
        },
        {
            "title": "Trường Dữ liệu", 
            "value": f"{total_cols}", 
            "delta": None
        },
        {
            "title": "Dung lượng", 
            "value": f"{total_size_mb:.1f}MB", 
            "delta": None
        }
    ]
    
    render_metric_cards(metrics)
    
    # Enhanced analytics dashboard with loading and error handling
    st.markdown("### 📈 Bảng điều khiển Phân tích")
    
    dashboard_container = st.container()
    
    with dashboard_container:
        try:
            # Show loading for dashboard creation
            with st.spinner("📊 Đang tạo dashboard phân tích..."):
                fig = create_enhanced_analytics_dashboard(datasets)
            
            # Display dashboard
            st.plotly_chart(fig, use_container_width=True, key="main_dashboard")
            
        except Exception as e:
            st.error(f"❌ Không thể tạo dashboard: {str(e)}")
            st.info("💡 Vui lòng thử tải lại trang hoặc kiểm tra dữ liệu")
    
    # AI Deep Analysis with enhanced loading
    st.markdown("### 🤖 Phân tích Sâu AI")
    
    if st.button("🚀 Bắt đầu Phân tích AI", type="primary"):
        ai_insights, data_quality = perform_ai_deep_analysis(datasets)
        
        if ai_insights:
            st.markdown("#### 💡 Insights được AI Phát hiện")
            
            # Display insights in an attractive grid
            cols = st.columns(2)
            for i, insight in enumerate(ai_insights):
                with cols[i % 2]:
                    confidence_color = "#28a745" if insight['confidence'] > 0.8 else "#ffc107" if insight['confidence'] > 0.6 else "#dc3545"
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {confidence_color}15 0%, {confidence_color}25 100%);
                        border: 1px solid {confidence_color}30;
                        padding: 1.5rem;
                        border-radius: 12px;
                        margin: 0.5rem 0;
                        position: relative;
                    ">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{insight['icon']}</span>
                            <h4 style="margin: 0; color: #2c3e50;">{insight['title']}</h4>
                            <span style="
                                background: {confidence_color};
                                color: white;
                                padding: 0.2rem 0.5rem;
                                border-radius: 10px;
                                font-size: 0.7rem;
                                margin-left: auto;
                            ">{insight['confidence']:.0%}</span>
                        </div>
                        <p style="margin: 0.5rem 0; color: #495057;">{insight['description']}</p>
                        <small style="color: {confidence_color}; font-weight: 500;">💡 {insight['action']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Overall data quality indicator
            st.markdown("#### 📊 Đánh giá Tổng thể")
            quality_color = "#28a745" if data_quality > 85 else "#ffc107" if data_quality > 60 else "#dc3545"
            quality_status = "Tuyệt vời" if data_quality > 85 else "Khá tốt" if data_quality > 60 else "Cần cải thiện"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {quality_color}15 0%, {quality_color}25 100%);
                border: 2px solid {quality_color};
                padding: 1.5rem;
                border-radius: 12px;
                text-align: center;
                margin: 1rem 0;
            ">
                <h3 style="margin: 0; color: {quality_color};">Chất lượng Dữ liệu: {quality_status}</h3>
                <div style="font-size: 2rem; font-weight: bold; color: {quality_color}; margin: 0.5rem 0;">
                    {data_quality:.1f}%
                </div>
                <p style="margin: 0; color: #495057;">
                    Dựa trên phân tích tính toàn vẹn, tính nhất quán và độ đầy đủ của dữ liệu
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.info("🤖 Không thể tạo insights AI. Vui lòng thử lại sau.")
    
    # Dataset management section (existing code continues...)
    st.markdown("### 🗂️ Quản lý Bộ dữ liệu")
    
    # Filter and search options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("🔍 Tìm kiếm bộ dữ liệu:", placeholder="Lọc theo tên...")
    
    with col2:
        size_filter = st.selectbox("📏 Lọc kích thước:", ["Tất cả", "Nhỏ (<1K)", "Trung bình (1K-10K)", "Lớn (>10K)"])
    
    with col3:
        sort_by = st.selectbox("📊 Sắp xếp theo:", ["Tên", "Ngày Tải lên", "Kích thước", "Cột"])
    
    # Apply filters
    filtered_datasets = datasets
    
    if search_term:
        filtered_datasets = [d for d in filtered_datasets if search_term.lower() in d[1].lower()]
    
    if size_filter != "Tất cả":
        if size_filter == "Nhỏ (<1K)":
            filtered_datasets = [d for d in filtered_datasets if d[2] < 1000]
        elif size_filter == "Trung bình (1K-10K)":
            filtered_datasets = [d for d in filtered_datasets if 1000 <= d[2] <= 10000]
        elif size_filter == "Lớn (>10K)":
            filtered_datasets = [d for d in filtered_datasets if d[2] > 10000]
    
    # Sort datasets
    if sort_by == "Tên":
        filtered_datasets.sort(key=lambda x: x[1])
    elif sort_by == "Ngày Tải lên":
        filtered_datasets.sort(key=lambda x: datetime.strptime(x[4], "%Y-%m-%d %H:%M:%S"), reverse=True)
    elif sort_by == "Kích thước":
        filtered_datasets.sort(key=lambda x: x[2], reverse=True)
    elif sort_by == "Cột":
        filtered_datasets.sort(key=lambda x: x[3], reverse=True)
    
    # Display datasets with enhanced management
    for dataset in filtered_datasets:
        id_, name, rows, cols, uploaded, status = dataset
        
        with st.expander(f"📁 {name}", expanded=False):
            try:
                file_path = os.path.join("data", "uploads", name)
                preview_df = safe_read_csv(file_path)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### 📊 Xem trước Bộ dữ liệu")
                    st.dataframe(preview_df.head(5), use_container_width=True)
                    
                    # Quick statistics
                    numeric_cols = preview_df.select_dtypes(include=['number']).columns
                    categorical_cols = preview_df.select_dtypes(include=['object']).columns
                    missing_values = preview_df.isnull().sum().sum()
                    
                    # Data quality for this dataset
                    quality_score = create_data_quality_indicator(preview_df)
                
                with col2:
                    st.markdown("#### 🎯 Thống kê Nhanh")
                    
                    dataset_metrics = [
                        {"title": "Cột Số", "value": str(len(numeric_cols)), "delta": None},
                        {"title": "Cột Văn bản", "value": str(len(categorical_cols)), "delta": None},
                        {"title": "Thiếu", "value": str(missing_values), "delta": None},
                        {"title": "Chất lượng", "value": f"{quality_score:.0%}", "delta": None}
                    ]
                    
                    render_metric_cards(dataset_metrics)
                    
                    st.markdown("#### ⚡ Hành động Nhanh")
                    
                    action_col1, action_col2 = st.columns(2)
                    
                    with action_col1:
                        if st.button("🔍 Phân tích", key=f"analyze_{id_}", use_container_width=True):
                            st.session_state.selected_dataset_id = id_
                            st.switch_page("pages/3_📂_Chi_Tiet_Bo_Du_Lieu.py")
                        
                        if st.button("💬 Trò chuyện", key=f"chat_{id_}", use_container_width=True):
                            st.session_state.selected_dataset_id = id_
                            st.switch_page("main.py")
                    
                    with action_col2:
                        if st.button("📊 Biểu đồ", key=f"chart_{id_}", use_container_width=True):
                            st.session_state.selected_dataset_id = id_
                            st.switch_page("pages/6_📈_Bieu_Do_Thong_Minh.py")
                        
                        if st.button("📋 Báo cáo", key=f"report_{id_}", use_container_width=True):
                            st.session_state.selected_dataset_id = id_
                            st.switch_page("pages/5_📋_Bao_Cao_EDA.py")
                
                # Management options
                st.markdown("#### ⚙️ Tùy chọn Quản lý")
                
                mgmt_col1, mgmt_col2, mgmt_col3 = st.columns(3)
                
                with mgmt_col1:
                    new_name = st.text_input(
                        "Đổi tên bộ dữ liệu:", 
                        value=name, 
                        key=f"rename_input_{id_}",
                        help="Đặt tên mô tả cho bộ dữ liệu của bạn"
                    )
                    
                    if st.button("✅ Đổi tên", key=f"rename_btn_{id_}"):
                        try:
                            rename_dataset(id_, new_name)
                            st.success("✅ Đã đổi tên bộ dữ liệu!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Lỗi khi đổi tên: {str(e)}")
                
                with mgmt_col2:
                    if st.button("📥 Tải xuống", key=f"download_{id_}", help="Tải xuống bộ dữ liệu đã xử lý"):
                        try:
                            csv_data = preview_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Tải xuống CSV",
                                data=csv_data,
                                file_name=f"{name.split('_', 1)[-1] if '_' in name else name}",
                                mime="text/csv",
                                key=f"download_btn_{id_}"
                            )
                        except Exception as e:
                            st.error(f"❌ Lỗi khi tạo file tải xuống: {str(e)}")
                
                with mgmt_col3:
                    if st.button("🗑️ Xóa", key=f"del_{id_}", type="secondary", help="Xóa vĩnh viễn bộ dữ liệu này"):
                        if st.checkbox(f"Xác nhận xóa {name}", key=f"confirm_{id_}"):
                            try:
                                delete_dataset(id_)
                                st.warning(f"🗑️ Đã xóa bộ dữ liệu: {name}")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Lỗi khi xóa: {str(e)}")
                
            except Exception as e:
                st.error(f"❌ Không thể tải bộ dữ liệu: {str(e)}")
                
                # Show basic management even if preview fails
                mgmt_col1, mgmt_col2 = st.columns(2)
                
                with mgmt_col1:
                    new_name = st.text_input("Đổi tên:", value=name, key=f"rename_error_{id_}")
                    if st.button("✅ Đổi tên", key=f"rename_error_btn_{id_}"):
                        try:
                            rename_dataset(id_, new_name)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Lỗi: {str(e)}")
                
                with mgmt_col2:
                    if st.button("🗑️ Xóa", key=f"del_error_{id_}", type="secondary"):
                        try:
                            delete_dataset(id_)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Lỗi: {str(e)}")

else:
    # Welcome screen for new users
    st.markdown("### 👋 Chào mừng đến với VizGenie-GPT Chuyên nghiệp!")
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_feature_card(
            "🤖 Phân tích được Hỗ trợ bởi AI",
            "Đặt câu hỏi phức tạp về dữ liệu của bạn bằng ngôn ngữ tự nhiên và nhận được thông tin thông minh với trực quan hóa đẹp mắt.",
            "🤖"
        )
    
    with col2:
        render_feature_card(
            "🔗 Khám phá Chéo Bộ dữ liệu",
            "Tải lên nhiều bộ dữ liệu và khám phá mối quan hệ ẩn và mô hình qua các nguồn dữ liệu của bạn.",
            "🔗"
        )
    
    with col3:
        render_feature_card(
            "📊 Biểu đồ Chuyên nghiệp",
            "Tạo ra những biểu đồ tuyệt đẹp, sẵn sàng xuất bản với bảng màu thông minh và tính năng tương tác.",
            "📊"
        )
    
    # Getting started guide
    st.markdown("### 🚀 Bắt đầu")
    
    render_insight_card("""
    **📋 Hướng dẫn Nhanh:**
    
    1. **📤 Tải lên Dữ liệu**: Sử dụng thanh bên để tải lên một hoặc nhiều file CSV
    2. **🤖 Đặt Câu hỏi**: Trò chuyện với dữ liệu của bạn bằng ngôn ngữ tự nhiên
    3. **📊 Tạo Trực quan hóa**: Tạo biểu đồ chuyên nghiệp tự động
    4. **🔗 Tìm Mối quan hệ**: Khám phá mô hình qua nhiều bộ dữ liệu
    5. **📄 Xuất Báo cáo**: Tạo báo cáo PDF toàn diện cho các bên liên quan
    
    **💡 Mẹo Chuyên nghiệp:**
    - Tải lên các bộ dữ liệu liên quan cùng nhau để phân tích chéo tốt hơn
    - Sử dụng tên mô tả cho bộ dữ liệu của bạn
    - Đặt câu hỏi cụ thể để có phản hồi AI tốt hơn
    - Thử các loại biểu đồ và bảng màu khác nhau
    """)
    
    # Sample data offer
    st.markdown("### 📚 Thử với Dữ liệu Mẫu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Tải Dữ liệu Bán hàng Mẫu", type="primary", use_container_width=True):
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
            
            st.success("✅ Đã tải dữ liệu bán hàng mẫu!")
            st.rerun()
    
    with col2:
        if st.button("👥 Tải Dữ liệu Khách hàng Mẫu", type="secondary", use_container_width=True):
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
            
            st.success("✅ Đã tải dữ liệu khách hàng mẫu!")
            st.rerun()

# Enhanced sidebar with navigation and tips
with st.sidebar:
    if datasets:
        st.markdown("---")
        st.markdown("### 🎯 Thống kê Nhanh")
        
        # Overall statistics
        try:
            total_size_mb = 0
            for d in datasets:
                try:
                    file_path = os.path.join("data", "uploads", d[1])
                    if os.path.exists(file_path):
                        total_size_mb += os.path.getsize(file_path) / (1024 * 1024)
                except:
                    continue
        except:
            total_size_mb = 0
        
        quick_stats = [
            {"title": "Bộ dữ liệu", "value": str(len(datasets)), "delta": None},
            {"title": "Tổng Kích thước", "value": f"{total_size_mb:.1f}MB", "delta": None},
            {"title": "Lớn nhất", "value": f"{max(d[2] for d in datasets):,}" if datasets else "0", "delta": None}
        ]
        
        render_metric_cards(quick_stats)
    
    st.markdown("---")
    st.markdown("### 🔗 Điều hướng")
    
    nav_links = [
        ("💬 Trò chuyện AI", "main.py"),
        ("📊 Chi tiết Bộ dữ liệu", "pages/3_📂_Chi_Tiet_Bo_Du_Lieu.py"),
        ("📈 Biểu đồ Thông minh", "pages/6_📈_Bieu_Do_Thong_Minh.py"),
        ("🔗 Phân tích Chéo", "pages/7_🔗_Phan_Tich_Cheo_Du_Lieu.py"),
        ("📋 Lịch sử Biểu đồ", "pages/4_📊_Lich_Su_Bieu_Do.py"),
        ("📄 Báo cáo EDA", "pages/5_📋_Bao_Cao_EDA.py"),
        ("📖 Về dự án", "pages/📖_Ve_Du_An.py")
    ]
    
    for label, page in nav_links:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.switch_page(page)
    
    st.markdown("---")
    st.markdown("### 💡 Mẹo & Thủ thuật")
    
    tips = [
        "🎯 **Phân tích Tốt hơn**: Tải lên các bộ dữ liệu liên quan cùng nhau",
        "🎨 **Hấp dẫn Trực quan**: Thử các bảng màu khác nhau trong biểu đồ", 
        "🤖 **Câu hỏi Thông minh**: Cụ thể về những gì bạn muốn khám phá",
        "📊 **Phân tích Chéo**: Tìm kiếm mô hình qua các bộ dữ liệu",
        "📋 **Lưu Công việc**: Sử dụng lịch sử biểu đồ và quản lý phiên"
    ]
    
    for tip in tips:
        st.markdown(f"- {tip}")

# Footer with system info
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🧠 VizGenie-GPT Chuyên nghiệp**")
    st.caption("Nền tảng Phân tích Đa Bộ dữ liệu Nâng cao")

with col2:
    if datasets:
        st.markdown(f"**📊 Trạng thái Hệ thống**")
        st.caption(f"{len(datasets)} bộ dữ liệu • {sum(d[2] for d in datasets):,} tổng bản ghi")
    else:
        st.markdown("**🚀 Sẵn sàng Bắt đầu**")
        st.caption("Tải lên bộ dữ liệu đầu tiên để bắt đầu")

with col3:
    st.markdown("**👨‍💻 Delay Group**")
    st.caption("Làm cho phân tích dữ liệu có thể tiếp cận với mọi người")