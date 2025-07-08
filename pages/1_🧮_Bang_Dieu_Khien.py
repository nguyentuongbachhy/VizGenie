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

st.set_page_config(page_title="📂 Bảng điều khiển Chuyên nghiệp", layout="wide", page_icon="📊")

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

# Enhanced sidebar for dataset upload
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e1e5e9; margin-bottom: 1rem;">
        <h3 style="color: #667eea; margin: 0;">📂 Quản lý Bộ dữ liệu</h3>
        <small style="color: #666;">Tải lên & Tổ chức</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced multi-file upload
    st.markdown("#### 📤 Tải lên Bộ dữ liệu")
    uploaded_files = st.file_uploader(
        "Chọn các file CSV (hỗ trợ nhiều file)", 
        type=["csv"], 
        accept_multiple_files=True,
        help="💡 Tải lên nhiều bộ dữ liệu để khám phá mối quan hệ chéo dữ liệu"
    )
    
    # Upload progress and processing
    if uploaded_files:
        upload_progress = st.progress(0)
        upload_status = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            if f"uploaded_{uploaded_file.name}" not in st.session_state:
                upload_status.text(f"Đang xử lý {uploaded_file.name}...")
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
        
        upload_status.text("✅ Tất cả file đã được tải lên thành công!")
        upload_progress.progress(1.0)
        st.rerun()

# Load datasets
datasets = get_all_datasets()

if datasets:
    # Enhanced dashboard metrics with animations
    st.markdown("### 📊 Tổng quan Bảng điều khiển")
    
    total_datasets = len(datasets)
    total_rows = sum([d[2] for d in datasets])
    total_cols = sum([d[3] for d in datasets])
    avg_size = total_rows / total_datasets if total_datasets > 0 else 0
    
    # Calculate additional metrics
    largest_dataset = max(datasets, key=lambda x: x[2])
    newest_dataset = max(datasets, key=lambda x: datetime.strptime(x[4], "%Y-%m-%d %H:%M:%S"))
    
    # Professional metric cards
    metrics = [
        {"title": "Tổng Bộ dữ liệu", "value": f"{total_datasets}", "delta": "+3 tuần này"},
        {"title": "Tổng Bản ghi", "value": f"{total_rows:,}", "delta": f"+{total_rows//10:,} gần đây"},
        {"title": "Trường Dữ liệu", "value": f"{total_cols}", "delta": None},
        {"title": "Kích thước TB Bộ dữ liệu", "value": f"{avg_size:,.0f}", "delta": None}
    ]
    
    render_metric_cards(metrics)
    
    # Enhanced analytics dashboard
    st.markdown("### 📈 Bảng điều khiển Phân tích")
    
    # Create comprehensive dashboard visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Kích thước Bộ dữ liệu (Bản ghi)', 'Dòng thời gian Tải lên', 'Phân phối Cột', 'Điểm Mật độ Dữ liệu'),
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
            name="Bản ghi",
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
            name="Tải lên Tích lũy",
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
            name="Cột",
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
            name="Điểm Mật độ",
            marker=dict(
                size=[min(50, max(10, size//1000)) for size in dataset_sizes],
                color=density_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Điểm Mật độ")
            ),
            text=[f"{d[1]}<br>Mật độ: {score:.1f}" for d, score in zip(datasets, density_scores)],
            hovertemplate="<b>%{text}</b><br>Bản ghi/Cột: %{y:.1f}<extra></extra>"
        ),
        row=2, col=2
    )
    
    # Update layout with professional styling
    fig.update_layout(
        height=600,
        showlegend=False,
        title=dict(
            text="📊 Tổng quan Phân tích Bộ dữ liệu",
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
    st.markdown("### 🤖 Thông tin được Hỗ trợ bởi AI")
    
    # Generate insights about the dataset collection
    insights = []
    
    if total_datasets >= 3:
        insights.append("🎯 **Sẵn sàng Phân tích Đa Bộ dữ liệu**: Bạn có đủ bộ dữ liệu cho phân tích chéo toàn diện")
    
    if max(dataset_sizes) > 10000:
        insights.append(f"📈 **Phát hiện Bộ dữ liệu Lớn**: {largest_dataset[1]} có {largest_dataset[2]:,} bản ghi - phù hợp cho phân tích sâu")
    
    if len(set(d[3] for d in datasets)) > 3:
        insights.append("🔗 **Cấu trúc Dữ liệu Đa dạng**: Số lượng cột khác nhau cho thấy các loại dữ liệu khác nhau - tốt cho phân tích toàn diện")
    
    upload_recency = (datetime.now() - datetime.strptime(newest_dataset[4], "%Y-%m-%d %H:%M:%S")).days
    if upload_recency < 7:
        insights.append(f"⚡ **Dữ liệu Mới**: Tải lên mới nhất ({newest_dataset[1]}) chỉ cách đây {upload_recency} ngày")
    
    # Display insights in cards
    if insights:
        for insight in insights:
            render_insight_card(insight)
    else:
        render_insight_card("📊 **Bắt đầu**: Tải lên thêm bộ dữ liệu để mở khóa thông tin AI nâng cao và phân tích chéo dữ liệu!")
    
    # Multi-dataset relationship analysis
    st.markdown("### 🔗 Phân tích Chéo Bộ dữ liệu")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_datasets = st.multiselect(
            "🎯 Chọn bộ dữ liệu để phân tích mối quan hệ:",
            options=[f"{d[0]} - {d[1]}" for d in datasets],
            help="Chọn 2+ bộ dữ liệu để khám phá mối quan hệ và mô hình ẩn",
            placeholder="Chọn nhiều bộ dữ liệu..."
        )
    
    with col2:
        analysis_type = st.selectbox(
            "Loại Phân tích:",
            ["Tương đồng Cột", "Tương quan Thống kê", "Mối quan hệ Ngữ nghĩa", "Phân tích Sâu AI"]
        )
    
    if len(selected_datasets) >= 2:
        if st.button("🚀 Phân tích Mối quan hệ", type="primary", use_container_width=True):
            with st.spinner("🤖 Đang phân tích mối quan hệ chéo bộ dữ liệu..."):
                render_animated_loading("Đang khám phá mô hình qua các bộ dữ liệu của bạn...")
                
                # Store selection for cross-analysis page
                st.session_state.cross_analysis_datasets = selected_datasets
                st.session_state.cross_analysis_type = analysis_type
                
                st.success("✅ Phân tích sẵn sàng! Nhấp bên dưới để xem kết quả chi tiết.")
                
                if st.button("📊 Xem Phân tích Chi tiết", type="secondary"):
                    st.switch_page("pages/7_🔗_Phan_Tich_Cheo_Du_Lieu.py")
    
    # Enhanced dataset management with professional cards
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
                    st.markdown("#### 📊 Xem trước Bộ dữ liệu")
                    st.dataframe(preview_df.head(5), use_container_width=True)
                    
                    # Quick statistics
                    numeric_cols = preview_df.select_dtypes(include=['number']).columns
                    categorical_cols = preview_df.select_dtypes(include=['object']).columns
                    missing_values = preview_df.isnull().sum().sum()
                    
                    # Data quality indicator
                    quality_score = create_data_quality_indicator(preview_df)
                    
                with col2:
                    st.markdown("#### 🎯 Thống kê Nhanh")
                    
                    # Mini metrics for this dataset
                    dataset_metrics = [
                        {"title": "Cột Số", "value": str(len(numeric_cols))},
                        {"title": "Cột Văn bản", "value": str(len(categorical_cols))},
                        {"title": "Thiếu", "value": str(missing_values)},
                        {"title": "Chất lượng", "value": f"{quality_score:.0%}"}
                    ]
                    
                    render_metric_cards(dataset_metrics)
                    
                    # Action buttons
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
                
                # AI recommendations for this specific dataset
                st.markdown("#### 🤖 Khuyến nghị AI")
                create_ai_recommendation_panel(preview_df)
                
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
                        rename_dataset(id_, new_name)
                        st.success("✅ Đã đổi tên bộ dữ liệu!")
                        st.rerun()
                
                with mgmt_col2:
                    if st.button("📥 Tải xuống", key=f"download_{id_}", help="Tải xuống bộ dữ liệu đã xử lý"):
                        # Create download functionality
                        csv_data = preview_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Tải xuống CSV",
                            data=csv_data,
                            file_name=f"{name.split('_', 1)[-1] if '_' in name else name}",
                            mime="text/csv",
                            key=f"download_btn_{id_}"
                        )
                
                with mgmt_col3:
                    if st.button("🗑️ Xóa", key=f"del_{id_}", type="secondary", help="Xóa vĩnh viễn bộ dữ liệu này"):
                        # Confirmation dialog
                        if st.checkbox(f"Xác nhận xóa {name}", key=f"confirm_{id_}"):
                            delete_dataset(id_)
                            st.warning(f"🗑️ Đã xóa bộ dữ liệu: {name}")
                            st.rerun()
                
                # Interactive data explorer for this dataset
                if st.checkbox("🔍 Mở Khám phá Dữ liệu", key=f"explorer_{id_}"):
                    render_interactive_data_explorer(preview_df)
                
            except Exception as e:
                st.error(f"❌ Không thể tải bộ dữ liệu: {e}")
                
                # Still show management options even if preview fails
                mgmt_col1, mgmt_col2 = st.columns(2)
                
                with mgmt_col1:
                    new_name = st.text_input("Đổi tên:", value=name, key=f"rename_error_{id_}")
                    if st.button("✅ Đổi tên", key=f"rename_error_btn_{id_}"):
                        rename_dataset(id_, new_name)
                        st.rerun()
                
                with mgmt_col2:
                    if st.button("🗑️ Xóa", key=f"del_error_{id_}", type="secondary"):
                        delete_dataset(id_)
                        st.rerun()

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
        total_size_mb = sum(os.path.getsize(os.path.join("data", "uploads", d[1])) for d in datasets if os.path.exists(os.path.join("data", "uploads", d[1]))) / (1024 * 1024)
        
        quick_stats = [
            {"title": "Bộ dữ liệu", "value": str(len(datasets)), "delta": None},
            {"title": "Tổng Kích thước", "value": f"{total_size_mb:.1f}MB", "delta": None},
            {"title": "Lớn nhất", "value": f"{max(d[2] for d in datasets):,}", "delta": None}
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