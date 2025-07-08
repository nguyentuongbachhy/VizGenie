import streamlit as st
import pandas as pd
import numpy as np
from src.utils import get_all_datasets, get_dataset, safe_read_csv
from src.models.llms import load_llm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🔗 Phân Tích Chéo Dữ Liệu", layout="wide")

# Enhanced styling
st.markdown("""
<style>
    .analysis-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .insight-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .correlation-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="analysis-header"><h1>🔗 Phân Tích Mối Quan Hệ Chéo Bộ Dữ Liệu</h1><p>Khám phá các mẫu và mối quan hệ ẩn qua nhiều bộ dữ liệu</p></div>', unsafe_allow_html=True)

llm = load_llm("gpt-3.5-turbo")

# Tải datasets có sẵn
datasets = get_all_datasets()
if not datasets:
    st.warning("Vui lòng tải lên datasets trước.")
    st.stop()

# Giao diện chọn dataset
st.subheader("📂 Chọn Bộ Dữ Liệu để Phân Tích")
col1, col2 = st.columns(2)

with col1:
    dataset1_options = {f"{d[0]} - {d[1]}": d[0] for d in datasets}
    dataset1_selection = st.selectbox("Bộ Dữ Liệu Chính:", list(dataset1_options.keys()))
    dataset1_id = dataset1_options[dataset1_selection]

with col2:
    dataset2_options = {f"{d[0]} - {d[1]}": d[0] for d in datasets if d[0] != dataset1_id}
    if dataset2_options:
        dataset2_selection = st.selectbox("Bộ Dữ Liệu Phụ:", list(dataset2_options.keys()))
        dataset2_id = dataset2_options[dataset2_selection]
    else:
        st.warning("Cần ít nhất 2 bộ dữ liệu để phân tích chéo")
        st.stop()

# Tải datasets
dataset1 = get_dataset(dataset1_id)
dataset2 = get_dataset(dataset2_id)
df1 = safe_read_csv(dataset1[2])
df2 = safe_read_csv(dataset2[2])

st.success(f"✅ Đã tải: **{dataset1[1]}** ({df1.shape[0]} hàng) và **{dataset2[1]}** ({df2.shape[0]} hàng)")

# Tùy chọn phân tích
st.subheader("🎯 Loại Phân Tích")
analysis_type = st.radio(
    "Chọn phương pháp phân tích:",
    ["Tương Đồng Cột", "Tương Quan Thống Kê", "Mối Quan Hệ Ngữ Nghĩa", "Phân Tích Tổng Hợp"],
    horizontal=True
)

def find_similar_columns(df1, df2, similarity_threshold=0.7):
    """Tìm các cột có tên hoặc kiểu dữ liệu tương tự"""
    similar_pairs = []
    
    for col1 in df1.columns:
        for col2 in df2.columns:
            # Tương đồng tên
            name_sim = len(set(col1.lower().split()) & set(col2.lower().split())) / max(len(set(col1.lower().split())), len(set(col2.lower().split())))
            
            # Tương đồng kiểu dữ liệu
            type_sim = 1.0 if df1[col1].dtype == df2[col2].dtype else 0.5
            
            # Tương đồng tổng hợp
            combined_sim = (name_sim + type_sim) / 2
            
            if combined_sim >= similarity_threshold:
                similar_pairs.append({
                    'col1': col1,
                    'col2': col2,
                    'similarity': combined_sim,
                    'type1': str(df1[col1].dtype),
                    'type2': str(df2[col2].dtype)
                })
    
    return sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)

def calculate_cross_correlations(df1, df2):
    """Tính tương quan giữa các cột số qua các bộ dữ liệu"""
    num_cols1 = df1.select_dtypes(include=[np.number]).columns
    num_cols2 = df2.select_dtypes(include=[np.number]).columns
    
    correlations = []
    
    for col1 in num_cols1:
        for col2 in num_cols2:
            try:
                # Căn chỉnh độ dài cho tương quan
                min_len = min(len(df1[col1]), len(df2[col2]))
                
                # Tính tương quan Pearson
                pearson_r, pearson_p = pearsonr(df1[col1][:min_len].fillna(0), df2[col2][:min_len].fillna(0))
                
                # Tính tương quan Spearman
                spearman_r, spearman_p = spearmanr(df1[col1][:min_len].fillna(0), df2[col2][:min_len].fillna(0))
                
                correlations.append({
                    'col1': col1,
                    'col2': col2,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'significance': 'Cao' if min(pearson_p, spearman_p) < 0.01 else 'Trung Bình' if min(pearson_p, spearman_p) < 0.05 else 'Thấp'
                })
            except:
                continue
    
    return sorted(correlations, key=lambda x: abs(x['pearson_r']), reverse=True)

def generate_ai_insights(df1, df2, dataset1_name, dataset2_name, analysis_results):
    """Tạo insights AI về mối quan hệ"""
    prompt = f"""
    Là một nhà khoa học dữ liệu, hãy phân tích mối quan hệ giữa hai bộ dữ liệu:

    Bộ Dữ Liệu 1: {dataset1_name}
    - Kích thước: {df1.shape}
    - Các cột: {list(df1.columns)[:10]}...
    - Dữ liệu mẫu: {df1.head(2).to_dict()}

    Bộ Dữ Liệu 2: {dataset2_name}
    - Kích thước: {df2.shape}
    - Các cột: {list(df2.columns)[:10]}...
    - Dữ liệu mẫu: {df2.head(2).to_dict()}

    Kết quả Phân tích: {str(analysis_results)[:1000]}...

    Cung cấp insights theo định dạng này:
    
    ## 🔍 Mối Quan Hệ Chính Được Tìm Thấy
    [Liệt kê 3-5 mối quan hệ quan trọng nhất]
    
    ## 📊 Ý Nghĩa Kinh Doanh
    [Giải thích ý nghĩa của các mối quan hệ này trong bối cảnh kinh doanh]
    
    ## 🎯 Hành Động Được Đề Xuất
    [Đề xuất các hành động cụ thể dựa trên phát hiện]
    
    ## ⚠️ Hạn Chế & Cân Nhắc
    [Đề cập đến bất kỳ lưu ý hoặc hạn chế nào]
    
    Hãy cụ thể và có thể hành động. Tập trung vào insights thực tế.
    """
    
    return llm.invoke(prompt)

# Thực hiện phân tích dựa trên lựa chọn
if st.button("🚀 Chạy Phân Tích", type="primary"):
    with st.spinner("Đang phân tích mối quan hệ..."):
        
        if analysis_type == "Tương Đồng Cột":
            similar_cols = find_similar_columns(df1, df2)
            
            st.subheader("📋 Các Cột Tương Tự Được Tìm Thấy")
            if similar_cols:
                for pair in similar_cols[:10]:  # Hiển thị top 10
                    st.markdown(f"""
                    <div class="insight-card">
                        <strong>{pair['col1']}</strong> ↔️ <strong>{pair['col2']}</strong><br>
                        Tương đồng: {pair['similarity']:.2%} | Loại: {pair['type1']} vs {pair['type2']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Không tìm thấy cột tương tự với ngưỡng hiện tại.")
        
        elif analysis_type == "Tương Quan Thống Kê":
            correlations = calculate_cross_correlations(df1, df2)
            
            st.subheader("📈 Tương Quan Chéo Bộ Dữ Liệu")
            if correlations:
                # Tạo trực quan hóa ma trận tương quan
                correlation_data = []
                for corr in correlations[:20]:  # Top 20
                    correlation_data.append({
                        'Mối Quan Hệ': f"{corr['col1']} × {corr['col2']}",
                        'Pearson R': corr['pearson_r'],
                        'Spearman R': corr['spearman_r'],
                        'Ý Nghĩa': corr['significance']
                    })
                
                corr_df = pd.DataFrame(correlation_data)
                
                # Biểu đồ plotly tương tác
                fig = px.bar(corr_df, x='Mối Quan Hệ', y='Pearson R', 
                           color='Ý Nghĩa', 
                           title="Tương Quan Chéo Bộ Dữ Liệu",
                           color_discrete_map={'Cao': '#e74c3c', 'Trung Bình': '#f39c12', 'Thấp': '#95a5a6'})
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Hiển thị bảng
                st.dataframe(corr_df, use_container_width=True)
            else:
                st.info("Không tìm thấy tương quan đáng kể.")
        
        elif analysis_type == "Mối Quan Hệ Ngữ Nghĩa":
            # Phân tích ngữ nghĩa bằng AI
            prompt = f"""
            Phân tích hai bộ dữ liệu này để tìm mối quan hệ ngữ nghĩa:
            
            Các cột Bộ dữ liệu 1: {list(df1.columns)}
            Các cột Bộ dữ liệu 2: {list(df2.columns)}
            
            Tìm các mối quan hệ ngữ nghĩa tiềm năng như:
            - Kết nối địa lý (thành phố, bang, quốc gia)
            - Kết nối thời gian (ngày, thời gian, giai đoạn)
            - Kết nối phân loại (loại, danh mục, lớp)
            - Kết nối phân cấp (mối quan hệ cha-con)
            
            Trả về danh sách JSON của các mối quan hệ tiềm năng với điểm tin cậy.
            """
            
            ai_relationships = llm.invoke(prompt)
            
            st.subheader("🧠 Mối Quan Hệ Ngữ Nghĩa Được AI Phát Hiện")
            st.markdown(ai_relationships)
        
        elif analysis_type == "Phân Tích Tổng Hợp":
            # Chạy tất cả phân tích
            similar_cols = find_similar_columns(df1, df2)
            correlations = calculate_cross_correlations(df1, df2)
            
            # Tạo insights AI tổng hợp
            all_results = {
                'similar_columns': similar_cols[:5],
                'correlations': correlations[:5]
            }
            
            ai_insights = generate_ai_insights(df1, df2, dataset1[1], dataset2[1], all_results)
            
            # Hiển thị kết quả trong tabs
            tab1, tab2, tab3 = st.tabs(["🔍 Insights AI", "📋 Cột Tương Tự", "📈 Tương Quan"])
            
            with tab1:
                st.markdown(ai_insights)
            
            with tab2:
                if similar_cols:
                    for pair in similar_cols[:10]:
                        st.markdown(f"""
                        <div class="insight-card">
                            <strong>{pair['col1']}</strong> ↔️ <strong>{pair['col2']}</strong><br>
                            Tương đồng: {pair['similarity']:.2%}
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab3:
                if correlations:
                    correlation_data = []
                    for corr in correlations[:15]:
                        correlation_data.append({
                            'Cột 1': corr['col1'],
                            'Cột 2': corr['col2'],
                            'Tương Quan': f"{corr['pearson_r']:.3f}",
                            'P-value': f"{corr['pearson_p']:.3f}",
                            'Ý Nghĩa': corr['significance']
                        })
                    
                    st.dataframe(pd.DataFrame(correlation_data), use_container_width=True)

# Giao diện Truy vấn Nâng cao
st.subheader("💬 Đặt Câu Hỏi Qua Các Bộ Dữ Liệu")
query_placeholder = st.text_area(
    "Đặt câu hỏi phức tạp trải rộng cả hai bộ dữ liệu:",
    placeholder="Ví dụ:\n- Có bao nhiêu giáo viên nữ ở trường tiểu học Hà Nội?\n- Tương quan tỷ lệ bỏ học giữa các vùng như thế nào?\n- So sánh hiệu suất học sinh qua các loại trường khác nhau",
    height=100
)

if st.button("🎯 Trả Lời Câu Hỏi") and query_placeholder:
    with st.spinner("Đang xử lý truy vấn phức tạp..."):
        enhanced_prompt = f"""
        Bạn có quyền truy cập vào hai bộ dữ liệu:
        
        Bộ Dữ Liệu 1: {dataset1[1]}
        Các cột: {list(df1.columns)}
        Mẫu: {df1.head(2).to_dict()}
        
        Bộ Dữ Liệu 2: {dataset2[1]}
        Các cột: {list(df2.columns)}
        Mẫu: {df2.head(2).to_dict()}
        
        Câu Hỏi Người Dùng: {query_placeholder}
        
        Cung cấp câu trả lời toàn diện bao gồm:
        1. Xác định bộ dữ liệu và cột nào liên quan
        2. Giải thích bất kỳ giả định nào được đưa ra
        3. Cung cấp số liệu/insights cụ thể khi có thể
        4. Đề xuất phân tích tiếp theo
        5. Lưu ý bất kỳ hạn chế nào
        
        Hãy cụ thể và dựa trên dữ liệu trong phản hồi của bạn.
        """
        
        response = llm.invoke(enhanced_prompt)
        
        st.markdown("### 🎯 Kết Quả Phân Tích")
        st.markdown(f"""
        <div class="insight-card">
            {response}
        </div>
        """, unsafe_allow_html=True)

# Xuất kết quả
if st.button("📥 Xuất Báo Cáo Phân Tích"):
    st.success("Báo cáo phân tích sẽ được tạo và tải xuống ở đây")