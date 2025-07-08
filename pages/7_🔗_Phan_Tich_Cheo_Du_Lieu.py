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
import time
import json
import re
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
    .loading-spinner {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .spinner {
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
    .analysis-result {
        background: linear-gradient(135deg, #56CCF215 0%, #2F80ED15 100%);
        border: 1px solid #56CCF230;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .error-card {
        background: linear-gradient(135deg, #ff6b6b15 0%, #ee5a2415 100%);
        border: 1px solid #ff6b6b30;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="analysis-header"><h1>🔗 Phân Tích Mối Quan Hệ Chéo Bộ Dữ Liệu</h1><p>Khám phá các mẫu và mối quan hệ ẩn qua nhiều bộ dữ liệu với AI nâng cao</p></div>', unsafe_allow_html=True)

llm = load_llm("gpt-3.5-turbo")

def show_loading(text="Đang xử lý..."):
    """Show loading animation"""
    return st.markdown(f"""
    <div class="loading-spinner">
        <div class="spinner"></div>
        <span style="color: #667eea; font-weight: 500;">{text}</span>
    </div>
    """, unsafe_allow_html=True)

def safe_llm_invoke(prompt, max_retries=3):
    """Safely invoke LLM with retries and error handling"""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            if isinstance(response, str):
                return response
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Lỗi LLM: {str(e)}"
            time.sleep(1)
    return "Không thể kết nối đến AI"

def extract_json_from_response(response):
    """Extract JSON from LLM response"""
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        
        # Try to parse the entire response as JSON
        return json.loads(response)
    except:
        # Return structured fallback
        return {
            "relationships": [
                {
                    "type": "text_analysis",
                    "description": response,
                    "confidence": 0.5
                }
            ]
        }

def find_similar_columns(df1, df2, similarity_threshold=0.6):
    """Find columns with similar names or data types"""
    try:
        similar_pairs = []
        
        for col1 in df1.columns:
            for col2 in df2.columns:
                try:
                    # Name similarity (case insensitive)
                    name1_words = set(col1.lower().replace('_', ' ').split())
                    name2_words = set(col2.lower().replace('_', ' ').split())
                    
                    if name1_words and name2_words:
                        name_sim = len(name1_words & name2_words) / max(len(name1_words), len(name2_words))
                    else:
                        name_sim = 0
                    
                    # Type similarity
                    type1 = str(df1[col1].dtype)
                    type2 = str(df2[col2].dtype)
                    type_sim = 1.0 if type1 == type2 else 0.3
                    
                    # Data pattern similarity for object columns
                    pattern_sim = 0
                    if df1[col1].dtype == 'object' and df2[col2].dtype == 'object':
                        sample1 = df1[col1].dropna().head(10).astype(str).tolist()
                        sample2 = df2[col2].dropna().head(10).astype(str).tolist()
                        
                        if sample1 and sample2:
                            # Check average length similarity
                            avg_len1 = np.mean([len(s) for s in sample1])
                            avg_len2 = np.mean([len(s) for s in sample2])
                            len_diff = abs(avg_len1 - avg_len2) / max(avg_len1, avg_len2, 1)
                            pattern_sim = max(0, 1 - len_diff)
                    
                    # Combined similarity
                    combined_sim = (name_sim * 0.5 + type_sim * 0.3 + pattern_sim * 0.2)
                    
                    if combined_sim >= similarity_threshold:
                        similar_pairs.append({
                            'col1': col1,
                            'col2': col2,
                            'similarity': combined_sim,
                            'name_sim': name_sim,
                            'type1': type1,
                            'type2': type2,
                            'pattern_sim': pattern_sim
                        })
                except Exception as e:
                    continue
        
        return sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)
    
    except Exception as e:
        st.error(f"Lỗi tìm cột tương tự: {str(e)}")
        return []

def calculate_cross_correlations(df1, df2, max_pairs=50):
    """Calculate correlations between numeric columns across datasets"""
    try:
        num_cols1 = df1.select_dtypes(include=[np.number]).columns[:10]  # Limit for performance
        num_cols2 = df2.select_dtypes(include=[np.number]).columns[:10]
        
        if len(num_cols1) == 0 or len(num_cols2) == 0:
            return []
        
        correlations = []
        pair_count = 0
        
        for col1 in num_cols1:
            for col2 in num_cols2:
                if pair_count >= max_pairs:
                    break
                    
                try:
                    # Align data lengths
                    min_len = min(len(df1[col1]), len(df2[col2]))
                    data1 = df1[col1][:min_len].fillna(df1[col1].median())
                    data2 = df2[col2][:min_len].fillna(df2[col2].median())
                    
                    # Skip if not enough data
                    if len(data1) < 10 or data1.std() == 0 or data2.std() == 0:
                        continue
                    
                    # Calculate correlations
                    pearson_r, pearson_p = pearsonr(data1, data2)
                    spearman_r, spearman_p = spearmanr(data1, data2)
                    
                    # Skip very weak correlations
                    if abs(pearson_r) < 0.1 and abs(spearman_r) < 0.1:
                        continue
                    
                    correlations.append({
                        'col1': col1,
                        'col2': col2,
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'significance': 'Cao' if min(pearson_p, spearman_p) < 0.01 else 'Trung Bình' if min(pearson_p, spearman_p) < 0.05 else 'Thấp',
                        'strength': 'Mạnh' if max(abs(pearson_r), abs(spearman_r)) > 0.7 else 'Trung Bình' if max(abs(pearson_r), abs(spearman_r)) > 0.3 else 'Yếu'
                    })
                    
                    pair_count += 1
                    
                except Exception as e:
                    continue
        
        return sorted(correlations, key=lambda x: max(abs(x['pearson_r']), abs(x['spearman_r'])), reverse=True)
    
    except Exception as e:
        st.error(f"Lỗi tính tương quan: {str(e)}")
        return []

def generate_ai_insights(df1, df2, dataset1_name, dataset2_name, analysis_results):
    """Generate comprehensive AI insights about relationships"""
    try:
        # Prepare data summary
        data_summary = f"""
        Bộ Dữ Liệu 1: {dataset1_name}
        - Kích thước: {df1.shape}
        - Cột số: {len(df1.select_dtypes(include=[np.number]).columns)}
        - Cột văn bản: {len(df1.select_dtypes(include=['object']).columns)}
        - Dữ liệu mẫu: {df1.head(2).to_dict() if not df1.empty else 'Trống'}

        Bộ Dữ Liệu 2: {dataset2_name}
        - Kích thước: {df2.shape}
        - Cột số: {len(df2.select_dtypes(include=[np.number]).columns)}
        - Cột văn bản: {len(df2.select_dtypes(include=['object']).columns)}
        - Dữ liệu mẫu: {df2.head(2).to_dict() if not df2.empty else 'Trống'}
        """
        
        results_summary = str(analysis_results)[:1000] + "..." if len(str(analysis_results)) > 1000 else str(analysis_results)
        
        prompt = f"""
        Phân tích mối quan hệ giữa hai bộ dữ liệu:

        {data_summary}

        Kết quả phân tích: {results_summary}

        Hãy đưa ra insights theo định dạng markdown:
        
        ## 🔍 Mối Quan Hệ Chính Được Tìm Thấy
        [Liệt kê 3-5 mối quan hệ quan trọng nhất với số liệu cụ thể]
        
        ## 📊 Ý Nghĩa Kinh Doanh
        [Giải thích ý nghĩa thực tế của các mối quan hệ này]
        
        ## 🎯 Hành Động Được Đề Xuất
        [3-4 hành động cụ thể có thể thực hiện dựa trên phát hiện]
        
        ## ⚠️ Hạn Chế & Cân Nhắc
        [Những lưu ý quan trọng về độ tin cậy và giới hạn của phân tích]
        
        Hãy cụ thể, dựa trên dữ liệu thực và có thể hành động.
        """
        
        return safe_llm_invoke(prompt)
    
    except Exception as e:
        return f"Lỗi tạo insights AI: {str(e)}"

def perform_semantic_analysis(df1, df2):
    """Perform semantic relationship analysis using AI"""
    try:
        cols1_info = {col: df1[col].dtype for col in df1.columns[:10]}
        cols2_info = {col: df2[col].dtype for col in df2.columns[:10]}
        
        prompt = f"""
        Phân tích các mối quan hệ ngữ nghĩa tiềm năng giữa hai bộ dữ liệu:
        
        Cột Bộ dữ liệu 1: {cols1_info}
        Cột Bộ dữ liệu 2: {cols2_info}
        
        Tìm các mối quan hệ ngữ nghĩa như:
        - Kết nối địa lý (thành phố ↔ khu vực)
        - Kết nối thời gian (ngày ↔ tháng ↔ năm)
        - Kết nối phân loại (loại ↔ danh mục)
        - Kết nối định danh (ID khách hàng ↔ mã khách hàng)
        - Kết nối phân cấp (chi nhánh ↔ công ty)
        
        Trả về JSON format:
        {{
            "relationships": [
                {{
                    "col1": "tên_cột_1",
                    "col2": "tên_cột_2", 
                    "type": "loại_quan_hệ",
                    "description": "mô_tả_chi_tiết",
                    "confidence": 0.8
                }}
            ]
        }}
        """
        
        response = safe_llm_invoke(prompt)
        return extract_json_from_response(response)
    
    except Exception as e:
        return {
            "relationships": [
                {
                    "col1": "error",
                    "col2": "error",
                    "type": "error",
                    "description": f"Lỗi phân tích ngữ nghĩa: {str(e)}",
                    "confidence": 0.0
                }
            ]
        }

def create_correlation_visualization(correlations):
    """Create enhanced correlation visualization"""
    try:
        if not correlations:
            fig = go.Figure()
            fig.add_annotation(text="Không có dữ liệu tương quan", 
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Prepare data for visualization
        df_corr = pd.DataFrame(correlations[:20])  # Top 20 correlations
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Tương quan Pearson',
                'Tương quan Spearman', 
                'Mức độ Ý nghĩa',
                'Tổng hợp Tương quan'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Pearson correlation
        fig.add_trace(
            go.Bar(
                x=[f"{r['col1']} × {r['col2']}" for r in correlations[:10]],
                y=[r['pearson_r'] for r in correlations[:10]],
                name="Pearson",
                marker=dict(color=[r['pearson_r'] for r in correlations[:10]], 
                          colorscale='RdBu', cmin=-1, cmax=1),
                text=[f"{r['pearson_r']:.3f}" for r in correlations[:10]],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Spearman correlation
        fig.add_trace(
            go.Bar(
                x=[f"{r['col1']} × {r['col2']}" for r in correlations[:10]],
                y=[r['spearman_r'] for r in correlations[:10]],
                name="Spearman",
                marker=dict(color=[r['spearman_r'] for r in correlations[:10]], 
                          colorscale='RdBu', cmin=-1, cmax=1),
                text=[f"{r['spearman_r']:.3f}" for r in correlations[:10]],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # Significance levels
        sig_counts = {}
        for corr in correlations:
            sig = corr['significance']
            sig_counts[sig] = sig_counts.get(sig, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(sig_counts.keys()),
                values=list(sig_counts.values()),
                name="Ý nghĩa"
            ),
            row=2, col=1
        )
        
        # Strength distribution
        strength_counts = {}
        for corr in correlations:
            strength = corr['strength']
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(strength_counts.keys()),
                values=list(strength_counts.values()),
                name="Độ mạnh"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Phân tích Tương quan Chéo Bộ dữ liệu",
            showlegend=False
        )
        
        # Update x-axes for better readability
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        
        return fig
    
    except Exception as e:
        # Fallback simple chart
        fig = go.Figure()
        fig.add_annotation(text=f"Lỗi tạo biểu đồ: {str(e)}", 
                         x=0.5, y=0.5, showarrow=False)
        return fig

# Load datasets
datasets = get_all_datasets()
if not datasets:
    st.warning("Vui lòng tải lên datasets trước.")
    st.stop()

# Dataset selection interface
st.subheader("📂 Chọn Bộ Dữ Liệu để Phân Tích")
col1, col2 = st.columns(2)

with col1:
    dataset1_options = {f"{d[0]} - {d[1]}": d[0] for d in datasets}
    dataset1_selection = st.selectbox("🗂️ Bộ Dữ Liệu Chính:", list(dataset1_options.keys()))
    dataset1_id = dataset1_options[dataset1_selection]

with col2:
    dataset2_options = {f"{d[0]} - {d[1]}": d[0] for d in datasets if d[0] != dataset1_id}
    if dataset2_options:
        dataset2_selection = st.selectbox("📋 Bộ Dữ Liệu Phụ:", list(dataset2_options.keys()))
        dataset2_id = dataset2_options[dataset2_selection]
    else:
        st.warning("⚠️ Cần ít nhất 2 bộ dữ liệu để phân tích chéo")
        st.stop()

# Load datasets with error handling
try:
    dataset1 = get_dataset(dataset1_id)
    dataset2 = get_dataset(dataset2_id)
    df1 = safe_read_csv(dataset1[2])
    df2 = safe_read_csv(dataset2[2])
    
    if df1.empty or df2.empty:
        st.error("❌ Một trong các bộ dữ liệu trống. Vui lòng kiểm tra lại.")
        st.stop()
    
    st.success(f"✅ Đã tải: **{dataset1[1]}** ({df1.shape[0]:,} hàng, {df1.shape[1]} cột) và **{dataset2[1]}** ({df2.shape[0]:,} hàng, {df2.shape[1]} cột)")
    
except Exception as e:
    st.error(f"❌ Lỗi tải dữ liệu: {str(e)}")
    st.stop()

# Analysis type selection
st.subheader("🎯 Loại Phân Tích")
analysis_type = st.radio(
    "Chọn phương pháp phân tích:",
    ["Tương Đồng Cột", "Tương Quan Thống Kê", "Mối Quan Hệ Ngữ Nghĩa", "Phân Tích Tổng Hợp"],
    horizontal=True,
    help="Chọn loại phân tích phù hợp với mục đích nghiên cứu của bạn"
)

# Main analysis execution
if st.button("🚀 Chạy Phân Tích", type="primary"):
    # Show loading
    loading_placeholder = st.empty()
    
    try:
        if analysis_type == "Tương Đồng Cột":
            with loading_placeholder:
                show_loading("🔍 Đang tìm các cột tương tự...")
            
            time.sleep(1)
            similar_cols = find_similar_columns(df1, df2)
            loading_placeholder.empty()
            
            st.subheader("📋 Các Cột Tương Tự Được Tìm Thấy")
            
            if similar_cols:
                # Display results in a nice format
                for i, pair in enumerate(similar_cols[:15]):  # Show top 15
                    confidence_color = "#28a745" if pair['similarity'] > 0.8 else "#ffc107" if pair['similarity'] > 0.6 else "#dc3545"
                    
                    st.markdown(f"""
                    <div class="analysis-result">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin: 0; color: #2c3e50;">
                                    📊 {pair['col1']} ↔️ {pair['col2']}
                                </h4>
                                <p style="margin: 0.5rem 0; color: #495057;">
                                    <strong>Loại:</strong> {pair['type1']} vs {pair['type2']}<br>
                                    <strong>Tương đồng tên:</strong> {pair['name_sim']:.1%}<br>
                                    <strong>Tương đồng mẫu:</strong> {pair['pattern_sim']:.1%}
                                </p>
                            </div>
                            <div style="text-align: center;">
                                <div style="
                                    background: {confidence_color};
                                    color: white;
                                    padding: 0.5rem 1rem;
                                    border-radius: 20px;
                                    font-weight: bold;
                                ">
                                    {pair['similarity']:.1%}
                                </div>
                                <small style="color: #666;">Độ tin cậy</small>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Summary statistics
                st.markdown("### 📊 Tóm tắt Phân tích")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Tổng cặp tương tự", len(similar_cols))
                
                with col2:
                    high_conf = len([p for p in similar_cols if p['similarity'] > 0.8])
                    st.metric("Độ tin cậy cao", high_conf)
                
                with col3:
                    avg_sim = np.mean([p['similarity'] for p in similar_cols])
                    st.metric("Tương đồng TB", f"{avg_sim:.1%}")
                
            else:
                st.info("🔍 Không tìm thấy cột tương tự với ngưỡng hiện tại. Thử giảm ngưỡng tương đồng.")
        
        elif analysis_type == "Tương Quan Thống Kê":
            with loading_placeholder:
                show_loading("📊 Đang tính toán tương quan chéo...")
            
            time.sleep(2)
            correlations = calculate_cross_correlations(df1, df2)
            loading_placeholder.empty()
            
            st.subheader("📈 Tương Quan Chéo Bộ Dữ Liệu")
            
            if correlations:
                # Create interactive visualization
                fig = create_correlation_visualization(correlations)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display top correlations table
                st.markdown("### 🏆 Top Tương Quan")
                
                corr_data = []
                for corr in correlations[:20]:
                    corr_data.append({
                        'Mối Quan Hệ': f"{corr['col1']} × {corr['col2']}",
                        'Pearson': f"{corr['pearson_r']:.3f}",
                        'Spearman': f"{corr['spearman_r']:.3f}",
                        'P-value': f"{min(corr['pearson_p'], corr['spearman_p']):.3f}",
                        'Ý Nghĩa': corr['significance'],
                        'Độ Mạnh': corr['strength']
                    })
                
                corr_df = pd.DataFrame(corr_data)
                st.dataframe(corr_df, use_container_width=True)
                
                # Export option
                csv_data = corr_df.to_csv(index=False)
                st.download_button(
                    "📥 Tải xuống Kết quả Tương quan",
                    csv_data,
                    file_name="cross_correlation_analysis.csv",
                    mime="text/csv"
                )
                
            else:
                st.info("📊 Không tìm thấy tương quan đáng kể giữa các cột số.")
        
        elif analysis_type == "Mối Quan Hệ Ngữ Nghĩa":
            with loading_placeholder:
                show_loading("🤖 AI đang phân tích mối quan hệ ngữ nghĩa...")
            
            time.sleep(3)
            semantic_results = perform_semantic_analysis(df1, df2)
            loading_placeholder.empty()
            
            st.subheader("🧠 Mối Quan Hệ Ngữ Nghĩa Được AI Phát Hiện")
            
            if semantic_results and 'relationships' in semantic_results:
                for rel in semantic_results['relationships']:
                    if rel['type'] != 'error':
                        confidence_color = "#28a745" if rel['confidence'] > 0.7 else "#ffc107" if rel['confidence'] > 0.4 else "#dc3545"
                        
                        st.markdown(f"""
                        <div class="analysis-result">
                            <h4 style="color: #2c3e50;">
                                🔗 {rel['col1']} ↔️ {rel['col2']}
                            </h4>
                            <p><strong>Loại quan hệ:</strong> {rel['type']}</p>
                            <p><strong>Mô tả:</strong> {rel['description']}</p>
                            <div style="
                                background: {confidence_color};
                                color: white;
                                padding: 0.3rem 0.8rem;
                                border-radius: 15px;
                                display: inline-block;
                                font-size: 0.9rem;
                            ">
                                Tin cậy: {rel['confidence']:.1%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-card">
                            <p>{rel['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("🔍 Không phát hiện mối quan hệ ngữ nghĩa rõ ràng.")
        
        elif analysis_type == "Phân Tích Tổng Hợp":
            with loading_placeholder:
                show_loading("🔄 Đang thực hiện phân tích tổng hợp...")
            
            # Run all analyses
            similar_cols = find_similar_columns(df1, df2)
            time.sleep(1)
            
            correlations = calculate_cross_correlations(df1, df2)
            time.sleep(1)
            
            semantic_results = perform_semantic_analysis(df1, df2)
            time.sleep(2)
            
            # Combine results for AI analysis
            all_results = {
                'similar_columns': similar_cols[:5],
                'correlations': correlations[:5],
                'semantic_relationships': semantic_results
            }
            
            ai_insights = generate_ai_insights(df1, df2, dataset1[1], dataset2[1], all_results)
            loading_placeholder.empty()
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["🔍 Insights AI", "📋 Cột Tương Tự", "📈 Tương Quan", "🔗 Ngữ Nghĩa"])
            
            with tab1:
                st.markdown("### 🤖 Phân Tích Tổng Hợp AI")
                st.markdown(ai_insights)
                
                # Summary metrics
                st.markdown("### 📊 Tóm tắt Kết quả")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Cột tương tự", len(similar_cols))
                
                with col2:
                    st.metric("Tương quan mạnh", len([c for c in correlations if max(abs(c['pearson_r']), abs(c['spearman_r'])) > 0.5]))
                
                with col3:
                    semantic_count = len(semantic_results.get('relationships', [])) if semantic_results else 0
                    st.metric("Quan hệ ngữ nghĩa", semantic_count)
                
                with col4:
                    total_connections = len(similar_cols) + len(correlations) + semantic_count
                    st.metric("Tổng kết nối", total_connections)
            
            with tab2:
                if similar_cols:
                    for pair in similar_cols[:10]:
                        st.markdown(f"""
                        <div class="insight-card">
                            <strong>{pair['col1']}</strong> ↔️ <strong>{pair['col2']}</strong><br>
                            Tương đồng: {pair['similarity']:.2%} | Loại: {pair['type1']} vs {pair['type2']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Không tìm thấy cột tương tự")
            
            with tab3:
                if correlations:
                    corr_data = []
                    for corr in correlations[:15]:
                        corr_data.append({
                            'Cột 1': corr['col1'],
                            'Cột 2': corr['col2'],
                            'Pearson': f"{corr['pearson_r']:.3f}",
                            'Spearman': f"{corr['spearman_r']:.3f}",
                            'Ý Nghĩa': corr['significance']
                        })
                    
                    st.dataframe(pd.DataFrame(corr_data), use_container_width=True)
                else:
                    st.info("Không tìm thấy tương quan đáng kể")
            
            with tab4:
                if semantic_results and 'relationships' in semantic_results:
                    for rel in semantic_results['relationships']:
                        if rel['type'] != 'error':
                            st.markdown(f"**{rel['col1']} ↔️ {rel['col2']}**")
                            st.write(f"Loại: {rel['type']}")
                            st.write(f"Mô tả: {rel['description']}")
                            st.write(f"Tin cậy: {rel['confidence']:.1%}")
                            st.markdown("---")
                else:
                    st.info("Không phát hiện mối quan hệ ngữ nghĩa")
    
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"❌ Lỗi trong quá trình phân tích: {str(e)}")
        
        # Show debug info
        with st.expander("🐛 Thông tin Debug"):
            st.write(f"**Lỗi:** {str(e)}")
            st.write(f"**Loại phân tích:** {analysis_type}")
            st.write(f"**Dataset 1:** {dataset1[1]} - {df1.shape}")
            st.write(f"**Dataset 2:** {dataset2[1]} - {df2.shape}")

# Advanced query section
st.markdown("---")
st.subheader("💬 Đặt Câu Hỏi Qua Các Bộ Dữ Liệu")

# Provide example questions
with st.expander("💡 Câu hỏi ví dụ", expanded=False):
    example_questions = [
        "Có mối quan hệ nào giữa doanh thu và số lượng khách hàng không?",
        "Xu hướng theo thời gian giữa hai bộ dữ liệu có giống nhau không?",
        "Các yếu tố nào ảnh hưởng chung đến cả hai bộ dữ liệu?",
        "Có thể dự đoán dữ liệu bộ 2 dựa trên bộ 1 không?",
        "Phân khúc khách hàng nào xuất hiện ở cả hai nguồn dữ liệu?"
    ]
    
    for q in example_questions:
        if st.button(f"📝 {q}", key=f"example_{q[:20]}"):
            st.session_state.query_input = q

query_input = st.text_area(
    "Đặt câu hỏi phức tạp trải rộng cả hai bộ dữ liệu:",
    value=st.session_state.get('query_input', ''),
    placeholder="Ví dụ: Tương quan giữa doanh thu và satisfaction score như thế nào?",
    height=100,
    help="Đặt câu hỏi cụ thể để nhận được phân tích chi tiết từ AI"
)

if st.button("🎯 Trả Lời Câu Hỏi", type="secondary") and query_input:
    with st.spinner("🤖 AI đang phân tích câu hỏi của bạn..."):
        enhanced_prompt = f"""
        Bạn là một chuyên gia phân tích dữ liệu. Hãy trả lời câu hỏi sau dựa trên hai bộ dữ liệu:
        
        Bộ Dữ Liệu 1: {dataset1[1]}
        - Kích thước: {df1.shape}
        - Các cột: {list(df1.columns)}
        - Mẫu dữ liệu: {df1.head(2).to_dict()}
        
        Bộ Dữ Liệu 2: {dataset2[1]}
        - Kích thước: {df2.shape}
        - Các cột: {list(df2.columns)}
        - Mẫu dữ liệu: {df2.head(2).to_dict()}
        
        Câu Hỏi: {query_input}
        
        Hãy phân tích và trả lời một cách chi tiết, bao gồm:
        1. Xác định các cột và dữ liệu liên quan
        2. Phương pháp phân tích phù hợp
        3. Kết quả và insights cụ thể
        4. Khuyến nghị hành động
        5. Các giới hạn của phân tích
        
        Sử dụng markdown để format câu trả lời một cách đẹp mắt.
        """
        
        response = safe_llm_invoke(enhanced_prompt)
        
        st.markdown("### 🎯 Kết Quả Phân Tích")
        st.markdown(f"""
        <div class="analysis-result">
            {response}
        </div>
        """, unsafe_allow_html=True)

# Export and save options
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Xuất Báo Cáo Tóm Tắt", use_container_width=True):
        st.info("🔄 Tính năng xuất báo cáo đang được phát triển...")

with col2:
    if st.button("💾 Lưu Kết Quả Phân Tích", use_container_width=True):
        st.info("🔄 Tính năng lưu kết quả đang được phát triển...")

with col3:
    if st.button("📊 Tạo Dashboard Tổng Hợp", use_container_width=True):
        try:
            with st.spinner("📊 Đang tạo dashboard..."):
                # Create a comprehensive dashboard
                dashboard_fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        'So sánh Kích thước Dữ liệu',
                        'Phân phối Kiểu Dữ liệu',
                        'Chất lượng Dữ liệu',
                        'Tổng quan Mối quan hệ'
                    ],
                    specs=[[{"type": "bar"}, {"type": "pie"}],
                           [{"type": "bar"}, {"type": "scatter"}]]
                )
                
                # Chart 1: Data size comparison
                dashboard_fig.add_trace(
                    go.Bar(
                        x=[dataset1[1][:20], dataset2[1][:20]],
                        y=[df1.shape[0], df2.shape[0]],
                        name="Số hàng",
                        marker_color=['#667eea', '#764ba2']
                    ),
                    row=1, col=1
                )
                
                # Chart 2: Data type distribution
                type_dist1 = df1.dtypes.value_counts()
                dashboard_fig.add_trace(
                    go.Pie(
                        labels=[f"{dataset1[1][:10]}: {idx}" for idx in type_dist1.index],
                        values=type_dist1.values,
                        name="Kiểu dữ liệu"
                    ),
                    row=1, col=2
                )
                
                # Chart 3: Data quality comparison
                missing1 = (df1.isnull().sum().sum() / (df1.shape[0] * df1.shape[1])) * 100
                missing2 = (df2.isnull().sum().sum() / (df2.shape[0] * df2.shape[1])) * 100
                
                dashboard_fig.add_trace(
                    go.Bar(
                        x=[dataset1[1][:20], dataset2[1][:20]],
                        y=[100-missing1, 100-missing2],
                        name="Chất lượng (%)",
                        marker_color=['#28a745', '#20c997']
                    ),
                    row=2, col=1
                )
                
                # Chart 4: Relationship summary (if analysis was run)
                if hasattr(st.session_state, 'analysis_results'):
                    # Use stored results
                    pass
                else:
                    # Simple overview
                    common_cols = len(set(df1.columns) & set(df2.columns))
                    total_cols = len(set(df1.columns) | set(df2.columns))
                    
                    dashboard_fig.add_trace(
                        go.Scatter(
                            x=[common_cols],
                            y=[total_cols],
                            mode='markers',
                            marker=dict(size=50, color='#ff6b6b'),
                            name="Mối quan hệ",
                            text=[f"Chung: {common_cols}/{total_cols}"],
                            textposition="middle center"
                        ),
                        row=2, col=2
                    )
                
                dashboard_fig.update_layout(
                    height=800,
                    title_text=f"Dashboard Phân tích: {dataset1[1]} vs {dataset2[1]}",
                    showlegend=True
                )
                
                st.plotly_chart(dashboard_fig, use_container_width=True)
                st.success("✅ Dashboard đã được tạo!")
                
        except Exception as e:
            st.error(f"❌ Lỗi tạo dashboard: {str(e)}")

# Tips and best practices
st.markdown("---")
st.subheader("💡 Mẹo Phân tích Chéo Hiệu quả")

with st.expander("📚 Hướng dẫn Sử dụng", expanded=False):
    st.markdown("""
    ### 🎯 Lựa chọn Phương pháp Phân tích
    
    **🔍 Tương Đồng Cột:**
    - Sử dụng khi: Muốn tìm các trường dữ liệu tương tự giữa hai bộ dữ liệu
    - Phù hợp cho: Việc hợp nhất dữ liệu, chuẩn hóa schema
    - Ví dụ: Tìm "customer_id" và "khach_hang_id" có cùng ý nghĩa
    
    **📊 Tương Quan Thống Kê:**
    - Sử dụng khi: Muốn tìm mối quan hệ số học giữa các biến
    - Phù hợp cho: Phân tích xu hướng, dự đoán, mô hình hóa
    - Ví dụ: Mối quan hệ giữa doanh thu và chi phí marketing
    
    **🧠 Mối Quan Hệ Ngữ Nghĩa:**
    - Sử dụng khi: Muốn hiểu ý nghĩa logic giữa các trường
    - Phù hợp cho: Thiết kế data warehouse, integration
    - Ví dụ: Mối quan hệ giữa "city" và "region"
    
    **🔄 Phân Tích Tổng Hợp:**
    - Sử dụng khi: Cần cái nhìn toàn diện về mối quan hệ
    - Phù hợp cho: Báo cáo tổng thể, ra quyết định strategice
    - Ví dụ: Đánh giá khả năng tích hợp toàn bộ hệ thống
    
    ### 🚀 Mẹo Tối ưu
    
    1. **Chuẩn bị Dữ liệu:**
       - Đảm bảo dữ liệu sạch và có cấu trúc
       - Thống nhất format ngày tháng, số liệu
       - Loại bỏ các cột không cần thiết
    
    2. **Đặt Câu hỏi Đúng:**
       - Cụ thể về mục tiêu phân tích
       - Đề cập đến tên cột và bối cảnh
       - Yêu cầu insights có thể hành động
    
    3. **Diễn giải Kết quả:**
       - Xem xét độ tin cậy và p-value
       - Cân nhắc kích thước mẫu
       - Kiểm tra tính hợp lý của kết quả
    
    4. **Hành động Tiếp theo:**
       - Lưu các phát hiện quan trọng
       - Tạo workflow cho việc cập nhật định kỳ
       - Chia sẻ insights với team
    """)

with st.expander("⚠️ Lưu ý Quan trọng", expanded=False):
    st.markdown("""
    ### 🔔 Những điều Cần Lưu ý
    
    **📊 Về Tương quan:**
    - Tương quan ≠ Nhân quả
    - Kiểm tra outliers có thể ảnh hưởng kết quả
    - P-value thấp không có nghĩa là tương quan có ý nghĩa thực tế
    
    **🔍 Về Phân tích Ngữ nghĩa:**
    - AI có thể đưa ra gợi ý sai
    - Cần kiểm tra lại với hiểu biết domain
    - Các mối quan hệ phức tạp có thể bị bỏ qua
    
    **⚡ Về Hiệu năng:**
    - Bộ dữ liệu lớn có thể mất nhiều thời gian
    - Giới hạn số lượng cột để tối ưu tốc độ
    - Cache kết quả cho việc phân tích lặp lại
    
    **🎯 Về Kết quả:**
    - Luôn validate kết quả với business logic
    - Xem xét context và thời gian thu thập dữ liệu
    - Cần có plan backup nếu phân tích thất bại
    """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔗 VizGenie-GPT Cross Analysis**")
    st.caption("Khám phá mối quan hệ ẩn trong dữ liệu")

with col2:
    if hasattr(st.session_state, 'analysis_results'):
        st.markdown("**✅ Trạng thái**")
        st.caption("Phân tích đã hoàn thành")
    else:
        st.markdown("**⏳ Trạng thái**")
        st.caption("Sẵn sàng phân tích")

with col3:
    st.markdown("**💡 Mẹo**")
    st.caption("Thử các loại phân tích khác nhau để có cái nhìn toàn diện!")