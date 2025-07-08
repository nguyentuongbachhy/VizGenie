import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from src.utils import (get_all_datasets, get_dataset, 
                       save_dataset_analysis, get_dataset_analysis, 
                       delete_dataset_analysis, is_analysis_outdated)
from src.models.llms import load_llm
import json
import time

st.set_page_config(page_title="📂 Chi Tiết Bộ Dữ Liệu", layout="wide")
st.title("📂 Chi Tiết Bộ Dữ Liệu")

# Add custom CSS for better styling
st.markdown("""
<style>
    .data-description-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 2px solid #667eea30;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .column-analysis-card {
        color: black;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .insight-badge {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.25rem 0;
    }
    .warning-badge {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.25rem 0;
    }
    .error-badge {
        background: linear-gradient(135deg, #dc3545 0%, #e55353 100%);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.25rem 0;
    }
    .cache-info {
        background: #e3f2fd;
        border: 1px solid #2196f3;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

llm = load_llm("gpt-3.5-turbo")

# ---------- Helper functions with enhanced error handling ----------
def safe_read_csv(file_path):
    """Safely read CSV with multiple encoding attempts"""
    for enc in ['utf-8', 'ISO-8859-1', 'utf-16', 'cp1252']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Error reading file with {enc}: {str(e)}")
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode file with common encodings.")

def extract_llm_content(response):
    """Trích xuất nội dung từ LLM response object"""
    try:
        # Nếu response có thuộc tính content
        if hasattr(response, 'content'):
            return response.content
        
        # Nếu response là string
        elif isinstance(response, str):
            return response
        
        # Nếu response có thuộc tính text
        elif hasattr(response, 'text'):
            return response.text
        
        # Nếu response có thuộc tính message và content
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            return response.message.content
        
        # Fallback: convert to string
        else:
            return str(response)
            
    except Exception as e:
        st.warning(f"Không thể trích xuất nội dung LLM: {str(e)}")
        return "Không xác định được ý nghĩa"

def analyze_column(col_name, series):
    """Enhanced column analysis with better error handling"""
    try:
        info = {
            'name': col_name, 
            'dtype': str(series.dtype), 
            'missing_pct': series.isna().mean() * 100, 
            'unique': series.nunique(),
            'total_count': len(series)
        }
        
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            info.update({
                'min': desc['min'], 
                'max': desc['max'], 
                'mean': desc['mean'],
                'median': series.median(), 
                'std': desc['std'],
                'outliers': ((series < (desc['25%'] - 1.5*(desc['75%'] - desc['25%']))) | 
                           (series > (desc['75%'] + 1.5*(desc['75%'] - desc['25%'])))).sum(),
                'type': 'Numeric',
                'skewness': series.skew(),
                'kurtosis': series.kurtosis()
            })
        elif series.nunique() == 2:
            info['type'] = 'Boolean'
            info['value_counts'] = series.value_counts().to_dict()
        elif info['unique'] == len(series):
            info['type'] = 'ID'
        elif info['unique'] <= 20:
            info['type'] = 'Category'
            info['value_counts'] = series.value_counts().head(10).to_dict()
        else:
            info['type'] = 'Text'
            info['avg_length'] = series.astype(str).str.len().mean()
            info['max_length'] = series.astype(str).str.len().max()
        
        return info
    except Exception as e:
        return {
            'name': col_name,
            'dtype': 'Error',
            'missing_pct': 100,
            'unique': 0,
            'type': 'Error',
            'error': str(e)
        }

def guess_column_semantic_llm(col_name, sample_values=None):
    """Enhanced semantic analysis with sample values"""
    try:
        sample_text = ""
        if sample_values is not None and len(sample_values) > 0:
            # Convert sample values to string and take first few
            sample_str = [str(v) for v in sample_values[:5] if pd.notna(v)]
            if sample_str:
                sample_text = f" Giá trị mẫu: {', '.join(sample_str)}"
        
        prompt = f"Loại ngữ nghĩa của cột '{col_name}'{sample_text} là gì? Trả lời bằng 3-5 từ tiếng Việt mô tả ý nghĩa (ví dụ: 'ID khách hàng', 'Ngày sinh', 'Tên sản phẩm')."
        
        response = llm.invoke(prompt)
        
        # Sử dụng hàm extract_llm_content để lấy nội dung
        result = extract_llm_content(response)
        return result.strip()
        
    except Exception as e:
        return f"Không xác định ({str(e)[:50]}...)"

@st.cache_data(show_spinner=False)
def get_cleaning_suggestions(col_stats, user_description=""):
    """Enhanced cleaning suggestions with user context"""
    try:
        cols_description = "\n".join([
            f"Cột: {col['name']} | Loại: {col['dtype']} | Thiếu: {col['missing_pct']:.2f}% | Duy nhất: {col['unique']}" 
            for col in col_stats if 'error' not in col
        ])
        
        context_text = f"\nMô tả người dùng: {user_description}" if user_description else ""
        
        prompt = f"""
Dựa trên tóm tắt sau về các cột trong bộ dữ liệu:
{cols_description}{context_text}

Hãy đề xuất kế hoạch làm sạch với các quy tắc sau:
- Chỉ xóa các cột nếu tỷ lệ thiếu > 70% hoặc toàn bộ là ID không cần thiết.
- Đối với các cột có giá trị thiếu ≤ 70%:
    - Nếu là số: điền bằng trung vị hoặc trung bình.
    - Nếu là phân loại: điền bằng mode hoặc 'Unknown'.
- Chỉ loại bỏ ngoại lệ từ các cột số có outliers > 5% tổng dữ liệu.
- Chuẩn hóa các cột số chỉ khi cần thiết cho phân tích.
- Ưu tiên giữ nguyên dữ liệu nếu có thể.
- Đề xuất chuyển đổi kiểu dữ liệu nếu phù hợp.

Trả về kế hoạch dưới dạng danh sách có cấu trúc rõ ràng với lý do.
"""
        response = llm.invoke(prompt)
        return extract_llm_content(response)
    except Exception as e:
        return f"Lỗi tạo đề xuất làm sạch: {str(e)}"

@st.cache_data(show_spinner=False)
def refine_cleaning_strategy(user_input, _base_plan):
    """Refine cleaning strategy based on user input"""
    try:
        base_plan_text = extract_llm_content(_base_plan)
        
        prompt = f"""
Kế hoạch làm sạch hiện tại:
{base_plan_text}

Người dùng muốn điều chỉnh: {user_input}

Cập nhật kế hoạch làm sạch phù hợp với yêu cầu của người dùng. Giữ nguyên các phần tốt và chỉ thay đổi theo yêu cầu.
"""
        response = llm.invoke(prompt)
        return extract_llm_content(response)
    except Exception as e:
        return f"Lỗi cập nhật kế hoạch: {str(e)}"

@st.cache_data(show_spinner=False)
def generate_cleaning_code_from_plan(_plan):
    """Enhanced code generation with better error handling"""
    try:
        plan_text = extract_llm_content(_plan)
        
        prompt = f"""
Chuyển đổi kế hoạch làm sạch sau thành mã Python hợp lệ sử dụng pandas.
Chỉ trả về mã Python có thể thực thi trực tiếp.
Giả định dataframe được đặt tên là `df`.

Quan trọng:
- Luôn kiểm tra dtype trước khi áp dụng .str methods
- Xử lý lỗi với try-except cho từng bước
- Thêm print statements để theo dõi quá trình
- Đảm bảo mã hoạt động với dữ liệu thực

Kế hoạch Làm sạch:
{plan_text}

Trả về mã Python đầy đủ:
"""
        response = llm.invoke(prompt)
        return extract_llm_content(response)
    except Exception as e:
        return f"# Lỗi tạo mã: {str(e)}\nprint('Không thể tạo mã làm sạch')"

def generate_insight(info):
    """Generate insights for column analysis"""
    try:
        if 'error' in info:
            return f"❌ Lỗi phân tích: {info['error']}"
        
        if info['type'] == 'ID':
            return "🔹 Đây là cột định danh duy nhất."
        
        if info['missing_pct'] > 50:
            return f"⚠️ {info['missing_pct']:.1f}% giá trị thiếu - cần xem xét loại bỏ."
        elif info['missing_pct'] > 10:
            return f"⚠️ {info['missing_pct']:.1f}% giá trị thiếu - cần điền bổ sung."
        
        if info['type'] == 'Numeric':
            if 'std' in info and info['std'] < 1e-3:
                return "⚠️ Độ biến thiên rất thấp - có thể là hằng số."
            if 'outliers' in info and info['outliers'] > 0:
                outlier_pct = (info['outliers'] / info['total_count']) * 100
                if outlier_pct > 5:
                    return f"⚠️ {info['outliers']} ngoại lệ ({outlier_pct:.1f}%) - cần kiểm tra."
        
        if info['unique'] < 5 and info['type'] == 'Category':
            return "ℹ️ Phân loại với ít giá trị - phù hợp cho grouping."
        
        if info['type'] == 'Text' and 'avg_length' in info:
            if info['avg_length'] > 100:
                return f"📝 Văn bản dài (TB: {info['avg_length']:.0f} ký tự) - có thể cần xử lý NLP."
        
        return "✅ Không phát hiện vấn đề lớn."
    except Exception as e:
        return f"❌ Lỗi tạo insight: {str(e)}"

def plot_distribution(col_name, series):
    """Enhanced distribution plotting with error handling"""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if pd.api.types.is_numeric_dtype(series):
            # Numeric distribution
            clean_series = series.dropna()
            if len(clean_series) > 0:
                ax.hist(clean_series, bins=min(30, len(clean_series.unique())), 
                       color='#69b3a2', alpha=0.7, edgecolor='black')
                ax.axvline(clean_series.mean(), color='red', linestyle='--', 
                          label=f'Trung bình: {clean_series.mean():.2f}')
                ax.axvline(clean_series.median(), color='orange', linestyle='--', 
                          label=f'Trung vị: {clean_series.median():.2f}')
                ax.legend()
                ax.set_xlabel(col_name)
                ax.set_ylabel('Tần suất')
        else:
            # Categorical distribution
            vc = series.fillna("NaN").value_counts().head(15)  # Show top 15
            if len(vc) > 0:
                bars = ax.bar(range(len(vc)), vc.values, color='#8c54ff', alpha=0.7)
                ax.set_xticks(range(len(vc)))
                ax.set_xticklabels([str(x) for x in vc.index], rotation=45, ha='right')
                ax.set_ylabel('Số lượng')
                
                # Add value labels on bars
                for bar, value in zip(bars, vc.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vc.values)*0.01,
                           str(value), ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f"Phân phối: {col_name}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.error(f"Lỗi vẽ biểu đồ cho {col_name}: {str(e)}")

def perform_column_analysis(df, dataset_id):
    """Thực hiện phân tích cột với progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    col_analyses = []
    
    for i, col in enumerate(df.columns):
        status_text.text(f"Đang phân tích cột: {col}")
        progress_bar.progress((i + 1) / len(df.columns))
        
        # Column analysis
        stats = analyze_column(col, df[col])
        
        # Get sample values for semantic analysis
        sample_vals = df[col].dropna().head(5).tolist()
        semantic = guess_column_semantic_llm(col, sample_vals)
        stats['semantic'] = semantic
        
        col_analyses.append(stats)
        
        time.sleep(0.1)  # Brief pause to show progress
    
    status_text.text("✅ Hoàn thành phân tích!")
    progress_bar.progress(1.0)
    
    # Lưu kết quả phân tích vào database
    save_dataset_analysis(dataset_id, col_analyses)
    
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    return col_analyses

def extract_valid_code(llm_response):
    """Extract valid Python code from LLM response"""
    try:
        # Try to extract code between ```python and ```
        match = re.search(r"```(?:python)?\n(.*?)```", llm_response.strip(), re.DOTALL)
        if match:
            return match.group(1)
        
        # If no code blocks, try to extract lines that look like Python code
        lines = llm_response.splitlines()
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("Kế hoạch"):
                # Basic check if it looks like Python code
                if any(keyword in stripped for keyword in ['df[', 'df.', 'pd.', 'np.', '=', 'print(', 'try:', 'except:']):
                    code_lines.append(line)
        
        return "\n".join(code_lines) if code_lines else llm_response
    except Exception as e:
        return f"# Error extracting code: {str(e)}\n{llm_response}"

def fix_numeric_strings(df):
    """Enhanced numeric string fixing"""
    fixed_cols = []
    for col in df.select_dtypes(include='object').columns:
        try:
            if df[col].dropna().apply(lambda x: isinstance(x, str)).all():
                # Try to convert numeric strings
                original_type = df[col].dtype
                test_series = df[col].str.replace(',', '', regex=False)
                test_series = pd.to_numeric(test_series, errors='coerce')
                
                # If more than 80% can be converted, do the conversion
                valid_ratio = test_series.notna().sum() / len(test_series)
                if valid_ratio > 0.8:
                    df[col] = test_series
                    fixed_cols.append(col)
        except Exception as e:
            continue
    
    if fixed_cols:
        st.info(f"✅ Đã chuyển đổi các cột số: {', '.join(fixed_cols)}")
    
    return df

def show_skew_kurtosis(df, cleaned_df):
    """Enhanced skewness and kurtosis analysis"""
    try:
        raw_cols = df.select_dtypes(include='number').columns
        clean_cols = cleaned_df.select_dtypes(include='number').columns
        numeric_cols = list(set(raw_cols).intersection(set(clean_cols)))

        if not numeric_cols:
            st.info("Không có cột số chung nào khả dụng cho báo cáo độ lệch/độ nhọn.")
            return

        # Create comprehensive report
        report = pd.DataFrame(index=numeric_cols)
        report['Độ lệch (Trước)'] = df[numeric_cols].skew()
        report['Độ nhọn (Trước)'] = df[numeric_cols].kurtosis()
        report['Độ lệch (Sau)'] = cleaned_df[numeric_cols].skew()
        report['Độ nhọn (Sau)'] = cleaned_df[numeric_cols].kurtosis()
        
        # Calculate improvements
        report['Cải thiện Độ lệch'] = abs(report['Độ lệch (Trước)']) - abs(report['Độ lệch (Sau)'])
        report['Cải thiện Độ nhọn'] = abs(report['Độ nhọn (Trước)']) - abs(report['Độ nhọn (Sau)'])
        
        st.dataframe(report.round(3), use_container_width=True)

        # Visualization with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Skewness comparison
        x_pos = np.arange(len(numeric_cols))
        width = 0.35
        
        ax1.bar(x_pos - width/2, report['Độ lệch (Trước)'], width, 
               label='Trước', alpha=0.8, color='#ff7f7f')
        ax1.bar(x_pos + width/2, report['Độ lệch (Sau)'], width,
               label='Sau', alpha=0.8, color='#7fbf7f')
        
        ax1.set_xlabel('Đặc trưng')
        ax1.set_ylabel('Độ lệch')
        ax1.set_title('So sánh Độ lệch Trước vs Sau Làm sạch')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Kurtosis comparison
        ax2.bar(x_pos - width/2, report['Độ nhọn (Trước)'], width,
               label='Trước', alpha=0.8, color='#ff7f7f')
        ax2.bar(x_pos + width/2, report['Độ nhọn (Sau)'], width,
               label='Sau', alpha=0.8, color='#7fbf7f')
        
        ax2.set_xlabel('Đặc trưng')
        ax2.set_ylabel('Độ nhọn')
        ax2.set_title('So sánh Độ nhọn Trước vs Sau Làm sạch')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Generate AI insights
        try:
            interpretation_prompt = f"""
Phân tích báo cáo độ lệch và độ nhọn sau:

{report.to_string()}

Hãy đưa ra nhận xét về:
1. Những cải thiện đáng kể trong phân phối dữ liệu
2. Các cột còn cần xử lý thêm
3. Tác động đến chất lượng phân tích
4. Đề xuất bước tiếp theo

Trả lời bằng markdown với format đẹp và dễ hiểu.
"""
            
            response = llm.invoke(interpretation_prompt)
            interpretation = extract_llm_content(response)
            st.markdown("### 🤖 Phân tích AI")
            st.markdown(interpretation)
            
        except Exception as e:
            st.warning(f"Không thể tạo phân tích AI: {str(e)}")

    except Exception as e:
        st.error(f"Lỗi trong phân tích độ lệch/độ nhọn: {str(e)}")

# Main application
def main():
    # Load datasets
    datasets = get_all_datasets()
    if not datasets:
        st.warning("Không tìm thấy bộ dữ liệu nào. Vui lòng tải lên một bộ dữ liệu trong Bảng điều khiển.")
        st.stop()

    # Dataset selection
    selected = st.selectbox("Chọn bộ dữ liệu:", [f"{d[0]} - {d[1]}" for d in datasets])
    dataset_id = int(selected.split(" - ")[0])
    dataset = get_dataset(dataset_id)
    
    try:
        df = safe_read_csv(dataset[2])
    except Exception as e:
        st.error(f"❌ Không thể đọc file: {str(e)}")
        st.stop()

    st.markdown(f"### Bộ dữ liệu: `{dataset[1]}` — {df.shape[0]:,} hàng × {df.shape[1]} cột")

    # Add data description section
    st.markdown("### 📝 Mô tả Dữ liệu")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User data description input
        user_description = st.text_area(
            "✍️ Mô tả bộ dữ liệu của bạn:",
            height=100,
            placeholder="Ví dụ: Dữ liệu bán hàng từ 2023-2024, bao gồm thông tin khách hàng, sản phẩm và doanh thu. Được thu thập từ hệ thống POS...",
            help="Mô tả chi tiết giúp AI hiểu rõ hơn về ngữ cảnh và đưa ra gợi ý chính xác hơn"
        )
        
        if user_description:
            st.session_state.user_data_description = user_description
    
    with col2:
        st.markdown("**💡 Mẹo viết mô tả tốt:**")
        st.markdown("""
        - Nguồn gốc dữ liệu
        - Mục đích thu thập  
        - Khoảng thời gian
        - Ý nghĩa các cột chính
        - Đơn vị đo lường
        - Lưu ý đặc biệt
        """)

    # Create tabs for different analysis sections
    tab1, tab2, tab3 = st.tabs(["📊 Tổng quan Chi tiết", "🧼 Làm sạch Thông minh", "📈 Phân tích Phân phối"])

    with tab1:
        st.markdown("### 🔍 Phân tích Chi tiết từng Cột")
        
        # Kiểm tra xem đã có phân tích cached không
        cached_analysis = get_dataset_analysis(dataset_id)
        
        # Hiển thị thông tin cache
        if cached_analysis:
            is_outdated = is_analysis_outdated(cached_analysis, dataset[4])  # dataset[4] là upload_time
            
            if is_outdated:
                st.markdown("""
                <div class="cache-info">
                    ⚠️ <strong>Phân tích cũ được tìm thấy</strong> - Dataset đã được cập nhật sau lần phân tích cuối. 
                    Nên chạy phân tích lại để có kết quả chính xác nhất.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="cache-info">
                    ✅ <strong>Phân tích có sẵn</strong> - Đã phân tích lúc {cached_analysis['updated_at']}. 
                    Bạn có thể sử dụng kết quả này hoặc chạy phân tích lại.
                </div>
                """, unsafe_allow_html=True)
        
        # Buttons cho việc phân tích
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if cached_analysis and not is_analysis_outdated(cached_analysis, dataset[4]):
                if st.button("📋 Sử dụng Phân tích Có sẵn", type="secondary"):
                    st.session_state.col_analyses = cached_analysis['analysis']
                    st.success("✅ Đã tải phân tích từ cache!")
                    st.rerun()
        
        with col_btn2:
            if st.button("🚀 Chạy Phân tích Mới", type="primary"):
                with st.spinner("🔄 Đang phân tích dữ liệu..."):
                    col_analyses = perform_column_analysis(df, dataset_id)
                    st.session_state.col_analyses = col_analyses
                    st.success("✅ Hoàn thành phân tích mới!")
                    st.rerun()
        
        with col_btn3:
            if cached_analysis:
                if st.button("🗑️ Xóa Cache", type="secondary"):
                    delete_dataset_analysis(dataset_id)
                    st.success("🗑️ Đã xóa cache phân tích!")
                    st.rerun()
        
        # Display analysis results
        if hasattr(st.session_state, 'col_analyses'):
            st.markdown("---")
            
            for analysis in st.session_state.col_analyses:
                col_name = analysis['name']
                
                with st.container():
                    st.markdown(f"""
                    <div class="column-analysis-card">
                        <h4>📌 {col_name}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_left, col_right = st.columns([2, 3])
                    
                    with col_left:
                        # Basic statistics
                        st.markdown(f"**🏷️ Loại:** `{analysis['type']}`")
                        st.markdown(f"**📊 Kiểu dữ liệu:** `{analysis['dtype']}`")
                        st.markdown(f"**🧩 Ý nghĩa:** {analysis['semantic']}")
                        st.markdown(f"**🔢 Duy nhất:** `{analysis['unique']:,}`")
                        st.markdown(f"**❌ Thiếu:** `{analysis['missing_pct']:.2f}%`")
                        
                        # Type-specific information
                        if analysis['type'] == 'Numeric' and 'mean' in analysis:
                            st.markdown(f"**📈 Trung bình:** `{analysis['mean']:.2f}`")
                            st.markdown(f"**📊 Trung vị:** `{analysis['median']:.2f}`")
                            st.markdown(f"**📏 Độ lệch chuẩn:** `{analysis['std']:.2f}`")
                            st.markdown(f"**⚠️ Ngoại lệ:** `{analysis['outliers']}`")
                            if 'skewness' in analysis:
                                st.markdown(f"**↗️ Độ lệch:** `{analysis['skewness']:.2f}`")
                        
                        elif analysis['type'] == 'Category' and 'value_counts' in analysis:
                            st.markdown("**🏆 Top giá trị:**")
                            for val, count in list(analysis['value_counts'].items())[:3]:
                                st.markdown(f"  - `{val}`: {count}")
                        
                        elif analysis['type'] == 'Text' and 'avg_length' in analysis:
                            st.markdown(f"**📝 Độ dài TB:** `{analysis['avg_length']:.1f}`")
                            st.markdown(f"**📏 Độ dài tối đa:** `{analysis['max_length']}`")
                        
                        # Generate and display insight
                        insight = generate_insight(analysis)
                        if "✅" in insight:
                            badge_class = "insight-badge"
                        elif "⚠️" in insight:
                            badge_class = "warning-badge"
                        else:
                            badge_class = "error-badge"
                        
                        st.markdown(f'<span class="{badge_class}">{insight}</span>', 
                                  unsafe_allow_html=True)
                    
                    with col_right:
                        # Distribution plot
                        if analysis['type'] != 'Error':
                            plot_distribution(col_name, df[col_name])
                        else:
                            st.error(f"Lỗi phân tích cột: {analysis.get('error', 'Unknown error')}")
                    
                    st.markdown("---")

    with tab2:
        st.markdown("### 🧼 Làm sạch Dữ liệu Thông minh")
        
        # Get column statistics
        if not hasattr(st.session_state, 'col_analyses'):
            st.info("🔄 Vui lòng chạy phân tích trong tab 'Tổng quan Chi tiết' trước.")
        else:
            col_stats = st.session_state.col_analyses
            
            # Display summary table
            summary_data = []
            for stat in col_stats:
                if 'error' not in stat:
                    summary_data.append({
                        'Cột': stat['name'],
                        'Loại': stat['type'],
                        'Kiểu dữ liệu': stat['dtype'],
                        'Ý nghĩa': stat['semantic'][:30] + "..." if len(stat['semantic']) > 30 else stat['semantic'],
                        'Duy nhất': stat['unique'],
                        'Thiếu %': f"{stat['missing_pct']:.1f}%"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Get user description for context
            user_desc = st.session_state.get('user_data_description', '')
            
            # Generate cleaning suggestions
            st.markdown("### 🤖 Đề xuất Làm sạch AI")
            
            if st.button("📋 Tạo Kế hoạch Làm sạch", type="primary"):
                with st.spinner("🤖 AI đang phân tích và tạo kế hoạch..."):
                    base_plan = get_cleaning_suggestions(col_stats, user_desc)
                    st.session_state.base_cleaning_plan = base_plan
            
            # Display cleaning plan
            if hasattr(st.session_state, 'base_cleaning_plan'):
                st.markdown("#### 📋 Kế hoạch Làm sạch")
                st.markdown(st.session_state.base_cleaning_plan)
                
                # Allow user customization
                if st.toggle("🛠️ Tùy chỉnh Kế hoạch"):
                    user_input = st.text_area(
                        "✍️ Điều chỉnh kế hoạch làm sạch:",
                        placeholder="Ví dụ: Không xóa cột ID, điền giá trị thiếu bằng 0 thay vì trung vị...",
                        height=100
                    )
                    
                    if user_input and st.button("🔄 Cập nhật Kế hoạch"):
                        with st.spinner("🔄 Đang cập nhật kế hoạch..."):
                            updated_plan = refine_cleaning_strategy(user_input, st.session_state.base_cleaning_plan)
                            st.session_state.base_cleaning_plan = updated_plan
                            st.success("✅ Đã cập nhật kế hoạch!")
                            st.rerun()
                
                # Generate cleaning code
                st.markdown("#### 🐍 Mã Python Làm sạch")
                
                if st.button("🔧 Tạo Mã Làm sạch"):
                    with st.spinner("🔧 Đang tạo mã Python..."):
                        code_raw = generate_cleaning_code_from_plan(st.session_state.base_cleaning_plan)
                        code_clean = extract_valid_code(code_raw)
                        st.session_state.cleaning_code = code_clean
                
                # Display and execute cleaning code
                if hasattr(st.session_state, 'cleaning_code'):
                    
                    # Show the code
                    with st.expander("👀 Xem Mã Làm sạch", expanded=True):
                        st.code(st.session_state.cleaning_code, language="python")
                    
                    # Execute cleaning
                    if st.button("🚀 Thực thi Làm sạch", type="primary"):
                        try:
                            with st.spinner("🔄 Đang làm sạch dữ liệu..."):
                                # Prepare execution environment
                                exec_globals = {
                                    'df': df.copy(), 
                                    'pd': pd, 
                                    'np': np, 
                                    'fix_numeric_strings': fix_numeric_strings,
                                    'print': st.write  # Redirect print to streamlit
                                }
                                
                                # Execute cleaning code
                                exec("df = fix_numeric_strings(df)\n" + st.session_state.cleaning_code, exec_globals)
                                cleaned_df = exec_globals['df']
                                
                                # Store cleaned data
                                st.session_state.cleaned_df = cleaned_df
                                st.session_state.raw_df = df
                                
                                st.success("✅ Làm sạch dữ liệu thành công!")
                                
                        except Exception as e:
                            st.error(f"❌ Lỗi khi thực thi mã làm sạch: {str(e)}")
                            
                            # Show debugging info
                            with st.expander("🐛 Thông tin Debug"):
                                st.write("**Lỗi chi tiết:**", str(e))
                                st.write("**Mã đã thực thi:**")
                                st.code(st.session_state.cleaning_code, language="python")
                
                # Show cleaned data preview
                if hasattr(st.session_state, 'cleaned_df'):
                    st.markdown("### ✅ Dữ liệu Đã làm sạch")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Trước làm sạch:**")
                        st.write(f"Kích thước: {df.shape}")
                        st.write(f"Giá trị thiếu: {df.isnull().sum().sum()}")
                        st.dataframe(df.head(3), use_container_width=True)
                    
                    with col2:
                        st.markdown("**✨ Sau làm sạch:**")
                        st.write(f"Kích thước: {st.session_state.cleaned_df.shape}")
                        st.write(f"Giá trị thiếu: {st.session_state.cleaned_df.isnull().sum().sum()}")
                        st.dataframe(st.session_state.cleaned_df.head(3), use_container_width=True)
                    
                    # Download cleaned data
                    csv_data = st.session_state.cleaned_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Tải xuống Dữ liệu Đã làm sạch",
                        data=csv_data,
                        file_name="cleaned_dataset.csv",
                        mime="text/csv",
                        key="download_cleaned"
                    )

    with tab3:
        st.markdown("### 📈 Phân tích Phân phối & Thống kê")
        
        if hasattr(st.session_state, 'cleaned_df') and hasattr(st.session_state, 'raw_df'):
            show_skew_kurtosis(st.session_state.raw_df, st.session_state.cleaned_df)
        else:
            st.info("🔄 Vui lòng chạy làm sạch dữ liệu trong tab '🧼 Làm sạch Thông minh' để xem phân tích này.")
            
            # Show basic distribution analysis for original data
            st.markdown("#### 📊 Phân tích Phân phối Cơ bản")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Chọn cột để phân tích:", numeric_cols)
                
                if selected_col:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Basic statistics
                        stats = df[selected_col].describe()
                        st.markdown("**📊 Thống kê mô tả:**")
                        st.dataframe(stats)
                        
                        # Additional metrics
                        skewness = df[selected_col].skew()
                        kurtosis = df[selected_col].kurtosis()
                        
                        st.markdown(f"**↗️ Độ lệch:** {skewness:.3f}")
                        st.markdown(f"**📈 Độ nhọn:** {kurtosis:.3f}")
                        
                        # Interpretation
                        if abs(skewness) < 0.5:
                            skew_interp = "Gần đối xứng"
                        elif abs(skewness) < 1:
                            skew_interp = "Hơi lệch"
                        else:
                            skew_interp = "Lệch mạnh"
                        
                        st.info(f"🔍 **Phân tích:** Phân phối {skew_interp}")
                    
                    with col2:
                        # Distribution plot
                        plot_distribution(selected_col, df[selected_col])

if __name__ == "__main__":
    main()