import streamlit as st
import pandas as pd
import os
from src.utils import export_eda_report_to_pdf, init_db, get_all_datasets, rename_dataset, safe_read_csv
import matplotlib.pyplot as plt
import seaborn as sns
import json
import textwrap
import re

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

st.set_page_config(page_title="Báo cáo EDA", layout="wide")
st.title("🧠 Báo cáo Phân tích Khám phá Dữ liệu (EDA)")

# LangChain LLM setup
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def clean_and_fix_json(raw_response):
    """
    Làm sạch và sửa lỗi JSON từ LLM response một cách mạnh mẽ
    """
    try:
        # Loại bỏ markdown code blocks
        cleaned = re.sub(r"^```(?:json)?", "", raw_response.strip(), flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        cleaned = cleaned.strip()
        
        # Loại bỏ comments trong JSON (// hoặc /* */)
        cleaned = re.sub(r'//.*?$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        
        # Sửa trailing commas
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Sửa single quotes thành double quotes (chỉ cho keys và strings)
        # Pattern phức tạp hơn để tránh sửa nhầm apostrophes trong content
        cleaned = re.sub(r"'([^']*)':", r'"\1":', cleaned)  # Keys
        cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)  # String values
        
        # Sửa các ký tự escape phổ biến
        cleaned = cleaned.replace('\n', '\\n')
        cleaned = cleaned.replace('\t', '\\t')
        
        # Thử parse trực tiếp
        return json.loads(cleaned)
        
    except json.JSONDecodeError as e:
        # Nếu vẫn lỗi, thử các phương pháp khác
        try:
            # Tìm và extract JSON object đầu tiên
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                raise ValueError("Không tìm thấy JSON object trong response")
                
        except Exception as e2:
            # Fallback: tạo structure mặc định với thông tin từ response
            st.error(f"❌ Không thể parse JSON. Lỗi: {str(e)}")
            st.error(f"Vị trí lỗi: line {e.lineno}, column {e.colno}")
            
            # Hiển thị raw response để debug
            with st.expander("🐛 Raw LLM Response (để debug)", expanded=False):
                st.text(raw_response)
                st.text("=" * 50)
                st.text("Cleaned response:")
                st.text(cleaned)
            
            # Trả về structure mặc định
            return create_fallback_eda_structure(raw_response)

def create_fallback_eda_structure(raw_response):
    """
    Tạo structure EDA mặc định khi không parse được JSON
    """
    return {
        "introduction": f"**Phân tích EDA tự động**\n\nĐã phát hiện lỗi khi tạo báo cáo EDA tự động. Dưới đây là thông tin cơ bản về dữ liệu:\n\n{raw_response[:500]}...",
        "data_quality": "**Đánh giá Chất lượng Dữ liệu**\n\nVui lòng kiểm tra thủ công chất lượng dữ liệu.",
        "univariate": [
            {
                "insight": "Phân tích đơn biến cần được thực hiện thủ công do lỗi tự động.",
                "code": "# Phân tích thủ công\nprint('Vui lòng kiểm tra dữ liệu thủ công')",
                "insight_after_chart": "Cần phân tích thủ công."
            }
        ],
        "correlation": {
            "insight": "Phân tích tương quan cần được thực hiện thủ công.",
            "code": "# Phân tích tương quan thủ công\nprint('Vui lòng tạo correlation matrix thủ công')",
            "insight_after_chart": "Cần phân tích tương quan thủ công."
        },
        "insights": ["Cần phân tích thủ công do lỗi tự động"],
        "recommendations": ["Kiểm tra lại dữ liệu và prompt", "Thử chạy lại báo cáo EDA"]
    }

def generate_eda_report_with_llm(df):
    """
    Tạo báo cáo EDA với error handling mạnh mẽ
    """
    prompt = f"""
Bạn là một nhà phân tích dữ liệu chuyên nghiệp. Phân tích bộ dữ liệu và trả về kết quả JSON hợp lệ CHÍNH XÁC.

QUAN TRỌNG: Trả về CHÍNH XÁC JSON hợp lệ, không có markdown, không có comments, không có trailing commas.

Cấu trúc JSON bắt buộc:
{{
  "introduction": "string",
  "data_quality": "string", 
  "univariate": [
    {{
      "insight": "string",
      "code": "string",
      "insight_after_chart": "string"
    }}
  ],
  "correlation": {{
    "insight": "string",
    "code": "string", 
    "insight_after_chart": "string"
  }},
  "insights": ["string1", "string2"],
  "recommendations": ["string1", "string2"]
}}

Metadata bộ dữ liệu:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Data types: {df.dtypes.to_dict()}
- Missing values: {df.isnull().sum().to_dict()}
- Numeric columns: {df.select_dtypes(include=['number']).columns.tolist()}

Tạo phân tích EDA chuyên nghiệp. Đảm bảo JSON hợp lệ 100%.
"""

    try:
        # Gọi LLM với error handling
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON với error handling mạnh mẽ
        return clean_and_fix_json(raw_content)
        
    except Exception as e:
        st.error(f"❌ Lỗi khi gọi LLM: {str(e)}")
        
        # Tạo báo cáo EDA cơ bản thay thế
        return create_manual_eda_report(df)

def create_manual_eda_report(df):
    """
    Tạo báo cáo EDA cơ bản khi LLM fail
    """
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    missing_info = df.isnull().sum()
    
    univariate_analyses = []
    
    # Tạo phân tích cho một vài cột đầu tiên
    for col in df.columns:
        if col in numeric_cols and len(df[col] == 0) < 0.5 * len(df):
            code = f"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df['{col}'], bins=30, kde=True)
plt.title('Phân phối của {col}')
plt.xlabel('{col}')
plt.ylabel('Tần suất')
plt.grid(True, alpha=0.3)
plt.tight_layout()
"""
        else:
            code = f"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
value_counts = df['{col}'].value_counts().head(10)
sns.barplot(x=value_counts.values, y=value_counts.index)
plt.title('Top 10 giá trị của {col}')
plt.xlabel('Số lượng')
plt.tight_layout()
"""
        
        univariate_analyses.append({
            "insight": f"Phân tích cột {col} - loại {'số' if col in numeric_cols else 'phân loại'}",
            "code": code,
            "insight_after_chart": f"Cần xem xét phân phối và patterns của {col}"
        })
    
    return {
        "introduction": f"""
## 📊 Giới thiệu Bộ dữ liệu

Bộ dữ liệu này có **{df.shape[0]:,} hàng** và **{df.shape[1]} cột**.

**Phân loại cột:**
- Cột số: {len(numeric_cols)} ({', '.join(numeric_cols)})
- Cột phân loại: {len(categorical_cols)} ({', '.join(categorical_cols)})

**Tổng quan nhanh:**
- Tổng số ô dữ liệu: {df.shape[0] * df.shape[1]:,}
- Ô trống: {df.isnull().sum().sum():,}
- Tỉ lệ hoàn thiện: {((df.shape[0] * df.shape[1] - df.isnull().sum().sum()) / (df.shape[0] * df.shape[1]) * 100):.1f}%
""",
        "data_quality": f"""
## 🧹 Đánh giá Chất lượng Dữ liệu

**Giá trị thiếu theo cột:**
{chr(10).join([f"- {col}: {count} ({count/len(df)*100:.1f}%)" for col, count in missing_info[missing_info > 0].items()][:10])}

**Đánh giá tổng quan:**
- Chất lượng dữ liệu: {'Tốt' if df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) < 0.05 else 'Cần cải thiện'}
- Các cột có vấn đề: {len(missing_info[missing_info > df.shape[0] * 0.1])} cột có >10% dữ liệu thiếu
""",
        "univariate": univariate_analyses,
        "correlation": {
            "insight": f"Phân tích tương quan giữa {len(numeric_cols)} biến số trong bộ dữ liệu.",
            "code": f"""
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = {numeric_cols}
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, cbar_kws={{'shrink': 0.8}})
    plt.title('Ma trận Tương quan')
    plt.tight_layout()
else:
    print('Cần ít nhất 2 cột số để tạo ma trận tương quan')
""",
            "insight_after_chart": "Ma trận tương quan giúp hiểu mối quan hệ giữa các biến số. Cần chú ý các cặp biến có tương quan mạnh (>0.7 hoặc <-0.7)."
        },
        "insights": [
            f"Bộ dữ liệu có kích thước vừa phải với {df.shape[0]:,} quan sát",
            f"Có {len(numeric_cols)} biến số và {len(categorical_cols)} biến phân loại",
            f"Tỉ lệ dữ liệu thiếu là {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%",
            "Cần kiểm tra thêm về outliers và data distribution"
        ],
        "recommendations": [
            "Xử lý dữ liệu thiếu trước khi phân tích sâu",
            "Kiểm tra và xử lý outliers trong các biến số", 
            "Thực hiện feature engineering nếu cần",
            "Xem xét normalizing/scaling cho machine learning"
        ]
    }

def generate_final_summary_prompt(sections):
    return textwrap.dedent("""
        Bạn là một nhà phân tích dữ liệu cấp cao được giao nhiệm vụ viết một báo cáo EDA cuối cùng chi tiết, chuyên nghiệp cho một bộ dữ liệu.
        Báo cáo của bạn sẽ được hiển thị trực tiếp cho các bên liên quan (ví dụ: quản trị trường học, nhóm khoa học dữ liệu) vì vậy nó phải toàn diện, sâu sắc và được viết bằng ngôn ngữ tự nhiên trôi chảy.

        Cấu trúc báo cáo trong markdown được định dạng tốt với các phần sau:

        ## 📘 Giới thiệu
        Tóm tắt nội dung bộ dữ liệu (số hàng, cột, loại dữ liệu) và mục đích của nó. Bao gồm bảng xem trước.

        ## 🧼 Chất lượng Dữ liệu
        Bình luận về giá trị thiếu, bản ghi trùng lặp và độ tin cậy tổng thể. Mô tả bất kỳ việc làm sạch dữ liệu nào cần thiết hoặc đã được thực hiện.

        ## 🔍 Phân tích Đơn biến
        Tóm tắt các mô hình chính được tìm thấy trong các cột riêng lẻ, đặc biệt là các cột số. Đề cập đến phân phối, giá trị phổ biến và giá trị ngoại lệ.
        Bao gồm tham chiếu đến các biểu đồ như biểu đồ tần suất hoặc biểu đồ cột, và đặt mỗi biểu đồ ngay sau điểm liên quan của nó. Đảm bảo thông tin chi tiết và biểu đồ liên quan xuất hiện cùng nhau trong kết xuất.

        ## 📊 Thông tin Tương quan
        Mô tả các mối quan hệ chính được khám phá giữa các cặp biến. Giải thích bản đồ nhiệt và chỉ ra các tương quan mạnh/yếu. Cung cấp ý nghĩa thực tế. Hiển thị bản đồ nhiệt tương quan gần mô tả.

        ## 💡 Thông tin Cuối cùng & Khuyến nghị
        Tóm tắt kết luận của bạn về dữ liệu.
        Đưa ra các khuyến nghị thực tế (ví dụ: cải thiện dữ liệu, lĩnh vực tập trung, đề xuất chính sách).

        Chỉ sử dụng markdown. Không có danh sách dấu đầu dòng trừ khi tóm tắt các hành động cuối cùng.
        Độ dài: khoảng 600-800 từ.
        Tông điệu: phân tích, có cấu trúc, hữu ích cho các bên liên quan.
    """) + f"""

Bối cảnh:
- Giới thiệu: {sections['introduction']}
- Chất lượng Dữ liệu: {sections['data_quality']}
- Đơn biến: {[b['insight_after_chart'] for b in sections['univariate'] if 'insight_after_chart' in b]}
- Tương quan: {sections['correlation']['insight_after_chart']}
- Thông tin Chính: {sections['insights']}
- Khuyến nghị: {sections['recommendations']}
"""

init_db()

# Load all datasets
datasets = get_all_datasets()
if not datasets:
    st.warning("Vui lòng tải lên một bộ dữ liệu trong Bảng điều khiển trước.")
    st.stop()

# Dataset selection
dataset_options = {f"{d[0]} - {d[1]}": d for d in datasets}
selected = st.selectbox("Chọn bộ dữ liệu để tạo báo cáo:", list(dataset_options.keys()))
dataset_id, name, rows, cols, uploaded, _ = dataset_options[selected]
file_path = os.path.join("data", "uploads", name)
df = safe_read_csv(file_path)

# Enhanced error handling cho việc tạo báo cáo
try:
    st.info("🤖 Đang tạo báo cáo EDA với AI...")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Đang phân tích dữ liệu...")
    progress_bar.progress(25)
    
    # Call LLM-generated EDA content với error handling
    eda_sections = generate_eda_report_with_llm(df)
    
    status_text.text("Đang tạo insights...")
    progress_bar.progress(75)
    
    status_text.text("Hoàn thành!")
    progress_bar.progress(100)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    st.success("✅ Đã tạo báo cáo EDA thành công!")
    
except Exception as e:
    st.error(f"❌ Lỗi khi tạo báo cáo EDA: {str(e)}")
    st.info("🔄 Đang chuyển sang chế độ báo cáo thủ công...")
    
    # Fallback to manual report
    eda_sections = create_manual_eda_report(df)
    st.warning("⚠️ Đã tạo báo cáo cơ bản. Một số tính năng AI có thể không khả dụng.")

# Call LLM-generated EDA content
tabs = st.tabs(["📘 Giới thiệu", "🧼 Chất lượng Dữ liệu", "🔍 Đơn biến", "📊 Tương quan", "💡 Thông tin", "📄 Báo cáo Đầy đủ"])

# --- 📘 Introduction ---
with tabs[0]:
    st.markdown(eda_sections['introduction'])
    st.subheader("📌 Xem trước Bộ dữ liệu")
    st.dataframe(df.head(10))

# --- 🧼 Data Quality ---
with tabs[1]:
    st.markdown(eda_sections['data_quality'])
    st.subheader("Giá trị Thiếu")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.dataframe(missing)
    else:
        st.success("Không phát hiện giá trị thiếu.")
    st.subheader("Hàng Trùng lặp")
    st.write(f"Số hàng trùng lặp: **{df.duplicated().sum()}**")

    # Detailed per-column analysis
    st.subheader("🔎 Phân tích theo Cột")
    for col in df.columns:
        st.markdown(f"### 📌 `{col}`")
        col_data = df[col]
        st.write(f"- Kiểu dữ liệu: `{col_data.dtype}`")
        st.write(f"- Giá trị thiếu: `{col_data.isnull().sum()}` ({col_data.isnull().mean():.2%})")

        if pd.api.types.is_numeric_dtype(col_data):
            desc = col_data.describe()
            st.dataframe(desc.to_frame())
            try:
                fig, ax = plt.subplots()
                sns.histplot(col_data.dropna(), kde=True, ax=ax)
                ax.set_title(f"Phân phối của {col}")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Không thể vẽ biểu đồ: {e}")
        elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == object:
            value_counts = col_data.value_counts().head(10)
            st.dataframe(value_counts.to_frame(name='Số lượng'))
            try:
                fig, ax = plt.subplots()
                sns.countplot(y=col_data, order=value_counts.index, ax=ax)
                ax.set_title(f"Giá trị hàng đầu trong {col}")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Không thể vẽ biểu đồ: {e}")

# --- 🔍 Univariate Analysis ---
with tabs[2]:
    for block in eda_sections['univariate']:
        st.markdown(block['insight'])
        st.code(block['code'], language='python')
        try:
            local_env = {"df": df, "plt": plt, "sns": sns}
            exec(block['code'], local_env)
            st.pyplot(plt.gcf())
            plt.clf()
            if 'insight_after_chart' in block:
                st.info(block['insight_after_chart'])
        except Exception as e:
            st.error(f"Lỗi khi hiển thị biểu đồ: {e}")

# --- 📊 Correlation ---
with tabs[3]:
    st.markdown(eda_sections['correlation']['insight'])
    st.code(eda_sections['correlation']['code'], language='python')
    try:
        local_env = {"df": df, "plt": plt, "sns": sns}
        exec(eda_sections['correlation']['code'], local_env)
        st.pyplot(plt.gcf())
        plt.clf()
        if 'insight_after_chart' in eda_sections['correlation']:
            st.info(eda_sections['correlation']['insight_after_chart'])
    except Exception as e:
        st.error(f"Lỗi khi hiển thị bản đồ nhiệt tương quan: {e}")

# --- 💡 Insights ---
with tabs[4]:
    st.subheader("🔖 Điểm chính & Khuyến nghị")
    prompt_summary = f"""
        Bạn là một nhà phân tích dữ liệu chuyên nghiệp. Với các tóm tắt sau từ quá trình EDA:

        1. Giới thiệu bộ dữ liệu:
        {eda_sections['introduction']}

        2. Vấn đề chất lượng dữ liệu:
        {eda_sections['data_quality']}

        3. Thông tin đơn biến:
        {[b['insight_after_chart'] for b in eda_sections['univariate'] if 'insight_after_chart' in b]}

        4. Thông tin tương quan:
        {eda_sections['correlation']['insight_after_chart']}

        Viết một đoạn tóm tắt gắn kết (~200-300 từ) mà:
        - Giải thích các mô hình hoặc vấn đề trong bộ dữ liệu.
        - Làm nổi bật các mối quan hệ quan trọng.
        - Đề cập đến những phát hiện bất ngờ.
        - Đề xuất những thông tin có thể hành động.

        Kết thúc bằng một danh sách ngắn gọn các khuyến nghị ở định dạng dấu đầu dòng.
        Trả lời bằng markdown.
        """
    
    try:
        summary_response = llm.invoke([HumanMessage(content=prompt_summary)]).content
    except Exception as e:
        st.error(f"Lỗi tạo summary: {e}")
        summary_response = """
## 📊 Tóm tắt Phân tích

Dựa trên phân tích EDA, bộ dữ liệu này cho thấy các đặc điểm sau:

### Điểm nổi bật:
- Kích thước dữ liệu phù hợp cho phân tích
- Chất lượng dữ liệu cần được cải thiện
- Có tiềm năng phát hiện insights quan trọng

### Khuyến nghị:
- Xử lý dữ liệu thiếu
- Kiểm tra outliers
- Thực hiện feature engineering
- Áp dụng các phương pháp phân tích nâng cao
"""
    
    st.markdown(summary_response)

# --- 📄 Full Report ---
with tabs[5]:
    st.markdown("## 📄 Tóm tắt Báo cáo Cuối cùng")

    # Render introduction + preview
    st.markdown("### 📘 Giới thiệu")
    st.markdown(eda_sections['introduction'])
    st.dataframe(df.head())

    # Data Quality
    st.markdown("### 🧼 Chất lượng Dữ liệu")
    st.markdown(eda_sections['data_quality'])

    # Univariate
    st.markdown("### 🔍 Phân tích Đơn biến")
    for block in eda_sections['univariate']:
        st.markdown(f"- {block['insight']}")
        st.code(block['code'], language="python")
        try:
            local_env = {"df": df, "plt": plt, "sns": sns}
            exec(block['code'], local_env)
            st.pyplot(plt.gcf())
            plt.clf()
            if 'insight_after_chart' in block:
                st.markdown(f"_{block['insight_after_chart']}_")
        except Exception as e:
            st.error(f"Lỗi: {e}")

    # Correlation
    st.markdown("### 📊 Thông tin Tương quan")
    st.markdown(eda_sections['correlation']['insight'])
    st.code(eda_sections['correlation']['code'], language="python")
    try:
        local_env = {"df": df, "plt": plt, "sns": sns}
        exec(eda_sections['correlation']['code'], local_env)
        st.pyplot(plt.gcf())
        plt.clf()
        st.markdown(f"_{eda_sections['correlation']['insight_after_chart']}_")
    except Exception as e:
        st.error(f"Lỗi bản đồ nhiệt: {e}")

    # Final Summary from tab 4
    st.markdown("### 💡 Thông tin Cuối cùng & Khuyến nghị")
    st.markdown(summary_response)

    # Export PDF
    try:
        pdf_bytes = export_eda_report_to_pdf(eda_sections, df, summary_response, dataset_name=name)
        st.download_button("📄 Tải xuống Báo cáo PDF", pdf_bytes, file_name=f"EDA_Report_{name}.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Lỗi xuất PDF: {e}")
        st.info("Vui lòng thử lại hoặc liên hệ hỗ trợ")