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

def clean_llm_json(raw_response):
    # Xoá markdown code block ```json hoặc ```
    cleaned = re.sub(r"^```(?:json)?", "", raw_response.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    return cleaned.strip()

def generate_eda_report_with_llm(df):
    prompt = f"""
Bạn là một nhà phân tích dữ liệu chuyên nghiệp. Với bộ dữ liệu `df` đã cho, thực hiện phân tích khám phá dữ liệu (EDA) chuyên sâu và trả về kết quả của bạn dưới dạng JSON. Phản hồi của bạn **phải** là JSON hợp lệ với các trường sau:

1. introduction: Giới thiệu markdown về kích thước bộ dữ liệu và các loại.
2. data_quality: Mô tả về giá trị thiếu, bản sao và chất lượng cột.
3. univariate: Một danh sách trong đó mỗi phần tử là một từ điển với các trường sau:
   - insight: Mô tả một câu về cột đại diện cho gì và điều gì làm cho nó thú vị.
   - code: Mã Python matplotlib hoặc seaborn để trực quan hóa cột. Sử dụng các loại biểu đồ thực hành tốt nhất:
       - Đối với cột số: sử dụng `sns.histplot(df['col'], bins=30, kde=True)`
       - Đối với cột phân loại: nếu `nunique <= 20`, sử dụng `sns.countplot`; ngược lại sử dụng barplot cho top 10 giá trị.
       - Bỏ qua các cột có hơn 100 giá trị duy nhất.
   - insight_after_chart: Giải thích markdown ngắn gọn về biểu đồ. Đề cập đến hình dạng phân phối (ví dụ: lệch phải, đối xứng), bất kỳ giá trị ngoại lệ nào, hoặc danh mục chiếm ưu thế. Ngắn gọn nhưng có ý nghĩa.
4. correlation: Một từ điển với các khóa sau:
- "insight": Một đoạn văn mô tả mục đích của phân tích tương quan, các biến nào dự kiến sẽ tương quan, và mối quan hệ nào thú vị nhất để khám phá.
- "code": Mã Python sử dụng seaborn hoặc matplotlib để tạo bản đồ nhiệt tương quan của tất cả các cột số trong bộ dữ liệu. Sử dụng `sns.heatmap(df.corr(), annot=True, cmap='coolwarm')` và thêm tiêu đề thông tin.
- "insight_after_chart": Giải thích chi tiết về bản đồ nhiệt. Bao gồm:
    - Đề cập đến các tương quan tích cực và tiêu cực mạnh nhất (với tên biến).
    - Liệu có tương quan nào bất ngờ hoặc trái ngược với trực giác không.
    - Bất kỳ biến nào dường như không liên quan đến các biến khác (tương quan thấp trên toàn bộ).
    - Kết luận ngắn về cách thông tin tương quan có thể hỗ trợ các nhiệm vụ xuôi dòng (ví dụ: dự đoán, lựa chọn tính năng).
5. insights: Danh sách các thông tin chi tiết
6. recommendations: Danh sách các khuyến nghị

Đảm bảo đầu ra của bạn chỉ là JSON và được thoát đúng cách.

Xem trước Metadata Bộ dữ liệu:
- Đầu:
{df.head().to_json(orient="records")}
- Giá trị thiếu:
{df.isnull().sum()[df.isnull().sum() > 0].to_dict()}
- Kiểu dữ liệu:
{df.dtypes.astype(str).to_dict()}
- Mô tả:
{df.describe().to_dict()}

Chỉ trả về JSON hợp lệ. Không bao gồm nó trong khối mã markdown (không có ba dấu gạch ngược).
"""

    response = llm.invoke([HumanMessage(content=prompt)]).content

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        cleaned = clean_llm_json(response)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            st.error("❌ Giải mã JSON thất bại. Đầu ra LLM thô được hiển thị bên dưới:")
            st.code(response)
            raise e

def generate_final_summary_prompt(sections):
    return textwrap.dedent("""
        Bạn là một nhà phân tích dữ liệu cấp cao được giao nhiệm vụ viết một báo cáo EDA cuối cùng chi tiết, chuyên nghiệp cho một bộ dữ liệu về hiệu suất học sinh.
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
        Tóm tắt kết luận của bạn về hành vi và hiệu suất của học sinh.
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

# Call LLM-generated EDA content
tabs = st.tabs(["📘 Giới thiệu", "🧼 Chất lượng Dữ liệu", "🔍 Đơn biến", "📊 Tương quan", "💡 Thông tin", "📄 Báo cáo Đầy đủ"])
eda_sections = generate_eda_report_with_llm(df)

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
    
    summary_response = llm.invoke([HumanMessage(content=prompt_summary)]).content
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

    # Export Markdown Report
    st.markdown("### 📤 Xuất Báo cáo")

    # Tạo phần Univariate Markdown trước để tránh lỗi f-string với \n
    univariate_md = ""
    for b in eda_sections['univariate']:
        univariate_md += f"- {b['insight']}\n\n```python\n{b['code']}\n```\n\n_{b.get('insight_after_chart', '')}_\n\n"

    # Gộp toàn bộ báo cáo (chỉ để xuất file, không hiển thị)
    full_report_md = f"""
## 📘 Giới thiệu
{eda_sections['introduction']}

## 🧼 Chất lượng Dữ liệu
{eda_sections['data_quality']}

## 🔍 Phân tích Đơn biến
{univariate_md}

## 📊 Thông tin Tương quan
{eda_sections['correlation']['insight']}

```python
{eda_sections['correlation']['code']}
```

_{eda_sections['correlation'].get('insight_after_chart', '')}_

## 💡 Thông tin Cuối cùng & Khuyến nghị
{summary_response}
"""

    # Nút tải xuống Markdown, không hiển thị nội dung
    # st.download_button(
    #     label="📥 Tải xuống Báo cáo Markdown",
    #     data=full_report_md,
    #     file_name=f"EDA_Report_{name}.md",
    #     mime="text/markdown"
    # )


    # Export PDF
    pdf_bytes = export_eda_report_to_pdf(eda_sections, df, summary_response, dataset_name=name)
    st.download_button("📄 Tải xuống Báo cáo PDF", pdf_bytes, file_name=f"EDA_Report_{name}.pdf", mime="application/pdf")



