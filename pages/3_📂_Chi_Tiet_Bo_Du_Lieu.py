import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from src.utils import get_all_datasets, get_dataset
from src.models.llms import load_llm

st.set_page_config(page_title="📂 Chi Tiết Bộ Dữ Liệu", layout="wide")
st.title("📂 Chi Tiết Bộ Dữ Liệu")

llm = load_llm("gpt-3.5-turbo")

# ---------- Hàm hỗ trợ ----------
def safe_read_csv(file_path):
    for enc in ['utf-8', 'ISO-8859-1', 'utf-16', 'cp1252']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Không thể giải mã file với các encoding phổ biến.")

def analyze_column(col_name, series):
    info = {'name': col_name, 'dtype': str(series.dtype), 'missing_pct': series.isna().mean() * 100, 'unique': series.nunique()}
    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe()
        info.update({
            'min': desc['min'], 'max': desc['max'], 'mean': desc['mean'],
            'median': series.median(), 'std': desc['std'],
            'outliers': ((series < (desc['25%'] - 1.5*(desc['75%'] - desc['25%']))) | (series > (desc['75%'] + 1.5*(desc['75%'] - desc['25%'])))).sum(),
            'type': 'Numeric'
        })
    elif series.nunique() == 2:
        info['type'] = 'Boolean'
    elif info['unique'] == len(series):
        info['type'] = 'ID'
    elif info['unique'] <= 20:
        info['type'] = 'Category'
    else:
        info['type'] = 'Text'
    return info

def guess_column_semantic_llm(col_name):
    prompt = f"Loại ngữ nghĩa hoặc ý nghĩa của cột có tên '{col_name}' trong bộ dữ liệu là gì? Trả lời bằng 3-5 từ tiếng Việt."
    return llm.invoke(prompt)

@st.cache_data(show_spinner=False)
def get_cleaning_suggestions(col_stats):
    cols_description = "\n".join([
        f"Cột: {col['name']} | Loại: {col['dtype']} | Thiếu: {col['missing_pct']:.2f}%" for col in col_stats
    ])
    prompt = f"""
Dựa trên tóm tắt sau về các cột trong bộ dữ liệu:
{cols_description}

Hãy đề xuất kế hoạch làm sạch với các quy tắc sau:
- Chỉ xóa các cột nếu tỷ lệ thiếu > 50%.
- Đối với các cột có giá trị thiếu ≤ 50%:
    - Nếu là số: điền bằng trung vị.
    - Nếu là phân loại: điền bằng mode.
- Chỉ loại bỏ ngoại lệ từ các cột có dữ liệu số không có giá trị thiếu.
- Chuẩn hóa các cột số chỉ khi chúng không được điền hoặc loại bỏ ngoại lệ, và giá trị tối đa lớn hơn nhiều so với 1.
- Không áp dụng quá hai bước làm sạch trên cùng một cột.
- Nhóm các cột một cách logic và giải thích ngắn gọn trong nhận xét.

Trả về kế hoạch dưới dạng danh sách dấu đầu dòng rõ ràng.
"""
    return llm.invoke(prompt)

@st.cache_data(show_spinner=False)
def refine_cleaning_strategy(user_input, base_plan):
    prompt = f"""
Kế hoạch làm sạch hiện tại:
{base_plan}

Người dùng muốn: {user_input}

Cập nhật kế hoạch làm sạch phù hợp.
"""
    return llm.invoke(prompt)

@st.cache_data(show_spinner=False)
def generate_cleaning_code_from_plan(plan):
    prompt = f"""
Chuyển đổi kế hoạch làm sạch sau thành mã Python hợp lệ sử dụng pandas.
Chỉ trả về mã Python có thể thực thi trực tiếp trong Python.
Giả định dataframe được đặt tên là `df`.

Trước khi áp dụng các phương thức `.str` (ví dụ `.str.replace`), luôn kiểm tra dtype của cột như thế này:
if df["tên_cột"].dtype == "object":
    df["tên_cột"] = df["tên_cột"].str.replace(",", "").astype(float)

Cũng đảm bảo rằng bất kỳ chuỗi nào như '1,000' hoặc '2,500.50' được chuyển đổi thành giá trị số trước khi làm sạch thêm.

Kế hoạch Làm sạch:
{plan}
"""
    return llm.invoke(prompt)

def extract_valid_code(llm_response):
    match = re.search(r"```(?:python)?\n(.*?)```", llm_response.strip(), re.DOTALL)
    if match:
        return match.group(1)
    lines = llm_response.splitlines()
    code_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
    return "\n".join(code_lines)

def generate_insight(info):
    if info['type'] == 'ID':
        return "🔹 Đây là cột định danh duy nhất."
    if info['missing_pct'] > 0:
        return f"⚠️ {info['missing_pct']:.1f}% giá trị thiếu."
    if 'std' in info and info['std'] < 1e-3:
        return "⚠️ Độ biến thiên rất thấp."
    if info['unique'] < 5 and info['type'] == 'Category':
        return "ℹ️ Phân loại với <5 giá trị riêng biệt."
    return "✅ Không phát hiện vấn đề lớn."

def plot_distribution(col_name, series):
    fig, ax = plt.subplots()
    if pd.api.types.is_numeric_dtype(series):
        ax.hist(series.dropna(), bins=20, color='#69b3a2')
        ax.set_xlabel(col_name)
        ax.set_ylabel('Tần suất')
    else:
        vc = series.fillna("NaN").value_counts().head(20)
        ax.bar(vc.index.astype(str), vc.values, color='#8c54ff')
        ax.set_xticks(range(len(vc.index)))  # Sửa warning
        ax.set_xticklabels(vc.index, rotation=45, ha='right')
        ax.set_ylabel('Số lượng')
    ax.set_title(f"Phân phối: {col_name}")
    st.pyplot(fig)

def fix_numeric_strings(df):
    for col in df.select_dtypes(include='object').columns:
        if df[col].dropna().apply(lambda x: isinstance(x, str)).all():
            try:
                df[col] = df[col].str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"Không thể làm sạch cột {col}: {e}")
    return df

def show_skew_kurtosis(df, cleaned_df):
    raw_cols = df.select_dtypes(include='number').columns
    clean_cols = cleaned_df.select_dtypes(include='number').columns
    numeric_cols = list(set(raw_cols).intersection(set(clean_cols)))

    if not numeric_cols:
        st.info("Không có cột số chung nào khả dụng cho báo cáo độ lệch/độ nhọn.")
        return

    report = pd.DataFrame(index=numeric_cols)
    report['Độ lệch (Trước)'] = df[numeric_cols].skew()
    report['Độ nhọn (Trước)'] = df[numeric_cols].kurtosis()
    report['Độ lệch (Sau)'] = cleaned_df[numeric_cols].skew()
    report['Độ nhọn (Sau)'] = cleaned_df[numeric_cols].kurtosis()
    st.dataframe(report.round(2), use_container_width=True)

    st.markdown("### 📊 Trực quan hóa")

    fig1, ax1 = plt.subplots()
    report[['Độ lệch (Trước)', 'Độ lệch (Sau)']].plot(kind='bar', ax=ax1)
    ax1.set_title('Độ lệch Trước vs Sau Làm sạch')
    ax1.set_ylabel('Độ lệch')
    ax1.set_xlabel('Đặc trưng')
    ax1.set_xticks(range(len(numeric_cols)))
    ax1.set_xticklabels(numeric_cols, rotation=45, ha='right')
    ax1.legend()
    st.pyplot(fig1)

    try:
        insight1 = llm.invoke(f"""
Hãy giải thích biểu đồ thanh độ lệch này so sánh trước vs sau làm sạch:
{report[['Độ lệch (Trước)', 'Độ lệch (Sau)']].to_markdown()}
""")
        st.markdown("#### 🤖 Nhận xét về Độ lệch")
        st.info(insight1)
    except Exception:
        st.warning("Không thể giải thích biểu đồ độ lệch qua LLM.")

    fig2, ax2 = plt.subplots()
    report[['Độ nhọn (Trước)', 'Độ nhọn (Sau)']].plot(kind='bar', ax=ax2)
    ax2.set_title('Độ nhọn Trước vs Sau Làm sạch')
    ax2.set_ylabel('Độ nhọn')
    ax2.set_xlabel('Đặc trưng')
    ax2.set_xticks(range(len(numeric_cols)))
    ax2.set_xticklabels(numeric_cols, rotation=45, ha='right')
    ax2.legend()
    st.pyplot(fig2)

    try:
        insight2 = llm.invoke(f"""
Hãy giải thích biểu đồ thanh độ nhọn này so sánh trước vs sau làm sạch:
{report[['Độ nhọn (Trước)', 'Độ nhọn (Sau)']].to_markdown()}
""")
        st.markdown("#### 🤖 Nhận xét về Độ nhọn")
        st.info(insight2)
    except Exception:
        st.warning("Không thể giải thích biểu đồ độ nhọn qua LLM.")

    try:
        interpretation = llm.invoke(f"""
Hãy phân tích những điều sau:
1. Bảng độ lệch và độ nhọn dưới đây.
2. Các biểu đồ thanh so sánh trước vs sau làm sạch.

Sau đó cung cấp:
- Giải thích về cách làm sạch ảnh hưởng đến tính đối xứng phân phối và hành vi của đuôi.
- Đánh giá liệu dữ liệu đã làm sạch có phù hợp hơn cho phân tích thống kê hay không.
- Đề xuất các bước tiếp theo nếu vẫn cần cải thiện.

Tóm tắt dữ liệu:
{report.to_markdown()}
""")
        st.markdown("### 📘 Giải thích bởi LLM")
        st.write(interpretation)
    except Exception:
        st.warning("Không thể giải thích báo cáo qua LLM.")




# Tải Bộ dữ liệu và Hiển thị Tabs
datasets = get_all_datasets()
if datasets:
    selected = st.selectbox("Chọn bộ dữ liệu:", [f"{d[0]} - {d[1]}" for d in datasets])
    dataset_id = int(selected.split(" - ")[0])
    dataset = get_dataset(dataset_id)
    df = safe_read_csv(dataset[2])
    st.markdown(f"### Bộ dữ liệu: `{dataset[1]}` — {df.shape[0]} hàng × {df.shape[1]} cột")

    tab1, tab2, tab3 = st.tabs(["📊 Tổng quan", "🧼 Làm sạch", "📈 Độ lệch & Độ nhọn"])

    with tab1:
        for col in df.columns:
            with st.container():
                stats = analyze_column(col, df[col])
                st.markdown(f"#### 📌 {col}")
                cols = st.columns([2, 3])
                with cols[0]:
                    st.markdown(f"**Loại:** `{stats['type']}`")
                    if 'min' in stats:
                        st.markdown(f"- Tối thiểu: `{stats['min']}`")
                        st.markdown(f"- Tối đa: `{stats['max']}`")
                        st.markdown(f"- Trung bình: `{stats['mean']:.2f}`")
                        st.markdown(f"- Trung vị: `{stats['median']}`")
                        st.markdown(f"- Độ lệch chuẩn: `{stats['std']:.2f}`")
                        st.markdown(f"- Ngoại lệ: `{stats['outliers']}`")
                    st.markdown(f"- Duy nhất: `{stats['unique']}`")
                    st.markdown(f"- Thiếu: `{stats['missing_pct']:.2f}%`")
                    st.info(generate_insight(stats))
                with cols[1]:
                    plot_distribution(col, df[col])
            st.markdown("---")

    with tab2:
        col_stats = [dict(analyze_column(col, df[col]), semantic=guess_column_semantic_llm(col)) for col in df.columns]
        summary_df = pd.DataFrame([{**c, 'Thiếu %': f"{c['missing_pct']:.2f}"} for c in col_stats])
        st.session_state.col_stats = col_stats
        st.session_state.summary_df = summary_df

        st.dataframe(summary_df[['name', 'dtype', 'semantic', 'type', 'unique', 'Thiếu %']])
        base_plan = get_cleaning_suggestions(col_stats)
        st.session_state.base_cleaning_plan = base_plan
        st.markdown("### 🧼 Kế hoạch Làm sạch")
        st.markdown(base_plan)

        if st.toggle("🛠 Tùy chỉnh Kế hoạch Làm sạch"):
            user_input = st.text_input("✍️ Sửa đổi kế hoạch làm sạch:")
            if user_input:
                st.session_state.base_cleaning_plan = refine_cleaning_strategy(user_input, base_plan)
                st.rerun()

        code_raw = generate_cleaning_code_from_plan(st.session_state.base_cleaning_plan)
        code_clean = extract_valid_code(code_raw)
        st.session_state.code_clean = code_clean
        with st.expander("🧪 Mã Làm sạch Thô (debug)"):
            st.code(code_raw, language="markdown")

        try:
            exec_globals = {'df': df.copy(), 'pd': pd, 'np': np, 'fix_numeric_strings': fix_numeric_strings}
            exec("df = fix_numeric_strings(df)\n" + code_clean, exec_globals)
            cleaned_df = exec_globals['df']

            # Chỉ khi không lỗi mới gán vào session_state
            st.session_state.cleaned_df = cleaned_df
            st.session_state.raw_df = df

            st.markdown("### ✅ Xem trước Dữ liệu Đã làm sạch")
            st.dataframe(cleaned_df.head())

        except Exception as e:
            st.error(f"Lỗi khi thực thi mã làm sạch: {e}")
            st.code(code_clean, language="python")

        if 'cleaned_df' in st.session_state:
            st.download_button(
                label="🧹 Làm sạch & Xuất",
                data=st.session_state.cleaned_df.to_csv(index=False).encode('utf-8'),
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

            with st.expander("🧾 Mã Python Đã sử dụng"):
                st.code(code_clean, language="python")




    with tab3:
        st.markdown("### 📈 Báo cáo Độ lệch & Độ nhọn")
        if "cleaned_df" in st.session_state and "raw_df" in st.session_state:
            show_skew_kurtosis(st.session_state.raw_df, st.session_state.cleaned_df)
        else:
            st.info("Vui lòng chạy làm sạch trong tab '🧼 Làm sạch' trước.")
else:
    st.warning("Không tìm thấy bộ dữ liệu nào. Vui lòng tải lên một bộ dữ liệu trong Bảng điều khiển.")
