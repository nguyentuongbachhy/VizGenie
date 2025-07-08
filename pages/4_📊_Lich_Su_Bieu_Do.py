# File: pages/4_📊_Visual_Summary.py
import streamlit as st
from src.utils import init_db, get_all_datasets, get_chart_cards_by_dataset, get_dataset, safe_read_csv, execute_plt_code, delete_chart_card


st.set_page_config(page_title="Tóm tắt Trực quan", layout="wide")
st.title("📊 Lịch sử Biểu đồ")

# Initialize database
init_db()

# Get list of datasets
datasets = get_all_datasets()
if not datasets:
    st.warning("Vui lòng tải lên một số bộ dữ liệu trước.")
    st.stop()

# Select dataset from dropdown
options = {f"{d[0]} - {d[1]}": d[0] for d in datasets}
selected = st.selectbox("Chọn bộ dữ liệu:", list(options.keys()))
dataset_id = options[selected]
dataset = get_dataset(dataset_id)

# Load dataframe safely
try:
    st.session_state.df = safe_read_csv(dataset[2])
except Exception as e:
    st.error(f"❌ Không thể tải dataframe: {e}")
    st.stop()

# Get chart cards related to this dataset
cards = get_chart_cards_by_dataset(dataset_id)
if not cards:
    st.info("Không tìm thấy thông tin biểu đồ nào cho bộ dữ liệu này.")
    st.stop()

# Display visual summaries
st.markdown("---")
st.subheader("🧩 Thông tin dựa trên Biểu đồ")

# for i, (question, answer, code, timestamp) in enumerate(cards):
#     with st.container():
#         cols = st.columns([2, 3])

#         with cols[0]:
#             st.markdown(f"### 🕓 {timestamp}")
#             st.markdown(f"❓ **Question {i+1}:** {question}")
#             st.markdown(f"💬 **Answer:** {answer}")
#             with st.expander("📄 View Code"):
#                 st.code(code, language="python")

#         with cols[1]:
#             fig = execute_plt_code(code, st.session_state.df)
#             if fig:
#                 st.pyplot(fig)
#             else:
#                 st.warning("⚠️ No chart could be rendered from saved code.")


# Ghi nhớ biểu đồ đã xoá để ẩn ngay mà không cần reload
if "deleted_charts" not in st.session_state:
    st.session_state.deleted_charts = set()

for i, (question, answer, code, timestamp) in enumerate(cards):
    unique_key = f"{question}-{timestamp}"
    if unique_key in st.session_state.deleted_charts:
        continue  # Ẩn khỏi UI nếu đã xoá

    with st.container():
        cols = st.columns([2, 3])

        with cols[0]:
            st.markdown(f"### 🕓 {timestamp}")
            st.markdown(f"❓ **Câu hỏi {i+1}:** {question}")
            st.markdown(f"💬 **Trả lời:** {answer}")
            with st.expander("📄 Xem Code"):
                st.code(code, language="python")

            if st.button("🗑️ Xóa Biểu đồ", key=f"del_card_{i}"):
                delete_chart_card(dataset_id, question, timestamp)
                st.toast("🗑️ Đã xóa biểu đồ!", icon="✅")
                st.rerun()  # 

        with cols[1]:
            fig = execute_plt_code(code, st.session_state.df)
            if fig:
                st.pyplot(fig)
            else:
                st.warning("⚠️ Không thể hiển thị biểu đồ từ code đã lưu.")

    st.markdown("---")
