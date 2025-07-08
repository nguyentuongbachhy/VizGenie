import streamlit as st
import os

st.set_page_config(page_title="📖 Về Dự Án", layout="wide")
st.title("📖 Về Dự Án Này")

st.markdown("""
Chào mừng đến với **VizGenie-GPT** — một nền tảng phân tích dữ liệu thông minh, được hỗ trợ bởi LLM, được xây dựng để EDA và trực quan hóa trực quan.


### 🚀 Tính Năng
- Tải lên bộ dữ liệu CSV và xem trước ngay lập tức  
- Tự động tạo báo cáo EDA đầy đủ sử dụng GPT-3.5 qua LangChain  
- Tạo biểu đồ động (Matplotlib, Seaborn, Plotly)  
- Lưu và tái sử dụng các biểu đồ có insights  
- Chatbot được hỗ trợ AI để đặt câu hỏi về dữ liệu  
- Xuất báo cáo PDF chuyên nghiệp với định dạng phong phú  

### 🧱 Quy Trình Dữ Liệu
""", unsafe_allow_html=True)

# Hiển thị ảnh pipeline
pipeline_path = os.path.join("assets", "img", "pipeline.png")
if os.path.exists(pipeline_path):
    st.image(pipeline_path, use_container_width=True)

else:
    st.warning(f"⚠️ Không thể tìm thấy ảnh pipeline tại {pipeline_path}")

st.markdown("""

### 🧰 Ngăn Xếp Công Nghệ
- **Frontend**: Streamlit  
- **AI Engine**: OpenAI GPT-3.5-turbo + LangChain  
- **Trực Quan Dữ Liệu**: Matplotlib, Seaborn, Plotly, PyGWalker  
- **Lưu Trữ**: SQLite  
- **Xuất**: pdfkit + wkhtmltopdf  


### 💡 Động Cơ
Dự án này được tạo ra để:

- Làm cho việc khám phá dữ liệu dễ dàng hơn cho những người không lập trình  
- Tích hợp lý luận LLM vào EDA  
- Tạo ra các báo cáo hấp dẫn về mặt thị giác với nỗ lực tối thiểu  

---

### 👨‍💻 Đóng Góp
Được tạo với ❤️ bởi nhóm Delay.
""", unsafe_allow_html=True)

with st.sidebar:
    st.page_link("pages/📖_Ve_Du_An.py", label="📖 Về Dự Án", icon="📘")
