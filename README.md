# VizGenie-GPT: Professional AI-Powered Data Analysis Platform

<div align="center">
  <!-- Language Toggle Buttons -->
  <div style="margin: 20px 0;">
    <button onclick="showLanguage('en')" id="btn-en" style="background: #667eea; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 5px; cursor: pointer; font-weight: bold;">English</button>
    <button onclick="showLanguage('vi')" id="btn-vi" style="background: #e0e0e0; color: #333; border: none; padding: 10px 20px; margin: 5px; border-radius: 5px; cursor: pointer; font-weight: bold;">Tiếng Việt</button>
  </div>
  
  ![VizGenie logo](assets/img/logo.png)
</div>

<script>
function showLanguage(lang) {
  // Hide all language sections
  var sections = document.querySelectorAll('[class*="lang-"]');
  sections.forEach(function(section) {
    section.style.display = 'none';
  });
  
  // Show selected language sections
  var selectedSections = document.querySelectorAll('.lang-' + lang);
  selectedSections.forEach(function(section) {
    section.style.display = 'block';
  });
  
  // Update button styles
  document.getElementById('btn-en').style.background = lang === 'en' ? '#667eea' : '#e0e0e0';
  document.getElementById('btn-en').style.color = lang === 'en' ? 'white' : '#333';
  document.getElementById('btn-vi').style.background = lang === 'vi' ? '#667eea' : '#e0e0e0';
  document.getElementById('btn-vi').style.color = lang === 'vi' ? 'white' : '#333';
}

// Show English by default
document.addEventListener('DOMContentLoaded', function() {
  showLanguage('en');
});
</script>

---

<!-- ENGLISH VERSION -->
<div class="lang-en">

## 🌟 Overview  

**VizGenie-GPT** – A comprehensive Streamlit-based data analysis platform that combines exploratory data analysis (EDA), AI-powered insights, and interactive visualizations. Upload your datasets, explore with intelligent charts, generate professional reports, and ask AI questions about your data—all in one powerful interface.

Data analysis can be complex and time-consuming: loading datasets, creating visualizations, tracking analysis history, and generating insights. **VizGenie-GPT** streamlines your entire data science workflow into one professional platform.

### Key Capabilities

1. **🔄 Multi-Dataset Management** - Upload, manage, and analyze multiple CSV datasets
2. **🤖 AI-Powered Analysis** - Ask questions in natural language and get intelligent insights
3. **📊 Smart Visualizations** - Auto-generated charts with professional styling and AI recommendations
4. **📈 Cross-Dataset Analysis** - Discover relationships and patterns across multiple data sources
5. **📋 Automated EDA Reports** - Generate comprehensive analysis reports with one click
6. **💬 Interactive Chat Sessions** - Persistent chat history with context-aware AI responses
7. **🎨 Professional UI** - Modern, responsive interface with dark/light mode support

---

## 🏗️ Architecture & Technology Stack

### Core Technologies

**🔧 Backend & AI**
- **OpenAI GPT-4o** - Advanced language model for data analysis and insights
- **LangChain** - Framework for LLM applications and agent creation
- **SQLite** - Lightweight database for session management and history

**📊 Data & Visualization**
- **Pandas & NumPy** - Data manipulation and numerical computation
- **Matplotlib & Seaborn** - Statistical visualization libraries
- **Plotly** - Interactive charts and advanced visualizations
- **PyGWalker** - Tableau-style visual analysis tool

**🌐 Frontend & UI**
- **Streamlit** - Multi-page web application framework
- **Professional CSS** - Custom styling with animations and responsive design
- **Component Architecture** - Modular UI components for consistency

### Application Structure

```
VizGenie/
├── main.py                 # Main chat interface with AI agent
├── pages/
│   ├── 1_🧮_Bang_Dieu_Khien.py     # Dashboard & dataset management
│   ├── 3_📂_Chi_Tiet_Bo_Du_Lieu.py  # Detailed dataset analysis
│   ├── 4_📊_Lich_Su_Bieu_Do.py      # Chart history & management
│   ├── 5_📋_Bao_Cao_EDA.py          # Automated EDA reports
│   ├── 6_📈_Bieu_Do_Thong_Minh.py   # Smart chart generation
│   ├── 7_🔗_Phan_Tich_Cheo_Du_Lieu.py # Cross-dataset analysis
│   └── 📖_Ve_Du_An.py               # Project information
├── src/
│   ├── models/
│   │   ├── llms.py          # LLM configuration and agents
│   │   └── config.py        # Application configuration
│   ├── components/
│   │   └── ui_components.py # Reusable UI components
│   ├── utils.py             # Core utilities and database functions
│   └── chart_enhancements.py # Advanced chart styling
└── assets/                  # Images and static resources
```

---

## 🎯 Features & Pages

### 1. 💬 Main Chat Interface (`main.py`)
**AI-Powered Data Conversation**
- Natural language queries about your data
- Context-aware responses with visualizations
- Session management with persistent history
- Real-time chart generation and insights
- Smart prompt enhancement with data context

### 2. 🧮 Dashboard (`1_🧮_Bang_Dieu_Khien.py`)
**Multi-Dataset Management Hub**
- Upload and manage multiple CSV files
- Dataset overview with quality metrics
- Quick action buttons for analysis
- Cross-dataset comparison tools
- File management and organization

### 3. 📂 Dataset Details (`3_📂_Chi_Tiet_Bo_Du_Lieu.py`)
**Deep Dataset Analysis**
- Comprehensive data profiling
- Missing data analysis and visualization
- Statistical summaries and distributions
- Data quality assessment
- Column-level insights and recommendations

### 4. 📊 Chart History (`4_📊_Lich_Su_Bieu_Do.py`)
**Visualization Management**
- View all generated charts by dataset
- Interactive chart reproduction
- Code viewing and export
- Chart organization and deletion
- Analysis timeline tracking

### 5. 📋 EDA Reports (`5_📋_Bao_Cao_EDA.py`)
**Automated Analysis Reports**
- One-click comprehensive EDA generation
- AI-powered insights and recommendations
- Professional PDF export
- Statistical analysis summaries
- Business intelligence insights

### 6. 📈 Smart Charts (`6_📈_Bieu_Do_Thong_Minh.py`)
**AI-Driven Visualization Creation**
- Intelligent chart type recommendations
- Professional color schemes and styling
- Interactive chart customization
- Advanced visualization options
- Export and sharing capabilities

### 7. 🔗 Cross-Dataset Analysis (`7_🔗_Phan_Tich_Cheo_Du_Lieu.py`)
**Multi-Dataset Intelligence**
- Relationship discovery across datasets
- Correlation analysis between sources
- Pattern identification and insights
- Semantic analysis and recommendations
- Combined dataset visualizations

### 8. 📖 About Project (`📖_Ve_Du_An.py`)
**Project Information & Documentation**
- Technology stack overview
- Feature explanations
- Usage guides and tips
- Development team information

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.8+**
- **Git**
- **OpenAI API Key**

### 1. Clone Repository
```bash
git clone https://github.com/nguyentuongbachhy/VizGenie.git
cd VizGenie
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**Important**: Ensure `.env` is in your `.gitignore` to keep your API key secure.

### 4. Initialize Database
The application will automatically create the SQLite database on first run.

### 5. Launch Application
```bash
streamlit run main.py
```

Navigate to `http://localhost:8501` to start using VizGenie-GPT!

---

## 💡 Key Features Deep Dive

### 🤖 AI-Powered Analysis
- **GPT-4o Integration**: Latest OpenAI model for superior understanding
- **Context-Aware Responses**: AI remembers your conversation history
- **Smart Recommendations**: Proactive suggestions based on your data
- **Natural Language Queries**: Ask complex questions in plain English

### 📊 Professional Visualizations
- **Multiple Chart Types**: Scatter plots, heatmaps, time series, box plots, bar charts
- **Smart Color Schemes**: Professional palettes with accessibility considerations
- **Interactive Elements**: Zoom, hover, and exploration capabilities
- **Export Options**: High-quality image and code export

### 🔄 Session Management
- **Persistent History**: All conversations and charts are saved
- **Session Organization**: Name and manage analysis sessions
- **Context Preservation**: AI maintains context across sessions
- **Collaborative Features**: Share insights and findings

### 📈 Advanced Analytics
- **Statistical Analysis**: Automated correlation, distribution, and outlier detection
- **Quality Assessment**: Data completeness and reliability scoring
- **Pattern Recognition**: Hidden relationship discovery
- **Business Intelligence**: Actionable insights and recommendations

---

## 🎨 User Interface

### Modern Design Principles
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Professional Styling**: Corporate-grade visual design
- **Intuitive Navigation**: Clear page structure and navigation
- **Accessibility**: High contrast and screen reader support

### Interactive Components
- **Smart Cards**: Information display with hover effects
- **Progress Indicators**: Visual feedback for long operations
- **Modal Dialogs**: Contextual actions and confirmations
- **Real-time Updates**: Live data refreshing and notifications

---

## 📊 Data Sources & Formats

### Supported File Types
- **CSV Files**: Primary data format with flexible encoding support
- **Excel Integration**: Coming soon
- **Database Connections**: Planned feature

### Data Requirements
- **Minimum Size**: No restrictions
- **Maximum Size**: Recommended under 100MB for optimal performance
- **Encoding**: UTF-8, ISO-8859-1, CP1252 auto-detection
- **Structure**: Any tabular data with headers

---

## 🔧 Configuration & Customization

### LLM Settings
```python
# In src/models/llms.py
def load_llm(model_name):
    if model_name == "gpt-4o":
        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=1500,
        )
```

### UI Customization
- Modify `src/components/ui_components.py` for styling changes
- Update color schemes in `src/chart_enhancements.py`
- Configure layouts in individual page files

---

## 🤝 Contributing

We welcome contributions to VizGenie-GPT! Please see our contributing guidelines for:
- Code style and standards
- Pull request process
- Issue reporting
- Feature requests

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Support & Community

- **Documentation**: Comprehensive guides and tutorials
- **Issues**: Bug reports and feature requests via GitHub
- **Community**: Join our discussions and share your insights
- **Updates**: Follow our development progress

---

## 🎯 Roadmap

### Upcoming Features
- **Real-time Collaboration**: Multi-user analysis sessions
- **Advanced ML Integration**: Automated model building and evaluation
- **API Endpoints**: Programmatic access to analysis functions
- **Cloud Integration**: Direct connection to cloud data sources
- **Custom Dashboards**: Personalized analytics dashboards

</div>

<!-- VIETNAMESE VERSION -->
<div class="lang-vi" style="display:none;">

## 🌟 Tổng Quan  

**VizGenie-GPT** – Nền tảng phân tích dữ liệu toàn diện dựa trên Streamlit, kết hợp khám phá dữ liệu (EDA), thông tin hỗ trợ AI và trực quan hóa tương tác. Tải lên datasets, khám phá với biểu đồ thông minh, tạo báo cáo chuyên nghiệp và đặt câu hỏi AI về dữ liệu của bạn—tất cả trong một giao diện mạnh mẽ.

Phân tích dữ liệu có thể phức tạp và tốn thời gian: tải datasets, tạo trực quan hóa, theo dõi lịch sử phân tích và tạo insights. **VizGenie-GPT** đơn giản hóa toàn bộ quy trình khoa học dữ liệu của bạn thành một nền tảng chuyên nghiệp.

### Khả Năng Chính

1. **🔄 Quản Lý Đa Bộ Dữ Liệu** - Tải lên, quản lý và phân tích nhiều datasets CSV
2. **🤖 Phân Tích Hỗ Trợ AI** - Đặt câu hỏi bằng ngôn ngữ tự nhiên và nhận insights thông minh
3. **📊 Trực Quan Hóa Thông Minh** - Biểu đồ tự động với styling chuyên nghiệp và đề xuất AI
4. **📈 Phân Tích Chéo Bộ Dữ Liệu** - Khám phá mối quan hệ và mô hình qua nhiều nguồn dữ liệu
5. **📋 Báo Cáo EDA Tự Động** - Tạo báo cáo phân tích toàn diện chỉ với một click
6. **💬 Phiên Chat Tương Tác** - Lịch sử chat bền vững với phản hồi AI có ngữ cảnh
7. **🎨 Giao Diện Chuyên Nghiệp** - Interface hiện đại, responsive với hỗ trợ chế độ tối/sáng

---

## 🏗️ Kiến Trúc & Công Nghệ

### Công Nghệ Cốt Lõi

**🔧 Backend & AI**
- **OpenAI GPT-4o** - Mô hình ngôn ngữ tiên tiến cho phân tích dữ liệu và insights
- **LangChain** - Framework cho ứng dụng LLM và tạo agent
- **SQLite** - Cơ sở dữ liệu nhẹ cho quản lý phiên và lịch sử

**📊 Dữ Liệu & Trực Quan Hóa**
- **Pandas & NumPy** - Thao tác dữ liệu và tính toán số học
- **Matplotlib & Seaborn** - Thư viện trực quan hóa thống kê
- **Plotly** - Biểu đồ tương tác và trực quan hóa nâng cao
- **PyGWalker** - Công cụ phân tích trực quan kiểu Tableau

**🌐 Frontend & UI**
- **Streamlit** - Framework ứng dụng web đa trang
- **Professional CSS** - Styling tùy chỉnh với animations và thiết kế responsive
- **Component Architecture** - Các thành phần UI modular để đảm bảo tính nhất quán

### Cấu Trúc Ứng Dụng

```
VizGenie/
├── main.py                 # Giao diện chat chính với AI agent
├── pages/
│   ├── 1_🧮_Bang_Dieu_Khien.py     # Dashboard & quản lý dataset
│   ├── 3_📂_Chi_Tiet_Bo_Du_Lieu.py  # Phân tích dataset chi tiết
│   ├── 4_📊_Lich_Su_Bieu_Do.py      # Lịch sử & quản lý biểu đồ
│   ├── 5_📋_Bao_Cao_EDA.py          # Báo cáo EDA tự động
│   ├── 6_📈_Bieu_Do_Thong_Minh.py   # Tạo biểu đồ thông minh
│   ├── 7_🔗_Phan_Tich_Cheo_Du_Lieu.py # Phân tích chéo dataset
│   └── 📖_Ve_Du_An.py               # Thông tin dự án
├── src/
│   ├── models/
│   │   ├── llms.py          # Cấu hình LLM và agents
│   │   └── config.py        # Cấu hình ứng dụng
│   ├── components/
│   │   └── ui_components.py # Thành phần UI tái sử dụng
│   ├── utils.py             # Tiện ích cốt lõi và functions cơ sở dữ liệu
│   └── chart_enhancements.py # Styling biểu đồ nâng cao
└── assets/                  # Hình ảnh và tài nguyên tĩnh
```

---

## 🎯 Tính Năng & Trang

### 1. 💬 Giao Diện Chat Chính (`main.py`)
**Cuộc Trò Chuyện Dữ Liệu Hỗ Trợ AI**
- Truy vấn ngôn ngữ tự nhiên về dữ liệu của bạn
- Phản hồi có ngữ cảnh với trực quan hóa
- Quản lý phiên với lịch sử bền vững
- Tạo biểu đồ và insights thời gian thực
- Cải tiến prompt thông minh với ngữ cảnh dữ liệu

### 2. 🧮 Dashboard (`1_🧮_Bang_Dieu_Khien.py`)
**Trung Tâm Quản Lý Đa Bộ Dữ Liệu**
- Tải lên và quản lý nhiều file CSV
- Tổng quan dataset với metrics chất lượng
- Nút hành động nhanh để phân tích
- Công cụ so sánh chéo dataset
- Quản lý và tổ chức file

### 3. 📂 Chi Tiết Dataset (`3_📂_Chi_Tiet_Bo_Du_Lieu.py`)
**Phân Tích Dataset Sâu**
- Profiling dữ liệu toàn diện
- Phân tích và trực quan hóa dữ liệu thiếu
- Tóm tắt thống kê và phân phối
- Đánh giá chất lượng dữ liệu
- Insights và khuyến nghị cấp cột

### 4. 📊 Lịch Sử Biểu Đồ (`4_📊_Lich_Su_Bieu_Do.py`)
**Quản Lý Trực Quan Hóa**
- Xem tất cả biểu đồ được tạo theo dataset
- Tái tạo biểu đồ tương tác
- Xem và xuất code
- Tổ chức và xóa biểu đồ
- Theo dõi timeline phân tích

### 5. 📋 Báo Cáo EDA (`5_📋_Bao_Cao_EDA.py`)
**Báo Cáo Phân Tích Tự Động**
- Tạo EDA toàn diện với một click
- Insights và khuyến nghị hỗ trợ AI
- Xuất PDF chuyên nghiệp
- Tóm tắt phân tích thống kê
- Insights business intelligence

### 6. 📈 Biểu Đồ Thông Minh (`6_📈_Bieu_Do_Thong_Minh.py`)
**Tạo Trực Quan Hóa Hướng Dẫn AI**
- Khuyến nghị loại biểu đồ thông minh
- Bảng màu và styling chuyên nghiệp
- Tùy chỉnh biểu đồ tương tác
- Tùy chọn trực quan hóa nâng cao
- Khả năng xuất và chia sẻ

### 7. 🔗 Phân Tích Chéo Dataset (`7_🔗_Phan_Tich_Cheo_Du_Lieu.py`)
**Trí Tuệ Đa Bộ Dữ Liệu**
- Khám phá mối quan hệ qua các dataset
- Phân tích tương quan giữa các nguồn
- Nhận dạng mô hình và insights
- Phân tích và khuyến nghị ngữ nghĩa
- Trực quan hóa dataset kết hợp

### 8. 📖 Về Dự Án (`📖_Ve_Du_An.py`)
**Thông Tin Dự Án & Tài Liệu**
- Tổng quan ngăn xếp công nghệ
- Giải thích tính năng
- Hướng dẫn sử dụng và mẹo
- Thông tin nhóm phát triển

---

## 🚀 Cài Đặt & Thiết Lập

### Yêu Cầu Tiên Quyết
- **Python 3.8+**
- **Git**
- **OpenAI API Key**

### 1. Clone Repository
```bash
git clone https://github.com/nguyentuongbachhy/VizGenie.git
cd VizGenie
```

### 2. Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```

### 3. Cấu Hình Môi Trường
Tạo file `.env` trong thư mục gốc dự án:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**Quan trọng**: Đảm bảo `.env` nằm trong `.gitignore` để giữ an toàn API key.

### 4. Khởi Tạo Cơ Sở Dữ Liệu
Ứng dụng sẽ tự động tạo cơ sở dữ liệu SQLite ở lần chạy đầu tiên.

### 5. Khởi Chạy Ứng Dụng
```bash
streamlit run main.py
```

Điều hướng đến `http://localhost:8501` để bắt đầu sử dụng VizGenie-GPT!

---

## 💡 Tính Năng Chính Chi Tiết

### 🤖 Phân Tích Hỗ Trợ AI
- **Tích Hợp GPT-4o**: Mô hình OpenAI mới nhất để hiểu biết vượt trội
- **Phản Hồi Có Ngữ Cảnh**: AI nhớ lịch sử cuộc trò chuyện của bạn
- **Khuyến Nghị Thông Minh**: Đề xuất chủ động dựa trên dữ liệu của bạn
- **Truy Vấn Ngôn Ngữ Tự Nhiên**: Đặt câu hỏi phức tạp bằng tiếng Việt đơn giản

### 📊 Trực Quan Hóa Chuyên Nghiệp
- **Nhiều Loại Biểu Đồ**: Scatter plots, heatmaps, time series, box plots, bar charts
- **Bảng Màu Thông Minh**: Palettes chuyên nghiệp với cân nhắc về accessibility
- **Phần Tử Tương Tác**: Khả năng zoom, hover và khám phá
- **Tùy Chọn Xuất**: Xuất hình ảnh chất lượng cao và code

### 🔄 Quản Lý Phiên
- **Lịch Sử Bền Vững**: Tất cả cuộc trò chuyện và biểu đồ được lưu
- **Tổ Chức Phiên**: Đặt tên và quản lý phiên phân tích
- **Bảo Tồn Ngữ Cảnh**: AI duy trì ngữ cảnh qua các phiên
- **Tính Năng Cộng Tác**: Chia sẻ insights và phát hiện

### 📈 Phân Tích Nâng Cao
- **Phân Tích Thống Kê**: Tự động phát hiện tương quan, phân phối và outlier
- **Đánh Giá Chất Lượng**: Chấm điểm độ hoàn thiện và độ tin cậy dữ liệu
- **Nhận Dạng Mô Hình**: Khám phá mối quan hệ ẩn
- **Business Intelligence**: Insights và khuyến nghị có thể hành động

---

## 🎨 Giao Diện Người Dùng

### Nguyên Tắc Thiết Kế Hiện Đại
- **Layout Responsive**: Hoạt động trên desktop, tablet và mobile
- **Styling Chuyên Nghiệp**: Thiết kế trực quan cấp doanh nghiệp
- **Điều Hướng Trực Quan**: Cấu trúc trang và điều hướng rõ ràng
- **Accessibility**: Độ tương phản cao và hỗ trợ screen reader

### Thành Phần Tương Tác
- **Smart Cards**: Hiển thị thông tin với hiệu ứng hover
- **Chỉ Báo Tiến Trình**: Phản hồi trực quan cho các thao tác dài
- **Modal Dialogs**: Hành động và xác nhận theo ngữ cảnh
- **Cập Nhật Thời Gian Thực**: Làm mới dữ liệu trực tiếp và thông báo

---

## 📊 Nguồn Dữ Liệu & Định Dạng

### Loại File Được Hỗ Trợ
- **File CSV**: Định dạng dữ liệu chính với hỗ trợ encoding linh hoạt
- **Tích Hợp Excel**: Sắp ra mắt
- **Kết Nối Cơ Sở Dữ Liệu**: Tính năng được lên kế hoạch

### Yêu Cầu Dữ Liệu
- **Kích Thước Tối Thiểu**: Không hạn chế
- **Kích Thước Tối Đa**: Khuyến nghị dưới 100MB để hiệu suất tối ưu
- **Encoding**: UTF-8, ISO-8859-1, CP1252 tự động phát hiện
- **Cấu Trúc**: Bất kỳ dữ liệu dạng bảng nào có headers

---

## 🔧 Cấu Hình & Tùy Chỉnh

### Cài Đặt LLM
```python
# Trong src/models/llms.py
def load_llm(model_name):
    if model_name == "gpt-4o":
        return ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=1500,
        )
```

### Tùy Chỉnh UI
- Sửa đổi `src/components/ui_components.py` để thay đổi styling
- Cập nhật bảng màu trong `src/chart_enhancements.py`
- Cấu hình layouts trong các file trang riêng lẻ

---

## 🤝 Đóng Góp

Chúng tôi hoan nghênh đóng góp cho VizGenie-GPT! Vui lòng xem hướng dẫn đóng góp của chúng tôi để biết:
- Chuẩn và tiêu chuẩn code
- Quy trình pull request
- Báo cáo vấn đề
- Yêu cầu tính năng

---

## 🎯 Lộ Trình

### Tính Năng Sắp Tới
- **Cộng Tác Thời Gian Thực**: Phiên phân tích nhiều người dùng
- **Tích Hợp ML Nâng Cao**: Xây dựng và đánh giá mô hình tự động
- **API Endpoints**: Truy cập lập trình vào functions phân tích
- **Tích Hợp Cloud**: Kết nối trực tiếp với nguồn dữ liệu cloud
- **Dashboard Tùy Chỉnh**: Dashboard analytics cá nhân hóa

</div>

---

<div align="center">

**🧠 VizGenie-GPT Professional**  
*Making Advanced Data Analysis Accessible to Everyone*  
*Làm Cho Phân Tích Dữ Liệu Nâng Cao Trở Nên Dễ Tiếp Cận Với Mọi Người*

Created with ❤️ by the Delay Group  
Được tạo với ❤️ bởi Delay Group

[🌟 Star us on GitHub](https://github.com/nguyentuongbachhy/VizGenie) | [📧 Contact Us](mailto:dangquach.dev@gmail.com) | [🐛 Report Issues](https://github.com/nguyentuongbachhy/VizGenie/issues)

</div>
