import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from src.models.llms import create_agent_from_csv, load_llm
from src.utils import (
    add_chart_card, init_db, get_all_datasets, get_dataset, safe_read_csv,
    create_chat_session, get_sessions_by_dataset, add_chat_message,
    get_chat_messages, execute_plt_code, delete_chat_message,
    delete_chat_session, rename_chat_session
)

from src.components.ui_components import (
    render_professional_header, render_metric_cards, render_feature_card,
    render_insight_card, render_status_indicator, create_data_quality_indicator,
    render_interactive_data_explorer, render_animated_loading, PROFESSIONAL_CSS
)

# Import chart enhancement functions
from src.chart_enhancements import (
    smart_patch_chart_code, apply_chart_enhancements,
    enhance_prompt_with_chart_suggestions, ENHANCED_COLOR_SCHEMES
)

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure page with professional styling
st.set_page_config(
    page_title="VizGenie-GPT Chuyên nghiệp", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Apply professional CSS
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# Professional header with animation
render_professional_header(
    "VizGenie-GPT Phân tích Chuyên nghiệp",
    "Phân tích dữ liệu tiên tiến được hỗ trợ bởi AI với thông tin thông minh và trực quan hóa đẹp mắt",
    "🧠"
)

# Load environment and initialize database FIRST
load_dotenv()
init_db()

def generate_chart_code(rec, df):
    """Generate Python visualization code based on recommendation"""
    try:
        chart_type = rec.get('chart_type', 'scatter')
        
        if chart_type == 'heatmap':
            # Correlation heatmap
            numeric_cols = rec.get('columns', df.select_dtypes(include=['number']).columns.tolist())
            code = f"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Select numeric columns and calculate correlation
numeric_cols = {numeric_cols}
correlation_data = df[numeric_cols].corr()

# Create heatmap
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(correlation_data, dtype=bool))
sns.heatmap(correlation_data, 
           annot=True, 
           cmap='RdBu_r', 
           center=0,
           square=True,
           mask=mask,
           cbar_kws={{'shrink': 0.8}})
plt.title('Correlation Heatmap - Discover Data Relationships', fontsize=16, fontweight='bold')
plt.tight_layout()
"""
        
        elif chart_type == 'scatter':
            x_col = rec.get('x_col')
            y_col = rec.get('y_col')
            code = f"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='{x_col}', y='{y_col}', alpha=0.7, s=60)
plt.title(f'Relationship: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
plt.xlabel('{x_col}', fontsize=12)
plt.ylabel('{y_col}', fontsize=12)
plt.grid(True, alpha=0.3)

# Add trend line
valid_data = df[['{x_col}', '{y_col}']].dropna()
if len(valid_data) > 1:
    z = np.polyfit(valid_data['{x_col}'], valid_data['{y_col}'], 1)
    p = np.poly1d(z)
    plt.plot(valid_data['{x_col}'], p(valid_data['{x_col}']), "r--", alpha=0.8, linewidth=2, label='Trend')
    plt.legend()
plt.tight_layout()
"""
        
        elif chart_type == 'boxplot':
            x_col = rec.get('x_col')
            y_col = rec.get('y_col')
            code = f"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='{x_col}', y='{y_col}', palette='Set2')
plt.title(f'Distribution of {y_col} by {x_col}', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.ylabel('{y_col}', fontsize=12)
plt.xlabel('{x_col}', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
"""
        
        elif chart_type == 'barplot':
            column = rec.get('column')
            code = f"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
value_counts = df['{column}'].value_counts().head(15)
sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
plt.title(f'Distribution of {column}', fontsize=14, fontweight='bold')
plt.xlabel('{column}', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, v in enumerate(value_counts.values):
    plt.text(i, v + max(value_counts.values)*0.01, str(v), ha='center', va='bottom')
plt.tight_layout()
"""
        
        elif chart_type == 'timeseries':
            x_col = rec.get('x_col')
            y_col = rec.get('y_col')
            code = f"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Convert to datetime if not already
df['{x_col}'] = pd.to_datetime(df['{x_col}'], errors='coerce')

plt.figure(figsize=(14, 6))
plt.plot(df['{x_col}'], df['{y_col}'], linewidth=2, marker='o', markersize=4, alpha=0.8)
plt.title(f'Time Series: {y_col} over {x_col}', fontsize=14, fontweight='bold')
plt.xlabel('{x_col}', fontsize=12)
plt.ylabel('{y_col}', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
"""
        
        elif chart_type == 'missing_data':
            code = f"""
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate missing data
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_data = missing_data[missing_data > 0]

if len(missing_data) > 0:
    plt.figure(figsize=(12, 6))
    
    # Missing data bar chart
    plt.subplot(1, 2, 1)
    sns.barplot(x=missing_data.values, y=missing_data.index, palette='Reds_r')
    plt.title('Missing Data Count by Column', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Missing Values')
    
    # Missing data heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
    plt.title('Missing Data Pattern', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
else:
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, 'No Missing Data Found!\\nYour dataset is complete.', 
             ha='center', va='center', fontsize=16, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    plt.axis('off')
"""
        
        else:
            # Default scatter plot
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                code = f"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='{numeric_cols[0]}', y='{numeric_cols[1]}', alpha=0.7, s=60)
plt.title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
"""
            else:
                code = "print('Insufficient numeric columns for visualization')"
        
        return code
        
    except Exception as e:
        return f"print('Error generating chart code: {str(e)}')"

def execute_chart_code_safely(code, df):
    """Safely execute chart code and return figure"""
    try:
        # Create safe execution environment
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        
        safe_globals = {
            'df': df,
            'plt': plt,
            'sns': sns,
            'np': np,
            'pd': pd
        }
        
        # Clear any existing plots
        plt.clf()
        
        # Execute the code
        exec(code, safe_globals)
        
        # Return the current figure
        return plt.gcf()
        
    except Exception as e:
        # Create error figure
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f'Error creating chart:\n{str(e)}', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        ax.axis('off')
        return fig

def generate_chart_insights(rec, df):
    """Generate AI insights about the created chart"""
    try:
        llm = load_llm("gpt-3.5-turbo")
        
        chart_type = rec.get('chart_type', 'unknown')
        title = rec.get('title', 'Chart Analysis')
        
        # Prepare data summary for AI
        data_summary = f"""
Dataset shape: {df.shape}
Chart type: {chart_type}
Recommendation: {rec.get('description', '')}
"""
        
        if chart_type == 'heatmap':
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()[:10]
            if len(numeric_cols) > 1:
                corr_data = df[numeric_cols].corr()
                strongest_corr = corr_data.abs().unstack().sort_values(ascending=False)
                strongest_corr = strongest_corr[strongest_corr < 1.0].head(3)
                data_summary += f"\nStrongest correlations: {strongest_corr.to_dict()}"
        
        elif chart_type == 'scatter':
            x_col = rec.get('x_col')
            y_col = rec.get('y_col')
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                correlation = df[x_col].corr(df[y_col])
                data_summary += f"\nCorrelation between {x_col} and {y_col}: {correlation:.3f}"
        
        prompt = f"""
Phân tích visualization này và đưa ra 3-4 insights quan trọng:

{data_summary}

Vui lòng đưa ra insights bằng tiếng Việt với format:

## 🔍 Insights Chính

1. **Khám phá Mô hình**: [Những mô hình nào bạn thấy?]
2. **Phát hiện Thống kê**: [Các con số cho chúng ta biết gì?] 
3. **Ý nghĩa Kinh doanh**: [Điều này có nghĩa gì cho việc ra quyết định?]
4. **Khuyến nghị**: [Nên làm gì tiếp theo?]

Giữ nó ngắn gọn nhưng có thể hành động.
"""
        
        response = llm.invoke(prompt)
        
        # Extract content from response
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
            
    except Exception as e:
        return f"""
## 🔍 Insights Chính

1. **Biểu đồ Được tạo**: Đã tạo thành công {rec.get('title', 'visualization')}
2. **Tổng quan Dữ liệu**: Dataset chứa {df.shape[0]:,} hàng và {df.shape[1]} cột
3. **Các bước Tiếp theo**: Khám phá các đề xuất khác hoặc đi sâu vào các mô hình cụ thể
4. **Ghi chú**: Việc tạo insight AI gặp vấn đề: {str(e)}
"""

def generate_and_display_chart(rec, df):
    """Generate, execute and display chart with insights"""
    
    # Create columns for layout
    chart_col, insight_col = st.columns([2, 1])
    
    with chart_col:
        st.markdown(f"#### 📊 {rec['title']}")
        
        # Show loading spinner
        with st.spinner(f"🎨 Đang tạo {rec['action'].lower()}..."):
            # Generate code
            code = generate_chart_code(rec, df)
            
            # Execute code and get figure
            fig = execute_chart_code_safely(code, df)
            
            # Display chart
            if fig:
                st.pyplot(fig)
                plt.close(fig)  # Clean up
            
        # Show code in expander
        with st.expander("📋 Xem Code Đã tạo", expanded=False):
            st.code(code, language='python')
    
    with insight_col:
        st.markdown("#### 🧠 AI Insights")
        
        with st.spinner("🤖 Đang tạo insights..."):
            insights = generate_chart_insights(rec, df)
            st.markdown(insights)
        
        # Action buttons
        st.markdown("#### ⚡ Hành động")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Lưu Biểu đồ", key=f"save_{rec['title']}", use_container_width=True):
                # Save to chart history
                try:
                    # Get dataset_id from session state
                    dataset_id = st.session_state.get('current_dataset_id', 1)
                    add_chart_card(dataset_id, rec['action'], insights, code)
                    st.success("✅ Đã lưu biểu đồ vào lịch sử!")
                except Exception as e:
                    st.error(f"❌ Lỗi khi lưu: {str(e)}")
        
        with col2:
            if st.button("🔄 Tạo lại", key=f"regen_{rec['title']}", use_container_width=True):
                st.rerun()

def create_ai_recommendation_panel(df, analysis_history=None):
    """Create an AI-powered recommendation panel with auto-visualization"""
    
    st.markdown("### 🤖 AI Recommendations")
    
    # Analyze data characteristics
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    recommendations = []
    
    # Data structure recommendations with enhanced details
    if len(numeric_cols) >= 2:
        recommendations.append({
            "type": "correlation",
            "priority": "high",
            "title": "Phân tích Tương quan",
            "description": f"Bạn có {len(numeric_cols)} cột số. Phân tích tương quan giữa các biến để tìm ra những mô hình ẩn.",
            "action": "Tạo correlation heatmap",
            "icon": "🔥",
            "chart_type": "heatmap",
            "columns": numeric_cols[:10]  # Limit for performance
        })
        
        # Add scatter plot recommendation for first two numeric columns
        recommendations.append({
            "type": "scatter",
            "priority": "medium", 
            "title": "Phân tích Mối quan hệ",
            "description": f"Khám phá mối quan hệ giữa {numeric_cols[0]} và {numeric_cols[1]}.",
            "action": "Tạo scatter plot",
            "icon": "📊",
            "chart_type": "scatter",
            "x_col": numeric_cols[0],
            "y_col": numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        })
    
    if categorical_cols and numeric_cols:
        recommendations.append({
            "type": "comparison",
            "priority": "medium",
            "title": "So sánh Nhóm",
            "description": f"So sánh {numeric_cols[0]} giữa các danh mục {categorical_cols[0]} khác nhau.",
            "action": "Tạo box plot analysis",
            "icon": "📦",
            "chart_type": "boxplot",
            "x_col": categorical_cols[0],
            "y_col": numeric_cols[0]
        })
        
        recommendations.append({
            "type": "distribution",
            "priority": "medium",
            "title": "Phân phối Danh mục",
            "description": f"Phân tích phân phối của các danh mục {categorical_cols[0]}.",
            "action": "Tạo bar chart",
            "icon": "📊",
            "chart_type": "barplot",
            "column": categorical_cols[0]
        })
    
    if datetime_cols and numeric_cols:
        recommendations.append({
            "type": "trend",
            "priority": "high",
            "title": "Phân tích Chuỗi thời gian",
            "description": f"Phân tích xu hướng theo thời gian cho {numeric_cols[0]}.",
            "action": "Tạo time series chart",
            "icon": "📈",
            "chart_type": "timeseries",
            "x_col": datetime_cols[0],
            "y_col": numeric_cols[0]
        })
    
    # Data quality recommendations
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 5:
        recommendations.append({
            "type": "quality",
            "priority": "high",
            "title": "Phân tích Chất lượng Dữ liệu",
            "description": f"Dataset có {missing_pct:.1f}% giá trị thiếu. Visualize mô hình dữ liệu thiếu.",
            "action": "Tạo missing data chart",
            "icon": "🧹",
            "chart_type": "missing_data",
            "missing_pct": missing_pct
        })
    
    # Display recommendations with enhanced functionality
    for i, rec in enumerate(recommendations):
        priority_color = "#dc3545" if rec['priority'] == 'high' else "#ffc107" if rec['priority'] == 'medium' else "#28a745"
        
        st.markdown(f'''
        <div class="feature-card" style="border-left-color: {priority_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4>{rec['icon']} {rec['title']}</h4>
                    <p>{rec['description']}</p>
                </div>
                <span style="background: {priority_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.7rem; text-transform: uppercase;">
                    {rec['priority']}
                </span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Enhanced button with auto-visualization
        if st.button(rec['action'], key=f"rec_{rec['title']}_{i}"):
            generate_and_display_chart(rec, df)

def generate_comprehensive_data_story(df: pd.DataFrame, chat_history: list, dataset_name: str) -> str:
    """Tạo một câu chuyện dữ liệu toàn diện với thông tin kinh doanh"""
    llm = load_llm("gpt-3.5-turbo")
    
    # Extract conversation patterns
    questions = [msg[2] for msg in chat_history if msg[1] == "user"][-10:]
    
    # Analyze data characteristics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    prompt = f"""
    Bạn là một nhà phân tích dữ liệu cấp cao đang tạo một tóm tắt điều hành cho bộ dữ liệu '{dataset_name}'.
    
    📊 TỔNG QUAN BỘ DỮ LIỆU:
    - Kích thước: {df.shape[0]:,} hàng × {df.shape[1]} cột
    - Biến số: {len(numeric_cols)} ({', '.join(numeric_cols[:5])})
    - Biến phân loại: {len(categorical_cols)} ({', '.join(categorical_cols[:5])})
    - Dữ liệu thiếu: {df.isnull().sum().sum():,} ô ({(df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100):.1f}%)
    
    🔍 CÁC CÂU HỎI PHÂN TÍCH GẦN ĐÂY:
    {questions}
    
    Tạo một tóm tắt điều hành hấp dẫn với:
    
    ## 📈 Tóm tắt Điều hành
    [2-3 câu làm nổi bật những phát hiện quan trọng nhất]
    
    ## 🎯 Thông tin Chính
    [4-5 thông tin cụ thể, có thể hành động với số liệu khi có thể]
    
    ## 📊 Đánh giá Chất lượng Dữ liệu  
    [Đánh giá ngắn gọn về độ tin cậy và tính đầy đủ của dữ liệu]
    
    ## 💼 Tác động Kinh doanh
    [Những thông tin này có thể thúc đẩy quyết định kinh doanh như thế nào]
    
    ## 🚀 Các Bước Tiếp theo Được Khuyến nghị
    [3-4 hành động cụ thể cần thực hiện dựa trên phân tích]
    
    ## ⚠️ Hạn chế & Cân nhắc
    [Những lưu ý quan trọng về dữ liệu hoặc phân tích]
    
    Làm cho nó sẵn sàng cho điều hành: chuyên nghiệp, súc tích và tập trung vào những thông tin có thể hành động.
    Sử dụng số cụ thể và tỷ lệ phần trăm khi có thể.
    """
    
    try:
        response = llm.invoke(prompt)
        # Handle different response types
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    except Exception as e:
        return f"❌ Lỗi tạo câu chuyện dữ liệu: {str(e)}"

def extract_enhanced_chart_insights(code: str, df: pd.DataFrame) -> str:
    """Trích xuất thông tin chi tiết về biểu đồ được tạo"""
    llm = load_llm("gpt-3.5-turbo")
    
    # Identify chart type from code
    chart_type = "Không xác định"
    if "scatter" in code.lower():
        chart_type = "Biểu đồ Phân tán"
    elif "bar" in code.lower():
        chart_type = "Biểu đồ Cột" 
    elif "hist" in code.lower():
        chart_type = "Biểu đồ Tần suất"
    elif "box" in code.lower():
        chart_type = "Biểu đồ Hộp"
    elif "line" in code.lower():
        chart_type = "Biểu đồ Đường"
    elif "heatmap" in code.lower():
        chart_type = "Bản đồ Nhiệt"
    
    prompt = f"""
    Phân tích {chart_type} này được tạo từ đoạn code sau:
    
    ```python
    {code}
    ```
    
    Đặc điểm bộ dữ liệu:
    - Kích thước: {df.shape}
    - Cột: {list(df.columns)}
    - Kiểu dữ liệu: {df.dtypes.to_dict()}
    
    Cung cấp thông tin chi tiết theo định dạng này:
    
    ## 📊 Phân tích Biểu đồ
    [Biểu đồ này hiển thị gì và tại sao nó hữu ích]
    
    ## 🔍 Mô hình Chính
    [Các mô hình, xu hướng hoặc mối quan hệ cụ thể có thể nhìn thấy]
    
    ## 📈 Thông tin Thống kê  
    [Quan sát định lượng với số thực tế]
    
    ## 💡 Giá trị Kinh doanh
    [Trực quan hóa này giúp quyết định kinh doanh như thế nào]
    
    ## 🎯 Gợi ý Theo dõi
    [Những phân tích bổ sung nào sẽ có giá trị]
    
    Hãy cụ thể và bao gồm tên cột thực tế và các giá trị tiềm năng.
    Tập trung vào những thông tin có thể hành động mà các bên liên quan có thể sử dụng.
    """
    
    try:
        response = llm.invoke(prompt)
        # Handle different response types
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    except Exception as e:
        return f"❌ Lỗi tạo insights biểu đồ: {str(e)}"

datasets = get_all_datasets()
if not datasets:
    render_feature_card(
        "Chào mừng đến với VizGenie-GPT",
        "Bắt đầu bằng cách tải lên bộ dữ liệu đầu tiên của bạn trong trang Bảng điều khiển để bắt đầu phân tích nâng cao.",
        "👋",
        "Đi đến Bảng điều khiển",
        "dashboard"
    )
    st.stop()

# Dataset selection with enhanced UI
st.markdown("### 📂 Lựa chọn Bộ dữ liệu")
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    dataset_options = {f"{d[0]} - {d[1]}": d[0] for d in datasets}
    selected = st.selectbox(
        "Chọn bộ dữ liệu của bạn:",
        list(dataset_options.keys()),
        help="Chọn bộ dữ liệu bạn muốn phân tích"
    )
    dataset_id = dataset_options[selected]
    dataset = get_dataset(dataset_id)
    
    # Store dataset_id in session state for saving charts
    st.session_state.current_dataset_id = dataset_id

with col2:
    if st.button("📊 Tạo Câu chuyện Dữ liệu", type="primary", use_container_width=True):
        st.session_state.generate_story = True

with col3:
    if st.button("🔍 Khám phá Dữ liệu", type="secondary", use_container_width=True):
        st.session_state.show_explorer = True

# Load and validate dataset
file_path = dataset[2]
num_rows, num_cols = dataset[3], dataset[4]

try:
    df = safe_read_csv(file_path)
    st.session_state.df = df
except Exception as e:
    st.error(f"❌ Lỗi khi tải bộ dữ liệu: {e}")
    st.stop()

# Dataset metrics with professional cards
st.markdown("### 📊 Tổng quan Bộ dữ liệu")
metrics = [
    {"title": "Tổng số Bản ghi", "value": f"{num_rows:,}", "delta": None},
    {"title": "Cột", "value": str(num_cols), "delta": None},
    {"title": "Trường Số", "value": str(df.select_dtypes(include=[np.number]).shape[1]), "delta": None},
    {"title": "Giá trị Thiếu", "value": f"{df.isnull().sum().sum():,}", "delta": None}
]

render_metric_cards(metrics)

# Data quality assessment
st.markdown("### 🎯 Đánh giá Chất lượng Dữ liệu")
quality_score = create_data_quality_indicator(df)

if quality_score < 0.7:
    render_status_indicator("Chất lượng Dữ liệu Cần Chú ý", "warning")
elif quality_score < 0.9:
    render_status_indicator("Chất lượng Dữ liệu Tốt", "success")
else:
    render_status_indicator("Chất lượng Dữ liệu Tuyệt vời", "success")

# Enhanced AI Recommendations Panel
create_ai_recommendation_panel(df)

# Interactive Data Explorer (if requested)
if st.session_state.get('show_explorer', False):
    with st.expander("🔍 Khám phá Dữ liệu Tương tác", expanded=True):
        render_interactive_data_explorer(df)
    st.session_state.show_explorer = False

# Chat session management with enhanced UI
st.markdown("### 💬 Phiên Phân tích AI")

# Session selection
sessions = get_sessions_by_dataset(dataset_id)
session_titles = {f"{s[0]} - {s[1]} ({s[2]})": s[0] for s in sessions}

col1, col2 = st.columns([3, 1])
with col1:
    new_session_title = st.text_input(
        "🆕 Tạo phiên phân tích mới:",
        placeholder="ví dụ: Phân tích Doanh thu, Phân khúc Khách hàng, Khám phá Xu hướng...",
        help="Đặt tên mô tả cho phiên phân tích của bạn"
    )

with col2:
    session_type = st.radio("Phiên:", ("Mới", "Hiện có"), horizontal=True)

if session_type == "Hiện có" and sessions:
    selected_session = st.selectbox("Chọn phiên hiện có:", list(session_titles.keys()))
    session_id = session_titles[selected_session]
    
    # Session management options
    with st.expander("⚙️ Quản lý Phiên"):
        col1, col2 = st.columns(2)
        with col1:
            rename_title = st.text_input("Đổi tên phiên:")
            if st.button("✏️ Đổi tên") and rename_title:
                rename_chat_session(session_id, rename_title)
                st.success("✅ Đã đổi tên phiên!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Xóa Phiên", type="secondary"):
                delete_chat_session(session_id)
                st.success("🗑️ Đã xóa phiên!")
                st.rerun()

else:
    # Create new session
    default_title = new_session_title or f"Phiên Phân tích {len(sessions) + 1}"
    session_id = create_chat_session(dataset_id, default_title)
    st.success(f"✅ Đã tạo phiên: **{default_title}**")

# Load chat history
chat_history = get_chat_messages(session_id)

# Generate comprehensive data story if requested
if st.session_state.get('generate_story', False):
    with st.spinner("🤖 Đang tạo câu chuyện dữ liệu toàn diện..."):
        render_animated_loading("Đang phân tích dữ liệu của bạn và tạo thông tin chi tiết...")
        
        story = generate_comprehensive_data_story(df, chat_history, dataset[1])
        
        render_insight_card(story)
        
        # Save story to chat history
        add_chat_message(session_id, "assistant", f"**📖 Đã Tạo Câu chuyện Dữ liệu**\n\n{story}")
        
    st.session_state.generate_story = False

# Enhanced chat history display
if chat_history:
    st.markdown("### 🗨️ Lịch sử Trò chuyện")
    
    for idx, (msg_id, role, content, ts) in enumerate(chat_history):
        with st.chat_message(role):
            cols = st.columns([10, 1])
            
            with cols[0]:
                # Enhanced message rendering
                if role == "assistant" and "📖 Câu chuyện Dữ liệu" in content:
                    # Special rendering for data stories
                    render_insight_card(content.replace("**📖 Đã Tạo Câu chuyện Dữ liệu**\n\n", ""))
                else:
                    st.markdown(content)
            
            with cols[1]:
                if role == "user":
                    with st.popover("⋮", use_container_width=True):
                        if st.button("✏️ Chỉnh sửa", key=f"edit_{idx}"):
                            st.session_state.edited_prompt = content
                            # Delete this message and the next AI response
                            delete_chat_message(session_id, msg_id)
                            if idx + 1 < len(chat_history) and chat_history[idx + 1][1] == "assistant":
                                delete_chat_message(session_id, chat_history[idx + 1][0])
                            st.rerun()
                        
                        if st.button("🗑️ Xóa", key=f"del_{msg_id}"):
                            delete_chat_message(session_id, msg_id)
                            # Also delete the AI response if it exists
                            if idx + 1 < len(chat_history) and chat_history[idx + 1][1] == "assistant":
                                delete_chat_message(session_id, chat_history[idx + 1][0])
                            st.rerun()
                        
                        if st.button("📋 Sao chép", key=f"copy_{idx}"):
                            st.session_state.clipboard = content
                            st.success("Đã sao chép vào clipboard!")

# Smart query suggestions with enhanced UI
st.markdown("### 💡 Gợi ý Truy vấn Thông minh")
with st.expander("🎯 Lấy Cảm hứng - Câu hỏi Mẫu", expanded=False):
    
    # Dynamic suggestions based on data characteristics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Câu hỏi Phân tích Dữ liệu:**")
        
        analysis_suggestions = [
            f"Hiển thị phân phối của {numeric_cols[0]}" if numeric_cols else "Phân tích phân phối dữ liệu",
            f"Mối tương quan giữa {numeric_cols[0]} và {numeric_cols[1]} là gì?" if len(numeric_cols) >= 2 else "Tìm tương quan trong dữ liệu",
            "Xác định các giá trị ngoại lệ và bất thường trong bộ dữ liệu",
            f"So sánh {numeric_cols[0]} giữa các nhóm {categorical_cols[0]} khác nhau" if numeric_cols and categorical_cols else "So sánh các nhóm trong dữ liệu",
            "Tạo tóm tắt thống kê toàn diện"
        ]
        
        for suggestion in analysis_suggestions:
            if st.button(suggestion, key=f"analysis_{suggestion[:20]}", use_container_width=True):
                st.session_state.suggested_prompt = suggestion
    
    with col2:
        st.markdown("**🎯 Câu hỏi Thông minh Kinh doanh:**")
        
        # Context-aware business questions
        business_questions = [
            "Các chỉ số hiệu suất chính trong dữ liệu này là gì?",
            "Những yếu tố nào có tác động mạnh nhất đến kết quả?",
            "Có những mô hình theo mùa hoặc theo thời gian không?", 
            "Những phân khúc hoặc nhóm nào cho thấy hiệu suất tốt nhất?",
            "Bạn có thể đưa ra những khuyến nghị nào dựa trên dữ liệu này?"
        ]
        
        for question in business_questions:
            if st.button(question, key=f"business_{question[:20]}", use_container_width=True):
                st.session_state.suggested_prompt = question

# Enhanced chart type suggestions
st.markdown("### 📈 Khuyến nghị Biểu đồ Thông minh")
with st.expander("🎨 Trực quan hóa được AI Đề xuất", expanded=False):
    
    chart_recommendations = []
    
    if len(numeric_cols) >= 2:
        chart_recommendations.extend([
            {"type": "Biểu đồ Phân tán", "desc": f"Khám phá mối quan hệ giữa {numeric_cols[0]} và {numeric_cols[1]}", "icon": "🔵"},
            {"type": "Bản đồ Nhiệt Tương quan", "desc": "Hiển thị tất cả tương quan số", "icon": "🔥"}
        ])
    
    if categorical_cols and numeric_cols:
        chart_recommendations.extend([
            {"type": "Biểu đồ Hộp", "desc": f"So sánh phân phối {numeric_cols[0]} theo {categorical_cols[0]}", "icon": "📦"},
            {"type": "Biểu đồ Cột", "desc": f"Hiển thị trung bình {numeric_cols[0]} theo {categorical_cols[0]}", "icon": "📊"}
        ])
    
    if any('date' in col.lower() or 'time' in col.lower() for col in df.columns):
        chart_recommendations.append(
            {"type": "Chuỗi Thời gian", "desc": "Theo dõi thay đổi theo thời gian", "icon": "📈"}
        )
    
    # Display recommendations in a grid
    if chart_recommendations:
        cols = st.columns(min(3, len(chart_recommendations)))
        for i, rec in enumerate(chart_recommendations[:6]):
            with cols[i % 3]:
                render_feature_card(
                    f"{rec['icon']} {rec['type']}", 
                    rec['desc'],
                    rec['icon']
                )
                if st.button(f"Tạo {rec['type']}", key=f"chart_rec_{i}", use_container_width=True):
                    st.session_state.suggested_prompt = f"Tạo một {rec['type'].lower()} hiển thị {rec['desc']}"

# Main chat input with enhanced processing
prompt = (st.session_state.pop("suggested_prompt", None) or 
          st.session_state.pop("edited_prompt", None) or 
          st.chat_input("🤖 Hỏi bất cứ điều gì về dữ liệu của bạn - Tôi sẽ tạo ra những trực quan hóa và thông tin đẹp mắt!"))

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    add_chat_message(session_id, "user", prompt)
    
    # Enhanced AI response with professional styling
    with st.chat_message("assistant"):
        try:
            # Create agent with enhanced prompting
            agent = create_agent_from_csv("gpt-3.5-turbo", file_path, return_steps=True)
            enhanced_prompt = enhance_prompt_with_chart_suggestions(prompt, df)
            
            # Show processing indicator
            with st.spinner("🧠 Đang phân tích dữ liệu của bạn với AI..."):
                response = agent.invoke(enhanced_prompt)
            
            # Extract execution details
            steps = response.get("intermediate_steps", [])
            action_code = steps[-1][0].tool_input["query"] if steps else ""
            
            # Display main response
            st.markdown(response["output"])
            add_chat_message(session_id, "assistant", response["output"])
            
            # Enhanced chart processing
            if action_code and ("plt" in action_code or "seaborn" in action_code or "sns" in action_code):
                
                # Apply intelligent chart enhancements
                patched_code = smart_patch_chart_code(action_code, df)
                
                # Create chart layout
                chart_col, controls_col = st.columns([3, 1])
                
                with chart_col:
                    st.markdown("#### 📊 Trực quan hóa được Tạo")
                    
                    # Execute and display chart
                    fig = execute_plt_code(patched_code, df)
                    if fig:
                        st.pyplot(fig)
                        
                        # Generate enhanced insights
                        with st.spinner("🔍 Đang trích xuất thông tin sâu..."):
                            chart_insights = extract_enhanced_chart_insights(patched_code, df)
                        
                        render_insight_card(chart_insights)
                
                with controls_col:
                    st.markdown("#### 🎨 Cải tiến Biểu đồ")
                    
                    # Color scheme selector
                    color_scheme = st.selectbox(
                        "Bảng Màu:",
                        list(ENHANCED_COLOR_SCHEMES.keys()),
                        index=0,
                        key=f"color_{len(chat_history)}"
                    )
                    
                    # Enhancement options
                    enhancements = st.multiselect(
                        "Thêm Tính năng:",
                        [
                            "Thêm đường xu hướng",
                            "Hiển thị nhãn dữ liệu",
                            "Thêm lưới",
                            "Sử dụng thang logarithm", 
                            "Làm nổi bật ngoại lệ",
                            "Thêm chú thích"
                        ],
                        key=f"enhance_{len(chat_history)}"
                    )
                    
                    # Apply enhancements
                    if st.button("🔄 Áp dụng Thay đổi", key=f"apply_{len(chat_history)}"):
                        enhanced_code = apply_chart_enhancements(patched_code, color_scheme, enhancements)
                        fig_enhanced = execute_plt_code(enhanced_code, df)
                        if fig_enhanced:
                            with chart_col:
                                st.markdown("#### ✨ Trực quan hóa được Cải tiến")
                                st.pyplot(fig_enhanced)
                    
                    # Chart actions
                    st.markdown("#### 💾 Hành động Biểu đồ")
                    
                    if st.button("Lưu vào Thư viện", key=f"save_{len(chat_history)}", use_container_width=True):
                        add_chart_card(dataset_id, prompt, response["output"], patched_code)
                        st.success("✅ Đã lưu biểu đồ!")
                    
                    if st.button("Tải xuống PNG", key=f"download_{len(chat_history)}", use_container_width=True):
                        st.info("📥 Chức năng tải xuống sẽ được triển khai ở đây")
                    
                    if st.button("Chia sẻ Biểu đồ", key=f"share_{len(chat_history)}", use_container_width=True):
                        st.info("📤 Chức năng chia sẻ sẽ được triển khai ở đây")
                
                # Code display with tabs
                with st.expander("📋 Xem Code được Tạo", expanded=False):
                    tab1, tab2 = st.tabs(["Code Cải tiến", "Code AI Gốc"])
                    
                    with tab1:
                        st.code(patched_code, language="python")
                        st.caption("Code này bao gồm kiểu dáng chuyên nghiệp và xử lý dữ liệu thông minh")
                    
                    with tab2:
                        st.code(action_code, language="python") 
                        st.caption("Code gốc được tạo bởi AI")
            
            # Handle Plotly charts
            elif action_code and ("plotly" in action_code or "px." in action_code):
                st.markdown("#### 📊 Trực quan hóa Tương tác")
                try:
                    exec_globals = {"df": df, "px": px, "go": go, "st": st}
                    exec(action_code, exec_globals)
                    
                    render_insight_card("🎯 **Đã Tạo Biểu đồ Tương tác!** Trực quan hóa Plotly này hỗ trợ phóng to, di chuột và khám phá tương tác.")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi tạo biểu đồ tương tác: {e}")
        
        except Exception as e:
            st.error(f"❌ Phân tích thất bại: {e}")
            render_insight_card(
                "💡 **Mẹo Khắc phục Sự cố:**\n"
                "- Thử diễn đạt lại câu hỏi của bạn cụ thể hơn\n" 
                "- Đề cập đến tên cột cụ thể bạn muốn phân tích\n"
                "- Yêu cầu một loại biểu đồ hoặc phân tích cụ thể\n"
                "- Kiểm tra xem dữ liệu của bạn có các cột cần thiết cho phân tích không"
            )

# Professional sidebar with navigation and stats
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e1e5e9; margin-bottom: 1rem;">
        <h3 style="color: #667eea; margin: 0;">🧠 VizGenie-GPT</h3>
        <small style="color: #666;">Nền tảng Phân tích Chuyên nghiệp</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick navigation
    st.markdown("### 🔗 Điều hướng Nhanh")
    nav_buttons = [
        ("📂 Bảng điều khiển", "pages/1_🧮_Bang_Dieu_Khien.py"),
        ("📊 Chi tiết Bộ dữ liệu", "pages/3_📂_Chi_Tiet_Bo_Du_Lieu.py"), 
        ("📈 Biểu đồ Thông minh", "pages/6_📈_Bieu_Do_Thong_Minh.py"),
        ("🔗 Phân tích Chéo", "pages/7_🔗_Phan_Tich_Cheo_Du_Lieu.py"),
        ("📋 Lịch sử Biểu đồ", "pages/4_📊_Lich_Su_Bieu_Do.py"),
        ("📄 Báo cáo EDA", "pages/5_📋_Bao_Cao_EDA.py")
    ]
    
    for label, page in nav_buttons:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.switch_page(page)
    
    # Session statistics
    if chat_history:
        st.markdown("---")
        st.markdown("### 📈 Thống kê Phiên")
        
        user_messages = [msg for msg in chat_history if msg[1] == "user"]
        charts_created = len([msg for msg in chat_history if "chart" in msg[2].lower() or "plot" in msg[2].lower()])
        
        render_metric_cards([
            {"title": "Câu hỏi", "value": str(len(user_messages))},
            {"title": "Biểu đồ", "value": str(charts_created)},
            {"title": "Chất lượng", "value": f"{quality_score:.0%}"}
        ])
        
        # Session summary
        if st.button("📊 Tạo Tóm tắt Phiên", use_container_width=True):
            summary_prompt = f"""
            Tóm tắt phiên phân tích dữ liệu này trong 3 điểm chính:
            
            Câu hỏi đã hỏi: {[msg[2] for msg in user_messages]}
            Bộ dữ liệu: {dataset[1]} ({df.shape[0]} hàng, {df.shape[1]} cột)
            
            Tập trung vào:
            - Các lĩnh vực phân tích chính đã khám phá
            - Những thông tin chính đã khám phá  
            - Các loại trực quan hóa đã tạo
            
            Giữ nó súc tích và thân thiện với điều hành.
            """
            
            with st.spinner("Đang tạo tóm tắt..."):
                summary = load_llm("gpt-3.5-turbo").invoke(summary_prompt)
                render_insight_card(f"**📋 Tóm tắt Phiên**\n\n{summary}")
    
    # Pro tips
    st.markdown("---")
    st.markdown("### 💡 Mẹo Chuyên nghiệp")
    st.markdown("""
    **🎯 Câu hỏi Tốt hơn:**
    - Cụ thể về các cột
    - Yêu cầu so sánh
    - Yêu cầu thông tin kinh doanh
    
    **📊 Mẹo Biểu đồ:**
    - Thử các bảng màu khác nhau
    - Sử dụng cải tiến để rõ ràng
    - Lưu các biểu đồ bạn thích
    
    **🤖 Tính năng AI:**
    - Tạo câu chuyện dữ liệu
    - Nhận khuyến nghị biểu đồ
    - Đặt câu hỏi tiếp theo
    """)

# Footer with credits and version
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🧠 VizGenie-GPT Chuyên nghiệp**")
    st.caption("Nền tảng Phân tích AI Nâng cao")

with col2:
    st.markdown("**🔧 Phiên bản 2.0**")
    st.caption("Cải tiến với Giao diện Chuyên nghiệp")

with col3:
    st.markdown("**👨‍💻 Được tạo bởi Delay Group**")
    st.caption("Với ❤️ cho khoa học dữ liệu")