import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.llms import load_llm
from src.utils import get_all_datasets, get_dataset, safe_read_csv, add_chart_card
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Biểu Đồ Thông Minh Nâng Cao", layout="wide")

# Professional styling with modern color schemes
st.markdown("""
<style>
    .chart-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #56CCF2 0%, #2F80ED 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .chart-option {
        color: black;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid transparent;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .chart-option:hover {
        border-color: #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
    }
    .color-palette {
        display: flex;
        gap: 10px;
        margin: 10px 0;
    }
    .color-box {
        width: 30px;
        height: 30px;
        border-radius: 5px;
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chart-header"><h1>📈 Công Cụ Tạo Biểu Đồ Thông Minh AI</h1><p>Nhận đề xuất biểu đồ thông minh và tạo ra các trực quan hóa tuyệt đẹp</p></div>', unsafe_allow_html=True)

llm = load_llm("gpt-4o")

# Bảng màu chuyên nghiệp
COLOR_PALETTES = {
    "Xanh Chuyên Nghiệp": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "Sống Động": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"],
    "Doanh Nghiệp": ["#2C3E50", "#3498DB", "#E74C3C", "#F39C12", "#27AE60", "#8E44AD", "#16A085", "#E67E22", "#34495E", "#1ABC9C"],
    "Hoàng Hôn": ["#FF6B35", "#F7931E", "#FFD23F", "#06FFA5", "#118AB2", "#073B4C", "#E63946", "#F77F00", "#FCBF49", "#003566"],
    "Đại Dương": ["#0077BE", "#00A8CC", "#0FA3B1", "#B5E2FA", "#F9E784", "#F8AD9D", "#F4975A", "#E8871E", "#DA627D", "#A53860"]
}

# Tải datasets
datasets = get_all_datasets()
if not datasets:
    st.warning("⚠️ Vui lòng tải lên bộ dữ liệu từ trang Bảng điều khiển.")
    st.stop()

dataset_options = {f"{d[0]} - {d[1]}": d[0] for d in datasets}
selected = st.selectbox("📂 Chọn bộ dữ liệu để phân tích:", list(dataset_options.keys()))
dataset_id = dataset_options[selected]
dataset = get_dataset(dataset_id)
file_path = dataset[2]

@st.cache_data
def load_csv(file_path):
    for enc in ['utf-8', 'ISO-8859-1', 'utf-16', 'cp1252']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except:
            continue
    raise ValueError("❌ Không thể giải mã file CSV.")

df = load_csv(file_path)

st.markdown(f"**🧾 Thông tin Bộ dữ liệu:** `{dataset[1]}` — {df.shape[0]:,} hàng × {df.shape[1]} cột")

def get_chart_recommendations(df, user_intent=""):
    """Đề xuất biểu đồ AI dựa trên đặc điểm dữ liệu"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Tìm cột có thể là thời gian (bao gồm cả year)
    datetime_cols = []
    for col in df.columns:
        if ('date' in col.lower() or 'time' in col.lower() or 
            'year' in col.lower() or col.lower() == 'year'):
            datetime_cols.append(col)
        # Kiểm tra nếu cột số có giá trị năm (1900-2100)
        elif (col in numeric_cols and 
              df[col].min() >= 1900 and df[col].max() <= 2100 and 
              df[col].nunique() < 50):
            datetime_cols.append(col)
    
    recommendations = []
    
    # Phân tích dựa trên kiểu dữ liệu
    if len(numeric_cols) >= 2:
        recommendations.append({
            "type": "Biểu Đồ Phân Tán",
            "description": "Hoàn hảo để khám phá mối quan hệ giữa hai biến số",
            "confidence": 0.9,
            "suggested_x": numeric_cols[0],
            "suggested_y": numeric_cols[1],
            "icon": "🔵",
            "color_scheme": "Xanh Chuyên Nghiệp"
        })
        
        recommendations.append({
            "type": "Bản Đồ Nhiệt Tương Quan",
            "description": "Hiển thị tương quan giữa tất cả các biến số",
            "confidence": 0.85,
            "suggested_x": "Tất cả biến số",
            "suggested_y": "Tất cả biến số",
            "icon": "🔥",
            "color_scheme": "Hoàng Hôn"
        })
    
    if categorical_cols and numeric_cols:
        recommendations.append({
            "type": "Biểu Đồ Hộp",
            "description": "So sánh phân phối dữ liệu số theo các danh mục",
            "confidence": 0.8,
            "suggested_x": categorical_cols[0],
            "suggested_y": numeric_cols[0],
            "icon": "📦",
            "color_scheme": "Sống Động"
        })
        
        recommendations.append({
            "type": "Biểu Đồ Cột",
            "description": "Hiển thị giá trị trung bình hoặc tổng theo danh mục",
            "confidence": 0.75,
            "suggested_x": categorical_cols[0],
            "suggested_y": numeric_cols[0],
            "icon": "📊",
            "color_scheme": "Doanh Nghiệp"
        })
    
    if datetime_cols and numeric_cols:
        # Ưu tiên cột year nếu có
        time_col = datetime_cols[0]
        for col in datetime_cols:
            if 'year' in col.lower() or col.lower() == 'year':
                time_col = col
                break
                
        recommendations.append({
            "type": "Chuỗi Thời Gian",
            "description": "Theo dõi thay đổi theo thời gian hoặc năm",
            "confidence": 0.95,
            "suggested_x": time_col,
            "suggested_y": numeric_cols[0],
            "icon": "📈",
            "color_scheme": "Đại Dương"
        })
    
    if categorical_cols:
        recommendations.append({
            "type": "Biểu Đồ Tròn",
            "description": "Hiển thị tỷ lệ của các danh mục",
            "confidence": 0.7,
            "suggested_x": categorical_cols[0],
            "suggested_y": "Đếm",
            "icon": "🥧",
            "color_scheme": "Hoàng Hôn"
        })
    
    # Sắp xếp theo độ tin cậy
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    
    return recommendations[:6]  # Trả về 6 đề xuất hàng đầu

def create_enhanced_chart(chart_type, df, x_col, y_col, color_col=None, palette="Xanh Chuyên Nghiệp", custom_prompt=""):
    """Tạo biểu đồ nâng cao với thiết kế chuyên nghiệp"""
    
    colors = COLOR_PALETTES[palette]
    
    fig = None
    code = ""
    
    if chart_type == "Biểu Đồ Phân Tán":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        color_discrete_sequence=colors,
                        title=f"Biểu Đồ Phân Tán: {x_col} vs {y_col}",
                        template="plotly_white")
        
        code = f"""
import plotly.express as px

fig = px.scatter(df, x='{x_col}', y='{y_col}', 
                color='{color_col}' if '{color_col}' != 'None' else None,
                color_discrete_sequence={colors},
                title="Biểu Đồ Phân Tán: {x_col} vs {y_col}",
                template="plotly_white")

fig.update_layout(
    font=dict(size=12),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
fig.show()
"""
    
    elif chart_type == "Biểu Đồ Cột":
        if df[x_col].dtype == 'object':
            agg_df = df.groupby(x_col)[y_col].mean().reset_index()
        else:
            agg_df = df
            
        fig = px.bar(agg_df, x=x_col, y=y_col,
                    color_discrete_sequence=colors,
                    title=f"Biểu Đồ Cột: {y_col} theo {x_col}",
                    template="plotly_white")
        
        code = f"""
import plotly.express as px

# Tổng hợp dữ liệu nếu cần
if df['{x_col}'].dtype == 'object':
    agg_df = df.groupby('{x_col}')['{y_col}'].mean().reset_index()
else:
    agg_df = df

fig = px.bar(agg_df, x='{x_col}', y='{y_col}',
            color_discrete_sequence={colors},
            title="Biểu Đồ Cột: {y_col} theo {x_col}",
            template="plotly_white")
fig.show()
"""
    
    elif chart_type == "Biểu Đồ Hộp":
        fig = px.box(df, x=x_col, y=y_col, color=color_col,
                    color_discrete_sequence=colors,
                    title=f"Biểu Đồ Hộp: Phân phối {y_col} theo {x_col}",
                    template="plotly_white")
        
        code = f"""
import plotly.express as px

fig = px.box(df, x='{x_col}', y='{y_col}', 
            color='{color_col}' if '{color_col}' != 'None' else None,
            color_discrete_sequence={colors},
            title="Biểu Đồ Hộp: Phân phối {y_col} theo {x_col}",
            template="plotly_white")
fig.show()
"""
    
    elif chart_type == "Chuỗi Thời Gian":
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        df_temp = df.copy()
        
        # Kiểm tra và xử lý dữ liệu thời gian
        try:
            # Nếu cột x chứa năm (year) thì sắp xếp theo năm
            if 'year' in x_col.lower() or df_temp[x_col].dtype in ['int64', 'float64']:
                df_temp = df_temp.sort_values(x_col)
                # Nếu dữ liệu có nhiều giá trị cho cùng một năm, tính trung bình
                if df_temp[x_col].duplicated().any():
                    df_temp = df_temp.groupby(x_col)[y_col].mean().reset_index()
            else:
                # Thử chuyển đổi sang datetime
                df_temp[x_col] = pd.to_datetime(df_temp[x_col])
                df_temp = df_temp.sort_values(x_col)
        except:
            # Nếu không thể chuyển đổi, sắp xếp theo giá trị gốc
            df_temp = df_temp.sort_values(x_col)
        
        fig = px.line(df_temp, x=x_col, y=y_col, color=color_col,
                     color_discrete_sequence=colors,
                     title=f"Biểu Đồ Đường: {y_col} theo {x_col}",
                     template="plotly_white",
                     markers=True)  # Thêm markers để dễ nhìn hơn
        
        code = f"""
import plotly.express as px
import pandas as pd

# Tạo bản sao và xử lý dữ liệu
df_temp = df.copy()

# Xử lý dữ liệu thời gian
try:
    if 'year' in '{x_col}'.lower() or df_temp['{x_col}'].dtype in ['int64', 'float64']:
        df_temp = df_temp.sort_values('{x_col}')
        # Tính trung bình nếu có nhiều giá trị cho cùng một năm
        if df_temp['{x_col}'].duplicated().any():
            df_temp = df_temp.groupby('{x_col}')['{y_col}'].mean().reset_index()
    else:
        df_temp['{x_col}'] = pd.to_datetime(df_temp['{x_col}'])
        df_temp = df_temp.sort_values('{x_col}')
except:
    df_temp = df_temp.sort_values('{x_col}')

fig = px.line(df_temp, x='{x_col}', y='{y_col}',
             color='{color_col}' if '{color_col}' != 'None' else None,
             color_discrete_sequence={colors},
             title="Biểu Đồ Đường: {y_col} theo {x_col}",
             template="plotly_white",
             markers=True)
fig.show()
"""
    
    elif chart_type == "Bản Đồ Nhiệt Tương Quan":
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, 
                       color_continuous_scale="RdBu_r",
                       title="Bản Đồ Nhiệt Tương Quan",
                       template="plotly_white")
        
        code = f"""
import plotly.express as px

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

fig = px.imshow(corr_matrix, 
               color_continuous_scale="RdBu_r",
               title="Bản Đồ Nhiệt Tương Quan",
               template="plotly_white")
fig.show()
"""
    
    elif chart_type == "Biểu Đồ Tròn":
        value_counts = df[x_col].value_counts().head(10)
        fig = px.pie(values=value_counts.values, names=value_counts.index,
                    color_discrete_sequence=colors,
                    title=f"Phân phối của {x_col}",
                    template="plotly_white")
        
        code = f"""
import plotly.express as px

value_counts = df['{x_col}'].value_counts().head(10)
fig = px.pie(values=value_counts.values, names=value_counts.index,
            color_discrete_sequence={colors},
            title="Phân phối của {x_col}",
            template="plotly_white")
fig.show()
"""
    
    elif chart_type == "Biểu Đồ Violin":
        fig = px.violin(df, x=x_col, y=y_col, color=color_col,
                       color_discrete_sequence=colors,
                       title=f"Biểu Đồ Violin: Phân phối {y_col} theo {x_col}",
                       template="plotly_white")
        
        code = f"""
import plotly.express as px

fig = px.violin(df, x='{x_col}', y='{y_col}',
               color='{color_col}' if '{color_col}' != 'None' else None,
               color_discrete_sequence={colors},
               title="Biểu Đồ Violin: Phân phối {y_col} theo {x_col}",
               template="plotly_white")
fig.show()
"""
    
    elif chart_type == "Biểu Đồ Tần Suất":
        fig = px.histogram(df, x=x_col, y=y_col, color=color_col,
                          color_discrete_sequence=colors,
                          title=f"Biểu Đồ Tần Suất: {x_col}",
                          template="plotly_white")
        
        code = f"""
import plotly.express as px

fig = px.histogram(df, x='{x_col}', y='{y_col}' if '{y_col}' != '{x_col}' else None,
                  color='{color_col}' if '{color_col}' != 'None' else None,
                  color_discrete_sequence={colors},
                  title="Biểu Đồ Tần Suất: {x_col}",
                  template="plotly_white")
fig.show()
"""
    
    # Apply custom styling
    if fig:
        fig.update_layout(
            font=dict(size=12, family="Arial, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title=dict(x=0.5, font=dict(size=16, color='#2c3e50')),
            margin=dict(t=60, l=60, r=60, b=60)
        )
        
        # Thêm sửa đổi prompt tùy chỉnh nếu được cung cấp
        if custom_prompt:
            prompt_modifications = f"""
            
# Sửa đổi tùy chỉnh dựa trên yêu cầu người dùng: "{custom_prompt}"
# Thêm bất kỳ thiết kế hoặc sửa đổi cụ thể nào ở đây
"""
            code += prompt_modifications
    
    return fig, code

# Giao diện chính
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 Đề xuất Biểu đồ AI")
    
    user_intent = st.text_input("💭 Bạn muốn kể câu chuyện gì?", 
                               placeholder="ví dụ: hiển thị xu hướng, so sánh danh mục, tìm ngoại lệ...")
    
    recommendations = get_chart_recommendations(df, user_intent)
    
    st.markdown("### 🤖 Biểu đồ Được đề xuất")
    for i, rec in enumerate(recommendations):
        with st.container():
            st.markdown(f"""
            <div class="chart-option">
                <h4>{rec['icon']} {rec['type']}</h4>
                <p>{rec['description']}</p>
                <small>Độ tin cậy: {rec['confidence']:.0%} | Màu sắc: {rec['color_scheme']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Sử dụng {rec['type']}", key=f"rec_{i}"):
                st.session_state.selected_chart = rec
                st.session_state.auto_x = rec['suggested_x']
                st.session_state.auto_y = rec['suggested_y']
                st.session_state.auto_palette = rec['color_scheme']

with col2:
    st.subheader("⚙️ Cấu hình Biểu đồ")
    
    # Lựa chọn loại biểu đồ
    chart_types = ["Biểu Đồ Phân Tán", "Biểu Đồ Cột", "Biểu Đồ Hộp", "Chuỗi Thời Gian", "Bản Đồ Nhiệt Tương Quan", "Biểu Đồ Tròn", "Biểu Đồ Violin", "Biểu Đồ Tần Suất"]
    selected_chart_type = st.selectbox("📊 Loại Biểu đồ:", chart_types, 
                                      index=chart_types.index(st.session_state.get('selected_chart', {}).get('type', 'Biểu Đồ Phân Tán')) if st.session_state.get('selected_chart', {}).get('type') in chart_types else 0)
    
    # Lựa chọn cột
    col_a, col_b = st.columns(2)
    with col_a:
        x_axis = st.selectbox("Trục X:", df.columns.tolist(), 
                             index=df.columns.tolist().index(st.session_state.get('auto_x', df.columns[0])) if st.session_state.get('auto_x') in df.columns else 0)
    
    with col_b:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            y_axis = st.selectbox("Trục Y:", numeric_cols,
                                 index=numeric_cols.index(st.session_state.get('auto_y', numeric_cols[0])) if st.session_state.get('auto_y') in numeric_cols else 0)
        else:
            y_axis = st.selectbox("Trục Y:", df.columns.tolist())
    
    # Màu sắc và nhóm
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    color_by = st.selectbox("🎨 Màu theo:", ["Không"] + categorical_cols)
    
    # Lựa chọn bảng màu
    palette_name = st.selectbox("🎨 Bảng Màu:", list(COLOR_PALETTES.keys()),
                               index=list(COLOR_PALETTES.keys()).index(st.session_state.get('auto_palette', 'Xanh Chuyên Nghiệp')))
    
    # Hiển thị xem trước bảng màu
    st.markdown("**Xem trước Màu sắc:**")
    palette_html = '<div class="color-palette">'
    for color in COLOR_PALETTES[palette_name][:8]:
        palette_html += f'<div class="color-box" style="background-color: {color}"></div>'
    palette_html += '</div>'
    st.markdown(palette_html, unsafe_allow_html=True)
    
    # Prompt thiết kế tùy chỉnh
    custom_prompt = st.text_area("✨ Hướng dẫn Thiết kế Bổ sung:", 
                                placeholder="ví dụ: thêm đường xu hướng, sử dụng thang log, làm nổi bật ngoại lệ, thêm chú thích...")
    
    # Tạo biểu đồ
    if st.button("🚀 Tạo Biểu đồ", type="primary"):
        with st.spinner("Đang tạo trực quan hóa của bạn..."):
            fig, code = create_enhanced_chart(
                selected_chart_type, df, x_axis, y_axis, 
                color_by if color_by != "Không" else None,
                palette_name, custom_prompt
            )
            
            if fig:
                st.session_state.current_fig = fig
                st.session_state.current_code = code
                st.session_state.chart_generated = True

# Hiển thị biểu đồ đã tạo
if st.session_state.get('chart_generated', False):
    st.subheader("📊 Trực quan hóa Đã tạo")
    st.plotly_chart(st.session_state.current_fig, use_container_width=True)
    
    # Insights được tạo bởi AI
    with st.spinner("Đang tạo insights AI..."):
        insight_prompt = f"""
        Phân tích biểu đồ {selected_chart_type} này hiển thị {x_axis} vs {y_axis} từ bộ dữ liệu.
        
        Thông tin bộ dữ liệu:
        - Kích thước: {df.shape}
        - Các cột: {list(df.columns)}
        
        Cung cấp 3-5 insights cụ thể về:
        1. Các mẫu hoặc xu hướng chính có thể nhìn thấy
        2. Ngoại lệ hoặc điểm dữ liệu thú vị
        3. Ý nghĩa kinh doanh
        4. Đề xuất phân tích tiếp theo
        
        Hãy cụ thể và có thể hành động. Bao gồm các con số thực tế khi có thể.
        """
        response = llm.invoke(insight_prompt)
        insights = response.content if hasattr(response, 'content') else str(response)
        
        st.markdown(f"""
        <div class="recommendation-card">
            <h3>🧠 Insights Được tạo bởi AI</h3>
            {insights}
        </div>
        """, unsafe_allow_html=True)
    
    # Tùy chọn mã và xuất
    with st.expander("📋 Xem Mã Đã tạo", expanded=False):
        st.code(st.session_state.current_code, language="python")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("💾 Lưu Biểu đồ"):
            add_chart_card(dataset_id, f"Biểu đồ: {selected_chart_type}", insights, st.session_state.current_code)
            st.success("✅ Biểu đồ đã lưu vào lịch sử!")
    
    with col_b:
        if st.button("📥 Tải PNG"):
            st.info("Tải PNG sẽ được triển khai ở đây")
    
    with col_c:
        if st.button("📊 Tạo Dashboard"):
            st.info("Tạo dashboard sẽ được triển khai ở đây")

# Thư viện biểu đồ và lịch sử
st.subheader("🖼️ Thư viện Biểu đồ & Cảm hứng")
with st.expander("Xem Ví dụ Biểu đồ", expanded=False):
    example_charts = [
        {"name": "Xu hướng Bán hàng", "type": "Chuỗi Thời Gian", "description": "Hiệu suất bán hàng hàng tháng theo thời gian"},
        {"name": "Phân khúc Khách hàng", "type": "Biểu Đồ Tròn", "description": "Phân phối các loại khách hàng"},
        {"name": "So sánh Hiệu suất", "type": "Biểu Đồ Hộp", "description": "So sánh các chỉ số qua các phòng ban"},
        {"name": "Phân tích Tương quan", "type": "Bản Đồ Nhiệt", "description": "Mối quan hệ giữa các biến"}
    ]
    
    for chart in example_charts:
        st.markdown(f"**{chart['name']}** ({chart['type']}): {chart['description']}")

# Gợi ý điều hướng
st.markdown("---")
st.info("💡 **Mẹo Chuyên Nghiệp:** Sử dụng các đề xuất AI để bắt đầu nhanh chóng, sau đó tùy chỉnh với sở thích thiết kế của riêng bạn!")