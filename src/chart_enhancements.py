import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.models.llms import load_llm
from src.utils import get_all_datasets, get_dataset, safe_read_csv, add_chart_card
import numpy as np
import warnings
import time
import base64
warnings.filterwarnings('ignore')

st.set_page_config(page_title="📈 Biểu Đồ Thông Minh Nâng Cao", layout="wide")

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
    .success-message {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .loading-spinner {
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chart-header"><h1>📈 Công Cụ Tạo Biểu Đồ Thông Minh AI</h1><p>Nhận đề xuất biểu đồ thông minh và tạo ra các trực quan hóa tuyệt đẹp</p></div>', unsafe_allow_html=True)

llm = load_llm("gpt-3.5-turbo")

# Enhanced color palettes
COLOR_PALETTES = {
    "Xanh Chuyên Nghiệp": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "Sống Động": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"],
    "Doanh Nghiệp": ["#2C3E50", "#3498DB", "#E74C3C", "#F39C12", "#27AE60", "#8E44AD", "#16A085", "#E67E22", "#34495E", "#1ABC9C"],
    "Hoàng Hôn": ["#FF6B35", "#F7931E", "#FFD23F", "#06FFA5", "#118AB2", "#073B4C", "#E63946", "#F77F00", "#FCBF49", "#003566"],
    "Đại Dương": ["#0077BE", "#00A8CC", "#0FA3B1", "#B5E2FA", "#F9E784", "#F8AD9D", "#F4975A", "#E8871E", "#DA627D", "#A53860"],
    "Tự Nhiên": ["#8FBC8F", "#32CD32", "#228B22", "#006400", "#9ACD32", "#ADFF2F", "#7CFC00", "#7FFF00", "#98FB98", "#90EE90"],
    "Gradient Tím": ["#9C27B0", "#8E24AA", "#7B1FA2", "#673AB7", "#5E35B1", "#512DA8", "#4527A0", "#3F51B5", "#3949AB", "#303F9F"]
}

# Load datasets
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
    """Enhanced AI-powered chart recommendations"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month'])]
    
    recommendations = []
    
    if len(numeric_cols) >= 2:
        recommendations.append({
            "type": "Biểu Đồ Phân Tán",
            "description": f"Khám phá mối quan hệ giữa {numeric_cols[0]} và {numeric_cols[1]}",
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
            "description": f"So sánh phân phối {numeric_cols[0]} theo {categorical_cols[0]}",
            "confidence": 0.8,
            "suggested_x": categorical_cols[0],
            "suggested_y": numeric_cols[0],
            "icon": "📦",
            "color_scheme": "Sống Động"
        })
        
        recommendations.append({
            "type": "Biểu Đồ Cột",
            "description": f"Hiển thị giá trị trung bình {numeric_cols[0]} theo {categorical_cols[0]}",
            "confidence": 0.75,
            "suggested_x": categorical_cols[0],
            "suggested_y": numeric_cols[0],
            "icon": "📊",
            "color_scheme": "Doanh Nghiệp"
        })
    
    if datetime_cols and numeric_cols:
        recommendations.append({
            "type": "Chuỗi Thời Gian",
            "description": f"Theo dõi thay đổi {numeric_cols[0]} theo {datetime_cols[0]}",
            "confidence": 0.95,
            "suggested_x": datetime_cols[0],
            "suggested_y": numeric_cols[0],
            "icon": "📈",
            "color_scheme": "Đại Dương"
        })
    
    if categorical_cols:
        recommendations.append({
            "type": "Biểu Đồ Tròn",
            "description": f"Hiển thị tỷ lệ của {categorical_cols[0]}",
            "confidence": 0.7,
            "suggested_x": categorical_cols[0],
            "suggested_y": "Đếm",
            "icon": "🥧",
            "color_scheme": "Tự Nhiên"
        })
    
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    return recommendations[:6]

def create_enhanced_chart(chart_type, df, x_col, y_col, color_col=None, palette="Xanh Chuyên Nghiệp", custom_prompt=""):
    """Create enhanced charts with proper data validation and error handling"""
    try:
        colors = COLOR_PALETTES[palette]
        fig = None
        code = ""
        
        # Data validation
        if x_col not in df.columns or (y_col != "Đếm" and y_col not in df.columns):
            st.error(f"❌ Cột không tồn tại: {x_col} hoặc {y_col}")
            return None, ""
        
        # Clean data
        working_df = df.copy()
        
        if chart_type == "Biểu Đồ Phân Tán":
            # Ensure both columns are numeric
            if pd.api.types.is_numeric_dtype(working_df[x_col]) and pd.api.types.is_numeric_dtype(working_df[y_col]):
                fig = px.scatter(
                    working_df, 
                    x=x_col, 
                    y=y_col, 
                    color=color_col if color_col and color_col != "Không" else None,
                    color_discrete_sequence=colors,
                    title=f"Biểu Đồ Phân Tán: {x_col} vs {y_col}",
                    template="plotly_white",
                    hover_data=[x_col, y_col]
                )
                
                code = f"""
import plotly.express as px

fig = px.scatter(df, x='{x_col}', y='{y_col}', 
                color='{color_col}' if '{color_col}' != 'Không' and '{color_col}' else None,
                color_discrete_sequence={colors},
                title="Biểu Đồ Phân Tán: {x_col} vs {y_col}",
                template="plotly_white")
fig.show()
"""
            else:
                st.error("❌ Biểu đồ phân tán cần cả hai cột đều là số")
                return None, ""
        
        elif chart_type == "Biểu Đồ Cột":
            if working_df[x_col].dtype == 'object' or pd.api.types.is_categorical_dtype(working_df[x_col]):
                # Group categorical data
                if pd.api.types.is_numeric_dtype(working_df[y_col]):
                    agg_df = working_df.groupby(x_col)[y_col].agg(['mean', 'count']).reset_index()
                    agg_df.columns = [x_col, f'Mean_{y_col}', 'Count']
                    
                    fig = px.bar(
                        agg_df, 
                        x=x_col, 
                        y=f'Mean_{y_col}',
                        color=x_col,
                        color_discrete_sequence=colors,
                        title=f"Biểu Đồ Cột: Trung bình {y_col} theo {x_col}",
                        template="plotly_white",
                        text=f'Mean_{y_col}'
                    )
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                else:
                    # Count plot for categorical y
                    value_counts = working_df[x_col].value_counts().reset_index()
                    value_counts.columns = [x_col, 'Count']
                    
                    fig = px.bar(
                        value_counts, 
                        x=x_col, 
                        y='Count',
                        color=x_col,
                        color_discrete_sequence=colors,
                        title=f"Biểu Đồ Cột: Số lượng theo {x_col}",
                        template="plotly_white",
                        text='Count'
                    )
                    fig.update_traces(texttemplate='%{text}', textposition='outside')
            else:
                # Numeric x-axis - create bins
                working_df['binned'] = pd.cut(working_df[x_col], bins=10)
                agg_df = working_df.groupby('binned')[y_col].mean().reset_index()
                
                fig = px.bar(
                    agg_df, 
                    x='binned', 
                    y=y_col,
                    color_discrete_sequence=colors,
                    title=f"Biểu Đồ Cột: {y_col} theo nhóm {x_col}",
                    template="plotly_white"
                )
            
            # Update layout for better spacing
            fig.update_layout(
                xaxis={'categoryorder': 'total descending'},
                bargap=0.2,
                bargroupgap=0.1
            )
            
            code = f"""
import plotly.express as px

if df['{x_col}'].dtype == 'object':
    agg_df = df.groupby('{x_col}')['{y_col}'].mean().reset_index()
    fig = px.bar(agg_df, x='{x_col}', y='{y_col}',
                color='{x_col}',
                color_discrete_sequence={colors},
                title="Biểu Đồ Cột: {y_col} theo {x_col}",
                template="plotly_white")
else:
    df['binned'] = pd.cut(df['{x_col}'], bins=10)
    agg_df = df.groupby('binned')['{y_col}'].mean().reset_index()
    fig = px.bar(agg_df, x='binned', y='{y_col}',
                color_discrete_sequence={colors},
                title="Biểu Đồ Cột: {y_col} theo {x_col}",
                template="plotly_white")

fig.update_layout(bargap=0.2, bargroupgap=0.1)
fig.show()
"""
        
        elif chart_type == "Biểu Đồ Hộp":
            fig = px.box(
                working_df, 
                x=x_col, 
                y=y_col, 
                color=color_col if color_col and color_col != "Không" else None,
                color_discrete_sequence=colors,
                title=f"Biểu Đồ Hộp: Phân phối {y_col} theo {x_col}",
                template="plotly_white",
                points="outliers"
            )
            
            code = f"""
import plotly.express as px

fig = px.box(df, x='{x_col}', y='{y_col}', 
            color='{color_col}' if '{color_col}' != 'Không' and '{color_col}' else None,
            color_discrete_sequence={colors},
            title="Biểu Đồ Hộp: Phân phối {y_col} theo {x_col}",
            template="plotly_white",
            points="outliers")
fig.show()
"""
        
        elif chart_type == "Chuỗi Thời Gian":
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(working_df[x_col]):
                working_df[x_col] = pd.to_datetime(working_df[x_col], errors='coerce')
            
            # Remove invalid dates
            working_df = working_df.dropna(subset=[x_col])
            
            if len(working_df) == 0:
                st.error("❌ Không có dữ liệu thời gian hợp lệ")
                return None, ""
            
            fig = px.line(
                working_df, 
                x=x_col, 
                y=y_col,
                color=color_col if color_col and color_col != "Không" else None,
                color_discrete_sequence=colors,
                title=f"Chuỗi Thời Gian: {y_col} theo {x_col}",
                template="plotly_white",
                markers=True
            )
            
            code = f"""
import plotly.express as px
import pandas as pd

df['{x_col}'] = pd.to_datetime(df['{x_col}'], errors='coerce')
df_clean = df.dropna(subset=['{x_col}'])

fig = px.line(df_clean, x='{x_col}', y='{y_col}',
             color='{color_col}' if '{color_col}' != 'Không' and '{color_col}' else None,
             color_discrete_sequence={colors},
             title="Chuỗi Thời Gian: {y_col} theo {x_col}",
             template="plotly_white",
             markers=True)
fig.show()
"""
        
        elif chart_type == "Bản Đồ Nhiệt Tương Quan":
            numeric_df = working_df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) < 2:
                st.error("❌ Cần ít nhất 2 cột số để tạo bản đồ nhiệt tương quan")
                return None, ""
            
            corr_matrix = numeric_df.corr()
            
            # Enhanced color scale options
            color_scales = {
                "Xanh Chuyên Nghiệp": "RdBu_r",
                "Sống Động": "Viridis",
                "Doanh Nghiệp": "Blues",
                "Hoàng Hôn": "Sunset",
                "Đại Dương": "thermal",
                "Tự Nhiên": "Greens",
                "Gradient Tím": "Purples"
            }
            
            color_scale = color_scales.get(palette, "RdBu_r")
            
            fig = px.imshow(
                corr_matrix, 
                color_continuous_scale=color_scale,
                title="Bản Đồ Nhiệt Tương Quan",
                template="plotly_white",
                aspect="auto",
                text_auto=True
            )
            
            fig.update_layout(
                width=800,
                height=600,
                xaxis_title="Biến",
                yaxis_title="Biến"
            )
            
            code = f"""
import plotly.express as px

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

fig = px.imshow(corr_matrix, 
               color_continuous_scale="{color_scale}",
               title="Bản Đồ Nhiệt Tương Quan",
               template="plotly_white",
               text_auto=True)
fig.show()
"""
        
        elif chart_type == "Biểu Đồ Tròn":
            # Enhanced pie chart logic
            if working_df[x_col].dtype == 'object' or pd.api.types.is_categorical_dtype(working_df[x_col]):
                value_counts = working_df[x_col].value_counts().head(10)
                
                fig = px.pie(
                    values=value_counts.values, 
                    names=value_counts.index,
                    color_discrete_sequence=colors,
                    title=f"Phân phối của {x_col}",
                    template="plotly_white",
                    hole=0.3  # Donut chart for modern look
                )
                
                fig.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Số lượng: %{value}<br>Tỷ lệ: %{percent}<extra></extra>'
                )
            else:
                # For numeric data, create bins
                working_df['binned'] = pd.cut(working_df[x_col], bins=5)
                value_counts = working_df['binned'].value_counts()
                
                fig = px.pie(
                    values=value_counts.values, 
                    names=[str(x) for x in value_counts.index],
                    color_discrete_sequence=colors,
                    title=f"Phân phối nhóm của {x_col}",
                    template="plotly_white",
                    hole=0.3
                )
            
            code = f"""
import plotly.express as px

if df['{x_col}'].dtype == 'object':
    value_counts = df['{x_col}'].value_counts().head(10)
    fig = px.pie(values=value_counts.values, names=value_counts.index,
                color_discrete_sequence={colors},
                title="Phân phối của {x_col}",
                template="plotly_white",
                hole=0.3)
else:
    df['binned'] = pd.cut(df['{x_col}'], bins=5)
    value_counts = df['binned'].value_counts()
    fig = px.pie(values=value_counts.values, names=value_counts.index,
                color_discrete_sequence={colors},
                title="Phân phối nhóm của {x_col}",
                template="plotly_white",
                hole=0.3)

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
"""
        
        # Apply custom styling to all charts
        if fig:
            fig.update_layout(
                font=dict(size=12, family="Arial, sans-serif"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title=dict(x=0.5, font=dict(size=16, color='#2c3e50')),
                margin=dict(t=60, l=60, r=60, b=60),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            
            # Add custom styling based on prompt
            if custom_prompt:
                fig.add_annotation(
                    text=f"Tùy chỉnh: {custom_prompt}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=10, color="gray")
                )
        
        return fig, code
        
    except Exception as e:
        st.error(f"❌ Lỗi khi tạo biểu đồ: {str(e)}")
        return None, ""

def save_chart_to_session(fig, code, chart_type, description):
    """Save chart data to session state"""
    if 'saved_charts' not in st.session_state:
        st.session_state.saved_charts = []
    
    chart_data = {
        'figure': fig,
        'code': code,
        'type': chart_type,
        'description': description,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    st.session_state.saved_charts.append(chart_data)

def download_chart_as_png(fig, filename):
    """Convert plotly figure to PNG and provide download"""
    try:
        # Convert to PNG bytes
        img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
        
        # Encode to base64 for download
        b64 = base64.b64encode(img_bytes).decode()
        
        # Create download link
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">📥 Tải PNG</a>'
        return href, img_bytes
    except Exception as e:
        st.error(f"❌ Lỗi khi tạo PNG: {str(e)}")
        return None, None

# Main interface
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
                st.rerun()

with col2:
    st.subheader("⚙️ Cấu hình Biểu đồ")
    
    # Chart type selection
    chart_types = ["Biểu Đồ Phân Tán", "Biểu Đồ Cột", "Biểu Đồ Hộp", "Chuỗi Thời Gian", "Bản Đồ Nhiệt Tương Quan", "Biểu Đồ Tròn"]
    selected_chart_type = st.selectbox("📊 Loại Biểu đồ:", chart_types, 
                                      index=chart_types.index(st.session_state.get('selected_chart', {}).get('type', 'Biểu Đồ Phân Tán')) if st.session_state.get('selected_chart', {}).get('type') in chart_types else 0)
    
    # Column selection with validation
    col_a, col_b = st.columns(2)
    with col_a:
        x_axis = st.selectbox("Trục X:", df.columns.tolist(), 
                             index=df.columns.tolist().index(st.session_state.get('auto_x', df.columns[0])) if st.session_state.get('auto_x') in df.columns else 0)
    
    with col_b:
        if selected_chart_type in ["Biểu Đồ Tròn"]:
            y_axis = "Đếm"
            st.markdown("**Trục Y:** Đếm (tự động)")
        else:
            available_cols = df.columns.tolist()
            if selected_chart_type in ["Biểu Đồ Phân Tán", "Biểu Đồ Hộp", "Chuỗi Thời Gian"]:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    available_cols = numeric_cols
            
            y_axis = st.selectbox("Trục Y:", available_cols,
                                 index=available_cols.index(st.session_state.get('auto_y', available_cols[0])) if st.session_state.get('auto_y') in available_cols else 0)
    
    # Color grouping
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    color_by = st.selectbox("🎨 Màu theo:", ["Không"] + categorical_cols)
    
    # Color palette selection
    palette_name = st.selectbox("🎨 Bảng Màu:", list(COLOR_PALETTES.keys()),
                               index=list(COLOR_PALETTES.keys()).index(st.session_state.get('auto_palette', 'Xanh Chuyên Nghiệp')))
    
    # Display color preview
    st.markdown("**Xem trước Màu sắc:**")
    palette_html = '<div class="color-palette">'
    for color in COLOR_PALETTES[palette_name][:8]:
        palette_html += f'<div class="color-box" style="background-color: {color}"></div>'
    palette_html += '</div>'
    st.markdown(palette_html, unsafe_allow_html=True)
    
    # Custom design prompt
    custom_prompt = st.text_area("✨ Hướng dẫn Thiết kế Bổ sung:", 
                                placeholder="ví dụ: thêm đường xu hướng, sử dụng thang log, làm nổi bật ngoại lệ, thêm chú thích...")
    
    # Generate chart button with loading
    if st.button("🚀 Tạo Biểu đồ", type="primary"):
        # Show loading
        loading_placeholder = st.empty()
        with loading_placeholder:
            st.markdown("""
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <span>Đang tạo trực quan hóa của bạn...</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Simulate processing time
        time.sleep(1)
        
        try:
            fig, code = create_enhanced_chart(
                selected_chart_type, df, x_axis, y_axis, 
                color_by if color_by != "Không" else None,
                palette_name, custom_prompt
            )
            
            if fig:
                st.session_state.current_fig = fig
                st.session_state.current_code = code
                st.session_state.chart_generated = True
                st.session_state.current_chart_type = selected_chart_type
                st.session_state.current_description = f"{selected_chart_type} hiển thị {x_axis} vs {y_axis}"
                
                # Clear loading
                loading_placeholder.empty()
                
                # Show success message
                st.markdown("""
                <div class="success-message">
                    ✅ Biểu đồ đã được tạo thành công!
                </div>
                """, unsafe_allow_html=True)
                
                st.rerun()
            else:
                loading_placeholder.empty()
                st.error("❌ Không thể tạo biểu đồ. Vui lòng kiểm tra dữ liệu và thử lại.")
                
        except Exception as e:
            loading_placeholder.empty()
            st.error(f"❌ Lỗi khi tạo biểu đồ: {str(e)}")

# Display generated chart
if st.session_state.get('chart_generated', False):
    st.subheader("📊 Trực quan hóa Đã tạo")
    
    # Chart display with enhanced layout
    chart_col, controls_col = st.columns([3, 1])
    
    with chart_col:
        # Display the chart
        st.plotly_chart(st.session_state.current_fig, use_container_width=True, key="main_chart")
        
        # AI Insights generation
        with st.spinner("🔍 Đang tạo insights AI..."):
            insight_prompt = f"""
            Phân tích biểu đồ {st.session_state.get('current_chart_type', 'này')} hiển thị {x_axis} vs {y_axis} từ bộ dữ liệu.
            
            Thông tin bộ dữ liệu:
            - Kích thước: {df.shape}
            - Loại biểu đồ: {st.session_state.get('current_chart_type')}
            - Trục X: {x_axis} ({df[x_axis].dtype})
            - Trục Y: {y_axis if y_axis != "Đếm" else "Số lượng"}
            
            Cung cấp 3-5 insights cụ thể về:
            1. Các mẫu hoặc xu hướng chính có thể nhìn thấy
            2. Ngoại lệ hoặc điểm dữ liệu thú vị
            3. Ý nghĩa kinh doanh tiềm năng
            4. Đề xuất phân tích tiếp theo
            
            Hãy cụ thể và có thể hành động. Bao gồm các con số thực tế khi có thể.
            Trả lời bằng markdown với format đẹp.
            """
            
            insights = llm.invoke(insight_prompt)
            
            st.markdown("### 🧠 Insights Được tạo bởi AI")
            st.markdown(f"""
            <div class="recommendation-card">
                {insights}
            </div>
            """, unsafe_allow_html=True)
    
    with controls_col:
        st.markdown("#### 🎨 Tùy chọn Biểu đồ")
        
        # Chart enhancement options
        if st.button("🔄 Tạo lại với cài đặt mới", use_container_width=True):
            # Clear current chart to force regeneration
            if 'chart_generated' in st.session_state:
                del st.session_state['chart_generated']
            st.rerun()
        
        st.markdown("#### 💾 Lưu & Xuất")
        
        # Save chart functionality - FIXED
        if st.button("💾 Lưu Biểu đồ", use_container_width=True, key="save_chart_btn"):
            try:
                # Save to database
                add_chart_card(
                    dataset_id, 
                    f"Biểu đồ: {st.session_state.get('current_chart_type')}", 
                    st.session_state.get('current_description', ''), 
                    st.session_state.get('current_code', '')
                )
                
                # Save to session for immediate access
                save_chart_to_session(
                    st.session_state.current_fig,
                    st.session_state.current_code,
                    st.session_state.get('current_chart_type'),
                    st.session_state.get('current_description')
                )
                
                st.success("✅ Biểu đồ đã được lưu thành công!")
                time.sleep(1)
                
            except Exception as e:
                st.error(f"❌ Lỗi khi lưu biểu đồ: {str(e)}")
        
        # Download PNG functionality - FIXED
        if st.button("📥 Tải PNG", use_container_width=True, key="download_png_btn"):
            try:
                with st.spinner("🔄 Đang tạo file PNG..."):
                    filename = f"chart_{int(time.time())}"
                    download_link, img_bytes = download_chart_as_png(st.session_state.current_fig, filename)
                    
                    if download_link:
                        # Provide direct download
                        st.download_button(
                            label="📥 Tải PNG",
                            data=img_bytes,
                            file_name=f"{filename}.png",
                            mime="image/png",
                            key="png_download_btn"
                        )
                        st.success("✅ File PNG đã sẵn sàng để tải!")
                    else:
                        st.error("❌ Không thể tạo file PNG")
                        
            except Exception as e:
                st.error(f"❌ Lỗi khi tạo PNG: {str(e)}")
        
        # Create Dashboard functionality - FIXED  
        if st.button("📊 Tạo Dashboard", use_container_width=True, key="create_dashboard_btn"):
            try:
                with st.spinner("🔄 Đang tạo dashboard..."):
                    # Create a multi-chart dashboard
                    dashboard_fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=[
                            f'Biểu đồ chính: {st.session_state.get("current_chart_type")}',
                            'Thống kê tóm tắt',
                            'Phân phối dữ liệu',
                            'Xu hướng theo thời gian'
                        ],
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]],
                        vertical_spacing=0.12,
                        horizontal_spacing=0.1
                    )
                    
                    # Add main chart (simplified version)
                    if st.session_state.get('current_chart_type') == 'Biểu Đồ Cột':
                        if df[x_axis].dtype == 'object':
                            agg_data = df.groupby(x_axis)[y_axis].mean().head(5)
                            dashboard_fig.add_trace(
                                go.Bar(x=agg_data.index, y=agg_data.values, name="Chính"),
                                row=1, col=1
                            )
                    
                    # Add summary statistics
                    numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
                    if len(numeric_cols) > 0:
                        summary_data = df[numeric_cols].mean()
                        dashboard_fig.add_trace(
                            go.Bar(x=summary_data.index, y=summary_data.values, name="Trung bình"),
                            row=1, col=2
                        )
                    
                    # Add distribution chart
                    if len(numeric_cols) > 0:
                        dashboard_fig.add_trace(
                            go.Histogram(x=df[numeric_cols[0]], name="Phân phối"),
                            row=2, col=1
                        )
                    
                    # Add trend if date column exists
                    date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time'])]
                    if date_cols and len(numeric_cols) > 0:
                        df_sorted = df.sort_values(date_cols[0])
                        dashboard_fig.add_trace(
                            go.Scatter(x=df_sorted[date_cols[0]], y=df_sorted[numeric_cols[0]], 
                                     mode='lines', name="Xu hướng"),
                            row=2, col=2
                        )
                    
                    dashboard_fig.update_layout(
                        height=800,
                        title_text=f"Dashboard: {dataset[1]}",
                        showlegend=True,
                        template="plotly_white"
                    )
                    
                    # Store dashboard
                    st.session_state.dashboard_fig = dashboard_fig
                    
                st.success("✅ Dashboard đã được tạo!")
                
                # Show dashboard
                st.plotly_chart(st.session_state.dashboard_fig, use_container_width=True, key="dashboard_chart")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi tạo dashboard: {str(e)}")
        
        # Chart statistics
        st.markdown("#### 📈 Thống kê Biểu đồ")
        if x_axis in df.columns:
            col_stats = {
                "Dữ liệu": f"{len(df)} điểm",
                "Kiểu X": str(df[x_axis].dtype),
                "Thiếu": f"{df[x_axis].isnull().sum()}",
                "Duy nhất": f"{df[x_axis].nunique()}"
            }
            
            for key, value in col_stats.items():
                st.metric(key, value)
    
    # Code display section
    with st.expander("📋 Xem Code được Tạo", expanded=False):
        st.markdown("### 🐍 Python Code")
        st.code(st.session_state.get('current_code', ''), language="python")
        
        st.markdown("### 📝 Hướng dẫn Sử dụng")
        st.markdown("""
        **Để sử dụng code này:**
        1. Đảm bảo bạn đã cài đặt: `pip install plotly pandas`
        2. Load dữ liệu của bạn vào DataFrame tên `df`
        3. Copy và paste code trên
        4. Chạy để xem biểu đồ
        
        **Tùy chỉnh thêm:**
        - Thay đổi `color_discrete_sequence` để đổi màu
        - Điều chỉnh `template` để đổi theme
        - Thêm `hover_data` để hiển thị thêm thông tin khi hover
        """)

# Chart History Section
st.markdown("---")
st.subheader("🖼️ Lịch sử Biểu đồ")

if 'saved_charts' in st.session_state and st.session_state.saved_charts:
    st.markdown(f"**📊 Bạn đã tạo {len(st.session_state.saved_charts)} biểu đồ trong phiên này**")
    
    # Display saved charts in tabs
    chart_tabs = st.tabs([f"{chart['type']} - {chart['timestamp']}" for chart in st.session_state.saved_charts[-3:]])
    
    for i, chart in enumerate(st.session_state.saved_charts[-3:]):
        with chart_tabs[i]:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.plotly_chart(chart['figure'], use_container_width=True, key=f"history_chart_{i}")
            
            with col2:
                st.markdown(f"**📅 Tạo lúc:** {chart['timestamp']}")
                st.markdown(f"**📊 Loại:** {chart['type']}")
                st.markdown(f"**📝 Mô tả:** {chart['description']}")
                
                if st.button(f"🔄 Tải lại", key=f"reload_chart_{i}"):
                    st.session_state.current_fig = chart['figure']
                    st.session_state.current_code = chart['code']
                    st.session_state.chart_generated = True
                    st.rerun()
else:
    st.info("📊 Chưa có biểu đồ nào được lưu trong phiên này. Tạo biểu đồ đầu tiên của bạn!")

# Tips and examples section
st.markdown("---")
st.subheader("💡 Mẹo & Ví dụ")

with st.expander("🎯 Mẹo Tạo Biểu đồ Hiệu quả", expanded=False):
    st.markdown("""
    ### 📊 Chọn Loại Biểu đồ Phù hợp
    
    **🔵 Biểu Đồ Phân Tán:**
    - Sử dụng khi: Muốn tìm mối quan hệ giữa 2 biến số
    - Tốt nhất cho: Dữ liệu liên tục, phát hiện xu hướng
    - Ví dụ: Mối quan hệ giữa tuổi và thu nhập
    
    **📊 Biểu Đồ Cột:**
    - Sử dụng khi: So sánh các danh mục
    - Tốt nhất cho: Dữ liệu phân loại, hiển thị tổng/trung bình
    - Ví dụ: Doanh thu theo tháng, số lượng theo khu vực
    
    **📦 Biểu Đồ Hộp:**
    - Sử dụng khi: Muốn xem phân phối và ngoại lệ
    - Tốt nhất cho: So sánh phân phối giữa các nhóm
    - Ví dụ: Điểm thi theo lớp, lương theo phòng ban
    
    **📈 Chuỗi Thời Gian:**
    - Sử dụng khi: Dữ liệu có yếu tố thời gian
    - Tốt nhất cho: Phát hiện xu hướng, mùa vụ, chu kỳ
    - Ví dụ: Giá cổ phiếu theo thời gian, doanh số theo ngày
    
    **🔥 Bản Đồ Nhiệt:**
    - Sử dụng khi: Muốn xem tương quan giữa nhiều biến
    - Tốt nhất cho: Phát hiện mối quan hệ ẩn
    - Ví dụ: Tương quan giữa các chỉ số KPI
    
    **🥧 Biểu Đồ Tròn:**
    - Sử dụng khi: Hiển thị tỷ lệ phần trăm
    - Tốt nhất cho: Ít hơn 7 danh mục
    - Ví dụ: Thị phần, phân bố khách hàng theo khu vực
    """)

with st.expander("🎨 Hướng dẫn Chọn Màu sắc", expanded=False):
    st.markdown("""
    ### 🎨 Bảng Màu và Ứng dụng
    
    **🔵 Xanh Chuyên Nghiệp:** Phù hợp cho báo cáo doanh nghiệp, thuyết trình
    **🌈 Sống Động:** Tốt cho dashboard tương tác, dữ liệu tiêu dùng
    **🏢 Doanh Nghiệp:** Thích hợp cho báo cáo tài chính, KPI
    **🌅 Hoàng Hôn:** Đẹp cho dữ liệu marketing, sáng tạo
    **🌊 Đại Dương:** Phù hợp cho dữ liệu môi trường, sức khỏe
    **🌿 Tự Nhiên:** Tốt cho dữ liệu nông nghiệp, xanh
    **💜 Gradient Tím:** Hiện đại cho tech, startup
    
    ### 💡 Mẹo Thiết kế
    - Sử dụng màu tương phản để làm nổi bật điểm quan trọng
    - Tránh dùng quá nhiều màu trong một biểu đồ (tối đa 5-7 màu)
    - Đảm bảo màu sắc phù hợp với thương hiệu công ty
    - Kiểm tra độ tương phản cho người khiếm thị màu
    """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📈 VizGenie-GPT Biểu đồ Thông minh**")
    st.caption("Tạo trực quan hóa chuyên nghiệp với AI")

with col2:
    if st.session_state.get('chart_generated'):
        st.markdown("**✅ Trạng thái**")
        st.caption("Biểu đồ đã sẵn sàng")
    else:
        st.markdown("**⏳ Trạng thái**") 
        st.caption("Sẵn sàng tạo biểu đồ")

with col3:
    st.markdown("**🎯 Mẹo**")
    st.caption("Thử các bảng màu và loại biểu đồ khác nhau!")

# Auto-clear old session data to prevent memory issues
if len(st.session_state.get('saved_charts', [])) > 10:
    st.session_state.saved_charts = st.session_state.saved_charts[-10:]