import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Enhanced color schemes for chart enhancements
ENHANCED_COLOR_SCHEMES = {
    "Professional Blue": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "Vibrant": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"],
    "Corporate": ["#2C3E50", "#3498DB", "#E74C3C", "#F39C12", "#27AE60", "#8E44AD", "#16A085", "#E67E22", "#34495E", "#1ABC9C"],
    "Sunset": ["#FF6B35", "#F7931E", "#FFD23F", "#06FFA5", "#118AB2", "#073B4C", "#E63946", "#F77F00", "#FCBF49", "#003566"],
    "Ocean": ["#0077BE", "#00A8CC", "#0FA3B1", "#B5E2FA", "#F9E784", "#F8AD9D", "#F4975A", "#E8871E", "#DA627D", "#A53860"],
    "Nature": ["#8FBC8F", "#32CD32", "#228B22", "#006400", "#9ACD32", "#ADFF2F", "#7CFC00", "#7FFF00", "#98FB98", "#90EE90"],
    "Purple Gradient": ["#9C27B0", "#8E24AA", "#7B1FA2", "#673AB7", "#5E35B1", "#512DA8", "#4527A0", "#3F51B5", "#3949AB", "#303F9F"]
}

def smart_patch_chart_code(original_code: str, df: pd.DataFrame) -> str:
    """
    Intelligently enhance matplotlib/seaborn code with professional styling
    """
    try:
        # Basic enhancements that work for most chart types
        enhancements = []
        
        # Add style and figure setup
        enhancements.append("import matplotlib.pyplot as plt")
        enhancements.append("import seaborn as sns")
        enhancements.append("plt.style.use('default')")
        enhancements.append("plt.rcParams['figure.facecolor'] = 'white'")
        enhancements.append("plt.rcParams['axes.facecolor'] = 'white'")
        enhancements.append("plt.rcParams['font.size'] = 10")
        
        # Add the original code
        enhancements.append(original_code)
        
        # Add professional finishing touches
        enhancements.append("plt.tight_layout()")
        enhancements.append("plt.grid(True, alpha=0.3)")
        
        # If it's a simple plot, add some color
        if "plt.plot" in original_code or "plt.scatter" in original_code:
            enhanced_code = original_code.replace("plt.plot(", "plt.plot(", 1)
            if "color=" not in enhanced_code and "c=" not in enhanced_code:
                enhanced_code = enhanced_code.replace("plt.plot(", "plt.plot(color='#667eea', ")
        
        return "\n".join(enhancements)
        
    except Exception as e:
        # If enhancement fails, return original code
        return original_code

def apply_chart_enhancements(code: str, color_scheme: str, enhancements: list) -> str:
    """
    Apply specific enhancements to chart code
    """
    try:
        enhanced_code = code
        colors = ENHANCED_COLOR_SCHEMES.get(color_scheme, ENHANCED_COLOR_SCHEMES["Professional Blue"])
        
        # Apply color scheme
        if "color_discrete_sequence" in code:
            enhanced_code = enhanced_code.replace(
                "color_discrete_sequence=colors",
                f"color_discrete_sequence={colors}"
            )
        
        # Apply specific enhancements
        for enhancement in enhancements:
            if enhancement == "Thêm đường xu hướng" and "px.scatter" in code:
                enhanced_code += "\nfig.add_traces(px.scatter(df, x=x_col, y=y_col, trendline='ols').data[1:])"
            
            elif enhancement == "Hiển thị nhãn dữ liệu" and "px.bar" in code:
                enhanced_code = enhanced_code.replace(
                    "template=\"plotly_white\"",
                    "template=\"plotly_white\", text_auto=True"
                )
            
            elif enhancement == "Thêm lưới":
                enhanced_code += "\nfig.update_layout(showgrid=True)"
            
            elif enhancement == "Sử dụng thang logarithm":
                enhanced_code += "\nfig.update_yaxes(type='log')"
        
        return enhanced_code
        
    except Exception as e:
        return code

def enhance_prompt_with_chart_suggestions(original_prompt: str, df: pd.DataFrame) -> str:
    """
    Enhance user prompts with intelligent chart suggestions based on data characteristics
    """
    try:
        # Analyze data characteristics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month'])]
        
        suggestions = []
        
        # Add context about data structure
        suggestions.append(f"Dữ liệu có {len(numeric_cols)} cột số và {len(categorical_cols)} cột phân loại.")
        
        # Suggest appropriate chart types
        if len(numeric_cols) >= 2:
            suggestions.append("Có thể tạo biểu đồ phân tán hoặc tương quan.")
        
        if categorical_cols and numeric_cols:
            suggestions.append("Phù hợp cho biểu đồ cột hoặc box plot.")
        
        if datetime_cols:
            suggestions.append("Có dữ liệu thời gian - có thể tạo time series.")
        
        # Enhanced prompt
        enhanced_prompt = f"""
{original_prompt}

Thông tin về dữ liệu:
{' '.join(suggestions)}

Vui lòng tạo biểu đồ phù hợp và chuyên nghiệp với:
- Màu sắc hài hòa
- Labels rõ ràng
- Title có ý nghĩa
- Styling chuyên nghiệp
"""
        
        return enhanced_prompt
        
    except Exception as e:
        return original_prompt

# Helper functions for chart analysis
def analyze_chart_effectiveness(chart_type: str, data_shape: tuple) -> dict:
    """
    Analyze how effective a chart type is for given data characteristics
    """
    effectiveness = {
        "score": 0.5,
        "reasons": [],
        "improvements": []
    }
    
    rows, cols = data_shape
    
    if chart_type == "scatter" and rows > 1000:
        effectiveness["score"] = 0.8
        effectiveness["reasons"].append("Good for large datasets")
    elif chart_type == "bar" and rows < 20:
        effectiveness["score"] = 0.9
        effectiveness["reasons"].append("Perfect for categorical comparison")
    elif chart_type == "line" and cols >= 2:
        effectiveness["score"] = 0.85
        effectiveness["reasons"].append("Great for showing trends")
    
    return effectiveness

def get_color_accessibility_score(colors: list) -> float:
    """
    Calculate accessibility score for color palette
    """
    try:
        # Simple accessibility check based on color diversity
        if len(set(colors)) == len(colors):  # All unique colors
            return 0.9
        else:
            return 0.6
    except:
        return 0.5

def suggest_chart_improvements(current_code: str, df: pd.DataFrame) -> list:
    """
    Suggest improvements for existing chart code
    """
    suggestions = []
    
    try:
        if "title=" not in current_code.lower():
            suggestions.append("Thêm tiêu đề mô tả")
        
        if "xlabel=" not in current_code.lower() and "x=" in current_code:
            suggestions.append("Thêm nhãn trục X")
        
        if "ylabel=" not in current_code.lower() and "y=" in current_code:
            suggestions.append("Thêm nhãn trục Y")
        
        if "color" not in current_code.lower():
            suggestions.append("Sử dụng màu sắc phù hợp")
        
        # Check for large datasets
        if len(df) > 1000 and "sample" not in current_code.lower():
            suggestions.append("Cân nhắc sampling cho dataset lớn")
        
        return suggestions
        
    except Exception as e:
        return ["Kiểm tra lại cú pháp code"]

__all__ = [
    'smart_patch_chart_code',
    'apply_chart_enhancements',
    'enhance_prompt_with_chart_suggestions',
    'ENHANCED_COLOR_SCHEMES',
    'analyze_chart_effectiveness',
    'get_color_accessibility_score',
    'suggest_chart_improvements'
]