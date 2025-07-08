import pandas as pd
import numpy as np
import re
from textwrap import dedent

# Enhanced color palettes for charts
ENHANCED_COLOR_SCHEMES = {
    "professional": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "vibrant": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
    "corporate": ["#2C3E50", "#3498DB", "#E74C3C", "#F39C12", "#27AE60"],
    "sunset": ["#FF6B35", "#F7931E", "#FFD23F", "#06FFA5", "#118AB2"],
    "ocean": ["#0077BE", "#00A8CC", "#0FA3B1", "#B5E2FA", "#F9E784"]
}

def smart_patch_chart_code(code: str, df: pd.DataFrame, max_categories=10) -> str:
    """
    Intelligent chart code patching with professional styling and data handling
    """
    import re

    patched_code = code

    # 1. Add professional styling imports at the beginning
    style_imports = dedent("""\
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd

        # Set professional style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False

        # Professional color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    """)

    # 2. Handle date columns automatically
    date_cols = [col for col in df.columns if "date" in col.lower() or df[col].dtype == "object" and any(df[col].astype(str).str.contains(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', na=False))]
    for col in date_cols:
        if col in code and f"{col}.dt.year" not in code:
            date_preprocessing = dedent(f"""\
                # Convert {col} to datetime and extract useful components
                df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')
                df['{col}_year'] = df['{col}'].dt.year
                df['{col}_month'] = df['{col}'].dt.month
                df['{col}_quarter'] = df['{col}'].dt.quarter
            """)
            patched_code = date_preprocessing + patched_code

    # 3. Limit categorical data for better visualization
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if col in patched_code and df[col].nunique() > max_categories:
            limit_categories = dedent(f"""\
                # Limit {col} to top {max_categories} categories for better visualization
                top_cats = df['{col}'].value_counts().nlargest({max_categories}).index
                df = df[df['{col}'].isin(top_cats)].copy()
            """)
            patched_code = limit_categories + patched_code
            break

    # 4. Fix pandas groupby warnings
    if "groupby" in patched_code:
        patched_code = re.sub(r"\.sum\(\)", r".sum(numeric_only=True)", patched_code)
        patched_code = re.sub(r"\.mean\(\)", r".mean(numeric_only=True)", patched_code)
        patched_code = re.sub(r"\.std\(\)", r".std(numeric_only=True)", patched_code)

    # 5. Enhanced seaborn plot styling
    seaborn_enhancements = {
        "sns.barplot": "sns.barplot(palette=colors, edgecolor='black', linewidth=0.5,",
        "sns.boxplot": "sns.boxplot(palette=colors, linewidth=2,",
        "sns.violinplot": "sns.violinplot(palette=colors, linewidth=2,",
        "sns.scatterplot": "sns.scatterplot(alpha=0.7, s=60,",
        "sns.lineplot": "sns.lineplot(linewidth=3, marker='o', markersize=6,",
        "sns.countplot": "sns.countplot(palette=colors, edgecolor='black', linewidth=0.5,",
        "sns.histplot": "sns.histplot(kde=True, alpha=0.7, edgecolor='black',",
        "sns.heatmap": "sns.heatmap(annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True,"
    }
    
    for plot_func, enhancement in seaborn_enhancements.items():
        if plot_func in patched_code:
            patched_code = patched_code.replace(f"{plot_func}(", enhancement)

    # 6. Enhanced matplotlib plot styling
    if "plt.scatter(" in patched_code:
        patched_code = re.sub(r"plt\.scatter\((.*?)\)", r"plt.scatter(\1, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)", patched_code)
    
    if "plt.bar(" in patched_code:
        patched_code = re.sub(r"plt\.bar\((.*?)\)", r"plt.bar(\1, color=colors[:len(data)] if 'data' in locals() else colors[0], edgecolor='black', linewidth=0.5)", patched_code)
    
    if "plt.plot(" in patched_code:
        patched_code = re.sub(r"plt\.plot\((.*?)\)", r"plt.plot(\1, linewidth=3, marker='o', markersize=6)", patched_code)

    # 7. Handle large numeric values with formatting
    large_number_formatting = dedent("""\
        # Format large numbers for better readability
        def format_large_numbers(x, pos):
            if abs(x) >= 1e9:
                return f'{x/1e9:.1f}B'
            elif abs(x) >= 1e6:
                return f'{x/1e6:.1f}M'
            elif abs(x) >= 1e3:
                return f'{x/1e3:.1f}K'
            else:
                return f'{x:.0f}'

        from matplotlib.ticker import FuncFormatter
    """)

    # 8. Professional finishing touches
    finishing_code = dedent("""\
        # Professional chart finishing
        ax = plt.gca()

        # Title and labels styling
        if hasattr(ax, 'get_title') and ax.get_title():
            ax.set_title(ax.get_title(), fontsize=16, fontweight='bold', pad=20, color='#2c3e50')

        # Axis styling
        ax.tick_params(axis='both', which='major', labelsize=11, colors='#2c3e50')
        ax.set_xlabel(ax.get_xlabel(), fontsize=12, fontweight='500', color='#2c3e50')
        ax.set_ylabel(ax.get_ylabel(), fontsize=12, fontweight='500', color='#2c3e50')

        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

        # Spine styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')

        # Rotate x-axis labels if they're long or numerous
        labels = [label.get_text() for label in ax.get_xticklabels()]
        if len(labels) > 10 or any(len(str(label)) > 12 for label in labels):
            plt.xticks(rotation=45, ha='right')

        # Format large numbers on y-axis if needed
        y_max = max([t.get_position()[1] for t in ax.get_yticklabels()] + [0])
        if y_max > 10000:
            ax.yaxis.set_major_formatter(FuncFormatter(format_large_numbers))

        # Tight layout for better spacing
        plt.tight_layout()

        # Add subtle background
        ax.set_facecolor('#fafafa')
    """)

    # Combine everything
    final_code = style_imports + large_number_formatting + patched_code + finishing_code
    
    # Remove duplicate imports
    final_code = remove_duplicate_imports(final_code)
    
    return final_code

def apply_chart_enhancements(code: str, color_scheme: str, enhancements: list) -> str:
    """
    Apply selected enhancements to chart code based on user preferences
    """
    enhanced_code = code
    colors = ENHANCED_COLOR_SCHEMES.get(color_scheme, ENHANCED_COLOR_SCHEMES["professional"])
    
    # Update color palette
    enhanced_code = re.sub(
        r"colors = \[.*?\]",
        f"colors = {colors}",
        enhanced_code,
        flags=re.DOTALL
    )
    
    # Apply specific enhancements
    enhancement_code = ""
    
    for enhancement in enhancements:
        if enhancement == "Add trend line":
            enhancement_code += dedent("""\
                # Add trend line for scatter plots
                if 'scatter' in locals() or 'plt.scatter' in code_above:
                    try:
                        # Extract x and y data (simplified approach)
                        x_data = df.iloc[:, 0] if len(df.columns) > 0 else range(len(df))
                        y_data = df.iloc[:, 1] if len(df.columns) > 1 else df.iloc[:, 0]
                        
                        # Calculate trend line
                        z = np.polyfit(x_data.dropna(), y_data.dropna(), 1)
                        p = np.poly1d(z)
                        
                        # Plot trend line
                        plt.plot(x_data, p(x_data), '--', color='red', linewidth=2, alpha=0.8, label='Trend')
                        plt.legend()
                    except:
                        pass
            """)
        
        elif enhancement == "Show data labels":
            enhancement_code += dedent("""\
                # Add data labels for bar charts
                if 'bar' in locals() or 'plt.bar' in code_above:
                    try:
                        ax = plt.gca()
                        for i, patch in enumerate(ax.patches):
                            height = patch.get_height()
                            if not np.isnan(height):
                                ax.text(patch.get_x() + patch.get_width()/2., height + height*0.01,
                                    f'{height:.0f}' if height > 1 else f'{height:.2f}',
                                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    except:
                        pass
            """)

        elif enhancement == "Add grid":
            enhancement_code += dedent("""
                # Enhanced grid styling
                ax = plt.gca()
                ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
                ax.set_axisbelow(True)
            """)

        elif enhancement == "Use log scale":
            enhancement_code += dedent("""\
                # Apply logarithmic scale to y-axis
                plt.yscale('log')
                ax = plt.gca()
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0e}' if x > 1000 else f'{x:.1f}'))
            """)

        elif enhancement == "Highlight outliers":
            enhancement_code += dedent("""\
                # Highlight outliers using IQR method
                try:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 2:
                        x_col, y_col = numeric_cols[0], numeric_cols[1]

                        # Calculate outliers for y-axis
                        Q1 = df[y_col].quantile(0.25)
                        Q3 = df[y_col].quantile(0.75)
                        IQR = Q3 - Q1

                        outlier_mask = (df[y_col] < Q1 - 1.5*IQR) | (df[y_col] > Q3 + 1.5*IQR)

                        if outlier_mask.any():
                            plt.scatter(df.loc[outlier_mask, x_col], df.loc[outlier_mask, y_col], 
                                    color='red', s=100, alpha=0.8, marker='x', linewidth=3, 
                                    label=f'Outliers ({outlier_mask.sum()})', zorder=5)
                            plt.legend()
                except:
                    pass
            """)

        elif enhancement == "Add annotations":
            enhancement_code += dedent("""
                # Add smart annotations for interesting data points
                try:
                    ax = plt.gca()

                    # For scatter plots, annotate extreme points
                    if hasattr(ax, 'collections') and ax.collections:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 2:
                            x_col, y_col = numeric_cols[0], numeric_cols[1]

                            # Annotate max and min points
                            max_idx = df[y_col].idxmax()
                            min_idx = df[y_col].idxmin()

                            plt.annotate(f'Max: {df.loc[max_idx, y_col]:.1f}', 
                                        xy=(df.loc[max_idx, x_col], df.loc[max_idx, y_col]),
                                        xytext=(10, 10), textcoords='offset points',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

                            plt.annotate(f'Min: {df.loc[min_idx, y_col]:.1f}', 
                                        xy=(df.loc[min_idx, x_col], df.loc[min_idx, y_col]),
                                        xytext=(10, -20), textcoords='offset points',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                except:
                    pass
            """)

    # Add enhancement code to the main code
    enhanced_code = enhanced_code + "\n" + enhancement_code
    
    return enhanced_code

def remove_duplicate_imports(code: str) -> str:
    """Remove duplicate import statements from code"""
    lines = code.split('\n')
    seen_imports = set()
    cleaned_lines = []
    
    for line in lines:
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            if line.strip() not in seen_imports:
                seen_imports.add(line.strip())
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def enhance_prompt_with_chart_suggestions(prompt: str, df: pd.DataFrame) -> str:
    """
    Enhance user prompts with intelligent chart type suggestions and styling
    """
    enhanced_prompt = prompt.strip()
    
    # Analyze data characteristics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    # Chart recommendations based on data structure
    chart_suggestions = "\n\nCHART RECOMMENDATIONS:\n"
    
    if len(numeric_cols) >= 2:
        chart_suggestions += "- For numeric relationships: Use scatter plots with trend lines\n"
        chart_suggestions += "- For correlation analysis: Create correlation heatmaps\n"
    
    if categorical_cols and numeric_cols:
        chart_suggestions += f"- For category comparisons: Use box plots or violin plots comparing {numeric_cols[0]} across {categorical_cols[0]}\n"
        chart_suggestions += f"- For group analysis: Create grouped bar charts\n"
    
    if datetime_cols and numeric_cols:
        chart_suggestions += f"- For time analysis: Create time series plots with {datetime_cols[0]} on x-axis\n"
    
    if len(categorical_cols) > 0:
        chart_suggestions += f"- For distribution: Create count plots for {categorical_cols[0]}\n"
    
    # Professional styling instructions
    styling_instructions = dedent("""\
        MANDATORY STYLING REQUIREMENTS:
        - Always use professional color palettes: colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        - For scatter plots: Add alpha=0.7, s=60, edgecolors='black'
        - For bar charts: Add edgecolor='black', linewidth=0.5
        - For line plots: Use linewidth=3, marker='o', markersize=6
        - Always include plt.tight_layout() before showing
        - Rotate x-axis labels if they're long: plt.xticks(rotation=45, ha='right')
        - Add grid with alpha=0.3: plt.grid(True, alpha=0.3)
        - Remove top and right spines for cleaner look
        - Use fontsize=12 for labels, fontsize=16 for titles
        - Add subtle background color: ax.set_facecolor('#fafafa')
        """)

    # Data handling instructions
    data_instructions = dedent(f"""\
        DATA CONTEXT:
        - Dataset shape: {df.shape}
        - Numeric columns: {numeric_cols}
        - Categorical columns: {categorical_cols}
        - Date columns: {datetime_cols}
        - Missing values: {df.isnull().sum().sum()} total missing
        - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

        SMART DATA HANDLING:
        - Limit categorical variables to top 10 values if more than 10 unique values
        - Convert date strings to datetime automatically
        - Handle missing values appropriately for the chart type
        - Format large numbers (>1000) with K, M, B suffixes
        - Use log scale for data with high variance
    """)

    return enhanced_prompt + chart_suggestions + styling_instructions + data_instructions

def create_chart_from_description(description: str, df: pd.DataFrame) -> tuple:
    """
    Create chart code based on natural language description
    """
    from src.models.llms import load_llm
    
    llm = load_llm("gpt-3.5-turbo")
    
    enhanced_description = enhance_prompt_with_chart_suggestions(description, df)
    
    prompt = dedent(f"""\
        You are an expert data visualization specialist. Create Python code for a chart based on this description:

        {enhanced_description}

        Requirements:
        1. Use matplotlib/seaborn for the visualization
        2. Apply all the styling requirements mentioned above
        3. Handle the data appropriately (missing values, large numbers, etc.)
        4. Return ONLY the Python code, no explanations
        5. Assume the dataframe is already loaded as 'df'
        6. Include error handling for edge cases

        Dataset columns available: {list(df.columns)}
        Sample data: {df.head(2).to_dict()}

        Return clean, executable Python code:
    """)

    code = llm.invoke(prompt)

    # Clean the code
    code = re.sub(r'^```python\n', '', code)
    code = re.sub(r'\n```$', '', code)

    # Apply smart patching
    enhanced_code = smart_patch_chart_code(code, df)

    return enhanced_code, code

# Export all functions
__all__ = [
    'smart_patch_chart_code',
    'apply_chart_enhancements',
    'enhance_prompt_with_chart_suggestions',
    'create_chart_from_description',
    'remove_duplicate_imports',
    'ENHANCED_COLOR_SCHEMES'
]