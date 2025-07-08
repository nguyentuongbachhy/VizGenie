import streamlit as st
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
    render_interactive_data_explorer, create_ai_recommendation_panel,
    render_animated_loading, PROFESSIONAL_CSS
)

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
    page_title="VizGenie-GPT Professional", 
    layout="wide", 
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Apply professional CSS
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# Professional header with animation
render_professional_header(
    "VizGenie-GPT Professional Analytics",
    "Advanced AI-powered data analysis with intelligent insights and beautiful visualizations",
    "🧠"
)

# Load environment and initialize
load_dotenv()
init_db()

def generate_comprehensive_data_story(df: pd.DataFrame, chat_history: list, dataset_name: str) -> str:
    """Generate a comprehensive data story with business insights"""
    llm = load_llm("gpt-3.5-turbo")
    
    # Extract conversation patterns
    questions = [msg[2] for msg in chat_history if msg[1] == "user"][-10:]
    
    # Analyze data characteristics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    prompt = f"""
    You are a senior data analyst creating an executive summary for dataset '{dataset_name}'.
    
    📊 DATASET OVERVIEW:
    - Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
    - Numeric variables: {len(numeric_cols)} ({', '.join(numeric_cols[:5])})
    - Categorical variables: {len(categorical_cols)} ({', '.join(categorical_cols[:5])})
    - Missing data: {df.isnull().sum().sum():,} cells ({(df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100):.1f}%)
    
    🔍 RECENT ANALYSIS QUESTIONS:
    {questions}
    
    Create a compelling executive summary with:
    
    ## 📈 Executive Summary
    [2-3 sentences highlighting the most important findings]
    
    ## 🎯 Key Insights
    [4-5 specific, actionable insights with numbers where possible]
    
    ## 📊 Data Quality Assessment  
    [Brief assessment of data reliability and completeness]
    
    ## 💼 Business Implications
    [How these insights can drive business decisions]
    
    ## 🚀 Recommended Next Steps
    [3-4 specific actions to take based on the analysis]
    
    ## ⚠️ Limitations & Considerations
    [Important caveats about the data or analysis]
    
    Make it executive-ready: professional, concise, and focused on actionable insights.
    Use specific numbers and percentages where possible.
    """
    
    return llm.invoke(prompt)

def extract_enhanced_chart_insights(code: str, df: pd.DataFrame) -> str:
    """Extract detailed insights about the generated chart"""
    llm = load_llm("gpt-3.5-turbo")
    
    # Identify chart type from code
    chart_type = "Unknown"
    if "scatter" in code.lower():
        chart_type = "Scatter Plot"
    elif "bar" in code.lower():
        chart_type = "Bar Chart" 
    elif "hist" in code.lower():
        chart_type = "Histogram"
    elif "box" in code.lower():
        chart_type = "Box Plot"
    elif "line" in code.lower():
        chart_type = "Line Plot"
    elif "heatmap" in code.lower():
        chart_type = "Heatmap"
    
    prompt = f"""
    Analyze this {chart_type} generated from the following code:
    
    ```python
    {code}
    ```
    
    Dataset characteristics:
    - Shape: {df.shape}
    - Columns: {list(df.columns)}
    - Data types: {df.dtypes.to_dict()}
    
    Provide detailed insights in this format:
    
    ## 📊 Chart Analysis
    [What this chart shows and why it's useful]
    
    ## 🔍 Key Patterns
    [Specific patterns, trends, or relationships visible]
    
    ## 📈 Statistical Insights  
    [Quantitative observations with actual numbers]
    
    ## 💡 Business Value
    [How this visualization helps business decisions]
    
    ## 🎯 Follow-up Suggestions
    [What additional analyses would be valuable]
    
    Be specific and include actual column names and potential values.
    Focus on actionable insights that stakeholders can use.
    """
    
    return llm.invoke(prompt)

# Load datasets with error handling
datasets = get_all_datasets()
if not datasets:
    render_feature_card(
        "Welcome to VizGenie-GPT",
        "Start by uploading your first dataset in the Dashboard page to begin advanced analytics.",
        "👋",
        "Go to Dashboard",
        "dashboard"
    )
    st.stop()

# Dataset selection with enhanced UI
st.markdown("### 📂 Dataset Selection")
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    dataset_options = {f"{d[0]} - {d[1]}": d[0] for d in datasets}
    selected = st.selectbox(
        "Choose your dataset:",
        list(dataset_options.keys()),
        help="Select the dataset you want to analyze"
    )
    dataset_id = dataset_options[selected]
    dataset = get_dataset(dataset_id)

with col2:
    if st.button("📊 Generate Data Story", type="primary", use_container_width=True):
        st.session_state.generate_story = True

with col3:
    if st.button("🔍 Data Explorer", type="secondary", use_container_width=True):
        st.session_state.show_explorer = True

# Load and validate dataset
file_path = dataset[2]
num_rows, num_cols = dataset[3], dataset[4]

try:
    df = safe_read_csv(file_path)
    st.session_state.df = df
except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# Dataset metrics with professional cards
st.markdown("### 📊 Dataset Overview")
metrics = [
    {"title": "Total Records", "value": f"{num_rows:,}", "delta": None},
    {"title": "Columns", "value": str(num_cols), "delta": None},
    {"title": "Numeric Fields", "value": str(df.select_dtypes(include=[np.number]).shape[1]), "delta": None},
    {"title": "Missing Values", "value": f"{df.isnull().sum().sum():,}", "delta": None}
]

render_metric_cards(metrics)

# Data quality assessment
st.markdown("### 🎯 Data Quality Assessment")
quality_score = create_data_quality_indicator(df)

if quality_score < 0.7:
    render_status_indicator("Data Quality Needs Attention", "warning")
elif quality_score < 0.9:
    render_status_indicator("Good Data Quality", "success")
else:
    render_status_indicator("Excellent Data Quality", "success")

# AI Recommendations Panel
create_ai_recommendation_panel(df)

# Interactive Data Explorer (if requested)
if st.session_state.get('show_explorer', False):
    with st.expander("🔍 Interactive Data Explorer", expanded=True):
        render_interactive_data_explorer(df)
    st.session_state.show_explorer = False

# Chat session management with enhanced UI
st.markdown("### 💬 AI Analysis Sessions")

# Session selection
sessions = get_sessions_by_dataset(dataset_id)
session_titles = {f"{s[0]} - {s[1]} ({s[2]})": s[0] for s in sessions}

col1, col2 = st.columns([3, 1])
with col1:
    new_session_title = st.text_input(
        "🆕 Create new analysis session:",
        placeholder="e.g., Revenue Analysis, Customer Segmentation, Trend Discovery...",
        help="Give your analysis session a descriptive name"
    )

with col2:
    session_type = st.radio("Session:", ("New", "Existing"), horizontal=True)

if session_type == "Existing" and sessions:
    selected_session = st.selectbox("Select existing session:", list(session_titles.keys()))
    session_id = session_titles[selected_session]
    
    # Session management options
    with st.expander("⚙️ Session Management"):
        col1, col2 = st.columns(2)
        with col1:
            rename_title = st.text_input("Rename session:")
            if st.button("✏️ Rename") and rename_title:
                rename_chat_session(session_id, rename_title)
                st.success("✅ Session renamed!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Delete Session", type="secondary"):
                delete_chat_session(session_id)
                st.success("🗑️ Session deleted!")
                st.rerun()

else:
    # Create new session
    default_title = new_session_title or f"Analysis Session {len(sessions) + 1}"
    session_id = create_chat_session(dataset_id, default_title)
    st.success(f"✅ Created session: **{default_title}**")

# Load chat history
chat_history = get_chat_messages(session_id)

# Generate comprehensive data story if requested
if st.session_state.get('generate_story', False):
    with st.spinner("🤖 Generating comprehensive data story..."):
        render_animated_loading("Analyzing your data and generating insights...")
        
        story = generate_comprehensive_data_story(df, chat_history, dataset[1])
        
        render_insight_card(story)
        
        # Save story to chat history
        add_chat_message(session_id, "assistant", f"**📖 Data Story Generated**\n\n{story}")
        
    st.session_state.generate_story = False

# Enhanced chat history display
if chat_history:
    st.markdown("### 🗨️ Conversation History")
    
    for idx, (msg_id, role, content, ts) in enumerate(chat_history):
        with st.chat_message(role):
            cols = st.columns([10, 1])
            
            with cols[0]:
                # Enhanced message rendering
                if role == "assistant" and "📖 Data Story" in content:
                    # Special rendering for data stories
                    render_insight_card(content.replace("**📖 Data Story Generated**\n\n", ""))
                else:
                    st.markdown(content)
            
            with cols[1]:
                if role == "user":
                    with st.popover("⋮", use_container_width=True):
                        if st.button("✏️ Edit", key=f"edit_{idx}"):
                            st.session_state.edited_prompt = content
                            # Delete this message and the next AI response
                            delete_chat_message(session_id, msg_id)
                            if idx + 1 < len(chat_history) and chat_history[idx + 1][1] == "assistant":
                                delete_chat_message(session_id, chat_history[idx + 1][0])
                            st.rerun()
                        
                        if st.button("🗑️ Delete", key=f"del_{msg_id}"):
                            delete_chat_message(session_id, msg_id)
                            # Also delete the AI response if it exists
                            if idx + 1 < len(chat_history) and chat_history[idx + 1][1] == "assistant":
                                delete_chat_message(session_id, chat_history[idx + 1][0])
                            st.rerun()
                        
                        if st.button("📋 Copy", key=f"copy_{idx}"):
                            st.session_state.clipboard = content
                            st.success("Copied to clipboard!")

# Smart query suggestions with enhanced UI
st.markdown("### 💡 Smart Query Suggestions")
with st.expander("🎯 Get Inspired - Sample Questions", expanded=False):
    
    # Dynamic suggestions based on data characteristics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Data Analysis Questions:**")
        
        analysis_suggestions = [
            f"Show the distribution of {numeric_cols[0]}" if numeric_cols else "Analyze the data distribution",
            f"What's the correlation between {numeric_cols[0]} and {numeric_cols[1]}?" if len(numeric_cols) >= 2 else "Find correlations in the data",
            "Identify outliers and anomalies in the dataset",
            f"Compare {numeric_cols[0]} across different {categorical_cols[0]} groups" if numeric_cols and categorical_cols else "Compare groups in the data",
            "Create a comprehensive statistical summary"
        ]
        
        for suggestion in analysis_suggestions:
            if st.button(suggestion, key=f"analysis_{suggestion[:20]}", use_container_width=True):
                st.session_state.suggested_prompt = suggestion
    
    with col2:
        st.markdown("**🎯 Business Intelligence Questions:**")
        
        # Context-aware business questions
        business_questions = [
            "What are the key performance indicators in this data?",
            "Which factors have the strongest impact on outcomes?",
            "Are there seasonal or time-based patterns?", 
            "What segments or groups show the best performance?",
            "What recommendations can you make based on this data?"
        ]
        
        for question in business_questions:
            if st.button(question, key=f"business_{question[:20]}", use_container_width=True):
                st.session_state.suggested_prompt = question

# Enhanced chart type suggestions
st.markdown("### 📈 Smart Chart Recommendations")
with st.expander("🎨 AI-Suggested Visualizations", expanded=False):
    
    chart_recommendations = []
    
    if len(numeric_cols) >= 2:
        chart_recommendations.extend([
            {"type": "Scatter Plot", "desc": f"Explore relationship between {numeric_cols[0]} and {numeric_cols[1]}", "icon": "🔵"},
            {"type": "Correlation Heatmap", "desc": "Show all numeric correlations", "icon": "🔥"}
        ])
    
    if categorical_cols and numeric_cols:
        chart_recommendations.extend([
            {"type": "Box Plot", "desc": f"Compare {numeric_cols[0]} distribution by {categorical_cols[0]}", "icon": "📦"},
            {"type": "Bar Chart", "desc": f"Show average {numeric_cols[0]} by {categorical_cols[0]}", "icon": "📊"}
        ])
    
    if any('date' in col.lower() or 'time' in col.lower() for col in df.columns):
        chart_recommendations.append(
            {"type": "Time Series", "desc": "Track changes over time", "icon": "📈"}
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
                if st.button(f"Create {rec['type']}", key=f"chart_rec_{i}", use_container_width=True):
                    st.session_state.suggested_prompt = f"Create a {rec['type'].lower()} showing {rec['desc']}"

# Main chat input with enhanced processing
prompt = (st.session_state.pop("suggested_prompt", None) or 
          st.session_state.pop("edited_prompt", None) or 
          st.chat_input("🤖 Ask anything about your data - I'll create beautiful visualizations and insights!"))

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
            with st.spinner("🧠 Analyzing your data with AI..."):
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
                    st.markdown("#### 📊 Generated Visualization")
                    
                    # Execute and display chart
                    fig = execute_plt_code(patched_code, df)
                    if fig:
                        st.pyplot(fig)
                        
                        # Generate enhanced insights
                        with st.spinner("🔍 Extracting deep insights..."):
                            chart_insights = extract_enhanced_chart_insights(patched_code, df)
                        
                        render_insight_card(chart_insights)
                
                with controls_col:
                    st.markdown("#### 🎨 Chart Enhancements")
                    
                    # Color scheme selector
                    color_scheme = st.selectbox(
                        "Color Palette:",
                        list(ENHANCED_COLOR_SCHEMES.keys()),
                        index=0,
                        key=f"color_{len(chat_history)}"
                    )
                    
                    # Enhancement options
                    enhancements = st.multiselect(
                        "Add Features:",
                        [
                            "Add trend line",
                            "Show data labels",
                            "Add grid",
                            "Use log scale", 
                            "Highlight outliers",
                            "Add annotations"
                        ],
                        key=f"enhance_{len(chat_history)}"
                    )
                    
                    # Apply enhancements
                    if st.button("🔄 Apply Changes", key=f"apply_{len(chat_history)}"):
                        enhanced_code = apply_chart_enhancements(patched_code, color_scheme, enhancements)
                        fig_enhanced = execute_plt_code(enhanced_code, df)
                        if fig_enhanced:
                            with chart_col:
                                st.markdown("#### ✨ Enhanced Visualization")
                                st.pyplot(fig_enhanced)
                    
                    # Chart actions
                    st.markdown("#### 💾 Chart Actions")
                    
                    if st.button("Save to Gallery", key=f"save_{len(chat_history)}", use_container_width=True):
                        add_chart_card(dataset_id, prompt, response["output"], patched_code)
                        st.success("✅ Chart saved!")
                    
                    if st.button("Download PNG", key=f"download_{len(chat_history)}", use_container_width=True):
                        st.info("📥 Download functionality would be implemented here")
                    
                    if st.button("Share Chart", key=f"share_{len(chat_history)}", use_container_width=True):
                        st.info("📤 Share functionality would be implemented here")
                
                # Code display with tabs
                with st.expander("📋 View Generated Code", expanded=False):
                    tab1, tab2 = st.tabs(["Enhanced Code", "Original AI Code"])
                    
                    with tab1:
                        st.code(patched_code, language="python")
                        st.caption("This code includes professional styling and smart data handling")
                    
                    with tab2:
                        st.code(action_code, language="python") 
                        st.caption("Original code generated by AI")
            
            # Handle Plotly charts
            elif action_code and ("plotly" in action_code or "px." in action_code):
                st.markdown("#### 📊 Interactive Visualization")
                try:
                    exec_globals = {"df": df, "px": px, "go": go, "st": st}
                    exec(action_code, exec_globals)
                    
                    render_insight_card("🎯 **Interactive Chart Created!** This Plotly visualization supports zooming, hovering, and interactive exploration.")
                    
                except Exception as e:
                    st.error(f"❌ Error creating interactive chart: {e}")
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            render_insight_card(
                "💡 **Troubleshooting Tips:**\n"
                "- Try rephrasing your question more specifically\n" 
                "- Mention specific column names you want to analyze\n"
                "- Ask for a particular type of chart or analysis\n"
                "- Check if your data has the required columns for the analysis"
            )

# Professional sidebar with navigation and stats
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e1e5e9; margin-bottom: 1rem;">
        <h3 style="color: #667eea; margin: 0;">🧠 VizGenie-GPT</h3>
        <small style="color: #666;">Professional Analytics Platform</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick navigation
    st.markdown("### 🔗 Quick Navigation")
    nav_buttons = [
        ("📂 Dashboard", "pages/1_🧮_Dashboard.py"),
        ("📊 Dataset Details", "pages/3_📂_Dataset_Details.py"), 
        ("📈 Smart Charts", "pages/6_📈_Smart_Charts.py"),
        ("🔗 Cross Analysis", "pages/7_🔗_Cross_Dataset_Analysis.py"),
        ("📋 Chart History", "pages/4_📊_Charts_History.py"),
        ("📄 EDA Report", "pages/5_📋_EDA Report.py")
    ]
    
    for label, page in nav_buttons:
        if st.button(label, key=f"nav_{label}", use_container_width=True):
            st.switch_page(page)
    
    # Session statistics
    if chat_history:
        st.markdown("---")
        st.markdown("### 📈 Session Stats")
        
        user_messages = [msg for msg in chat_history if msg[1] == "user"]
        charts_created = len([msg for msg in chat_history if "chart" in msg[2].lower() or "plot" in msg[2].lower()])
        
        render_metric_cards([
            {"title": "Questions", "value": str(len(user_messages))},
            {"title": "Charts", "value": str(charts_created)},
            {"title": "Quality", "value": f"{quality_score:.0%}"}
        ])
        
        # Session summary
        if st.button("📊 Generate Session Summary", use_container_width=True):
            summary_prompt = f"""
            Summarize this data analysis session in 3 key bullet points:
            
            Questions asked: {[msg[2] for msg in user_messages]}
            Dataset: {dataset[1]} ({df.shape[0]} rows, {df.shape[1]} cols)
            
            Focus on:
            - Main areas of analysis explored
            - Key insights discovered  
            - Types of visualizations created
            
            Keep it concise and executive-friendly.
            """
            
            with st.spinner("Generating summary..."):
                summary = load_llm("gpt-3.5-turbo").invoke(summary_prompt)
                render_insight_card(f"**📋 Session Summary**\n\n{summary}")
    
    # Pro tips
    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    st.markdown("""
    **🎯 Better Questions:**
    - Be specific about columns
    - Ask for comparisons
    - Request business insights
    
    **📊 Chart Tips:**
    - Try different color palettes
    - Use enhancements for clarity
    - Save charts you like
    
    **🤖 AI Features:**
    - Generate data stories
    - Get chart recommendations
    - Ask follow-up questions
    """)

# Footer with credits and version
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🧠 VizGenie-GPT Professional**")
    st.caption("Advanced AI Analytics Platform")

with col2:
    st.markdown("**🔧 Version 2.0**")
    st.caption("Enhanced with Professional UI")

with col3:
    st.markdown("**👨‍💻 Made by Delay Group**")
    st.caption("With ❤️ for data science")