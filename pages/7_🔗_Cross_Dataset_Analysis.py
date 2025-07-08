import streamlit as st
import pandas as pd
import numpy as np
from src.utils import get_all_datasets, get_dataset, safe_read_csv
from src.models.llms import load_llm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🔗 Cross-Dataset Analysis", layout="wide")

# Enhanced styling
st.markdown("""
<style>
    .analysis-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .insight-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .correlation-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="analysis-header"><h1>🔗 Cross-Dataset Relationship Analysis</h1><p>Discover hidden patterns and relationships across multiple datasets</p></div>', unsafe_allow_html=True)

llm = load_llm("gpt-3.5-turbo")

# Load available datasets
datasets = get_all_datasets()
if not datasets:
    st.warning("Please upload datasets first.")
    st.stop()

# Dataset selection interface
st.subheader("📂 Select Datasets for Analysis")
col1, col2 = st.columns(2)

with col1:
    dataset1_options = {f"{d[0]} - {d[1]}": d[0] for d in datasets}
    dataset1_selection = st.selectbox("Primary Dataset:", list(dataset1_options.keys()))
    dataset1_id = dataset1_options[dataset1_selection]

with col2:
    dataset2_options = {f"{d[0]} - {d[1]}": d[0] for d in datasets if d[0] != dataset1_id}
    if dataset2_options:
        dataset2_selection = st.selectbox("Secondary Dataset:", list(dataset2_options.keys()))
        dataset2_id = dataset2_options[dataset2_selection]
    else:
        st.warning("Need at least 2 datasets for cross-analysis")
        st.stop()

# Load datasets
dataset1 = get_dataset(dataset1_id)
dataset2 = get_dataset(dataset2_id)
df1 = safe_read_csv(dataset1[2])
df2 = safe_read_csv(dataset2[2])

st.success(f"✅ Loaded: **{dataset1[1]}** ({df1.shape[0]} rows) and **{dataset2[1]}** ({df2.shape[0]} rows)")

# Analysis options
st.subheader("🎯 Analysis Type")
analysis_type = st.radio(
    "Choose analysis method:",
    ["Column Similarity", "Statistical Correlation", "Semantic Relationships", "Combined Analysis"],
    horizontal=True
)

def find_similar_columns(df1, df2, similarity_threshold=0.7):
    """Find columns with similar names or data types"""
    similar_pairs = []
    
    for col1 in df1.columns:
        for col2 in df2.columns:
            # Name similarity
            name_sim = len(set(col1.lower().split()) & set(col2.lower().split())) / max(len(set(col1.lower().split())), len(set(col2.lower().split())))
            
            # Data type similarity
            type_sim = 1.0 if df1[col1].dtype == df2[col2].dtype else 0.5
            
            # Combined similarity
            combined_sim = (name_sim + type_sim) / 2
            
            if combined_sim >= similarity_threshold:
                similar_pairs.append({
                    'col1': col1,
                    'col2': col2,
                    'similarity': combined_sim,
                    'type1': str(df1[col1].dtype),
                    'type2': str(df2[col2].dtype)
                })
    
    return sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)

def calculate_cross_correlations(df1, df2):
    """Calculate correlations between numeric columns across datasets"""
    num_cols1 = df1.select_dtypes(include=[np.number]).columns
    num_cols2 = df2.select_dtypes(include=[np.number]).columns
    
    correlations = []
    
    for col1 in num_cols1:
        for col2 in num_cols2:
            try:
                # Align lengths for correlation
                min_len = min(len(df1[col1]), len(df2[col2]))
                
                # Calculate Pearson correlation
                pearson_r, pearson_p = pearsonr(df1[col1][:min_len].fillna(0), df2[col2][:min_len].fillna(0))
                
                # Calculate Spearman correlation
                spearman_r, spearman_p = spearmanr(df1[col1][:min_len].fillna(0), df2[col2][:min_len].fillna(0))
                
                correlations.append({
                    'col1': col1,
                    'col2': col2,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'significance': 'High' if min(pearson_p, spearman_p) < 0.01 else 'Medium' if min(pearson_p, spearman_p) < 0.05 else 'Low'
                })
            except:
                continue
    
    return sorted(correlations, key=lambda x: abs(x['pearson_r']), reverse=True)

def generate_ai_insights(df1, df2, dataset1_name, dataset2_name, analysis_results):
    """Generate AI insights about relationships"""
    prompt = f"""
    As a data scientist, analyze the relationship between two datasets:

    Dataset 1: {dataset1_name}
    - Shape: {df1.shape}
    - Columns: {list(df1.columns)[:10]}...
    - Sample data: {df1.head(2).to_dict()}

    Dataset 2: {dataset2_name}
    - Shape: {df2.shape}
    - Columns: {list(df2.columns)[:10]}...
    - Sample data: {df2.head(2).to_dict()}

    Analysis Results: {str(analysis_results)[:1000]}...

    Provide insights in this format:
    
    ## 🔍 Key Relationships Found
    [List 3-5 most important relationships]
    
    ## 📊 Business Implications
    [Explain what these relationships mean in business context]
    
    ## 🎯 Recommended Actions
    [Suggest specific actions based on findings]
    
    ## ⚠️ Limitations & Considerations
    [Mention any caveats or limitations]
    
    Be specific and actionable. Focus on practical insights.
    """
    
    return llm.invoke(prompt)

# Execute analysis based on selection
if st.button("🚀 Run Analysis", type="primary"):
    with st.spinner("Analyzing relationships..."):
        
        if analysis_type == "Column Similarity":
            similar_cols = find_similar_columns(df1, df2)
            
            st.subheader("📋 Similar Columns Found")
            if similar_cols:
                for pair in similar_cols[:10]:  # Show top 10
                    st.markdown(f"""
                    <div class="insight-card">
                        <strong>{pair['col1']}</strong> ↔️ <strong>{pair['col2']}</strong><br>
                        Similarity: {pair['similarity']:.2%} | Types: {pair['type1']} vs {pair['type2']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No similar columns found with current threshold.")
        
        elif analysis_type == "Statistical Correlation":
            correlations = calculate_cross_correlations(df1, df2)
            
            st.subheader("📈 Cross-Dataset Correlations")
            if correlations:
                # Create correlation matrix visualization
                correlation_data = []
                for corr in correlations[:20]:  # Top 20
                    correlation_data.append({
                        'Relationship': f"{corr['col1']} × {corr['col2']}",
                        'Pearson R': corr['pearson_r'],
                        'Spearman R': corr['spearman_r'],
                        'Significance': corr['significance']
                    })
                
                corr_df = pd.DataFrame(correlation_data)
                
                # Interactive plotly chart
                fig = px.bar(corr_df, x='Relationship', y='Pearson R', 
                           color='Significance', 
                           title="Cross-Dataset Correlations",
                           color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#95a5a6'})
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table
                st.dataframe(corr_df, use_container_width=True)
            else:
                st.info("No significant correlations found.")
        
        elif analysis_type == "Semantic Relationships":
            # AI-powered semantic analysis
            prompt = f"""
            Analyze these two datasets for semantic relationships:
            
            Dataset 1 columns: {list(df1.columns)}
            Dataset 2 columns: {list(df2.columns)}
            
            Find potential semantic relationships like:
            - Geographic connections (city, state, country)
            - Temporal connections (date, time, period)
            - Categorical connections (type, category, class)
            - Hierarchical connections (parent-child relationships)
            
            Return a JSON list of potential relationships with confidence scores.
            """
            
            ai_relationships = llm.invoke(prompt)
            
            st.subheader("🧠 AI-Detected Semantic Relationships")
            st.markdown(ai_relationships)
        
        elif analysis_type == "Combined Analysis":
            # Run all analyses
            similar_cols = find_similar_columns(df1, df2)
            correlations = calculate_cross_correlations(df1, df2)
            
            # Generate comprehensive AI insights
            all_results = {
                'similar_columns': similar_cols[:5],
                'correlations': correlations[:5]
            }
            
            ai_insights = generate_ai_insights(df1, df2, dataset1[1], dataset2[1], all_results)
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["🔍 AI Insights", "📋 Similar Columns", "📈 Correlations"])
            
            with tab1:
                st.markdown(ai_insights)
            
            with tab2:
                if similar_cols:
                    for pair in similar_cols[:10]:
                        st.markdown(f"""
                        <div class="insight-card">
                            <strong>{pair['col1']}</strong> ↔️ <strong>{pair['col2']}</strong><br>
                            Similarity: {pair['similarity']:.2%}
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab3:
                if correlations:
                    correlation_data = []
                    for corr in correlations[:15]:
                        correlation_data.append({
                            'Column 1': corr['col1'],
                            'Column 2': corr['col2'],
                            'Correlation': f"{corr['pearson_r']:.3f}",
                            'P-value': f"{corr['pearson_p']:.3f}",
                            'Significance': corr['significance']
                        })
                    
                    st.dataframe(pd.DataFrame(correlation_data), use_container_width=True)

# Advanced Query Interface
st.subheader("💬 Ask Questions Across Datasets")
query_placeholder = st.text_area(
    "Ask complex questions spanning both datasets:",
    placeholder="Examples:\n- How many female teachers are in Hanoi elementary schools?\n- What's the dropout rate correlation between regions?\n- Compare student performance across different school types",
    height=100
)

if st.button("🎯 Answer Question") and query_placeholder:
    with st.spinner("Processing complex query..."):
        enhanced_prompt = f"""
        You have access to two datasets:
        
        Dataset 1: {dataset1[1]}
        Columns: {list(df1.columns)}
        Sample: {df1.head(2).to_dict()}
        
        Dataset 2: {dataset2[1]}
        Columns: {list(df2.columns)}
        Sample: {df2.head(2).to_dict()}
        
        User Question: {query_placeholder}
        
        Provide a comprehensive answer that:
        1. Identifies which datasets and columns are relevant
        2. Explains any assumptions made
        3. Provides specific numbers/insights where possible
        4. Suggests follow-up analyses
        5. Notes any limitations
        
        Be specific and data-driven in your response.
        """
        
        response = llm.invoke(enhanced_prompt)
        
        st.markdown("### 🎯 Analysis Results")
        st.markdown(f"""
        <div class="insight-card">
            {response}
        </div>
        """, unsafe_allow_html=True)

# Export results
if st.button("📥 Export Analysis Report"):
    st.success("Analysis report would be generated and downloaded here")