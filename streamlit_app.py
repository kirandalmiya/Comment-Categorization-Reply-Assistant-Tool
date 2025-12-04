import streamlit as st
import pandas as pd
import altair as alt
import joblib
from pathlib import Path
from src.replies import generate_reply  # Fixed import

# Page config
st.set_page_config(
    page_title="Comment Categorization Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💬 Comment Categorization & Reply Assistant")
st.markdown("---")

# Load models (cached)
@st.cache_resource
def load_models():
    svm_model = joblib.load("models/svm_classifier.pkl")
    lr_model = joblib.load("models/logistic_classifier.pkl")
    return svm_model, lr_model

svm_model, lr_model = load_models()

# FIXED: classify_comment function with model parameter
def classify_comment(text: str, model):
    """Classify comment using specified model"""
    label = model.predict([text])[0]
    reply = generate_reply(label)
    return {"text": text, "label": label, "reply": reply}

# Sidebar
st.sidebar.header("📊 Model & Settings")
model_choice = st.sidebar.radio(
    "Select model:",
    ["SVM (Recommended)", "Logistic Regression"],
    index=0
)
selected_model = svm_model if model_choice == "SVM (Recommended)" else lr_model

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Comment", "📁 Batch Upload", "📈 Analytics"])

# Tab 1: Single comment - FIXED
with tab1:
    st.header("Classify Single Comment")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        comment = st.text_area(
            "Enter comment:",
            placeholder="e.g., The animation was okay but voiceover needs improvement",
            height=100
        )
    with col2:
        if st.button("🔍 Classify", type="primary", use_container_width=True):
            pass
    
    if comment.strip():
        # FIXED: Pass model argument
        result = classify_comment(comment, selected_model)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("📋 Category", result["label"].title())
        with col_b:
            st.metric("🤖 Reply Ready", "Yes")
        
        st.markdown("---")
        st.subheader("💬 Suggested Reply")
        st.info(result["reply"])
        
        # Confidence scores
        try:
            probs = selected_model.predict_proba([comment])[0]
            prob_df = pd.DataFrame({
                "Category": selected_model.classes_,
                "Probability": probs
            }).sort_values("Probability", ascending=False)
            
            chart = alt.Chart(prob_df.head(5)).mark_bar().encode(
                x=alt.X("Probability:Q", axis=alt.Axis(format=".1%")),
                y=alt.Y("Category:N", sort="-x"),
                color=alt.Color("Probability:Q", scale=alt.Scale(scheme="viridis"))
            ).properties(width=400, height=200)
            st.subheader("📊 Model Confidence")
            st.altair_chart(chart, use_container_width=True)
        except:
            st.info("Confidence scores not available for this model")

# Tab 2: Batch upload - FIXED
with tab2:
    st.header("Batch Comment Classification")
    
    uploaded_file = st.file_uploader(
        "📤 Upload CSV with 'text' column",
        type=["csv"],
        help="File should have a 'text' column with comments"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')
        if "text" not in df.columns:
            st.error("❌ CSV must have a 'text' column!")
        else:
            st.write("**Preview:**")
            st.dataframe(df.head())
            
            if st.button("🚀 Classify All", type="primary", use_container_width=True):
                with st.spinner(f"Classifying {len(df)} comments..."):
                    # FIXED: Pass model argument to each classification
                    results = df["text"].astype(str).apply(
                        lambda text: classify_comment(text, selected_model)
                    )
                    df["predicted_label"] = results.apply(lambda r: r["label"])
                    df["suggested_reply"] = results.apply(lambda r: r["reply"])
                    
                    st.session_state.classified_df = df
                    st.success(f"✅ Classified {len(df)} comments!")
    
    # Show results
    if 'classified_df' in st.session_state:
        df_display = st.session_state.classified_df
        
        # Filter
        st.subheader("🔎 Filter Results")
        category_filter = st.multiselect(
            "Select categories:",
            options=sorted(df_display["predicted_label"].unique()),
            default=df_display["predicted_label"].value_counts().index[:3].tolist()
        )
        
        filtered_df = df_display[df_display["predicted_label"].isin(category_filter)]
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Results",
            data=csv,
            file_name=f"categorized_comments_{model_choice.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Tab 3: Analytics - FIXED
with tab3:
    st.header("📈 Analytics Dashboard")
    
    if 'classified_df' in st.session_state:
        df_analytics = st.session_state.classified_df
        
        col1, col2 = st.columns(2)
        with col1:
            # Bar chart
            counts = df_analytics["predicted_label"].value_counts().reset_index()
            counts.columns = ["label", "count"]
            
            bar_chart = alt.Chart(counts).mark_bar().encode(
                x=alt.X("label:N", axis=alt.Axis(title="Category")),
                y=alt.Y("count:Q"),
                color=alt.Color("label:N", scale=alt.Scale(scheme="tableau10"))
            ).properties(width=400, height=350)
            st.altair_chart(bar_chart, use_container_width=True)
        
        with col2:
            # Metrics
            st.metric("Total Comments", len(df_analytics))
            st.metric("Categories Found", df_analytics["predicted_label"].nunique())
            top_cat = df_analytics["predicted_label"].value_counts().index[0]
            st.metric("Top Category", top_cat, delta="📊")
    
    st.info("👆 Upload comments in Batch tab to see analytics")

# Footer
st.markdown("---")
st.markdown("""
**✨ Comment Categorization Assistant**  
*Built with Python, scikit-learn, Streamlit* | *Dataset: Jigsaw Toxic Comments (Kaggle)*  
**Models:** SVM (50% acc) & Logistic Regression (52% acc)
""")
