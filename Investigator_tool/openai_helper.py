import streamlit as st
import pandas as pd
from main import extract_summary  # Assumes this function returns a DataFrame

st.set_page_config(page_title="Financial Crime Investigator", layout="wide")

# Inject CSS to wrap summary column and allow horizontal overflow
st.markdown("""
    <style>
        .wrap-summary td:nth-child(3) {
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            max-width: 500px !important;
        }
        .wrap-summary table {
            table-layout: auto;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("💰 Financial Crime Investigator Tool")

# Two-column layout
col1, col2 = st.columns([1, 2])  # Adjust ratio as needed

# Column 1: Text input
with col1:
    st.subheader("📋 Transaction Text Input")
    user_input = st.text_area(
        "Enter transaction data or JSON below:",
        height=500,
        placeholder="Paste transaction summary or JSON here..."
    )
    run_button = st.button("🔍 Analyze")

# Column 2: Display result
with col2:
    st.subheader("🧾 Extracted Crime Features")
    if run_button:
        if user_input.strip():
            with st.spinner("Analyzing with LLM..."):
                try:
                    df = extract_summary(user_input)
                    if df.empty:
                        st.warning("No suspicious patterns detected.")
                    else:
                        # Wrap summary column properly and show DataFrame
                        st.markdown('<div class="wrap-summary">', unsafe_allow_html=True)
                        st.dataframe(df, use_container_width=True, height=500)
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        else:
            st.warning("Please enter some input first.")
