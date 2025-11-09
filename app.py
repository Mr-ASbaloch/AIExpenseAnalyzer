import os
import io
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF
from urllib.parse import quote

# =============================
# CONFIGURATION
# =============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "gpt-4o-mini"

st.set_page_config(page_title="AI Expense Analyzer", layout="wide")

# =============================
# SESSION STATE
# =============================
if "expenses" not in st.session_state:
    st.session_state.expenses = []

# =============================
# FUNCTIONS
# =============================

def analyze_with_groq(prompt: str):
    """Send prompt to Groq LLM and return analysis text."""
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
    }
    try:
        response = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=30)
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error contacting Groq API: {e}"

def export_to_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Expense Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)

    for i, row in df.iterrows():
        pdf.cell(200, 8, f"{row['Date']} - {row['Category']} - Rs {row['Amount']} - {row['Description']}", ln=True)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

def export_to_excel(df):
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, sheet_name="Expenses")
    excel_buffer.seek(0)
    return excel_buffer

def export_to_image(fig):
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png", bbox_inches="tight")
    img_buffer.seek(0)
    return img_buffer

# =============================
# UI - DATA INPUT
# =============================

st.title("💸 AI Expense Analyzer (Groq + Streamlit)")

with st.expander("➕ Add New Expense", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        date = st.date_input("Date")
    with col2:
        category = st.selectbox("Category", ["Food", "Transport", "Bills", "Shopping", "Other"])
    with col3:
        amount = st.number_input("Amount (PKR)", min_value=0.0, step=100.0)
    with col4:
        description = st.text_input("Description")

    if st.button("Add Expense"):
        if amount > 0 and description:
            st.session_state.expenses.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Category": category,
                "Amount": amount,
                "Description": description
            })
            st.success("✅ Expense added successfully!")
        else:
            st.warning("⚠️ Please enter valid details before adding.")

# =============================
# DISPLAY DATA
# =============================

if st.session_state.expenses:
    df = pd.DataFrame(st.session_state.expenses)
    st.dataframe(df, use_container_width=True)

    # =============================
    # VISUALIZATION
    # =============================
    st.subheader("📊 Expense Insights")

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        df.groupby("Category")["Amount"].sum().plot.pie(autopct="%1.1f%%", ax=ax1)
        plt.ylabel("")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        df.groupby("Category")["Amount"].sum().plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

    # =============================
    # AI ANALYSIS
    # =============================
    st.subheader("🧠 AI Recommendations (Groq)")

    if st.button("Generate AI Expense Analysis"):
        prompt = f"""
        Analyze these expenses and give financial insights and saving tips:
        {df.to_string(index=False)}
        """
        with st.spinner("Analyzing your expenses with Groq LLM..."):
            analysis = analyze_with_groq(prompt)
        st.text_area("AI Analysis", analysis, height=200)

    # =============================
    # EXPORT OPTIONS
    # =============================
    st.subheader("📤 Export or Share")

    pdf_buffer = export_to_pdf(df)
    excel_buffer = export_to_excel(df)
    image_buffer = export_to_image(fig1)

    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.download_button("📘 Download Excel", data=excel_buffer, file_name="expenses.xlsx")
    with colB:
        st.download_button("📄 Download PDF", data=pdf_buffer, file_name="expenses.pdf")
    with colC:
        st.download_button("🖼️ Download Chart Image", data=image_buffer, file_name="chart.png")

    with colD:
        total = df["Amount"].sum()
        whatsapp_msg = f"Total Expenses: Rs {total}\n\nTop Categories:\n{df.groupby('Category')['Amount'].sum().to_string()}"
        whatsapp_link = f"https://wa.me/?text={quote(whatsapp_msg)}"
        st.markdown(f"[💬 Share via WhatsApp]({whatsapp_link})")

else:
    st.info("Add expenses to start analysis.")

