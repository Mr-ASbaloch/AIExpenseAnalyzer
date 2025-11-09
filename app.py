import os
import io
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# =============================
# CONFIGURATION
# =============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"  # Groq LLM model (adjust if needed)

st.set_page_config(page_title="💸 AI Expense Analyzer", layout="wide")
st.title("💸 AI Expense Analyzer (Groq + Streamlit)")

st.write("""
This app helps you **track**, **analyze**, and **optimize** your expenses using **AI-powered insights**.  
You can add daily expenses, visualize them, export reports, and get personalized saving tips.
""")

# =============================
# INITIALIZE STATE
# =============================
if "expenses" not in st.session_state:
    st.session_state.expenses = []

# =============================
# EXPENSE INPUT FORM
# =============================
st.divider()
st.subheader("➕ Add New Expense")

with st.form("add_expense"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        date = st.date_input("📅 Date", datetime.now())
    with col2:
        category = st.selectbox("🏷️ Category", ["Food", "Transport", "Bills", "Entertainment", "Shopping", "Other"])
    with col3:
        amount = st.number_input("💰 Amount (PKR)", min_value=0.0, format="%.2f")
    with col4:
        description = st.text_input("📝 Description")

    submitted = st.form_submit_button("Add Expense")

    if submitted:
        if amount <= 0:
            st.error("Please enter a valid amount.")
        else:
            st.session_state.expenses.append({
                "Date": date,
                "Month": date.strftime("%B %Y"),
                "Category": category,
                "Amount (PKR)": amount,
                "Description": description
            })
            st.success("✅ Expense added successfully!")

# =============================
# DISPLAY EXPENSES
# =============================
if st.session_state.expenses:
    df = pd.DataFrame(st.session_state.expenses)
    st.divider()
    st.subheader("📊 Expense Records")
    st.dataframe(df, use_container_width=True)

    # =============================
    # VISUALIZATION
    # =============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Bar Chart (Expenses by Category)")
        fig1, ax1 = plt.subplots()
        df.groupby("Category")["Amount (PKR)"].sum().plot(kind="bar", color="cornflowerblue", ax=ax1)
        ax1.set_ylabel("Total (PKR)")
        st.pyplot(fig1)

    with col2:
        st.subheader("🥧 Pie Chart (Category Distribution)")
        fig2, ax2 = plt.subplots()
        df.groupby("Category")["Amount (PKR)"].sum().plot(kind="pie", autopct='%1.1f%%', ax=ax2)
        ax2.set_ylabel("")
        st.pyplot(fig2)

    # =============================
    # MONTHLY DASHBOARD
    # =============================
    st.divider()
    st.header("📅 Monthly Summary Dashboard")

    monthly_data = df.groupby("Month")["Amount (PKR)"].sum().reset_index()
    avg_spending = monthly_data["Amount (PKR)"].mean()
    total_spent = df["Amount (PKR)"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("💸 Total Spent", f"PKR {total_spent:,.2f}")
    col2.metric("📆 Avg Monthly Spending", f"PKR {avg_spending:,.2f}")
    col3.metric("🧾 Total Transactions", len(df))

    st.subheader("📊 Monthly Spending Trend")
    fig3, ax3 = plt.subplots()
    ax3.plot(monthly_data["Month"], monthly_data["Amount (PKR)"], marker="o", color="green")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig3)

    # =============================
    # EXPORT OPTIONS
    # =============================
    st.divider()
    st.subheader("📤 Export Options")

    # Excel
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, sheet_name="Expenses")
    excel_buffer.seek(0)
    st.download_button("📘 Download Excel", excel_buffer, "expenses.xlsx")

    # Image (chart)
    img_buffer = io.BytesIO()
    fig1.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    st.download_button("🖼️ Download Chart Image", img_buffer, "chart.png")

    # PDF (basic text)
    pdf_buffer = io.BytesIO()
    text_data = df.to_string(index=False)
    pdf_buffer.write(text_data.encode("utf-8"))
    pdf_buffer.seek(0)
    st.download_button("📕 Download PDF", pdf_buffer, "expenses.pdf")

    # WhatsApp share
    summary = f"My total expenses are PKR {total_spent:.2f}. Avg monthly: PKR {avg_spending:.2f}.\n\nTop categories:\n" + df.groupby("Category")["Amount (PKR)"].sum().to_string()
    whatsapp_link = f"https://wa.me/?text={summary}"
    st.markdown(f"[📱 Share Summary on WhatsApp]({whatsapp_link})")

    # =============================
    # AI INSIGHTS USING GROQ
    # =============================
    st.divider()
    st.header("🤖 AI Expense Insights (Groq LLM)")

    ai_prompt = f"""
    Analyze the following expense data and generate:
    - A financial summary in plain English.
    - Budget improvement tips.
    - Smart recommendations to save money in each category.

    Expense Data:
    {df.to_string(index=False)}
    """

    if GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY":
        try:
            with st.spinner("💬 AI analyzing your expenses..."):
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": ai_prompt}],
                    "temperature": 0.7,
                }
                response = requests.post(GROQ_CHAT_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    ai_data = response.json()
                    message = ai_data["choices"][0]["message"]["content"]
                    st.markdown("### 💡 AI Recommendations")
                    st.write(message)
                else:
                    st.error(f"Groq API Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"❌ Error fetching AI insights: {e}")
    else:
        st.warning("⚠️ Add a valid Groq API key in your environment or .streamlit/secrets.toml file.")

else:
    st.info("👆 Add your first expense above to start tracking and get AI analysis.")





# -------- CHAT-STYLE FOLLOW-UP --------
st.markdown("---")
st.subheader("🧭 Ask AI Insights")
st.caption("Type a natural question about your spending (only uses Groq LLM, no stored data).")

followup = st.text_input("Ask your question:", placeholder="e.g. How can I save more on food?")
if st.button("💬 Ask Groq"):
    if not followup.strip():
        st.warning("Type a question first.")
    else:
        context = f"""
User question: {followup}
Previously analyzed expense summary: {results.get('summary','')}
Recommendations: {results.get('recommendations','')}
"""
        with st.spinner("Groq thinking..."):
            answer = groq_chat([
                {"role": "system", "content": "You are a financial assistant offering clear, empathetic guidance."},
                {"role": "user", "content": context}
            ])
        st.success("Groq's Response:")
        st.write(answer)
