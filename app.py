import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from datetime import datetime
from groq import Groq

# ===========================
# APP CONFIGURATION
# ===========================
st.set_page_config(page_title="💸 AI Expense Analyzer", layout="wide")
st.title("💸 AI Expense Analyzer (Groq + Streamlit)")

st.write("""
This app helps you **track**, **analyze**, and **optimize** your expenses using **AI-powered insights**.  
You can add your daily expenses, view category-wise charts, export your data, and get smart recommendations from the Groq LLM model.
""")

# ===========================
# SETUP GROQ API KEY & MODEL
# ===========================
# You can set your Groq API key directly here OR use Streamlit Secrets.
# ⚠️ RECOMMENDED: Store the key safely in .streamlit/secrets.toml as shown below:
# [secrets.toml]
# GROQ_API_KEY = "your_groq_api_key_here"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
# ===========================
# INITIALIZE STATE
# ===========================
if "expenses" not in st.session_state:
    st.session_state.expenses = []

# ===========================
# EXPENSE INPUT FORM
# ===========================
st.divider()
st.subheader("➕ Add a New Expense")

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

# ===========================
# DISPLAY EXPENSE TABLE
# ===========================
if st.session_state.expenses:
    df = pd.DataFrame(st.session_state.expenses)
    st.divider()
    st.subheader("📊 Expense Records")
    st.dataframe(df, use_container_width=True)

    # ===========================
    # CHARTS
    # ===========================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Bar Chart (Expenses by Category)")
        fig1, ax1 = plt.subplots()
        df.groupby("Category")["Amount (PKR)"].sum().plot(kind="bar", color="skyblue", ax=ax1)
        ax1.set_ylabel("Total (PKR)")
        st.pyplot(fig1)

    with col2:
        st.subheader("🥧 Pie Chart (Category Distribution)")
        fig2, ax2 = plt.subplots()
        df.groupby("Category")["Amount (PKR)"].sum().plot(kind="pie", autopct='%1.1f%%', ax=ax2)
        ax2.set_ylabel("")
        st.pyplot(fig2)

    # ===========================
    # MONTHLY SUMMARY DASHBOARD
    # ===========================
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

    # ===========================
    # EXPORT OPTIONS
    # ===========================
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

    # PDF (basic text format)
    pdf_buffer = io.BytesIO()
    text_data = df.to_string(index=False)
    pdf_buffer.write(text_data.encode("utf-8"))
    pdf_buffer.seek(0)
    st.download_button("📕 Download PDF", pdf_buffer, "expenses.pdf")

    # WhatsApp share link
    summary = f"My total expenses are PKR {total_spent:.2f}. Avg monthly: PKR {avg_spending:.2f}.\n\nTop categories:\n" + df.groupby("Category")["Amount (PKR)"].sum().to_string()
    whatsapp_link = f"https://wa.me/?text={summary}"
    st.markdown(f"[📱 Share Summary on WhatsApp]({whatsapp_link})")

    # ===========================
    # AI INSIGHTS (GROQ MODEL)
    # ===========================
    st.divider()
    st.header("🤖 AI Expense Insights (Groq LLM)")

    ai_prompt = f"""
    Analyze the following expense data and generate:
    - A short financial summary.
    - Bullet-point recommendations for better budgeting.
    - Suggestions for saving money based on categories.
    
    Expense Data:
    {df.to_string(index=False)}
    """

    if client:
        try:
            with st.spinner("💬 AI analyzing your expenses..."):
                response = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[{"role": "user", "content": ai_prompt}]
                )
                st.markdown("### 💡 AI Recommendations:")
                st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ Error fetching AI insights: {e}")
    else:
        st.info("⚠️ Please enter your Groq API key above to enable AI-powered analysis.")

else:
    st.info("👆 Add your first expense above to start analysis and visualization.")
