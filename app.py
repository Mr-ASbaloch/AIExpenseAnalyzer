import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =============================
# CONFIGURATION
# =============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"  # Groq model name (adjust per your account)

# =============================
# HELPER: Groq Chat API
# =============================
def groq_chat(messages, temperature=0.3, max_tokens=800):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    response = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# =============================
# HELPER: Analyze Expenses
# =============================
def analyze_expenses(raw_text):
    system_prompt = """You are a financial data expert.
Read the user's provided expense text or CSV-like input and return a JSON:
{
 "transactions":[{"date":"YYYY-MM-DD","merchant":"string","amount":float,"category":"string"}],
 "summary":"concise paragraph about spending patterns, totals, and category trends",
 "recommendations":"personalized saving and optimization strategies"
}
Return only valid JSON.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": raw_text}
    ]
    reply = groq_chat(messages)

    # Try parsing JSON safely
    try:
        data = json.loads(reply)
    except Exception:
        import re
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        data = json.loads(match.group(0)) if match else {"transactions": [], "summary": reply, "recommendations": ""}
    return data

# =============================
# STREAMLIT APP
# =============================
st.set_page_config(page_title="💸 AI Expense Analyzer", layout="wide")
st.title("💸 AI Expense Analyzer using Groq LLM + Streamlit")

st.markdown("""
This AI-powered expense analyzer reads your expenses, categorizes them,  
**summarizes patterns**, and gives **smart saving strategies** 💡.  
Now with **Auto Budget Warnings** and dual visualization (Bar + Pie).
""")

# -------- INPUT SECTION --------
input_type = st.radio("Choose Input Type:", ["📝 Paste Text", "📂 Upload CSV"])
raw_text = ""

if input_type == "📝 Paste Text":
    raw_text = st.text_area("Paste your expenses here:", height=200,
                            placeholder="e.g.\nUber 2500 PKR Nov 3\nKFC 1200 PKR Nov 4\nInternet Bill 3500 PKR Nov 5")
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        raw_text = df.to_csv(index=False)

# -------- ANALYSIS --------
if st.button("🔍 Analyze with Groq"):
    if not raw_text.strip():
        st.warning("⚠️ Please provide expense data first.")
    else:
        with st.spinner("Analyzing your expenses with Groq..."):
            results = analyze_expenses(raw_text)

        st.success("✅ Analysis Complete!")

        # Display summary and recommendations
        st.subheader("📊 Expense Summary")
        st.write(results.get("summary", "No summary generated."))

        st.subheader("💡 AI Recommendations")
        st.write(results.get("recommendations", "No recommendations generated."))

        # Display transactions
        transactions = results.get("transactions", [])
        if transactions:
            df = pd.DataFrame(transactions)
            st.dataframe(df, use_container_width=True)

            # -------- CATEGORY ANALYSIS --------
            if "category" in df.columns and "amount" in df.columns:
                cat_sum = df.groupby("category")["amount"].sum().sort_values(ascending=False)
                total_spent = cat_sum.sum()

                st.markdown("### 💰 Spending by Category")

                # Auto Budget Warnings
                for cat, amt in cat_sum.items():
                    percent = (amt / total_spent) * 100
                    if percent > 40:
                        st.error(f"⚠️ High Spending in **{cat}**: {percent:.1f}% of total — consider reducing this category.")
                    elif percent > 25:
                        st.warning(f"⚠️ Notable Spending in **{cat}**: {percent:.1f}% of total.")
                    else:
                        st.info(f"✅ Balanced Spending in **{cat}**: {percent:.1f}% of total.")

                # Display charts
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 📊 Bar Chart")
                    fig_bar, ax_bar = plt.subplots()
                    cat_sum.plot(kind="bar", ax=ax_bar, color="skyblue")
                    ax_bar.set_title("Category-wise Spending (Bar Chart)")
                    ax_bar.set_xlabel("Category")
                    ax_bar.set_ylabel("Amount")
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig_bar)

                with col2:
                    st.markdown("#### 🥧 Pie Chart")
                    fig_pie, ax_pie = plt.subplots()
                    ax_pie.pie(cat_sum, labels=cat_sum.index, autopct='%1.1f%%', startangle=90)
                    ax_pie.axis("equal")
                    st.pyplot(fig_pie)

            # -------- MONTHLY TREND --------
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"])
                    df["month"] = df["date"].dt.to_period("M").astype(str)
                    trend = df.groupby("month")["amount"].sum()

                    st.markdown("### 📈 Monthly Spending Trend")
                    fig2, ax2 = plt.subplots()
                    trend.plot(ax=ax2, marker="o", color="orange")
                    ax2.set_title("Monthly Spending Trend")
                    ax2.set_xlabel("Month")
                    ax2.set_ylabel("Amount")
                    st.pyplot(fig2)
                except Exception:
                    pass

# -------- FOLLOW-UP CHAT --------
st.markdown("---")
st.subheader("🧭 Ask Groq for Deeper Insights")
st.caption("Ask questions like 'How can I reduce my shopping expenses by 20%?'")

user_query = st.text_input("Ask your question:")
if st.button("💬 Ask AI"):
    if not user_query.strip():
        st.warning("Please type a question first.")
    else:
        context = f"""
User question: {user_query}
Previous summary: {results.get('summary', '')}
Recommendations: {results.get('recommendations', '')}
"""
        with st.spinner("Groq thinking..."):
            answer = groq_chat([
                {"role": "system", "content": "You are a financial advisor offering precise, empathetic guidance."},
                {"role": "user", "content": context}
            ])
        st.success("Groq's Insight:")
        st.write(answer)
