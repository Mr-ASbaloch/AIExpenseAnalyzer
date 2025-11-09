import os
import io
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =============================
# CONFIGURATION
# =============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"  # Groq LLM model (adjust if needed)

# =============================
# HELPER: Groq Chat Request
# =============================
def groq_chat(messages, temperature=0.3, max_tokens=800):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    response = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# =============================
# HELPER: Analyze Expenses via Groq
# =============================
def analyze_expenses(raw_text):
    system_prompt = """You are an expert financial data analyst.
Read the user's provided expense text or CSV-like input and produce a JSON object:
{
 "transactions":[{"date":"YYYY-MM-DD","merchant":"string","amount":float,"category":"string"}],
 "summary":"concise paragraph about spending patterns, totals, and category trends",
 "recommendations":"personalized saving and optimization strategies"
}
Return only valid JSON, no explanations.
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
This AI tool uses **Groq's LLM** to understand, categorize, and analyze your spending habits.  
Just paste your expenses or upload a CSV — it will summarize your spending, visualize it,  
and recommend **smart saving strategies** 💡.
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

        # Display results
        st.subheader("📊 Expense Summary")
        st.write(results.get("summary", "No summary generated."))

        st.subheader("💡 AI Recommendations")
        st.write(results.get("recommendations", "No recommendations generated."))

        # Display transactions
        transactions = results.get("transactions", [])
        if transactions:
            df = pd.DataFrame(transactions)
            st.dataframe(df, use_container_width=True)

            # Category Breakdown Chart
            if "category" in df.columns and "amount" in df.columns:
                cat_sum = df.groupby("category")["amount"].sum().sort_values(ascending=False)
                fig, ax = plt.subplots()
                cat_sum.plot(kind="bar", ax=ax)
                ax.set_title("Spending by Category")
                ax.set_ylabel("Amount")
                st.pyplot(fig)

            # Monthly Trend
            if "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"])
                    df["month"] = df["date"].dt.to_period("M").astype(str)
                    trend = df.groupby("month")["amount"].sum()
                    fig2, ax2 = plt.subplots()
                    trend.plot(ax=ax2, marker="o")
                    ax2.set_title("Monthly Spending Trend")
                    ax2.set_ylabel("Amount")
                    st.pyplot(fig2)
                except Exception:
                    pass

# -------- FOLLOW-UP CHAT --------
st.markdown("---")
st.subheader("🧭 Ask Groq for Deeper Insights")
st.caption("Ask AI any question about your spending — like 'Where can I save most?'")

user_query = st.text_input("Ask your question:")
if st.button("💬 Ask AI"):
    if not user_query.strip():
        st.warning("Please type a question first.")
    else:
        context = f"""
User question: {user_query}
Context: Previous summary and recommendations.
"""
        with st.spinner("Groq thinking..."):
            answer = groq_chat([
                {"role": "system", "content": "You are a helpful AI financial assistant."},
                {"role": "user", "content": context}
            ])
        st.success("Groq's Insight:")
        st.write(answer)
