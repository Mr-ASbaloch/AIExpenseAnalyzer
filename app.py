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
MODEL = "llama-3.3-70b-versatile"  # Groq LLM model

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
def analyze_expenses(df):
    system_prompt = """You are an expert financial data analyst.
Read the provided expense data and produce a JSON object:
{
 "summary": "concise paragraph about spending patterns, totals, and category trends",
 "recommendations": "personalized saving and optimization strategies"
}
Return only valid JSON, no explanations.
"""
    csv_data = df.to_csv(index=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": csv_data}
    ]
    reply = groq_chat(messages)

    # Try parsing JSON safely
    try:
        data = json.loads(reply)
    except Exception:
        import re
        match = re.search(r"\{.*\}", reply, re.DOTALL)
        data = json.loads(match.group(0)) if match else {"summary": reply, "recommendations": ""}
    return data

# =============================
# STREAMLIT APP
# =============================
st.set_page_config(page_title="💸 AI Expense Analyzer", layout="wide")
st.title("💸 AI Expense Analyzer using Groq LLM + Streamlit")

st.markdown("""
Easily add your daily expenses and let **Groq LLM** analyze your spending.  
It will summarize your habits and recommend smart saving strategies 💡.
""")

# -----------------------------
# Initialize Session
# -----------------------------
if "entries" not in st.session_state:
    st.session_state.entries = []

# -----------------------------
# Input Fields
# -----------------------------
st.subheader("➕ Add Expense Entry")

col1, col2, col3, col4 = st.columns(4)
with col1:
    date = st.date_input("Date")
with col2:
    category = st.selectbox("Category", ["Food", "Transport", "Bills", "Shopping", "Other"])
with col3:
    amount = st.number_input("Amount (PKR)", min_value=0.0, step=100.0)
with col4:
    description = st.text_input("Description")

# Add Entry Button
if st.button("Add New Row"):
    if amount > 0 and description:
        st.session_state.entries.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Category": category,
            "Amount": amount,
            "Description": description
        })
        st.success("✅ Entry added successfully!")
    else:
        st.warning("⚠️ Please fill all fields correctly.")

# -----------------------------
# Display Table
# -----------------------------
if st.session_state.entries:
    df = pd.DataFrame(st.session_state.entries)
    st.dataframe(df, use_container_width=True)

    # Analyze Button
    if st.button("🔍 Analyze Expenses"):
        with st.spinner("Analyzing your expenses using Groq..."):
            results = analyze_expenses(df)

        st.subheader("📊 Expense Summary")
        st.write(results.get("summary", "No summary generated."))

        st.subheader("💡 AI Recommendations")
        st.write(results.get("recommendations", "No recommendations generated."))

        # Visualizations
        if not df.empty:
            cat_sum = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Bar Chart")
                fig_bar, ax_bar = plt.subplots()
                cat_sum.plot(kind="bar", ax=ax_bar, color="skyblue")
                ax_bar.set_title("Spending by Category")
                st.pyplot(fig_bar)

            with col2:
                st.markdown("#### 🥧 Pie Chart")
                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie(cat_sum, labels=cat_sum.index, autopct="%1.1f%%", startangle=90)
                ax_pie.axis("equal")
                st.pyplot(fig_pie)

else:
    st.info("No entries yet. Add your first expense above!")



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
