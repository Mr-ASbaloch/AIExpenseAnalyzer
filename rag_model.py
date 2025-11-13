"""
RAG (Retrieval-Augmented Generation) Model Module
This module implements a simple RAG system for expense analysis using:
- In-memory vector storage
- TF-IDF based document retrieval
- Context-aware response generation
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple


class RAGModel:
    """
    RAG Model for expense analysis with document retrieval capabilities.
    """
    
    def __init__(self):
        """Initialize the RAG model with financial knowledge base."""
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.documents = []
        self.document_vectors = None
        self.knowledge_base = self._initialize_knowledge_base()
        self.expense_history = []
        
    def _initialize_knowledge_base(self) -> List[Dict[str, str]]:
        """
        Initialize the financial advice knowledge base.
        
        Returns:
            List of knowledge documents with category and content.
        """
        knowledge_base = [
            {
                "category": "Food",
                "content": "To reduce food expenses, consider meal planning, cooking at home more often, "
                          "buying in bulk, using grocery lists, and avoiding impulse purchases. "
                          "Track food waste and plan meals around sales and seasonal produce."
            },
            {
                "category": "Transport",
                "content": "Save on transport by carpooling, using public transportation, combining trips, "
                          "maintaining your vehicle properly, and considering fuel-efficient routes. "
                          "Walk or bike for short distances when possible."
            },
            {
                "category": "Bills",
                "content": "Optimize bills by reviewing subscriptions, negotiating rates, switching providers, "
                          "using energy-efficient appliances, and setting up automatic payments to avoid late fees. "
                          "Consider bundling services for discounts."
            },
            {
                "category": "Shopping",
                "content": "Reduce shopping expenses by creating a budget, waiting 24 hours before non-essential purchases, "
                          "using coupons and cashback apps, buying quality items that last longer, "
                          "and distinguishing between wants and needs."
            },
            {
                "category": "General",
                "content": "Build an emergency fund with 3-6 months of expenses, follow the 50/30/20 rule "
                          "(50% needs, 30% wants, 20% savings), track all expenses, and review spending monthly. "
                          "Set specific financial goals and automate savings."
            },
            {
                "category": "Savings",
                "content": "Maximize savings by automating transfers to savings accounts, taking advantage of "
                          "employer matching for retirement accounts, reducing high-interest debt first, "
                          "and using high-yield savings accounts. Start small if needed but be consistent."
            },
            {
                "category": "Budget",
                "content": "Create an effective budget by tracking income and expenses, categorizing spending, "
                          "identifying areas to cut back, and reviewing regularly. Use budgeting apps or "
                          "spreadsheets to monitor progress and adjust as needed."
            }
        ]
        
        # Extract content for vectorization
        self.documents = [doc["content"] for doc in knowledge_base]
        if self.documents:
            self.document_vectors = self.vectorizer.fit_transform(self.documents)
        
        return knowledge_base
    
    def add_expense_to_history(self, expense_data: Dict) -> None:
        """
        Add expense data to the historical context.
        
        Args:
            expense_data: Dictionary containing expense information
        """
        self.expense_history.append(expense_data)
    
    def retrieve_relevant_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Retrieve the most relevant knowledge base documents for a query.
        
        Args:
            query: The search query
            top_k: Number of top documents to retrieve
            
        Returns:
            List of relevant knowledge documents
        """
        if not self.documents or self.document_vectors is None:
            return []
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return relevant documents
        relevant_docs = [self.knowledge_base[i] for i in top_indices if similarities[i] > 0]
        
        return relevant_docs
    
    def build_context_from_expenses(self, df) -> str:
        """
        Build context from expense dataframe for RAG.
        
        Args:
            df: Pandas DataFrame with expense data
            
        Returns:
            Context string summarizing expenses
        """
        if df.empty:
            return "No expense data available."
        
        # Calculate statistics
        total_spending = df["Amount"].sum()
        category_totals = df.groupby("Category")["Amount"].sum().to_dict()
        avg_expense = df["Amount"].mean()
        
        # Build context
        context = f"Total spending: {total_spending:.2f} PKR. "
        context += f"Average expense: {avg_expense:.2f} PKR. "
        context += "Spending by category: "
        context += ", ".join([f"{cat}: {amt:.2f} PKR" for cat, amt in category_totals.items()])
        
        return context
    
    def generate_rag_enhanced_prompt(self, query: str, expense_context: str) -> Tuple[str, List[Dict]]:
        """
        Generate an enhanced prompt with retrieved knowledge for RAG.
        
        Args:
            query: User query
            expense_context: Context built from expense data
            
        Returns:
            Tuple of (enhanced prompt, relevant documents)
        """
        # Retrieve relevant knowledge
        relevant_docs = self.retrieve_relevant_knowledge(query, top_k=2)
        
        # Build enhanced context
        enhanced_prompt = f"Expense Context: {expense_context}\n\n"
        
        if relevant_docs:
            enhanced_prompt += "Relevant Financial Advice:\n"
            for i, doc in enumerate(relevant_docs, 1):
                enhanced_prompt += f"{i}. [{doc['category']}] {doc['content']}\n"
        
        enhanced_prompt += f"\nUser Question: {query}\n"
        enhanced_prompt += "Please provide a detailed, context-aware response based on the expense data and financial advice above."
        
        return enhanced_prompt, relevant_docs
    
    def get_category_specific_advice(self, category: str) -> str:
        """
        Get specific advice for a spending category.
        
        Args:
            category: The expense category
            
        Returns:
            Advice string for the category
        """
        for doc in self.knowledge_base:
            if doc["category"].lower() == category.lower():
                return doc["content"]
        
        # Return general advice if category not found
        for doc in self.knowledge_base:
            if doc["category"] == "General":
                return doc["content"]
        
        return "No specific advice available for this category."
