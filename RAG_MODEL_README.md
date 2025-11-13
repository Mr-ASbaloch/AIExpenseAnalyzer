# RAG Model for AI Expense Analyzer

## Overview

This implementation adds Retrieval-Augmented Generation (RAG) capabilities to the AI Expense Analyzer application. The RAG model enhances the expense analysis by retrieving relevant financial knowledge and providing context-aware recommendations.

## What is RAG?

RAG (Retrieval-Augmented Generation) is an AI technique that combines:
1. **Retrieval**: Finding relevant information from a knowledge base
2. **Generation**: Using that information to generate better, more informed responses

## Features

### 1. Financial Knowledge Base
The RAG model includes a comprehensive knowledge base with financial advice for:
- **Food**: Meal planning, bulk buying, reducing waste
- **Transport**: Carpooling, public transportation, fuel efficiency
- **Bills**: Subscription management, negotiating rates, energy efficiency
- **Shopping**: Budget creation, impulse control, quality over quantity
- **Savings**: Emergency funds, 50/30/20 rule, automated savings
- **Budget**: Expense tracking, categorization, regular reviews
- **General**: Overall financial wellness strategies

### 2. Document Retrieval
Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and cosine similarity to:
- Find the most relevant financial advice based on user queries
- Retrieve context-specific recommendations for spending categories
- Match user questions with appropriate knowledge base documents

### 3. Context-Aware Responses
Generates enhanced prompts that include:
- Current expense data and statistics
- Retrieved relevant financial advice
- User's specific question
- Historical expense patterns

### 4. Category-Specific Insights
Automatically provides targeted advice for:
- Top spending categories
- Areas where users spend the most
- Specific recommendations for each expense type

## Technical Implementation

### Core Components

1. **RAGModel Class** (`rag_model.py`)
   - Manages the financial knowledge base
   - Implements TF-IDF vectorization for document retrieval
   - Builds context from expense data
   - Generates enhanced prompts for LLM

2. **Knowledge Base**
   - 7 categories of financial advice
   - Vectorized for efficient retrieval
   - Easily extensible with more documents

3. **Integration with Streamlit**
   - Session state management for RAG model
   - Expense history tracking
   - Real-time knowledge retrieval

### Key Methods

- `retrieve_relevant_knowledge(query, top_k)`: Finds most relevant documents
- `build_context_from_expenses(df)`: Creates expense summary context
- `generate_rag_enhanced_prompt(query, context)`: Builds enhanced prompts
- `get_category_specific_advice(category)`: Returns category-specific tips

## Usage

### 1. Adding Expenses
When you add an expense, it's automatically:
- Stored in the session
- Added to RAG model history
- Available for context-aware queries

### 2. Analyzing Expenses
Click "🔍 Analyze Expenses" to:
- Get AI-powered summary from Groq
- Receive personalized recommendations
- View RAG-enhanced category-specific tips (expandable sections)

### 3. Asking Questions with RAG
Use the "Ask AI with RAG" feature to:
- Ask specific questions about your spending
- Get responses enhanced with relevant financial knowledge
- See which knowledge sources were used (expandable "Knowledge Sources Used")

## Benefits of RAG Implementation

1. **Better Recommendations**: Combines real expense data with proven financial advice
2. **Context-Aware**: Understands both your spending patterns and general best practices
3. **Transparent**: Shows which knowledge sources were used for each response
4. **Scalable**: Easy to add more financial knowledge documents
5. **Efficient**: Fast retrieval using TF-IDF vectorization

## Technical Requirements

- Python 3.7+
- scikit-learn (for TF-IDF and similarity calculations)
- numpy (for array operations)
- pandas (for data manipulation)
- streamlit (for web interface)

## Future Enhancements

Potential improvements for the RAG model:
1. **Persistent Storage**: Save expense history and learned patterns
2. **Advanced Embeddings**: Use neural embeddings (e.g., sentence-transformers)
3. **Vector Database**: Integrate with Pinecone, Weaviate, or FAISS
4. **Dynamic Knowledge Base**: Allow users to add custom financial advice
5. **Multi-modal RAG**: Include images, charts, and graphs in retrieval
6. **Personalized Learning**: Adapt recommendations based on user behavior
7. **External Data**: Retrieve information from financial websites and APIs

## Example Queries

Try asking:
- "Where can I save the most money?"
- "How can I reduce my food expenses?"
- "What's the best way to manage my transport costs?"
- "Give me tips for my highest spending category"
- "How should I budget my monthly expenses?"

## Testing

Run the test suite to verify RAG functionality:
```bash
cd /home/runner/work/AIExpenseAnalyzer/AIExpenseAnalyzer
PYTHONPATH=. python /tmp/test_rag_model.py
```

All tests should pass, verifying:
- RAG initialization
- Knowledge retrieval
- Expense context building
- Enhanced prompt generation
- Category advice retrieval
- Expense history management

## Architecture

```
User Query
    ↓
RAG Model
    ↓
1. Build expense context from data
2. Retrieve relevant financial knowledge (TF-IDF + Cosine Similarity)
3. Combine context + knowledge + query
    ↓
Enhanced Prompt → Groq LLM
    ↓
Context-Aware Response + Knowledge Sources
```

## Conclusion

The RAG model implementation transforms the AI Expense Analyzer from a simple analysis tool into an intelligent financial assistant that combines your personal spending data with expert financial knowledge to provide truly personalized, actionable advice.
