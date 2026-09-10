from langchain_core.prompts import ChatPromptTemplate

# Question analysis prompt template
question_analysis_prompt = ChatPromptTemplate.from_template("""
Analyze the user question for a Wikipedia-based RAG system.

Return:
- question_type: one of ["comparison", "single_entity", "multi_entity", "list", "recent_or_latest", "explanation_or_broad_topic"]
- entities: Python list of named people, organizations, places, theories, or topics that should be retrieved separately if is question_type is "comparison", "single_entity" or "multi_entity"; otherwise return an empty list
- aspects: list of 3 distinct subtopics or angles if question_type is "list" or "recent_or_latest"; otherwise return an empty list
- queries: list of 3 concise Wikipedia search queries that improve retrieval quality

Rules:
- For "comparison" questions, identify the compared entities explicitly
- For "recent_or_latest" questions, make aspects diverse rather than near-duplicate paraphrases
- For "recent_or_latest" questions, still produce Wikipedia-friendly topic queries instead of news-style wording
- Keep queries short, specific, and useful for Wikipedia search
- Avoid generic filler such as "explained in simple terms"

Question:
{question}
""")

# Final RAG prompt template
final_prompt = ChatPromptTemplate.from_template("""
You are a helpful Wikipedia chatbot based on RAG. You have already retrieved some context from Wikipedia. Now, answer the question using the context.

# Rules:
- Use ONLY the retrieved Wikipedia context
- Combine information from multiple contexts when helpful
- Provide a complete but concise answer
- If information the retrieved Wikipedia context is incomplete, mention that clearly
- Do NOT invent unsupported facts
- If the question asks for a list or latest/recent developments and the context includes aspect labels, prefer one distinct answer item per aspect
- If the question asks for the latest or recent developments, clarify when Wikipedia context may not reflect the absolute newest real-time updates
- Return in text/markdown format
- Do NOT return in tables unless necessary

# Context:
```
{context}
```

# Question:
{question}
""")
