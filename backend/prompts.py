from langchain_core.prompts import ChatPromptTemplate

# Final prompt template
final_prompt = ChatPromptTemplate.from_template("""
Answer the question using the retrieved Wikipedia context.

Rules:
- Use ONLY the provided context
- Combine information from multiple contexts when helpful
- Provide a complete but concise answer
- If information the retrieved Wikipedia context is incomplete, mention that clearly
- Do NOT invent unsupported facts
- If the question asks for a list or latest/recent developments and the context includes aspect labels, prefer one distinct answer item per aspect
- If the question asks for the latest or recent developments, clarify when Wikipedia context may not reflect the absolute newest real-time updates

Context:
{context}

Question:
{question}
""")
