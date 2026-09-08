import os, re
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from model_names import planner_model

load_dotenv()

class QuestionPlan(BaseModel):
    question_type: str
    queries: List[str]
    entities: List[str]
    aspects: List[str]

planner_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=planner_model,
    temperature=0.5
).with_structured_output(QuestionPlan)

# Question analysis prompt template
question_analysis_prompt = ChatPromptTemplate.from_template("""
Analyze the user question for a Wikipedia-based RAG system.

Return:
- question_type: one of [comparison, single_entity, multi_entity, broad_topic, list, recent_or_latest, explanation]
- entities: named people, organizations, places, theories, or topics that should be retrieved separately when useful
- aspects: 3 distinct subtopics or angles for list/latest questions; otherwise return an empty list
- queries: 3 concise Wikipedia search queries that improve retrieval quality

Rules:
- For comparison questions, identify the compared entities explicitly
- For list or latest questions, make aspects diverse rather than near-duplicate paraphrases
- For "latest" or "recent" questions, still produce Wikipedia-friendly topic queries instead of news-style wording
- Keep queries short, specific, and useful for Wikipedia search
- Avoid generic filler such as "explained in simple terms"

Question:
{question}
""")

def analyze_question(question):
    result = (
        question_analysis_prompt
        | planner_llm
    ).invoke({"question": question})

    def clean_query(q):
        q = q.replace('"', '')
        q = q.replace("Wikipedia", "")
        q = q.strip()
        return q[:100]

    cleaned_queries = [clean_query(q) for q in result.queries[:3] if clean_query(q)]
    cleaned_entities = [entity.strip() for entity in result.entities if entity.strip()]
    cleaned_aspects = [clean_query(aspect) for aspect in result.aspects[:3] if clean_query(aspect)]
    question_type = (result.question_type or "broad_topic").strip().lower()

    return {
        "question_type": question_type,
        "queries": cleaned_queries,
        "entities": cleaned_entities,
        "aspects": cleaned_aspects,
    }

def clean_topic_query(question_text):
    cleaned = question_text.strip()
    cleaned = re.sub(r"(?i)\b(explain|describe|summarize|tell me about|in easy way|simply)\b", "", cleaned)
    cleaned = re.sub(r"(?i)\b(latest|recent|current)\b", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.,")
    return cleaned[:100]

def build_retrieval_queries(question_text, plan):
    queries = list(plan["queries"])
    question_type = plan["question_type"]
    entities = plan["entities"]
    aspects = plan["aspects"]
    broad_topic_query = clean_topic_query(question_text)

    if question_type == "comparison" and len(entities) >= 2:
        for entity in entities[:3]:
            queries.append(entity)
            queries.append(f"{entity} research")
            queries.append(f"{entity} contributions")
    elif question_type in {"single_entity", "multi_entity"}:
        for entity in entities[:3]:
            queries.append(entity)
    elif question_type == "list":
        queries.extend(aspects[:3])
        if broad_topic_query:
            queries.append(broad_topic_query)
    elif question_type in {"recent_or_latest", "broad_topic", "explanation"}:
        if question_type == "recent_or_latest":
            queries.extend(aspects[:3])
        if broad_topic_query:
            queries.append(broad_topic_query)

    deduped_queries = []
    seen_queries = set()

    for query in queries:
        cleaned_query = query.strip()
        if cleaned_query and cleaned_query not in seen_queries:
            seen_queries.add(cleaned_query)
            deduped_queries.append(cleaned_query[:100])

    return deduped_queries
