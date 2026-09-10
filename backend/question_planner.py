import os, re
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from langchain_groq import ChatGroq
from model_names import planner_model
from prompts import question_analysis_prompt

load_dotenv()

class QuestionPlan(BaseModel):
    question_type: str
    queries: List[str]
    entities: List[str]
    aspects: List[str]

planner_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=planner_model,
    temperature=0.2
).with_structured_output(
    QuestionPlan,
    # The default is function_calling, which makes Groq require a tool call.
    # GPT-OSS can return the schema directly via Groq's native structured output.
    method="json_schema",
    strict=True,
)

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
    cleaned_topic_query = clean_topic_query(question_text)

    if question_type in {"comparison", "single_entity", "multi_entity"}:
        queries.extend(entities[:3])
        for entity in entities[:3]:
            queries.append(f"about {entity}")
    elif question_type in {"list", "recent_or_latest"}:
        queries.extend(aspects[:3])
        if cleaned_topic_query:
            queries.append(cleaned_topic_query)
    elif cleaned_topic_query:
        queries.append(cleaned_topic_query)

    deduped_queries = []
    seen_queries = set()

    for query in queries:
        cleaned_query = query.strip()
        if cleaned_query and cleaned_query not in seen_queries:
            seen_queries.add(cleaned_query)
            deduped_queries.append(cleaned_query[:100])

    return deduped_queries
