import json, os, re, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests, wikipediaapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

CACHE_BASE_DIR = Path(__file__).resolve().parent / "cache"
RAW_CACHE_DIR = CACHE_BASE_DIR / "raw"
CHUNK_CACHE_DIR = CACHE_BASE_DIR / "chunks"
INDEX_PATH = CACHE_BASE_DIR / "index.json"

MAX_RAW_CACHE_PAGES = 100
MAX_CHUNK_CACHE_PAGES = 100
MAX_TOTAL_CACHE_SIZE_BYTES = 100 * 1024 * 1024

for cache_dir in (RAW_CACHE_DIR, CHUNK_CACHE_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)

class QueryList(BaseModel):
    queries: List[str]

class QuestionPlan(BaseModel):
    question_type: str
    queries: List[str]
    entities: List[str]
    aspects: List[str]

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='wikipedia-rag-chatbot/1.0'
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

planner_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.5
).with_structured_output(QuestionPlan)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=70
)

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def get_default_cache_index():
    return {
        "version": 1,
        "entries": {}
    }

def load_cache_index():
    if not INDEX_PATH.exists():
        return get_default_cache_index()

    try:
        with INDEX_PATH.open("r", encoding="utf-8") as index_file:
            payload = json.load(index_file)
            if "entries" not in payload or not isinstance(payload["entries"], dict):
                return get_default_cache_index()
            return payload
    except (json.JSONDecodeError, OSError):
        return get_default_cache_index()

def save_cache_index(index_data):
    with INDEX_PATH.open("w", encoding="utf-8") as index_file:
        json.dump(index_data, index_file, ensure_ascii=False, indent=2)

def get_canonical_key(page):
    source_url = getattr(page, "fullurl", "") or ""

    if source_url:
        parsed = urlparse(source_url)
        marker = "/wiki/"
        if marker in parsed.path:
            return unquote(parsed.path.split(marker, 1)[1]).strip()

    fallback_title = getattr(page, "title", "") or ""
    return fallback_title.strip().replace(" ", "_")

def make_safe_filename(key):
    normalized_key = key.strip().replace(" ", "_")
    safe_name = re.sub(r"[^A-Za-z0-9_.()-]", "_", normalized_key)
    safe_name = safe_name.strip(" .")

    return safe_name or "untitled_page"

def resolve_filename_collision(directory, base_name, suffix=".json"):
    candidate = directory / f"{base_name}{suffix}"
    counter = 1

    while candidate.exists():
        candidate = directory / f"{base_name}__{counter}{suffix}"
        counter += 1

    return candidate

def build_page_metadata(page):
    canonical_key = get_canonical_key(page)
    source_url = getattr(page, "fullurl", "") or ""
    raw_base_name = make_safe_filename(canonical_key)

    return {
        "title": page.title,
        "canonical_key": canonical_key,
        "source_url": source_url,
        "cache_key": raw_base_name,
    }

def get_cache_paths(cache_key):
    return {
        "raw": RAW_CACHE_DIR / f"{cache_key}.json",
        "chunk": CHUNK_CACHE_DIR / f"{cache_key}.json",
    }

def get_cache_stats(index_data):
    entries = index_data["entries"].values()
    return {
        "total_size": sum(entry.get("size_bytes", 0) for entry in entries),
        "raw_count": sum(1 for entry in entries if entry.get("type") == "raw"),
        "chunk_count": sum(1 for entry in entries if entry.get("type") == "chunk"),
    }

def format_cache_size(size_bytes):
    return f"{size_bytes / (1024 * 1024):.2f} MB"

def log_cache_summary(index_data, prefix):
    stats = get_cache_stats(index_data)
    print(
        f"{prefix}: raw_pages={stats['raw_count']}, "
        f"chunk_pages={stats['chunk_count']}, "
        f"total_size={format_cache_size(stats['total_size'])}"
    )

def remove_cache_entry_file(entry):
    try:
        Path(entry["path"]).unlink(missing_ok=True)
    except OSError:
        pass

def remove_page_cache(index_data, cache_key):
    entry_keys = [
        entry_key
        for entry_key in list(index_data["entries"].keys())
        if entry_key.startswith(f"{cache_key}:")
    ]

    for entry_key in entry_keys:
        remove_cache_entry_file(index_data["entries"][entry_key])
        index_data["entries"].pop(entry_key, None)

def evict_cache_if_needed(index_data):
    while True:
        stats = get_cache_stats(index_data)
        limits_ok = (
            stats["total_size"] <= MAX_TOTAL_CACHE_SIZE_BYTES
            and stats["raw_count"] <= MAX_RAW_CACHE_PAGES
            and stats["chunk_count"] <= MAX_CHUNK_CACHE_PAGES
        )

        if limits_ok or not index_data["entries"]:
            break

        oldest_entry_key = min(
            index_data["entries"],
            key=lambda key: index_data["entries"][key].get("last_accessed", "")
        )
        cache_key = oldest_entry_key.split(":", 1)[0]
        print(f"CACHE EVICTED: {cache_key}")
        remove_page_cache(index_data, cache_key)
        log_cache_summary(index_data, "CACHE SUMMARY")

    save_cache_index(index_data)

def touch_cache_entry(index_data, entry_key):
    if entry_key not in index_data["entries"]:
        return

    index_data["entries"][entry_key]["last_accessed"] = utc_now_iso()
    save_cache_index(index_data)

def register_cache_entry(index_data, cache_key, entry_type, title, path, source_url):
    file_path = Path(path)
    entry_key = f"{cache_key}:{entry_type}"
    now = utc_now_iso()
    existing_entry = index_data["entries"].get(entry_key, {})

    index_data["entries"][entry_key] = {
        "type": entry_type,
        "title": title,
        "path": str(file_path),
        "source_url": source_url,
        "size_bytes": file_path.stat().st_size,
        "created_at": existing_entry.get("created_at", now),
        "last_accessed": now,
    }

    save_cache_index(index_data)
    evict_cache_if_needed(index_data)

def load_raw_page_from_cache(cache_key):
    index_data = load_cache_index()
    entry_key = f"{cache_key}:raw"
    entry = index_data["entries"].get(entry_key)

    if not entry:
        return None

    cache_path = Path(entry["path"])
    if not cache_path.exists():
        index_data["entries"].pop(entry_key, None)
        save_cache_index(index_data)
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except (json.JSONDecodeError, OSError):
        index_data["entries"].pop(entry_key, None)
        save_cache_index(index_data)
        return None

    touch_cache_entry(index_data, entry_key)
    print(f"RAW CACHE HIT: {cache_key}")
    return payload

def save_raw_page_to_cache(page_metadata, page_text):
    cache_key = page_metadata["cache_key"]
    cache_path = get_cache_paths(cache_key)["raw"]
    index_data = load_cache_index()

    payload = {
        "version": 1,
        "title": page_metadata["title"],
        "canonical_key": page_metadata["canonical_key"],
        "source_url": page_metadata["source_url"],
        "cache_key": cache_key,
        "page_content": page_text,
        "retrieved_at": utc_now_iso(),
    }

    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, ensure_ascii=False, indent=2)

    print(f"RAW CACHE SAVE: {cache_key}")
    register_cache_entry(
        index_data=index_data,
        cache_key=cache_key,
        entry_type="raw",
        title=page_metadata["title"],
        path=cache_path,
        source_url=page_metadata["source_url"],
    )
    log_cache_summary(load_cache_index(), "CACHE SUMMARY")

def load_chunk_cache(cache_key):
    index_data = load_cache_index()
    entry_key = f"{cache_key}:chunk"
    entry = index_data["entries"].get(entry_key)

    if not entry:
        return None

    cache_path = Path(entry["path"])
    if not cache_path.exists():
        index_data["entries"].pop(entry_key, None)
        save_cache_index(index_data)
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except (json.JSONDecodeError, OSError):
        index_data["entries"].pop(entry_key, None)
        save_cache_index(index_data)
        return None

    touch_cache_entry(index_data, entry_key)
    print(f"CHUNK CACHE HIT: {cache_key}")
    return payload

def save_chunk_cache(page_metadata, docs):
    cache_key = page_metadata["cache_key"]
    cache_path = get_cache_paths(cache_key)["chunk"]
    index_data = load_cache_index()

    payload = {
        "version": 1,
        "title": page_metadata["title"],
        "canonical_key": page_metadata["canonical_key"],
        "source_url": page_metadata["source_url"],
        "cache_key": cache_key,
        "generated_at": utc_now_iso(),
        "chunks": [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ],
    }

    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, ensure_ascii=False, indent=2)

    print(f"CHUNK CACHE SAVE: {cache_key}")
    register_cache_entry(
        index_data=index_data,
        cache_key=cache_key,
        entry_type="chunk",
        title=page_metadata["title"],
        path=cache_path,
        source_url=page_metadata["source_url"],
    )
    log_cache_summary(load_cache_index(), "CACHE SUMMARY")

def hydrate_docs_from_chunk_payload(payload):
    return [
        Document(
            page_content=chunk["page_content"],
            metadata=chunk["metadata"],
        )
        for chunk in payload.get("chunks", [])
    ]

def split_by_sections(docs):
    split_docs = []

    for doc in docs:
        title = doc.metadata.get("title", "")
        canonical_key = doc.metadata.get("canonical_key", "")
        source_url = doc.metadata.get("source_url", "")
        cache_key = doc.metadata.get("cache_key", "")
        retrieval_aspect = doc.metadata.get("retrieval_aspect")

        # Split on Wikipedia headings
        sections = re.split(r"(==+.*?==+)", doc.page_content)

        current_heading = "Introduction"

        for part in sections:
            if re.match(r"==+.*?==+", part):
                current_heading = part.strip("=").strip()
            else:
                content = part.strip()

                if len(content) > 100:
                    split_docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "title": title,
                                "section": current_heading,
                                "canonical_key": canonical_key,
                                "source_url": source_url,
                                "cache_key": cache_key,
                                "retrieval_aspect": retrieval_aspect,
                            }
                        )
                    )

    return split_docs

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

wiki_cache = {}

# Multi-Query Retriever
def wiki_retriever_multi(question):
    section_cache = {}

    def clean_topic_query(question_text):
        cleaned = question_text.strip()
        cleaned = re.sub(r"(?i)\b(explain|describe|summarize|tell me about|in easy way|simply)\b", "", cleaned)
        cleaned = re.sub(r"(?i)\b(latest|recent|current)\b", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.,")
        return cleaned[:100]

    def safe_wikipedia_load(query, retrieval_aspect=None):
        docs = []

        try:
            url = "https://en.wikipedia.org/w/api.php"

            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 5
            }

            headers = {"User-Agent": "wikipedia-rag-chatbot/1.0"}

            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()
            search_results = data["query"]["search"]

            for result in search_results:
                title = result["title"]
                page = wiki.page(title)

                if page.exists():
                    page_metadata = build_page_metadata(page)
                    cached_page = load_raw_page_from_cache(page_metadata["cache_key"])
                    page_text = cached_page["page_content"] if cached_page else page.text

                    if not cached_page:
                        save_raw_page_to_cache(page_metadata, page_text)

                    if retrieval_aspect:
                        page_metadata = {
                            **page_metadata,
                            "retrieval_aspect": retrieval_aspect,
                        }

                    docs.append(
                        Document(
                            page_content=page_text,
                            metadata=page_metadata
                        )
                    )

                time.sleep(0.2)

        except Exception as e:
            print(f"Wikipedia fetch failed: {e}")

        return docs

    def load_entity_docs(entity_name):
        docs = []

        try:
            exact_page = wiki.page(entity_name)

            if exact_page.exists():
                page_metadata = build_page_metadata(exact_page)
                cached_page = load_raw_page_from_cache(page_metadata["cache_key"])
                page_text = cached_page["page_content"] if cached_page else exact_page.text

                if not cached_page:
                    save_raw_page_to_cache(page_metadata, page_text)

                docs.append(
                    Document(
                        page_content=page_text,
                        metadata=page_metadata
                    )
                )

        except Exception as e:
            print(f"Exact Wikipedia page load failed for {entity_name}: {e}")

        docs.extend(safe_wikipedia_load(entity_name))
        return docs

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

    def collect_docs_for_aspects(aspects, seen_titles):
        aspect_docs = []

        for aspect in aspects[:3]:
            docs = safe_wikipedia_load(aspect, retrieval_aspect=aspect)

            for doc in docs:
                canonical_key = doc.metadata.get("canonical_key", "")
                if canonical_key not in seen_titles:
                    seen_titles.add(canonical_key)
                    aspect_docs.append(doc)

        return aspect_docs

    plan = analyze_question(question)
    queries = build_retrieval_queries(question, plan)
    print(f"question_plan:\n{plan}\n")
    print(f"queries:\n{queries}\n")

    all_docs = []
    seen_titles = set()

    if plan["question_type"] == "comparison" and len(plan["entities"]) >= 2:
        for entity in plan["entities"][:3]:
            docs = load_entity_docs(entity)

            for doc in docs:
                canonical_key = doc.metadata.get("canonical_key", "")
                if canonical_key not in seen_titles:
                    seen_titles.add(canonical_key)
                    all_docs.append(doc)
    elif plan["question_type"] in {"list", "recent_or_latest"} and plan["aspects"]:
        all_docs.extend(collect_docs_for_aspects(plan["aspects"], seen_titles))

    # Fetch docs for each query
    for q in queries:
        docs = safe_wikipedia_load(q)

        # Deduplicate by title
        for doc in docs:
            canonical_key = doc.metadata.get("canonical_key", "")
            if canonical_key not in seen_titles:
                seen_titles.add(canonical_key)
                all_docs.append(doc)

    if not all_docs:
        raise ValueError("No Wikipedia documents retrieved.")

    # Chunk with cache reuse
    section_docs = []
    for doc in all_docs:
        cache_key = doc.metadata.get("cache_key", "")

        if cache_key in section_cache:
            section_docs.extend(section_cache[cache_key])
            continue

        cached_chunks = load_chunk_cache(cache_key)
        if cached_chunks:
            hydrated_docs = hydrate_docs_from_chunk_payload(cached_chunks)
            section_cache[cache_key] = hydrated_docs
            section_docs.extend(hydrated_docs)
            continue

        page_section_docs = split_by_sections([doc])
        section_cache[cache_key] = page_section_docs
        section_docs.extend(page_section_docs)
        save_chunk_cache(doc.metadata, page_section_docs)

    splits = splitter.split_documents(section_docs)

    # Embed + MMR retrieval
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 15}
    )

    all_results = []

    for q in queries:
        results = retriever.invoke(q)
        all_results.extend(results)

    unique_chunks = {}

    for doc in all_results:
        key = doc.page_content[:200]

        if key not in unique_chunks:
            unique_chunks[key] = doc

    return list(unique_chunks.values())

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

# Format retrieved docs
def format_docs(docs):
    formatted_docs = "\n\n".join(
        f"[Article: {doc.metadata.get('title','')}]\n"
        f"{'[Aspect: ' + doc.metadata.get('retrieval_aspect', '') + ']\n' if doc.metadata.get('retrieval_aspect') else ''}"
        f"[Section: {doc.metadata.get('section','Unknown')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )
    print(f"formatted_docs:\n{formatted_docs}\n")   # Print formatted_docs in the terminal
    return formatted_docs

# RAG chain
rag_chain = (
    {"context": RunnablePassthrough() | wiki_retriever_multi | format_docs, "question": RunnablePassthrough()}
    | final_prompt
    | llm
    | StrOutputParser()
)

def ask_question(question):
    response = rag_chain.invoke(question)
    print(f"response:\n{response}\n")   # Print response in the terminal
    return response
