import wiki_client, cache_store, chunking, question_planner, prompts, os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from model_names import final_model

load_dotenv()

for cache_dir in (cache_store.RAW_CACHE_DIR, cache_store.CHUNK_CACHE_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=final_model,
    temperature=0.2
)

# Multi-Query Retriever
def wiki_retriever_multi(question):
    section_cache = {}

    def collect_docs_for_aspects(aspects, seen_titles):
        aspect_docs = []

        for aspect in aspects[:3]:
            docs = wiki_client.safe_wikipedia_load(aspect, retrieval_aspect=aspect)

            for doc in docs:
                canonical_key = doc.metadata.get("canonical_key", "")
                if canonical_key not in seen_titles:
                    seen_titles.add(canonical_key)
                    aspect_docs.append(doc)

        return aspect_docs

    plan = question_planner.analyze_question(question)
    queries = question_planner.build_retrieval_queries(question, plan)
    print(f"question_plan:\n{plan}\n")
    print(f"queries:\n{queries}\n")

    all_docs = []
    seen_titles = set()

    if plan["question_type"] == "comparison" and len(plan["entities"]) >= 2:
        for entity in plan["entities"][:3]:
            docs = wiki_client.load_entity_docs(entity)

            for doc in docs:
                canonical_key = doc.metadata.get("canonical_key", "")
                if canonical_key not in seen_titles:
                    seen_titles.add(canonical_key)
                    all_docs.append(doc)
    elif plan["question_type"] in {"list", "recent_or_latest"} and plan["aspects"]:
        all_docs.extend(collect_docs_for_aspects(plan["aspects"], seen_titles))

    # Fetch docs for each query
    for q in queries:
        docs = wiki_client.safe_wikipedia_load(q)

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

        cached_chunks = cache_store.load_chunk_cache(cache_key)
        if cached_chunks:
            hydrated_docs = chunking.hydrate_docs_from_chunk_payload(cached_chunks)
            section_cache[cache_key] = hydrated_docs
            section_docs.extend(hydrated_docs)
            continue

        page_section_docs = chunking.split_by_sections([doc])
        section_cache[cache_key] = page_section_docs
        section_docs.extend(page_section_docs)
        cache_store.save_chunk_cache(doc.metadata, page_section_docs)

    splits = chunking.splitter.split_documents(section_docs)

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
    | prompts.final_prompt
    | llm
    | StrOutputParser()
)

def ask_question(question):
    response = rag_chain.invoke(question)
    print(f"response:\n{response}\n")   # Print response in the terminal
    return response
