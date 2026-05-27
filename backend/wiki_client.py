import cache_store, wikipediaapi, re, requests, time
from urllib.parse import unquote, urlparse
from langchain_core.documents import Document

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='wikipedia-rag-chatbot/1.0'
)

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
                cached_page = cache_store.load_raw_page_from_cache(page_metadata["cache_key"])
                page_text = cached_page["page_content"] if cached_page else page.text

                if not cached_page:
                    cache_store.save_raw_page_to_cache(page_metadata, page_text)

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
            cached_page = cache_store.load_raw_page_from_cache(page_metadata["cache_key"])
            page_text = cached_page["page_content"] if cached_page else exact_page.text

            if not cached_page:
                cache_store.save_raw_page_to_cache(page_metadata, page_text)

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
