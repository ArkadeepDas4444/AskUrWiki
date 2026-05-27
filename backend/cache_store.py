import json
from datetime import datetime, timezone
from pathlib import Path

CACHE_BASE_DIR = Path(__file__).resolve().parent / "cache"
RAW_CACHE_DIR = CACHE_BASE_DIR / "raw"
CHUNK_CACHE_DIR = CACHE_BASE_DIR / "chunks"
INDEX_PATH = CACHE_BASE_DIR / "index.json"

MAX_RAW_CACHE_PAGES = 100
MAX_CHUNK_CACHE_PAGES = 100
MAX_TOTAL_CACHE_SIZE_BYTES = 100 * 1024 * 1024

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
