import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=70
)

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
