import os
import time
from datetime import datetime

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


class LinkedInPostStore:
    def __init__(self):
        self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "linkedin-posts")

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for the index to be ready
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        self._index = pc.Index(index_name)

    def _embed(self, text: str) -> list[float]:
        response = self._openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def search_similar(
        self,
        post_text: str,
        top_k: int = 5,
        distance_threshold: float = 0.3,
    ) -> list[dict]:
        """
        Return stored posts whose cosine distance from post_text is <= distance_threshold.
        Pinecone returns cosine similarity scores (1 = identical), so we convert:
            distance = 1 - score  →  filter: score >= (1 - distance_threshold)
        """
        stats = self._index.describe_index_stats()
        if stats.total_vector_count == 0:
            return []

        embedding = self._embed(post_text)
        similarity_threshold = 1 - distance_threshold

        results = self._index.query(
            vector=embedding,
            top_k=min(top_k, stats.total_vector_count),
            include_metadata=True,
        )

        similar = []
        for match in results.matches:
            if match.score >= similarity_threshold:
                meta = dict(match.metadata)
                text = meta.pop("text", "")
                similar.append({
                    "text": text,
                    "distance": round(1 - match.score, 4),
                    "metadata": meta,
                })

        return similar

    def add_post(self, post_text: str, metadata: dict | None = None) -> str:
        post_id = f"post_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        embedding = self._embed(post_text)

        self._index.upsert(vectors=[{
            "id": post_id,
            "values": embedding,
            "metadata": {"text": post_text, **(metadata or {})},
        }])

        return post_id

    def count(self) -> int:
        return self._index.describe_index_stats().total_vector_count
