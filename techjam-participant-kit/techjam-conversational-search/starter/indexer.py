from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items() if v)
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    return str(value)


class CatalogIndexer:
    """Dedicated indexer for FAISS vector search with disk caching and progress tracking."""

    def __init__(
        self,
        catalog_path: Path,
        encoder: SentenceTransformer,
        cache_dir: Path = Path("data/index_cache"),
        batch_size: int = 256
    ) -> None:
        self.catalog_path = catalog_path
        self.encoder = encoder
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "catalog_faiss.index"
        self.asins_file = self.cache_dir / "asins.json"

        self.parent_asins: List[str] = []
        self.faiss_index: faiss.IndexFlatIP | None = None

        self._load_or_build()

    def _load_or_build(self) -> None:
        """Loads cached FAISS index from disk or builds it with progress tracking."""
        if self.index_file.exists() and self.asins_file.exists():
            print(f"[FAISS Indexer] Loading cached index from {self.cache_dir}...")
            self.faiss_index = faiss.read_index(str(self.index_file))
            with self.asins_file.open("r", encoding="utf-8") as f:
                self.parent_asins = json.load(f)
            print(f"[FAISS Indexer] Cached index loaded successfully ({len(self.parent_asins)} products).")
        else:
            print(f"[FAISS Indexer] No cache found. Building FAISS index from {self.catalog_path}...")
            self._build_index()

    def _build_index(self) -> None:
        documents: List[str] = []
        asins: List[str] = []

        # Read catalog records
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                title = _clean_text(product.get("title"))
                cats = _clean_text(product.get("categories"))
                feats = _clean_text(product.get("features"))
                
                embed_text = f"{title}. Category: {cats}. Features: {feats}".strip()
                asins.append(asin)
                documents.append(embed_text)

        self.parent_asins = asins

        # Batched Encoding with TQDM Progress Bar
        print(f"[FAISS Indexer] Encoding {len(documents)} documents in batches of {self.batch_size}...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(documents), self.batch_size), desc="Embedding Catalog"):
            batch_docs = documents[i : i + self.batch_size]
            embeddings = self.encoder.encode(
                batch_docs, 
                show_progress_bar=False, 
                normalize_embeddings=True
            )
            all_embeddings.append(embeddings)

        full_matrix = np.vstack(all_embeddings).astype(np.float32)
        dimension = full_matrix.shape[1]

        # Construct FAISS Inner Product Index
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(full_matrix)

        # Save cache to disk
        print(f"[FAISS Indexer] Saving index cache to {self.cache_dir}...")
        faiss.write_index(self.faiss_index, str(self.index_file))
        with self.asins_file.open("w", encoding="utf-8") as f:
            json.dump(self.parent_asins, f)
        print("[FAISS Indexer] Build and caching complete!")

    # def search(self, query: str, top_k: int = 50) -> List[str]:
    #     """Performs vector similarity search."""
    #     if not self.faiss_index:
    #         return []
        
    #     q_emb = self.encoder.encode([query], normalize_embeddings=True)
    #     _, indices = self.faiss_index.search(np.array(q_emb, dtype=np.float32), top_k)
        
    #     results = []
    #     for idx in indices[0]:
    #         if idx != -1 and idx < len(self.parent_asins):
    #             results.append(self.parent_asins[idx])
    #     return results

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Executes dense vector search and returns (asin, cosine_similarity) pairs."""
        query_vector = self.encoder.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        ).astype("float32")

        distances, indices = self.faiss_index.search(query_vector, top_k)
        
        results: List[Tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.parent_asins):
                asin = self.parent_asins[idx]
                results.append((asin, float(dist)))
                
        return results