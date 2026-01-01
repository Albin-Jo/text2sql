# text2sql_mvp/data/few_shot_pool.py
"""
Few-shot example pool for Text-to-SQL generation.
Supports static and dynamic (similarity-based) example selection.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class FewShotExample:
    """A few-shot example for prompting."""
    question: str
    sql: str
    explanation: Optional[str] = None
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)
    tables_used: list[str] = field(default_factory=list)
    query_type: str = "select"
    
    # Embedding for similarity search (populated lazily)
    embedding: Optional[np.ndarray] = None
    
    def to_prompt_string(self, include_explanation: bool = False) -> str:
        """Convert to prompt format."""
        parts = [
            f"Question: {self.question}",
            f"SQL: {self.sql}"
        ]
        if include_explanation and self.explanation:
            parts.insert(1, f"Reasoning: {self.explanation}")
        return "\n".join(parts)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "sql": self.sql,
            "explanation": self.explanation,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "tables_used": self.tables_used,
            "query_type": self.query_type,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FewShotExample":
        """Create from dictionary."""
        return cls(
            question=data["question"],
            sql=data["sql"],
            explanation=data.get("explanation"),
            difficulty=data.get("difficulty", "medium"),
            tags=data.get("tags", []),
            tables_used=data.get("tables_used", []),
            query_type=data.get("query_type", "select"),
        )


class FewShotPool:
    """
    Pool of few-shot examples with selection strategies.
    
    Supports:
    - Static selection (by difficulty, tags, etc.)
    - Random selection
    - Similarity-based selection (with embeddings)
    - Maximum Marginal Relevance (MMR) selection
    
    Example:
        ```python
        pool = FewShotPool()
        pool.load_from_json("examples.json")
        
        # Static selection
        examples = pool.get_examples(n=3, difficulty="medium")
        
        # Similarity-based selection
        examples = pool.get_similar_examples("my question", n=3)
        ```
    """
    
    def __init__(self, embedding_model: Optional[str] = None):
        """
        Initialize the pool.
        
        Args:
            embedding_model: Name of sentence-transformers model for similarity
        """
        self._examples: list[FewShotExample] = []
        self._embedding_model_name = embedding_model or "all-MiniLM-L6-v2"
        self._embedding_model = None
        self._embeddings_computed = False
    
    def load_from_json(self, path: str | Path) -> None:
        """Load examples from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        
        examples_data = data.get("examples", data)
        if isinstance(examples_data, list):
            self._examples = [FewShotExample.from_dict(ex) for ex in examples_data]
        
        self._embeddings_computed = False
    
    def load_from_list(self, examples: list[dict[str, Any]]) -> None:
        """Load examples from list of dictionaries."""
        self._examples = [FewShotExample.from_dict(ex) for ex in examples]
        self._embeddings_computed = False
    
    def add_example(self, example: FewShotExample) -> None:
        """Add a single example to the pool."""
        self._examples.append(example)
        self._embeddings_computed = False
    
    def add_examples(self, examples: list[FewShotExample]) -> None:
        """Add multiple examples."""
        self._examples.extend(examples)
        self._embeddings_computed = False
    
    @property
    def size(self) -> int:
        """Number of examples in pool."""
        return len(self._examples)
    
    def get_examples(
        self,
        n: int = 3,
        difficulty: Optional[str] = None,
        tags: Optional[list[str]] = None,
        tables: Optional[list[str]] = None,
        query_type: Optional[str] = None,
        random_order: bool = False,
    ) -> list[FewShotExample]:
        """
        Get examples with filtering.
        
        Args:
            n: Number of examples to return
            difficulty: Filter by difficulty
            tags: Filter by tags (any match)
            tables: Filter by tables used (any match)
            query_type: Filter by query type
            random_order: Randomize selection order
            
        Returns:
            List of matching examples
        """
        filtered = self._examples.copy()
        
        if difficulty:
            filtered = [ex for ex in filtered if ex.difficulty == difficulty]
        
        if tags:
            tags_set = set(tags)
            filtered = [ex for ex in filtered if tags_set & set(ex.tags)]
        
        if tables:
            tables_set = set(t.lower() for t in tables)
            filtered = [
                ex for ex in filtered 
                if tables_set & set(t.lower() for t in ex.tables_used)
            ]
        
        if query_type:
            filtered = [ex for ex in filtered if ex.query_type == query_type]
        
        if random_order:
            filtered = random.sample(filtered, min(n, len(filtered)))
        else:
            filtered = filtered[:n]
        
        return filtered
    
    def get_random_examples(self, n: int = 3) -> list[FewShotExample]:
        """Get random examples from the pool."""
        return random.sample(self._examples, min(n, len(self._examples)))
    
    def _ensure_embeddings(self) -> None:
        """Compute embeddings for all examples if not done."""
        if self._embeddings_computed:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            if self._embedding_model is None:
                self._embedding_model = SentenceTransformer(self._embedding_model_name)
            
            questions = [ex.question for ex in self._examples]
            embeddings = self._embedding_model.encode(questions, convert_to_numpy=True)
            
            for i, ex in enumerate(self._examples):
                ex.embedding = embeddings[i]
            
            self._embeddings_computed = True
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for similarity-based selection. "
                "Install with: pip install sentence-transformers"
            )
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a single text."""
        self._ensure_embeddings()  # Ensures model is loaded
        return self._embedding_model.encode([text], convert_to_numpy=True)[0]
    
    def get_similar_examples(
        self,
        question: str,
        n: int = 3,
        min_similarity: float = 0.0,
    ) -> list[FewShotExample]:
        """
        Get most similar examples to the question.
        
        Args:
            question: Question to find similar examples for
            n: Number of examples to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of most similar examples
        """
        self._ensure_embeddings()
        
        # Compute question embedding
        query_embedding = self._compute_embedding(question)
        
        # Compute similarities
        similarities = []
        for ex in self._examples:
            if ex.embedding is not None:
                sim = self._cosine_similarity(query_embedding, ex.embedding)
                similarities.append((ex, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and return top n
        result = [
            ex for ex, sim in similarities 
            if sim >= min_similarity
        ][:n]
        
        return result
    
    def get_mmr_examples(
        self,
        question: str,
        n: int = 3,
        lambda_param: float = 0.5,
    ) -> list[FewShotExample]:
        """
        Get examples using Maximum Marginal Relevance.
        
        MMR balances relevance to the query with diversity among selected examples.
        
        Args:
            question: Question to find examples for
            n: Number of examples to return
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            
        Returns:
            List of selected examples
        """
        self._ensure_embeddings()
        
        query_embedding = self._compute_embedding(question)
        
        # Compute relevance scores
        candidates = []
        for ex in self._examples:
            if ex.embedding is not None:
                relevance = self._cosine_similarity(query_embedding, ex.embedding)
                candidates.append((ex, relevance))
        
        # Sort by relevance first
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy MMR selection
        selected: list[FewShotExample] = []
        selected_embeddings: list[np.ndarray] = []
        
        while len(selected) < n and candidates:
            best_score = -float('inf')
            best_idx = 0
            
            for i, (ex, relevance) in enumerate(candidates):
                # Compute max similarity to already selected
                if selected_embeddings:
                    max_sim = max(
                        self._cosine_similarity(ex.embedding, sel_emb)
                        for sel_emb in selected_embeddings
                    )
                else:
                    max_sim = 0.0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            # Add best candidate
            best_ex = candidates.pop(best_idx)[0]
            selected.append(best_ex)
            selected_embeddings.append(best_ex.embedding)
        
        return selected
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def get_by_tables(self, tables: list[str], n: int = 3) -> list[FewShotExample]:
        """Get examples that use specific tables."""
        tables_lower = set(t.lower() for t in tables)
        
        # Score by number of matching tables
        scored = []
        for ex in self._examples:
            ex_tables = set(t.lower() for t in ex.tables_used)
            overlap = len(tables_lower & ex_tables)
            if overlap > 0:
                scored.append((ex, overlap))
        
        # Sort by overlap
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [ex for ex, _ in scored[:n]]
    
    def get_diverse_examples(
        self,
        n: int = 3,
        by: str = "difficulty"
    ) -> list[FewShotExample]:
        """
        Get diverse examples by a specific attribute.
        
        Args:
            n: Number of examples
            by: Attribute to diversify by ("difficulty", "query_type", "tags")
            
        Returns:
            Diverse set of examples
        """
        if by == "difficulty":
            # One from each difficulty level
            by_diff = {}
            for ex in self._examples:
                if ex.difficulty not in by_diff:
                    by_diff[ex.difficulty] = []
                by_diff[ex.difficulty].append(ex)
            
            result = []
            for diff in ["easy", "medium", "hard"]:
                if diff in by_diff and len(result) < n:
                    result.append(random.choice(by_diff[diff]))
            return result
        
        elif by == "query_type":
            by_type = {}
            for ex in self._examples:
                if ex.query_type not in by_type:
                    by_type[ex.query_type] = []
                by_type[ex.query_type].append(ex)
            
            result = []
            for qtype in by_type:
                if len(result) < n:
                    result.append(random.choice(by_type[qtype]))
            return result
        
        else:
            return self.get_random_examples(n)
    
    def to_json(self, path: str | Path) -> None:
        """Save examples to JSON file."""
        path = Path(path)
        data = {
            "examples": [ex.to_dict() for ex in self._examples]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def __len__(self) -> int:
        """Return number of examples."""
        return len(self._examples)
    
    def __iter__(self):
        """Iterate over examples."""
        return iter(self._examples)
