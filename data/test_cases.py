# text2sql_mvp/data/test_cases.py
"""
Test case management for Text-to-SQL evaluation.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class TestCase:
    """A test case for evaluation."""
    id: str
    question: str
    expected_sql: str
    difficulty: str = "medium"
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    tables_used: list[str] = field(default_factory=list)
    description: Optional[str] = None
    
    # Expected result for execution-based evaluation
    expected_result: Optional[Any] = None
    expected_columns: Optional[list[str]] = None
    expected_row_count: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "expected_sql": self.expected_sql,
            "difficulty": self.difficulty,
            "category": self.category,
            "tags": self.tags,
            "tables_used": self.tables_used,
            "description": self.description,
            "expected_result": self.expected_result,
            "expected_columns": self.expected_columns,
            "expected_row_count": self.expected_row_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestCase":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            question=data["question"],
            expected_sql=data["expected_sql"],
            difficulty=data.get("difficulty", "medium"),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
            tables_used=data.get("tables_used", []),
            description=data.get("description"),
            expected_result=data.get("expected_result"),
            expected_columns=data.get("expected_columns"),
            expected_row_count=data.get("expected_row_count"),
        )


class TestCaseLoader:
    """
    Loader for test cases with filtering and sampling.
    
    Example:
        ```python
        loader = TestCaseLoader()
        loader.load_from_json("test_cases.json")
        
        easy_cases = loader.filter(difficulty="easy")
        sample = loader.sample(n=10)
        ```
    """
    
    def __init__(self):
        """Initialize the loader."""
        self._cases: list[TestCase] = []
    
    def load_from_json(self, path: str | Path) -> None:
        """Load test cases from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        
        cases_data = data.get("test_cases", data)
        if isinstance(cases_data, list):
            self._cases = []
            for i, case in enumerate(cases_data):
                if "id" not in case:
                    case["id"] = f"test_{i+1}"
                self._cases.append(TestCase.from_dict(case))
    
    def load_from_list(self, cases: list[dict[str, Any]]) -> None:
        """Load test cases from list of dictionaries."""
        self._cases = []
        for i, case in enumerate(cases):
            if "id" not in case:
                case["id"] = f"test_{i+1}"
            self._cases.append(TestCase.from_dict(case))
    
    @property
    def cases(self) -> list[TestCase]:
        """Get all test cases."""
        return self._cases
    
    @property
    def size(self) -> int:
        """Number of test cases."""
        return len(self._cases)
    
    def filter(
        self,
        difficulty: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
        tables: Optional[list[str]] = None,
    ) -> list[TestCase]:
        """
        Filter test cases.
        
        Args:
            difficulty: Filter by difficulty
            category: Filter by category
            tags: Filter by tags (any match)
            tables: Filter by tables used (any match)
            
        Returns:
            Filtered list of test cases
        """
        filtered = self._cases.copy()
        
        if difficulty:
            filtered = [c for c in filtered if c.difficulty == difficulty]
        
        if category:
            filtered = [c for c in filtered if c.category == category]
        
        if tags:
            tags_set = set(tags)
            filtered = [c for c in filtered if tags_set & set(c.tags)]
        
        if tables:
            tables_set = set(t.lower() for t in tables)
            filtered = [
                c for c in filtered 
                if tables_set & set(t.lower() for t in c.tables_used)
            ]
        
        return filtered
    
    def sample(
        self,
        n: int,
        stratified: bool = False,
        difficulty_weights: Optional[dict[str, float]] = None,
    ) -> list[TestCase]:
        """
        Sample test cases.
        
        Args:
            n: Number of cases to sample
            stratified: If True, sample evenly across difficulties
            difficulty_weights: Custom weights for each difficulty level
            
        Returns:
            Sampled test cases
        """
        if not self._cases:
            return []
        
        if n >= len(self._cases):
            return self._cases.copy()
        
        if stratified:
            # Group by difficulty
            by_diff = {}
            for case in self._cases:
                if case.difficulty not in by_diff:
                    by_diff[case.difficulty] = []
                by_diff[case.difficulty].append(case)
            
            # Sample from each group
            result = []
            per_group = n // len(by_diff)
            for diff, cases in by_diff.items():
                sample_n = per_group if not difficulty_weights else int(n * difficulty_weights.get(diff, 1.0 / len(by_diff)))
                sample_n = min(sample_n, len(cases))
                result.extend(random.sample(cases, sample_n))
            
            # Fill remaining slots
            while len(result) < n:
                remaining = [c for c in self._cases if c not in result]
                if remaining:
                    result.append(random.choice(remaining))
                else:
                    break
            
            return result[:n]
        
        return random.sample(self._cases, n)
    
    def get_by_id(self, case_id: str) -> Optional[TestCase]:
        """Get test case by ID."""
        for case in self._cases:
            if case.id == case_id:
                return case
        return None
    
    def get_by_ids(self, ids: list[str]) -> list[TestCase]:
        """Get multiple test cases by IDs."""
        id_set = set(ids)
        return [c for c in self._cases if c.id in id_set]
    
    def get_difficulties(self) -> list[str]:
        """Get list of unique difficulties."""
        return list(set(c.difficulty for c in self._cases))
    
    def get_categories(self) -> list[str]:
        """Get list of unique categories."""
        return list(set(c.category for c in self._cases))
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the test cases."""
        stats = {
            "total": len(self._cases),
            "by_difficulty": {},
            "by_category": {},
        }
        
        for case in self._cases:
            # Count by difficulty
            if case.difficulty not in stats["by_difficulty"]:
                stats["by_difficulty"][case.difficulty] = 0
            stats["by_difficulty"][case.difficulty] += 1
            
            # Count by category
            if case.category not in stats["by_category"]:
                stats["by_category"][case.category] = 0
            stats["by_category"][case.category] += 1
        
        return stats
    
    def to_json(self, path: str | Path) -> None:
        """Save test cases to JSON file."""
        path = Path(path)
        data = {
            "test_cases": [c.to_dict() for c in self._cases]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def __len__(self) -> int:
        """Return number of test cases."""
        return len(self._cases)
    
    def __iter__(self):
        """Iterate over test cases."""
        return iter(self._cases)
    
    def __getitem__(self, idx: int) -> TestCase:
        """Get test case by index."""
        return self._cases[idx]
