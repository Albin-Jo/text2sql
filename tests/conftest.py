# text2sql_mvp/tests/conftest.py
"""
Pytest configuration and fixtures for Text-to-SQL MVP tests.
"""

import asyncio
import json
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def finance_schema_path(project_root: Path) -> Path:
    """Path to finance domain schema."""
    return project_root / "data" / "domains" / "finance" / "schema.sql"


@pytest.fixture
def finance_test_cases_path(project_root: Path) -> Path:
    """Path to finance domain test cases."""
    return project_root / "data" / "domains" / "finance" / "test_cases.json"


@pytest.fixture
def finance_examples_path(project_root: Path) -> Path:
    """Path to finance domain few-shot examples."""
    return project_root / "data" / "domains" / "finance" / "few_shot_examples.json"


@pytest.fixture
def sample_schema_sql() -> str:
    """Sample SQL DDL for testing."""
    return """
    CREATE TABLE customers (
        customer_id STRING NOT NULL,
        first_name STRING NOT NULL,
        last_name STRING NOT NULL,
        email STRING,
        created_at TIMESTAMP,
        PRIMARY KEY (customer_id)
    );
    
    CREATE TABLE orders (
        order_id STRING NOT NULL,
        customer_id STRING NOT NULL,
        order_date DATE,
        total_amount NUMERIC(10, 2),
        status STRING,
        PRIMARY KEY (order_id),
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    """


@pytest.fixture
def sample_schema(sample_schema_sql: str):
    """Load sample schema."""
    from schema.manager import SchemaManager
    
    # Write to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
        f.write(sample_schema_sql)
        temp_path = f.name
    
    manager = SchemaManager()
    schema = manager.load_from_sql(temp_path, schema_name="test")
    
    # Cleanup
    Path(temp_path).unlink()
    
    return schema


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for testing without API calls."""
    from core.gemini_client import GeminiClient, GeminiResponse, GenerationMetrics
    
    client = MagicMock(spec=GeminiClient)
    client.model = "gemini-2.0-flash-exp"
    client.default_temperature = 0.0
    
    # Default successful response
    async def mock_generate(*args, **kwargs):
        return GeminiResponse(
            text="SELECT * FROM customers",
            success=True,
            metrics=GenerationMetrics(
                latency_ms=100.0,
                prompt_tokens=50,
                completion_tokens=10,
                total_tokens=60,
                model="gemini-2.0-flash-exp",
                temperature=0.0,
            ),
        )
    
    client.generate = AsyncMock(side_effect=mock_generate)
    return client


@pytest.fixture
def sample_few_shot_examples() -> list[dict]:
    """Sample few-shot examples for testing."""
    return [
        {
            "question": "How many customers are there?",
            "sql": "SELECT COUNT(*) FROM customers",
            "difficulty": "easy",
            "tags": ["count", "aggregation"],
            "tables_used": ["customers"],
        },
        {
            "question": "List all orders from last month",
            "sql": "SELECT * FROM orders WHERE order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)",
            "difficulty": "medium",
            "tags": ["filter", "date"],
            "tables_used": ["orders"],
        },
        {
            "question": "What is the total revenue per customer?",
            "sql": "SELECT customer_id, SUM(total_amount) as revenue FROM orders GROUP BY customer_id",
            "difficulty": "medium",
            "tags": ["aggregation", "group_by"],
            "tables_used": ["orders"],
        },
    ]


@pytest.fixture
def generation_context(sample_schema):
    """Create a generation context for testing."""
    from strategies.base import GenerationContext
    
    return GenerationContext(
        schema=sample_schema,
        question="How many customers are there?",
        dialect="bigquery",
    )
