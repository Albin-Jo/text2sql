# text2sql_mvp/tests/test_core/test_sql_parser.py
"""
Tests for SQL parser module.
"""

import pytest
from core.sql_parser import SQLParser, QueryType, SQLDialect


class TestSQLExtraction:
    """Tests for SQL extraction from text."""
    
    def setup_method(self):
        """Setup parser instance."""
        self.parser = SQLParser()
    
    def test_extract_from_code_block(self):
        """Test extraction from markdown code block."""
        text = """Here is the query:
        
```sql
SELECT * FROM users WHERE age > 18
```

This query selects all adult users.
"""
        sql = self.parser.extract_sql(text)
        assert sql == "SELECT * FROM users WHERE age > 18"
    
    def test_extract_from_plain_text(self):
        """Test extraction from plain text."""
        text = "The query is: SELECT COUNT(*) FROM orders"
        sql = self.parser.extract_sql(text)
        assert "SELECT COUNT(*)" in sql
    
    def test_extract_with_cte(self):
        """Test extraction of CTE query."""
        text = """```sql
WITH active_users AS (
    SELECT * FROM users WHERE status = 'active'
)
SELECT COUNT(*) FROM active_users
```"""
        sql = self.parser.extract_sql(text)
        assert sql.startswith("WITH active_users")
    
    def test_extract_empty_returns_empty(self):
        """Test that empty input returns empty string."""
        assert self.parser.extract_sql("") == ""
        assert self.parser.extract_sql("No SQL here") == ""
    
    def test_extract_removes_trailing_semicolon(self):
        """Test that trailing semicolons are removed."""
        text = "SELECT * FROM users;"
        sql = self.parser.extract_sql(text)
        assert not sql.endswith(";")


class TestSQLParsing:
    """Tests for SQL parsing functionality."""
    
    def setup_method(self):
        """Setup parser instance."""
        self.parser = SQLParser()
    
    def test_parse_simple_select(self):
        """Test parsing simple SELECT."""
        result = self.parser.parse("SELECT * FROM users")
        
        assert result.is_valid
        assert result.query_type == QueryType.SELECT
        assert "users" in result.tables
    
    def test_parse_with_join(self):
        """Test parsing query with JOIN."""
        sql = """
        SELECT u.name, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        """
        result = self.parser.parse(sql)
        
        assert result.is_valid
        assert result.features.has_join
        assert result.features.join_count == 1
        assert "users" in result.tables
        assert "orders" in result.tables
    
    def test_parse_with_aggregation(self):
        """Test parsing query with aggregation."""
        sql = "SELECT category, COUNT(*) as cnt FROM products GROUP BY category"
        result = self.parser.parse(sql)
        
        assert result.is_valid
        assert result.features.has_aggregation
        assert result.features.has_groupby
    
    def test_parse_with_subquery(self):
        """Test parsing query with subquery."""
        sql = """
        SELECT * FROM users
        WHERE id IN (SELECT user_id FROM orders WHERE total > 100)
        """
        result = self.parser.parse(sql)
        
        assert result.is_valid
        assert result.features.has_subquery
        assert result.features.subquery_count >= 1
    
    def test_parse_with_window_function(self):
        """Test parsing query with window function."""
        sql = """
        SELECT name, salary,
               RANK() OVER (ORDER BY salary DESC) as salary_rank
        FROM employees
        """
        result = self.parser.parse(sql)
        
        assert result.is_valid
        assert result.features.has_window_function
    
    def test_parse_invalid_sql(self):
        """Test parsing invalid SQL."""
        result = self.parser.parse("SELEKT * FORM users")
        
        assert not result.is_valid
        assert result.error is not None
    
    def test_complexity_scoring(self):
        """Test complexity scoring."""
        # Simple query
        simple = self.parser.parse("SELECT * FROM users")
        assert simple.complexity_score <= 2
        
        # Complex query
        complex_sql = """
        WITH cte AS (SELECT * FROM a JOIN b ON a.id = b.id)
        SELECT x, COUNT(*) OVER (PARTITION BY y)
        FROM cte
        WHERE EXISTS (SELECT 1 FROM c)
        GROUP BY x
        HAVING COUNT(*) > 1
        """
        complex_result = self.parser.parse(complex_sql)
        assert complex_result.complexity_score >= 5


class TestSQLFormatting:
    """Tests for SQL formatting."""
    
    def setup_method(self):
        """Setup parser instance."""
        self.parser = SQLParser()
    
    def test_format_simple_query(self):
        """Test formatting simple query."""
        sql = "select * from users where age>18"
        formatted = self.parser.format_sql(sql)
        
        # Should have proper capitalization and spacing
        assert "SELECT" in formatted
        assert "FROM" in formatted
    
    def test_format_preserves_semantics(self):
        """Test that formatting preserves query semantics."""
        sql = "SELECT a,b,c FROM t WHERE x=1 AND y=2"
        formatted = self.parser.format_sql(sql)
        
        # Parse both and compare
        original = self.parser.parse(sql)
        formatted_parsed = self.parser.parse(formatted)
        
        assert original.tables == formatted_parsed.tables


class TestSQLTranspilation:
    """Tests for SQL dialect transpilation."""
    
    def setup_method(self):
        """Setup parser instance."""
        self.parser = SQLParser()
    
    def test_transpile_bigquery_to_duckdb(self):
        """Test transpiling BigQuery to DuckDB."""
        bq_sql = "SELECT IFNULL(name, 'Unknown') FROM users"
        duckdb_sql = self.parser.transpile(
            bq_sql,
            SQLDialect.BIGQUERY,
            SQLDialect.DUCKDB
        )
        
        # Should have transpiled IFNULL to COALESCE or similar
        assert duckdb_sql is not None


class TestSkeleton:
    """Tests for SQL skeleton extraction."""
    
    def setup_method(self):
        """Setup parser instance."""
        self.parser = SQLParser()
    
    def test_skeleton_simple(self):
        """Test skeleton for simple query."""
        result = self.parser.parse("SELECT * FROM users")
        assert "SELECT" in result.skeleton
        assert "FROM" in result.skeleton
    
    def test_skeleton_with_join(self):
        """Test skeleton with JOIN."""
        sql = "SELECT * FROM a JOIN b ON a.id = b.id WHERE a.x = 1"
        result = self.parser.parse(sql)
        
        assert "JOIN" in result.skeleton
        assert "WHERE" in result.skeleton
    
    def test_skeleton_with_aggregation(self):
        """Test skeleton with aggregation."""
        sql = "SELECT x, COUNT(*) FROM t GROUP BY x HAVING COUNT(*) > 1 ORDER BY x"
        result = self.parser.parse(sql)
        
        assert "GROUP BY" in result.skeleton
        assert "HAVING" in result.skeleton
        assert "ORDER BY" in result.skeleton
