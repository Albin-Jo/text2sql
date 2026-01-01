# text2sql_mvp/core/sql_parser.py
"""
SQL parsing utilities using sqlglot.
Extracts SQL from LLM responses and analyzes query structure.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError


class QueryType(str, Enum):
    """SQL query type classification."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    ALTER = "alter"
    DROP = "drop"
    UNKNOWN = "unknown"


class SQLDialect(str, Enum):
    """Supported SQL dialects."""
    BIGQUERY = "bigquery"
    DUCKDB = "duckdb"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"


@dataclass
class QueryFeatures:
    """Features detected in a SQL query."""
    has_aggregation: bool = False
    has_groupby: bool = False
    has_having: bool = False
    has_orderby: bool = False
    has_limit: bool = False
    has_distinct: bool = False
    has_subquery: bool = False
    has_join: bool = False
    has_union: bool = False
    has_window_function: bool = False
    has_cte: bool = False
    has_case_when: bool = False
    join_count: int = 0
    subquery_count: int = 0
    table_count: int = 0
    column_count: int = 0


@dataclass
class SQLParseResult:
    """Result of SQL parsing."""
    sql: str
    is_valid: bool
    query_type: QueryType = QueryType.UNKNOWN
    tables: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    features: QueryFeatures = field(default_factory=QueryFeatures)
    complexity_score: int = 1
    error: Optional[str] = None
    formatted_sql: Optional[str] = None
    skeleton: Optional[str] = None


class SQLParser:
    """
    SQL parser for Text-to-SQL pipeline.
    
    Features:
    - Extract SQL from LLM responses (markdown, plain text)
    - Parse and validate SQL syntax
    - Extract query metadata (tables, columns, features)
    - Calculate complexity score
    - Format and transpile SQL
    
    Example:
        ```python
        parser = SQLParser()
        result = parser.parse("SELECT * FROM users WHERE age > 18")
        print(f"Tables: {result.tables}")
        print(f"Complexity: {result.complexity_score}")
        ```
    """
    
    # Patterns for extracting SQL from LLM responses
    SQL_PATTERNS = [
        # Code blocks with sql/SQL tag
        r"```sql\s*(.*?)\s*```",
        r"```SQL\s*(.*?)\s*```",
        # Generic code blocks
        r"```\s*(SELECT.*?)\s*```",
        r"```\s*(WITH.*?)\s*```",
        # Inline backticks
        r"`(SELECT[^`]+)`",
        r"`(WITH[^`]+)`",
    ]
    
    # Aggregation functions
    AGG_FUNCTIONS = {"COUNT", "SUM", "AVG", "MIN", "MAX", "GROUP_CONCAT", "STRING_AGG", "ARRAY_AGG"}
    
    # Window functions
    WINDOW_FUNCTIONS = {"ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE", "LAG", "LEAD", "FIRST_VALUE", "LAST_VALUE"}
    
    def __init__(self, dialect: SQLDialect = SQLDialect.BIGQUERY):
        """
        Initialize SQL parser.
        
        Args:
            dialect: SQL dialect for parsing
        """
        self.dialect = dialect
        self._sqlglot_dialect = self._get_sqlglot_dialect(dialect)
    
    def _get_sqlglot_dialect(self, dialect: SQLDialect) -> str:
        """Map our dialect enum to sqlglot dialect string."""
        mapping = {
            SQLDialect.BIGQUERY: "bigquery",
            SQLDialect.DUCKDB: "duckdb",
            SQLDialect.POSTGRES: "postgres",
            SQLDialect.MYSQL: "mysql",
            SQLDialect.SQLITE: "sqlite",
        }
        return mapping.get(dialect, "bigquery")
    
    def extract_sql(self, text: str) -> str:
        """
        Extract SQL from LLM response text.
        
        Handles various formats:
        - Markdown code blocks (```sql ... ```)
        - Plain SQL statements
        - Multiple statements (returns first SELECT/WITH)
        
        Args:
            text: Raw LLM response text
            
        Returns:
            Extracted SQL string (empty if none found)
        """
        if not text:
            return ""
        
        # Try code block patterns first
        for pattern in self.SQL_PATTERNS:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                sql = matches[0].strip()
                # Clean up any remaining artifacts
                sql = self._clean_sql(sql)
                if sql:
                    return sql
        
        # Try to find raw SQL starting with SELECT or WITH
        lines = text.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()
            
            # Start capturing at SELECT or WITH
            if not in_sql and (upper.startswith("SELECT") or upper.startswith("WITH")):
                in_sql = True
            
            if in_sql:
                # Stop at explanatory text
                if stripped and not any([
                    upper.startswith(kw) for kw in 
                    ["SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
                     "GROUP", "ORDER", "HAVING", "LIMIT", "OFFSET", "UNION", "WITH", "AND",
                     "OR", "ON", "AS", "CASE", "WHEN", "THEN", "ELSE", "END", "IN", "NOT",
                     "BETWEEN", "LIKE", "IS", "NULL", "TRUE", "FALSE", "DISTINCT", "ALL",
                     "(", ")", ",", "--"]
                ]) and not stripped.startswith("--") and len(stripped) > 50 and " " in stripped:
                    # Looks like prose, stop
                    if not any(c in stripped for c in ["(", ")", "=", "<", ">", ","]):
                        break
                
                sql_lines.append(line)
        
        if sql_lines:
            sql = '\n'.join(sql_lines).strip()
            sql = self._clean_sql(sql)
            if sql:
                return sql
        
        return ""
    
    def _clean_sql(self, sql: str) -> str:
        """Clean up extracted SQL."""
        # Remove markdown artifacts
        sql = re.sub(r'^```\w*\s*', '', sql)
        sql = re.sub(r'\s*```$', '', sql)
        
        # Remove trailing semicolons (can cause issues)
        sql = sql.rstrip(';').strip()
        
        # Remove any leading/trailing whitespace
        sql = sql.strip()
        
        return sql
    
    def parse(self, sql: str, extract_first: bool = True) -> SQLParseResult:
        """
        Parse SQL and extract metadata.
        
        Args:
            sql: SQL string (can include markdown formatting)
            extract_first: If True, extract SQL from text first
            
        Returns:
            SQLParseResult with parsing results
        """
        # Extract SQL if needed
        if extract_first:
            extracted = self.extract_sql(sql)
            if extracted:
                sql = extracted
        
        if not sql:
            return SQLParseResult(
                sql="",
                is_valid=False,
                error="No SQL found in input"
            )
        
        # Clean the SQL
        sql = self._clean_sql(sql)
        
        try:
            # Parse with sqlglot
            parsed = sqlglot.parse_one(sql, dialect=self._sqlglot_dialect)
            
            # Extract metadata
            query_type = self._get_query_type(parsed)
            tables = self._extract_tables(parsed)
            columns = self._extract_columns(parsed)
            features = self._extract_features(parsed)
            complexity = self._calculate_complexity(features, tables, columns)
            
            # Format SQL
            try:
                formatted = parsed.sql(dialect=self._sqlglot_dialect, pretty=True)
            except Exception:
                formatted = sql
            
            # Extract skeleton
            skeleton = self._extract_skeleton(parsed)
            
            return SQLParseResult(
                sql=sql,
                is_valid=True,
                query_type=query_type,
                tables=tables,
                columns=columns,
                features=features,
                complexity_score=complexity,
                formatted_sql=formatted,
                skeleton=skeleton,
            )
            
        except ParseError as e:
            return SQLParseResult(
                sql=sql,
                is_valid=False,
                error=f"Parse error: {str(e)}"
            )
        except Exception as e:
            return SQLParseResult(
                sql=sql,
                is_valid=False,
                error=f"Unexpected error: {str(e)}"
            )
    
    def _get_query_type(self, parsed: exp.Expression) -> QueryType:
        """Determine query type from parsed expression."""
        if isinstance(parsed, exp.Select):
            return QueryType.SELECT
        elif isinstance(parsed, exp.Insert):
            return QueryType.INSERT
        elif isinstance(parsed, exp.Update):
            return QueryType.UPDATE
        elif isinstance(parsed, exp.Delete):
            return QueryType.DELETE
        elif isinstance(parsed, exp.Create):
            return QueryType.CREATE
        elif isinstance(parsed, exp.Alter):
            return QueryType.ALTER
        elif isinstance(parsed, exp.Drop):
            return QueryType.DROP
        return QueryType.UNKNOWN
    
    def _extract_tables(self, parsed: exp.Expression) -> list[str]:
        """Extract table names from parsed SQL."""
        tables = set()
        
        for table in parsed.find_all(exp.Table):
            name = table.name
            if name:
                tables.add(name)
        
        return sorted(list(tables))
    
    def _extract_columns(self, parsed: exp.Expression) -> list[str]:
        """Extract column names from parsed SQL."""
        columns = set()
        
        for col in parsed.find_all(exp.Column):
            name = col.name
            if name and name != "*":
                columns.add(name)
        
        return sorted(list(columns))
    
    def _extract_features(self, parsed: exp.Expression) -> QueryFeatures:
        """Extract query features."""
        features = QueryFeatures()
        
        # Check for various SQL features
        features.has_groupby = bool(parsed.find(exp.Group))
        features.has_having = bool(parsed.find(exp.Having))
        features.has_orderby = bool(parsed.find(exp.Order))
        features.has_limit = bool(parsed.find(exp.Limit))
        features.has_distinct = bool(parsed.find(exp.Distinct))
        features.has_union = bool(parsed.find(exp.Union))
        features.has_cte = bool(parsed.find(exp.CTE))
        features.has_case_when = bool(parsed.find(exp.Case))
        
        # Count joins
        joins = list(parsed.find_all(exp.Join))
        features.has_join = len(joins) > 0
        features.join_count = len(joins)
        
        # Count subqueries
        subqueries = list(parsed.find_all(exp.Subquery))
        features.has_subquery = len(subqueries) > 0
        features.subquery_count = len(subqueries)
        
        # Check for aggregation functions
        for func in parsed.find_all(exp.Func):
            func_name = func.name.upper() if hasattr(func, 'name') else ""
            if func_name in self.AGG_FUNCTIONS:
                features.has_aggregation = True
            if func_name in self.WINDOW_FUNCTIONS:
                features.has_window_function = True
        
        # Also check for window expressions
        if parsed.find(exp.Window):
            features.has_window_function = True
        
        # Count tables and columns
        features.table_count = len(self._extract_tables(parsed))
        features.column_count = len(self._extract_columns(parsed))
        
        return features
    
    def _calculate_complexity(
        self,
        features: QueryFeatures,
        tables: list[str],
        columns: list[str]
    ) -> int:
        """
        Calculate query complexity score (1-10).
        
        Scoring:
        - Base: 1
        - Per join: +1 (max +3)
        - Subqueries: +1 each (max +2)
        - Aggregation: +1
        - Window functions: +2
        - CTE: +1
        - HAVING: +1
        """
        score = 1
        
        # Joins
        score += min(features.join_count, 3)
        
        # Subqueries
        score += min(features.subquery_count, 2)
        
        # Other features
        if features.has_aggregation:
            score += 1
        if features.has_window_function:
            score += 2
        if features.has_cte:
            score += 1
        if features.has_having:
            score += 1
        if features.has_union:
            score += 1
        
        return min(score, 10)
    
    def _extract_skeleton(self, parsed: exp.Expression) -> str:
        """
        Extract SQL skeleton (structure without specific values).
        
        Example: "SELECT ... FROM table JOIN table WHERE ... GROUP BY ... ORDER BY ..."
        """
        parts = []
        
        # Check for WITH clause
        if parsed.find(exp.CTE):
            parts.append("WITH")
        
        # SELECT clause
        if isinstance(parsed, exp.Select):
            if parsed.find(exp.Distinct):
                parts.append("SELECT DISTINCT")
            else:
                parts.append("SELECT")
            parts.append("...")
        
        # FROM clause with joins
        from_clause = parsed.find(exp.From)
        if from_clause:
            parts.append("FROM")
            parts.append("table")
            
            # Add joins
            for join in parsed.find_all(exp.Join):
                join_type = "JOIN"
                if hasattr(join, 'kind') and join.kind:
                    join_type = f"{join.kind.upper()} JOIN"
                parts.append(join_type)
                parts.append("table")
        
        # WHERE clause
        if parsed.find(exp.Where):
            parts.append("WHERE ...")
        
        # GROUP BY
        if parsed.find(exp.Group):
            parts.append("GROUP BY ...")
        
        # HAVING
        if parsed.find(exp.Having):
            parts.append("HAVING ...")
        
        # ORDER BY
        if parsed.find(exp.Order):
            parts.append("ORDER BY ...")
        
        # LIMIT
        if parsed.find(exp.Limit):
            parts.append("LIMIT ...")
        
        return " ".join(parts)
    
    def validate_syntax(self, sql: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL syntax.
        
        Args:
            sql: SQL to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        result = self.parse(sql, extract_first=False)
        return result.is_valid, result.error
    
    def format_sql(self, sql: str, dialect: Optional[SQLDialect] = None) -> str:
        """
        Format SQL for readability.
        
        Args:
            sql: SQL to format
            dialect: Target dialect (uses default if not specified)
            
        Returns:
            Formatted SQL string
        """
        target_dialect = self._get_sqlglot_dialect(dialect) if dialect else self._sqlglot_dialect
        
        try:
            parsed = sqlglot.parse_one(sql, dialect=self._sqlglot_dialect)
            return parsed.sql(dialect=target_dialect, pretty=True)
        except Exception:
            return sql
    
    def transpile(
        self,
        sql: str,
        source_dialect: SQLDialect,
        target_dialect: SQLDialect
    ) -> str:
        """
        Transpile SQL between dialects.
        
        Args:
            sql: SQL to transpile
            source_dialect: Source SQL dialect
            target_dialect: Target SQL dialect
            
        Returns:
            Transpiled SQL string
        """
        source = self._get_sqlglot_dialect(source_dialect)
        target = self._get_sqlglot_dialect(target_dialect)
        
        try:
            return sqlglot.transpile(sql, read=source, write=target)[0]
        except Exception:
            return sql
    
    def get_tables_from_sql(self, sql: str) -> list[str]:
        """Quick method to extract table names from SQL."""
        result = self.parse(sql)
        return result.tables if result.is_valid else []
    
    def get_complexity(self, sql: str) -> int:
        """Quick method to get complexity score."""
        result = self.parse(sql)
        return result.complexity_score if result.is_valid else 0
