# text2sql_mvp/schema/manager.py
"""
Schema management for Text-to-SQL MVP.
Handles loading, parsing, and caching database schemas.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ColumnInfo:
    """Information about a database column."""
    name: str
    data_type: str
    nullable: bool = True
    description: Optional[str] = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_table: Optional[str] = None
    foreign_key_column: Optional[str] = None
    sample_values: list[Any] = field(default_factory=list)


@dataclass 
class TableInfo:
    """Information about a database table."""
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    description: Optional[str] = None
    primary_key: Optional[str] = None
    foreign_keys: dict[str, tuple[str, str]] = field(default_factory=dict)
    row_count: Optional[int] = None
    
    def get_column(self, name: str) -> Optional[ColumnInfo]:
        """Get column by name (case-insensitive)."""
        name_lower = name.lower()
        for col in self.columns:
            if col.name.lower() == name_lower:
                return col
        return None
    
    def column_names(self) -> list[str]:
        """Get list of column names."""
        return [col.name for col in self.columns]


@dataclass
class SchemaInfo:
    """Complete database schema information."""
    name: str
    tables: list[TableInfo] = field(default_factory=list)
    description: Optional[str] = None
    dialect: str = "bigquery"
    
    def get_table(self, name: str) -> Optional[TableInfo]:
        """Get table by name (case-insensitive)."""
        name_lower = name.lower()
        for table in self.tables:
            if table.name.lower() == name_lower:
                return table
        return None
    
    def table_names(self) -> list[str]:
        """Get list of table names."""
        return [t.name for t in self.tables]
    
    def get_relationships(self) -> list[tuple[str, str, str, str]]:
        """
        Get foreign key relationships.
        
        Returns:
            List of (from_table, from_column, to_table, to_column)
        """
        relationships = []
        for table in self.tables:
            for col_name, (ref_table, ref_col) in table.foreign_keys.items():
                relationships.append((table.name, col_name, ref_table, ref_col))
        return relationships


class SchemaManager:
    """
    Manager for database schemas.
    
    Features:
    - Load schemas from SQL DDL files
    - Load schemas from JSON
    - Cache parsed schemas
    - Validate schema structure
    
    Example:
        ```python
        manager = SchemaManager()
        schema = manager.load_from_sql("path/to/schema.sql")
        print(schema.table_names())
        ```
    """
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize schema manager.
        
        Args:
            cache_enabled: Whether to cache loaded schemas
        """
        self._cache: dict[str, SchemaInfo] = {}
        self._cache_enabled = cache_enabled
    
    def load_from_sql(
        self,
        path: str | Path,
        schema_name: Optional[str] = None,
        dialect: str = "bigquery"
    ) -> SchemaInfo:
        """
        Load schema from SQL DDL file.
        
        Args:
            path: Path to SQL file
            schema_name: Name for the schema (defaults to filename)
            dialect: SQL dialect
            
        Returns:
            SchemaInfo object
        """
        path = Path(path)
        cache_key = str(path.absolute())
        
        # Check cache
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Read SQL file
        sql_content = path.read_text()
        
        # Parse DDL
        schema = self._parse_ddl(
            sql_content,
            schema_name or path.stem,
            dialect
        )
        
        # Cache
        if self._cache_enabled:
            self._cache[cache_key] = schema
        
        return schema
    
    def load_from_json(
        self,
        path: str | Path,
        schema_name: Optional[str] = None
    ) -> SchemaInfo:
        """
        Load schema from JSON file.
        
        Args:
            path: Path to JSON file
            schema_name: Name for the schema
            
        Returns:
            SchemaInfo object
        """
        path = Path(path)
        cache_key = str(path.absolute())
        
        # Check cache
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Read JSON
        with open(path) as f:
            data = json.load(f)
        
        schema = self._parse_json(data, schema_name or path.stem)
        
        # Cache
        if self._cache_enabled:
            self._cache[cache_key] = schema
        
        return schema
    
    def load_from_dict(
        self,
        data: dict[str, Any],
        schema_name: str = "schema"
    ) -> SchemaInfo:
        """
        Load schema from dictionary.
        
        Args:
            data: Schema dictionary
            schema_name: Name for the schema
            
        Returns:
            SchemaInfo object
        """
        return self._parse_json(data, schema_name)
    
    def _parse_ddl(
        self,
        sql: str,
        schema_name: str,
        dialect: str
    ) -> SchemaInfo:
        """Parse SQL DDL statements."""
        tables = []
        
        # Pattern for CREATE TABLE statements
        create_pattern = re.compile(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?(\w+)[`\"]?\s*\((.*?)\)",
            re.IGNORECASE | re.DOTALL
        )
        
        for match in create_pattern.finditer(sql):
            table_name = match.group(1)
            columns_str = match.group(2)
            
            table = self._parse_table_definition(table_name, columns_str, dialect)
            tables.append(table)
        
        return SchemaInfo(
            name=schema_name,
            tables=tables,
            dialect=dialect,
        )
    
    def _parse_table_definition(
        self,
        table_name: str,
        columns_str: str,
        dialect: str
    ) -> TableInfo:
        """Parse a single table definition."""
        columns = []
        primary_key = None
        foreign_keys = {}
        
        # Split by commas, but handle nested parentheses
        parts = self._split_columns(columns_str)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            upper_part = part.upper()
            
            # Primary key constraint
            if upper_part.startswith("PRIMARY KEY"):
                pk_match = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", part, re.IGNORECASE)
                if pk_match:
                    primary_key = pk_match.group(1).strip().strip('`"')
                continue
            
            # Foreign key constraint
            if upper_part.startswith("FOREIGN KEY"):
                fk_match = re.search(
                    r"FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+[`\"]?(\w+)[`\"]?\s*\(([^)]+)\)",
                    part,
                    re.IGNORECASE
                )
                if fk_match:
                    col = fk_match.group(1).strip().strip('`"')
                    ref_table = fk_match.group(2).strip()
                    ref_col = fk_match.group(3).strip().strip('`"')
                    foreign_keys[col] = (ref_table, ref_col)
                continue
            
            # Skip other constraints
            if any(kw in upper_part for kw in ["CONSTRAINT", "UNIQUE", "CHECK", "INDEX"]):
                continue
            
            # Parse column definition
            col = self._parse_column(part, dialect)
            if col:
                columns.append(col)
        
        # Mark primary key column
        if primary_key:
            for col in columns:
                if col.name.lower() == primary_key.lower():
                    col.is_primary_key = True
                    break
        
        # Mark foreign key columns
        for col in columns:
            if col.name in foreign_keys:
                col.is_foreign_key = True
                col.foreign_key_table, col.foreign_key_column = foreign_keys[col.name]
        
        return TableInfo(
            name=table_name,
            columns=columns,
            primary_key=primary_key,
            foreign_keys=foreign_keys,
        )
    
    def _split_columns(self, columns_str: str) -> list[str]:
        """Split column definitions, handling nested parentheses."""
        parts = []
        current = []
        depth = 0
        
        for char in columns_str:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(char)
        
        if current:
            parts.append(''.join(current))
        
        return parts
    
    def _parse_column(self, col_str: str, dialect: str) -> Optional[ColumnInfo]:
        """Parse a single column definition."""
        col_str = col_str.strip()
        if not col_str:
            return None
        
        # Pattern: column_name DATA_TYPE [constraints]
        match = re.match(
            r"[`\"]?(\w+)[`\"]?\s+(\w+(?:\([^)]+\))?)\s*(.*)",
            col_str,
            re.IGNORECASE
        )
        
        if not match:
            return None
        
        name = match.group(1)
        data_type = self._normalize_data_type(match.group(2), dialect)
        constraints = match.group(3).upper() if match.group(3) else ""
        
        nullable = "NOT NULL" not in constraints
        is_pk = "PRIMARY KEY" in constraints
        
        return ColumnInfo(
            name=name,
            data_type=data_type,
            nullable=nullable,
            is_primary_key=is_pk,
        )
    
    def _normalize_data_type(self, data_type: str, dialect: str) -> str:
        """Normalize data type for consistency."""
        dt = data_type.upper()
        
        # BigQuery type mappings
        type_map = {
            "INT64": "INTEGER",
            "INT": "INTEGER",
            "BIGINT": "INTEGER",
            "SMALLINT": "INTEGER",
            "FLOAT64": "FLOAT",
            "FLOAT": "FLOAT",
            "DOUBLE": "FLOAT",
            "BOOL": "BOOLEAN",
            "STRING": "STRING",
            "VARCHAR": "STRING",
            "TEXT": "STRING",
            "CHAR": "STRING",
            "TIMESTAMP": "TIMESTAMP",
            "DATETIME": "DATETIME",
            "DATE": "DATE",
            "TIME": "TIME",
            "NUMERIC": "NUMERIC",
            "DECIMAL": "NUMERIC",
        }
        
        # Remove size specifications for mapping
        base_type = re.sub(r"\([^)]+\)", "", dt).strip()
        
        return type_map.get(base_type, data_type)
    
    def _parse_json(self, data: dict[str, Any], schema_name: str) -> SchemaInfo:
        """Parse schema from JSON format."""
        tables = []
        
        for table_data in data.get("tables", []):
            columns = []
            for col_data in table_data.get("columns", []):
                col = ColumnInfo(
                    name=col_data["name"],
                    data_type=col_data.get("type", col_data.get("data_type", "STRING")),
                    nullable=col_data.get("nullable", True),
                    description=col_data.get("description"),
                    is_primary_key=col_data.get("is_primary_key", False),
                    is_foreign_key=col_data.get("is_foreign_key", False),
                    foreign_key_table=col_data.get("foreign_key_table"),
                    foreign_key_column=col_data.get("foreign_key_column"),
                )
                columns.append(col)
            
            # Parse foreign keys
            foreign_keys = {}
            for fk in table_data.get("foreign_keys", []):
                col = fk.get("column")
                ref = fk.get("references", "").split(".")
                if col and len(ref) == 2:
                    foreign_keys[col] = (ref[0], ref[1])
            
            table = TableInfo(
                name=table_data["name"],
                columns=columns,
                description=table_data.get("description"),
                primary_key=table_data.get("primary_key"),
                foreign_keys=foreign_keys,
            )
            tables.append(table)
        
        return SchemaInfo(
            name=schema_name,
            tables=tables,
            description=data.get("description"),
            dialect=data.get("dialect", "bigquery"),
        )
    
    def clear_cache(self) -> None:
        """Clear the schema cache."""
        self._cache.clear()
    
    def validate_schema(self, schema: SchemaInfo) -> list[str]:
        """
        Validate schema for common issues.
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        if not schema.tables:
            issues.append("Schema has no tables")
            return issues
        
        # Check for duplicate table names
        table_names = [t.name.lower() for t in schema.tables]
        if len(table_names) != len(set(table_names)):
            issues.append("Schema has duplicate table names")
        
        for table in schema.tables:
            # Check for empty tables
            if not table.columns:
                issues.append(f"Table '{table.name}' has no columns")
                continue
            
            # Check for duplicate column names
            col_names = [c.name.lower() for c in table.columns]
            if len(col_names) != len(set(col_names)):
                issues.append(f"Table '{table.name}' has duplicate column names")
            
            # Check foreign key references
            for col_name, (ref_table, ref_col) in table.foreign_keys.items():
                ref_table_obj = schema.get_table(ref_table)
                if not ref_table_obj:
                    issues.append(
                        f"Foreign key in '{table.name}.{col_name}' references "
                        f"non-existent table '{ref_table}'"
                    )
                elif not ref_table_obj.get_column(ref_col):
                    issues.append(
                        f"Foreign key in '{table.name}.{col_name}' references "
                        f"non-existent column '{ref_table}.{ref_col}'"
                    )
        
        return issues
