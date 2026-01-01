# text2sql_mvp/schema/formatter.py
"""
Schema formatting utilities.
Converts schema information to various text formats for LLM prompts.
"""

from enum import Enum
from typing import Optional

from schema.manager import SchemaInfo, TableInfo, ColumnInfo


class SchemaFormat(str, Enum):
    """Available schema output formats."""
    DDL = "ddl"
    MARKDOWN = "markdown"
    JSON = "json"
    M_SCHEMA = "m_schema"
    COMPACT = "compact"


class SchemaFormatter:
    """
    Format database schemas for LLM prompts.
    
    Supports multiple formats optimized for different use cases:
    - DDL: Standard CREATE TABLE statements
    - MARKDOWN: Human-readable table format
    - JSON: Structured JSON representation
    - M_SCHEMA: Compact structured format (research-backed)
    - COMPACT: Minimal single-line format
    
    Example:
        ```python
        formatter = SchemaFormatter()
        ddl = formatter.format(schema, SchemaFormat.DDL)
        markdown = formatter.format(schema, SchemaFormat.MARKDOWN)
        ```
    """
    
    def format(
        self,
        schema: SchemaInfo,
        format_type: SchemaFormat = SchemaFormat.DDL,
        include_descriptions: bool = True,
        include_foreign_keys: bool = True,
        tables: Optional[list[str]] = None,
    ) -> str:
        """
        Format schema to specified format.
        
        Args:
            schema: Schema to format
            format_type: Output format
            include_descriptions: Include column descriptions
            include_foreign_keys: Include foreign key info
            tables: Optional list of specific tables to include
            
        Returns:
            Formatted schema string
        """
        # Filter tables if specified
        if tables:
            tables_lower = [t.lower() for t in tables]
            filtered_tables = [
                t for t in schema.tables 
                if t.name.lower() in tables_lower
            ]
        else:
            filtered_tables = schema.tables
        
        formatters = {
            SchemaFormat.DDL: self._format_ddl,
            SchemaFormat.MARKDOWN: self._format_markdown,
            SchemaFormat.JSON: self._format_json,
            SchemaFormat.M_SCHEMA: self._format_m_schema,
            SchemaFormat.COMPACT: self._format_compact,
        }
        
        formatter_func = formatters.get(format_type, self._format_ddl)
        return formatter_func(
            filtered_tables,
            include_descriptions,
            include_foreign_keys
        )
    
    def _format_ddl(
        self,
        tables: list[TableInfo],
        include_descriptions: bool,
        include_foreign_keys: bool
    ) -> str:
        """Format as DDL CREATE TABLE statements."""
        statements = []
        
        for table in tables:
            lines = [f"CREATE TABLE {table.name} ("]
            
            col_defs = []
            for col in table.columns:
                col_def = f"  {col.name} {col.data_type}"
                if not col.nullable:
                    col_def += " NOT NULL"
                if col.is_primary_key:
                    col_def += " PRIMARY KEY"
                if include_descriptions and col.description:
                    col_def += f"  -- {col.description}"
                col_defs.append(col_def)
            
            # Add foreign key constraints
            if include_foreign_keys:
                for col_name, (ref_table, ref_col) in table.foreign_keys.items():
                    fk_def = f"  FOREIGN KEY ({col_name}) REFERENCES {ref_table}({ref_col})"
                    col_defs.append(fk_def)
            
            lines.append(",\n".join(col_defs))
            lines.append(");")
            
            # Add table description as comment
            if include_descriptions and table.description:
                statements.append(f"-- {table.description}")
            statements.append("\n".join(lines))
        
        return "\n\n".join(statements)
    
    def _format_markdown(
        self,
        tables: list[TableInfo],
        include_descriptions: bool,
        include_foreign_keys: bool
    ) -> str:
        """Format as markdown tables."""
        sections = []
        
        for table in tables:
            lines = []
            
            # Table header
            if include_descriptions and table.description:
                lines.append(f"### {table.name}")
                lines.append(f"*{table.description}*")
            else:
                lines.append(f"### {table.name}")
            
            lines.append("")
            
            # Column table header
            if include_descriptions:
                lines.append("| Column | Type | Nullable | Key | Description |")
                lines.append("|--------|------|----------|-----|-------------|")
            else:
                lines.append("| Column | Type | Nullable | Key |")
                lines.append("|--------|------|----------|-----|")
            
            # Column rows
            for col in table.columns:
                key = ""
                if col.is_primary_key:
                    key = "PK"
                elif col.is_foreign_key:
                    key = f"FK→{col.foreign_key_table}"
                
                nullable = "Yes" if col.nullable else "No"
                desc = col.description or ""
                
                if include_descriptions:
                    lines.append(f"| {col.name} | {col.data_type} | {nullable} | {key} | {desc} |")
                else:
                    lines.append(f"| {col.name} | {col.data_type} | {nullable} | {key} |")
            
            sections.append("\n".join(lines))
        
        return "\n\n".join(sections)
    
    def _format_json(
        self,
        tables: list[TableInfo],
        include_descriptions: bool,
        include_foreign_keys: bool
    ) -> str:
        """Format as JSON structure."""
        import json
        
        schema_dict = {"tables": []}
        
        for table in tables:
            table_dict = {
                "name": table.name,
                "columns": []
            }
            
            if include_descriptions and table.description:
                table_dict["description"] = table.description
            
            for col in table.columns:
                col_dict = {
                    "name": col.name,
                    "type": col.data_type,
                    "nullable": col.nullable,
                }
                if col.is_primary_key:
                    col_dict["primary_key"] = True
                if include_foreign_keys and col.is_foreign_key:
                    col_dict["foreign_key"] = f"{col.foreign_key_table}.{col.foreign_key_column}"
                if include_descriptions and col.description:
                    col_dict["description"] = col.description
                    
                table_dict["columns"].append(col_dict)
            
            if include_foreign_keys and table.foreign_keys:
                table_dict["foreign_keys"] = [
                    {"column": col, "references": f"{ref[0]}.{ref[1]}"}
                    for col, ref in table.foreign_keys.items()
                ]
            
            schema_dict["tables"].append(table_dict)
        
        return json.dumps(schema_dict, indent=2)
    
    def _format_m_schema(
        self,
        tables: list[TableInfo],
        include_descriptions: bool,
        include_foreign_keys: bool
    ) -> str:
        """
        Format as M-Schema (research-backed compact format).
        
        Format: table_name(column1: type1, column2: type2, ...)
        With relationships shown as arrows.
        """
        lines = []
        
        for table in tables:
            # Build column list
            col_parts = []
            for col in table.columns:
                col_str = f"{col.name}: {col.data_type}"
                if col.is_primary_key:
                    col_str = f"*{col_str}"  # Mark primary key
                col_parts.append(col_str)
            
            line = f"{table.name}({', '.join(col_parts)})"
            
            if include_descriptions and table.description:
                line += f"  // {table.description}"
            
            lines.append(line)
        
        # Add relationships section
        if include_foreign_keys:
            relationships = []
            for table in tables:
                for col_name, (ref_table, ref_col) in table.foreign_keys.items():
                    relationships.append(f"{table.name}.{col_name} -> {ref_table}.{ref_col}")
            
            if relationships:
                lines.append("")
                lines.append("Relationships:")
                for rel in relationships:
                    lines.append(f"  {rel}")
        
        return "\n".join(lines)
    
    def _format_compact(
        self,
        tables: list[TableInfo],
        include_descriptions: bool,
        include_foreign_keys: bool
    ) -> str:
        """
        Format as minimal compact representation.
        
        Format: table: col1, col2, col3 | table2: col1, col2
        """
        table_strs = []
        
        for table in tables:
            cols = ", ".join(col.name for col in table.columns)
            table_strs.append(f"{table.name}: {cols}")
        
        return " | ".join(table_strs)
    
    def format_table(
        self,
        table: TableInfo,
        format_type: SchemaFormat = SchemaFormat.DDL,
        include_descriptions: bool = True,
        include_foreign_keys: bool = True,
    ) -> str:
        """Format a single table."""
        return self.format(
            SchemaInfo(name="temp", tables=[table]),
            format_type,
            include_descriptions,
            include_foreign_keys,
        )
    
    def get_column_descriptions(self, schema: SchemaInfo) -> str:
        """
        Get all column descriptions as a formatted string.
        
        Returns:
            String with table.column: description format
        """
        lines = []
        
        for table in schema.tables:
            for col in table.columns:
                if col.description:
                    lines.append(f"{table.name}.{col.name}: {col.description}")
        
        return "\n".join(lines) if lines else "No column descriptions available."
    
    def get_relationships_summary(self, schema: SchemaInfo) -> str:
        """
        Get a summary of table relationships.
        
        Returns:
            Human-readable relationship summary
        """
        relationships = schema.get_relationships()
        
        if not relationships:
            return "No foreign key relationships defined."
        
        lines = ["Table Relationships:"]
        for from_table, from_col, to_table, to_col in relationships:
            lines.append(f"  • {from_table}.{from_col} → {to_table}.{to_col}")
        
        return "\n".join(lines)
