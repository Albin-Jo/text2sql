# text2sql_mvp/strategies/prompting/zero_shot_enhanced.py
"""
Enhanced zero-shot Text-to-SQL generation strategy.
Adds SQL dialect rules, common mistakes to avoid, and best practices.
"""

import time
from typing import Any, Optional

from core.gemini_client import GeminiClient
from core.sql_parser import SQLParser
from core.prompt_builder import PromptBuilder, SchemaFormat
from schema.formatter import SchemaFormatter
from strategies.base import (
    BaseStrategy,
    GenerationResult,
    GenerationContext,
    StrategyType,
)
from strategies.registry import register_strategy


# SQL best practices and rules by dialect
DIALECT_RULES = {
    "bigquery": [
        "Use backticks (`) for reserved keywords and special characters in identifiers",
        "Use TIMESTAMP functions (TIMESTAMP_DIFF, TIMESTAMP_ADD) for date/time operations",
        "Use SAFE_DIVIDE(a, b) instead of a/b to handle division by zero",
        "Use IFNULL(expr, default) or COALESCE for NULL handling",
        "String comparisons are case-sensitive; use LOWER() for case-insensitive matching",
        "Use EXTRACT(part FROM date) for extracting date components",
        "Array and STRUCT types are supported; use UNNEST for array operations",
        "Use LIMIT at the end of queries when only top N results are needed",
    ],
    "duckdb": [
        "Use double quotes for identifiers with special characters",
        "Use strftime for date formatting",
        "COALESCE handles NULL values",
        "Use ILIKE for case-insensitive pattern matching",
        "Supports window functions with OVER clause",
    ],
    "default": [
        "Use explicit JOIN syntax instead of comma-separated tables",
        "Always specify JOIN conditions in ON clause",
        "Use table aliases for clarity in multi-table queries",
        "Handle NULL values explicitly with IS NULL or COALESCE",
        "Use aggregate functions only with GROUP BY when selecting non-aggregated columns",
        "Add ORDER BY when the question implies ranking or sorting",
        "Use DISTINCT to eliminate duplicates when appropriate",
        "Use LIMIT when asking for top/first N results",
    ],
}

COMMON_MISTAKES = [
    "Don't use columns that don't exist in the schema",
    "Don't forget JOIN conditions when combining tables",
    "Don't use aggregate functions without GROUP BY for non-aggregated columns",
    "Don't forget to handle NULL values in comparisons",
    "Don't use = NULL; use IS NULL instead",
    "Don't forget table aliases when column names are ambiguous",
    "Don't use ORDER BY on columns not in SELECT when using DISTINCT",
]


@register_strategy
class ZeroShotEnhancedStrategy(BaseStrategy):
    """
    Enhanced zero-shot SQL generation with dialect-specific rules.
    
    Improves upon basic zero-shot by including:
    - SQL dialect-specific rules and best practices
    - Common mistakes to avoid
    - Explicit output format instructions
    
    Expected accuracy: 55-65% on typical benchmarks.
    
    Example:
        ```python
        strategy = ZeroShotEnhancedStrategy(gemini_client, dialect="bigquery")
        result = await strategy.generate(question, context)
        ```
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        schema_format: SchemaFormat = SchemaFormat.DDL,
        temperature: float = 0.0,
        dialect: str = "bigquery",
        include_common_mistakes: bool = True,
    ):
        """
        Initialize enhanced zero-shot strategy.
        
        Args:
            llm_client: Gemini client instance
            schema_format: Format for schema representation
            temperature: Generation temperature
            dialect: SQL dialect for rules
            include_common_mistakes: Whether to include mistake warnings
        """
        self._llm = llm_client
        self._schema_format = schema_format
        self._temperature = temperature
        self._dialect = dialect
        self._include_common_mistakes = include_common_mistakes
        self._parser = SQLParser()
        self._formatter = SchemaFormatter()
    
    @property
    def name(self) -> str:
        return "zero_shot_enhanced"
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PROMPTING
    
    @property
    def description(self) -> str:
        return "Enhanced zero-shot with SQL rules and best practices"
    
    def _ensure_client(self) -> GeminiClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = GeminiClient()
        return self._llm
    
    def _get_rules(self) -> list[str]:
        """Get dialect-specific rules."""
        rules = DIALECT_RULES.get("default", []).copy()
        dialect_rules = DIALECT_RULES.get(self._dialect, [])
        rules.extend(dialect_rules)
        return rules
    
    def _build_system_prompt(self) -> str:
        """Build enhanced system prompt."""
        dialect_name = self._dialect.upper()
        
        prompt = f"""You are an expert {dialect_name} SQL developer. Your task is to convert natural language questions into accurate, executable SQL queries.

## Your Responsibilities:
1. Analyze the question carefully to understand what data is being requested
2. Use only tables and columns that exist in the provided schema
3. Generate syntactically correct {dialect_name} SQL
4. Follow the rules and best practices provided

## SQL Generation Rules:
{chr(10).join(f"- {rule}" for rule in self._get_rules())}
"""
        
        if self._include_common_mistakes:
            prompt += f"""
## Common Mistakes to Avoid:
{chr(10).join(f"- {mistake}" for mistake in COMMON_MISTAKES)}
"""
        
        prompt += """
## Output Format:
- Generate ONLY the SQL query
- Do NOT include explanations, markdown formatting, or code blocks
- Do NOT include comments in the SQL
- Ensure the query is syntactically valid and executable
"""
        
        return prompt
    
    def _build_prompt(
        self,
        question: str,
        context: GenerationContext,
    ) -> tuple[str, str]:
        """Build the enhanced prompt."""
        # Format schema
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
            include_foreign_keys=True,
        )
        
        # Build user prompt
        user_parts = [
            "## Database Schema",
            schema_text,
            "",
        ]
        
        # Add column descriptions if available
        if context.column_descriptions:
            user_parts.extend([
                "## Column Descriptions",
                context.column_descriptions,
                "",
            ])
        
        # Add sample data if available
        if context.sample_data:
            user_parts.extend([
                "## Sample Data",
                context.sample_data,
                "",
            ])
        
        # Add hints if provided
        if context.hints:
            user_parts.extend([
                "## Hints",
                *[f"- {hint}" for hint in context.hints],
                "",
            ])
        
        # Add question
        user_parts.extend([
            "## Question",
            question,
            "",
            "## SQL Query:",
        ])
        
        system_prompt = self._build_system_prompt()
        user_prompt = "\n".join(user_parts)
        
        return system_prompt, user_prompt
    
    async def generate(
        self,
        question: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """
        Generate SQL with enhanced prompting.
        
        Args:
            question: Natural language question
            context: Generation context with schema
            
        Returns:
            GenerationResult with generated SQL
        """
        start_time = time.perf_counter()
        
        # Build prompt
        system_prompt, user_prompt = self._build_prompt(question, context)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Generate
        client = self._ensure_client()
        response = await client.generate(
            prompt=full_prompt,
            temperature=self._temperature,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Handle failure
        if not response.success:
            return GenerationResult(
                sql=None,
                success=False,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                error=response.error,
                error_type=response.error_type.value if response.error_type else None,
                prompt_used=full_prompt,
                model=response.metrics.model,
            )
        
        # Parse SQL from response
        parse_result = self._parser.parse(response.text)
        
        if not parse_result.is_valid:
            return GenerationResult(
                sql=response.text.strip(),
                success=False,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                prompt_tokens=response.metrics.prompt_tokens,
                completion_tokens=response.metrics.completion_tokens,
                total_tokens=response.metrics.total_tokens,
                temperature=self._temperature,
                model=response.metrics.model,
                error=f"SQL parse error: {parse_result.error}",
                error_type="parse_error",
                raw_response=response.text,
                prompt_used=full_prompt,
            )
        
        return GenerationResult(
            sql=parse_result.sql,
            success=True,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            latency_ms=latency_ms,
            prompt_tokens=response.metrics.prompt_tokens,
            completion_tokens=response.metrics.completion_tokens,
            total_tokens=response.metrics.total_tokens,
            temperature=self._temperature,
            model=response.metrics.model,
            raw_response=response.text,
            prompt_used=full_prompt,
            metadata={
                "schema_format": self._schema_format.value,
                "dialect": self._dialect,
                "complexity_score": parse_result.complexity_score,
                "tables_used": parse_result.tables,
                "include_common_mistakes": self._include_common_mistakes,
            },
        )
    
    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "version": self.version,
            "schema_format": self._schema_format.value,
            "temperature": self._temperature,
            "dialect": self._dialect,
            "include_common_mistakes": self._include_common_mistakes,
            "num_rules": len(self._get_rules()),
        }
