# text2sql_mvp/strategies/generation/synthetic_examples.py
"""
Synthetic example generation strategy for Text-to-SQL.

Generates tailored few-shot examples for each input question,
creating examples that are specifically relevant to the query pattern,
tables involved, and complexity level needed.

This is inspired by CHASE-SQL's instance-aware synthetic example generation.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.gemini_client import GeminiClient
from core.sql_parser import SQLParser
from core.prompt_builder import SchemaFormat
from schema.formatter import SchemaFormatter
from schema.manager import SchemaInfo
from strategies.base import (
    BaseStrategy,
    GenerationResult,
    GenerationContext,
    StrategyType,
)
from strategies.registry import register_strategy


@dataclass
class SyntheticExample:
    """A synthetically generated example."""
    question: str
    sql: str
    explanation: Optional[str] = None
    relevance_reason: Optional[str] = None
    tables_used: list[str] = field(default_factory=list)


@dataclass 
class SyntheticExampleConfig:
    """Configuration for synthetic example generation."""
    num_examples: int = 3
    include_explanations: bool = True
    match_complexity: bool = True
    match_tables: bool = True
    generation_temperature: float = 0.3  # Slightly creative for variety
    sql_temperature: float = 0.0  # Deterministic for final SQL


@register_strategy
class SyntheticExamplesStrategy(BaseStrategy):
    """
    Generate SQL using synthetically created relevant examples.
    
    This strategy:
    1. Analyzes the input question for tables, patterns, and complexity
    2. Generates tailored few-shot examples that match the question
    3. Uses these examples to guide final SQL generation
    
    Benefits:
    - Examples are always relevant to the specific question
    - No need to maintain a large example pool
    - Adapts to any schema/domain automatically
    
    Example:
        ```python
        strategy = SyntheticExamplesStrategy(llm_client=client)
        result = await strategy.generate(
            "Find the average loan amount by branch",
            context
        )
        # result.metadata contains generated examples
        ```
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        config: Optional[SyntheticExampleConfig] = None,
        num_examples: int = 3,
        schema_format: SchemaFormat = SchemaFormat.DDL,
        include_explanations: bool = True,
    ):
        """
        Initialize synthetic examples strategy.
        
        Args:
            llm_client: Gemini client instance
            config: Full configuration
            num_examples: Number of examples to generate (shortcut)
            schema_format: Schema format to use
            include_explanations: Include explanations in examples
        """
        self._llm = llm_client
        self._config = config or SyntheticExampleConfig()
        
        if num_examples != 3:
            self._config.num_examples = num_examples
        if not include_explanations:
            self._config.include_explanations = include_explanations
        
        self._schema_format = schema_format
        self._parser = SQLParser()
        self._formatter = SchemaFormatter()
    
    @property
    def name(self) -> str:
        return "synthetic_examples"
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.GENERATION
    
    @property
    def description(self) -> str:
        return f"Generates {self._config.num_examples} tailored examples per question"
    
    def _ensure_client(self) -> GeminiClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = GeminiClient()
        return self._llm
    
    def _analyze_question(self, question: str, schema: SchemaInfo) -> dict[str, Any]:
        """
        Analyze question to determine example requirements.
        
        Returns:
            Analysis dict with complexity hints, likely tables, etc.
        """
        question_lower = question.lower()
        
        analysis = {
            "likely_tables": [],
            "patterns": [],
            "complexity_hints": [],
        }
        
        # Find likely tables mentioned in question
        for table in schema.tables:
            table_name_lower = table.name.lower()
            # Check for table name or related terms
            if table_name_lower in question_lower:
                analysis["likely_tables"].append(table.name)
            # Check for column names that might indicate the table
            for col in table.columns:
                if col.name.lower() in question_lower:
                    if table.name not in analysis["likely_tables"]:
                        analysis["likely_tables"].append(table.name)
                    break
        
        # Detect query patterns
        patterns = {
            "aggregation": ["count", "sum", "average", "avg", "total", "max", "min"],
            "grouping": ["by", "per", "each", "group"],
            "filtering": ["where", "with", "having", "only", "specific"],
            "sorting": ["top", "bottom", "highest", "lowest", "rank", "order"],
            "comparison": ["more than", "less than", "greater", "between"],
            "joining": ["with their", "and their", "along with", "associated"],
            "subquery": ["who have", "that have", "those with", "which have"],
        }
        
        for pattern_name, keywords in patterns.items():
            if any(kw in question_lower for kw in keywords):
                analysis["patterns"].append(pattern_name)
        
        # Estimate complexity
        if len(analysis["patterns"]) >= 3:
            analysis["complexity_hints"].append("complex")
        elif len(analysis["patterns"]) >= 2:
            analysis["complexity_hints"].append("medium")
        else:
            analysis["complexity_hints"].append("simple")
        
        if len(analysis["likely_tables"]) >= 2:
            analysis["complexity_hints"].append("multi_table")
        
        return analysis
    
    async def _generate_synthetic_examples(
        self,
        question: str,
        schema: SchemaInfo,
        analysis: dict[str, Any],
        client: GeminiClient,
    ) -> list[SyntheticExample]:
        """Generate synthetic examples tailored to the question."""
        schema_text = self._formatter.format(
            schema,
            format_type=self._schema_format,
            include_descriptions=True,
        )
        
        # Build requirements based on analysis
        requirements = []
        if analysis["patterns"]:
            requirements.append(f"Use similar SQL patterns: {', '.join(analysis['patterns'])}")
        if analysis["likely_tables"]:
            requirements.append(f"Focus on tables: {', '.join(analysis['likely_tables'])}")
        if "complex" in analysis.get("complexity_hints", []):
            requirements.append("Include examples with JOINs and subqueries")
        elif "medium" in analysis.get("complexity_hints", []):
            requirements.append("Include examples with GROUP BY and aggregations")
        
        requirements_text = "\n".join(f"- {r}" for r in requirements) if requirements else "- Generate varied examples"
        
        explanation_instruction = """
- explanation: Brief explanation of the SQL pattern used""" if self._config.include_explanations else ""
        
        prompt = f"""You are an expert SQL teacher. Generate {self._config.num_examples} example question-SQL pairs that would help someone write SQL for a similar question.

Database Schema:
{schema_text}

Target Question (generate examples SIMILAR to this, not the same):
{question}

Requirements for examples:
{requirements_text}

Generate examples that:
1. Use similar tables and relationships as needed for the target question
2. Demonstrate relevant SQL patterns (JOINs, GROUP BY, aggregations, etc.)
3. Have varied but related complexity
4. Are different from each other

Output format for each example:
---
question: [natural language question]
sql: [complete SQL query]
relevance: [why this example is relevant]{explanation_instruction}
---

Generate {self._config.num_examples} examples:"""
        
        response = await client.generate(
            prompt, 
            temperature=self._config.generation_temperature
        )
        
        if not response.success:
            return []
        
        return self._parse_examples(response.text)
    
    def _parse_examples(self, llm_response: str) -> list[SyntheticExample]:
        """Parse LLM response into SyntheticExample objects."""
        examples = []
        
        # Split by example delimiter
        parts = llm_response.split('---')
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            example_dict = {}
            current_key = None
            current_value = []
            
            for line in part.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Check for field markers
                if line.startswith('question:'):
                    if current_key:
                        example_dict[current_key] = '\n'.join(current_value).strip()
                    current_key = 'question'
                    current_value = [line.replace('question:', '').strip()]
                elif line.startswith('sql:'):
                    if current_key:
                        example_dict[current_key] = '\n'.join(current_value).strip()
                    current_key = 'sql'
                    current_value = [line.replace('sql:', '').strip()]
                elif line.startswith('relevance:'):
                    if current_key:
                        example_dict[current_key] = '\n'.join(current_value).strip()
                    current_key = 'relevance'
                    current_value = [line.replace('relevance:', '').strip()]
                elif line.startswith('explanation:'):
                    if current_key:
                        example_dict[current_key] = '\n'.join(current_value).strip()
                    current_key = 'explanation'
                    current_value = [line.replace('explanation:', '').strip()]
                elif current_key:
                    current_value.append(line)
            
            # Don't forget last field
            if current_key:
                example_dict[current_key] = '\n'.join(current_value).strip()
            
            # Create example if we have question and sql
            if 'question' in example_dict and 'sql' in example_dict:
                # Parse SQL to get tables
                parse_result = self._parser.parse(example_dict['sql'])
                tables_used = parse_result.tables if parse_result.is_valid else []
                
                examples.append(SyntheticExample(
                    question=example_dict['question'],
                    sql=parse_result.sql if parse_result.is_valid else example_dict['sql'],
                    explanation=example_dict.get('explanation'),
                    relevance_reason=example_dict.get('relevance'),
                    tables_used=tables_used,
                ))
        
        return examples
    
    def _format_examples_for_prompt(self, examples: list[SyntheticExample]) -> str:
        """Format synthetic examples for the generation prompt."""
        if not examples:
            return ""
        
        formatted = []
        for i, ex in enumerate(examples, 1):
            parts = [f"Example {i}:"]
            parts.append(f"Q: {ex.question}")
            if self._config.include_explanations and ex.explanation:
                parts.append(f"Explanation: {ex.explanation}")
            parts.append(f"SQL: {ex.sql}")
            formatted.append('\n'.join(parts))
        
        return '\n\n'.join(formatted)
    
    async def generate(
        self,
        question: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """
        Generate SQL using synthetically created examples.
        
        Args:
            question: Natural language question
            context: Generation context with schema
            
        Returns:
            GenerationResult with generated SQL
        """
        start_time = time.perf_counter()
        client = self._ensure_client()
        
        # Step 1: Analyze the question
        analysis = self._analyze_question(question, context.schema)
        
        # Step 2: Generate synthetic examples
        synthetic_examples = await self._generate_synthetic_examples(
            question, context.schema, analysis, client
        )
        
        # Step 3: Build final generation prompt with examples
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
            include_foreign_keys=True,
        )
        
        examples_text = self._format_examples_for_prompt(synthetic_examples)
        
        system_prompt = """You are an expert SQL developer. Convert natural language questions into accurate SQL queries.

Study the examples provided carefully - they demonstrate relevant SQL patterns for your task.

Guidelines:
- Use exact table and column names from the schema
- Follow the patterns shown in the examples
- Generate only the SQL query without explanations
- Handle NULL values appropriately
- Use proper JOIN conditions"""
        
        user_parts = [
            "## Database Schema",
            schema_text,
        ]
        
        if examples_text:
            user_parts.extend([
                "",
                "## Relevant Examples",
                examples_text,
            ])
        
        if context.hints:
            user_parts.extend([
                "",
                "## Hints",
                *[f"- {hint}" for hint in context.hints],
            ])
        
        user_parts.extend([
            "",
            "## Your Task",
            f"Question: {question}",
            "",
            "SQL:",
        ])
        
        full_prompt = f"{system_prompt}\n\n" + "\n".join(user_parts)
        
        # Step 4: Generate final SQL
        response = await client.generate(
            prompt=full_prompt,
            temperature=self._config.sql_temperature,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        if not response.success:
            return GenerationResult(
                sql=None,
                success=False,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                error=response.error,
                error_type=response.error_type.value if response.error_type else None,
                model=response.metrics.model,
                metadata={
                    "analysis": analysis,
                    "num_examples_generated": len(synthetic_examples),
                },
            )
        
        # Parse SQL
        parse_result = self._parser.parse(response.text)
        
        if parse_result.is_valid:
            return GenerationResult(
                sql=parse_result.sql,
                success=True,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                prompt_tokens=response.metrics.prompt_tokens,
                completion_tokens=response.metrics.completion_tokens,
                total_tokens=response.metrics.total_tokens,
                temperature=self._config.sql_temperature,
                model=response.metrics.model,
                raw_response=response.text,
                prompt_used=full_prompt,
                metadata={
                    "analysis": analysis,
                    "synthetic_examples": [
                        {"question": ex.question, "sql": ex.sql, "relevance": ex.relevance_reason}
                        for ex in synthetic_examples
                    ],
                    "num_examples_generated": len(synthetic_examples),
                    "complexity_score": parse_result.complexity_score,
                    "tables_used": parse_result.tables,
                },
            )
        else:
            return GenerationResult(
                sql=response.text.strip(),
                success=False,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                prompt_tokens=response.metrics.prompt_tokens,
                completion_tokens=response.metrics.completion_tokens,
                total_tokens=response.metrics.total_tokens,
                temperature=self._config.sql_temperature,
                model=response.metrics.model,
                error=f"SQL parse error: {parse_result.error}",
                error_type="parse_error",
                raw_response=response.text,
                prompt_used=full_prompt,
                metadata={
                    "analysis": analysis,
                    "num_examples_generated": len(synthetic_examples),
                },
            )
    
    async def generate_examples_only(
        self,
        question: str,
        schema: SchemaInfo,
    ) -> list[SyntheticExample]:
        """
        Generate synthetic examples without final SQL generation.
        
        Useful for debugging or combining with other strategies.
        
        Args:
            question: Natural language question
            schema: Database schema
            
        Returns:
            List of generated synthetic examples
        """
        client = self._ensure_client()
        analysis = self._analyze_question(question, schema)
        return await self._generate_synthetic_examples(question, schema, analysis, client)
    
    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "version": self.version,
            "num_examples": self._config.num_examples,
            "include_explanations": self._config.include_explanations,
            "match_complexity": self._config.match_complexity,
            "match_tables": self._config.match_tables,
            "generation_temperature": self._config.generation_temperature,
            "sql_temperature": self._config.sql_temperature,
            "schema_format": self._schema_format.value,
        }
