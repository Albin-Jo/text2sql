# text2sql_mvp/core/prompt_builder.py
"""
Dynamic prompt construction for Text-to-SQL generation.
Provides a fluent interface for building prompts with various components.

FIXED: FewShotExample is now imported from data.few_shot_pool instead of being duplicated.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

# FIX: Import FewShotExample from single source of truth
from data.few_shot_pool import FewShotExample


class SchemaFormat(str, Enum):
    """Schema representation format."""
    DDL = "ddl"
    MARKDOWN = "markdown"
    JSON = "json"
    M_SCHEMA = "m_schema"
    COMPACT = "compact"


# NOTE: FewShotExample class has been removed from here.
# It is now imported from data.few_shot_pool to avoid duplication.
# The FewShotExample in few_shot_pool has these fields:
#   - question: str
#   - sql: str
#   - explanation: Optional[str] = None
#   - difficulty: str = "medium"
#   - tags: list[str] = field(default_factory=list)
#   - tables_used: list[str] = field(default_factory=list)
#   - embedding: Optional[np.ndarray] = None


@dataclass
class PromptTemplate:
    """A reusable prompt template."""
    name: str
    system_prompt: str
    user_template: str
    description: str = ""

    def format(self, **kwargs: Any) -> tuple[str, str]:
        """
        Format template with provided values.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        return (
            self.system_prompt.format(**kwargs) if kwargs else self.system_prompt,
            self.user_template.format(**kwargs) if kwargs else self.user_template,
        )


class PromptBuilder:
    """
    Fluent builder for constructing Text-to-SQL prompts.

    Example:
        ```python
        builder = PromptBuilder()
        system, user = (
            builder
            .set_task("Convert natural language to SQL")
            .set_schema(schema_text, format="ddl")
            .add_examples(examples)
            .set_question(question)
            .add_rules(["Use BigQuery syntax", "Handle NULL values"])
            .build()
        )
        ```
    """

    # Pre-built system prompts
    SYSTEM_PROMPTS = {
        "default": """You are an expert SQL developer. Your task is to convert natural language questions into accurate SQL queries.

Guidelines:
- Generate only the SQL query, no explanations unless asked
- Use the exact table and column names from the provided schema
- Handle NULL values appropriately
- Use proper JOIN conditions based on foreign key relationships
- Follow the specified SQL dialect conventions""",

        "bigquery": """You are an expert BigQuery SQL developer. Convert natural language questions into BigQuery-compatible SQL queries.

BigQuery-specific guidelines:
- Use TIMESTAMP functions for date operations
- Use SAFE_DIVIDE for division to handle zeros
- Use IFNULL or COALESCE for NULL handling
- String comparisons are case-sensitive
- Use backticks for reserved words if needed
- Arrays and structs are supported""",

        "analytical": """You are an expert data analyst specializing in SQL. Convert business questions into analytical SQL queries.

Guidelines:
- Generate precise, efficient SQL queries
- Use appropriate aggregation functions
- Include proper GROUP BY for aggregations
- Add ORDER BY when ranking or sorting is implied
- Consider edge cases like empty results or NULL values
- Optimize for readability and performance""",

        "strict": """You are a SQL query generator. Generate ONLY the SQL query with no additional text.

Rules:
- Output must be a valid SQL query only
- No explanations, comments, or markdown formatting
- No backticks or code blocks
- Query must be syntactically correct
- Use exact schema names provided""",
    }

    def __init__(self):
        """Initialize the prompt builder."""
        self._system_prompt: str = self.SYSTEM_PROMPTS["default"]
        self._task: str = ""
        self._schema: str = ""
        self._schema_format: SchemaFormat = SchemaFormat.DDL
        self._examples: list[FewShotExample] = []
        self._question: str = ""
        self._rules: list[str] = []
        self._hints: list[str] = []
        self._context: dict[str, Any] = {}
        self._output_format: str = ""
        self._chain_of_thought: bool = False
        self._decomposition: bool = False
        self._sample_data: str = ""
        self._column_descriptions: str = ""

    def reset(self) -> "PromptBuilder":
        """Reset builder to initial state."""
        self.__init__()
        return self

    def use_system_prompt(self, name: str) -> "PromptBuilder":
        """Use a pre-built system prompt."""
        if name in self.SYSTEM_PROMPTS:
            self._system_prompt = self.SYSTEM_PROMPTS[name]
        return self

    def set_system_prompt(self, prompt: str) -> "PromptBuilder":
        """Set custom system prompt."""
        self._system_prompt = prompt
        return self

    def set_task(self, task: str) -> "PromptBuilder":
        """Set the task description."""
        self._task = task
        return self

    def set_schema(
            self,
            schema: str,
            format: SchemaFormat = SchemaFormat.DDL
    ) -> "PromptBuilder":
        """Set the database schema."""
        self._schema = schema
        self._schema_format = format
        return self

    def add_sample_data(self, sample_data: str) -> "PromptBuilder":
        """Add sample data from tables."""
        self._sample_data = sample_data
        return self

    def add_column_descriptions(self, descriptions: str) -> "PromptBuilder":
        """Add column descriptions/documentation."""
        self._column_descriptions = descriptions
        return self

    def add_example(self, example: FewShotExample) -> "PromptBuilder":
        """Add a single few-shot example."""
        self._examples.append(example)
        return self

    def add_examples(self, examples: list[FewShotExample]) -> "PromptBuilder":
        """Add multiple few-shot examples."""
        self._examples.extend(examples)
        return self

    def set_question(self, question: str) -> "PromptBuilder":
        """Set the user's natural language question."""
        self._question = question
        return self

    def add_rule(self, rule: str) -> "PromptBuilder":
        """Add a single rule/constraint."""
        self._rules.append(rule)
        return self

    def add_rules(self, rules: list[str]) -> "PromptBuilder":
        """Add multiple rules/constraints."""
        self._rules.extend(rules)
        return self

    def add_hint(self, hint: str) -> "PromptBuilder":
        """Add a hint for query generation."""
        self._hints.append(hint)
        return self

    def add_hints(self, hints: list[str]) -> "PromptBuilder":
        """Add multiple hints."""
        self._hints.extend(hints)
        return self

    def add_context(self, key: str, value: Any) -> "PromptBuilder":
        """Add contextual information."""
        self._context[key] = value
        return self

    def set_output_format(self, format_instruction: str) -> "PromptBuilder":
        """Set expected output format."""
        self._output_format = format_instruction
        return self

    def enable_chain_of_thought(self, enabled: bool = True) -> "PromptBuilder":
        """Enable chain-of-thought reasoning."""
        self._chain_of_thought = enabled
        return self

    def enable_decomposition(self, enabled: bool = True) -> "PromptBuilder":
        """Enable question decomposition."""
        self._decomposition = enabled
        return self

    def _format_example(self, example: FewShotExample, include_explanation: bool) -> str:
        """
        Format a FewShotExample for prompt inclusion.

        This helper method ensures compatibility with FewShotExample from few_shot_pool.
        """
        parts = [
            f"Question: {example.question}",
        ]
        if include_explanation and example.explanation:
            parts.append(f"Reasoning: {example.explanation}")
        parts.append(f"SQL: {example.sql}")
        return "\n".join(parts)

    def build(self) -> tuple[str, str]:
        """
        Build the final prompt.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        user_parts = []

        # Task description
        if self._task:
            user_parts.append(f"Task: {self._task}")

        # Schema
        if self._schema:
            user_parts.append(f"Database Schema ({self._schema_format.value}):")
            user_parts.append(self._schema)

        # Column descriptions
        if self._column_descriptions:
            user_parts.append("\nColumn Descriptions:")
            user_parts.append(self._column_descriptions)

        # Sample data
        if self._sample_data:
            user_parts.append("\nSample Data:")
            user_parts.append(self._sample_data)

        # Rules
        if self._rules:
            user_parts.append("\nRules:")
            for i, rule in enumerate(self._rules, 1):
                user_parts.append(f"{i}. {rule}")

        # Hints
        if self._hints:
            user_parts.append("\nHints:")
            for hint in self._hints:
                user_parts.append(f"- {hint}")

        # Few-shot examples
        if self._examples:
            user_parts.append("\nExamples:")
            for i, example in enumerate(self._examples, 1):
                user_parts.append(f"\nExample {i}:")
                # Use helper method for formatting
                user_parts.append(self._format_example(
                    example,
                    include_explanation=self._chain_of_thought
                ))

        # Chain of thought instruction
        if self._chain_of_thought:
            user_parts.append("\nBefore writing the SQL, think through the following:")
            user_parts.append("1. What tables are needed?")
            user_parts.append("2. What columns should be selected?")
            user_parts.append("3. What join conditions are required?")
            user_parts.append("4. What filters/conditions apply?")
            user_parts.append("5. Is aggregation or grouping needed?")

        # Decomposition instruction
        if self._decomposition:
            user_parts.append("\nBreak down the question into sub-questions:")
            user_parts.append("1. Identify the main question components")
            user_parts.append("2. Determine what data each component needs")
            user_parts.append("3. Plan how to combine the results")

        # Question
        if self._question:
            user_parts.append(f"\nQuestion: {self._question}")

        # Output format
        if self._output_format:
            user_parts.append(f"\nOutput Format: {self._output_format}")
        else:
            user_parts.append("\nGenerate the SQL query:")

        # Additional context
        if self._context:
            user_parts.append("\nAdditional Context:")
            for key, value in self._context.items():
                user_parts.append(f"- {key}: {value}")

        user_prompt = "\n".join(user_parts)

        return self._system_prompt, user_prompt

    def build_user_prompt_only(self) -> str:
        """Build only the user prompt (for APIs that don't support system prompts)."""
        system, user = self.build()
        return f"{system}\n\n{user}"


# Pre-built templates for common patterns
class PromptTemplates:
    """Collection of pre-built prompt templates."""

    ZERO_SHOT = PromptTemplate(
        name="zero_shot",
        description="Basic zero-shot SQL generation",
        system_prompt=PromptBuilder.SYSTEM_PROMPTS["default"],
        user_template="""Database Schema:
{schema}

Question: {question}

Generate the SQL query:"""
    )

    ZERO_SHOT_ENHANCED = PromptTemplate(
        name="zero_shot_enhanced",
        description="Enhanced zero-shot with rules",
        system_prompt=PromptBuilder.SYSTEM_PROMPTS["bigquery"],
        user_template="""Database Schema:
{schema}

Rules:
1. Use only tables and columns from the schema
2. Handle NULL values with COALESCE or IFNULL
3. Use appropriate JOIN types based on requirements
4. Add ORDER BY when the question implies sorting
5. Use LIMIT when asking for top/first N results

Question: {question}

Generate the SQL query:"""
    )

    FEW_SHOT = PromptTemplate(
        name="few_shot",
        description="Few-shot with examples",
        system_prompt=PromptBuilder.SYSTEM_PROMPTS["default"],
        user_template="""Database Schema:
{schema}

Examples:
{examples}

Question: {question}

Generate the SQL query:"""
    )

    CHAIN_OF_THOUGHT = PromptTemplate(
        name="chain_of_thought",
        description="Chain-of-thought reasoning",
        system_prompt=PromptBuilder.SYSTEM_PROMPTS["analytical"],
        user_template="""Database Schema:
{schema}

Question: {question}

Let's approach this step by step:
1. First, identify which tables contain the required data
2. Determine what columns are needed in the output
3. Figure out the join conditions between tables
4. Identify any filtering conditions (WHERE clause)
5. Check if aggregation or grouping is needed
6. Consider if sorting is required

Now, based on this analysis, generate the SQL query:"""
    )

    DECOMPOSITION = PromptTemplate(
        name="decomposition",
        description="Question decomposition approach",
        system_prompt=PromptBuilder.SYSTEM_PROMPTS["analytical"],
        user_template="""Database Schema:
{schema}

Question: {question}

Break down this question:
1. What is the main information being requested?
2. What entities/tables are involved?
3. What conditions or filters apply?
4. Are there any calculations or aggregations needed?
5. How should the results be organized?

Now generate the SQL query that addresses all these components:"""
    )

    CORRECTION = PromptTemplate(
        name="correction",
        description="SQL correction prompt",
        system_prompt=PromptBuilder.SYSTEM_PROMPTS["strict"],
        user_template="""Database Schema:
{schema}

Original Question: {question}

Previous SQL attempt:
{previous_sql}

Error encountered:
{error}

Fix the SQL query to address the error:"""
    )

    @classmethod
    def get_template(cls, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        templates = {
            "zero_shot": cls.ZERO_SHOT,
            "zero_shot_enhanced": cls.ZERO_SHOT_ENHANCED,
            "few_shot": cls.FEW_SHOT,
            "chain_of_thought": cls.CHAIN_OF_THOUGHT,
            "decomposition": cls.DECOMPOSITION,
            "correction": cls.CORRECTION,
        }
        return templates.get(name)

    @classmethod
    def list_templates(cls) -> list[str]:
        """List available template names."""
        return [
            "zero_shot",
            "zero_shot_enhanced",
            "few_shot",
            "chain_of_thought",
            "decomposition",
            "correction",
        ]
