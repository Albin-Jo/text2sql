# text2sql_mvp/strategies/generation/divide_and_conquer.py
"""
Divide and Conquer strategy for complex SQL generation.

Decomposes complex questions into simpler sub-questions, generates SQL
for each sub-question, then combines them into a final query. This is
inspired by CHASE-SQL's divide-and-conquer approach.
"""

import re
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
class SubQuestion:
    """A decomposed sub-question."""
    question: str
    purpose: str
    dependencies: list[int] = field(default_factory=list)  # Indices of dependent sub-questions
    sql: Optional[str] = None
    success: bool = False


@dataclass
class DecompositionResult:
    """Result of question decomposition."""
    original_question: str
    sub_questions: list[SubQuestion]
    combination_strategy: str  # "join", "union", "subquery", "cte", "direct"
    is_complex: bool  # Whether decomposition was needed


@register_strategy
class DivideAndConquerStrategy(BaseStrategy):
    """
    Divide and conquer SQL generation for complex queries.
    
    This strategy:
    1. Analyzes the question complexity
    2. Decomposes complex questions into sub-questions
    3. Generates SQL for each sub-question
    4. Combines sub-queries into a final unified query
    
    Best for:
    - Multi-step analytical questions
    - Questions requiring multiple aggregations
    - Questions with implicit comparisons
    
    Example:
        ```python
        strategy = DivideAndConquerStrategy(llm_client=client)
        result = await strategy.generate(
            "Which branch has the highest total loans and what percentage 
             of all loans does it represent?",
            context
        )
        ```
    """
    
    COMPLEXITY_INDICATORS = [
        r'\band\b.*\band\b',           # Multiple "and" conditions
        r'compare|comparison|versus|vs',  # Comparisons
        r'percentage|percent|ratio|proportion',  # Ratios
        r'highest.*lowest|lowest.*highest',  # Multiple extremes
        r'(both|each|every|all).*and',  # Multiple entities
        r'average.*total|total.*average',  # Multiple aggregations
        r'more than.*less than',        # Range comparisons
        r'before.*after|after.*before',  # Temporal ranges
        r'difference|change|growth',    # Calculations
        r'rank.*among|top.*bottom',     # Rankings
    ]
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        schema_format: SchemaFormat = SchemaFormat.DDL,
        temperature: float = 0.0,
        complexity_threshold: int = 2,  # Min indicators to trigger decomposition
        max_sub_questions: int = 5,
        use_cte_combination: bool = True,
    ):
        """
        Initialize divide and conquer strategy.
        
        Args:
            llm_client: Gemini client instance
            schema_format: Schema format to use
            temperature: Generation temperature
            complexity_threshold: Number of complexity indicators to trigger decomposition
            max_sub_questions: Maximum number of sub-questions
            use_cte_combination: Use CTEs to combine sub-queries (vs subqueries)
        """
        self._llm = llm_client
        self._schema_format = schema_format
        self._temperature = temperature
        self._complexity_threshold = complexity_threshold
        self._max_sub_questions = max_sub_questions
        self._use_cte_combination = use_cte_combination
        self._parser = SQLParser()
        self._formatter = SchemaFormatter()
    
    @property
    def name(self) -> str:
        return "divide_and_conquer"
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.GENERATION
    
    @property
    def description(self) -> str:
        return "Decomposes complex questions and combines sub-query results"
    
    def _ensure_client(self) -> GeminiClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = GeminiClient()
        return self._llm
    
    def _assess_complexity(self, question: str) -> tuple[bool, list[str]]:
        """
        Assess if a question is complex enough for decomposition.
        
        Returns:
            (is_complex, list of matched indicators)
        """
        question_lower = question.lower()
        matched_indicators = []
        
        for pattern in self.COMPLEXITY_INDICATORS:
            if re.search(pattern, question_lower):
                matched_indicators.append(pattern)
        
        # Also check question length and clause count
        words = question.split()
        if len(words) > 25:
            matched_indicators.append("long_question")
        
        # Count potential query components
        components = ['who', 'what', 'which', 'how many', 'how much', 'when', 'where']
        component_count = sum(1 for c in components if c in question_lower)
        if component_count >= 2:
            matched_indicators.append("multiple_components")
        
        is_complex = len(matched_indicators) >= self._complexity_threshold
        return is_complex, matched_indicators
    
    async def _decompose_question(
        self,
        question: str,
        schema: SchemaInfo,
        client: GeminiClient,
    ) -> DecompositionResult:
        """Decompose a complex question into sub-questions."""
        schema_text = self._formatter.format(
            schema,
            format_type=self._schema_format,
            include_descriptions=True,
        )
        
        prompt = f"""You are an expert at breaking down complex database questions into simpler parts.

Given a complex question, decompose it into simpler sub-questions that can each be answered with a straightforward SQL query.

Database Schema:
{schema_text}

Complex Question: {question}

Instructions:
1. Identify the main components/steps needed to answer the question
2. Create 2-{self._max_sub_questions} simpler sub-questions
3. Specify how results should be combined (JOIN, UNION, CTE, SUBQUERY, or DIRECT if no combination needed)
4. Note dependencies between sub-questions

Output format:
SUB_QUESTIONS:
1. [sub-question 1] | PURPOSE: [what this calculates] | DEPENDS_ON: [none or comma-separated indices]
2. [sub-question 2] | PURPOSE: [what this calculates] | DEPENDS_ON: [none or 1]
...

COMBINATION_STRATEGY: [JOIN/UNION/CTE/SUBQUERY/DIRECT]
COMBINATION_EXPLANATION: [how to combine the results]

If the question is simple enough to answer directly, respond with:
SUB_QUESTIONS:
1. {question} | PURPOSE: direct answer | DEPENDS_ON: none
COMBINATION_STRATEGY: DIRECT
"""
        
        response = await client.generate(prompt, temperature=0.0)
        
        if not response.success:
            # Fall back to treating as simple question
            return DecompositionResult(
                original_question=question,
                sub_questions=[SubQuestion(question=question, purpose="direct answer")],
                combination_strategy="direct",
                is_complex=False,
            )
        
        return self._parse_decomposition(question, response.text)
    
    def _parse_decomposition(
        self,
        original_question: str,
        llm_response: str,
    ) -> DecompositionResult:
        """Parse LLM decomposition response."""
        sub_questions = []
        combination_strategy = "direct"
        
        lines = llm_response.strip().split('\n')
        in_sub_questions = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('SUB_QUESTIONS:'):
                in_sub_questions = True
                continue
            
            if line.startswith('COMBINATION_STRATEGY:'):
                in_sub_questions = False
                strategy_match = re.search(r'COMBINATION_STRATEGY:\s*(\w+)', line)
                if strategy_match:
                    combination_strategy = strategy_match.group(1).lower()
                continue
            
            if in_sub_questions and line and line[0].isdigit():
                # Parse sub-question line
                # Format: "1. [question] | PURPOSE: [purpose] | DEPENDS_ON: [deps]"
                parts = line.split('|')
                if parts:
                    # Extract question (remove leading number)
                    q_part = parts[0].strip()
                    q_match = re.match(r'\d+\.\s*(.+)', q_part)
                    question_text = q_match.group(1).strip() if q_match else q_part
                    
                    # Extract purpose
                    purpose = ""
                    for part in parts:
                        if 'PURPOSE:' in part.upper():
                            purpose = part.split(':', 1)[1].strip()
                            break
                    
                    # Extract dependencies
                    dependencies = []
                    for part in parts:
                        if 'DEPENDS_ON:' in part.upper():
                            deps_str = part.split(':', 1)[1].strip()
                            if deps_str.lower() != 'none':
                                try:
                                    dependencies = [
                                        int(d.strip()) - 1  # Convert to 0-indexed
                                        for d in deps_str.split(',')
                                        if d.strip().isdigit()
                                    ]
                                except ValueError:
                                    pass
                            break
                    
                    sub_questions.append(SubQuestion(
                        question=question_text,
                        purpose=purpose,
                        dependencies=dependencies,
                    ))
        
        # If parsing failed, treat as simple
        if not sub_questions:
            sub_questions = [SubQuestion(question=original_question, purpose="direct answer")]
            combination_strategy = "direct"
        
        return DecompositionResult(
            original_question=original_question,
            sub_questions=sub_questions,
            combination_strategy=combination_strategy,
            is_complex=len(sub_questions) > 1,
        )
    
    async def _generate_sub_query(
        self,
        sub_question: SubQuestion,
        schema: SchemaInfo,
        context: GenerationContext,
        client: GeminiClient,
        previous_results: dict[int, str],  # idx -> sql
    ) -> str:
        """Generate SQL for a sub-question."""
        schema_text = self._formatter.format(
            schema,
            format_type=self._schema_format,
            include_descriptions=True,
        )
        
        # Build context from previous sub-queries if there are dependencies
        dependency_context = ""
        if sub_question.dependencies:
            dep_parts = []
            for idx in sub_question.dependencies:
                if idx in previous_results:
                    dep_parts.append(f"Sub-query {idx + 1}:\n{previous_results[idx]}")
            if dep_parts:
                dependency_context = "\n\nPrevious sub-queries (you may reference these):\n" + "\n\n".join(dep_parts)
        
        prompt = f"""You are an expert SQL developer. Generate a SQL query for the following sub-question.

Database Schema:
{schema_text}

Sub-question: {sub_question.question}
Purpose: {sub_question.purpose}
{dependency_context}

Generate ONLY the SQL query, nothing else. The query should:
- Be self-contained and executable
- Use proper table/column names from the schema
- Include appropriate JOINs if needed

SQL:"""
        
        response = await client.generate(prompt, temperature=self._temperature)
        
        if not response.success:
            return ""
        
        parse_result = self._parser.parse(response.text)
        return parse_result.sql if parse_result.is_valid else response.text.strip()
    
    async def _combine_sub_queries(
        self,
        decomposition: DecompositionResult,
        sub_queries: dict[int, str],
        schema: SchemaInfo,
        client: GeminiClient,
    ) -> str:
        """Combine sub-queries into a final query."""
        if decomposition.combination_strategy == "direct":
            # Just return the single query
            return sub_queries.get(0, "")
        
        schema_text = self._formatter.format(
            schema,
            format_type=self._schema_format,
            include_descriptions=False,  # Shorter version for combination
        )
        
        # Build sub-queries section
        sub_queries_text = []
        for idx, sq in enumerate(decomposition.sub_questions):
            sql = sub_queries.get(idx, "")
            if sql:
                sub_queries_text.append(f"""
Sub-question {idx + 1}: {sq.question}
Purpose: {sq.purpose}
SQL:
{sql}
""")
        
        combination_method = "WITH (CTE)" if self._use_cte_combination else "subqueries"
        
        prompt = f"""You are an expert SQL developer. Combine the following sub-queries into a single final query.

Database Schema:
{schema_text}

Original Question: {decomposition.original_question}

Sub-queries to combine:
{"".join(sub_queries_text)}

Combination Strategy: {decomposition.combination_strategy}

Instructions:
- Combine using {combination_method}
- The final query should answer the original question completely
- Ensure all sub-query results are properly integrated
- Use meaningful aliases for CTEs/subqueries

Generate ONLY the final combined SQL query:"""
        
        response = await client.generate(prompt, temperature=self._temperature)
        
        if not response.success:
            # Fall back to first sub-query
            return sub_queries.get(0, "")
        
        parse_result = self._parser.parse(response.text)
        return parse_result.sql if parse_result.is_valid else response.text.strip()
    
    async def generate(
        self,
        question: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """
        Generate SQL using divide and conquer approach.
        
        Args:
            question: Natural language question
            context: Generation context with schema
            
        Returns:
            GenerationResult with generated SQL
        """
        start_time = time.perf_counter()
        client = self._ensure_client()
        
        # Step 1: Assess complexity
        is_complex, indicators = self._assess_complexity(question)
        
        # Step 2: Decompose if complex
        if is_complex:
            decomposition = await self._decompose_question(
                question, context.schema, client
            )
        else:
            decomposition = DecompositionResult(
                original_question=question,
                sub_questions=[SubQuestion(question=question, purpose="direct answer")],
                combination_strategy="direct",
                is_complex=False,
            )
        
        # Step 3: Generate SQL for each sub-question
        sub_queries: dict[int, str] = {}
        
        # Sort by dependencies to process in correct order
        processed = set()
        to_process = list(range(len(decomposition.sub_questions)))
        
        while to_process:
            for idx in to_process[:]:
                sq = decomposition.sub_questions[idx]
                # Check if all dependencies are satisfied
                if all(dep in processed for dep in sq.dependencies):
                    sql = await self._generate_sub_query(
                        sq, context.schema, context, client, sub_queries
                    )
                    sub_queries[idx] = sql
                    sq.sql = sql
                    sq.success = bool(sql)
                    processed.add(idx)
                    to_process.remove(idx)
                    break
            else:
                # No progress - break to avoid infinite loop
                break
        
        # Step 4: Combine sub-queries
        if decomposition.is_complex:
            final_sql = await self._combine_sub_queries(
                decomposition, sub_queries, context.schema, client
            )
        else:
            final_sql = sub_queries.get(0, "")
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse and validate final SQL
        parse_result = self._parser.parse(final_sql) if final_sql else None
        
        if parse_result and parse_result.is_valid:
            return GenerationResult(
                sql=parse_result.sql,
                success=True,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                temperature=self._temperature,
                model=client.model,
                metadata={
                    "is_complex": decomposition.is_complex,
                    "complexity_indicators": indicators,
                    "num_sub_questions": len(decomposition.sub_questions),
                    "combination_strategy": decomposition.combination_strategy,
                    "sub_questions": [
                        {"question": sq.question, "purpose": sq.purpose, "sql": sq.sql}
                        for sq in decomposition.sub_questions
                    ],
                    "complexity_score": parse_result.complexity_score,
                    "tables_used": parse_result.tables,
                },
            )
        else:
            return GenerationResult(
                sql=final_sql if final_sql else None,
                success=False,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                temperature=self._temperature,
                model=client.model,
                error="Failed to generate valid combined SQL",
                metadata={
                    "is_complex": decomposition.is_complex,
                    "complexity_indicators": indicators,
                    "num_sub_questions": len(decomposition.sub_questions),
                    "sub_queries_generated": len(sub_queries),
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
            "complexity_threshold": self._complexity_threshold,
            "max_sub_questions": self._max_sub_questions,
            "use_cte_combination": self._use_cte_combination,
            "complexity_indicators": len(self.COMPLEXITY_INDICATORS),
        }
