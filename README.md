# Text-to-SQL MVP

Convert natural language questions into SQL queries using Google Gemini Flash.

## 🎯 Overview

This project implements a state-of-the-art Text-to-SQL system with:

- **Multiple Generation Strategies**: Zero-shot, few-shot (static/dynamic), chain-of-thought, and decomposition
- **Multi-Level Correction**: Schema, skeleton, execution, and semantic validation (coming soon)
- **Flexible Schema Support**: DDL, Markdown, JSON, M-Schema formats
- **Production Ready**: Caching, confidence scoring, and security guardrails (coming soon)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd text2sql_mvp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For embedding-based example selection
pip install sentence-transformers

# Copy environment template
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Configuration

Edit `.env` with your settings:

```env
GOOGLE_API_KEY=your-api-key-here
DATABASE_MODE=duckdb  # or bigquery
GEMINI_MODEL=gemini-2.0-flash-exp
```

### Basic Usage

```bash
# Show system info
python main.py info

# Validate domain configuration
python main.py validate --domain finance

# Generate SQL from a question
python main.py generate "How many customers do we have?" --domain finance

# Generate with specific strategy
python main.py generate "List top 10 orders by amount" --strategy few_shot_dynamic

# Test connection
python main.py test-connection

# List available strategies
python main.py list-strategies

# Parse and analyze SQL
python main.py parse "SELECT * FROM customers WHERE status = 'active'"
```

## 📁 Project Structure

```
text2sql_mvp/
├── config/                 # Configuration
│   ├── settings.py        # Pydantic settings
│   └── domains/           # Domain configs (YAML)
├── core/                   # Core components
│   ├── gemini_client.py   # Gemini API client
│   ├── sql_parser.py      # SQL parsing (sqlglot)
│   └── prompt_builder.py  # Prompt construction
├── schema/                 # Schema management
│   ├── manager.py         # Schema loading
│   └── formatter.py       # Multiple formats
├── strategies/             # Generation strategies
│   ├── base.py            # Base interfaces
│   ├── registry.py        # Strategy registry
│   └── prompting/         # Prompting strategies
│       ├── zero_shot.py
│       ├── zero_shot_enhanced.py
│       ├── few_shot_static.py
│       ├── few_shot_dynamic.py
│       ├── chain_of_thought.py
│       └── decomposition.py
├── data/                   # Test data
│   └── domains/finance/   # Finance domain
└── tests/                  # Test suite
```

## 🔧 Strategies

### Available Strategies

| Strategy | Description | Expected Accuracy |
|----------|-------------|-------------------|
| `zero_shot` | Basic schema + question | ~52% |
| `zero_shot_enhanced` | + SQL rules and best practices | ~62% |
| `few_shot_static` | + Fixed examples | ~68% |
| `few_shot_dynamic` | + Similarity-based examples | ~73% |
| `chain_of_thought` | + Step-by-step reasoning | ~74% |
| `decomposition` | + Question decomposition | ~75% |

### Using Strategies Programmatically

```python
import asyncio
from strategies.registry import get_strategy
from strategies.base import GenerationContext
from schema.manager import SchemaManager

# Load schema
manager = SchemaManager()
schema = manager.load_from_sql("data/domains/finance/schema.sql")

# Create context
context = GenerationContext(
    schema=schema,
    question="What is the average account balance?",
    dialect="bigquery",
)

# Get strategy
strategy = get_strategy("few_shot_dynamic")

# Generate
result = asyncio.run(strategy.generate(
    "What is the average account balance?",
    context
))

print(result.sql)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_core/test_sql_parser.py

# Run tests matching pattern
pytest -k "test_zero_shot"
```

## 📊 Finance Domain

The included finance domain has:

- **6 Tables**: customers, accounts, transactions, loans, branches, employees
- **20 Test Cases**: Easy, medium, and hard difficulty
- **15 Few-Shot Examples**: Covering various query patterns

### Tables

| Table | Description |
|-------|-------------|
| customers | Customer personal information |
| accounts | Bank accounts (checking, savings) |
| transactions | Account transactions |
| loans | Customer loans |
| branches | Bank branch locations |
| employees | Bank employees |

## 🔄 Development Phases

- [x] **Phase 1**: Foundation (config, core clients, schema management)
- [x] **Phase 2**: Basic Strategies (zero-shot, few-shot, CoT, decomposition)
- [ ] **Phase 3**: Multi-Path Generation (CHASE-SQL style)
- [ ] **Phase 4**: Multi-Level Correction
- [ ] **Phase 5**: Candidate Selection
- [ ] **Phase 6**: Context & Schema Linking
- [ ] **Phase 7**: Production Patterns
- [ ] **Phase 8**: Evaluation Framework
- [ ] **Phase 9**: CLI & Documentation

## 🤝 Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Run tests before submitting

## 📝 License

MIT License
