# Active System Files

This document lists the currently active files in the system.

## Core Type Theory & Foundations
- `lean_type_theory.py` - Lean 4 dependent type theory formalization

## Semantic Processing Pipeline
1. `flexible_semantic_parsing.py` - Normalization & parse forest generation (handles surface variation)
2. `z3_semantic_synthesis.py` - Z3-driven semantic synthesis (treats English as under-specified λ-calculus)
3. `compositional_semantics.py` - Static compositional rules (grammar with semantic functions)

## DAG & Code Generation
- `dependency_dag.py` - Extract dependency graph from papers
- `canonical_lean_generator.py` - Generate canonical Lean 4 code from DAG

## Learning & Discovery
- `rule_discovery_from_arxiv.py` - RL-based compositional rule discovery from arXiv corpus

## Verification
- `z3_type_checker.py` - Z3-based formal type checking

## Documentation
- `CAV_README.md` - System overview and architecture
- `USAGE_GUIDE.md` - Comprehensive usage instructions

## Full Pipeline Architecture

```
arXiv Papers
    ↓
[rule_discovery_from_arxiv.py]
    ↓ (discovers compositional rules)
    ↓
Mathematical English Text
    ↓
[flexible_semantic_parsing.py]
    ↓ (normalize to semantic primitives)
    ↓
[dependency_dag.py]
    ↓ (extract theorem/definition DAG)
    ↓
[z3_semantic_synthesis.py]
    ↓ (synthesize valid Lean types via Z3)
    ↓
[compositional_semantics.py]
    ↓ (apply compositional rules)
    ↓
[canonical_lean_generator.py]
    ↓ (generate canonical Lean code)
    ↓
[z3_type_checker.py]
    ↓ (verify type correctness)
    ↓
Canonical Lean 4 Code
```

## Multi-Paradigm Approach

The system combines **four complementary approaches**:

1. **Static Rules** (`compositional_semantics.py`)
   - Hand-crafted production rules
   - Deterministic semantic functions
   
2. **Flexible Normalization** (`flexible_semantic_parsing.py`)
   - Handle surface form variation
   - Equivalence classes for 10+ phrasings
   - Parse forests with Z3 disambiguation
   
3. **Z3 Synthesis** (`z3_semantic_synthesis.py`)
   - Treat English as holes in typed programs
   - Z3 searches valid interpretations
   - CEGIS loop for learning
   
4. **RL Discovery** (`rule_discovery_from_arxiv.py`)
   - Learn new compositional rules from corpus
   - Reward = coverage + correctness + generalization
   - Discovers rule structures, not just weights

## Deprecated Files (moved to `deprecated/`)

- `advanced_statement_extractor.py` - Superseded by `flexible_semantic_parsing.py`
- `arxiv_paper_harvester.py` - Superseded by `rule_discovery_from_arxiv.py`
- `arxiv_corpus_trainer.py` - Superseded by `rule_discovery_from_arxiv.py`
- `comprehensive_grammar.py` - Superseded by `compositional_semantics.py`
- `data_driven_mce.py` - MCE approach abandoned
- `mathematical_controlled_english.py` - MCE approach abandoned
- `orchestrator.py` - Superseded by integrated pipeline
- `paper_to_lean.py` - Superseded by integrated pipeline
- `rl_semantic_learning.py` - Superseded by `rule_discovery_from_arxiv.py`
- `z3_knowledge_dag.py` - Superseded by `z3_semantic_synthesis.py`
- `z3_natural_language_verifier.py` - Superseded by `z3_type_checker.py`
- `MCE_EXAMPLES.md` - MCE approach abandoned
- `MCE_GRAMMAR_SPEC.md` - MCE approach abandoned
- `*.prompt.md` - Old prompt engineering files
- `foundations_*.tex/md` - Example content, not core system
- `run_lean_verified_foundations.py` - Old runner script
