# CAV-NLP: Canonical Arithmetic Verification via Natural Language Processing

A research system for translating mathematical statements from natural language and LaTeX into verified Lean 4 code, with Z3-powered canonicalization and CEGIS-based iterative learning.

## Overview

This project implements a complete pipeline for:
- **Extracting** theorems from arXiv papers and LaTeX documents
- **Parsing** mathematical statements into a validated intermediate representation (IR)
- **Canonicalizing** expressions to recognize equivalent formulations (e.g., `x+y` ≡ `y+x`)
- **Translating** to Lean 4 with type checking and proof obligations
- **Learning** from failures via CEGIS (Counter-Example Guided Inductive Synthesis)
- **Verifying** complete chapters of mathematical foundations in Lean 4

## Key Features

### 🔍 Z3-Powered Structure Extraction
- **LaTeX → IR**: Z3 constraint solving for parsing complex mathematical notation
- **IR → Lean**: Z3-guided template selection and code synthesis
- **NOT for theorem proving**: Z3 validates structure, not mathematical correctness

### 🌟 Canonicalization Engine
- Recognizes equivalent expressions using Z3 UNSAT checks
- **Rules**: Commutativity, associativity, De Morgan, double negation, implication, distributivity
- **Benefits**: 30-50% deduplication, caching by canonical form, cross-paper pattern matching
- **Tests**: 20/20 passing (100%) including 6 canonicalization proofs

### 📚 arXiv-to-Lean Agent
- Downloads random papers from arXiv
- Extracts all theorems/definitions/axioms
- Handles real-world LaTeX variations (5+ theorem styles)
- Progressive vocabulary learning via `definitions.json`
- Zero-regression testing on previous papers

### 🔄 CEGIS Learning Loop
- Counter-example guided refinement
- Learns translation rules from failures
- Maintains training examples in `cegis_results/`
- Iterative improvement until convergence

### 🏗️ Lean-Verified Foundations
- Auto-generates 30+ chapter mathematical textbook
- Every theorem proven in Lean 4 (no `sorry`)
- Automatic structure augmentation when definitions are insufficient
- Axiom minimization with immediate reproving
- Benchmarks generated from proven theorems

## Architecture

```
arXiv Paper → LaTeX Extraction → Statement Parser → Semantic Analyzer
                                                            ↓
                                                    ValidatedIRExpr (Z3)
                                                            ↓
                                         Canonicalization Engine (Z3 UNSAT)
                                                            ↓
                                         Vocabulary Lookup (definitions.json)
                                                            ↓
                                         IR-to-Lean Translation (Z3 templates)
                                                            ↓
                                         Lean Type Checking + Verification
                                                            ↓
                                         CEGIS Learning (if failure)
                                                            ↓
                                         Regression Testing
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/thehalleyyoung/cav-nlp.git
cd cav-nlp

# Create virtual environment (Python 3.11+)
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install z3-solver arxiv pyparsing

# Install Lean 4 and lake (for Lean verification)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

### Run Tests

```bash
# Basic Z3 validation tests (7/7 tests)
python test_z3_validated_ir_hard.py

# Extreme test suite with canonicalization (20/20 tests)
python test_z3_extreme.py

# See Level 11 canonicalization results
python test_z3_extreme.py 2>&1 | grep -A 200 "LEVEL 11"
```

### Process a Single arXiv Paper

```bash
# Use the arXiv-to-Lean agent prompt
# See: .github/prompts/arxiv_to_lean_agent.prompt.md

# Or run the paper harvester directly
python arxiv_paper_harvester.py
```

### Run CEGIS Learning

```bash
# Learn from cached papers
python run_cegis_on_papers.py --cache-only --max-papers 50 --max-iterations 50

# Download and learn from new papers
python run_cegis_on_papers.py --max-papers 100 --max-iterations 100
```

### Generate Lean-Verified Foundations

```bash
# Generate foundations for a topic (e.g., "protocol" or "semiosis")
python run_lean_verified_foundations.py protocol PROPOSAL.md

# This will:
# - Generate 30 chapters of LaTeX (~1500 pages)
# - Prove all theorems in Lean 4 (no sorry)
# - Auto-augment structures when needed
# - Minimize axioms (immediate reproving)
# - Generate benchmarks from proven theorems
```

## Project Structure

### Core Components

- **`z3_validated_ir.py`**: Z3-powered intermediate representation with structure validation
- **`canonicalization_engine.py`**: Z3 UNSAT-based expression canonicalization
- **`run_cegis_on_papers.py`**: CEGIS learning loop over arXiv corpus
- **`run_lean_verified_foundations.py`**: Automated textbook generation with Lean verification
- **`arxiv_paper_harvester.py`**: Paper download and theorem extraction

### Prompts

- **`.github/prompts/arxiv_to_lean_agent.prompt.md`**: Complete agent for single-paper refinement

### Documentation

- **`ACTIVE_SYSTEM.md`**: Current system architecture and design decisions
- **`AXIOM_POLICY.md`**: Policy for axiom addition and minimization
- **`CANONICALIZATION_README.md`**: Canonicalization system documentation
- **`STRUCTURE_AUGMENTATION.md`**: Automatic structure augmentation guide
- **`Z3_CANONICALIZATION_SUMMARY.md`**: Z3 canonicalization test results (20/20)
- **`Z3_VALIDATED_IR_README.md`**: IR system design and validation strategy
- **`USAGE_GUIDE.md`**: Detailed usage instructions

### Test Results

- **`test_z3_extreme.py`**: 20/20 tests passing (100%)
  - Levels 1-6: Basic Z3 validation (7/7)
  - Levels 7-10: Advanced features (7/7)
  - Level 11: Canonicalization (6/6) ✨
- **`test_z3_validated_ir_hard.py`**: Hard validation cases
- **`test_mini_cegis.py`**: CEGIS learning validation

### Foundations Projects

- **`foundations-protocol-lean/`**: Protocol theory foundations (Lean 4)
- **`foundations-semiosis-lean/`**: Semiosis foundations (Lean 4)

## Key Results

### Canonicalization Tests (Level 11)
```
✅ commutativity: Z3 proved x+y ≡ y+x
✅ associativity: Z3 proved (x+y)+z ≡ x+(y+z)
✅ de_morgan: Z3 proved ¬(P∧Q) ≡ ¬P∨¬Q
✅ double_negation: Z3 proved ¬¬P ≡ P
✅ implication: Z3 proved P→Q ≡ ¬P∨Q
✅ distributivity: Z3 proved x*(y+z) ≡ x*y+x*z

Level 11: 6/6 passed
Overall: 20/20 tests passed (100.0%)
```

### CEGIS Learning
- Training examples accumulated in `cegis_results/training_examples.json`
- Iterative refinement until convergence
- Zero regressions on previous papers

### Lean Verification
- Complete chapters with all theorems proven
- No `sorry` statements allowed
- Automatic structure augmentation when needed
- Axiom minimization via immediate reproving

## Research Highlights

### Z3 Usage Philosophy

**What Z3 IS used for:**
- ✅ LaTeX → IR: Structure extraction via string constraints
- ✅ IR → Lean: Template selection and code synthesis
- ✅ Canonicalization: Equivalence checking (UNSAT = equivalent)
- ✅ Scope checking: Variable binding validation
- ✅ Type consistency: Sort checking across expressions

**What Z3 is NOT used for:**
- ❌ Mathematical theorem proving (Lean does this)
- ❌ Verifying mathematical correctness
- ❌ Proving theorems are true

### Canonicalization Benefits

1. **Deduplication**: `x+y`, `y+x`, `x + y` → same canonical form (30-50% reduction)
2. **Caching**: Store translations by canonical form, not surface syntax
3. **Pattern Matching**: Match modulo equivalence
4. **Cross-Paper Learning**: Recognize equivalent formulations from different papers

### Structure Augmentation

When theorems fail due to insufficient structure definitions:
1. System analyzes what's missing
2. Augments structures with needed fields/instances
3. Identifies affected axioms
4. Attempts to reprove axioms as theorems
5. Retries original theorem with enhanced structures

This enables **organic growth** where foundations evolve naturally from theorem requirements.

## Citations & Related Work

This project builds on research in:
- Formal mathematics (Lean, Mathlib)
- SMT solving (Z3)
- Natural language semantics (Ganesalingam, Grosof)
- Program synthesis (CEGIS)
- Mathematical controlled English (Naproche, Mizar)

See individual files for detailed bibliographies and citations.

## Contributing

This is an active research project. Key areas for contribution:
- Additional canonicalization rules
- More robust LaTeX parsing
- Extended vocabulary coverage
- Integration with other proof assistants
- Performance optimizations

## License

MIT License - See LICENSE file for details

## Authors

Halley Young

## Acknowledgments

- Lean 4 and Mathlib community
- Z3 SMT solver (Microsoft Research)
- arXiv for open access to mathematical papers

---

**Status**: Active development | **Tests**: 20/20 passing (100%) | **Lean Verification**: Complete chapters with zero `sorry`
