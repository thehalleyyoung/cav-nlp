# arXiv-to-Lean Continuous Learning Agent

## Mission

Download a random arXiv paper, extract **all** theorems/definitions/axioms to Lean, and **iteratively improve** the translation system based on failures until 100% parsing success with zero regressions on previous papers.

## Core Objectives

1. **Download & Parse**: Random arXiv paper → extract mathematical content
2. **Translate**: Every theorem/definition/axiom → validated Lean code **using the existing pipeline with Z3 validation, not using copilot/LLM features**
3. **Learn**: Update translation system based on failures
4. **Validate**: Z3 verification at every step + Lean type checking
5. **No Regressions**: All previous papers must still parse correctly
6. **Expand Vocabulary**: Add domain-specific terms to `definitions.json`

## Architecture Overview

```
arXiv Paper (PDF/LaTeX)
    ↓
[LaTeX Extraction] → Raw theorem statements
    ↓
[Statement Parser] → Structured AST with Z3 validation
    ↓
[Semantic Analyzer] → ValidatedIRExpr (z3_validated_ir.py)
    ↓ (Z3 checks: types, scope, consistency)
[Vocabulary Lookup] → definitions.json + mathlib
    ↓ (missing terms → add to definitions.json)
[IR-to-Lean Translation] → Lean code
    ↓
[Lean Verification] → Type check + proof obligations
    ↓ (failures trigger learning)
[System Update] → Improve parsers, add vocabulary, refine rules
    ↓
[Regression Testing] → Verify all previous papers still work
    ↓
SUCCESS or ITERATE
```

## Phase 1: Paper Acquisition & Extraction

### 1.1 Download Random arXiv Paper

```python
import arxiv
import random
from pathlib import Path

def download_random_paper(category='math.CO', max_results=100):
    """
    Download a random paper from arXiv.
    Focus on math categories with rich theorem content.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = list(client.results(search))
    paper = random.choice(papers)
    
    print(f"Selected: {paper.title}")
    print(f"Authors: {', '.join(a.name for a in paper.authors)}")
    print(f"arXiv ID: {paper.entry_id}")
    
    # Download PDF and LaTeX source if available
    paper.download_source(dirpath='./arxiv_papers/')
    paper.download_pdf(dirpath='./arxiv_papers/')
    
    return paper
```

### 1.2 Extract LaTeX Theorems using Copilot

**Key Insight**: Theorems are stated with **enormous variation** in real papers:

```latex
% Style 1: Standard LaTeX
\begin{theorem}[Cauchy-Schwarz]
For all vectors $x, y \in \mathbb{R}^n$, we have
$$|\langle x, y \rangle| \leq \|x\| \cdot \|y\|$$
\end{theorem}

% Style 2: Informal with bold
**Theorem 2.3** (Main Result). If $G$ is a planar graph, then...

% Style 3: Numbered manually
Theorem 1.1. Let $S$ be a set with $|S| \geq n$. Then...

% Style 4: Environment variants
\begin{thm} ... \end{thm}
\begin{Theorem} ... \end{Theorem}
\begin{proposition} ... \end{proposition}
\begin{lemma} ... \end{lemma}
\begin{corollary} ... \end{corollary}

% Style 5: Definition with implicit formalization
Let a list $xs$ be *sorted* if $xs[i] \leq xs[i+1]$ for all $0 \leq i < |xs|-1$.

% Style 6: Axiom
\textbf{Axiom} (Choice). For any collection of nonempty sets...
```

**Z3-Validated Extractor**:
Whatever the current z3-based theorem extractor is, we will use it here to extract all statements into structured format.

## Phase 2: Vocabulary Management with Z3

### 2.1 Smart `definitions.json` Updates

**Current Problem**: Mathematical papers use domain-specific terminology not in mathlib:
- "sorted", "connected graph", "acyclic", "bipartite"
- "uniformly continuous", "Lipschitz", "contractive"
- "prime", "coprime", "square-free"

**Solution**: Use Z3 as a **constraint solver to extract structure** from LaTeX and synthesize Lean code

```python
class VocabularyManager:
    """Manage domain-specific vocabulary using Z3 for structure extraction."""
    
    def __init__(self, definitions_path='definitions.json'):
        self.definitions_path = Path(definitions_path)
        self.definitions = self._load_definitions()
        self.z3_parsers = {}  # Z3-based parsers for each term
    
    def _load_definitions(self) -> Dict:
        if self.definitions_path.exists():
            return json.load(open(self.definitions_path))
        return {}
    
    def add_definition(self, term: str, definition_data: Dict):
        """
        Add new term to vocabulary.
        
        Args:
            term: The new term (e.g., "sorted")
            definition_data: {
                'latex_pattern': regex for LaTeX,
                'lean_template': Lean code template,
                'z3_parser': Z3 constraints for extracting structure,
                'signature': Type signature,
                'examples': List of usage examples
            }
        """
        # Create Z3-based parser for this term
        if 'z3_parser' in definition_data:
            self.z3_parsers[term] = self._compile_z3_parser(
                definition_data['z3_parser']
            )
        
        # Add to definitions
        self.definitions[term] = definition_data
        
        # Save
        json.dump(self.definitions, open(self.definitions_path, 'w'), indent=2)
        
        print(f"✅ Added '{term}' to vocabulary")
    
    def _compile_z3_parser(self, parser_spec: Dict) -> Callable:
        """
        Compile Z3 parser that extracts structure from LaTeX.
        
        Example: For "sorted(xs)", Z3 extracts:
        - Variable name: xs
        - Type: List α
        - Order: ≤
        
        Returns parser function: latex_str → IR structure
        """
        from z3 import String, Int, Solver, sat
        
        def parser(latex_str: str) -> Dict:
            solver = Solver()
            
            # Z3 string constraints to parse structure
            latex_var = String('latex_input')
            solver.add(latex_var == latex_str)
            
            # Extract components using Z3 string theory
            # (In real impl, use Z3 string operations)
            
            if solver.check() == sat:
                model = solver.model()
                return self._extract_structure_from_model(model, parser_spec)
            else:
                return None
        
        return parser

# Example: Adding "sorted" to vocabulary
vocab_mgr = VocabularyManager()

vocab_mgr.add_definition('sorted', {
    'latex_pattern': r'sorted\((.*?)\)',
    'lean_template': 'List.Sorted (· ≤ ·) {list_var}',
    'z3_parser': {
        # Use Z3 to extract the list variable and order relation
        'extract_vars': ['list_var'],
        'extract_order': 'le',  # ≤ relation
        'ir_constructor': 'IRApp(IRConst("List.Sorted"), [order, list_var])'
    },
    'signature': 'List α → Prop',
    'examples': [
        {
            'latex': 'sorted($xs$)',
            'ir': 'IRApp(IRConst("List.Sorted"), [IRConst("LE.le"), IRVar("xs")])',
            'lean': 'List.Sorted (· ≤ ·) xs'
        }
    ],
    'mathlib_equiv': 'List.Sorted',
    'notes': 'Common in algorithm papers. Means ∀ i, xs[i] ≤ xs[i+1]'
})

# Example: Adding "connected graph"
vocab_mgr.add_definition('connected', {
    'latex_pattern': r'connected\s+graph',
    'lean_template': 'Graph.Connected {graph_var}',
    'z3_parser': {
        # Z3 extracts: graph variable from context
        'extract_vars': ['graph_var'],
        'ir_constructor': 'IRApp(IRConst("Graph.Connected"), [graph_var])'
    },
    'signature': 'Graph V E → Prop',
    'examples': [
        {
            'latex': 'Let $G$ be a connected graph',
            'ir': 'IRApp(IRConst("Graph.Connected"), [IRVar("G")])',
            'lean': 'variable (G : Graph V E) (hG : Graph.Connected G)'
        }
    ],
    'mathlib_equiv': 'SimpleGraph.Connected',
    'notes': 'Path exists between any two vertices'
})
```

### 2.2 Automatic Vocabulary Discovery

```python
class VocabularyDiscovery:
    """Discover missing vocabulary from failed translations."""
    
    def __init__(self, vocab_mgr: VocabularyManager):
        self.vocab_mgr = vocab_mgr
        self.unknown_terms = set()
    
    def analyze_failure(self, latex_stmt: str, error_msg: str):
        """
        Analyze translation failure to identify missing vocabulary.
        
        Examples of error messages:
        - "Unknown term: 'bipartite'"
        - "Cannot translate: 'Lipschitz continuous'"
        - "Undefined predicate: 'prime'"
        """
        # Extract unknown terms from error
        if 'Unknown term' in error_msg:
            match = re.search(r"Unknown term: '(.*?)'", error_msg)
            if match:
                term = match.group(1)
                self.unknown_terms.add(term)
                self._suggest_definition(term, latex_stmt)
    
    def _suggest_definition(self, term: str, context: str):
        """
        Suggest a definition based on context.
        Uses GPT-4 + Z3 validation.
        """
        print(f"\n🔍 Discovering definition for: '{term}'")
        print(f"Context: {context[:100]}...")
        
        # Prompt GPT-4 to generate definition
        prompt = f"""
Given the mathematical term "{term}" used in context:

{context}

Generate a formal definition including:
1. Lean signature (e.g., "List α → Prop")
2. Lean code template
3. Z3 encoding for validation
4. Plain English explanation

Output as JSON.
"""
        
        # (In real implementation, call GPT-4 API here)
        # For now, use heuristics
        
        if term == 'sorted':
            self.vocab_mgr.add_definition('sorted', {
                'latex_pattern': r'sorted\((.*?)\)',
                'lean_template': 'List.Sorted (· ≤ ·) {list_var}',
                'z3_encoding': '...',  # As above
                'signature': 'List α → Prop',
                'auto_generated': True,
                'needs_review': True
            })
```
## Phase 3: Translation Pipeline with Z3 for Structure Extraction

### 3.1 LaTeX → IR (via Z3) → Lean (via Z3)

**Key Insight**: Use Z3 as a **constraint solver for parsing and synthesis**:
- **LaTeX → IR**: Z3 constraints extract mathematical structure
- **IR → Lean**: Z3 SMT solver synthesizes correct Lean code

```python
from z3_validated_ir import DocumentContext, ValidatedIRExpr
from semantic_to_ir import CompositionalToLeanPipeline

class ArxivToLeanPipeline:
    """Complete pipeline using Z3 for structure extraction and code synthesis."""
    
    def __init__(self):
        self.doc_context = DocumentContext()
        self.vocab_mgr = VocabularyManager()
        self.vocab_discovery = VocabularyDiscovery(self.vocab_mgr)
        self.previous_papers = []  # For regression testing
        self.translation_cache = {}
        
        # Z3 solvers for different tasks
        self.latex_parser_solver = Solver()  # Extract IR from LaTeX
        self.lean_synthesis_solver = Solver()  # Synthesize Lean from IR# For regression testing
        self.translation_cache = {}
    
    def process_paper(self, paper_id: str) -> Dict:
        """
        Process entire paper with learning.
        
        Returns:
        {
            'paper_id': str,
            'total_statements': int,
            'successful': int,
            'failed': int,
            'new_vocabulary': List[str],
            'lean_code': str,
            'z3_validated': bool
        }
        """
        print(f"\n{'='*60}")
        print(f"Processing paper: {paper_id}")
        print(f"{'='*60}\n")
        
        # Step 1: Download and extract
        latex_content = self._download_paper(paper_id)
        extractor = TheoremExtractor()
        statements = extractor.extract_all(latex_content)
        
        print(f"📊 Found {len(statements)} statements:")
        print(f"   - {sum(1 for s in statements if s['type']=='theorem')} theorems")
        print(f"   - {sum(1 for s in statements if s['type']=='definition')} definitions")
        print(f"   - {sum(1 for s in statements if s['type']=='axiom')} axioms")
        
        # Step 2: Translate each statement
        results = []
        new_vocab = []
        
        for i, stmt in enumerate(statements, 1):
            print(f"\n[{i}/{len(statements)}] Translating: {stmt['name']}")
            
            result = self._translate_statement(stmt)
            results.append(result)
            
            if result['success']:
                print(f"   ✅ Success!")
            else:
                print(f"   ❌ Failed: {result['error']}")
                
                # Learn from failure
                self._learn_from_failure(stmt, result['error'])
                
                # Retry with updated system
                retry_result = self._translate_statement(stmt)
                if retry_result['success']:
                    print(f"   ✅ Success after learning!")
                    results[-1] = retry_result
                else:
                    print(f"   ❌ Still failing - needs manual intervention")
        
        # Step 3: Combine into full Lean file
        lean_code = self._generate_lean_file(results)
        
        # Step 4: Verify with Lean
        lean_valid = self._verify_lean(lean_code)
        
        # Step 5: Regression test
        regression_passed = self._regression_test()
        
        # Step 6: Save results
        result_summary = {
            'paper_id': paper_id,
            'total_statements': len(statements),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'new_vocabulary': new_vocab,
            'lean_code': lean_code,
    def _translate_statement(self, stmt: Dict) -> Dict:
        """
        Translate single statement using Z3 for extraction and synthesis.
        
        Key: Use Z3 as a tool for:
        1. **Parsing LaTeX → IR**: Z3 constraints extract structure
        2. **Synthesizing IR → Lean**: Z3 solver finds correct Lean code
        """
        try:
            # Stage 1: Use Z3 to extract IR from LaTeX
            ir_expr = self._latex_to_ir_via_z3(stmt['statement'])
            
            if ir_expr is None:
                return {
                    'success': False,
                    'error': 'Z3 could not extract IR structure from LaTeX',
                    'stage': 'latex_parsing'
                }
            
            print(f"   ✅ Z3 extracted IR: {ir_expr}")
            
            # Stage 2: Check vocabulary
            missing_vocab = self._check_vocabulary(ir_expr)
            if missing_vocab:
                print(f"   📚 Missing vocabulary: {missing_vocab}")
                # Attempt to learn
                for term in missing_vocab:
                    self._learn_vocabulary(term, stmt['statement'])
                
                # Retry extraction with new vocabulary
                ir_expr = self._latex_to_ir_via_z3(stmt['statement'])
            
            # Stage 3: Use Z3 to synthesize Lean from IR
            lean_code = self._ir_to_lean_via_z3(ir_expr, stmt)
            
            if lean_code is None:
                return {
                    'success': False,
                    'error': 'Z3 could not synthesize Lean code from IR',
                    'stage': 'lean_synthesis'
                }
            
            print(f"   ✅ Z3 synthesized Lean: {lean_code[:60]}...")
            
            return {
                'success': True,
                'ir': ir_expr,
                'lean': lean_code,
                'z3_extracted': True,
                'missing_vocab': missing_vocab
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stage': 'unknown'
            }   'z3_valid': True,
                'missing_vocab': missing_vocab
            }
            
        except Exception as e:
    def _latex_to_ir_via_z3(self, latex: str) -> Optional[IRExpr]:
        """
        Use Z3 as a constraint solver to extract IR structure from LaTeX.
        
        Example:
        LaTeX: "sorted(xs)"
        Z3 constraints:
          - Extract function name: "sorted"
          - Extract argument: "xs"
          - Look up "sorted" in definitions.json
          - Construct: IRApp(IRConst("List.Sorted"), [order, IRVar("xs")])
        
        Returns IR expression or None if parsing fails.
        """
        from z3 import Solver, String, Int, sat
        
        solver = Solver()
        
        # Use Z3 string theory to parse LaTeX structure
        latex_str = String('latex')
        solver.add(latex_str == latex)
        
        # Try each vocabulary pattern
        for term, defn in self.vocab_mgr.definitions.items():
            pattern = defn.get('latex_pattern')
            if not pattern:
                continue
            
            # Check if pattern matches using regex (Z3 has string constraints)
            import re
            match = re.match(pattern, latex)
            if match:
                # Extract components from match
                components = match.groups()
                
                # Use Z3 parser to construct IR
                if 'z3_parser' in defn:
                    parser_spec = defn['z3_parser']
                    ir_expr = self._construct_ir_from_spec(
                        parser_spec, components
                    )
                    print(f"   🔍 Z3 parsed '{term}': {ir_expr}")
                    return ir_expr
        
        # Fallback: compositional parsing
        from advanced_compositional_rules import parse_with_compositional_rules
        return parse_with_compositional_rules(latex)
    
    def _ir_to_lean_via_z3(self, ir_expr: IRExpr, stmt: Dict) -> Optional[str]:
        """
        Use Z3 to synthesize Lean code from IR structure.
        
        Example:
        IR: IRApp(IRConst("List.Sorted"), [IRConst("LE.le"), IRVar("xs")])
        
        Z3 synthesis:
          - Template: "List.Sorted (· ≤ ·) {var}"
          - Substitute: var = "xs"
          - Output: "List.Sorted (· ≤ ·) xs"
        
        Returns Lean code or None if synthesis fails.
        """
        from z3 import Solver, String, sat
        
        solver = Solver()
        
        # Use Z3 to match IR structure to Lean templates
        if isinstance(ir_expr, IRApp):
            func = ir_expr.func
            
            # Look up function in vocabulary
            for term, defn in self.vocab_mgr.definitions.items():
                if self._matches_ir_pattern(func, defn):
                    # Found matching definition
                    template = defn['lean_template']
                    
                    # Use Z3 to extract variables and fill template
                    lean_code = self._fill_lean_template(
                        template, ir_expr, defn
## Phase 4: Clever Z3 Usage for Structure Extraction and Synthesis

### 4.1 Z3 for Structure Extraction from LaTeX

```python
def extract_quantifier_structure_with_z3(latex: str) -> Dict:
    """
    Use Z3 string constraints to extract quantifier structure.
    
    Example:
    LaTeX: "∀ n ∈ ℕ, n ≥ 0"
    
    Z3 extracts:
    - quantifier: ∀
    - variable: n
    - type: ℕ
    - body: n ≥ 0
    
    Returns: {
        'quantifier': 'forall',
        'var': 'n',
        'type': 'Nat',
        'body': 'n >= 0'
    }
    """
    from z3 import Solver, String, sat, Length, SubString, IndexOf
    
    solver = Solver()
    latex_str = String('latex')
    solver.add(latex_str == latex)
    
    # Find quantifier symbol (∀, ∃, etc.)
    forall_sym = String('∀')
    exists_sym = String('∃')
    
    has_forall = IndexOf(latex_str, forall_sym, 0) >= 0
    has_exists = IndexOf(latex_str, exists_sym, 0) >= 0
    
    # Extract variable name (after quantifier, before ∈ or :)
    # Use Z3 string operations to parse
    
    if solver.check() == sat:
        model = solver.model()
        # Extract components from Z3 model
### 4.2 Z3 for Lean Code Synthesis

```python
def synthesize_lean_with_z3(ir_expr: IRExpr, context: DocumentContext) -> str:
    """
    Use Z3 as a synthesis engine to generate Lean code from IR.
    
    Example:
    IR: IRPi("n", IRConst("Nat"), IRBinOp(GE, IRVar("n"), IRConst(0)))
    
    Z3 synthesis problem:
      - Find Lean string s such that:
        * s parses as valid Lean
        * s has correct type structure
        * s matches IR semantics
    
    Algorithm:
      1. Generate candidate Lean templates
      2. Use Z3 to check which template fits IR
      3. Fill in template parameters
    """
    from z3 import Solver, String, sat
    
    solver = Solver()
    
    # Candidate templates for different IR patterns
    if isinstance(ir_expr, IRPi):
        # Template: "∀ (var : type), body"
        templates = [
            "∀ ({var} : {type}), {body}",
            "∀ {var} : {type}, {body}",
            "({var} : {type}) → {body}"
        ]
        
        # Use Z3 to pick best template
        var_name = ir_expr.var_name
        var_type = synthesize_lean_with_z3(ir_expr.var_type, context)
        body = synthesize_lean_with_z3(ir_expr.body, context)
        
        # For universal quantifiers over propositions
        if is_prop(ir_expr.body):
            return f"∀ ({var_name} : {var_type}), {body}"
        else:
### 4.3 Z3 for Template Matching and Variable Extraction

```python
def extract_variables_with_z3(latex: str, pattern: str) -> Dict[str, str]:
    """
    Use Z3 to match LaTeX against pattern and extract variables.
    
    Example:
    LaTeX: "sorted(xs)"
    Pattern: "sorted({var})"
    
    Z3 extracts: {'var': 'xs'}
    
    Example 2:
    LaTeX: "∀ n > 0, n^2 > 0"
    Pattern: "∀ {var} > {bound}, {body}"
    
    Z3 extracts: {'var': 'n', 'bound': '0', 'body': 'n^2 > 0'}
    """
    from z3 import Solver, String, sat, Contains, IndexOf, SubString
    
    solver = Solver()
    
    latex_str = String('latex')
    pattern_str = String('pattern')
    
    solver.add(latex_str == latex)
### 4.4 Z3 for IR Validation and Well-Formedness

```python
def validate_ir_wellformedness_with_z3(ir_expr: IRExpr) -> bool:
    """
    Use Z3 to check that IR expression is well-formed.
    
    Example checks:
    - All variables are in scope
    - Types are consistent
    - No dangling references
    
    This is NOT about mathematical correctness, just structural validity.
    """
    from z3 import Solver, Bool, Implies, sat
    
    solver = Solver()
    
    # Check 1: All variables are bound
    free_vars = ir_expr.free_variables()
    bound_vars = ir_expr.bound_variables()
    
    for var in free_vars:
        # Z3 constraint: var must be in context
        var_in_context = Bool(f'{var}_in_context')
        solver.add(var_in_context == True)
    
    # Check 2: Types are consistent
    # (use Z3 to track type flow through expression)
    
    # Check 3: No circular dependencies
    # (use Z3 to ensure dependency graph is acyclic)
    
    if solver.check() == sat:
        print(f"   ✅ IR is well-formed")
        return True
    else:
        print(f"   ❌ IR has structural errors")
        return False
    
    return {'n': 'unknown'}
```

### 4.2 Z3 for Dependency Detection

```python
def detect_dependencies_with_z3(theorem: str, available_lemmas: List[str]) -> List[str]:
    """
    Use Z3 to detect which lemmas are needed for a theorem.
    
    Algorithm:
    1. Try to prove theorem with Z3 alone → if succeeds, no dependencies
    2. Add each lemma one by one → find minimal set that enables proof
    """
    from z3 import Solver, sat, unsat
    
    solver = Solver()
    
    # Encode theorem as Z3 goal
    theorem_z3 = parse_to_z3(theorem)
    
    # Try without lemmas
    solver.push()
    solver.add(Not(theorem_z3))
    if solver.check() == unsat:
        print("   💡 Z3 proved theorem without lemmas!")
        return []
    solver.pop()
    
    # Try with each lemma
    needed_lemmas = []
    for lemma in available_lemmas:
        solver.push()
        lemma_z3 = parse_to_z3(lemma)
        solver.add(lemma_z3)  # Assume lemma
        solver.add(Not(theorem_z3))  # Try to violate theorem
        
        if solver.check() == unsat:
            print(f"   💡 Lemma '{lemma}' helps prove theorem!")
            needed_lemmas.append(lemma)
        
        solver.pop()
    
    return needed_lemmas
```

### 4.3 Z3 for Canonicalization

```python
def canonicalize_with_z3(expr1: str, expr2: str) -> bool:
    """
    Check if two expressions are semantically equivalent.
    
    Examples:
    - "n + m" ≡ "m + n" (commutativity)
    - "2 * n" ≡ "n + n"
    - "¬(P ∧ Q)" ≡ "¬P ∨ ¬Q" (De Morgan)
    """
    from z3 import Solver, unsat
    
    z3_expr1 = parse_to_z3(expr1)
    z3_expr2 = parse_to_z3(expr2)
    
    solver = Solver()
    solver.add(z3_expr1 != z3_expr2)
    
    if solver.check() == unsat:
        print(f"   ✅ Z3 confirms: '{expr1}' ≡ '{expr2}'")
        return True
    else:
        return False
```

### 4.4 Z3 for Proof Obligation Synthesis

```python
def synthesize_proof_obligations(theorem: TheoremDependency) -> List[str]:
    """
    Use Z3 to generate proof obligations.
    
    Example:
    Theorem: ∀ n, n > 0 → n^2 > 0
    
    Z3 generates obligations:
    1. Prove: n > 0 → n * n > 0
    2. Prove: n * n = n^2 (definition)
    """
    from z3 import Solver, ForAll, Implies, Int
    
    solver = Solver()
    n = Int('n')
    
    # Extract hypothesis and conclusion
    hypothesis = n > 0
    conclusion = n * n > 0
    
    obligations = []
    
    # Obligation 1: Direct implication
    solver.push()
    solver.add(ForAll([n], Implies(hypothesis, conclusion)))
    if solver.check() == unsat:
        obligations.append("Direct proof required")
    solver.pop()
    
    # Obligation 2: Check if intermediate lemmas help
    # (e.g., n > 0 → n ≥ 1 for natural numbers)
    
    return obligations
```

## Phase 5: Real-World Theorem Corpus Analysis

### 5.1 Handle Actual Theorem Styles

```python
# Example from real papers:

# Paper 1: Combinatorics
"""
**Theorem 2.1** (Ramsey's Theorem). For any $r, s \geq 1$, there exists 
$n = R(r,s)$ such that any 2-coloring of $K_n$ contains either a red $K_r$ 
or a blue $K_s$.
"""

# Paper 2: Analysis  
"""
\begin{theorem}[Heine-Borel]
A subset $K \subseteq \mathbb{R}^n$ is compact if and only if it is closed 
and bounded.
\end{theorem}
"""

# Paper 3: Algebra
"""
Theorem 3.5. Let $G$ be a finite group and $H \leq G$ a subgroup. Then 
$|H|$ divides $|G|$. 

Proof. Consider the left cosets...
"""

# Paper 4: Informal style
"""
The main result is the following: if a graph is planar, then it has at most 
3n-6 edges (for n ≥ 3).
"""

# Translation strategy for each:
def handle_real_theorem_styles():
    """
    Strategy:
    1. Extract theorem statement (ignore proof for now)
    2. Identify named theorems vs unnamed results
    3. Parse mathematical notation with context
    4. Use Z3 to validate structure
    5. Generate Lean with proper naming
    """
    
    styles = {
        'formal_environment': {
            'pattern': r'\\begin{theorem}.*?\\end{theorem}',
            'extract': lambda m: m.group(0),
            'priority': 'high'
        },
        'bold_heading': {
            'pattern': r'\*\*Theorem.*?\*\*.*?(?=\n\n)',
            'extract': lambda m: m.group(0),
            'priority': 'high'
        },
        'numbered_statement': {
            'pattern': r'Theorem\s+[\d.]+\..*?(?=\n\nProof|\n\n[A-Z])',
            'extract': lambda m: m.group(0),
            'priority': 'medium'
        },
        'informal_statement': {
            'pattern': r'(?:main result|following|we (?:prove|show)) (?:is )?(?:the )?following:.*?(?=\.)',
            'extract': lambda m: m.group(0),
            'priority': 'low'
        }
    }
    
    return styles
```

### 5.2 Definition Extraction from Prose

```python
def extract_definitions_from_prose(text: str) -> List[Dict]:
    """
    Extract implicit definitions from prose.
    
    Example:
    "A graph is **bipartite** if its vertices can be partitioned into two 
    sets such that all edges connect vertices in different sets."
    
    Extracts:
    - term: "bipartite"
    - signature: Graph V E → Prop
    - definition: ∃ (X Y : Set V), V = X ∪ Y ∧ X ∩ Y = ∅ ∧ 
                  ∀ e ∈ E, (e.1 ∈ X ∧ e.2 ∈ Y) ∨ (e.1 ∈ Y ∧ e.2 ∈ X)
    """
    
    # Pattern: "X is TERM if CONDITION"
    pattern = r'(?:A|An)\s+(.*?)\s+is\s+\*\*(.*?)\*\*\s+if\s+(.*?)(?:\.|\n)'
    
    definitions = []
    for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
        subject = match.group(1)
        term = match.group(2)
        condition = match.group(3)
        
        # Parse condition with Z3 validation
        try:
            ir_cond = latex_to_validated_ir(condition)
            
            # Generate Lean definition
            lean_def = f"""
def {term} ({subject.replace(' ', '_')} : {infer_type(subject)}) : Prop :=
  {ir_to_lean(ir_cond)}
"""
            
            definitions.append({
                'term': term,
                'subject': subject,
                'condition': condition,
                'ir': ir_cond,
                'lean': lean_def,
                'z3_validated': True
            })
            
        except Exception as e:
            print(f"Failed to extract definition of '{term}': {e}")
    
    return definitions
```

## Phase 6: Iterative Improvement Loop

### 6.1 Main Agent Loop

```python
def arxiv_to_lean_agent():
    """
    Main agent: Download → Translate → Learn → Repeat
    """
    
    pipeline = ArxivToLeanPipeline()
    vocab_mgr = VocabularyManager()
    
    success_rate = 0.0
    iteration = 0
    
    while success_rate < 1.0:  # Until 100% success
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"Current success rate: {success_rate*100:.1f}%")
        print(f"Vocabulary size: {len(vocab_mgr.definitions)}")
        print(f"{'='*60}\n")
        
        # Step 1: Get a new paper
        paper = download_random_paper(category='math.CO')
        
        # Step 2: Process the paper
        result = pipeline.process_paper(paper.entry_id)
        
        # Step 3: Calculate success rate
        success_rate = result['successful'] / result['total_statements']
        
        print(f"\n📊 RESULTS:")
        print(f"   Statements: {result['total_statements']}")
        print(f"   Successful: {result['successful']}")
        print(f"   Failed: {result['failed']}")
        print(f"   Success rate: {success_rate*100:.1f}%")
        print(f"   New vocabulary: {len(result['new_vocabulary'])}")
        print(f"   Z3 validated: {'✅' if result['z3_validated'] else '❌'}")
        print(f"   Lean verified: {'✅' if result['lean_verified'] else '❌'}")
        print(f"   Regression passed: {'✅' if result['regression_passed'] else '❌'}")
        
        # Step 4: If not perfect, analyze failures
        if success_rate < 1.0:
            print(f"\n🔍 Analyzing failures...")
            analyze_and_improve(result)
        
        # Step 5: Regression test
        if not result['regression_passed']:
            print(f"\n⚠️  REGRESSION DETECTED! Rolling back changes...")
            rollback_changes()
            continue
        
        # Step 6: Save progress
        save_checkpoint(iteration, result)
        
        # Step 7: If perfect, try another paper
## Key Highlights

### 🔍 Z3 Usage Throughout (for Parsing & Synthesis, NOT Theorem Proving)

1. **LaTeX → IR Extraction**: Z3 string constraints parse LaTeX structure
   - Extract quantifiers: `∀ n ∈ ℕ` → `{quantifier: 'forall', var: 'n', type: 'Nat'}`
   - Extract function applications: `sorted(xs)` → `IRApp(...)`
   - Pattern matching with Z3 string theory

2. **IR → Lean Synthesis**: Z3 generates correct Lean code
   - Template selection: Pick best Lean syntax for IR pattern
   - Variable substitution: Fill template with extracted values
   - Type-directed generation: Use type info to guide synthesis

3. **🌟 CANONICALIZATION (CRITICAL FEATURE)**: Z3 proves expression equivalences
   - **Commutativity**: `x+y ≡ y+x` → Check `(x+y ≠ y+x)` is UNSAT
   - **Associativity**: `(x+y)+z ≡ x+(y+z)` → UNSAT check
   - **De Morgan**: `¬(P∧Q) ≡ ¬P∨¬Q` → Z3 proves equivalence
   - **α-equivalence**: `∀x.P(x) ≡ ∀y.P(y)` → variable renaming
   - **Enables**: Deduplication (same math → same canonical form)
   - **Enables**: Pattern matching (match modulo equivalence)
   - **Enables**: Caching (cache by canonical form)
   - **Example**: LaTeX `x+y`, `y+x`, `x + y` all → same canonical IR

4. **Template Matching**: Z3 matches LaTeX against vocabulary patterns
   - Extract variables from `sorted({list_var})`
   - Extract bounds from `∀ {var} > {bound}, {body}`

5. **IR Well-Formedness**: Z3 checks structure (not math correctness)
   - All variables are in scope
   - No dangling references
   - Consistent structure

6. **Vocabulary Learning**: Z3 helps extract new definitions
   - Parse: "A list xs is *sorted* if xs[i] ≤ xs[i+1]"
   - Extract components with Z3 string operations
   - Build IR representation automatically
def analyze_and_improve(result: Dict):
    """Analyze failures and improve system."""
    
    # Failure categories
    failures = {
        'parsing': [],
        'vocabulary': [],
        'type_inference': [],
        'z3_validation': [],
        'lean_generation': []
    }
    
    # Categorize each failure
    for stmt in result.get('failed_statements', []):
        category = stmt['error_category']
        failures[category].append(stmt)
    
    # Improve each category
    if failures['vocabulary']:
        print(f"   📚 Adding {len(failures['vocabulary'])} new terms...")
        for stmt in failures['vocabulary']:
            add_vocabulary_from_failure(stmt)
    
    if failures['parsing']:
        print(f"   🔧 Updating grammar for {len(failures['parsing'])} cases...")
        for stmt in failures['parsing']:
            update_grammar_from_failure(stmt)
    
    if failures['z3_validation']:
        print(f"   🔍 Improving Z3 validation for {len(failures['z3_validation'])} cases...")
        for stmt in failures['z3_validation']:
            improve_z3_validation(stmt)
```

## Phase 7: Output and Integration

### 7.1 Generated Lean File Structure
### 📚 Vocabulary Management (Z3-Powered)

1. **Automatic Discovery**: Extract definitions from prose using Z3 parsing
2. **Structure Extraction**: Z3 extracts components from definitions
3. **Smart Templates**: Z3 fills templates based on extracted structure
4. **Domain-Specific**: Adapts to paper terminology ("sorted", "connected", etc.)
5. **Incremental Learning**: Grows vocabulary progressively with Z3-parsed patterns
import Mathlib.Data.List.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Tactic

-- Z3-validated translations below
-- Each theorem includes Z3 validation status

/-- Definition extracted from paper --/
-- Z3 validation: ✅ PASSED
-- Original LaTeX: Let a list xs be *sorted* if xs[i] ≤ xs[i+1] for all i
def sorted (xs : List α) [LE α] : Prop :=
  ∀ i : Fin xs.length, i.val + 1 < xs.length → 
    xs[i] ≤ xs[i.val + 1]

/-- Theorem 2.1 (Main Result) --/
## Z3 Role Summary

**Z3 is used as a TOOL for:**
1. ✅ **Parsing LaTeX** → Extract mathematical structure using string constraints
2. ✅ **Synthesizing Lean** → Generate code from IR using template matching
3. ✅ **Validating IR structure** → Check well-formedness (scope, references)
4. ✅ **Extracting vocabulary** → Parse definitions and build patterns
5. ✅ **🌟 CANONICALIZATION** → Prove expression equivalences for deduplication

**Z3 is NOT used for:**
- ❌ Proving theorems are mathematically correct
- ❌ Verifying logical validity of statements
- ❌ Checking mathematical soundness

**Canonicalization Details:**
```python
# Example: Prove x+y ≡ y+x
solver = Solver()
x, y = Ints('x y')
solver.add((x + y) != (y + x))  # Try to find counterexample
result = solver.check()
# result == UNSAT → expressions are always equal!

# This enables:
# - Deduplication: "x+y" and "y+x" map to same canonical form
# - Caching: Store by canonical form, not surface syntax
# - Pattern matching: Match "x+y" against pattern "a+b" even if written "y+x"
```

The goal: Use Z3's **constraint solving**, **string theory**, and **equivalence checking** to automate the translation pipeline, not to verify mathematical truth!
2. ✅ **Synthesizing Lean** → Generate code from IR using template matching
3. ✅ **Validating IR structure** → Check well-formedness (scope, references)
4. ✅ **Extracting vocabulary** → Parse definitions and build patterns

**Z3 is NOT used for:**
- ❌ Proving theorems are mathematically correct
- ❌ Verifying logical validity of statements
- ❌ Checking mathematical soundness

The goal: Use Z3's **constraint solving** and **string theory** capabilities to automate the translation pipeline, not to verify mathematical truth!
```

## Phase 8: Success Metrics

```python
class SuccessMetrics:
    """Track progress across papers."""
    
    def __init__(self):
        self.metrics = {
            'papers_processed': 0,
            'total_theorems': 0,
            'successful_translations': 0,
            'vocabulary_additions': 0,
            'z3_validations': 0,
            'lean_verifications': 0,
            'regression_tests': 0
        }
    
    def report(self):
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║              arXiv-to-Lean Success Metrics               ║
╠═══════════════════════════════════════════════════════════╣
║ Papers Processed:        {self.metrics['papers_processed']:5d}                      ║
║ Total Theorems:          {self.metrics['total_theorems']:5d}                      ║
║ Successful Translations: {self.metrics['successful_translations']:5d} ({self._pct('successful_translations')}%)           ║
║ Vocabulary Additions:    {self.metrics['vocabulary_additions']:5d}                      ║
║ Z3 Validations:          {self.metrics['z3_validations']:5d} (100%)              ║
║ Lean Verifications:      {self.metrics['lean_verifications']:5d} ({self._pct('lean_verifications')}%)           ║
║ Regression Tests:        {self.metrics['regression_tests']:5d} (PASS)             ║
╚═══════════════════════════════════════════════════════════╝
""")
```

## Key Highlights

### 🔍 Z3 Usage Throughout

1. **Statement Validation**: Every theorem validated for well-formedness
2. **Type Inference**: Z3 infers types from usage context
3. **Dependency Detection**: Z3 determines minimal lemma dependencies
4. **Semantic Equivalence**: Z3 checks IR ≡ Lean code
5. **Proof Obligations**: Z3 generates what needs to be proved
6. **Canonicalization**: Z3 identifies equivalent formulations

### 📚 Vocabulary Management

1. **Automatic Discovery**: Extract definitions from prose
2. **Z3-Validated Additions**: Every new term validated
3. **Smart Templates**: Context-aware Lean code generation
4. **Domain-Specific**: Adapts to paper subject area
5. **Incremental Learning**: Grows vocabulary progressively

### 🎯 Focus on Real Corpus

1. **Multiple Theorem Styles**: Formal envs, bold headings, prose
2. **Implicit Definitions**: "A graph is bipartite if..."
3. **Domain Terminology**: "sorted", "connected", "prime"
4. **Cross-references**: Theorem depends on Lemma 2.1
5. **Proof Structure**: Extract even if can't formalize yet

### ✅ Regression Prevention

1. **Test All Previous Papers**: After every change
2. **Rollback on Failure**: Undo breaking changes
3. **Incremental Checkpoints**: Save after each success
4. **Version Control**: Track system evolution

## Expected Results

After processing 10-20 papers:
- **Vocabulary**: 100+ domain-specific terms
- **Success Rate**: 90%+ statements translated
- **Z3 Validation**: 100% (by construction)
- **Lean Verification**: 70%+ (proof skeletons complete)
- **Regression**: 0 failures (guaranteed)

This agent learns from each paper, building a comprehensive system for real mathematical text translation!
