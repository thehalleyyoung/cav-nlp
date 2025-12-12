"""
Flexible Semantic Parsing via Z3-Driven Normalization
======================================================

Handle the massive variation in mathematical English by:
1. Normalizing to canonical semantic forms using Z3
2. Learning equivalence classes of phrasings
3. Using type constraints to disambiguate

Key insight: Don't parse surface forms directly. Instead:
- Extract semantic primitives (quantifiers, predicates, relations)
- Let Z3 find valid type-theoretic interpretations
- Learn that "for all", "for every", "given any" → same ∀
"""

from z3 import *
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Callable
import re
from collections import defaultdict

from lean_type_theory import LeanType, LeanExpr, Context


@dataclass
class SemanticPrimitive:
    """
    Canonical semantic primitive extracted from text.
    
    These are invariant across surface form variations.
    """
    kind: str  # 'QUANTIFIER', 'PREDICATE', 'RELATION', 'OPERATOR'
    canonical_form: str  # Normalized representation
    surface_spans: List[Tuple[int, int]]  # Where it appears in text
    confidence: float  # How certain we are
    
    # Z3 constraints this primitive must satisfy
    type_constraints: List[BoolRef] = field(default_factory=list)


class SemanticNormalizer:
    """
    Normalize diverse English phrasings to canonical semantic forms.
    
    Uses patterns + Z3 to handle variations like:
    - "for all x" ≡ "for every x" ≡ "given any x" ≡ "let x be arbitrary"
    - "x > 0" ≡ "x is positive" ≡ "x greater than zero"
    - "continuous" ≡ "is continuous" ≡ "continuity holds"
    """
    
    def __init__(self):
        self.equivalence_classes = self._build_equivalence_classes()
        self.pattern_library = self._build_pattern_library()
        
    def _build_equivalence_classes(self) -> Dict[str, Set[str]]:
        """Define equivalence classes of phrasings."""
        return {
            'UNIVERSAL_QUANTIFIER': {
                'for all', 'for every', 'for each', 'for any',
                'given any', 'given arbitrary', 'let any', 'let arbitrary',
                'take any', 'take arbitrary', 'consider any', 'consider arbitrary',
                'for arbitrary', 'whenever', 'suppose', 'assume',
                'if we take any', 'if we choose any'
            },
            
            'EXISTENTIAL_QUANTIFIER': {
                'there exists', 'there is', 'there are',
                'for some', 'we can find', 'one can find',
                'it is possible to find', 'we have some',
                'there is at least one', 'at least one'
            },
            
            'IMPLICATION': {
                'implies', 'if then', 'only if', 'provided that',
                'entails', 'it follows that', 'we have',
                'we obtain', 'we get', 'whence', 'hence', 'thus',
                'therefore', 'consequently', 'so'
            },
            
            'CONJUNCTION': {
                'and', 'moreover', 'furthermore', 'also',
                'in addition', 'as well as', 'together with',
                'along with', 'both and', 'simultaneously'
            },
            
            'DISJUNCTION': {
                'or', 'alternatively', 'either or',
                'at least one of', 'one of the following'
            },
            
            'NEGATION': {
                'not', 'no', 'never', 'without', 'fails to',
                'does not', 'is not', 'cannot', 'it is false that'
            },
            
            'EQUALITY': {
                'equals', 'is equal to', 'is the same as',
                '=', 'identical to', 'coincides with'
            },
            
            'MEMBERSHIP': {
                'in', 'belongs to', 'is an element of',
                'is a member of', '∈', 'lies in', 'is contained in'
            },
            
            'COMPARISON': {
                'greater than': 'GT',
                'less than': 'LT',
                'at least': 'GE',
                'at most': 'LE',
                'exceeds': 'GT',
                'below': 'LT',
                'above': 'GT'
            },
            
            'PROPERTIES': {
                'continuous': 'continuous',
                'is continuous': 'continuous',
                'continuity holds': 'continuous',
                'satisfies continuity': 'continuous',
                'differentiable': 'differentiable',
                'is differentiable': 'differentiable',
                'has a derivative': 'differentiable',
                'bounded': 'bounded',
                'is bounded': 'bounded',
                'has a bound': 'bounded',
            }
        }
    
    def _build_pattern_library(self) -> List['FlexiblePattern']:
        """
        Build library of flexible patterns.
        
        Each pattern matches multiple surface forms but extracts
        the same semantic primitive.
        """
        patterns = []
        
        # Universal quantification patterns
        quant_words = '|'.join(self.equivalence_classes['UNIVERSAL_QUANTIFIER'])
        patterns.extend([
            FlexiblePattern(
                name='universal_simple',
                regex=rf'({quant_words})\s+(\w+)\s+in\s+(\w+)',
                semantic_template='∀ {1} : {2}, {body}',
                priority=10
            ),
            FlexiblePattern(
                name='universal_typed',
                regex=rf'({quant_words})\s+(\w+)\s*:\s*(\w+)',
                semantic_template='∀ {1} : {2}, {body}',
                priority=9
            ),
            FlexiblePattern(
                name='universal_such_that',
                regex=rf'({quant_words})\s+(\w+)\s+such that\s+(.+?),',
                semantic_template='∀ {1} : {{type}}, {2} → {body}',
                priority=8
            ),
            FlexiblePattern(
                name='universal_implicit',
                regex=rf'let\s+(\w+)\s+be\s+(?:an?|any)\s+(\w+)',
                semantic_template='∀ {0} : {1}, {body}',
                priority=7
            ),
        ])
        
        # Existential patterns
        exist_words = '|'.join(self.equivalence_classes['EXISTENTIAL_QUANTIFIER'])
        patterns.extend([
            FlexiblePattern(
                name='existential_simple',
                regex=rf'({exist_words})\s+(?:an?|some)\s+(\w+)\s+in\s+(\w+)',
                semantic_template='∃ {1} : {2}, {body}',
                priority=10
            ),
            FlexiblePattern(
                name='existential_such_that',
                regex=rf'({exist_words})\s+(\w+)\s+such that\s+(.+)',
                semantic_template='∃ {1} : {{type}}, {2}',
                priority=9
            ),
        ])
        
        # Implication patterns
        impl_words = '|'.join(self.equivalence_classes['IMPLICATION'])
        patterns.extend([
            FlexiblePattern(
                name='implication_if_then',
                regex=r'if\s+(.+?)\s+then\s+(.+)',
                semantic_template='{0} → {1}',
                priority=10
            ),
            FlexiblePattern(
                name='implication_when',
                regex=r'when\s+(.+?),\s*(?:then\s+)?(.+)',
                semantic_template='{0} → {1}',
                priority=9
            ),
            FlexiblePattern(
                name='implication_implies',
                regex=r'(.+?)\s+(?:implies|entails)\s+(.+)',
                semantic_template='{0} → {1}',
                priority=8
            ),
            FlexiblePattern(
                name='implication_only_if',
                regex=r'(.+?)\s+only if\s+(.+)',
                semantic_template='{1} → {0}',  # Reversed!
                priority=8
            ),
        ])
        
        # Property patterns (flexible predicate application)
        patterns.extend([
            FlexiblePattern(
                name='property_is_adj',
                regex=r'(\w+)\s+is\s+(\w+)',
                semantic_template='{1}({0})',
                priority=5
            ),
            FlexiblePattern(
                name='property_adj_noun',
                regex=r'(?:a|an)\s+(\w+)\s+(\w+)',
                semantic_template='{0}({1})',
                priority=4
            ),
            FlexiblePattern(
                name='property_has',
                regex=r'(\w+)\s+has\s+(?:a|an|the)\s+(\w+)',
                semantic_template='Has{1}({0})',
                priority=4
            ),
        ])
        
        # Comparison patterns
        patterns.extend([
            FlexiblePattern(
                name='comparison_symbolic',
                regex=r'(\w+)\s*([<>≤≥=])\s*(\w+)',
                semantic_template='{1}({0}, {2})',
                priority=10
            ),
            FlexiblePattern(
                name='comparison_verbal',
                regex=r'(\w+)\s+(?:is\s+)?(greater than|less than|at least|at most)\s+(\w+)',
                semantic_template='{1}({0}, {2})',
                priority=9
            ),
        ])
        
        return patterns
    
    def normalize(self, text: str) -> List[SemanticPrimitive]:
        """
        Extract semantic primitives from text.
        
        Returns canonical forms regardless of surface variation.
        """
        primitives = []
        
        # Apply all patterns
        for pattern in self.pattern_library:
            matches = pattern.find_all(text)
            
            for match in matches:
                # Determine canonical form
                canonical = pattern.to_canonical(match)
                
                primitive = SemanticPrimitive(
                    kind=pattern.infer_kind(),
                    canonical_form=canonical,
                    surface_spans=[match.span()],
                    confidence=pattern.priority / 10.0
                )
                
                primitives.append(primitive)
        
        # Merge overlapping primitives
        primitives = self._merge_overlapping(primitives)
        
        # Resolve ambiguities using Z3
        primitives = self._resolve_with_z3(primitives, text)
        
        return primitives
    
    def _merge_overlapping(self, primitives: List[SemanticPrimitive]) -> List[SemanticPrimitive]:
        """Merge primitives that span overlapping text."""
        # Sort by position
        primitives.sort(key=lambda p: p.surface_spans[0][0])
        
        merged = []
        i = 0
        while i < len(primitives):
            current = primitives[i]
            
            # Check for overlaps with next
            j = i + 1
            while j < len(primitives):
                next_prim = primitives[j]
                
                # Check if spans overlap
                curr_end = current.surface_spans[0][1]
                next_start = next_prim.surface_spans[0][0]
                
                if next_start < curr_end:
                    # Overlap! Choose higher confidence
                    if next_prim.confidence > current.confidence:
                        current = next_prim
                    j += 1
                else:
                    break
            
            merged.append(current)
            i = j if j > i + 1 else i + 1
        
        return merged
    
    def _resolve_with_z3(self, primitives: List[SemanticPrimitive], 
                        text: str) -> List[SemanticPrimitive]:
        """
        Use Z3 to resolve ambiguous primitives.
        
        If multiple interpretations are possible, Z3 picks the one
        that satisfies type constraints.
        """
        # Group primitives by kind
        by_kind = defaultdict(list)
        for p in primitives:
            by_kind[p.kind].append(p)
        
        # For each group with ambiguity, use Z3
        resolved = []
        
        for kind, group in by_kind.items():
            if len(group) > 1 and self._are_overlapping(group):
                # Ambiguity! Use Z3 to pick best
                best = self._z3_disambiguate(group, text)
                resolved.append(best)
            else:
                resolved.extend(group)
        
        return resolved
    
    def _are_overlapping(self, primitives: List[SemanticPrimitive]) -> bool:
        """Check if primitives overlap in text span."""
        for i, p1 in enumerate(primitives):
            for p2 in primitives[i+1:]:
                span1 = p1.surface_spans[0]
                span2 = p2.surface_spans[0]
                if span1[0] < span2[1] and span2[0] < span1[1]:
                    return True
        return False
    
    def _z3_disambiguate(self, candidates: List[SemanticPrimitive],
                        text: str) -> SemanticPrimitive:
        """Use Z3 to pick best interpretation."""
        solver = Solver()
        
        # Create Z3 variables for each candidate
        choice_vars = [Bool(f'choose_{i}') for i in range(len(candidates))]
        
        # Exactly one must be chosen
        solver.add(Or(choice_vars))
        solver.add(Sum([If(v, 1, 0) for v in choice_vars]) == 1)
        
        # Add type constraints for each candidate
        for i, (candidate, choice) in enumerate(zip(candidates, choice_vars)):
            for constraint in candidate.type_constraints:
                solver.add(Implies(choice, constraint))
        
        # Prefer higher confidence
        # (Use Optimize to maximize confidence)
        opt = Optimize()
        for constraint in solver.assertions():
            opt.add(constraint)
        
        # Maximize confidence
        confidence_sum = Sum([
            If(choice, int(cand.confidence * 100), 0)
            for choice, cand in zip(choice_vars, candidates)
        ])
        opt.maximize(confidence_sum)
        
        if opt.check() == sat:
            model = opt.model()
            for i, choice in enumerate(choice_vars):
                if model.evaluate(choice):
                    return candidates[i]
        
        # Fallback: highest confidence
        return max(candidates, key=lambda c: c.confidence)


@dataclass
class FlexiblePattern:
    """
    A flexible pattern that matches multiple surface forms.
    """
    name: str
    regex: str
    semantic_template: str
    priority: int
    
    def find_all(self, text: str) -> List[re.Match]:
        """Find all matches in text."""
        pattern = re.compile(self.regex, re.IGNORECASE)
        return list(pattern.finditer(text))
    
    def to_canonical(self, match: re.Match) -> str:
        """Convert match to canonical form."""
        groups = match.groups()
        
        # Simple template substitution
        result = self.semantic_template
        for i, group in enumerate(groups):
            result = result.replace(f'{{{i}}}', group if group else '')
        
        return result
    
    def infer_kind(self) -> str:
        """Infer semantic kind from pattern name."""
        if 'universal' in self.name:
            return 'UNIVERSAL_QUANTIFIER'
        elif 'existential' in self.name:
            return 'EXISTENTIAL_QUANTIFIER'
        elif 'implication' in self.name:
            return 'IMPLICATION'
        elif 'comparison' in self.name:
            return 'COMPARISON'
        else:
            return 'PREDICATE'


class Z3FlexibleSemanticSynthesizer:
    """
    Extend Z3SemanticSynthesizer to handle surface variation.
    
    Pipeline:
    1. Normalize text → semantic primitives
    2. Construct ambiguous parse forest
    3. Z3 searches for valid interpretation
    4. Learn from successful interpretations
    """
    
    def __init__(self):
        from z3_semantic_synthesis import Z3SemanticAlgebra
        self.algebra = Z3SemanticAlgebra()
        self.normalizer = SemanticNormalizer()
        self.learned_normalizations: Dict[str, str] = {}
    
    def synthesize_flexible(self, text: str, 
                           context: Optional[Context] = None) -> List[LeanExpr]:
        """
        Synthesize semantics from potentially varied text.
        
        Args:
            text: English mathematical statement (any phrasing)
            context: Optional typing context
            
        Returns:
            List of valid Lean expressions
        """
        # Step 1: Normalize to semantic primitives
        primitives = self.normalizer.normalize(text)
        
        # Step 2: Generate parse forest (all possible combinations)
        parse_forest = self._generate_parse_forest(primitives)
        
        # Step 3: For each parse, try Z3 synthesis
        valid_interpretations = []
        
        for parse in parse_forest:
            # Create Z3 synthesis problem
            solver = Solver()
            
            # Encode parse as constraints
            result_var = Const('result', self.algebra.LeanType)
            
            # Add constraints from primitives
            for prim in parse.primitives:
                self._encode_primitive_constraints(solver, prim, result_var)
            
            # Check satisfiability
            if solver.check() == sat:
                model = solver.model()
                lean_expr = self._extract_lean_expr(model, result_var)
                valid_interpretations.append(lean_expr)
        
        # Step 4: Rank by simplicity and learn
        valid_interpretations.sort(key=lambda e: self._complexity(e))
        
        # Learn successful normalizations
        if valid_interpretations:
            self._learn_normalization(text, primitives, valid_interpretations[0])
        
        return valid_interpretations
    
    def _generate_parse_forest(self, primitives: List[SemanticPrimitive]) -> List['Parse']:
        """Generate all possible parse trees from primitives."""
        # Build parse forest by combining primitives
        
        if len(primitives) == 0:
            return []
        
        if len(primitives) == 1:
            return [Parse(primitives=[primitives[0]], structure='atomic')]
        
        # Try different structural combinations
        forest = []
        
        # Linear composition (left-to-right)
        forest.append(Parse(primitives=primitives, structure='linear'))
        
        # Hierarchical (quantifier scopes body)
        for i, p in enumerate(primitives):
            if p.kind in ['UNIVERSAL_QUANTIFIER', 'EXISTENTIAL_QUANTIFIER']:
                # This quantifier scopes everything after it
                body_prims = primitives[i+1:]
                body_parse = self._generate_parse_forest(body_prims)
                forest.append(Parse(
                    primitives=[p] + body_prims,
                    structure='scoped',
                    head=p,
                    body=body_parse[0] if body_parse else None
                ))
        
        # Binary operations (implications, conjunctions)
        for i in range(len(primitives)):
            p = primitives[i]
            if p.kind in ['IMPLICATION', 'CONJUNCTION']:
                left_prims = primitives[:i]
                right_prims = primitives[i+1:]
                
                if left_prims and right_prims:
                    forest.append(Parse(
                        primitives=primitives,
                        structure='binary',
                        operator=p,
                        left=Parse(primitives=left_prims, structure='linear'),
                        right=Parse(primitives=right_prims, structure='linear')
                    ))
        
        return forest
    
    def _encode_primitive_constraints(self, solver: Solver, 
                                     prim: SemanticPrimitive,
                                     result_var):
        """Encode primitive's constraints in Z3."""
        
        if prim.kind == 'UNIVERSAL_QUANTIFIER':
            # Result must be Pi type
            # Extract variable and domain from canonical form
            # ∀ x : T, body
            match = re.match(r'∀\s+(\w+)\s+:\s+(\w+)', prim.canonical_form)
            if match:
                var_name, domain = match.groups()
                
                # Create Pi type constraint
                pi_type = self.algebra.LeanType.Pi_Type(
                    String(var_name),
                    Const(f'domain_{var_name}', self.algebra.LeanType),
                    Const(f'body_{var_name}', self.algebra.LeanType)
                )
                
                solver.add(result_var == pi_type)
        
        elif prim.kind == 'IMPLICATION':
            # Result must be arrow type
            antecedent = Const('antecedent', self.algebra.LeanType)
            consequent = Const('consequent', self.algebra.LeanType)
            
            arrow_type = self.algebra.LeanType.Pi_Type(
                String('_'),
                antecedent,
                consequent
            )
            
            solver.add(result_var == arrow_type)
        
        # Add any additional constraints from primitive
        for constraint in prim.type_constraints:
            solver.add(constraint)
    
    def _extract_lean_expr(self, model: ModelRef, result_var) -> LeanExpr:
        """Extract Lean expression from Z3 model."""
        # Convert Z3 model to Lean AST
        from lean_type_theory import VarExpr
        return VarExpr("synthesized")
    
    def _complexity(self, expr: LeanExpr) -> float:
        """Compute complexity score for ranking."""
        return 1.0
    
    def _learn_normalization(self, original_text: str, 
                            primitives: List[SemanticPrimitive],
                            successful_expr: LeanExpr):
        """Learn that this normalization was successful."""
        # Store mapping from surface form to canonical form
        canonical_key = ' '.join(p.canonical_form for p in primitives)
        self.learned_normalizations[original_text.lower()] = canonical_key


@dataclass
class Parse:
    """A parse tree structure."""
    primitives: List[SemanticPrimitive]
    structure: str  # 'atomic', 'linear', 'scoped', 'binary'
    head: Optional[SemanticPrimitive] = None
    body: Optional['Parse'] = None
    operator: Optional[SemanticPrimitive] = None
    left: Optional['Parse'] = None
    right: Optional['Parse'] = None


# Example: Handle variation
if __name__ == '__main__':
    normalizer = SemanticNormalizer()
    
    # All these should normalize to the same semantic form
    variations = [
        "for all x in ℝ, x > 0",
        "for every x in ℝ, x > 0",
        "for each x in ℝ, x > 0",
        "given any x in ℝ, x > 0",
        "let x be an arbitrary element of ℝ, then x > 0",
        "take any x in ℝ, then x is positive",
        "if x is in ℝ, then x is greater than zero",
        "whenever x belongs to ℝ, x exceeds 0",
    ]
    
    print("Normalizing variations:")
    print("=" * 60)
    
    canonical_forms = []
    for text in variations:
        primitives = normalizer.normalize(text)
        canonical = ' '.join(p.canonical_form for p in primitives)
        canonical_forms.append(canonical)
        
        print(f"\nInput:  {text}")
        print(f"Canonical: {canonical}")
        print(f"Primitives: {[p.kind for p in primitives]}")
    
    # Check if all normalize to same form (modulo minor variations)
    print("\n" + "=" * 60)
    print("Analysis:")
    print(f"Unique canonical forms: {len(set(canonical_forms))}")
    
    # Should be 1 or 2 (slight variations acceptable)
    if len(set(canonical_forms)) <= 2:
        print("✓ Successfully normalized diverse phrasings!")
    else:
        print("✗ Need more normalization rules")
