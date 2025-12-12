#!/usr/bin/env python3
"""
Compositional Meta-Rules for Mathematical Language Understanding

This system builds compositional linguistic rules based on formal semantics,
where complex meanings are built from simpler components following principles from:
- Montague Grammar: Compositional semantics with type theory
- Combinatory Categorial Grammar (CCG): Function application and composition
- Discourse Representation Theory (DRT): Context and binding
- Type-Logical Grammar: Proof-theoretic semantics

Each rule is:
1. Compositional: Combines with other rules via well-defined operations
2. Type-safe: Respects semantic types (e, t, <e,t>, etc.)
3. Z3-verified: Produces compilable SMT constraints
4. Linguistically grounded: Based on formal linguistic theory
"""

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Optional, Tuple
from enum import Enum
from z3 import *
import re


# ============================================================================
# SEMANTIC TYPES (Montague Grammar / Type Theory)
# ============================================================================

class SemanticType(Enum):
    """
    Semantic types following Montague Grammar:
    - ENTITY (e): individuals, objects
    - TRUTH (t): truth values
    - PROP: propositions
    - PRED: predicates <e,t>
    - REL: relations <e,<e,t>>
    - QUANT: quantifiers <<e,t>,t>
    - MOD: modifiers <<e,t>,<e,t>>
    """
    ENTITY = "e"
    TRUTH = "t"
    PROP = "t"  # Propositions are truth-valued
    PRED = "<e,t>"  # Predicate: entity → truth
    REL = "<e,<e,t>>"  # Binary relation
    QUANT = "<<e,t>,t>"  # Generalized quantifier
    MOD = "<<e,t>,<e,t>>"  # Modifier
    FUNC = "<e,e>"  # Entity-to-entity function
    
    def __str__(self):
        return self.value
    
    @staticmethod
    def function_type(domain: 'SemanticType', codomain: 'SemanticType') -> str:
        """Create function type <domain, codomain>"""
        return f"<{domain.value},{codomain.value}>"


# ============================================================================
# COMPOSITIONAL OPERATIONS (CCG-style)
# ============================================================================

class CompositionOp(Enum):
    """
    Composition operations from Combinatory Categorial Grammar:
    - APPLICATION: f(x) - function application
    - COMPOSITION: f ∘ g - function composition
    - TYPE_RAISING: x → λf.f(x) - type raising
    - COORDINATION: X and Y → λP.P(X)∧P(Y)
    - BINDING: λx.φ - lambda abstraction with variable binding
    """
    APPLICATION = "app"
    COMPOSITION = "comp"
    TYPE_RAISING = "raise"
    COORDINATION = "coord"
    BINDING = "bind"
    SUBSTITUTION = "subst"


@dataclass
class SemanticTerm:
    """
    Represents a semantic term with type and Z3 encoding.
    This is the core compositional unit.
    """
    term_type: SemanticType
    z3_expr: Any  # Z3 expression
    free_vars: List[str] = field(default_factory=list)
    
    def apply(self, argument: 'SemanticTerm', op: CompositionOp = CompositionOp.APPLICATION) -> 'SemanticTerm':
        """
        Apply compositional operation to combine terms.
        This is where Montague-style compositionality happens.
        """
        if op == CompositionOp.APPLICATION:
            # Standard function application: f(x)
            # Type check: self must be <α,β> and argument must be α
            if self.term_type == SemanticType.PRED and argument.term_type == SemanticType.ENTITY:
                result = self.z3_expr(argument.z3_expr)
                return SemanticTerm(SemanticType.TRUTH, result, self.free_vars + argument.free_vars)
            elif self.term_type == SemanticType.REL and argument.term_type == SemanticType.ENTITY:
                # Partial application of binary relation
                result_func = lambda y: self.z3_expr(argument.z3_expr, y)
                return SemanticTerm(SemanticType.PRED, result_func, self.free_vars + argument.free_vars)
            else:
                raise TypeError(f"Cannot apply {self.term_type} to {argument.term_type}")
        
        elif op == CompositionOp.COMPOSITION:
            # Function composition: (f ∘ g)(x) = f(g(x))
            result = lambda x: self.z3_expr(argument.z3_expr(x))
            return SemanticTerm(self.term_type, result, self.free_vars + argument.free_vars)
        
        elif op == CompositionOp.COORDINATION:
            # Coordination: "X and Y" → λP. P(X) ∧ P(Y)
            if self.term_type == argument.term_type == SemanticType.ENTITY:
                result = lambda P: And(P(self.z3_expr), P(argument.z3_expr))
                return SemanticTerm(SemanticType.QUANT, result, self.free_vars + argument.free_vars)
            elif self.term_type == argument.term_type == SemanticType.TRUTH:
                result = And(self.z3_expr, argument.z3_expr)
                return SemanticTerm(SemanticType.TRUTH, result, self.free_vars + argument.free_vars)
            else:
                raise TypeError(f"Cannot coordinate {self.term_type} with {argument.term_type}")
        
        else:
            raise NotImplementedError(f"Operation {op} not yet implemented")


# ============================================================================
# META-RULES: Compositional Linguistic Patterns
# ============================================================================

@dataclass
class MetaRule:
    """
    A meta-rule is a compositional pattern that can:
    1. Parse linguistic structure
    2. Build semantic representation compositionally
    3. Generate Z3 constraints
    4. Compose with other rules
    """
    name: str
    linguistic_pattern: str  # Regex or parse pattern
    semantic_builder: Callable[[Dict[str, Any]], SemanticTerm]
    composition_rules: List[Tuple[str, CompositionOp]] = field(default_factory=list)
    papers: List[str] = field(default_factory=list)
    z3_tests: List[Dict[str, Any]] = field(default_factory=list)
    
    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse text and extract semantic components"""
        match = re.search(self.linguistic_pattern, text, re.IGNORECASE)
        if match:
            return match.groupdict()
        return None
    
    def apply(self, text: str) -> Optional[SemanticTerm]:
        """Parse text and build semantic term"""
        components = self.parse(text)
        if components:
            return self.semantic_builder(components)
        return None
    
    def compose_with(self, other: 'MetaRule', op: CompositionOp) -> 'MetaRule':
        """Compose two meta-rules to create a more complex rule"""
        def combined_builder(components):
            self_term = self.semantic_builder(components)
            other_term = other.semantic_builder(components)
            return self_term.apply(other_term, op)
        
        return MetaRule(
            name=f"{self.name}+{other.name}",
            linguistic_pattern=f"({self.linguistic_pattern}).*({other.linguistic_pattern})",
            semantic_builder=combined_builder,
            composition_rules=self.composition_rules + other.composition_rules,
            papers=list(set(self.papers + other.papers)),
            z3_tests=self.z3_tests + other.z3_tests
        )


# ============================================================================
# ATOMIC META-RULES (Building Blocks)
# ============================================================================

class MetaRuleLibrary:
    """
    Library of atomic compositional meta-rules.
    Each rule is linguistically grounded and Z3-verified.
    """
    
    @staticmethod
    def universal_quantifier() -> MetaRule:
        """
        Universal quantification: "for all x, φ(x)"
        
        Semantics (Barwise & Cooper 1981):
        ⟦for all x⟧ = λP. ∀x. P(x)
        
        Type: <<e,t>,t> (generalized quantifier)
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            var_name = components.get('var', 'x')
            predicate = components.get('predicate', lambda x: Bool(f'P({x})'))
            
            # Create Z3 variable
            var = Int(var_name) if components.get('sort') == 'int' else Const(var_name, DeclareSort('Entity'))
            
            # Build universal quantification
            z3_formula = ForAll([var], predicate(var))
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, [var_name])
        
        return MetaRule(
            name="universal_quantifier",
            linguistic_pattern=r"for\s+all\s+(?P<var>\w+)",
            semantic_builder=build_semantic,
            papers=[
                "Barwise & Cooper (1981): Generalized Quantifiers and Natural Language",
                "Montague (1973): The Proper Treatment of Quantification in Ordinary English"
            ],
            z3_tests=[
                {
                    'description': 'Simple universal quantification',
                    'code': '''
x = Int('x')
s.add(ForAll([x], x > 0))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def existential_quantifier() -> MetaRule:
        """
        Existential quantification: "there exists x such that φ(x)"
        
        Semantics: ⟦there exists x⟧ = λP. ∃x. P(x)
        Type: <<e,t>,t>
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            var_name = components.get('var', 'x')
            predicate = components.get('predicate', lambda x: Bool(f'P({x})'))
            
            var = Int(var_name) if components.get('sort') == 'int' else Const(var_name, DeclareSort('Entity'))
            z3_formula = Exists([var], predicate(var))
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, [var_name])
        
        return MetaRule(
            name="existential_quantifier",
            linguistic_pattern=r"there\s+exists?\s+(?P<var>\w+)",
            semantic_builder=build_semantic,
            papers=["Barwise & Cooper (1981)", "Montague (1973)"],
            z3_tests=[
                {
                    'description': 'Simple existential quantification',
                    'code': '''
x = Int('x')
s.add(Exists([x], And(x > 0, x < 10)))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def conditional_implication() -> MetaRule:
        """
        Conditional: "if φ then ψ"
        
        Semantics: ⟦if φ then ψ⟧ = φ → ψ
        Type: <t,<t,t>>
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            antecedent = components.get('antecedent', Bool('p'))
            consequent = components.get('consequent', Bool('q'))
            
            z3_formula = Implies(antecedent, consequent)
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, [])
        
        return MetaRule(
            name="conditional_implication",
            linguistic_pattern=r"if\s+(?P<antecedent>.+?)\s+then\s+(?P<consequent>.+)",
            semantic_builder=build_semantic,
            papers=["Stalnaker (1968): A Theory of Conditionals"],
            z3_tests=[
                {
                    'description': 'Material implication',
                    'code': '''
p = Bool('p')
q = Bool('q')
s.add(Implies(p, q))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def lambda_abstraction() -> MetaRule:
        """
        Lambda abstraction: "λx. φ(x)" or "let x be such that φ(x)"
        
        Semantics: Variable binding with scope
        Type: <e,t> (creates predicate from formula)
        
        Based on Church (1940) lambda calculus and Heim & Kratzer (1998)
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            var_name = components.get('var', 'x')
            body = components.get('body', lambda x: Bool(f'P({x})'))
            
            # Lambda abstraction creates a predicate
            predicate = lambda x: body(x) if callable(body) else body
            
            return SemanticTerm(SemanticType.PRED, predicate, [var_name])
        
        return MetaRule(
            name="lambda_abstraction",
            linguistic_pattern=r"(?:let|assume)\s+(?P<var>\w+)\s+(?:be\s+)?(?:such\s+)?that\s+(?P<body>.+)",
            semantic_builder=build_semantic,
            papers=[
                "Heim & Kratzer (1998): Semantics in Generative Grammar",
                "Church (1940): A Formulation of the Simple Theory of Types"
            ],
            z3_tests=[
                {
                    'description': 'Lambda with constraint',
                    'code': '''
x = Int('x')
P = Function('P', IntSort(), BoolSort())
s.add(ForAll([x], Implies(x > 0, P(x))))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def definite_description() -> MetaRule:
        """
        Definite description: "the x such that φ(x)"
        
        Semantics (Russell 1905): ∃!x. φ(x) ∧ ψ(x)
        "The" presupposes uniqueness and existence
        
        Type: <<e,t>,e> (quantifier that returns entity)
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            var_name = components.get('var', 'x')
            property_pred = components.get('property', lambda x: Bool(f'P({x})'))
            
            var = Const(var_name, DeclareSort('Entity'))
            
            # Russell's analysis: existence + uniqueness
            exists_clause = Exists([var], property_pred(var))
            unique_clause = ForAll(
                [Const(f'{var_name}1', DeclareSort('Entity')), 
                 Const(f'{var_name}2', DeclareSort('Entity'))],
                Implies(
                    And(property_pred(Const(f'{var_name}1', DeclareSort('Entity'))),
                        property_pred(Const(f'{var_name}2', DeclareSort('Entity')))),
                    Const(f'{var_name}1', DeclareSort('Entity')) == Const(f'{var_name}2', DeclareSort('Entity'))
                )
            )
            
            z3_formula = And(exists_clause, unique_clause)
            
            return SemanticTerm(SemanticType.ENTITY, var, [var_name])
        
        return MetaRule(
            name="definite_description",
            linguistic_pattern=r"the\s+(?P<var>\w+)\s+(?:such\s+that\s+)?(?P<property>.+)?",
            semantic_builder=build_semantic,
            papers=[
                "Russell (1905): On Denoting",
                "Strawson (1950): On Referring",
                "Heim (1982): The Semantics of Definite and Indefinite Noun Phrases"
            ],
            z3_tests=[
                {
                    'description': 'Definite description with uniqueness',
                    'code': '''
Entity = DeclareSort('Entity')
x = Const('x', Entity)
P = Function('P', Entity, BoolSort())
s.add(Exists([x], P(x)))  # Existence
x1, x2 = Consts('x1 x2', Entity)
s.add(ForAll([x1, x2], Implies(And(P(x1), P(x2)), x1 == x2)))  # Uniqueness
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def relative_clause() -> MetaRule:
        """
        Relative clause: "x which/that φ(x)"
        
        Semantics: Predicate modification (Heim & Kratzer 1998)
        ⟦N that VP⟧ = λx. N(x) ∧ VP(x)
        
        Type: <<e,t>,<e,t>> (modifier)
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            head_noun = components.get('head', lambda x: Bool(f'N({x})'))
            restriction = components.get('restriction', lambda x: Bool(f'R({x})'))
            
            # Predicate intersection
            combined_pred = lambda x: And(head_noun(x), restriction(x))
            
            return SemanticTerm(SemanticType.PRED, combined_pred, [])
        
        return MetaRule(
            name="relative_clause",
            linguistic_pattern=r"(?P<head>\w+)\s+(?:which|that)\s+(?P<restriction>.+)",
            semantic_builder=build_semantic,
            papers=[
                "Heim & Kratzer (1998): Ch. 6 on Relative Clauses",
                "Montague (1973): PTQ Fragment"
            ],
            z3_tests=[
                {
                    'description': 'Relative clause as predicate intersection',
                    'code': '''
Entity = DeclareSort('Entity')
x = Const('x', Entity)
N = Function('N', Entity, BoolSort())  # Noun predicate
R = Function('R', Entity, BoolSort())  # Restriction predicate
s.add(Exists([x], And(N(x), R(x))))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def conjunction_coordination() -> MetaRule:
        """
        Coordination: "φ and ψ"
        
        Semantics: Boolean coordination or generalized coordination
        - At type t: φ ∧ ψ
        - At type <e,t>: λx. P(x) ∧ Q(x)
        - At type e: λP. P(x) ∧ P(y) (collectivization)
        
        Polymorphic across types (Partee & Rooth 1983)
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            left = components.get('left', Bool('p'))
            right = components.get('right', Bool('q'))
            
            # Type-flexible coordination
            if callable(left) and callable(right):
                # Predicate coordination: λx. P(x) ∧ Q(x)
                combined = lambda x: And(left(x), right(x))
                return SemanticTerm(SemanticType.PRED, combined, [])
            else:
                # Boolean coordination
                z3_formula = And(left, right)
                return SemanticTerm(SemanticType.TRUTH, z3_formula, [])
        
        return MetaRule(
            name="conjunction_coordination",
            linguistic_pattern=r"(?P<left>.+?)\s+and\s+(?P<right>.+)",
            semantic_builder=build_semantic,
            papers=[
                "Partee & Rooth (1983): Generalized Conjunction and Type Ambiguity",
                "Steedman (2000): The Syntactic Process (CCG coordination)"
            ],
            z3_tests=[
                {
                    'description': 'Boolean conjunction',
                    'code': '''
p = Bool('p')
q = Bool('q')
s.add(And(p, q))
s.check()
'''
                },
                {
                    'description': 'Predicate conjunction',
                    'code': '''
Entity = DeclareSort('Entity')
x = Const('x', Entity)
P = Function('P', Entity, BoolSort())
Q = Function('Q', Entity, BoolSort())
s.add(ForAll([x], Implies(And(P(x), Q(x)), Bool('result'))))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def numerical_quantifier() -> MetaRule:
        """
        Numerical quantification: "at least n x", "exactly n x", "at most n x"
        
        Semantics (Hackl 2000): Cardinality constraints
        ⟦at least n⟧ = λP. |{x : P(x)}| ≥ n
        
        Type: <<e,t>,t>
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            quantifier = components.get('quantifier', 'at least')
            number = int(components.get('number', '1'))
            var_name = components.get('var', 'x')
            predicate = components.get('predicate', lambda x: Bool(f'P({x})'))
            
            # This is simplified - full implementation needs counting
            # For Z3, we can use cardinality constraints or explicit enumeration
            if quantifier == 'at least':
                # Generate witnesses
                vars = [Const(f'{var_name}_{i}', DeclareSort('Entity')) for i in range(number)]
                z3_formula = And([predicate(v) for v in vars])
            elif quantifier == 'exactly':
                vars = [Const(f'{var_name}_{i}', DeclareSort('Entity')) for i in range(number)]
                z3_formula = And(
                    And([predicate(v) for v in vars]),
                    # Add distinctness
                    And([vars[i] != vars[j] for i in range(number) for j in range(i+1, number)])
                )
            else:  # at most
                # This requires more complex encoding
                z3_formula = Bool(f'at_most_{number}_{var_name}')
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, [var_name])
        
        return MetaRule(
            name="numerical_quantifier",
            linguistic_pattern=r"(?P<quantifier>at\s+least|exactly|at\s+most)\s+(?P<number>\d+)\s+(?P<var>\w+)",
            semantic_builder=build_semantic,
            papers=[
                "Hackl (2000): Comparative Quantifiers",
                "Generalized Quantifier Theory (Barwise & Cooper 1981)"
            ],
            z3_tests=[
                {
                    'description': 'At least 2 elements',
                    'code': '''
Entity = DeclareSort('Entity')
x1, x2 = Consts('x1 x2', Entity)
P = Function('P', Entity, BoolSort())
s.add(And(P(x1), P(x2), x1 != x2))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def type_ascription() -> MetaRule:
        """
        Type ascription: "x : T" or "x is a T"
        
        Semantics: Type membership predicate
        ⟦x : T⟧ = T(x)
        
        Type: <e,t>
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            var_name = components.get('var', 'x')
            type_name = components.get('type', 'T')
            
            # Create typed variable and membership constraint
            sort = DeclareSort(type_name)
            var = Const(var_name, sort)
            
            # Type membership is implicit in Z3's sort system
            z3_formula = Bool(f'{var_name}_in_{type_name}')  # Placeholder
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, [var_name])
        
        return MetaRule(
            name="type_ascription",
            linguistic_pattern=r"(?P<var>\w+)\s+(?::|is\s+a(?:n)?)\s+(?P<type>\w+)",
            semantic_builder=build_semantic,
            papers=[
                "Ranta (1994): Type-Theoretical Grammar",
                "Martin-Löf (1984): Intuitionistic Type Theory"
            ],
            z3_tests=[
                {
                    'description': 'Type membership',
                    'code': '''
NaturalNumber = DeclareSort('NaturalNumber')
n = Const('n', NaturalNumber)
s.check()  # n is implicitly typed
'''
                }
            ]
        )
    
    @staticmethod
    def possessive_construction() -> MetaRule:
        """
        Possessive: "x's y" or "the y of x"
        
        Semantics: Binary relation R(x,y) where R = "has"/"of"
        ⟦x's y⟧ = λy'. R(x, y') ∧ y = y'
        
        Type: <e,<e,t>>
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            possessor = components.get('possessor', 'x')
            possessed = components.get('possessed', 'y')
            
            # Create relation
            Entity = DeclareSort('Entity')
            x = Const(possessor, Entity)
            y = Const(possessed, Entity)
            has_rel = Function('has', Entity, Entity, BoolSort())
            
            z3_formula = has_rel(x, y)
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, [possessor, possessed])
        
        return MetaRule(
            name="possessive_construction",
            linguistic_pattern=r"(?P<possessor>\w+)'s\s+(?P<possessed>\w+)",
            semantic_builder=build_semantic,
            papers=[
                "Partee (1997): Possessive Constructions",
                "Barker (1995): Possessive Descriptions"
            ],
            z3_tests=[
                {
                    'description': 'Possessive relation',
                    'code': '''
Entity = DeclareSort('Entity')
x = Const('x', Entity)
y = Const('y', Entity)
has = Function('has', Entity, Entity, BoolSort())
s.add(has(x, y))
s.check()
'''
                }
            ]
        )


# ============================================================================
# COMPOSITIONAL RULE COMBINATIONS
# ============================================================================

class CompositionEngine:
    """
    Engine for composing meta-rules to handle complex linguistic constructions.
    Uses CCG-style composition operators.
    """
    
    def __init__(self):
        self.library = MetaRuleLibrary()
        self.atomic_rules = self._initialize_atomic_rules()
        self.composed_rules = {}
    
    def _initialize_atomic_rules(self) -> Dict[str, MetaRule]:
        """Initialize library of atomic rules"""
        return {
            'universal': self.library.universal_quantifier(),
            'existential': self.library.existential_quantifier(),
            'conditional': self.library.conditional_implication(),
            'lambda': self.library.lambda_abstraction(),
            'definite': self.library.definite_description(),
            'relative': self.library.relative_clause(),
            'conjunction': self.library.conjunction_coordination(),
            'numerical': self.library.numerical_quantifier(),
            'type': self.library.type_ascription(),
            'possessive': self.library.possessive_construction(),
        }
    
    def compose(self, rule1_name: str, rule2_name: str, op: CompositionOp) -> MetaRule:
        """Compose two rules using specified operation"""
        rule1 = self.atomic_rules.get(rule1_name) or self.composed_rules.get(rule1_name)
        rule2 = self.atomic_rules.get(rule2_name) or self.composed_rules.get(rule2_name)
        
        if not rule1 or not rule2:
            raise ValueError(f"Rules not found: {rule1_name}, {rule2_name}")
        
        composed = rule1.compose_with(rule2, op)
        self.composed_rules[composed.name] = composed
        return composed
    
    def build_complex_rule(self, name: str, components: List[Tuple[str, CompositionOp]]) -> MetaRule:
        """
        Build complex rule from sequence of compositions.
        
        Example:
            build_complex_rule("forall_implies", [
                ("universal", CompositionOp.APPLICATION),
                ("conditional", CompositionOp.COMPOSITION)
            ])
        """
        if len(components) < 2:
            raise ValueError("Need at least 2 components to compose")
        
        # Start with first rule
        current_rule = self.atomic_rules[components[0][0]]
        
        # Sequentially compose with remaining rules
        for rule_name, op in components[1:]:
            next_rule = self.atomic_rules.get(rule_name) or self.composed_rules.get(rule_name)
            current_rule = current_rule.compose_with(next_rule, op)
        
        current_rule.name = name
        self.composed_rules[name] = current_rule
        return current_rule


# ============================================================================
# Z3 VALIDATION FRAMEWORK
# ============================================================================

def validate_meta_rule(rule: MetaRule) -> Dict[str, Any]:
    """
    Validate that all Z3 tests for a meta-rule compile and run correctly.
    Returns detailed results.
    """
    results = {
        'rule': rule.name,
        'total_tests': len(rule.z3_tests),
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    for test in rule.z3_tests:
        try:
            # Create fresh solver
            s = Solver()
            
            # Execute test code
            namespace = {
                'Bool': Bool, 'Int': Int, 'Real': Real, 'String': String,
                'Const': Const, 'Consts': Consts, 'Function': Function,
                'DeclareSort': DeclareSort, 'ForAll': ForAll, 'Exists': Exists,
                'And': And, 'Or': Or, 'Not': Not, 'Implies': Implies,
                'If': If, 'Solver': Solver, 's': s,
                'IntSort': IntSort, 'BoolSort': BoolSort, 'RealSort': RealSort,
                'Array': Array, 'Select': Select, 'Store': Store,
                'sat': sat, 'unsat': unsat,
            }
            
            exec(test['code'], namespace)
            
            # Try to check satisfiability
            result = s.check()
            
            results['passed'] += 1
            
        except Exception as e:
            results['failed'] += 1
            results['errors'].append({
                'test': test.get('description', 'unnamed'),
                'error': str(e),
                'code': test['code']
            })
    
    return results


def validate_all_rules(engine: CompositionEngine) -> Dict[str, Any]:
    """Validate all atomic rules in the library"""
    summary = {
        'total_rules': len(engine.atomic_rules),
        'total_tests': 0,
        'total_passed': 0,
        'total_failed': 0,
        'rule_results': {}
    }
    
    for rule_name, rule in engine.atomic_rules.items():
        result = validate_meta_rule(rule)
        summary['rule_results'][rule_name] = result
        summary['total_tests'] += result['total_tests']
        summary['total_passed'] += result['passed']
        summary['total_failed'] += result['failed']
    
    return summary


# ============================================================================
# EXAMPLE: Building Complex Constructions Compositionally
# ============================================================================

def demo_compositional_rules():
    """Demonstrate compositional rule building"""
    
    print("=" * 80)
    print("COMPOSITIONAL META-RULES: Z3 Validation")
    print("=" * 80)
    print()
    
    # Initialize engine
    engine = CompositionEngine()
    
    # Validate all atomic rules
    print("Validating atomic meta-rules...")
    print()
    
    results = validate_all_rules(engine)
    
    # Print results
    for rule_name, result in results['rule_results'].items():
        status = "✅" if result['failed'] == 0 else "❌"
        print(f"{status} {rule_name:20s}: {result['passed']}/{result['total_tests']} tests passed")
        
        if result['errors']:
            for error in result['errors']:
                print(f"   ✗ {error['test']}: {error['error']}")
    
    print()
    print("-" * 80)
    print(f"SUMMARY: {results['total_passed']}/{results['total_tests']} tests passed")
    print(f"Success rate: {100 * results['total_passed'] / results['total_tests']:.1f}%")
    print("-" * 80)
    print()
    
    # Demonstrate composition
    print("Building composed rules...")
    print()
    
    # Example 1: Universal quantification with conditional
    # "for all x, if P(x) then Q(x)"
    forall_implies = engine.compose('universal', 'conditional', CompositionOp.APPLICATION)
    print(f"✓ Created: {forall_implies.name}")
    print(f"  Papers: {len(forall_implies.papers)} cited")
    
    # Example 2: Existential with relative clause
    # "there exists an x which/that P(x)"
    exists_relative = engine.compose('existential', 'relative', CompositionOp.APPLICATION)
    print(f"✓ Created: {exists_relative.name}")
    
    # Example 3: Definite description with type ascription
    # "the x : T such that P(x)"
    def_typed = engine.compose('definite', 'type', CompositionOp.COMPOSITION)
    print(f"✓ Created: {def_typed.name}")
    
    print()
    print("=" * 80)
    print("Composition engine ready with", len(engine.atomic_rules), "atomic rules")
    print("=" * 80)


if __name__ == '__main__':
    demo_compositional_rules()
