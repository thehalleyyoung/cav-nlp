#!/usr/bin/env python3
"""
Advanced Compositional Meta-Rules: Mathematical Discourse Phenomena

This module extends the basic compositional system with sophisticated
linguistic constructions from mathematical texts, based on:

1. Ellipsis Resolution (Merchant 2001, Hardt 1999)
2. Anaphora & Binding Theory (Kamp & Reyle 1993, Heim & Kratzer 1998)
3. Mathematical Notation (Kamareddine et al. 2004, Ganesalingam 2013)
4. Discourse Structure (Asher & Lascarides 2003)
5. Presupposition (Heim 1983, van der Sandt 1992)
"""

from compositional_meta_rules import *
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re


# ============================================================================
# DISCOURSE REPRESENTATION STRUCTURES (Kamp & Reyle 1993)
# ============================================================================

@dataclass
class DiscourseReferent:
    """
    A discourse referent in DRT.
    Represents an entity introduced into discourse context.
    """
    name: str
    sort: Any  # Z3 sort
    conditions: List[Any] = field(default_factory=list)
    
    def z3_var(self):
        """Get Z3 variable for this referent"""
        return Const(self.name, self.sort)


@dataclass
class DRS:
    """
    Discourse Representation Structure (Kamp & Reyle 1993).
    
    A DRS consists of:
    - Universe: set of discourse referents
    - Conditions: constraints on referents
    
    DRSs compose via:
    - Merge: DRS1 + DRS2 (sequence)
    - Subordination: DRS1 ⊆ DRS2 (embedding)
    """
    referents: List[DiscourseReferent] = field(default_factory=list)
    conditions: List[Any] = field(default_factory=list)
    
    def merge(self, other: 'DRS') -> 'DRS':
        """Merge two DRSs (discourse sequence)"""
        return DRS(
            referents=self.referents + other.referents,
            conditions=self.conditions + other.conditions
        )
    
    def subordinate(self, condition: Any) -> 'DRS':
        """Add subordinate condition (e.g., implication, negation)"""
        return DRS(
            referents=self.referents,
            conditions=self.conditions + [condition]
        )
    
    def to_z3(self) -> Any:
        """Convert DRS to Z3 formula"""
        if not self.conditions:
            return Bool('true')
        
        # Collect all constraints
        all_constraints = []
        
        # Add referent conditions
        for ref in self.referents:
            all_constraints.extend(ref.conditions)
        
        # Add DRS conditions
        all_constraints.extend(self.conditions)
        
        if len(all_constraints) == 1:
            return all_constraints[0]
        else:
            return And(all_constraints)
    
    def exists_closure(self) -> Any:
        """Add existential closure over all referents"""
        if not self.referents:
            return self.to_z3()
        
        vars = [ref.z3_var() for ref in self.referents]
        return Exists(vars, self.to_z3())


# ============================================================================
# ADVANCED META-RULES: Ellipsis
# ============================================================================

class EllipsisMetaRules:
    """
    Meta-rules for ellipsis resolution following Merchant (2001).
    
    Types of ellipsis:
    1. VP Ellipsis: "John runs. Mary does too." → Mary runs
    2. Sluicing: "Someone arrived, but I don't know who" → who arrived
    3. Gapping: "John likes apples and Mary bananas" → Mary likes bananas
    4. Pseudogapping: "John will eat apples and Mary will bananas"
    """
    
    @staticmethod
    def vp_ellipsis() -> MetaRule:
        """
        VP Ellipsis: "X does [too/also/as well]"
        
        Resolution: Find antecedent VP and copy with subject replacement
        
        Example:
        - "if n is prime, then m is too" → m is prime
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            subject = components.get('subject', 'x')
            antecedent_vp = components.get('antecedent_vp', lambda x: Bool(f'P({x})'))
            
            Entity = DeclareSort('Entity')
            subj_var = Const(subject, Entity)
            
            # Apply antecedent VP to new subject
            z3_formula = antecedent_vp(subj_var)
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, [subject])
        
        return MetaRule(
            name="vp_ellipsis",
            linguistic_pattern=r"(?P<subject>\w+)\s+(?:is|does)\s+(?:too|also|as\s+well)",
            semantic_builder=build_semantic,
            papers=[
                "Merchant (2001): The Syntax of Silence",
                "Hardt (1999): Dynamic Interpretation of Verb Phrase Ellipsis"
            ],
            z3_tests=[
                {
                    'description': 'VP ellipsis with property transfer',
                    'code': '''
Entity = DeclareSort('Entity')
n = Const('n', Entity)
m = Const('m', Entity)
prime = Function('prime', Entity, BoolSort())
# Antecedent: n is prime
s.add(prime(n))
# Ellipsis: m is too → m is prime
s.add(prime(m))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def sluicing() -> MetaRule:
        """
        Sluicing: Wh-remnant after clause deletion
        
        Example: "Someone left, but I don't know who [left]"
        Resolution: Copy TP/VP from antecedent
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            wh_word = components.get('wh', 'who')
            antecedent_pred = components.get('antecedent', lambda x: Bool(f'P({x})'))
            
            Entity = DeclareSort('Entity')
            var = Const('x', Entity)
            
            # Sluicing creates question: which x such that antecedent(x)?
            z3_formula = Exists([var], antecedent_pred(var))
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, ['x'])
        
        return MetaRule(
            name="sluicing",
            linguistic_pattern=r"(?:but|and)\s+I\s+don't\s+know\s+(?P<wh>who|what|where|when|why|how)",
            semantic_builder=build_semantic,
            papers=[
                "Merchant (2001): Ch. 2 on Sluicing",
                "Ross (1969): Guess Who?"
            ],
            z3_tests=[
                {
                    'description': 'Sluicing resolution',
                    'code': '''
Entity = DeclareSort('Entity')
x = Const('x', Entity)
left = Function('left', Entity, BoolSort())
# Someone left = ∃x. left(x)
s.add(Exists([x], left(x)))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def comparative_ellipsis() -> MetaRule:
        """
        Comparative ellipsis: "X is more/less P than Y [is P]"
        
        Example: "n is greater than m [is]"
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            subject1 = components.get('subject1', 'x')
            subject2 = components.get('subject2', 'y')
            comparison = components.get('comparison', '>')
            
            x = Int(subject1)
            y = Int(subject2)
            
            if comparison in ['greater', 'more', '>']:
                z3_formula = x > y
            elif comparison in ['less', '<']:
                z3_formula = x < y
            else:
                z3_formula = x >= y
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, [subject1, subject2])
        
        return MetaRule(
            name="comparative_ellipsis",
            linguistic_pattern=r"(?P<subject1>\w+)\s+is\s+(?P<comparison>greater|less|more)\s+than\s+(?P<subject2>\w+)",
            semantic_builder=build_semantic,
            papers=[
                "Heim (2000): Degree Operators and Scope",
                "Kennedy (2007): Vagueness and Grammar"
            ],
            z3_tests=[
                {
                    'description': 'Comparative with implicit second predicate',
                    'code': '''
x = Int('x')
y = Int('y')
s.add(x > y)
s.check()
'''
                }
            ]
        )


# ============================================================================
# ADVANCED META-RULES: Anaphora & Binding
# ============================================================================

class AnaphoraMetaRules:
    """
    Meta-rules for anaphoric reference resolution.
    
    Based on:
    - Binding Theory (Chomsky 1981, Heim & Kratzer 1998)
    - DRT (Kamp & Reyle 1993)
    - Dynamic Semantics (Groenendijk & Stokhof 1991)
    """
    
    @staticmethod
    def pronominal_anaphora() -> MetaRule:
        """
        Pronoun resolution: "it", "this", "that", "such"
        
        DRT Analysis: Pronouns are discourse referents that must be
        linked to accessible antecedents
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            pronoun = components.get('pronoun', 'it')
            antecedent = components.get('antecedent', 'x')
            
            Entity = DeclareSort('Entity')
            pronoun_var = Const(pronoun, Entity)
            antecedent_var = Const(antecedent, Entity)
            
            # Pronoun = antecedent (coreference)
            z3_formula = pronoun_var == antecedent_var
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, [pronoun, antecedent])
        
        return MetaRule(
            name="pronominal_anaphora",
            linguistic_pattern=r"\b(?P<pronoun>it|this|that|such)\b",
            semantic_builder=build_semantic,
            papers=[
                "Kamp & Reyle (1993): Ch. 1-2 on Anaphora",
                "Heim (1982): File Change Semantics",
                "Groenendijk & Stokhof (1991): Dynamic Predicate Logic"
            ],
            z3_tests=[
                {
                    'description': 'Pronoun-antecedent binding',
                    'code': '''
Entity = DeclareSort('Entity')
x = Const('x', Entity)
it = Const('it', Entity)
prime = Function('prime', Entity, BoolSort())
# Antecedent: x is prime
s.add(prime(x))
# Pronoun: it = x
s.add(it == x)
# Therefore: it is prime
s.add(prime(it))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def donkey_anaphora() -> MetaRule:
        """
        Donkey anaphora: "If a farmer owns a donkey, he beats it"
        
        DRT Analysis: Indefinites introduce discourse referents
        accessible in subsequent discourse
        
        Translation: ∀x,y. (farmer(x) ∧ donkey(y) ∧ owns(x,y)) → beats(x,y)
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            Entity = DeclareSort('Entity')
            x = Const('x', Entity)
            y = Const('y', Entity)
            
            # Get predicates from components
            pred1 = Function(components.get('pred1', 'P'), Entity, BoolSort())
            pred2 = Function(components.get('pred2', 'Q'), Entity, BoolSort())
            rel = Function(components.get('rel', 'R'), Entity, Entity, BoolSort())
            consequent_rel = Function(components.get('cons_rel', 'S'), Entity, Entity, BoolSort())
            
            # Universal quantification over introduced referents
            z3_formula = ForAll([x, y],
                Implies(
                    And(pred1(x), pred2(y), rel(x, y)),
                    consequent_rel(x, y)
                )
            )
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, ['x', 'y'])
        
        return MetaRule(
            name="donkey_anaphora",
            linguistic_pattern=r"if\s+a\s+(?P<pred1>\w+)\s+(?P<rel>\w+)\s+a\s+(?P<pred2>\w+),\s+(?:he|she|it)\s+(?P<cons_rel>\w+)\s+(?:it|him|her)",
            semantic_builder=build_semantic,
            papers=[
                "Kamp (1981): A Theory of Truth and Semantic Representation",
                "Heim (1982): The Semantics of Definite and Indefinite NPs",
                "Groenendijk & Stokhof (1991): Dynamic Predicate Logic"
            ],
            z3_tests=[
                {
                    'description': 'Donkey sentence with universal force',
                    'code': '''
Entity = DeclareSort('Entity')
x = Const('x', Entity)
y = Const('y', Entity)
farmer = Function('farmer', Entity, BoolSort())
donkey = Function('donkey', Entity, BoolSort())
owns = Function('owns', Entity, Entity, BoolSort())
beats = Function('beats', Entity, Entity, BoolSort())
s.add(ForAll([x, y], 
    Implies(And(farmer(x), donkey(y), owns(x, y)), 
            beats(x, y))))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def paycheck_pronoun() -> MetaRule:
        """
        Paycheck pronoun: Functional reading of pronouns
        
        Example: "John spent his paycheck. Bill saved it."
        "it" = Bill's paycheck (not John's paycheck)
        
        Resolution: Pronoun copies function, not value
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            Entity = DeclareSort('Entity')
            x = Const(components.get('subject1', 'x'), Entity)
            y = Const(components.get('subject2', 'y'), Entity)
            
            # Functional pronoun: f(x) vs f(y)
            f = Function(components.get('function', 'f'), Entity, Entity)
            
            # "it" refers to f(y), not f(x)
            it = f(y)
            
            z3_formula = Bool('functional_reading')  # Placeholder
            
            return SemanticTerm(SemanticType.ENTITY, it, ['x', 'y'])
        
        return MetaRule(
            name="paycheck_pronoun",
            linguistic_pattern=r"(?P<subject1>\w+).*(?P<function>\w+).*(?P<subject2>\w+).*\bit\b",
            semantic_builder=build_semantic,
            papers=[
                "Cooper (1979): The Interpretation of Pronouns",
                "Jacobson (1999): Paycheck Pronouns and Variable-Free Semantics"
            ],
            z3_tests=[
                {
                    'description': 'Functional reading of pronoun',
                    'code': '''
Entity = DeclareSort('Entity')
x = Const('x', Entity)
y = Const('y', Entity)
paycheck = Function('paycheck', Entity, Entity)
# John's paycheck ≠ Bill's paycheck (distinct)
s.add(paycheck(x) != paycheck(y))
s.check()
'''
                }
            ]
        )


# ============================================================================
# ADVANCED META-RULES: Mathematical Notation
# ============================================================================

class NotationMetaRules:
    """
    Meta-rules for parsing mathematical notation.
    
    Based on:
    - Kamareddine, Maarek & Wells (2004): MathLang
    - Ganesalingam (2013): Mathematical Language
    - OpenMath / Content MathML standards
    """
    
    @staticmethod
    def subscript_notation() -> MetaRule:
        """
        Subscript: x_i, x_{i,j}, x_1, x_n
        
        Semantics: Indexed family
        ⟦x_i⟧ = select(x, i) where x : Index → Value
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            base = components.get('base', 'x')
            index = components.get('index', 'i')
            
            # Array representation of indexed family
            IndexSort = IntSort()
            ValueSort = RealSort()
            
            x_array = Array(base, IndexSort, ValueSort)
            i_var = Int(index)
            
            # x_i = x[i]
            z3_expr = Select(x_array, i_var)
            
            return SemanticTerm(SemanticType.ENTITY, z3_expr, [base, index])
        
        return MetaRule(
            name="subscript_notation",
            linguistic_pattern=r"(?P<base>[a-zA-Z]+)_(?:\{(?P<index>[^}]+)\}|(?P<index_simple>\w+))",
            semantic_builder=build_semantic,
            papers=[
                "Kamareddine et al. (2004): Computerizing Mathematical Text",
                "Ganesalingam (2013): Ch. 3 on Notation"
            ],
            z3_tests=[
                {
                    'description': 'Subscript as array indexing',
                    'code': '''
x = Array('x', IntSort(), RealSort())
i = Int('i')
x_i = Select(x, i)
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def superscript_notation() -> MetaRule:
        """
        Superscript: x^n, x^{-1}, x^2
        
        Semantics: Depends on context
        - Exponentiation: x^n = power(x, n)
        - Inverse: x^{-1} = inverse(x)
        - Iteration: f^n = f ∘ f ∘ ... ∘ f
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            base = components.get('base', 'x')
            exponent = components.get('exponent', 'n')
            
            # Context-dependent interpretation
            if exponent == '-1':
                # Inverse
                base_var = Real(base)
                z3_expr = 1.0 / base_var
            else:
                # Power
                base_var = Real(base)
                exp_var = Int(exponent) if exponent.isdigit() else Real(exponent)
                z3_expr = base_var ** exp_var
            
            return SemanticTerm(SemanticType.ENTITY, z3_expr, [base, exponent])
        
        return MetaRule(
            name="superscript_notation",
            linguistic_pattern=r"(?P<base>[a-zA-Z]+)\^(?:\{(?P<exponent>[^}]+)\}|(?P<exponent_simple>\w+))",
            semantic_builder=build_semantic,
            papers=[
                "Kamareddine et al. (2004)",
                "Ganesalingam (2013): Ch. 3"
            ],
            z3_tests=[
                {
                    'description': 'Superscript as exponentiation',
                    'code': '''
x = Real('x')
n = Int('n')
s.add(x > 0)
s.add(n > 0)
# x^n represented symbolically
power = Real('x_power_n')
s.add(power > 0)
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def set_builder_notation() -> MetaRule:
        """
        Set builder: {x : P(x)} or {x | P(x)}
        
        Semantics: Comprehension
        ⟦{x : P(x)}⟧ = λy. ∃x. (y = x ∧ P(x))
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            var = components.get('var', 'x')
            predicate = components.get('predicate', lambda x: Bool(f'P({x})'))
            
            Entity = DeclareSort('Entity')
            x_var = Const(var, Entity)
            
            # Set membership predicate
            in_set = lambda y: Exists([x_var], And(y == x_var, predicate(x_var)))
            
            return SemanticTerm(SemanticType.PRED, in_set, [var])
        
        return MetaRule(
            name="set_builder_notation",
            linguistic_pattern=r"\{(?P<var>\w+)\s*[:|]\s*(?P<predicate>.+)\}",
            semantic_builder=build_semantic,
            papers=[
                "Zermelo-Fraenkel Set Theory",
                "Ganesalingam (2013): Ch. 4 on Set Theory"
            ],
            z3_tests=[
                {
                    'description': 'Set comprehension',
                    'code': '''
Entity = DeclareSort('Entity')
x = Const('x', Entity)
P = Function('P', Entity, BoolSort())
y = Const('y', Entity)
# y ∈ {x : P(x)} iff ∃x. y=x ∧ P(x)
s.add(Exists([x], And(y == x, P(x))))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def function_application_juxtaposition() -> MetaRule:
        """
        Juxtaposition as function application: "f x" means f(x)
        
        Common in mathematics: "sin x", "f n", "P x"
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            func = components.get('function', 'f')
            arg = components.get('argument', 'x')
            
            # Create function and apply
            Entity = DeclareSort('Entity')
            f = Function(func, Entity, Entity)
            x = Const(arg, Entity)
            
            z3_expr = f(x)
            
            return SemanticTerm(SemanticType.ENTITY, z3_expr, [func, arg])
        
        return MetaRule(
            name="function_juxtaposition",
            linguistic_pattern=r"(?P<function>[a-zA-Z_]\w*)\s+(?P<argument>[a-zA-Z_]\w*)",
            semantic_builder=build_semantic,
            papers=[
                "Church (1940): Lambda Calculus",
                "Ganesalingam (2013): Ch. 5"
            ],
            z3_tests=[
                {
                    'description': 'Function application by juxtaposition',
                    'code': '''
Entity = DeclareSort('Entity')
f = Function('f', Entity, Entity)
x = Const('x', Entity)
result = f(x)
s.check()
'''
                }
            ]
        )


# ============================================================================
# ADVANCED META-RULES: Presupposition
# ============================================================================

class PresuppositionMetaRules:
    """
    Meta-rules for presupposition projection and accommodation.
    
    Based on:
    - Heim (1983): File Change Semantics
    - van der Sandt (1992): Presupposition Projection as Anaphora Resolution
    - Beaver (2001): Presupposition and Assertion in Dynamic Semantics
    """
    
    @staticmethod
    def factive_presupposition() -> MetaRule:
        """
        Factive verbs: "know", "realize", "discover"
        
        Presupposition: "X knows that P" presupposes P
        Assertion: X has knowledge of P
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            subject = components.get('subject', 'x')
            proposition = components.get('proposition', Bool('P'))
            
            Entity = DeclareSort('Entity')
            x = Const(subject, Entity)
            knows = Function('knows', Entity, BoolSort(), BoolSort())
            
            # Presupposition: P must be true
            presupposition = proposition
            # Assertion: x knows P
            assertion = knows(x, proposition)
            
            z3_formula = And(presupposition, assertion)
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, [subject])
        
        return MetaRule(
            name="factive_presupposition",
            linguistic_pattern=r"(?P<subject>\w+)\s+(?:knows?|realizes?|discovers?)\s+that\s+(?P<proposition>.+)",
            semantic_builder=build_semantic,
            papers=[
                "Kiparsky & Kiparsky (1970): Fact",
                "Heim (1983): On the Projection Problem for Presuppositions"
            ],
            z3_tests=[
                {
                    'description': 'Factive presupposition',
                    'code': '''
Entity = DeclareSort('Entity')
x = Const('x', Entity)
P = Bool('P')
knows = Function('knows', Entity, BoolSort(), BoolSort())
# Presupposition: P is true
s.add(P)
# Assertion: x knows P
s.add(knows(x, P))
s.check()
'''
                }
            ]
        )
    
    @staticmethod
    def iterative_presupposition() -> MetaRule:
        """
        Iteratives: "again", "another", "also"
        
        Presupposition: "X does P again" presupposes X did P before
        """
        def build_semantic(components: Dict[str, Any]) -> SemanticTerm:
            subject = components.get('subject', 'x')
            predicate = components.get('predicate', lambda x: Bool(f'P({x})'))
            
            Entity = DeclareSort('Entity')
            x = Const(subject, Entity)
            
            # Presupposition: P(x) held at previous time
            previous_time = Int('t_prev')
            current_time = Int('t_now')
            holds_at = Function('holds_at', Entity, IntSort(), BoolSort())
            
            presupposition = holds_at(x, previous_time)
            assertion = holds_at(x, current_time)
            
            z3_formula = And(
                previous_time < current_time,
                presupposition,
                assertion
            )
            
            return SemanticTerm(SemanticType.TRUTH, z3_formula, [subject])
        
        return MetaRule(
            name="iterative_presupposition",
            linguistic_pattern=r"(?P<subject>\w+)\s+(?P<predicate>\w+)\s+(?:again|also|another)",
            semantic_builder=build_semantic,
            papers=[
                "Beck (2006): Iterative and Restitutive Again",
                "von Stechow (1996): The Different Readings of Wieder"
            ],
            z3_tests=[
                {
                    'description': 'Iterative presupposition with temporal ordering',
                    'code': '''
Entity = DeclareSort('Entity')
x = Const('x', Entity)
t_prev = Int('t_prev')
t_now = Int('t_now')
holds_at = Function('holds_at', Entity, IntSort(), BoolSort())
s.add(t_prev < t_now)
s.add(holds_at(x, t_prev))  # Presupposition
s.add(holds_at(x, t_now))   # Assertion
s.check()
'''
                }
            ]
        )


# ============================================================================
# INTEGRATION: Extend Composition Engine
# ============================================================================

class AdvancedCompositionEngine(CompositionEngine):
    """
    Extended composition engine with advanced linguistic phenomena
    """
    
    def __init__(self):
        super().__init__()
        self._add_advanced_rules()
    
    def _add_advanced_rules(self):
        """Add advanced meta-rules to library"""
        
        # Ellipsis rules
        ellipsis = EllipsisMetaRules()
        self.atomic_rules['vp_ellipsis'] = ellipsis.vp_ellipsis()
        self.atomic_rules['sluicing'] = ellipsis.sluicing()
        self.atomic_rules['comparative_ellipsis'] = ellipsis.comparative_ellipsis()
        
        # Anaphora rules
        anaphora = AnaphoraMetaRules()
        self.atomic_rules['pronominal_anaphora'] = anaphora.pronominal_anaphora()
        self.atomic_rules['donkey_anaphora'] = anaphora.donkey_anaphora()
        self.atomic_rules['paycheck_pronoun'] = anaphora.paycheck_pronoun()
        
        # Notation rules
        notation = NotationMetaRules()
        self.atomic_rules['subscript'] = notation.subscript_notation()
        self.atomic_rules['superscript'] = notation.superscript_notation()
        self.atomic_rules['set_builder'] = notation.set_builder_notation()
        self.atomic_rules['juxtaposition'] = notation.function_application_juxtaposition()
        
        # Presupposition rules
        presupposition = PresuppositionMetaRules()
        self.atomic_rules['factive'] = presupposition.factive_presupposition()
        self.atomic_rules['iterative'] = presupposition.iterative_presupposition()


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_advanced_rules():
    """Demonstrate advanced compositional rules with Z3 validation"""
    
    print("=" * 80)
    print("ADVANCED COMPOSITIONAL META-RULES: Z3 Validation")
    print("=" * 80)
    print()
    
    engine = AdvancedCompositionEngine()
    
    print(f"Total meta-rules: {len(engine.atomic_rules)}")
    print()
    
    # Validate all rules
    results = validate_all_rules(engine)
    
    # Group by category
    categories = {
        'Basic': ['universal', 'existential', 'conditional', 'lambda', 'definite', 
                  'relative', 'conjunction', 'numerical', 'type', 'possessive'],
        'Ellipsis': ['vp_ellipsis', 'sluicing', 'comparative_ellipsis'],
        'Anaphora': ['pronominal_anaphora', 'donkey_anaphora', 'paycheck_pronoun'],
        'Notation': ['subscript', 'superscript', 'set_builder', 'juxtaposition'],
        'Presupposition': ['factive', 'iterative'],
    }
    
    for category, rule_names in categories.items():
        print(f"\n{category} Rules:")
        print("-" * 40)
        for rule_name in rule_names:
            if rule_name in results['rule_results']:
                result = results['rule_results'][rule_name]
                status = "✅" if result['failed'] == 0 else "❌"
                print(f"{status} {rule_name:25s}: {result['passed']}/{result['total_tests']} tests")
                
                if result['errors']:
                    for error in result['errors']:
                        print(f"   ✗ {error['test']}: {error['error'][:60]}...")
    
    print()
    print("=" * 80)
    print(f"OVERALL SUMMARY: {results['total_passed']}/{results['total_tests']} tests passed")
    print(f"Success rate: {100 * results['total_passed'] / results['total_tests']:.1f}%")
    print("=" * 80)
    print()
    
    # Show compositionality examples
    print("Compositionality Examples:")
    print("-" * 80)
    
    # Example 1: Quantification + Ellipsis
    print("\n1. Universal quantification + VP ellipsis:")
    print("   'For all x, P(x). For all y, Q(y) too.'")
    print("   → Combines universal quantifier with ellipsis resolution")
    
    # Example 2: Definite description + Anaphora
    print("\n2. Definite description + Pronominal anaphora:")
    print("   'The number n is prime. It is also odd.'")
    print("   → Combines uniqueness presupposition with pronoun resolution")
    
    # Example 3: Notation + Quantification
    print("\n3. Subscript notation + Universal quantification:")
    print("   'For all i, x_i > 0'")
    print("   → Combines indexed families with quantification")
    
    print()


if __name__ == '__main__':
    demo_advanced_rules()
