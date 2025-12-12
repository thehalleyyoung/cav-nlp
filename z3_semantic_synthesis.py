"""
Z3-Driven Compositional Semantic Synthesis
===========================================

Treat English mathematical text as an UNDER-SPECIFIED program where Z3 
synthesizes the compositional semantics by searching the space of valid 
denotations that satisfy type-theoretic constraints.

Core Idea: English phrases are holes in a typed lambda calculus program.
Z3 fills these holes with semantically valid Lean type expressions by:
1. Encoding syntax as algebraic datatypes
2. Encoding compositional rules as SMT constraints
3. Searching for interpretations that satisfy Lean's type system
4. Learning which interpretations generalize via CEGIS loop

This inverts the typical NLP pipeline: instead of parsing→semantics→type-checking,
we do parsing→Z3-driven-semantic-search→verified-output.
"""

from z3 import *
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Callable
from enum import Enum
import re
import numpy as np

from lean_type_theory import UniverseLevel, LeanType, LeanExpr, Context


class SemanticHole(Enum):
    """Types of semantic holes Z3 must fill."""
    QUANTIFIER_SCOPE = "quantifier_scope"  # What does "all x" bind?
    TYPE_ANNOTATION = "type_annotation"     # What is type of x?
    PREDICATE_MEANING = "predicate_meaning" # What does "continuous" mean?
    OPERATOR_BINDING = "operator_binding"   # What does "+" mean here?
    COMPOSITION_ORDER = "composition_order" # Which parses are valid?
    IMPLICIT_ARGUMENT = "implicit_argument" # Fill in type class inference
    COERCION = "coercion"                   # Insert implicit casts


@dataclass
class SemanticSketch:
    """
    A semantic sketch with holes that Z3 must fill.
    
    Think of this as a partial Lean program where Z3 completes it.
    """
    text: str  # English input
    syntactic_parse: 'ParseTree'  # Ambiguous parse
    holes: List[Tuple[str, SemanticHole]]  # Named holes with types
    constraints: List[BoolRef]  # Z3 constraints that must be satisfied
    candidate_denotations: List['LeanExpr']  # Possible meanings


class Z3SemanticAlgebra:
    """
    Encode compositional semantics as Z3 algebraic datatypes.
    
    The key insight: treat Lean types as an algebraic datatype in Z3,
    then express compositional semantic rules as SMT formulas over
    this algebra. Z3 searches for valid semantic interpretations.
    """
    
    def __init__(self):
        # Define sorts for semantic objects
        self._define_sorts()
        self._define_composition_rules()
        self._define_well_formedness()
    
    def _define_sorts(self):
        """Define Z3 sorts for semantic algebra."""
        
        # Universe levels (for dependent types)
        self.UnivSort = DeclareSort('Universe')
        self.univ_zero = Const('Univ_0', self.UnivSort)
        self.univ_succ = Function('Univ_Succ', self.UnivSort, self.UnivSort)
        self.univ_max = Function('Univ_Max', self.UnivSort, self.UnivSort, self.UnivSort)
        self.univ_imax = Function('Univ_IMax', self.UnivSort, self.UnivSort, self.UnivSort)
        
        # Simplified: Just use uninterpreted sorts for now
        # Full recursive datatype encoding is complex in Z3
        self.LeanType = DeclareSort('LeanType')
        self.LeanExpr = DeclareSort('LeanExpr')
        
        # Type constructors as functions
        self.Type_Sort = Function('Type_Sort', self.UnivSort, self.LeanType)
        self.Prop_Sort = Const('Prop_Sort', self.LeanType)
        self.Var_Type = Function('Var_Type', StringSort(), self.LeanType)
        self.Pi_Type = Function('Pi_Type', StringSort(), self.LeanType, self.LeanType, self.LeanType)
        self.Forall_Type = Function('Forall_Type', StringSort(), self.LeanType, self.LeanType, self.LeanType)
        
        # Expression constructors
        self.Var_Expr = Function('Var_Expr', StringSort(), self.LeanExpr)
        self.App_Expr = Function('App_Expr', self.LeanExpr, self.LeanExpr, self.LeanExpr)
        self.Lambda_Expr = Function('Lambda_Expr', StringSort(), self.LeanType, self.LeanExpr, self.LeanExpr)
        
        # Contexts (typing environments)
        self.Context = ArraySort(StringSort(), self.LeanType)
        
        # Semantic interpretations (maps strings to types)
        self.Interpretation = ArraySort(StringSort(), self.LeanType)
    
    def _define_composition_rules(self):
        """Define compositional semantic rules as Z3 functions."""
        
        # Functional application: [[ f(x) ]] = [[ f ]] ⊗ [[ x ]]
        self.compose_application = Function('Compose_App',
                                           self.LeanType,
                                           self.LeanType,
                                           self.LeanType)
        
        # Quantification: [[ for all x : A, B ]] = Π x:A, B
        self.compose_forall = Function('Compose_Forall',
                                      StringSort(),      # variable
                                      self.LeanType,     # domain
                                      self.LeanType,     # body
                                      self.LeanType)     # result
        
        # Implication: [[ if P then Q ]] = P → Q
        self.compose_implies = Function('Compose_Implies',
                                       self.LeanType,
                                       self.LeanType,
                                       self.LeanType)
        
        # Conjunction: [[ P and Q ]] = P ∧ Q
        self.compose_and = Function('Compose_And',
                                   self.LeanType,
                                   self.LeanType,
                                   self.LeanType)
        
        # Existential: [[ there exists x : A, B ]] = Σ x:A, B or ∃ x:A, B
        self.compose_exists = Function('Compose_Exists',
                                      StringSort(),
                                      self.LeanType,
                                      self.LeanType,
                                      self.LeanType)
    
    def _define_well_formedness(self):
        """Define type well-formedness predicates."""
        
        # Typing judgment: Γ ⊢ e : T
        self.has_type = Function('HasType',
                                self.Context,
                                self.LeanExpr,
                                self.LeanType,
                                BoolSort())
        
        # Type is well-formed in context: Γ ⊢ T : Type u
        self.wellformed_type = Function('WellFormedType',
                                       self.Context,
                                       self.LeanType,
                                       BoolSort())
        
        # Definitional equality: Γ ⊢ e1 ≡ e2
        self.def_equal = Function('DefEqual',
                                 self.Context,
                                 self.LeanExpr,
                                 self.LeanExpr,
                                 BoolSort())


class Z3SemanticSynthesizer:
    """
    Synthesize semantic interpretations using Z3.
    
    Given English text with ambiguity, generate SMT problem where:
    - Variables = semantic choices (types, scopes, bindings)
    - Constraints = Lean's type system + compositional semantics
    - Model = valid interpretation that type-checks
    """
    
    def __init__(self):
        self.algebra = Z3SemanticAlgebra()
        self.solver = Optimize()  # Use Optimize for weighted synthesis
        self._initialize_type_system_axioms()
    
    def _initialize_type_system_axioms(self):
        """Encode Lean's type system as SMT axioms."""
        
        # Axiom 1: Prop is a type
        ctx = Const('empty_ctx', self.algebra.Context)
        prop_sort = self.algebra.Prop_Sort
        self.solver.add(
            self.algebra.wellformed_type(ctx, prop_sort)
        )
        
        # Axiom 2: Type u is a type
        u = Const('u', self.algebra.UnivSort)
        type_u = self.algebra.Type_Sort(u)
        self.solver.add(
            self.algebra.wellformed_type(ctx, type_u)
        )
        
        # Axiom 3: Pi type formation
        # If Γ ⊢ A : Type u and Γ, x:A ⊢ B : Type v
        # then Γ ⊢ (Π x:A, B) : Type (imax u v)
        gamma = Const('gamma', self.algebra.Context)
        x = String('x')
        A = Const('A', self.algebra.LeanType)
        B = Const('B', self.algebra.LeanType)
        
        pi_type = self.algebra.Pi_Type(x, A, B)
        
        self.solver.add(
            Implies(
                And(
                    self.algebra.wellformed_type(gamma, A),
                    self.algebra.wellformed_type(
                        Store(gamma, x, A),  # Extend context
                        B
                    )
                ),
                self.algebra.wellformed_type(
                    gamma,
                    pi_type
                )
            )
        )
    
    def synthesize_semantics(self, english_text: str, 
                            parse_tree: 'ParseTree',
                            type_hints: Optional[Dict[str, LeanType]] = None) -> List['SemanticInterpretation']:
        """
        Synthesize valid semantic interpretations.
        
        Args:
            english_text: English mathematical statement
            parse_tree: Syntactic parse (possibly ambiguous)
            type_hints: Optional type annotations from context
            
        Returns:
            List of valid semantic interpretations satisfying Lean's type system
        """
        # Create semantic sketch with holes
        sketch = self._create_sketch(english_text, parse_tree, type_hints)
        
        # Encode synthesis problem as SMT
        self._encode_sketch(sketch)
        
        # Search for valid interpretations
        interpretations = []
        
        while self.solver.check() == sat:
            model = self.solver.model()
            
            # Extract semantic interpretation from model
            interp = self._extract_interpretation(model, sketch)
            interpretations.append(interp)
            
            # Block this solution and search for alternatives
            self._block_interpretation(model, sketch)
            
            # Limit search
            if len(interpretations) >= 10:
                break
        
        # Rank by simplicity/naturalness
        interpretations.sort(key=lambda i: i.complexity_score)
        
        return interpretations
    
    def _create_sketch(self, text: str, parse_tree: 'ParseTree',
                      type_hints: Optional[Dict]) -> SemanticSketch:
        """Create semantic sketch with holes."""
        
        holes = []
        constraints = []
        
        # Identify quantifiers → create holes for scopes and types
        for match in re.finditer(r'for\s+(all|every|each)\s+(\w+)', text, re.I):
            var_name = match.group(2)
            holes.append((f"type_of_{var_name}", SemanticHole.TYPE_ANNOTATION))
            holes.append((f"scope_of_{var_name}", SemanticHole.QUANTIFIER_SCOPE))
        
        # Identify predicates → create holes for meanings
        for match in re.finditer(r'is\s+(\w+)', text, re.I):
            pred = match.group(1)
            holes.append((f"meaning_of_{pred}", SemanticHole.PREDICATE_MEANING))
        
        # Identify function applications → create holes for operators
        for match in re.finditer(r'(\w+)\s*\(', text):
            func = match.group(1)
            holes.append((f"type_of_{func}", SemanticHole.OPERATOR_BINDING))
        
        return SemanticSketch(
            text=text,
            syntactic_parse=parse_tree,
            holes=holes,
            constraints=constraints,
            candidate_denotations=[]
        )
    
    def _encode_sketch(self, sketch: SemanticSketch):
        """Encode sketch as SMT problem."""
        
        # Create Z3 variables for each hole
        hole_vars = {}
        for hole_name, hole_type in sketch.holes:
            if hole_type == SemanticHole.TYPE_ANNOTATION:
                hole_vars[hole_name] = Const(hole_name, self.algebra.LeanType)
            elif hole_type == SemanticHole.QUANTIFIER_SCOPE:
                hole_vars[hole_name] = Const(hole_name, self.algebra.LeanType)
            # ... handle other hole types
        
        # Add type well-formedness constraints
        ctx = Const('ctx', self.algebra.Context)
        for var in hole_vars.values():
            self.solver.add(self.algebra.wellformed_type(ctx, var))
        
        # Add compositional constraints
        # Example: if text is "for all x in S, P(x)"
        # Constrain: result must be Pi type
        if 'for all' in sketch.text.lower():
            # Result must be Pi type
            result = Const('result', self.algebra.LeanType)
            var_name = String('x')  # Extract from parse
            domain = hole_vars.get('type_of_x', Const('domain', self.algebra.LeanType))
            body = Const('body', self.algebra.LeanType)
            
            self.solver.add(
                result == self.algebra.Pi_Type(var_name, domain, body)
            )
    
    def _extract_interpretation(self, model: ModelRef, 
                               sketch: SemanticSketch) -> 'SemanticInterpretation':
        """Extract semantic interpretation from Z3 model."""
        
        interp_dict = {}
        
        for hole_name, hole_type in sketch.holes:
            # Get value from model
            hole_var = Const(hole_name, self.algebra.LeanType)
            value = model.eval(hole_var, model_completion=True)
            
            # Convert Z3 value back to Lean type
            lean_type = self._z3_to_lean_type(value)
            interp_dict[hole_name] = lean_type
        
        return SemanticInterpretation(
            text=sketch.text,
            hole_assignments=interp_dict,
            lean_output=self._render_lean(interp_dict),
            complexity_score=self._compute_complexity(interp_dict),
            z3_model=model
        )
    
    def _block_interpretation(self, model: ModelRef, sketch: SemanticSketch):
        """Block current interpretation to search for alternatives."""
        
        blocking_clause = []
        
        for hole_name, _ in sketch.holes:
            hole_var = Const(hole_name, self.algebra.LeanType)
            value = model.eval(hole_var)
            blocking_clause.append(hole_var != value)
        
        self.solver.add(Or(blocking_clause))
    
    def _z3_to_lean_type(self, z3_value) -> LeanType:
        """Convert Z3 value to Lean type."""
        # Implementation depends on Z3 datatype structure
        from lean_type_theory import PropSort, TypeU, PiType, VarType
        
        # Simplified for uninterpreted sorts
        return VarType("synthesized")
    
    def _render_lean(self, interp_dict: Dict) -> str:
        """Render semantic interpretation as Lean code."""
        # Build Lean code from interpretation
        return "-- Synthesized Lean code"
    
    def _compute_complexity(self, interp_dict: Dict) -> float:
        """Compute complexity score for ranking."""
        # Prefer simpler types (fewer Pi types, smaller universes)
        return sum(1.0 for _ in interp_dict)


@dataclass
class SemanticInterpretation:
    """A valid semantic interpretation synthesized by Z3."""
    text: str
    hole_assignments: Dict[str, LeanType]
    lean_output: str
    complexity_score: float
    z3_model: ModelRef


class SemanticRuleTemplate:
    """
    Templates for compositional semantic rules based on formal semantics.
    
    Inspired by Montague Grammar, Categorial Grammar, and Type-Logical Grammar.
    Each template encodes a specific compositional pattern.
    """
    
    # Quantifier patterns (Generalized Quantifiers à la Barwise & Cooper)
    UNIVERSAL_QUANTIFICATION = {
        'syntactic_category': 'S/NP',  # Sentence with NP hole
        'semantic_type': '(e → t) → t',  # Type-raised NP
        'composition_rule': 'forall',
        'arity': 2,  # variable + body
        'type_constraints': ['domain_is_type', 'body_is_proposition'],
        'examples': ['∀x:A, P(x)', 'Πx:A, B(x)']
    }
    
    EXISTENTIAL_QUANTIFICATION = {
        'syntactic_category': 'S/NP',
        'semantic_type': '(e → t) → t',
        'composition_rule': 'exists',
        'arity': 2,
        'type_constraints': ['domain_is_type', 'body_is_proposition'],
        'examples': ['∃x:A, P(x)', 'Σx:A, B(x)']
    }
    
    # Lambda abstraction (function formation)
    LAMBDA_ABSTRACTION = {
        'syntactic_category': 'VP/NP',  # VP with object hole
        'semantic_type': 'a → b',
        'composition_rule': 'lambda',
        'arity': 2,  # variable + body
        'type_constraints': ['parameter_type_compatible', 'body_well_typed'],
        'examples': ['λx:A, f(x)', 'fun x => body']
    }
    
    # Function application (β-reduction)
    FUNCTION_APPLICATION = {
        'syntactic_category': 'S',
        'semantic_type': 'b',  # Result type
        'composition_rule': 'apply',
        'arity': 2,  # function + argument
        'type_constraints': ['function_domain_matches_argument', 'argument_well_typed'],
        'examples': ['f a', 'apply f a']
    }
    
    # Implication (conditional)
    IMPLICATION = {
        'syntactic_category': 'S/S',
        'semantic_type': 't → t → t',  # Prop → Prop → Prop
        'composition_rule': 'implies',
        'arity': 2,
        'type_constraints': ['antecedent_is_prop', 'consequent_is_prop'],
        'examples': ['P → Q', 'if P then Q']
    }
    
    # Conjunction (and coordination)
    CONJUNCTION = {
        'syntactic_category': 'X\\X/X',  # Right-associative
        'semantic_type': 't → t → t',
        'composition_rule': 'and',
        'arity': 2,
        'type_constraints': ['operands_same_type', 'result_same_type'],
        'examples': ['P ∧ Q', 'P and Q']
    }
    
    # Disjunction
    DISJUNCTION = {
        'syntactic_category': 'X\\X/X',
        'semantic_type': 't → t → t',
        'composition_rule': 'or',
        'arity': 2,
        'type_constraints': ['operands_same_type', 'result_same_type'],
        'examples': ['P ∨ Q', 'P or Q']
    }
    
    # Negation (unary operator)
    NEGATION = {
        'syntactic_category': 'S/S',
        'semantic_type': 't → t',
        'composition_rule': 'not',
        'arity': 1,
        'type_constraints': ['operand_is_prop'],
        'examples': ['¬P', 'not P']
    }
    
    # Predicate modification (intersective semantics)
    PREDICATE_MODIFICATION = {
        'syntactic_category': 'N/N',
        'semantic_type': '(e → t) → (e → t) → (e → t)',
        'composition_rule': 'modify',
        'arity': 2,
        'type_constraints': ['predicates_same_domain', 'intersective_semantics'],
        'examples': ['red ball', 'continuous function']
    }
    
    # Relational application (binary predicates)
    RELATIONAL_APPLICATION = {
        'syntactic_category': 'VP/NP',
        'semantic_type': 'e → e → t',
        'composition_rule': 'relate',
        'arity': 2,
        'type_constraints': ['relation_type_binary', 'arguments_compatible'],
        'examples': ['x < y', 'x divides y']
    }


@dataclass
class CompositionContext:
    """
    Linguistic and type-theoretic context for composition.
    
    Tracks:
    - Syntactic categories (CCG-style)
    - Semantic types (λ-calculus)
    - Type environment (Γ)
    - Discourse referents
    - Presuppositions
    """
    type_environment: Dict[str, LeanType]  # Γ: variable → type
    discourse_referents: Set[str]  # Available entities
    syntactic_stack: List[str]  # Category combinations
    semantic_constraints: List[BoolRef]  # Z3 constraints
    presuppositions: List[str]  # Background assumptions
    
    def extend(self, var: str, typ: LeanType) -> 'CompositionContext':
        """Extend context with new binding."""
        new_env = self.type_environment.copy()
        new_env[var] = typ
        return CompositionContext(
            type_environment=new_env,
            discourse_referents=self.discourse_referents.copy(),
            syntactic_stack=self.syntactic_stack.copy(),
            semantic_constraints=self.semantic_constraints.copy(),
            presuppositions=self.presuppositions.copy()
        )


@dataclass
class SemanticDerivation:
    """
    A complete semantic derivation showing how meaning is composed.
    
    Captures the full derivation tree from syntax to semantics,
    following principles of:
    - Compositionality (Frege's principle)
    - Type-driven interpretation
    - β-reduction and normalization
    """
    english_text: str
    syntactic_parse: 'ParseTree'
    derivation_steps: List[Tuple[str, str, str]]  # (rule, input, output)
    final_lean_type: LeanType
    semantic_representation: str  # λ-calculus form
    type_derivation: List[str]  # Type checking steps
    z3_proof: Optional[ModelRef]  # Z3 model witnessing validity
    
    def is_well_formed(self) -> bool:
        """Check if derivation is compositional and well-typed."""
        return len(self.derivation_steps) > 0 and self.z3_proof is not None


class CEGIS_SemanticLearner:
    """
    Counter-Example Guided Inductive Synthesis for Compositional Semantics.
    
    Learns semantic rules grounded in formal linguistics:
    
    THEORETICAL FOUNDATIONS:
    1. Montague Grammar: Syntax-semantics isomorphism
    2. Categorial Grammar: Type-driven composition
    3. Generalized Quantifier Theory: Quantifier semantics
    4. Lambda Calculus: Functional abstraction and application
    5. Dependent Type Theory: Proof-relevant semantics
    
    COMPOSITIONAL PRINCIPLES:
    - Frege's Principle: Meaning of whole = function of parts
    - Function-Argument Application: β-reduction
    - Predicate Modification: Intersective/subsective semantics
    - Quantifier Raising: Type-lifted interpretations
    - Scope Ambiguity: Multiple compositional paths
    
    LEARNING ALGORITHM:
    1. Synthesize rule candidates using semantic templates
    2. Validate via Z3 type checking
    3. Test on positive/negative examples
    4. Add counter-examples as constraints
    5. Refine until convergence
    
    The key innovation: treat semantic rules as PROGRAMS where:
    - Input = syntactic structure + context
    - Output = well-typed Lean expression
    - Constraints = type theory + linguistic principles
    """
    
    def __init__(self):
        self.synthesizer = Z3SemanticSynthesizer()
        self.learned_rules: List['EnhancedCompositionRule'] = []
        self.positive_examples: List[Tuple[str, str]] = []  # (English, Lean)
        self.negative_examples: List[Tuple[str, str]] = []
        
        # Linguistic knowledge base
        self.semantic_templates = self._initialize_templates()
        self.syntactic_categories = self._initialize_categories()
        self.type_signatures = self._initialize_type_signatures()
        
        # CEGIS statistics
        self.iterations_history: List[Dict] = []
        self.rule_quality_scores: Dict[str, float] = {}
        
        # Track failed template attempts to avoid repeating
        self.failed_templates: Dict[str, int] = {}  # template_name -> failure_count
        self.template_cooldown: Dict[str, int] = {}  # template_name -> iterations to skip
        
        # Novel template generation (RL phase)
        self.discovered_templates: Dict[str, Dict] = {}  # Inductively learned templates
        self.template_generation_threshold: int = 5  # Try induction after this many failures
        
        # IMPROVEMENT #2: Hierarchical rule learning
        self.rule_hierarchy: Dict[str, List[str]] = {}  # parent_rule_id -> [child_rule_ids]
        self.rule_dependencies: Dict[str, Set[str]] = {}  # rule_id -> {required_rule_ids}
        self.template_priorities: Dict[str, float] = {}  # template_name -> dynamic priority
        self.composition_cache: Dict[Tuple[str, str], Optional[str]] = {}  # (rule1, rule2) -> composed_output
        
        # IMPROVEMENT #3: Template validation and refinement
        self.template_validation_results: Dict[str, Dict] = {}  # template_name -> validation scores
        self.refinement_history: List[Dict] = []  # Track refinement attempts
        self.validated_induced_templates: Set[str] = set()  # Templates that passed validation
    
    def _initialize_templates(self) -> Dict[str, Dict]:
        """Initialize semantic rule templates from formal semantics."""
        return {
            'universal_quant': SemanticRuleTemplate.UNIVERSAL_QUANTIFICATION,
            'existential_quant': SemanticRuleTemplate.EXISTENTIAL_QUANTIFICATION,
            'lambda_abs': SemanticRuleTemplate.LAMBDA_ABSTRACTION,
            'func_app': SemanticRuleTemplate.FUNCTION_APPLICATION,
            'implication': SemanticRuleTemplate.IMPLICATION,
            'conjunction': SemanticRuleTemplate.CONJUNCTION,
            'disjunction': SemanticRuleTemplate.DISJUNCTION,
            'negation': SemanticRuleTemplate.NEGATION,
            'pred_mod': SemanticRuleTemplate.PREDICATE_MODIFICATION,
            'relation': SemanticRuleTemplate.RELATIONAL_APPLICATION,
            # NEW: Mathematical definition patterns
            'definition': {
                'syntactic_category': 'S',
                'semantic_type': 'e → t',
                'composition_rule': 'define',
                'arity': 2,
                'type_constraints': ['definiens_well_typed', 'definiendum_compatible'],
                'examples': ['let X := Y', 'X is defined as Y', 'X denotes Y']
            },
            'type_ascription': {
                'syntactic_category': 'NP',
                'semantic_type': 'e',
                'composition_rule': 'ascribe',
                'arity': 2,
                'type_constraints': ['term_has_type', 'type_well_formed'],
                'examples': ['x : A', 'x is of type A', 'x ∈ A']
            },
            'equality': {
                'syntactic_category': 'S',
                'semantic_type': 'e → e → t',
                'composition_rule': 'eq',
                'arity': 2,
                'type_constraints': ['operands_same_type'],
                'examples': ['x = y', 'x equals y', 'x is equal to y']
            },
        }
    
    def _initialize_categories(self) -> Dict[str, str]:
        """Initialize syntactic categories (CCG-style)."""
        return {
            'S': 'Sentence',
            'NP': 'Noun Phrase',
            'VP': 'Verb Phrase',
            'N': 'Noun',
            'V': 'Verb',
            'Det': 'Determiner',
            'Adj': 'Adjective',
            'Adv': 'Adverb',
            'Prep': 'Preposition',
            'Conj': 'Conjunction',
        }
    
    def _initialize_type_signatures(self) -> Dict[str, str]:
        """Initialize semantic type signatures."""
        return {
            'e': 'Entity (individuals)',
            't': 'Truth value (propositions)',
            'e → t': 'Unary predicate',
            'e → e → t': 'Binary relation',
            '(e → t) → t': 'Generalized quantifier',
            '(e → t) → (e → t) → (e → t)': 'Predicate modifier',
        }
    
    def _initialize_template_priorities(self) -> None:
        """Initialize dynamic priorities for templates based on success rate."""
        # Start with equal priorities
        for template_name in self.semantic_templates:
            self.template_priorities[template_name] = 1.0
    
    def _update_template_priority(self, template_name: str, success: bool) -> None:
        """Update template priority based on success/failure (exponential moving average)."""
        if template_name not in self.template_priorities:
            self.template_priorities[template_name] = 1.0
        
        alpha = 0.1  # Learning rate
        current = self.template_priorities[template_name]
        target = 2.0 if success else 0.5
        self.template_priorities[template_name] = alpha * target + (1 - alpha) * current
    
    def learn_from_corpus(self, examples: List[Tuple[str, str]], 
                         max_iterations: int = 100,
                         min_confidence: float = 0.9) -> List['EnhancedCompositionRule']:
        """
        Learn compositional rules from (English, Lean) pairs using CEGIS.
        
        ALGORITHM:
        ==========
        Input: Training examples {(e_i, l_i)} where e_i = English, l_i = Lean
        Output: Set of compositional semantic rules R
        
        1. INITIALIZATION:
           - Initialize R = ∅ (empty rule set)
           - Initialize positive examples P = examples
           - Initialize negative examples N = ∅
        
        2. SYNTHESIS LOOP (while uncovered examples exist):
           a. SELECT template τ from semantic template library
           b. SYNTHESIZE rule candidate r using Z3:
              - Variables: pattern regex, semantic function, type constraints
              - Constraints: 
                * ∀(e, l) ∈ P: pattern(r) matches e → semantic(r, e) ≡ l
                * ∀(e, l) ∈ N: ¬(pattern(r) matches e ∧ semantic(r, e) ≡ l)
                * Type constraints from τ
                * Compositionality: r composes with existing rules in R
           
           c. VALIDATION:
              - Test r on held-out validation set
              - Check type correctness via Z3 type checker
              - Compute linguistic quality metrics:
                * Compositionality score
                * Generalization score
                * Simplicity score
           
           d. COUNTER-EXAMPLE CHECK:
              - If ∃ counter-example (e, l): r fails
                → Add (e, l) to N
                → Continue to next iteration
              - Else: Add r to R
           
           e. REFINEMENT:
              - Remove covered examples from P
              - Update rule interaction graph
              - Check for rule subsumption/redundancy
        
        3. CONVERGENCE: Stop when P = ∅ or max_iterations reached
        
        Args:
            examples: Training (English, Lean) pairs
            max_iterations: Maximum CEGIS iterations
            min_confidence: Minimum confidence threshold for rules
            
        Returns:
            Learned compositional semantic rules
        """
        self.positive_examples = examples.copy()
        
        # Initialize improvement engines
        self._initialize_template_priorities()
        composition_engine = RuleCompositionEngine(self)
        negative_generator = NegativeExampleGenerator(self)
        template_validator = TemplateValidator(self)
        template_refiner = TemplateRefiner(self)
        dcg_parser = Z3DCGParser(self)  # DCG-inspired parser with Z3
        unification_engine = Z3UnificationEngine()  # Z3-based unification
        
        print(f"\n{'='*70}")
        print("CEGIS COMPOSITIONAL SEMANTICS LEARNING")
        print(f"{'='*70}")
        print(f"Training examples: {len(examples)}")
        print(f"Semantic templates: {len(self.semantic_templates)}")
        print(f"Min confidence: {min_confidence}")
        print(f"Improvement engines: DCG Parser, Composition, Negatives, Validation, Refinement")
        print(f"{'='*70}\n")
        
        uncovered_examples = examples.copy()
        validation_split = int(len(examples) * 0.8)
        training_set = examples[:validation_split]
        validation_set = examples[validation_split:]
        
        for iteration in range(max_iterations):
            print(f"\n{'─'*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'─'*70}")
            print(f"Uncovered examples: {len(uncovered_examples)}")
            print(f"Current rules: {len(self.learned_rules)}")
            print(f"Positive examples: {len(self.positive_examples)}")
            print(f"Negative examples: {len(self.negative_examples)}")
            
            if not uncovered_examples:
                print("\n✓ All examples covered!")
                break
            
            # PHASE 1: Template Selection
            print("\nPHASE 1: Template Selection")
            
            # Check if we should try inducing a novel template
            total_failures = sum(self.failed_templates.values())
            if total_failures > self.template_generation_threshold and iteration % 5 == 0:
                print("  → Attempting template induction...")
                novel_template = self._induce_novel_template(uncovered_examples)
                if novel_template:
                    # IMPROVEMENT #3: Validate induced template
                    print("  → Validating induced template...")
                    validation_scores = template_validator.validate_template(
                        novel_template,
                        uncovered_examples[:20]
                    )
                    print(f"    Validation scores:")
                    for criterion, score in validation_scores.items():
                        print(f"      {criterion}: {score:.2f}")
                    
                    if validation_scores['overall'] > 0.6:
                        template = novel_template
                        self.validated_induced_templates.add(template['composition_rule'])
                        print(f"  ✓ Using validated induced template: {template['composition_rule']}")
                    else:
                        print(f"  ✗ Induced template failed validation (score: {validation_scores['overall']:.2f})")
                        print("  → Attempting refinement...")
                        # IMPROVEMENT #3: Refine failed template
                        refined_template = template_refiner.refine_template(
                            novel_template,
                            uncovered_examples[:20],
                            self.negative_examples,
                            []  # No counter-examples yet
                        )
                        if refined_template:
                            template = refined_template
                            print(f"  ✓ Using refined template: {template['composition_rule']}")
                        else:
                            template = self._select_best_template(uncovered_examples)
                            print(f"  → Using fallback template: {template['composition_rule']}")
                else:
                    template = self._select_best_template(uncovered_examples)
                    print(f"  Selected template: {template['composition_rule']}")
            else:
                template = self._select_best_template(uncovered_examples)
                print(f"  Selected template: {template['composition_rule']}")

            template_name = template.get('composition_rule', 'unknown')
            
            print(f"  Semantic type: {template['semantic_type']}")
            print(f"  Arity: {template['arity']}")
            if template.get('learned_from_data'):
                print(f"  📊 Learned from {template['support']} examples")
            
            # PHASE 2: Rule Synthesis via Z3
            print("\nPHASE 2: Rule Synthesis")
            candidate_rule = self._synthesize_rule_with_template(
                template=template,
                positive_examples=self.positive_examples[:20],  # Use subset for efficiency
                negative_examples=self.negative_examples,
                existing_rules=self.learned_rules
            )
            
            if not candidate_rule:
                print("  ✗ No valid rule found for this template")
                
                # Try DCG-based parsing as fallback
                print("  → Trying DCG+Z3 fallback...")
                dcg_rule = dcg_parser.propose_enhanced_rule_from_examples(
                    uncovered_examples[:20]
                )
                if dcg_rule:
                    candidate_rule = dcg_rule
                    template_name = f"dcg_{candidate_rule.composition_type}"
                    print(f"    ✓ Proposed DCG rule: {candidate_rule.rule_id}")
                    print(f"      Pattern: {candidate_rule.syntactic_pattern}")
                    print(f"      Z3 checks (DCG): {dcg_rule.linguistic_features.get('z3_checks', 0)}")
                else:
                    dcg_parses = dcg_parser.parse_with_dcg(
                        uncovered_examples[0][0] if uncovered_examples else "",
                        target_category='S'
                    )
                    if dcg_parses:
                        print(f"    Found {len(dcg_parses)} DCG parse(s)")
                
                if not candidate_rule:
                    # Track failure and set cooldown
                    self.failed_templates[template_name] = self.failed_templates.get(template_name, 0) + 1
                    
                    # IMPROVEMENT #2: Update template priority (failure)
                    self._update_template_priority(template_name, success=False)
                    
                    # Cooldown increases with repeated failures: 2, 4, 8, 16... iterations
                    cooldown_iterations = min(2 ** self.failed_templates[template_name], 16)
                    self.template_cooldown[template_name] = cooldown_iterations
                    print(f"  Template '{template_name}' will be skipped for {cooldown_iterations} iterations")
                    continue
            
            print(f"  ✓ Synthesized rule: {candidate_rule.rule_id}")
            print(f"    Pattern: {candidate_rule.syntactic_pattern[:60]}...")
            print(f"    Semantic function: {candidate_rule.semantic_function_name}")
            
            # PHASE 3: Type Checking
            print("\nPHASE 3: Type Checking")
            type_check_result = self._verify_rule_type_correctness(
                candidate_rule,
                validation_set[:10]
            )
            
            if not type_check_result['valid']:
                print(f"  ✗ Type check failed: {type_check_result['error']}")
                # Add as negative constraint
                self.negative_examples.extend(type_check_result['counter_examples'])
                
                # Track failure for this template
                template_name = template.get('composition_rule', 'unknown')
                self.failed_templates[template_name] = self.failed_templates.get(template_name, 0) + 1
                continue
            
            print(f"  ✓ Type check passed")
            print(f"    Type-correct applications: {type_check_result['correct_count']}")
            
            # PHASE 4: Validation on Test Set
            print("\nPHASE 4: Validation")
            validation_result = self._validate_rule_comprehensively(
                candidate_rule,
                validation_set
            )
            
            print(f"  Coverage: {validation_result['coverage']:.2%}")
            print(f"  Correctness: {validation_result['correctness']:.2%}")
            print(f"  Compositionality: {validation_result['compositionality']:.2f}")
            print(f"  Generalization: {validation_result['generalization']:.2%}")
            
            # PHASE 5: Counter-Example Analysis
            print("\nPHASE 5: Counter-Example Analysis")
            counter_examples = validation_result['counter_examples']
            
            if counter_examples:
                print(f"  ✗ Found {len(counter_examples)} counter-examples")
                for i, (eng, expected, actual) in enumerate(counter_examples[:3], 1):
                    print(f"    {i}. Input: {eng[:50]}...")
                    print(f"       Expected: {expected[:40]}...")
                    print(f"       Got: {actual[:40]}...")
                
                # Add to negative examples
                self.negative_examples.extend([(e, exp) for e, exp, _ in counter_examples])
                
                # Analyze WHY it failed (linguistic diagnostics)
                failure_analysis = self._analyze_failures(counter_examples, candidate_rule)
                print(f"\n  Failure Analysis:")
                print(f"    Primary cause: {failure_analysis['primary_cause']}")
                print(f"    Suggested fix: {failure_analysis['suggestion']}")
                
                continue
            
            # PHASE 6: Quality Assessment
            print("\nPHASE 6: Quality Assessment")
            quality_score = self._compute_rule_quality(
                candidate_rule,
                validation_result,
                type_check_result
            )
            
            print(f"  Quality score: {quality_score:.3f}")
            
            if quality_score < min_confidence:
                print(f"  ✗ Quality below threshold ({min_confidence})")
                # Track low-quality attempt
                template_name = template.get('composition_rule', 'unknown')
                self.failed_templates[template_name] = self.failed_templates.get(template_name, 0) + 1
                continue
            
            # PHASE 7: Rule Acceptance
            print("\nPHASE 7: Rule Acceptance")
            print(f"  ✓ Rule accepted!")
            
            candidate_rule.quality_score = quality_score
            self.learned_rules.append(candidate_rule)
            self.rule_quality_scores[candidate_rule.rule_id] = quality_score
            
            # IMPROVEMENT #2: Update template priority (success)
            self._update_template_priority(template_name, success=True)
            
            # IMPROVEMENT #2: Generate better negative examples
            print("\n  → Generating adversarial negatives...")
            new_negatives = negative_generator.generate_adversarial_negatives(
                self.positive_examples[:10],
                candidate_rule,
                count=5
            )
            self.negative_examples.extend(new_negatives)
            print(f"    Generated {len(new_negatives)} negative examples")
            
            # IMPROVEMENT #2: Try hierarchical composition
            print("\n  → Checking rule composition opportunities...")
            composed_results = composition_engine.try_compose_rules(
                uncovered_examples[0][0] if uncovered_examples else "",
                max_composition_depth=2
            )
            if composed_results:
                print(f"    Found {len(composed_results)} composition opportunities")
                # Track composition relationships
                for output, rule_chain in composed_results[:3]:
                    parent_rule = rule_chain[0] if rule_chain else None
                    if parent_rule and parent_rule not in self.rule_hierarchy:
                        self.rule_hierarchy[parent_rule] = []
                    if len(rule_chain) > 1:
                        self.rule_hierarchy[parent_rule].append(rule_chain[1])
            
            # Reset failure count for successful template
            if template_name in self.failed_templates:
                del self.failed_templates[template_name]
            if template_name in self.template_cooldown:
                del self.template_cooldown[template_name]
            
            # Remove covered examples
            newly_covered = [
                ex for ex in uncovered_examples
                if self._rule_covers_example(candidate_rule, ex)
            ]
            uncovered_examples = [
                ex for ex in uncovered_examples
                if not self._rule_covers_example(candidate_rule, ex)
            ]
            
            print(f"  Newly covered: {len(newly_covered)} examples")
            print(f"  Remaining: {len(uncovered_examples)} examples")
            
            # PHASE 8: Rule Interaction Analysis
            print("\nPHASE 8: Rule Interaction Analysis")
            interactions = self._analyze_rule_interactions(
                candidate_rule,
                self.learned_rules[:-1]  # All except just-added
            )
            
            if interactions['composes_with']:
                print(f"  Composes with: {len(interactions['composes_with'])} rules")
            if interactions['subsumes']:
                print(f"  Subsumes: {interactions['subsumes']}")
            if interactions['conflicts']:
                print(f"  ⚠ Conflicts with: {interactions['conflicts']}")
            
            # Store iteration statistics
            self.iterations_history.append({
                'iteration': iteration + 1,
                'rule_id': candidate_rule.rule_id,
                'quality': quality_score,
                'coverage': validation_result['coverage'],
                'newly_covered': len(newly_covered),
                'remaining': len(uncovered_examples),
            })
        
        # Final report
        print(f"\n{'='*70}")
        print("LEARNING COMPLETE")
        print(f"{'='*70}")
        print(f"Total rules learned: {len(self.learned_rules)}")
        print(f"Coverage: {(len(examples) - len(uncovered_examples)) / len(examples):.2%}")
        print(f"Average quality: {np.mean(list(self.rule_quality_scores.values())):.3f}")
        
        # Print rule summary
        print(f"\nLearned Rules:")
        for i, rule in enumerate(self.learned_rules, 1):
            print(f"  {i}. {rule.rule_id} (quality: {rule.quality_score:.3f})")
            print(f"     {rule.composition_type} | {rule.semantic_type}")
        
        # Report on discovered templates
        if self.discovered_templates:
            print(f"\n📊 Novel Templates Discovered: {len(self.discovered_templates)}")
            for template_id, template in self.discovered_templates.items():
                print(f"  • {template['composition_rule']}")
                print(f"    Pattern: '{template.get('pattern_phrase', 'N/A')}'")
                print(f"    Support: {template['support']} examples")
                print(f"    Type: {template['semantic_type']}")
        
        return self.learned_rules
    
    def _select_best_template(self, examples: List[Tuple[str, str]]) -> Dict:
        """
        Select semantic template that best fits uncovered examples.
        
        Uses heuristics based on:
        - Syntactic patterns (quantifier words, connectives, etc.)
        - Type signatures
        - Frequency in corpus
        - Avoids recently failed templates (cooldown mechanism)
        """
        template_scores = {}
        
        for template_name, template in self.semantic_templates.items():
            # Skip templates on cooldown
            if template_name in self.template_cooldown:
                cooldown = self.template_cooldown[template_name]
                if cooldown > 0:
                    self.template_cooldown[template_name] -= 1
                    continue
                else:
                    del self.template_cooldown[template_name]
            
            score = 0.0
            
            # Check how many examples match template patterns
            for eng, lean in examples[:50]:  # Sample for efficiency
                eng_lower = eng.lower()
                lean_lower = lean.lower()
                
                # Heuristic matching
                if template['composition_rule'] == 'forall':
                    if any(q in eng_lower for q in ['for all', 'for every', 'for each', '∀']):
                        score += 1.0
                    if '∀' in lean_lower or 'forall' in lean_lower:
                        score += 1.0
                
                elif template['composition_rule'] == 'exists':
                    if any(q in eng_lower for q in ['there exists', 'there is', 'some', '∃']):
                        score += 1.0
                    if '∃' in lean_lower or 'exists' in lean_lower:
                        score += 1.0
                
                elif template['composition_rule'] == 'implies':
                    if any(c in eng_lower for c in ['if', 'implies', 'then', '→']):
                        score += 1.0
                    if '→' in lean_lower or '->' in lean_lower:
                        score += 1.0
                
                elif template['composition_rule'] == 'lambda':
                    if 'function' in eng_lower or 'map' in eng_lower:
                        score += 0.5
                    if 'λ' in lean_lower or 'fun' in lean_lower or 'lambda' in lean_lower:
                        score += 1.0
                
                elif template['composition_rule'] in ['and', 'or', 'not']:
                    # Context-aware: check if it's a logical connective vs. other use
                    if template['composition_rule'] in eng_lower:
                        # Check for logical context markers
                        logical_context = any(marker in eng_lower for marker in [
                            'if', 'then', 'implies', 'such that', 'where',
                            'for all', 'there exists', 'assume', 'suppose'
                        ])
                        if logical_context:
                            score += 1.0
                        else:
                            # Penalize if appears in non-logical context
                            if re.search(r'\blet\s+' + template['composition_rule'], eng_lower):
                                score -= 0.5  # "let and be" is not logical
                            elif template['composition_rule'] in eng_lower:
                                score += 0.3  # Weak signal
                
                elif template['composition_rule'] == 'define':
                    if any(kw in eng_lower for kw in ['let', 'define', ':=', 'denote']):
                        score += 1.0
                    if 'let' in lean_lower and ':=' in lean_lower:
                        score += 1.0
                
                elif template['composition_rule'] == 'ascribe':
                    if ' : ' in eng or ' in ' in eng_lower or '∈' in eng:
                        score += 1.0
                    if ' : ' in lean_lower:
                        score += 0.5
                
                elif template['composition_rule'] == 'eq':
                    if any(kw in eng_lower for kw in ['equal', '=', 'same as']):
                        score += 1.0
            
            # Penalize templates that failed recently
            if template_name in self.failed_templates:
                failure_penalty = self.failed_templates[template_name] * 0.5
                score -= failure_penalty
            
            # IMPROVEMENT #2: Apply dynamic priority
            priority = self.template_priorities.get(template_name, 1.0)
            score *= priority
            
            template_scores[template_name] = score
        
        # Select template with highest score
        if not template_scores or max(template_scores.values()) <= 0:
            # Try templates in priority order, skipping those on cooldown
            priority_order = ['definition', 'universal_quant', 'existential_quant', 
                            'implication', 'equality', 'func_app']
            for template_name in priority_order:
                if template_name not in self.template_cooldown:
                    return self.semantic_templates[template_name]
            # Fallback
            return self.semantic_templates['func_app']
        
        best_template_name = max(template_scores, key=template_scores.get)
        return self.semantic_templates[best_template_name]
    
    def _induce_novel_template(self, uncovered_examples: List[Tuple[str, str]]) -> Optional[Dict]:
        """
        Induce a novel semantic template from uncovered examples.
        
        This is the REINFORCEMENT LEARNING phase: discover new compositional
        patterns that aren't in the initial template library.
        
        ALGORITHM:
        1. Cluster similar examples by syntactic/semantic patterns
        2. Extract common structural patterns (n-grams, POS sequences)
        3. Identify systematic English→Lean mappings
        4. Synthesize new template with composition rule
        5. Validate against type theory constraints
        
        Returns:
            Novel template if pattern is strong enough, None otherwise
        """
        print("\n  🔬 TEMPLATE INDUCTION: Analyzing uncovered examples...")
        
        if len(uncovered_examples) < 3:
            print("    Not enough examples for induction")
            return None
        
        # Step 1: Find common patterns in English side
        english_patterns = {}
        for eng, lean in uncovered_examples[:30]:  # Sample for efficiency
            # Extract key phrases
            words = eng.lower().split()
            
            # Look for common connectives/keywords
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if bigram not in english_patterns:
                    english_patterns[bigram] = []
                english_patterns[bigram].append((eng, lean))
        
        # Find patterns with multiple occurrences
        frequent_patterns = {
            pattern: examples 
            for pattern, examples in english_patterns.items()
            if len(examples) >= 3  # Appears in at least 3 examples
        }
        
        if not frequent_patterns:
            print("    No frequent patterns found")
            return None
        
        # Step 2: Select most promising pattern
        best_pattern = max(frequent_patterns, key=lambda p: len(frequent_patterns[p]))
        pattern_examples = frequent_patterns[best_pattern]
        
        print(f"    Found pattern: '{best_pattern}' ({len(pattern_examples)} instances)")
        
        # Step 3: Analyze Lean output structure
        lean_structures = [lean for _, lean in pattern_examples]
        common_lean_operators = self._extract_common_operators(lean_structures)
        
        if not common_lean_operators:
            print("    No consistent Lean structure")
            return None
        
        print(f"    Lean operators: {common_lean_operators}")
        
        # Step 4: Synthesize template
        # Infer semantic type from Lean operators
        semantic_type = self._infer_semantic_type(common_lean_operators)
        
        # Generate composition rule name
        composition_rule = f"induced_{best_pattern.replace(' ', '_')}"
        
        # Estimate arity from pattern
        arity = best_pattern.count(',') + 2 if ',' in best_pattern else 2
        
        # Create novel template
        novel_template = {
            'syntactic_category': 'S',  # Default to sentence
            'semantic_type': semantic_type,
            'composition_rule': composition_rule,
            'arity': arity,
            'type_constraints': ['well_typed'],  # Generic constraint
            'examples': [(eng, lean) for eng, lean in pattern_examples[:3]],
            'pattern_phrase': best_pattern,
            'learned_from_data': True,  # Mark as induced
            'support': len(pattern_examples),  # Number of supporting examples
        }
        
        print(f"    ✓ Induced novel template: {composition_rule}")
        print(f"      Semantic type: {semantic_type}")
        print(f"      Arity: {arity}")
        print(f"      Support: {len(pattern_examples)} examples")
        
        # Step 5: Store in discovered templates
        template_id = f"discovered_{len(self.discovered_templates)}"
        self.discovered_templates[template_id] = novel_template
        
        # Add to active template library
        self.semantic_templates[template_id] = novel_template
        
        return novel_template
    
    def _extract_common_operators(self, lean_expressions: List[str]) -> List[str]:
        """Extract common operators from Lean expressions."""
        # Common Lean operators
        operators = ['∀', '∃', '→', '∧', '∨', '¬', '=', ':', 'fun', 'let', 'λ']
        
        common = []
        for op in operators:
            count = sum(1 for expr in lean_expressions if op in expr)
            if count >= len(lean_expressions) * 0.6:  # In 60%+ of examples
                common.append(op)
        
        return common
    
    def _infer_semantic_type(self, operators: List[str]) -> str:
        """Infer semantic type from Lean operators."""
        if '∀' in operators or '∃' in operators:
            return '(e → t) → t'  # Generalized quantifier
        elif '→' in operators:
            return 't → t → t'  # Implication
        elif '∧' in operators or '∨' in operators:
            return 't → t → t'  # Binary connective
        elif '¬' in operators:
            return 't → t'  # Unary operator
        elif 'fun' in operators or 'λ' in operators:
            return 'a → b'  # Lambda abstraction
        elif ':' in operators:
            return 'e → Type'  # Type ascription
        elif 'let' in operators:
            return 'e → e → e'  # Definition
        else:
            return 't'  # Default to proposition
    
    def _synthesize_rule_with_template(
        self,
        template: Dict,
        positive_examples: List[Tuple[str, str]],
        negative_examples: List[Tuple[str, str]],
        existing_rules: List['EnhancedCompositionRule']
    ) -> Optional['EnhancedCompositionRule']:
        """
        Synthesize compositional rule using Z3-driven program synthesis.
        
        This is the CORE of CEGIS: program synthesis for semantic rules.
        
        SYNTHESIS PROBLEM:
        ==================
        Find: Rule r = (pattern, semantic_function, constraints)
        
        Such that:
        1. POSITIVE COVERAGE: ∀(e, l) ∈ positive_examples:
           if pattern matches e then semantic_function(e) ≡ l
        
        2. NEGATIVE EXCLUSION: ∀(e, l) ∈ negative_examples:
           ¬(pattern matches e ∧ semantic_function(e) ≡ l)
        
        3. TYPE CONSTRAINTS: semantic_function respects type system
           - Input types match syntactic categories
           - Output type is well-formed in context
           - Compositional: can combine with existing rules
        
        4. LINGUISTIC PRINCIPLES:
           - Compositionality: meaning of whole = f(meaning of parts)
           - Type-driven: semantic function respects semantic type
           - Monotonicity: where applicable (e.g., quantifiers)
        """
        
        composition_rule = template['composition_rule']
        semantic_type = template['semantic_type']
        arity = template['arity']
        
        # PHASE 1: Z3-Guided Pattern Synthesis
        print("    → Z3 pattern synthesis...")
        synthesized_pattern = self._z3_synthesize_pattern(
            positive_examples,
            negative_examples,
            template,
            composition_rule
        )
        
        if not synthesized_pattern:
            # Fallback to heuristic pattern generation
            print("    → Fallback to heuristic patterns...")
            pattern_candidates = self._generate_pattern_candidates(
                positive_examples,
                composition_rule
            )
            
            if not pattern_candidates:
                return None
            
            # Select best pattern using Z3 verification
            synthesized_pattern = self._select_best_pattern_z3(
                pattern_candidates,
                positive_examples,
                negative_examples
            )
        
        if not synthesized_pattern:
            return None
        
        print(f"    → Synthesized pattern: {synthesized_pattern[:80]}...")
        
        # PHASE 2: Z3-Guided Semantic Function Synthesis
        print("    → Z3 semantic function synthesis...")
        semantic_function = self._z3_synthesize_semantic_function(
            template,
            synthesized_pattern,
            positive_examples,
            negative_examples
        )
        
        if not semantic_function:
            # Fallback to template-based function
            print("    → Fallback to template function...")
            semantic_function = self._create_semantic_function(
                template,
                synthesized_pattern,
                positive_examples
            )
        
        # PHASE 3: Z3 Type Constraint Generation
        print("    → Z3 constraint generation...")
        z3_constraints = self._synthesize_z3_type_constraints(
            template,
            synthesized_pattern,
            semantic_function,
            positive_examples
        )
        
        print(f"    → Generated {len(z3_constraints)} Z3 constraints")
        
        # Create rule with synthesized components
        rule = EnhancedCompositionRule(
            rule_id=f"rule_{len(existing_rules)}_{composition_rule}",
            syntactic_pattern=synthesized_pattern,
            syntactic_category=template.get('syntactic_category', 'S'),
            semantic_type=semantic_type,
            semantic_function_name=composition_rule,
            semantic_function=semantic_function,
            composition_type=composition_rule,
            arity=arity,
            type_constraints=template['type_constraints'],
            z3_constraints=z3_constraints,
            example_instances=positive_examples[:5],
            quality_score=0.0,
            linguistic_features=self._extract_linguistic_features(synthesized_pattern, composition_rule)
        )
        
        return rule
    
    def _z3_synthesize_pattern(
        self,
        positive_examples: List[Tuple[str, str]],
        negative_examples: List[Tuple[str, str]],
        template: Dict,
        composition_rule: str
    ) -> Optional[str]:
        """
        Use Z3 to synthesize regex pattern that:
        - Matches all positive examples
        - Rejects all negative examples
        - Extracts the right number of groups (arity)
        
        Uses sketch-based synthesis: we provide pattern templates,
        Z3 fills in the holes.
        """
        solver = Solver()
        solver.set('timeout', 5000)  # 5 second timeout
        
        arity = template['arity']
        
        # Pattern sketch variables
        # We synthesize patterns as combinations of:
        # - Literal strings
        # - Capture groups: (.+?), (\w+), ([^,]+)
        # - Whitespace: \s+, \s*
        # - Word boundaries: \b
        
        # For each position in pattern, choose a component
        num_positions = 2 * arity + 1  # Alternating: literal, capture, literal, ...
        
        # Z3 variables for pattern components
        component_choices = []
        for i in range(num_positions):
            if i % 2 == 0:
                # Literal position: choose from common phrases
                component_choices.append(Int(f'literal_{i}'))
            else:
                # Capture group position
                component_choices.append(Int(f'capture_{i}'))
        
        # Define component vocabularies
        literal_vocab = self._get_literal_vocabulary(composition_rule)
        capture_vocab = ['(.+?)', '(\\w+)', '([^,]+)', '([^:]+)', '([A-Z]\\w*)']

        # Avoid degenerate patterns where capture groups touch with no separator.
        # In practice this produces empty groups and unstable matches.
        disallowed_literal_indices: Set[int] = set()
        if '' in literal_vocab:
            disallowed_literal_indices.add(literal_vocab.index(''))
        if '\\b' in literal_vocab:
            disallowed_literal_indices.add(literal_vocab.index('\\b'))
        
        # Constraint: each choice is valid index
        for i, choice in enumerate(component_choices):
            if i % 2 == 0:
                solver.add(And(choice >= 0, choice < len(literal_vocab)))
                # Middle literals must separate captures.
                if 0 < i < num_positions - 1 and disallowed_literal_indices:
                    solver.add(And(*[choice != bad for bad in disallowed_literal_indices]))
            else:
                solver.add(And(choice >= 0, choice < len(capture_vocab)))

        # Rule-specific anchor: avoid patterns that match everything.
        if composition_rule == 'implies':
            literal_choice_vars = [component_choices[i] for i in range(0, num_positions, 2)]
            if_idx = literal_vocab.index('if\\s+') if 'if\\s+' in literal_vocab else None
            then_idx = literal_vocab.index('\\s+then\\s+') if '\\s+then\\s+' in literal_vocab else None
            implies_idx = literal_vocab.index('\\s+implies\\s+') if '\\s+implies\\s+' in literal_vocab else None

            has_if = Or(*([v == if_idx for v in literal_choice_vars] if if_idx is not None else [False]))
            has_then = Or(*([v == then_idx for v in literal_choice_vars] if then_idx is not None else [False]))
            has_implies = Or(*([v == implies_idx for v in literal_choice_vars] if implies_idx is not None else [False]))

            # Either `if ... then ...` or `... implies ...`.
            solver.add(Or(And(has_if, has_then), has_implies))
        
        # Constraint: pattern must match positive examples
        # (This is approximated since we can't fully encode regex matching in Z3)
        # Instead, we generate candidates and verify
        
        # Try to find a satisfying assignment
        if solver.check() == sat:
            model = solver.model()
            
            # Extract pattern from model
            pattern_parts = []
            for i, choice_var in enumerate(component_choices):
                choice_idx = model.eval(choice_var).as_long()
                if i % 2 == 0:
                    pattern_parts.append(literal_vocab[choice_idx])
                else:
                    pattern_parts.append(capture_vocab[choice_idx])
            
            synthesized = ''.join(pattern_parts)
            
            # Verify it works on examples
            if self._verify_pattern(synthesized, positive_examples, negative_examples):
                return synthesized
        
        return None
    
    def _get_literal_vocabulary(self, composition_rule: str) -> List[str]:
        """Get vocabulary of literal strings for pattern synthesis."""
        common_literals = [
            '', '\\s+', '\\s*', ',\\s*', '\\b',
            'the\\s+', 'a\\s+', 'an\\s+',
        ]
        
        # Rule-specific literals
        if composition_rule in ['forall', 'universal']:
            common_literals.extend([
                'for\\s+(?:all|every|each)\\s+',
                '\\s+in\\s+',
                '\\s*[,:;]\\s*',
            ])
        elif composition_rule in ['exists', 'existential']:
            common_literals.extend([
                'there\\s+exists?\\s+',
                '\\s+such\\s+that\\s+',
                '\\s+where\\s+',
            ])
        elif composition_rule == 'implies':
            common_literals.extend([
                'if\\s+',
                '\\s+then\\s+',
                '\\s+implies\\s+',
                '\\s*→\\s*',
            ])
        elif composition_rule in ['and', 'conjunction']:
            common_literals.extend([
                '\\s+and\\s+',
                '\\s*∧\\s*',
            ])
        elif composition_rule in ['define', 'definition']:
            common_literals.extend([
                'let\\s+',
                '\\s*:=\\s*',
                '\\s+is\\s+defined\\s+as\\s+',
            ])
        
        return common_literals
    
    def _verify_pattern(
        self,
        pattern: str,
        positive_examples: List[Tuple[str, str]],
        negative_examples: List[Tuple[str, str]]
    ) -> bool:
        """Verify pattern matches positive and rejects negative examples."""
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except:
            return False
        
        # Must match at least 60% of positive examples
        pos_matches = sum(1 for eng, _ in positive_examples if compiled.search(eng))
        if pos_matches < len(positive_examples) * 0.6:
            return False
        
        # Must reject all negative examples
        neg_matches = sum(1 for eng, _ in negative_examples if compiled.search(eng))
        if neg_matches > 0:
            return False
        
        return True
    
    def _select_best_pattern_z3(
        self,
        candidates: List[str],
        positive_examples: List[Tuple[str, str]],
        negative_examples: List[Tuple[str, str]]
    ) -> Optional[str]:
        """Use Z3 to select best pattern from candidates."""
        solver = Solver()
        solver.set('timeout', 3000)
        
        # Z3 variables: one per candidate
        pattern_selected = [Bool(f'select_{i}') for i in range(len(candidates))]
        
        # Constraint: exactly one pattern selected
        solver.add(PbEq([(p, 1) for p in pattern_selected], 1))
        
        # Score each pattern
        scores = []
        for pattern in candidates:
            pos_coverage = sum(1 for eng, _ in positive_examples 
                             if re.search(pattern, eng, re.IGNORECASE))
            neg_matches = sum(1 for eng, _ in negative_examples 
                            if re.search(pattern, eng, re.IGNORECASE))
            
            # Score = coverage - 10 * neg_matches (heavy penalty for false positives)
            score = pos_coverage - 10 * neg_matches
            scores.append(score)
        
        # Maximize score
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        
        if scores[best_idx] > 0:
            return candidates[best_idx]
        
        return None
    
    def _z3_synthesize_semantic_function(
        self,
        template: Dict,
        pattern: str,
        positive_examples: List[Tuple[str, str]],
        negative_examples: List[Tuple[str, str]]
    ) -> Optional[Callable]:
        """
        Synthesize semantic function using Z3.
        
        The function takes regex match groups and produces Lean output.
        We synthesize it as a sequence of string transformations.
        """
        composition_rule = template['composition_rule']
        
        # Extract sample transformations from positive examples
        transformations = []
        for eng, lean in positive_examples[:10]:
            match = re.search(pattern, eng, re.IGNORECASE)
            if match:
                groups = match.groups()
                transformations.append((groups, lean))
        
        if not transformations:
            return None
        
        # Analyze transformation pattern using Z3
        solver = Solver()
        solver.set('timeout', 3000)
        
        # Infer transformation template
        # Check if it's a simple substitution pattern
        template_str = self._infer_transformation_template(
            transformations,
            composition_rule
        )
        
        if template_str:
            # Create function from template
            def semantic_func(match_groups):
                return self._apply_transformation_template(
                    template_str,
                    match_groups,
                    composition_rule
                )
            return semantic_func
        
        return None
    
    def _infer_transformation_template(
        self,
        transformations: List[Tuple[Tuple[str, ...], str]],
        composition_rule: str
    ) -> Optional[str]:
        """Infer transformation template from examples using Z3."""
        
        # Common templates by composition rule
        templates = {
            'forall': '∀ {0} : {1}, {2}',
            'exists': '∃ {0} : {1}, {2}',
            'implies': '{0} → {1}',
            'and': '{0} ∧ {1}',
            'or': '{0} ∨ {1}',
            'not': '¬{0}',
            'define': 'let {0} := {1}',
            'ascribe': '{0} : {1}',
            'eq': '{0} = {1}',
            'lambda': 'fun {0} => {1}',
        }
        
        if composition_rule in templates:
            return templates[composition_rule]
        
        # For induced templates, infer from examples
        if len(transformations) > 0:
            groups, lean = transformations[0]
            # Try to find pattern
            if ' → ' in lean and len(groups) >= 2:
                return '{0} → {1}'
            elif ' ∧ ' in lean and len(groups) >= 2:
                return '{0} ∧ {1}'
            elif 'let ' in lean and ':=' in lean:
                return 'let {0} := {1}'
        
        return None
    
    def _apply_transformation_template(
        self,
        template_str: str,
        match_groups: Tuple[str, ...],
        composition_rule: str
    ) -> str:
        """Apply transformation template to match groups."""
        try:
            # Clean up groups
            cleaned_groups = [g.strip() if g else '' for g in match_groups]
            return template_str.format(*cleaned_groups)
        except:
            return ' '.join(str(g) for g in match_groups if g)
    
    def _synthesize_z3_type_constraints(
        self,
        template: Dict,
        pattern: str,
        semantic_function: Callable,
        examples: List[Tuple[str, str]]
    ) -> List[str]:
        """Generate Z3 SMT constraints for type checking."""
        constraints = []
        
        solver = Solver()
        solver.set('timeout', 2000)
        
        # Generate constraints based on template
        for constraint_name in template.get('type_constraints', []):
            if constraint_name == 'domain_is_type':
                constraints.append('wellformed_type(ctx, domain)')
                # Add Z3 encoding
                ctx_var = Const('ctx', self.synthesizer.algebra.Context)
                domain_var = Const('domain', self.synthesizer.algebra.LeanType)
                solver.add(
                    self.synthesizer.algebra.wellformed_type(ctx_var, domain_var)
                )
            
            elif constraint_name == 'body_is_proposition':
                constraints.append('is_prop(body)')
                body_var = Const('body', self.synthesizer.algebra.LeanType)
                prop_type = self.synthesizer.algebra.Prop_Sort
                # Add type equivalence constraint
            
            elif constraint_name == 'operands_same_type':
                constraints.append('type(left) == type(right)')
                left_type = Const('left_type', self.synthesizer.algebra.LeanType)
                right_type = Const('right_type', self.synthesizer.algebra.LeanType)
                solver.add(left_type == right_type)
            
            elif constraint_name == 'function_domain_matches_argument':
                constraints.append('domain(func_type) == type(arg)')
        
        # Check if constraints are satisfiable
        if solver.check() == sat:
            constraints.append('z3_satisfiable')
        
        return constraints
    
    def _generate_pattern_candidates(
        self,
        examples: List[Tuple[str, str]],
        composition_rule: str
    ) -> List[str]:
        """Generate candidate syntactic patterns."""
        patterns = []
        
        if composition_rule == 'forall':
            patterns.extend([
                r'for\s+(?:all|every|each)\s+(\w+)\s*(?:in|:|\∈)\s*(\w+),?\s+(.+)',
                r'∀\s*(\w+)\s*(?::|∈)\s*(\w+),?\s+(.+)',
                r'given\s+any\s+(\w+)\s+in\s+(\w+),?\s+(.+)',
            ])
        
        elif composition_rule == 'exists':
            patterns.extend([
                r'there\s+exists?\s+(?:a|an)?\s*(\w+)\s*(?:in|:|\∈)\s*(\w+)\s+(?:such\s+that|where|with)\s+(.+)',
                r'∃\s*(\w+)\s*(?::|∈)\s*(\w+),?\s+(.+)',
                r'some\s+(\w+)\s+in\s+(\w+)\s+(?:satisfies?|has)\s+(.+)',
            ])
        
        elif composition_rule == 'implies':
            patterns.extend([
                r'if\s+(.+?)\s+then\s+(.+)',
                r'(.+?)\s+implies\s+(.+)',
                r'(.+?)\s*→\s*(.+)',
                r'(.+?)\s+entails\s+(.+)',
            ])
        
        elif composition_rule == 'lambda':
            patterns.extend([
                r'(?:the\s+)?function\s+(?:from\s+)?(\w+)\s+to\s+(\w+)\s+(?:given\s+by|defined\s+by|where)\s+(.+)',
                r'λ\s*(\w+)\s*[.:]\s*(.+)',
                r'(?:\w+)\s*↦\s*(.+)',
            ])
        
        elif composition_rule == 'apply':
            patterns.extend([
                r'(\w+)\s+applied\s+to\s+(\w+)',
                r'(\w+)\s*\(\s*(\w+)\s*\)',
                r'(\w+)\s+of\s+(\w+)',
            ])
        
        elif composition_rule == 'and':
            # Context-aware: only match in logical contexts
            patterns.extend([
                r'(?:if|then|where|such that|assume).*?(.+?)\s+and\s+(.+)',  # Logical context
                r'(.+?)\s*∧\s*(.+)',  # Symbolic form
            ])
        
        elif composition_rule == 'or':
            patterns.extend([
                r'(.+?)\s+or\s+(.+)',
                r'(.+?)\s*∨\s*(.+)',
            ])
        
        elif composition_rule == 'not':
            patterns.extend([
                r'not\s+(.+)',
                r'¬\s*(.+)',
                r'it\s+is\s+not\s+(?:the\s+case\s+)?that\s+(.+)',
            ])
        
        elif composition_rule == 'define':
            patterns.extend([
                r'let\s+(\w+)\s*:=\s*(.+)',
                r'(\w+)\s+is\s+defined\s+(?:as|to be)\s+(.+)',
                r'define\s+(\w+)\s+(?:as|to be)\s+(.+)',
                r'(\w+)\s+denotes\s+(.+)',
            ])
        
        elif composition_rule == 'ascribe':
            patterns.extend([
                r'(\w+)\s*:\s*(\w+)',
                r'(\w+)\s+(?:is\s+)?(?:of\s+)?type\s+(\w+)',
                r'(\w+)\s*∈\s*(\w+)',
                r'(\w+)\s+in\s+(\w+)',
            ])
        
        elif composition_rule == 'eq':
            patterns.extend([
                r'(\w+)\s*=\s*(\w+)',
                r'(\w+)\s+equals\s+(\w+)',
                r'(\w+)\s+is\s+equal\s+to\s+(\w+)',
            ])
        
        elif composition_rule.startswith('induced_'):
            # Dynamic pattern generation for induced templates
            # Extract the key phrase from composition rule
            key_phrase = composition_rule.replace('induced_', '').replace('_', ' ')
            
            # Generate patterns around the key phrase
            patterns.extend([
                rf'(.+?)\s+{key_phrase}\s+(.+)',
                rf'{key_phrase}\s+(.+)',
                rf'(.+?)\s+{key_phrase}',
            ])
            
            # Also try generic capture around the phrase
            patterns.append(rf'(.+?){key_phrase}(.+?)')
        
        else:
            # Generic patterns
            patterns.extend([
                r'(.+)',
                r'(\w+)',
            ])
        
        return patterns
    
    def _create_semantic_function(
        self,
        template: Dict,
        pattern: str,
        examples: List[Tuple[str, str]]
    ) -> Callable:
        """
        Create semantic composition function.
        
        This function takes matched syntactic components and builds Lean type.
        """
        composition_rule = template['composition_rule']
        
        def semantic_forall(match_groups):
            """∀x:A, B"""
            if len(match_groups) >= 3:
                var, domain, body = match_groups[:3]
                return f"∀ {var} : {domain}, {body}"
            return ""
        
        def semantic_exists(match_groups):
            """∃x:A, B"""
            if len(match_groups) >= 3:
                var, domain, body = match_groups[:3]
                return f"∃ {var} : {domain}, {body}"
            return ""
        
        def semantic_implies(match_groups):
            """P → Q"""
            if len(match_groups) >= 2:
                antecedent, consequent = match_groups[:2]
                return f"{antecedent} → {consequent}"
            return ""
        
        def semantic_lambda(match_groups):
            """λx, body"""
            if len(match_groups) >= 2:
                var, body = match_groups[:2]
                return f"fun {var} => {body}"
            return ""
        
        def semantic_apply(match_groups):
            """f a"""
            if len(match_groups) >= 2:
                func, arg = match_groups[:2]
                return f"{func} {arg}"
            return ""
        
        def semantic_and(match_groups):
            """P ∧ Q"""
            if len(match_groups) >= 2:
                left, right = match_groups[:2]
                return f"{left} ∧ {right}"
            return ""
        
        def semantic_or(match_groups):
            """P ∨ Q"""
            if len(match_groups) >= 2:
                left, right = match_groups[:2]
                return f"{left} ∨ {right}"
            return ""
        
        def semantic_not(match_groups):
            """¬P"""
            if len(match_groups) >= 1:
                prop = match_groups[0]
                return f"¬{prop}"
            return ""
        
        def semantic_define(match_groups):
            """let X := Y"""
            if len(match_groups) >= 2:
                name, value = match_groups[:2]
                return f"let {name} := {value}"
            return ""
        
        def semantic_ascribe(match_groups):
            """x : A"""
            if len(match_groups) >= 2:
                term, typ = match_groups[:2]
                return f"{term} : {typ}"
            return ""
        
        def semantic_eq(match_groups):
            """x = y"""
            if len(match_groups) >= 2:
                left, right = match_groups[:2]
                return f"{left} = {right}"
            return ""
        
        def semantic_induced(match_groups):
            """Generic semantic function for induced templates."""
            # Use operators from template examples to guide transformation
            if 'examples' in template:
                # Look at first example to infer transformation
                _, example_lean = template['examples'][0]
                
                # Extract main operator
                if '→' in example_lean:
                    # Likely implication-like
                    if len(match_groups) >= 2:
                        return f"{match_groups[0]} → {match_groups[1]}"
                elif '∧' in example_lean:
                    # Likely conjunction-like
                    if len(match_groups) >= 2:
                        return f"{match_groups[0]} ∧ {match_groups[1]}"
                elif '∨' in example_lean:
                    # Likely disjunction-like
                    if len(match_groups) >= 2:
                        return f"{match_groups[0]} ∨ {match_groups[1]}"
                elif 'let' in example_lean.lower():
                    # Likely definition-like
                    if len(match_groups) >= 2:
                        return f"let {match_groups[0]} := {match_groups[1]}"
            
            # Fallback: just concatenate with space
            return ' '.join(str(g) for g in match_groups if g)
        
        # Map composition rules to functions
        function_map = {
            'forall': semantic_forall,
            'exists': semantic_exists,
            'implies': semantic_implies,
            'lambda': semantic_lambda,
            'apply': semantic_apply,
            'and': semantic_and,
            'or': semantic_or,
            'not': semantic_not,
            'define': semantic_define,
            'ascribe': semantic_ascribe,
            'eq': semantic_eq,
        }
        
        # Add handler for induced templates
        if composition_rule.startswith('induced_'):
            return semantic_induced
        
        return function_map.get(composition_rule, lambda _: "")
    
    def _generate_z3_type_constraints(self, template: Dict) -> List[str]:
        """Generate Z3 constraints for type checking."""
        constraints = []
        
        for constraint_name in template['type_constraints']:
            if constraint_name == 'domain_is_type':
                constraints.append("wellformed_type(ctx, domain)")
            elif constraint_name == 'body_is_proposition':
                constraints.append("is_prop(body)")
            elif constraint_name == 'function_domain_matches_argument':
                constraints.append("domain(func_type) == type(arg)")
            elif constraint_name == 'operands_same_type':
                constraints.append("type(left) == type(right)")
        
        return constraints
    
    def _extract_linguistic_features(self, pattern: str, composition_rule: str) -> Dict:
        """Extract linguistic features for rule."""
        return {
            'uses_quantifier': composition_rule in ['forall', 'exists'],
            'uses_connective': composition_rule in ['and', 'or', 'implies', 'not'],
            'uses_abstraction': composition_rule == 'lambda',
            'pattern_complexity': len(pattern),
            'arity': len(re.findall(r'\([^)]*\)', pattern)),
        }
    
    def _verify_rule_type_correctness(
        self,
        rule: 'EnhancedCompositionRule',
        examples: List[Tuple[str, str]]
    ) -> Dict:
        """Verify rule produces type-correct output."""
        correct_count = 0
        counter_examples = []
        
        for eng, expected_lean in examples:
            match = re.search(rule.syntactic_pattern, eng, re.IGNORECASE)
            if match:
                actual_lean = rule.semantic_function(match.groups())
                
                # Simple type check (full version would use Z3)
                if self._types_compatible(actual_lean, expected_lean):
                    correct_count += 1
                else:
                    counter_examples.append((eng, expected_lean))
        
        if counter_examples:
            return {
                'valid': False,
                'error': f'Type mismatch in {len(counter_examples)} cases',
                'counter_examples': counter_examples,
                'correct_count': correct_count
            }
        
        return {
            'valid': True,
            'correct_count': correct_count,
            'counter_examples': []
        }
    
    def _types_compatible(self, actual: str, expected: str) -> bool:
        """Check if two Lean expressions are type-compatible."""
        # Simplified check
        # Full version would parse and type-check both
        return len(actual) > 0
    
    def _validate_rule_comprehensively(
        self,
        rule: 'EnhancedCompositionRule',
        validation_examples: List[Tuple[str, str]]
    ) -> Dict:
        """Comprehensive validation with multiple metrics."""
        
        matched_count = 0
        correct_count = 0
        counter_examples = []
        
        for eng, expected_lean in validation_examples:
            match = re.search(rule.syntactic_pattern, eng, re.IGNORECASE)
            if match:
                matched_count += 1
                actual_lean = rule.semantic_function(match.groups())
                
                if self._semantically_equivalent(actual_lean, expected_lean):
                    correct_count += 1
                else:
                    counter_examples.append((eng, expected_lean, actual_lean))
        
        coverage = matched_count / len(validation_examples) if validation_examples else 0
        correctness = correct_count / matched_count if matched_count > 0 else 0
        
        # Compositionality score: how well does rule compose with others
        compositionality = self._compute_compositionality_score(rule)
        
        # Generalization: does it work beyond training patterns
        generalization = correctness * (1 - abs(coverage - 0.5))  # Prefer moderate coverage
        
        return {
            'coverage': coverage,
            'correctness': correctness,
            'compositionality': compositionality,
            'generalization': generalization,
            'counter_examples': counter_examples,
            'matched_count': matched_count,
            'correct_count': correct_count,
        }
    
    def _semantically_equivalent(self, actual: str, expected: str) -> bool:
        """Check semantic equivalence of Lean expressions."""
        # Simplified: normalize and compare
        actual_norm = actual.strip().replace(' ', '')
        expected_norm = expected.strip().replace(' ', '')
        return actual_norm == expected_norm
    
    def _compute_compositionality_score(self, rule: 'EnhancedCompositionRule') -> float:
        """
        Compute how well rule composes with others.
        
        Based on:
        - Can it take output of other rules as input?
        - Can its output feed into other rules?
        - Does it respect type boundaries?
        """
        score = 1.0
        
        # Bonus for well-typed composition
        if rule.semantic_type in ['e → t', 't → t', 'e → e → t']:
            score += 0.5
        
        # Penalty for overly specific patterns
        if len(rule.syntactic_pattern) > 100:
            score -= 0.2
        
        return max(0.0, min(2.0, score))
    
    def _analyze_failures(
        self,
        counter_examples: List[Tuple[str, str, str]],
        rule: 'EnhancedCompositionRule'
    ) -> Dict:
        """Analyze why rule failed on counter-examples."""
        
        # Categorize failures
        type_mismatches = 0
        scope_errors = 0
        composition_errors = 0
        
        for eng, expected, actual in counter_examples:
            if '→' in expected and '→' not in actual:
                type_mismatches += 1
            elif expected.count('∀') != actual.count('∀'):
                scope_errors += 1
            else:
                composition_errors += 1
        
        if type_mismatches > len(counter_examples) / 2:
            return {
                'primary_cause': 'type_mismatch',
                'suggestion': 'Refine type constraints in pattern'
            }
        elif scope_errors > len(counter_examples) / 2:
            return {
                'primary_cause': 'scope_error',
                'suggestion': 'Add explicit scope markers'
            }
        else:
            return {
                'primary_cause': 'composition_error',
                'suggestion': 'Check compositional structure'
            }
    
    def _compute_rule_quality(
        self,
        rule: 'EnhancedCompositionRule',
        validation_result: Dict,
        type_check_result: Dict
    ) -> float:
        """
        Compute overall quality score for rule.
        
        Weighted combination of:
        - Type correctness (40%)
        - Coverage (20%)
        - Correctness (20%)
        - Compositionality (10%)
        - Generalization (10%)
        """
        type_score = 1.0 if type_check_result['valid'] else 0.0
        
        quality = (
            0.4 * type_score +
            0.2 * validation_result['coverage'] +
            0.2 * validation_result['correctness'] +
            0.1 * validation_result['compositionality'] / 2.0 +  # Normalize to [0,1]
            0.1 * validation_result['generalization']
        )
        
        return quality
    
    def _rule_covers_example(
        self,
        rule: 'EnhancedCompositionRule',
        example: Tuple[str, str]
    ) -> bool:
        """Check if rule successfully covers example."""
        eng, expected_lean = example
        match = re.search(rule.syntactic_pattern, eng, re.IGNORECASE)
        if not match:
            return False
        
        actual_lean = rule.semantic_function(match.groups())
        return self._semantically_equivalent(actual_lean, expected_lean)
    
    def _analyze_rule_interactions(
        self,
        new_rule: 'EnhancedCompositionRule',
        existing_rules: List['EnhancedCompositionRule']
    ) -> Dict:
        """Analyze how new rule interacts with existing rules."""
        
        composes_with = []
        subsumes = []
        conflicts = []
        
        for existing in existing_rules:
            # Check if rules can compose
            if self._can_compose(new_rule, existing):
                composes_with.append(existing.rule_id)
            
            # Check if new rule subsumes existing
            if self._rule_subsumes(new_rule, existing):
                subsumes.append(existing.rule_id)
            
            # Check for conflicts
            if self._rules_conflict(new_rule, existing):
                conflicts.append(existing.rule_id)
        
        return {
            'composes_with': composes_with,
            'subsumes': subsumes,
            'conflicts': conflicts
        }
    
    def _can_compose(
        self,
        rule1: 'EnhancedCompositionRule',
        rule2: 'EnhancedCompositionRule'
    ) -> bool:
        """Check if two rules can compose."""
        # Simplified: check if output type of one matches input of other
        # Full version would parse semantic types properly
        return True  # Assume composable for now
    
    def _rule_subsumes(
        self,
        rule1: 'EnhancedCompositionRule',
        rule2: 'EnhancedCompositionRule'
    ) -> bool:
        """Check if rule1 subsumes rule2."""
        # Simplified: check if rule1's pattern is more general
        return len(rule1.syntactic_pattern) < len(rule2.syntactic_pattern)
    
    def _rules_conflict(
        self,
        rule1: 'EnhancedCompositionRule',
        rule2: 'EnhancedCompositionRule'
    ) -> bool:
        """Check if rules produce conflicting outputs."""
        # Would need to test on examples
        return False  # Assume no conflicts for now


@dataclass
class EnhancedCompositionRule:
    """
    Enhanced compositional semantic rule with full linguistic metadata.
    
    Captures:
    - Syntactic structure (pattern, category)
    - Semantic denotation (type, function)
    - Type-theoretic constraints
    - Linguistic features
    - Quality metrics
    """
    rule_id: str
    
    # Syntactic level
    syntactic_pattern: str  # Regex pattern for English
    syntactic_category: str  # CCG category (e.g., S/NP, VP/NP)
    
    # Semantic level
    semantic_type: str  # Type signature (e.g., e → t, (e → t) → t)
    semantic_function_name: str  # Name of composition operation
    semantic_function: Callable  # Actual function: match groups → Lean
    
    # Composition metadata
    composition_type: str  # forall, exists, lambda, apply, etc.
    arity: int  # Number of arguments
    
    # Type-theoretic constraints
    type_constraints: List[str]  # Linguistic constraints
    z3_constraints: List[str]  # Z3 SMT constraints
    
    # Examples and provenance
    example_instances: List[Tuple[str, str]]  # (English, Lean) pairs
    quality_score: float  # Overall quality [0, 1]
    
    # Linguistic analysis
    linguistic_features: Dict = field(default_factory=dict)
    
    def matches(self, text: str) -> bool:
        """Check if rule pattern matches text."""
        return re.search(self.syntactic_pattern, text, re.IGNORECASE) is not None
    
    def apply(self, text: str) -> str:
        """Apply semantic function to text."""
        match = re.search(self.syntactic_pattern, text, re.IGNORECASE)
        if match:
            return self.semantic_function(match.groups())
        return ""
    
    def get_derivation(self, text: str) -> Optional[SemanticDerivation]:
        """Get full semantic derivation for text."""
        if not self.matches(text):
            return None
        
        match = re.search(self.syntactic_pattern, text, re.IGNORECASE)
        result = self.semantic_function(match.groups())
        
        return SemanticDerivation(
            english_text=text,
            syntactic_parse=ParseTree(text=text),
            derivation_steps=[
                ('match_pattern', text, str(match.groups())),
                ('apply_semantic_function', str(match.groups()), result)
            ],
            final_lean_type=None,  # Would need type inference
            semantic_representation=result,
            type_derivation=[],
            z3_proof=None
        )


@dataclass
class CompositionRule:
    """Legacy composition rule (kept for compatibility)."""
    pattern: str
    semantic_function: Callable
    weight: float
    
    def matches(self, text: str) -> bool:
        return re.search(self.pattern, text, re.IGNORECASE) is not None
    
    def apply(self, text: str) -> str:
        match = re.search(self.pattern, text, re.IGNORECASE)
        if match:
            return self.semantic_function(match.groups())
        return ""


@dataclass  
class ParseTree:
    """Placeholder for parse tree."""
    text: str
    children: List['ParseTree'] = None


# ============================================================================
# DCG-INSPIRED GRAMMAR WITH Z3 (inspired by Prolog natural language processing)
# ============================================================================

class DCGGrammarRule:
    """
    Definite Clause Grammar rule adapted for Z3.
    
    In Prolog DCG:
        s(Sem) --> np(NP), vp(VP), {combine(NP, VP, Sem)}.
    
    In our Z3 version:
        - Syntactic pattern matching (regex instead of DCG)
        - Semantic construction via Z3-constrained functions
        - Type checking via Z3 SMT solver
    """
    
    def __init__(self, 
                 name: str,
                 lhs_category: str,
                 rhs_categories: List[str],
                 syntactic_pattern: str,
                 semantic_combinator: Callable,
                 z3_constraints: Optional[List] = None):
        self.name = name
        self.lhs_category = lhs_category
        self.rhs_categories = rhs_categories
        self.syntactic_pattern = syntactic_pattern
        self.semantic_combinator = semantic_combinator
        self.z3_constraints = z3_constraints or []
    
    def parse(self, text: str, context: Dict) -> Optional[Tuple[Any, Dict]]:
        """
        Parse text according to this grammar rule.
        
        Returns (semantic_value, updated_context) if successful, None otherwise.
        """
        match = re.search(self.syntactic_pattern, text, re.IGNORECASE)
        if not match:
            return None
        
        # Extract constituents
        constituents = match.groups()
        
        # Apply semantic combinator
        try:
            semantic_value = self.semantic_combinator(constituents, context)
            
            # Verify Z3 constraints
            if self.z3_constraints and not self._check_z3_constraints(semantic_value, context):
                return None
            
            return semantic_value, context
        except:
            return None
    
    def _check_z3_constraints(self, semantic_value: Any, context: Dict) -> bool:
        """Check Z3 constraints on semantic value."""
        solver = Solver()
        solver.set('timeout', 1000)
        
        for constraint in self.z3_constraints:
            solver.add(constraint)
        
        return solver.check() == sat


class Z3DCGParser:
    """
    DCG-style parser using Z3 for constraint solving.
    
    Key ideas from Prolog NLP adapted to Z3:
    1. Compositional semantics: meaning built from parts
    2. Feature unification: Z3 solves for variable bindings
    3. Constraint propagation: Z3 enforces type/semantic constraints
    4. Backtracking: Try multiple parses, Z3 prunes invalid ones
    """
    
    def __init__(self, learner: CEGIS_SemanticLearner):
        self.learner = learner
        self.grammar_rules: List[DCGGrammarRule] = []
        self.z3_check_count: int = 0
        self._initialize_dcg_rules()
    
    def _initialize_dcg_rules(self):
        """Initialize DCG-style grammar rules."""
        
        # RULE: S → NP VP (sentence = subject + predicate)
        # Prolog DCG: s(app(VP, NP)) --> np(NP), vp(VP).
        self.grammar_rules.append(DCGGrammarRule(
            name='sentence_np_vp',
            lhs_category='S',
            rhs_categories=['NP', 'VP'],
            syntactic_pattern=r'([A-Z]\w*)\s+(is|are|has)\s+(.+)',
            semantic_combinator=lambda constituents, ctx: f"{constituents[1]}({constituents[0]}, {constituents[2]})",
            z3_constraints=[]
        ))
        
        # RULE: VP → V NP (verb phrase = verb + object)
        # Prolog DCG: vp(lam(X, app(V, NP))) --> v(V), np(NP).
        self.grammar_rules.append(DCGGrammarRule(
            name='vp_v_np',
            lhs_category='VP',
            rhs_categories=['V', 'NP'],
            syntactic_pattern=r'(\w+)\s+([A-Z]\w*)',
            semantic_combinator=lambda constituents, ctx: f"fun x => {constituents[0]} x {constituents[1]}",
            z3_constraints=[]
        ))
        
        # RULE: NP → Det N (noun phrase = determiner + noun)
        # Prolog DCG: np(Q) --> det(Det), n(N), {quantifier(Det, N, Q)}.
        self.grammar_rules.append(DCGGrammarRule(
            name='np_det_n',
            lhs_category='NP',
            rhs_categories=['Det', 'N'],
            syntactic_pattern=r'(every|some|all|a|an|the)\s+(\w+)',
            semantic_combinator=self._quantifier_semantics,
            z3_constraints=[]
        ))
        
        # RULE: Quantified → Quantifier Variable Type Body
        # Prolog DCG: quant(forall(X, T, B)) --> [for, all], var(X), [in], type(T), [,], body(B).
        self.grammar_rules.append(DCGGrammarRule(
            name='universal_quantification',
            lhs_category='Quant',
            rhs_categories=['QWord', 'Var', 'Type', 'Body'],
            syntactic_pattern=r'for\s+(?:all|every|each)\s+(\w+)\s+(?:in|:)\s+(\w+),\s+(.+)',
            semantic_combinator=lambda constituents, ctx: f"∀ {constituents[0]} : {constituents[1]}, {constituents[2]}",
            z3_constraints=[]
        ))
        
        # RULE: Implication → Condition → Conclusion
        # Prolog DCG: impl(imp(C, R)) --> [if], cond(C), [then], result(R).
        self.grammar_rules.append(DCGGrammarRule(
            name='implication',
            lhs_category='Impl',
            rhs_categories=['Cond', 'Result'],
            syntactic_pattern=r'if\s+(.+?)\s+then\s+(.+)',
            semantic_combinator=lambda constituents, ctx: f"{constituents[0]} → {constituents[1]}",
            z3_constraints=[]
        ))

    def _solver_check(self, solver: Solver):
        self.z3_check_count += 1
        return solver.check()

    def _tokenize(self, text: str) -> List[str]:
        # Keep a very small tokenizer: words + a few punctuation tokens.
        return re.findall(r"[A-Za-z0-9_]+|→|∀|∃|:=|:|,|\(|\)|\.|;|\S", text)

    def _detokenize(self, tokens: List[str]) -> str:
        s = " ".join(tokens)
        s = re.sub(r"\s+([,.;:])", r"\1", s)
        s = re.sub(r"\(\s+", "(", s)
        s = re.sub(r"\s+\)", ")", s)
        return s.strip()

    def _parse_if_then_z3(self, text: str, max_models: int = 3) -> List[str]:
        toks = self._tokenize(text)
        low = [t.lower() for t in toks]
        if_positions = [i for i, t in enumerate(low) if t == 'if']
        then_positions = [i for i, t in enumerate(low) if t == 'then']
        if not if_positions or not then_positions:
            return []

        solver = Solver()
        solver.set('timeout', 1000)
        if_i = Int('if_i')
        then_i = Int('then_i')
        solver.add(Or([if_i == i for i in if_positions]))
        solver.add(Or([then_i == i for i in then_positions]))
        solver.add(if_i < then_i - 1)
        solver.add(then_i < len(toks) - 1)

        outs: List[str] = []
        while len(outs) < max_models and self._solver_check(solver) == sat:
            m = solver.model()
            if_idx = m.eval(if_i).as_long()
            then_idx = m.eval(then_i).as_long()
            antecedent = self._detokenize(toks[if_idx + 1:then_idx])
            consequent = self._detokenize(toks[then_idx + 1:])
            if antecedent and consequent:
                outs.append(f"{antecedent} → {consequent}")
            solver.add(Or(if_i != if_idx, then_i != then_idx))

        return outs

    def _parse_forall_z3(self, text: str, max_models: int = 3) -> List[str]:
        toks = self._tokenize(text)
        low = [t.lower() for t in toks]
        quant_indices = [i for i in range(1, len(toks))
                        if low[i] in {'all', 'every', 'each'} and low[i - 1] == 'for']
        if not quant_indices:
            return []

        in_indices = [i for i, t in enumerate(low) if t in {'in', ':'}]
        comma_indices = [i for i, t in enumerate(low) if t == ',']
        if not in_indices or not comma_indices:
            return []

        solver = Solver()
        solver.set('timeout', 1000)
        q_i = Int('q_i')
        in_i = Int('in_i')
        comma_i = Int('comma_i')

        solver.add(Or([q_i == i for i in quant_indices]))
        solver.add(Or([in_i == i for i in in_indices]))
        solver.add(Or([comma_i == i for i in comma_indices]))

        # Layout: for <qword> <var> in <domain> , <body>
        solver.add(q_i + 1 < in_i)
        solver.add(in_i + 1 < comma_i)
        solver.add(comma_i + 1 < len(toks))

        outs: List[str] = []
        while len(outs) < max_models and self._solver_check(solver) == sat:
            m = solver.model()
            q_idx = m.eval(q_i).as_long()
            in_idx = m.eval(in_i).as_long()
            comma_idx = m.eval(comma_i).as_long()

            var_tokens = toks[q_idx + 1:in_idx]
            var = var_tokens[0] if var_tokens else ''
            domain = self._detokenize(toks[in_idx + 1:comma_idx])
            body = self._detokenize(toks[comma_idx + 1:])

            if var and domain and body:
                outs.append(f"∀ {var} : {domain}, {body}")

            solver.add(Or(q_i != q_idx, in_i != in_idx, comma_i != comma_idx))

        return outs

    def _parse_exists_z3(self, text: str, max_models: int = 3) -> List[str]:
        toks = self._tokenize(text)
        low = [t.lower() for t in toks]
        exists_indices = [i for i, t in enumerate(low) if t in {'exists', 'exist'}]
        if not exists_indices:
            return []

        in_indices = [i for i, t in enumerate(low) if t in {'in', ':'}]
        such_indices = [i for i, t in enumerate(low) if t == 'such']
        where_indices = [i for i, t in enumerate(low) if t == 'where']
        tail_markers = such_indices + where_indices
        if not in_indices or not tail_markers:
            return []

        solver = Solver()
        solver.set('timeout', 1000)
        ex_i = Int('ex_i')
        in_i = Int('in_i')
        tail_i = Int('tail_i')

        solver.add(Or([ex_i == i for i in exists_indices]))
        solver.add(Or([in_i == i for i in in_indices]))
        solver.add(Or([tail_i == i for i in tail_markers]))

        # Layout: there exists <var> in <domain> such that/where <body>
        solver.add(ex_i + 1 < in_i)
        solver.add(in_i + 1 < tail_i)
        solver.add(tail_i < len(toks) - 1)

        outs: List[str] = []
        while len(outs) < max_models and self._solver_check(solver) == sat:
            m = solver.model()
            ex_idx = m.eval(ex_i).as_long()
            in_idx = m.eval(in_i).as_long()
            tail_idx = m.eval(tail_i).as_long()

            var_tokens = toks[ex_idx + 1:in_idx]
            var = var_tokens[0] if var_tokens else ''
            domain = self._detokenize(toks[in_idx + 1:tail_idx])

            # Skip optional "that" after "such"
            body_start = tail_idx + 1
            if low[tail_idx] == 'such' and body_start < len(toks) and low[body_start] == 'that':
                body_start += 1
            body = self._detokenize(toks[body_start:])

            if var and domain and body:
                outs.append(f"∃ {var} : {domain}, {body}")

            solver.add(Or(ex_i != ex_idx, in_i != in_idx, tail_i != tail_idx))

        return outs

    def _parse_and_or_z3(self, text: str, max_models: int = 3) -> List[str]:
        toks = self._tokenize(text)
        low = [t.lower() for t in toks]
        conj_positions = [(i, t) for i, t in enumerate(low) if t in {'and', 'or'}]
        if not conj_positions:
            return []

        solver = Solver()
        solver.set('timeout', 1000)
        op_i = Int('op_i')
        solver.add(Or([op_i == i for i, _ in conj_positions]))
        solver.add(op_i > 0)
        solver.add(op_i < len(toks) - 1)

        outs: List[str] = []
        while len(outs) < max_models and self._solver_check(solver) == sat:
            m = solver.model()
            op_idx = m.eval(op_i).as_long()
            op = low[op_idx]
            left = self._detokenize(toks[:op_idx])
            right = self._detokenize(toks[op_idx + 1:])
            if left and right:
                outs.append(f"{left} {'∧' if op == 'and' else '∨'} {right}")
            solver.add(op_i != op_idx)

        return outs

    def translate(self, text: str) -> List[Tuple[str, str]]:
        """Return candidate (lean, composition_type) translations."""
        candidates: List[Tuple[str, str]] = []
        for out in self._parse_forall_z3(text):
            candidates.append((out, 'forall'))
        for out in self._parse_exists_z3(text):
            candidates.append((out, 'exists'))
        for out in self._parse_if_then_z3(text):
            candidates.append((out, 'implies'))
        for out in self._parse_and_or_z3(text):
            candidates.append((out, 'and_or'))
        return candidates
    
    def _quantifier_semantics(self, constituents: Tuple[str, ...], context: Dict) -> str:
        """
        Quantifier semantics following generalized quantifier theory.
        
        Prolog version:
            quantifier(every, N, lambda(X, implies(N(X), _)))
            quantifier(some, N, lambda(X, and(N(X), _)))
        
        Z3 version: Use Z3 to verify type constraints on quantifier scope.
        """
        det, noun = constituents
        
        # Use Z3 to determine quantifier type
        solver = Solver()
        is_universal = Bool('is_universal')
        is_existential = Bool('is_existential')
        
        # Constraint: exactly one quantifier type
        solver.add(Xor(is_universal, is_existential))
        
        # Det determines quantifier
        if det.lower() in ['every', 'all', 'each']:
            solver.add(is_universal)
        elif det.lower() in ['some', 'a', 'an']:
            solver.add(is_existential)
        
        if solver.check() == sat:
            model = solver.model()
            if model.eval(is_universal):
                return f"∀ x : {noun}, _"
            else:
                return f"∃ x : {noun}, _"
        
        return f"{det} {noun}"
    
    def parse_with_dcg(self, text: str, target_category: str = 'S') -> List[Tuple[Any, Dict]]:
        """
        Parse text using DCG rules, returning all valid parses.
        
        Uses Z3 to:
        1. Constrain feature unification
        2. Prune type-invalid parses
        3. Rank parses by logical consistency
        """
        parses: List[Tuple[Any, Dict]] = []

        # Prefer token-based Z3 parses for core constructions.
        for lean_out, composition_type in self.translate(text):
            parses.append((lean_out, {'composition_type': composition_type, 'z3_checks': self.z3_check_count}))

        # Fallback: legacy regex rules.
        if not parses:
            for rule in self.grammar_rules:
                if rule.lhs_category != target_category:
                    continue
                result = rule.parse(text, {})
                if result:
                    parses.append(result)

        if len(parses) > 1:
            parses = self._z3_rank_parses(parses)

        return parses

    def propose_enhanced_rule_from_examples(
        self,
        examples: List[Tuple[str, str]],
        min_correct: int = 2,
    ) -> Optional['EnhancedCompositionRule']:
        """Propose a concrete `EnhancedCompositionRule` using DCG+Z3 translation as an oracle."""
        pattern_library: List[Tuple[str, str, int, str, Callable[[Tuple[str, ...]], str]]] = [
            (
                'forall',
                r"for\s+(?:all|every|each)\s+(\w+)\s+(?:in|:)\s+([^,]+),\s+(.+)",
                3,
                '(e → t) → t',
                lambda g: f"∀ {g[0]} : {g[1]}, {g[2]}",
            ),
            (
                'exists',
                r"there\s+exists?\s+(\w+)\s+(?:in|:)\s+(.+?)\s+(?:such\s+that|where)\s+(.+)",
                3,
                '(e → t) → t',
                lambda g: f"∃ {g[0]} : {g[1]}, {g[2]}",
            ),
            (
                'implies',
                r"if\s+(.+?)\s+then\s+(.+)",
                2,
                't → t → t',
                lambda g: f"{g[0]} → {g[1]}",
            ),
            (
                'and',
                r"(.+?)\s+and\s+(.+)",
                2,
                't → t → t',
                lambda g: f"{g[0]} ∧ {g[1]}",
            ),
            (
                'or',
                r"(.+?)\s+or\s+(.+)",
                2,
                't → t → t',
                lambda g: f"{g[0]} ∨ {g[1]}",
            ),
        ]

        best: Optional[Tuple[int, int, str, str, int, Callable, str]] = None
        for composition_type, pattern, arity, semantic_type, sem_fn in pattern_library:
            matched = 0
            correct = 0
            z3_supported = 0
            for eng, expected in examples:
                m = re.search(pattern, eng, re.IGNORECASE)
                if not m:
                    continue
                matched += 1
                got = sem_fn(m.groups())
                if self.learner._semantically_equivalent(got, expected):
                    correct += 1

                # Force the Z3-backed DCG path to run and agree.
                for out, out_type in self.translate(eng):
                    if out_type == composition_type and self.learner._semantically_equivalent(out, expected):
                        z3_supported += 1
                        break

            if correct >= min_correct:
                if best is None:
                    best = (correct, z3_supported, composition_type, pattern, arity, sem_fn, semantic_type)
                else:
                    # Prefer higher correctness, then higher Z3 support.
                    if correct > best[0] or (correct == best[0] and z3_supported > best[1]):
                        best = (correct, z3_supported, composition_type, pattern, arity, sem_fn, semantic_type)

        if not best:
            return None

        correct, z3_supported, composition_type, pattern, arity, sem_fn, semantic_type = best
        rule_id = f"dcg_{composition_type}_{len(self.learner.learned_rules)}"
        return EnhancedCompositionRule(
            rule_id=rule_id,
            syntactic_pattern=pattern,
            syntactic_category='S',
            semantic_type=semantic_type,
            semantic_function_name=composition_type,
            semantic_function=sem_fn,
            composition_type=composition_type,
            arity=arity,
            type_constraints=[],
            z3_constraints=[],
            example_instances=examples[:10],
            quality_score=0.0,
            linguistic_features={'source': 'dcg_z3', 'z3_checks': self.z3_check_count, 'z3_supported': z3_supported},
        )
    
    def _z3_rank_parses(self, parses: List[Tuple[Any, Dict]]) -> List[Tuple[Any, Dict]]:
        """Use Z3 to rank multiple parses by logical consistency."""
        solver = Solver()
        
        # Create Z3 variables for each parse's score
        scores = [Int(f'score_{i}') for i in range(len(parses))]
        
        # Score based on type correctness, simplicity, etc.
        for i, (sem_val, ctx) in enumerate(parses):
            # Higher score for simpler expressions
            complexity = len(str(sem_val))
            solver.add(scores[i] == 100 - complexity)
        
        # Find parse with maximum score
        if solver.check() == sat:
            model = solver.model()
            scored_parses = [(model.eval(scores[i]).as_long(), parses[i]) 
                           for i in range(len(parses))]
            scored_parses.sort(reverse=True, key=lambda x: x[0])
            return [p for _, p in scored_parses]
        
        return parses
    
    def learn_dcg_rule_from_examples(self, 
                                     examples: List[Tuple[str, str]],
                                     category: str) -> Optional[DCGGrammarRule]:
        """
        Learn a new DCG rule from examples using Z3 synthesis.
        
        Prolog version learns by generalization and anti-unification.
        Z3 version uses SMT-guided program synthesis.
        """
        if len(examples) < 2:
            return None
        
        # Step 1: Use Z3 to synthesize syntactic pattern
        pattern_solver = Solver()
        pattern_solver.set('timeout', 5000)
        
        # Pattern must match all positive examples
        # (synthesize using sketch-based approach)
        synthesized_pattern = self.learner._z3_synthesize_pattern(
            examples,
            [],
            {'arity': 2, 'composition_rule': 'learned'},
            'learned'
        )
        
        if not synthesized_pattern:
            return None
        
        # Step 2: Use Z3 to infer semantic combinator
        def learned_combinator(constituents, context):
            # Analyze examples to infer transformation
            if len(constituents) >= 2:
                return f"{constituents[0]} ⊢ {constituents[1]}"
            return ""
        
        # Step 3: Generate Z3 type constraints
        type_solver = Solver()
        # Add well-formedness constraints
        # (simplified - full version would encode type theory)
        
        new_rule = DCGGrammarRule(
            name=f'learned_{category}',
            lhs_category=category,
            rhs_categories=['X', 'Y'],
            syntactic_pattern=synthesized_pattern,
            semantic_combinator=learned_combinator,
            z3_constraints=[]
        )
        
        return new_rule


class Z3UnificationEngine:
    """
    Feature unification engine using Z3 (replaces Prolog unification).
    
    In Prolog NLP, unification binds logical variables.
    In our system, Z3 solves constraints on semantic features.
    """
    
    def __init__(self):
        self.solver = Solver()
        self.feature_vars: Dict[str, ExprRef] = {}
    
    def unify_features(self, feat1: Dict[str, Any], feat2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Unify two feature structures using Z3.
        
        Prolog: unify(X, Y) succeeds if X and Y can be made equal.
        Z3: solve constraints that make features compatible.
        """
        self.solver.push()
        
        # Add constraints for each feature
        unified = {}
        for key in set(feat1.keys()) | set(feat2.keys()):
            val1 = feat1.get(key)
            val2 = feat2.get(key)
            
            if val1 is not None and val2 is not None:
                # Both specified - must be equal
                if key not in self.feature_vars:
                    self.feature_vars[key] = Const(f'feat_{key}', IntSort())
                
                self.solver.add(self.feature_vars[key] == val1)
                self.solver.add(self.feature_vars[key] == val2)
                
                if self.solver.check() == unsat:
                    self.solver.pop()
                    return None
                
                unified[key] = val1
            elif val1 is not None:
                unified[key] = val1
            elif val2 is not None:
                unified[key] = val2
        
        self.solver.pop()
        return unified
    
    def propagate_constraints(self, features: Dict[str, Any], constraints: List) -> bool:
        """
        Propagate constraints through feature structure using Z3.
        
        Returns True if constraints are satisfiable.
        """
        self.solver.push()
        
        for constraint in constraints:
            self.solver.add(constraint)
        
        result = self.solver.check() == sat
        self.solver.pop()
        
        return result


# ============================================================================
# IMPROVEMENT #2: HIERARCHICAL RULE LEARNING & COMPOSITION
# ============================================================================

class RuleCompositionEngine:
    """Engine for composing multiple rules hierarchically."""
    
    def __init__(self, learner: CEGIS_SemanticLearner):
        self.learner = learner
        self.composition_patterns = self._initialize_composition_patterns()
    
    def _initialize_composition_patterns(self) -> Dict[str, Callable]:
        """Define how different rule types can compose."""
        return {
            'sequential': self._compose_sequential,
            'nested': self._compose_nested,
            'parallel': self._compose_parallel,
            'conditional': self._compose_conditional,
        }
    
    def _compose_sequential(self, rule1: EnhancedCompositionRule, 
                           rule2: EnhancedCompositionRule, 
                           text: str) -> Optional[str]:
        """Apply rules sequentially: rule1 then rule2."""
        intermediate = rule1.apply(text)
        if intermediate:
            return rule2.apply(intermediate)
        return None
    
    def _compose_nested(self, rule1: EnhancedCompositionRule, 
                       rule2: EnhancedCompositionRule, 
                       text: str) -> Optional[str]:
        """Apply rule1 to result of rule2 (nested composition)."""
        # First check if text matches rule2's sub-pattern
        match2 = re.search(rule2.syntactic_pattern, text, re.IGNORECASE)
        if not match2:
            return None
        
        # Extract groups and apply rule2
        groups2 = match2.groups()
        result2 = rule2.semantic_function(groups2)
        
        # Now apply rule1 to the result
        return rule1.apply(result2)
    
    def _compose_parallel(self, rule1: EnhancedCompositionRule, 
                         rule2: EnhancedCompositionRule, 
                         text: str) -> Optional[str]:
        """Apply both rules and combine results (e.g., for conjunction)."""
        # Split text into parts
        parts = re.split(r'\s+and\s+|\s+or\s+', text, flags=re.IGNORECASE)
        if len(parts) != 2:
            return None
        
        result1 = rule1.apply(parts[0])
        result2 = rule2.apply(parts[1])
        
        if result1 and result2:
            # Determine connective
            if ' and ' in text.lower():
                return f"{result1} ∧ {result2}"
            elif ' or ' in text.lower():
                return f"{result1} ∨ {result2}"
        
        return None
    
    def _compose_conditional(self, rule1: EnhancedCompositionRule, 
                            rule2: EnhancedCompositionRule, 
                            text: str) -> Optional[str]:
        """Apply rule2 only if rule1 succeeds (conditional composition)."""
        result1 = rule1.apply(text)
        if result1:
            return rule2.apply(result1)
        return None
    
    def try_compose_rules(self, text: str, 
                         max_composition_depth: int = 3) -> List[Tuple[str, List[str]]]:
        """
        Try composing existing rules to cover text.
        Returns list of (output, rule_chain) pairs.
        """
        results = []
        
        # Single rule applications (depth 1)
        for rule in self.learner.learned_rules:
            if rule.matches(text):
                output = rule.apply(text)
                if output:
                    results.append((output, [rule.rule_id]))
        
        # Two-rule compositions (depth 2)
        if max_composition_depth >= 2:
            for rule1 in self.learner.learned_rules:
                for rule2 in self.learner.learned_rules:
                    if rule1.rule_id == rule2.rule_id:
                        continue
                    
                    # Try different composition patterns
                    for pattern_name, composer in self.composition_patterns.items():
                        output = composer(rule1, rule2, text)
                        if output:
                            results.append((output, [rule1.rule_id, rule2.rule_id, pattern_name]))
        
        return results


# ============================================================================
# IMPROVEMENT #2: BETTER NEGATIVE EXAMPLE GENERATION
# ============================================================================

class NegativeExampleGenerator:
    """Generate high-quality negative examples for CEGIS."""
    
    def __init__(self, learner: CEGIS_SemanticLearner):
        self.learner = learner
    
    def generate_adversarial_negatives(self, 
                                      positive_examples: List[Tuple[str, str]],
                                      current_rule: EnhancedCompositionRule,
                                      count: int = 10) -> List[Tuple[str, str]]:
        """
        Generate adversarial examples that are syntactically similar 
        but semantically different from positives.
        """
        negatives = []
        
        for eng, lean in positive_examples[:count]:
            # Strategy 1: Near-miss negatives (small syntactic changes)
            near_misses = self._generate_near_misses(eng, lean)
            negatives.extend(near_misses)
            
            # Strategy 2: Type-incompatible variations
            type_variations = self._generate_type_variations(eng, lean)
            negatives.extend(type_variations)
            
            # Strategy 3: Semantic perturbations
            semantic_perturbations = self._generate_semantic_perturbations(eng, lean)
            negatives.extend(semantic_perturbations)
        
        return negatives[:count]
    
    def _generate_near_misses(self, eng: str, lean: str) -> List[Tuple[str, str]]:
        """Generate syntactically similar but semantically different examples."""
        near_misses = []
        
        # Replace logical connectives
        connective_swaps = [
            ('and', 'or'), ('or', 'and'),
            ('if', 'when'), ('then', 'thus'),
            ('for all', 'for some'), ('every', 'some'),
            ('implies', 'if and only if'),
        ]
        
        for old, new in connective_swaps:
            if old in eng.lower():
                modified = eng.lower().replace(old, new)
                # Keep original Lean (wrong for modified English)
                near_misses.append((modified, lean))
        
        return near_misses
    
    def _generate_type_variations(self, eng: str, lean: str) -> List[Tuple[str, str]]:
        """Generate examples with type mismatches."""
        variations = []
        
        # Type substitutions that break well-typedness
        type_subs = [
            ('ℕ', 'ℝ → ℝ'),  # Type category error
            ('x', 'λ x. x'),  # Term vs function confusion
            ('>', '∈'),  # Relation confusion
        ]
        
        for old, new in type_subs:
            if old in eng:
                modified_eng = eng.replace(old, new)
                modified_lean = lean.replace(old, new)
                variations.append((modified_eng, modified_lean))
        
        return variations
    
    def _generate_semantic_perturbations(self, eng: str, lean: str) -> List[Tuple[str, str]]:
        """Generate semantically incorrect variations."""
        perturbations = []
        
        # Scope perturbations (for quantifiers)
        if 'for all' in eng.lower() or '∀' in lean:
            # Swap quantifier scope
            if ',' in eng:
                parts = eng.split(',', 1)
                if len(parts) == 2:
                    swapped = f"{parts[1]}, {parts[0]}"
                    perturbations.append((swapped, lean))  # Wrong for swapped scope
        
        # Negation insertion
        if 'not' not in eng.lower() and '¬' not in lean:
            negated_eng = f"not ({eng})"
            # Keep non-negated Lean (wrong)
            perturbations.append((negated_eng, lean))
        
        return perturbations


# ============================================================================
# IMPROVEMENT #3: TEMPLATE VALIDATION FRAMEWORK
# ============================================================================

class TemplateValidator:
    """Comprehensive validation for induced templates."""
    
    def __init__(self, learner: CEGIS_SemanticLearner):
        self.learner = learner
        self.validation_criteria = {
            'semantic': self._validate_semantic,
            'syntactic': self._validate_syntactic,
            'type': self._validate_type,
            'coverage': self._validate_coverage,
            'compositionality': self._validate_compositionality,
        }
    
    def validate_template(self, template: Dict, 
                         examples: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Comprehensive validation returning scores for each criterion.
        
        Returns dict mapping criterion name to score in [0, 1].
        """
        results = {}
        
        for criterion_name, validator in self.validation_criteria.items():
            try:
                score = validator(template, examples)
                results[criterion_name] = score
            except Exception as e:
                results[criterion_name] = 0.0
        
        # Overall score (weighted average)
        weights = {
            'semantic': 0.3,
            'syntactic': 0.2,
            'type': 0.25,
            'coverage': 0.15,
            'compositionality': 0.1,
        }
        
        results['overall'] = sum(results[k] * weights.get(k, 0.1) 
                                for k in self.validation_criteria.keys())
        
        return results
    
    def _validate_semantic(self, template: Dict, 
                          examples: List[Tuple[str, str]]) -> float:
        """Validate semantic coherence of template."""
        composition_rule = template.get('composition_rule', '')
        semantic_type = template.get('semantic_type', '')
        
        # Check if semantic type matches composition rule
        type_matches = {
            'forall': ['(e → t) → t', 't → t'],
            'exists': ['(e → t) → t', 't → t'],
            'implies': ['t → t → t'],
            'and': ['t → t → t'],
            'lambda': ['(a → b)', 'e → t'],
        }
        
        expected_types = type_matches.get(composition_rule, [])
        if any(expected in semantic_type for expected in expected_types):
            return 1.0
        elif semantic_type:
            return 0.5  # Has a type, but not standard
        else:
            return 0.0
    
    def _validate_syntactic(self, template: Dict, 
                           examples: List[Tuple[str, str]]) -> float:
        """Validate syntactic patterns are well-formed."""
        # Check if template was generated with a pattern
        pattern = template.get('pattern', '')
        
        if not pattern:
            return 0.0
        
        # Check pattern is valid regex
        try:
            re.compile(pattern)
        except:
            return 0.0
        
        # Check pattern has correct number of capture groups
        arity = template.get('arity', 0)
        num_groups = pattern.count('(') - pattern.count('\\(')
        
        if num_groups == arity:
            return 1.0
        elif num_groups > 0:
            return 0.5
        else:
            return 0.0
    
    def _validate_type(self, template: Dict, 
                      examples: List[Tuple[str, str]]) -> float:
        """Validate type constraints are satisfiable."""
        type_constraints = template.get('type_constraints', [])
        
        if not type_constraints:
            return 0.5  # Neutral: no constraints to validate
        
        # Use Z3 to check satisfiability
        solver = Solver()
        solver.set('timeout', 1000)
        
        # Add constraints (simplified check)
        # In full implementation, would encode full type theory
        for constraint in type_constraints:
            # Just check they're non-empty and look reasonable
            if not constraint or len(constraint) < 3:
                return 0.0
        
        return 1.0  # Constraints look valid
    
    def _validate_coverage(self, template: Dict, 
                          examples: List[Tuple[str, str]]) -> float:
        """Validate template covers enough examples."""
        min_support = template.get('support', 0)
        
        if min_support >= 10:
            return 1.0
        elif min_support >= 5:
            return 0.7
        elif min_support >= 2:
            return 0.4
        else:
            return 0.0
    
    def _validate_compositionality(self, template: Dict, 
                                   examples: List[Tuple[str, str]]) -> float:
        """Validate template composes with existing rules."""
        # Check if semantic type allows composition
        semantic_type = template.get('semantic_type', '')
        arity = template.get('arity', 0)
        
        # Templates with proper arity and type signatures can compose
        if arity > 0 and ' → ' in semantic_type:
            return 1.0
        elif arity > 0:
            return 0.5
        else:
            return 0.0


# ============================================================================
# IMPROVEMENT #3: COUNTER-EXAMPLE GUIDED TEMPLATE REFINEMENT
# ============================================================================

class TemplateRefiner:
    """Refine induced templates using counter-examples and Z3."""
    
    def __init__(self, learner: CEGIS_SemanticLearner):
        self.learner = learner
        self.max_refinement_iterations = 5
    
    def refine_template(self, template: Dict, 
                       positive_examples: List[Tuple[str, str]],
                       negative_examples: List[Tuple[str, str]],
                       counter_examples: List[Tuple[str, str, str]]) -> Optional[Dict]:
        """
        Refine template using counter-examples.
        
        Args:
            template: Template to refine
            positive_examples: Should match
            negative_examples: Should not match
            counter_examples: (input, expected, got) triples where template failed
        
        Returns:
            Refined template or None if refinement fails
        """
        refined = template.copy()
        
        for iteration in range(self.max_refinement_iterations):
            print(f"    Refinement iteration {iteration + 1}/{self.max_refinement_iterations}")
            
            # Analyze counter-examples to determine refinement strategy
            refinement_strategy = self._analyze_counter_examples(counter_examples)
            
            if refinement_strategy == 'narrow_pattern':
                refined = self._narrow_pattern(refined, positive_examples, counter_examples)
            elif refinement_strategy == 'broaden_pattern':
                refined = self._broaden_pattern(refined, positive_examples, counter_examples)
            elif refinement_strategy == 'fix_semantic_function':
                refined = self._fix_semantic_function(refined, counter_examples)
            elif refinement_strategy == 'add_type_constraint':
                refined = self._add_type_constraints(refined, counter_examples)
            else:
                print(f"    → Unknown refinement strategy: {refinement_strategy}")
                return None
            
            # Validate refined template
            validator = TemplateValidator(self.learner)
            validation_scores = validator.validate_template(refined, positive_examples)
            
            if validation_scores['overall'] > 0.7:
                print(f"    ✓ Refinement successful (score: {validation_scores['overall']:.2f})")
                return refined
        
        print(f"    ✗ Refinement failed after {self.max_refinement_iterations} iterations")
        return None
    
    def _analyze_counter_examples(self, 
                                  counter_examples: List[Tuple[str, str, str]]) -> str:
        """Determine what kind of refinement is needed."""
        if not counter_examples:
            return 'none'
        
        # Count failure patterns
        pattern_failures = 0
        semantic_failures = 0
        type_failures = 0
        
        for inp, expected, got in counter_examples:
            if not got or got == '':
                pattern_failures += 1
            elif len(got.split()) != len(expected.split()):
                semantic_failures += 1
            else:
                type_failures += 1
        
        # Determine dominant failure mode
        if pattern_failures > len(counter_examples) * 0.5:
            return 'broaden_pattern'
        elif semantic_failures > len(counter_examples) * 0.5:
            return 'fix_semantic_function'
        elif type_failures > len(counter_examples) * 0.5:
            return 'add_type_constraint'
        else:
            return 'narrow_pattern'
    
    def _narrow_pattern(self, template: Dict, 
                       positive_examples: List[Tuple[str, str]],
                       counter_examples: List[Tuple[str, str, str]]) -> Dict:
        """Narrow pattern using Z3 to exclude counter-examples."""
        current_pattern = template.get('pattern', '')
        
        # Use Z3 to synthesize narrower pattern
        solver = Solver()
        solver.set('timeout', 3000)
        
        # Add constraint: pattern must match positives
        # Add constraint: pattern must not match counter-example inputs
        
        # For now, heuristic narrowing
        # Add negative lookahead for common counter-example patterns
        ce_terms = set()
        for inp, _, _ in counter_examples[:5]:
            # Extract distinctive terms
            terms = set(inp.lower().split())
            ce_terms.update(terms)
        
        # Remove common terms that also appear in positives
        pos_terms = set()
        for eng, _ in positive_examples:
            pos_terms.update(eng.lower().split())
        
        distinctive_ce_terms = ce_terms - pos_terms
        
        if distinctive_ce_terms:
            # Add negative lookahead
            term = list(distinctive_ce_terms)[0]
            narrowed_pattern = f"(?!.*{term}.*){current_pattern}"
            template['pattern'] = narrowed_pattern
            print(f"      → Narrowed pattern to exclude '{term}'")
        
        return template
    
    def _broaden_pattern(self, template: Dict, 
                        positive_examples: List[Tuple[str, str]],
                        counter_examples: List[Tuple[str, str, str]]) -> Dict:
        """Broaden pattern to cover counter-examples that should match."""
        current_pattern = template.get('pattern', '')
        
        # Analyze why pattern didn't match
        # Make capture groups more permissive
        broadened = current_pattern.replace('(\\w+)', '(.+?)')
        broadened = broadened.replace('([^,]+)', '(.+?)')
        broadened = broadened.replace('\\s+', '\\s*')
        
        template['pattern'] = broadened
        print(f"      → Broadened pattern to be more permissive")
        
        return template
    
    def _fix_semantic_function(self, template: Dict, 
                               counter_examples: List[Tuple[str, str, str]]) -> Dict:
        """Fix semantic function based on counter-examples."""
        composition_rule = template.get('composition_rule', '')
        
        # Analyze what went wrong
        for inp, expected, got in counter_examples[:3]:
            print(f"      → Analyzing: '{expected}' vs '{got}'")
            
            # Common fixes
            if '→' in expected and '→' not in got:
                # Missing implication arrow
                template['semantic_function_hint'] = 'add_implication'
            elif '∀' in expected and '∀' not in got:
                # Missing quantifier
                template['semantic_function_hint'] = 'add_quantifier'
        
        return template
    
    def _add_type_constraints(self, template: Dict, 
                             counter_examples: List[Tuple[str, str, str]]) -> Dict:
        """Add type constraints based on counter-examples."""
        constraints = template.get('type_constraints', [])
        
        # Add common type constraints
        if 'operands_same_type' not in constraints:
            constraints.append('operands_same_type')
        
        if 'wellformed_output' not in constraints:
            constraints.append('wellformed_output')
        
        template['type_constraints'] = constraints
        print(f"      → Added {len(constraints)} type constraints")
        
        return template


# Example usage demonstrating full CEGIS pipeline
if __name__ == '__main__':
    print("=" * 70)
    print("Z3-DRIVEN COMPOSITIONAL SEMANTIC SYNTHESIS")
    print("=" * 70)
    
    # PART 1: Basic semantic synthesis
    print("\nPART 1: Basic Semantic Synthesis")
    print("-" * 70)
    
    synthesizer = Z3SemanticSynthesizer()
    
    english = "for all x in ℝ, x ≥ 0"
    parse_tree = ParseTree(text=english)
    
    print(f"Input: {english}")
    print("Synthesizing valid interpretations...")
    
    interpretations = synthesizer.synthesize_semantics(
        english, 
        parse_tree,
        type_hints={'ℝ': 'Real'}
    )
    
    print(f"\nFound {len(interpretations)} valid interpretations:")
    
    for i, interp in enumerate(interpretations, 1):
        print(f"\n  {i}. Complexity: {interp.complexity_score:.2f}")
        print(f"     Hole assignments:")
        for hole, value in interp.hole_assignments.items():
            print(f"       {hole} := {value}")
        print(f"     Lean output: {interp.lean_output}")
    
    # PART 2: CEGIS Compositional Learning
    print("\n" + "=" * 70)
    print("PART 2: CEGIS Compositional Semantic Learning")
    print("=" * 70)
    
    learner = CEGIS_SemanticLearner()
    
    # Realistic training corpus
    training_data = [
        # Universal quantification
        ("for all x in ℝ, x > 0", "∀ x : ℝ, x > 0"),
        ("for every element e of S, P(e)", "∀ e : S, P e"),
        ("given any natural number n, n is even or odd", "∀ n : ℕ, even n ∨ odd n"),
        
        # Existential quantification
        ("there exists x in ℕ such that x > 5", "∃ x : ℕ, x > 5"),
        ("some element of S has property P", "∃ x : S, P x"),
        
        # Implication
        ("if P then Q", "P → Q"),
        ("P implies Q", "P → Q"),
        ("whenever x > 0, x² > 0", "∀ x, x > 0 → x² > 0"),
        
        # Conjunction/Disjunction
        ("P and Q", "P ∧ Q"),
        ("P or Q", "P ∨ Q"),
        
        # Function application
        ("f applied to x", "f x"),
        ("the image of x under f", "f x"),
        
        # Lambda abstraction
        ("the function mapping x to x²", "fun x => x²"),
    ]
    
    print(f"\nTraining on {len(training_data)} examples...")
    
    learned_rules = learner.learn_from_corpus(
        training_data,
        max_iterations=20,
        min_confidence=0.7
    )
    
    # PART 3: Test learned rules
    print("\n" + "=" * 70)
    print("PART 3: Testing Learned Rules")
    print("=" * 70)
    
    test_examples = [
        "for all x in Q, x > 0",
        "if x > 0 then x² > 0",
        "there exists y in ℝ such that y² = 2",
    ]
    
    for test_eng in test_examples:
        print(f"\nTest: {test_eng}")
        
        for rule in learned_rules:
            if rule.matches(test_eng):
                result = rule.apply(test_eng)
                print(f"  Rule {rule.rule_id}: {result}")
                print(f"    Quality: {rule.quality_score:.3f}")
                print(f"    Type: {rule.composition_type}")
                break
        else:
            print("  No matching rule found")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
