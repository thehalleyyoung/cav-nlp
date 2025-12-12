"""
Compositional Semantics for Mathematical English
=================================================

Formal semantic grammar where each production rule has:
1. Syntactic pattern (context-free grammar)
2. Semantic function (compositional meaning)
3. Type constraints (what must hold in Lean's type theory)

This is the core of the natural language → Lean translation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Callable, Any, Tuple
from enum import Enum
import re
from lean_type_theory import *


class Category(Enum):
    """Syntactic categories for mathematical English."""
    # Statements
    STATEMENT = "statement"
    DEFINITION = "definition"
    THEOREM = "theorem"
    LEMMA = "lemma"
    STRUCTURE = "structure"
    
    # Propositions
    PROPOSITION = "proposition"
    QUANTIFIED = "quantified_prop"
    IMPLICATION = "implication"
    CONJUNCTION = "conjunction"
    NEGATION = "negation"
    
    # Terms
    TERM = "term"
    VARIABLE = "variable"
    APPLICATION = "application"
    
    # Types
    TYPE_EXPR = "type_expression"
    TYPE_NAME = "type_name"
    FUNCTION_TYPE = "function_type"
    
    # Quantifiers
    UNIVERSAL = "universal"
    EXISTENTIAL = "existential"
    
    # Operators
    RELATION = "relation"
    BINARY_OP = "binary_op"
    
    # Structural
    FIELD_LIST = "field_list"
    CONSTRAINT = "constraint"


@dataclass
class ParseNode:
    """Node in parse tree with syntactic and semantic information."""
    category: Category
    text: str
    children: List['ParseNode'] = field(default_factory=list)
    semantic_value: Optional[Any] = None
    span: Tuple[int, int] = (0, 0)  # Character positions in source
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def add_child(self, child: 'ParseNode') -> None:
        self.children.append(child)
    
    def get_text_span(self, full_text: str) -> str:
        """Extract text for this node's span."""
        return full_text[self.span[0]:self.span[1]]


@dataclass
class SemanticFunction:
    """Compositional semantic function for a production rule."""
    name: str
    arity: int
    function: Callable
    
    def apply(self, *args) -> Any:
        """Apply semantic function to child denotations."""
        if len(args) != self.arity:
            raise ValueError(f"Expected {self.arity} arguments, got {len(args)}")
        return self.function(*args)


@dataclass
class ProductionRule:
    """Context-free production rule with compositional semantics."""
    lhs: Category
    rhs_pattern: str  # Regex pattern for matching
    rhs_categories: List[Category]  # Categories of constituents
    semantic_function: SemanticFunction
    type_constraint: Optional[Callable] = None  # Type checking function
    priority: int = 10  # For disambiguation (higher = prefer)
    
    def matches(self, text: str) -> Optional[re.Match]:
        """Check if text matches this production."""
        return re.match(self.rhs_pattern, text, re.IGNORECASE)
    
    def extract_constituents(self, match: re.Match) -> List[str]:
        """Extract constituent texts from regex match."""
        return [g for g in match.groups() if g is not None]


class SemanticGrammar:
    """Grammar with compositional semantic rules for mathematical English."""
    
    def __init__(self):
        self.rules: Dict[Category, List[ProductionRule]] = {}
        self.lexicon: Dict[str, Tuple[Category, Any]] = {}
        self._build_core_grammar()
    
    def add_rule(self, rule: ProductionRule) -> None:
        """Add production rule to grammar."""
        if rule.lhs not in self.rules:
            self.rules[rule.lhs] = []
        self.rules[rule.lhs].append(rule)
        # Sort by priority
        self.rules[rule.lhs].sort(key=lambda r: r.priority, reverse=True)
    
    def add_lexical_entry(self, word: str, category: Category, 
                         denotation: Any) -> None:
        """Add word to lexicon with its semantic denotation."""
        self.lexicon[word.lower()] = (category, denotation)
    
    def _build_core_grammar(self) -> None:
        """Build core compositional semantic rules."""
        
        # UNIVERSAL QUANTIFICATION
        # "for all x : T, P(x)" → ∀ (x : T), P
        self.add_rule(ProductionRule(
            lhs=Category.QUANTIFIED,
            rhs_pattern=r"for\s+(?:all|every|each)\s+([a-zA-Zα-ωΑ-Ω]+)\s*(?::\s*([^,]+))?,\s*(.+)",
            rhs_categories=[Category.VARIABLE, Category.TYPE_EXPR, Category.PROPOSITION],
            semantic_function=SemanticFunction(
                name="universal_quantification",
                arity=3,
                function=lambda var, var_type, body: create_forall_type(
                    var_name=var,
                    var_type=var_type if var_type else LeanVariable("_infer"),
                    body_type=body
                )
            ),
            priority=20
        ))
        
        # EXISTENTIAL QUANTIFICATION
        # "there exists x : T such that P(x)" → ∃ (x : T), P
        self.add_rule(ProductionRule(
            lhs=Category.QUANTIFIED,
            rhs_pattern=r"there\s+exists?\s+([a-zA-Zα-ωΑ-Ω]+)\s*(?::\s*([^,]+))?\s+such\s+that\s+(.+)",
            rhs_categories=[Category.VARIABLE, Category.TYPE_EXPR, Category.PROPOSITION],
            semantic_function=SemanticFunction(
                name="existential_quantification",
                arity=3,
                function=lambda var, var_type, predicate: create_exists_type(
                    var_type=var_type if var_type else LeanVariable("_infer"),
                    predicate_type=predicate
                )
            ),
            priority=20
        ))
        
        # IMPLICATION
        # "if P then Q" → P → Q
        self.add_rule(ProductionRule(
            lhs=Category.IMPLICATION,
            rhs_pattern=r"if\s+(.+?)\s+then\s+(.+)",
            rhs_categories=[Category.PROPOSITION, Category.PROPOSITION],
            semantic_function=SemanticFunction(
                name="implication",
                arity=2,
                function=lambda antecedent, consequent: LeanArrow(
                    from_type=antecedent,
                    to_type=consequent
                )
            ),
            priority=15
        ))
        
        # "P implies Q" → P → Q
        self.add_rule(ProductionRule(
            lhs=Category.IMPLICATION,
            rhs_pattern=r"(.+?)\s+implies\s+(.+)",
            rhs_categories=[Category.PROPOSITION, Category.PROPOSITION],
            semantic_function=SemanticFunction(
                name="implies",
                arity=2,
                function=lambda antecedent, consequent: LeanArrow(
                    from_type=antecedent,
                    to_type=consequent
                )
            ),
            priority=15
        ))
        
        # CONJUNCTION
        # "P and Q" → P ∧ Q
        self.add_rule(ProductionRule(
            lhs=Category.CONJUNCTION,
            rhs_pattern=r"(.+?)\s+and\s+(.+)",
            rhs_categories=[Category.PROPOSITION, Category.PROPOSITION],
            semantic_function=SemanticFunction(
                name="conjunction",
                arity=2,
                function=lambda left, right: create_and_type(left, right)
            ),
            priority=10
        ))
        
        # STRUCTURE DEFINITION
        # "A X is a Y together with Z" → structure X extends Y where ...
        self.add_rule(ProductionRule(
            lhs=Category.STRUCTURE,
            rhs_pattern=r"[Aa]\s+([A-Z][a-zA-Z]*)\s+is\s+a\s+([A-Z][a-zA-Z]*)\s+(?:together\s+with|with)\s+(.+)",
            rhs_categories=[Category.TYPE_NAME, Category.TYPE_NAME, Category.FIELD_LIST],
            semantic_function=SemanticFunction(
                name="structure_extension",
                arity=3,
                function=self._make_structure_extends
            ),
            priority=25
        ))
        
        # "A X consists of Y" → structure X where Y
        self.add_rule(ProductionRule(
            lhs=Category.STRUCTURE,
            rhs_pattern=r"[Aa]\s+([A-Z][a-zA-Z]*)\s+consists?\s+of\s+(.+)",
            rhs_categories=[Category.TYPE_NAME, Category.FIELD_LIST],
            semantic_function=SemanticFunction(
                name="structure_definition",
                arity=2,
                function=self._make_structure
            ),
            priority=25
        ))
        
        # FUNCTION TYPE
        # "f : X → Y" → X → Y
        self.add_rule(ProductionRule(
            lhs=Category.FUNCTION_TYPE,
            rhs_pattern=r"([^:→]+)\s*→\s*(.+)",
            rhs_categories=[Category.TYPE_EXPR, Category.TYPE_EXPR],
            semantic_function=SemanticFunction(
                name="function_type",
                arity=2,
                function=lambda from_ty, to_ty: LeanArrow(from_ty, to_ty)
            ),
            priority=15
        ))
        
        # LEXICAL ENTRIES
        self._build_lexicon()
    
    def _make_structure_extends(self, name: str, base: str, 
                               fields: List[Tuple[str, LeanType]]):
        """Create structure that extends another."""
        return {
            'kind': 'structure',
            'name': self._to_pascal_case(name),
            'extends': self._to_pascal_case(base),
            'fields': fields
        }
    
    def _make_structure(self, name: str, fields: List[Tuple[str, LeanType]]):
        """Create structure definition."""
        return {
            'kind': 'structure',
            'name': self._to_pascal_case(name),
            'extends': None,
            'fields': fields
        }
    
    def _to_pascal_case(self, text: str) -> str:
        """Convert to PascalCase (canonical for types)."""
        words = re.findall(r'[A-Za-z][a-z]*|[A-Z]+', text)
        return ''.join(word.capitalize() for word in words)
    
    def _to_snake_case(self, text: str) -> str:
        """Convert to snake_case (canonical for definitions/theorems)."""
        # Insert underscore before capitals
        text = re.sub('([a-z0-9])([A-Z])', r'\1_\2', text)
        # Convert to lowercase
        return text.lower().replace(' ', '_').replace('-', '_')
    
    def _build_lexicon(self) -> None:
        """Build lexicon of mathematical terms."""
        
        # Type names
        common_types = {
            'ℝ': LeanVariable('Real'),
            'ℕ': LeanVariable('Nat'),
            'ℤ': LeanVariable('Int'),
            'ℚ': LeanVariable('Rat'),
            'real': LeanVariable('Real'),
            'natural': LeanVariable('Nat'),
            'integer': LeanVariable('Int'),
            'rational': LeanVariable('Rat'),
            'set': LeanVariable('Set'),
            'type': LeanTypeSort(UniverseLevel(0)),
            'prop': LeanProp(),
        }
        
        for word, lean_type in common_types.items():
            self.add_lexical_entry(word, Category.TYPE_NAME, lean_type)
        
        # Relation symbols
        relations = {
            '<': 'LT.lt',
            '>': 'GT.gt',
            '≤': 'LE.le',
            '≥': 'GE.ge',
            '=': 'Eq',
            '≠': 'Ne',
            '∈': 'Membership.mem',
            '∉': 'Membership.notMem',
            '⊆': 'HasSubset.Subset',
            '⊂': 'HasSSubset.SSubset',
        }
        
        for symbol, lean_name in relations.items():
            self.add_lexical_entry(symbol, Category.RELATION, lean_name)
    
    def parse(self, text: str, context: Optional[Context] = None) -> List[ParseNode]:
        """
        Parse text and return possible parse trees.
        Returns multiple parses due to ambiguity.
        """
        text = text.strip()
        
        # Try to parse as each category
        all_parses = []
        
        for category in [Category.QUANTIFIED, Category.IMPLICATION, 
                        Category.STRUCTURE, Category.DEFINITION, 
                        Category.THEOREM]:
            parses = self._parse_category(text, category, context)
            all_parses.extend(parses)
        
        return all_parses
    
    def _parse_category(self, text: str, category: Category,
                       context: Optional[Context]) -> List[ParseNode]:
        """Try to parse text as specific category."""
        if category not in self.rules:
            return []
        
        parses = []
        
        for rule in self.rules[category]:
            match = rule.matches(text)
            if not match:
                continue
            
            # Extract constituents
            constituents = rule.extract_constituents(match)
            
            # Recursively parse constituents
            child_parses_list = []
            for i, (const_text, const_cat) in enumerate(zip(constituents, 
                                                            rule.rhs_categories)):
                child_parses = self._parse_category(const_text.strip(), 
                                                    const_cat, context)
                if not child_parses:
                    # Try as terminal
                    child_parses = [ParseNode(
                        category=const_cat,
                        text=const_text.strip(),
                        children=[],
                        semantic_value=const_text.strip()
                    )]
                child_parses_list.append(child_parses)
            
            # Combine all possible child parses
            import itertools
            for child_combination in itertools.product(*child_parses_list):
                node = ParseNode(
                    category=category,
                    text=text,
                    children=list(child_combination)
                )
                parses.append(node)
        
        return parses
    
    def compute_semantics(self, parse_tree: ParseNode, 
                         context: Context) -> Any:
        """
        Compute compositional semantics bottom-up.
        Returns Lean type expression.
        """
        # Base case: leaf node
        if parse_tree.is_leaf():
            return self._compute_leaf_semantics(parse_tree, context)
        
        # Recursive case: apply semantic function to children
        child_semantics = [
            self.compute_semantics(child, context)
            for child in parse_tree.children
        ]
        
        # Find matching rule
        for rule in self.rules.get(parse_tree.category, []):
            if rule.matches(parse_tree.text):
                try:
                    semantic_value = rule.semantic_function.apply(*child_semantics)
                    parse_tree.semantic_value = semantic_value
                    
                    # Type check if constraint provided
                    if rule.type_constraint:
                        if not rule.type_constraint(semantic_value, context):
                            continue  # Type check failed
                    
                    return semantic_value
                except Exception as e:
                    continue  # Semantic function failed
        
        # Fallback: return text
        return parse_tree.text
    
    def _compute_leaf_semantics(self, node: ParseNode, 
                               context: Context) -> Any:
        """Compute semantics for leaf nodes (lexical items)."""
        text = node.text.strip().lower()
        
        # Check lexicon
        if text in self.lexicon:
            category, denotation = self.lexicon[text]
            return denotation
        
        # Variable name
        if node.category == Category.VARIABLE:
            # Look up in context
            var_type = context.lookup(text)
            if var_type:
                return LeanVarExpr(text, var_type)
            return text  # Unknown variable
        
        # Type name
        if node.category == Category.TYPE_NAME:
            # Check if it's a known type
            pascal_case = self._to_pascal_case(text)
            return LeanVariable(pascal_case)
        
        # Number literal
        if text.isdigit():
            return LeanVarExpr(text, LeanVariable('Nat'))
        
        # Default: return text
        return text


class TypeInferenceEngine:
    """Infer types for semantic values using Z3."""
    
    def __init__(self):
        self.solver = Solver()
        self.type_checker = LeanTypeChecker()
        self.type_vars: Dict[Any, ExprRef] = {}
    
    def infer_types(self, semantic_value: Any, context: Context) -> Optional[LeanType]:
        """Infer Lean type for semantic value."""
        
        if isinstance(semantic_value, LeanType):
            return semantic_value
        
        if isinstance(semantic_value, LeanExpr):
            return self.type_checker.infer(semantic_value, context)
        
        if isinstance(semantic_value, str):
            # Variable name
            var_type = context.lookup(semantic_value)
            return var_type
        
        return None
    
    def unify_types(self, type1: LeanType, type2: LeanType) -> bool:
        """Check if types can be unified."""
        return self.type_checker.definitionally_equal(type1, type2, Context())
    
    def add_type_constraint(self, expr: Any, expected_type: LeanType) -> None:
        """Add constraint that expression must have given type."""
        # Create Z3 variable for this expression's type
        if expr not in self.type_vars:
            self.type_vars[expr] = Const(f"type_{id(expr)}", DeclareSort('LeanType'))
        
        # Add constraint
        # (This is simplified; real implementation would encode full type theory)
        pass
    
    def solve(self) -> Optional[Dict[Any, LeanType]]:
        """Solve type constraints and return type assignment."""
        if self.solver.check() == sat:
            model = self.solver.model()
            # Extract types from model
            return {}
        return None


def parse_and_compute_semantics(text: str, grammar: SemanticGrammar,
                                context: Context) -> List[Tuple[ParseNode, Any]]:
    """
    Parse text and compute compositional semantics.
    Returns list of (parse_tree, semantic_value) pairs.
    """
    parse_trees = grammar.parse(text, context)
    
    results = []
    for parse_tree in parse_trees:
        try:
            semantic_value = grammar.compute_semantics(parse_tree, context)
            results.append((parse_tree, semantic_value))
        except Exception as e:
            # Semantic computation failed for this parse
            continue
    
    return results
