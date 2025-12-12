#!/usr/bin/env python3
"""
Intermediate Representation (IR) for LaTeX → Lean Translation

This module implements a formal IR based on literature on mathematical notation parsing:

1. MathLang (Kamareddine et al. 2004) - Separates presentation from semantics
2. OpenMath / Content MathML - Standard for mathematical content representation
3. Naproche (Creutz et al. 2021) - Controlled natural language for proof checking
4. Grammatical Framework (Ranta 1994, 2011) - Abstract/concrete syntax separation
5. OMDoc (Kohlhase 2006) - Semantic markup for mathematical documents

Key Design Principles:
- PRESENTATION vs SEMANTICS separation (Kamareddine et al.)
- Type-theoretic foundation (Martin-Löf, Coquand)
- Compositional semantics (Montague, Steedman)
- Explicit binding and scope (Church, Barendregt)
- Preserves provenance for error reporting

The IR serves as a bridge:
  LaTeX/Natural Language → MathIR → Lean

Benefits:
- Reusable for multiple target languages (Lean, Coq, Isabelle, Z3)
- Type checking at IR level catches errors early
- Canonicalization and optimization possible
- Human-readable for debugging
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import json


# ============================================================================
# LITERATURE-BASED IR DESIGN
# ============================================================================

class MathIRSort(Enum):
    """
    Sorts in the IR type system.
    Based on Martin-Löf type theory and Lean's type hierarchy.
    """
    PROP = "Prop"           # Propositions (proof-relevant)
    TYPE = "Type"           # Types (universe)
    NAT = "Nat"             # Natural numbers
    INT = "Int"             # Integers
    REAL = "Real"           # Real numbers
    SET = "Set"             # Sets
    FUNCTION = "Function"   # Function types
    DEPENDENT = "Dependent" # Dependent types (Π-types)


@dataclass
class SourceLocation:
    """
    Source location for error reporting.
    Tracks both LaTeX source and natural language.
    """
    latex_source: Optional[str] = None
    natural_language: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    file: Optional[str] = None


# ============================================================================
# IR EXPRESSION TYPES (Based on Lambda Calculus + Dependent Types)
# ============================================================================

class MathIRExpr:
    """Base class for all IR expressions"""
    def __init__(self):
        self.source: Optional[SourceLocation] = None
        self.inferred_type: Optional['MathIRExpr'] = None


@dataclass
class IRVar(MathIRExpr):
    """
    Variable reference.
    Example: x, n, f
    """
    name: str
    de_bruijn_index: Optional[int] = None  # For explicit scope tracking
    
    def to_lean(self) -> str:
        return self.name
    
    def to_json(self) -> Dict:
        return {
            'type': 'Var',
            'name': self.name,
            'de_bruijn': self.de_bruijn_index
        }


@dataclass
class IRConst(MathIRExpr):
    """
    Constant (0, 1, true, false, etc.)
    """
    value: Any
    sort: MathIRSort
    
    def to_lean(self) -> str:
        if self.sort == MathIRSort.NAT:
            return str(self.value)
        elif self.sort == MathIRSort.PROP:
            return 'True' if self.value else 'False'
        else:
            return str(self.value)
    
    def to_json(self) -> Dict:
        return {
            'type': 'Const',
            'value': self.value,
            'sort': self.sort.value
        }


@dataclass
class IRApp(MathIRExpr):
    """
    Function application: f(x) or f x
    Based on lambda calculus application rule.
    """
    function: MathIRExpr
    argument: MathIRExpr
    
    def to_lean(self) -> str:
        # Lean uses juxtaposition for application
        func_str = self.function.to_lean()
        arg_str = self.argument.to_lean()
        
        # Add parentheses if argument is complex
        if isinstance(self.argument, (IRApp, IRLambda, IRPi)):
            arg_str = f"({arg_str})"
        
        return f"{func_str} {arg_str}"
    
    def to_json(self) -> Dict:
        return {
            'type': 'App',
            'function': self.function.to_json(),
            'argument': self.argument.to_json()
        }


@dataclass
class IRLambda(MathIRExpr):
    """
    Lambda abstraction: λx. e or fun x => e (Lean syntax)
    Church's lambda calculus foundation.
    """
    var_name: str
    var_type: Optional[MathIRExpr]
    body: MathIRExpr
    
    def to_lean(self) -> str:
        if self.var_type:
            return f"fun ({self.var_name} : {self.var_type.to_lean()}) => {self.body.to_lean()}"
        else:
            return f"fun {self.var_name} => {self.body.to_lean()}"
    
    def to_json(self) -> Dict:
        return {
            'type': 'Lambda',
            'var': self.var_name,
            'var_type': self.var_type.to_json() if self.var_type else None,
            'body': self.body.to_json()
        }


@dataclass
class IRPi(MathIRExpr):
    """
    Dependent product type: Π(x : A), B or ∀(x : A), B
    Foundation of dependent type theory (Martin-Löf).
    
    Special cases:
    - Function type: A → B (when B doesn't depend on x)
    - Universal quantification: ∀x : A, P(x)
    """
    var_name: str
    var_type: MathIRExpr
    body: MathIRExpr
    is_implicit: bool = False  # Lean's implicit arguments {x : A}
    
    def to_lean(self) -> str:
        # Check if this is a simple function type (non-dependent)
        if not self._is_dependent():
            return f"{self.var_type.to_lean()} → {self.body.to_lean()}"
        
        # Dependent function type
        if self.is_implicit:
            return f"∀ {{{self.var_name} : {self.var_type.to_lean()}}}, {self.body.to_lean()}"
        else:
            return f"∀ ({self.var_name} : {self.var_type.to_lean()}), {self.body.to_lean()}"
    
    def _is_dependent(self) -> bool:
        """Check if body actually depends on variable"""
        # Simplified check - in full implementation, traverse body AST
        return True
    
    def to_json(self) -> Dict:
        return {
            'type': 'Pi',
            'var': self.var_name,
            'var_type': self.var_type.to_json(),
            'body': self.body.to_json(),
            'implicit': self.is_implicit
        }


@dataclass
class IRLet(MathIRExpr):
    """
    Let binding: let x := e1 in e2
    Common in mathematical definitions.
    """
    var_name: str
    var_type: Optional[MathIRExpr]
    value: MathIRExpr
    body: MathIRExpr
    
    def to_lean(self) -> str:
        if self.var_type:
            return f"let {self.var_name} : {self.var_type.to_lean()} := {self.value.to_lean()}\n{self.body.to_lean()}"
        else:
            return f"let {self.var_name} := {self.value.to_lean()}\n{self.body.to_lean()}"
    
    def to_json(self) -> Dict:
        return {
            'type': 'Let',
            'var': self.var_name,
            'var_type': self.var_type.to_json() if self.var_type else None,
            'value': self.value.to_json(),
            'body': self.body.to_json()
        }


# ============================================================================
# MATHEMATICAL NOTATION IR (Based on Kamareddine et al. 2004)
# ============================================================================

@dataclass
class IRSubscript(MathIRExpr):
    """
    Subscript notation: x_i, x_{n,m}
    
    Kamareddine et al.: Separate presentation (x_i) from semantics (indexed access).
    Multiple interpretations:
    - Array/sequence indexing: x[i]
    - Family member: x_i as distinct variable
    - Component extraction: vector_i
    """
    base: MathIRExpr
    indices: List[MathIRExpr]
    interpretation: str = "indexed"  # "indexed", "family", "component"
    
    def to_lean(self) -> str:
        if self.interpretation == "indexed":
            # Array indexing in Lean
            base_str = self.base.to_lean()
            if len(self.indices) == 1:
                return f"{base_str}[{self.indices[0].to_lean()}]"
            else:
                # Multi-dimensional indexing
                indices_str = ", ".join(idx.to_lean() for idx in self.indices)
                return f"{base_str}[{indices_str}]"
        elif self.interpretation == "family":
            # Treat as distinct variable name
            base_str = self.base.to_lean()
            indices_str = "_".join(idx.to_lean() for idx in self.indices)
            return f"{base_str}_{indices_str}"
        else:  # component
            base_str = self.base.to_lean()
            return f"{base_str}.{self.indices[0].to_lean()}"
    
    def to_json(self) -> Dict:
        return {
            'type': 'Subscript',
            'base': self.base.to_json(),
            'indices': [idx.to_json() for idx in self.indices],
            'interpretation': self.interpretation
        }


@dataclass
class IRSuperscript(MathIRExpr):
    """
    Superscript notation: x^n, x^{-1}, f^{(n)}
    
    Multiple interpretations:
    - Exponentiation: x^n = x * x * ... * x
    - Inverse: x^{-1}
    - Derivative: f^{(n)} = n-th derivative
    - Iteration: f^n = f ∘ f ∘ ... ∘ f
    """
    base: MathIRExpr
    exponent: MathIRExpr
    interpretation: str = "power"  # "power", "inverse", "derivative", "iteration"
    
    def to_lean(self) -> str:
        base_str = self.base.to_lean()
        exp_str = self.exponent.to_lean()
        
        if self.interpretation == "power":
            return f"{base_str} ^ {exp_str}"
        elif self.interpretation == "inverse":
            return f"{base_str}⁻¹"
        elif self.interpretation == "derivative":
            return f"deriv^[{exp_str}] {base_str}"
        else:  # iteration
            return f"iterate {base_str} {exp_str}"
    
    def to_json(self) -> Dict:
        return {
            'type': 'Superscript',
            'base': self.base.to_json(),
            'exponent': self.exponent.to_json(),
            'interpretation': self.interpretation
        }


@dataclass
class IRBigOp(MathIRExpr):
    """
    Big operators: ∑, ∏, ⋃, ⋂, ∫
    
    OpenMath representation: operator with range and body.
    """
    operator: str  # "sum", "prod", "union", "inter", "integral"
    var_name: str
    lower_bound: MathIRExpr
    upper_bound: MathIRExpr
    body: MathIRExpr
    
    def to_lean(self) -> str:
        # Lean's big operator notation
        if self.operator == "sum":
            return f"∑ {self.var_name} in {self.lower_bound.to_lean()}..{self.upper_bound.to_lean()}, {self.body.to_lean()}"
        elif self.operator == "prod":
            return f"∏ {self.var_name} in {self.lower_bound.to_lean()}..{self.upper_bound.to_lean()}, {self.body.to_lean()}"
        else:
            return f"{self.operator} {self.var_name} in {self.lower_bound.to_lean()}..{self.upper_bound.to_lean()}, {self.body.to_lean()}"
    
    def to_json(self) -> Dict:
        return {
            'type': 'BigOp',
            'operator': self.operator,
            'var': self.var_name,
            'lower': self.lower_bound.to_json(),
            'upper': self.upper_bound.to_json(),
            'body': self.body.to_json()
        }


@dataclass
class IRSetBuilder(MathIRExpr):
    """
    Set comprehension: {x : A | P(x)} or {x ∈ A | P(x)}
    
    Based on ZFC set theory and Lean's set notation.
    """
    var_name: str
    var_type: MathIRExpr
    predicate: MathIRExpr
    
    def to_lean(self) -> str:
        return f"{{ {self.var_name} : {self.var_type.to_lean()} | {self.predicate.to_lean()} }}"
    
    def to_json(self) -> Dict:
        return {
            'type': 'SetBuilder',
            'var': self.var_name,
            'var_type': self.var_type.to_json(),
            'predicate': self.predicate.to_json()
        }


# ============================================================================
# LOGICAL CONNECTIVES & QUANTIFIERS
# ============================================================================

@dataclass
class IRBinOp(MathIRExpr):
    """
    Binary operations: +, -, *, /, ∧, ∨, →, ↔, =, <, >, etc.
    """
    operator: str
    left: MathIRExpr
    right: MathIRExpr
    
    # Operator precedence for parenthesization
    PRECEDENCE = {
        '↔': 1, '→': 2, '∨': 3, '∧': 4,
        '=': 5, '≠': 5, '<': 5, '>': 5, '≤': 5, '≥': 5,
        '+': 6, '-': 6,
        '*': 7, '/': 7, '%': 7,
        '^': 8
    }
    
    def to_lean(self) -> str:
        # Map operators to Lean syntax
        op_map = {
            '∧': '∧', '∨': '∨', '→': '→', '↔': '↔',
            '=': '=', '≠': '≠', '<': '<', '>': '>', '≤': '≤', '≥': '≥',
            '+': '+', '-': '-', '*': '*', '/': '/', '%': '%', '^': '^',
            '∈': '∈', '⊆': '⊆', '∪': '∪', '∩': '∩'
        }
        
        lean_op = op_map.get(self.operator, self.operator)
        left_str = self._parenthesize(self.left, True)
        right_str = self._parenthesize(self.right, False)
        
        return f"{left_str} {lean_op} {right_str}"
    
    def _parenthesize(self, expr: MathIRExpr, is_left: bool) -> str:
        """Add parentheses if needed based on precedence"""
        expr_str = expr.to_lean()
        
        if not isinstance(expr, IRBinOp):
            return expr_str
        
        my_prec = self.PRECEDENCE.get(self.operator, 0)
        expr_prec = self.PRECEDENCE.get(expr.operator, 0)
        
        # Left-associative: parenthesize if lower precedence
        # Right-associative: parenthesize if lower or equal precedence
        if is_left:
            needs_parens = expr_prec < my_prec
        else:
            needs_parens = expr_prec <= my_prec
        
        return f"({expr_str})" if needs_parens else expr_str
    
    def to_json(self) -> Dict:
        return {
            'type': 'BinOp',
            'operator': self.operator,
            'left': self.left.to_json(),
            'right': self.right.to_json()
        }


@dataclass
class IRUnOp(MathIRExpr):
    """
    Unary operations: ¬, -, |x|, etc.
    """
    operator: str
    operand: MathIRExpr
    
    def to_lean(self) -> str:
        op_str = self.operand.to_lean()
        
        # Add parentheses for complex operands
        if isinstance(self.operand, (IRBinOp, IRApp)):
            op_str = f"({op_str})"
        
        if self.operator == '¬':
            return f"¬ {op_str}"
        elif self.operator == '-':
            return f"- {op_str}"
        elif self.operator == 'abs':
            return f"|{op_str}|"
        else:
            return f"{self.operator} {op_str}"
    
    def to_json(self) -> Dict:
        return {
            'type': 'UnOp',
            'operator': self.operator,
            'operand': self.operand.to_json()
        }


# ============================================================================
# TYPE SYSTEM & CHECKING
# ============================================================================

class IRTypeChecker:
    """
    Type checker for MathIR.
    Based on bidirectional type checking (Pierce & Turner 2000).
    """
    
    def __init__(self):
        self.context: Dict[str, MathIRExpr] = {}
    
    def infer_type(self, expr: MathIRExpr) -> MathIRExpr:
        """Infer the type of an expression"""
        
        if isinstance(expr, IRVar):
            if expr.name in self.context:
                return self.context[expr.name]
            else:
                raise TypeError(f"Unbound variable: {expr.name}")
        
        elif isinstance(expr, IRConst):
            # Return sort as type
            return IRVar(expr.sort.value)
        
        elif isinstance(expr, IRApp):
            func_type = self.infer_type(expr.function)
            
            # Function type should be Pi type
            if isinstance(func_type, IRPi):
                arg_type = self.infer_type(expr.argument)
                # Check argument type matches parameter type
                # (simplified - full checking would use type equality)
                return func_type.body
            else:
                raise TypeError(f"Expected function type, got {func_type}")
        
        elif isinstance(expr, IRLambda):
            # Add variable to context
            if expr.var_type:
                self.context[expr.var_name] = expr.var_type
            
            body_type = self.infer_type(expr.body)
            
            # Remove variable from context
            if expr.var_name in self.context:
                del self.context[expr.var_name]
            
            # Return function type
            return IRPi(expr.var_name, expr.var_type or IRVar("?"), body_type)
        
        elif isinstance(expr, IRBinOp):
            # Simplified - would need full operator typing
            if expr.operator in ['∧', '∨', '→', '↔']:
                return IRVar("Prop")
            elif expr.operator in ['=', '≠', '<', '>', '≤', '≥']:
                return IRVar("Prop")
            else:
                # Arithmetic operators - infer from operands
                left_type = self.infer_type(expr.left)
                return left_type
        
        else:
            # Default: unknown type
            return IRVar("?")
    
    def check_type(self, expr: MathIRExpr, expected_type: MathIRExpr) -> bool:
        """Check if expression has expected type"""
        inferred_type = self.infer_type(expr)
        # Simplified type equality - full version would use unification
        return self._types_equal(inferred_type, expected_type)
    
    def _types_equal(self, t1: MathIRExpr, t2: MathIRExpr) -> bool:
        """Check if two types are equal (simplified)"""
        if isinstance(t1, IRVar) and isinstance(t2, IRVar):
            return t1.name == t2.name
        # Full implementation would handle definitional equality
        return False


# ============================================================================
# LATEX PARSER → IR (Based on Kamareddine et al. 2004)
# ============================================================================

class LaTeXToIR:
    """
    Parse LaTeX mathematical expressions into MathIR.
    
    Based on:
    - MathLang (Kamareddine et al. 2004)
    - Presentation vs Content separation
    """
    
    def __init__(self):
        self.context = {}
    
    def parse(self, latex: str, natural_language: str = None) -> MathIRExpr:
        """
        Parse LaTeX with optional natural language context.
        
        This is a simplified parser - full implementation would use
        proper LaTeX parsing (e.g., plasTeX, TexSoup).
        """
        source = SourceLocation(
            latex_source=latex,
            natural_language=natural_language
        )
        
        # Simple pattern matching for demonstration
        # Full parser would build proper AST from LaTeX tokens
        
        # Forall: \forall x, P(x)
        if '\\forall' in latex or 'for all' in (natural_language or ''):
            return self._parse_forall(latex, source)
        
        # Exists: \exists x, P(x)
        elif '\\exists' in latex or 'there exists' in (natural_language or ''):
            return self._parse_exists(latex, source)
        
        # Implication: P \to Q
        elif '\\to' in latex or '\\rightarrow' in latex or 'if' in (natural_language or ''):
            return self._parse_implication(latex, source)
        
        # Subscript: x_i, x_{n}
        elif '_' in latex:
            return self._parse_subscript(latex, source)
        
        # Superscript: x^n
        elif '^' in latex:
            return self._parse_superscript(latex, source)
        
        # Sum: \sum_{i=0}^{n}
        elif '\\sum' in latex:
            return self._parse_sum(latex, source)
        
        # Set builder: \{x | P(x)\}
        elif '\\{' in latex and '|' in latex:
            return self._parse_set_builder(latex, source)
        
        # Binary operations
        elif any(op in latex for op in ['+', '-', '*', '/', '=', '<', '>']):
            return self._parse_binop(latex, source)
        
        # Variable or constant
        else:
            return self._parse_atom(latex, source)
    
    def _parse_forall(self, latex: str, source: SourceLocation) -> MathIRExpr:
        """Parse universal quantification"""
        # Simplified: extract variable and body
        # Real parser would use proper LaTeX parsing
        
        # Pattern: \forall x : T, P(x)
        var_name = "x"  # Extract from LaTeX
        var_type = IRVar("Nat")  # Infer or parse type
        body = IRVar("P_x")  # Parse body
        
        result = IRPi(var_name, var_type, body)
        result.source = source
        return result
    
    def _parse_subscript(self, latex: str, source: SourceLocation) -> MathIRExpr:
        """Parse subscript notation"""
        # Pattern: x_i or x_{i,j}
        parts = latex.split('_')
        base = IRVar(parts[0].strip())
        
        # Parse indices
        index_str = parts[1].strip('{}')
        indices = [IRVar(idx.strip()) for idx in index_str.split(',')]
        
        result = IRSubscript(base, indices)
        result.source = source
        return result
    
    def _parse_atom(self, latex: str, source: SourceLocation) -> MathIRExpr:
        """Parse atomic expression (variable or constant)"""
        latex = latex.strip()
        
        # Check if it's a number
        try:
            value = int(latex)
            return IRConst(value, MathIRSort.NAT, source=source)
        except:
            pass
        
        # Otherwise it's a variable
        return IRVar(latex, source=source)
    
    # Additional parsing methods would go here...
    def _parse_exists(self, latex: str, source: SourceLocation) -> MathIRExpr:
        pass
    
    def _parse_implication(self, latex: str, source: SourceLocation) -> MathIRExpr:
        pass
    
    def _parse_superscript(self, latex: str, source: SourceLocation) -> MathIRExpr:
        pass
    
    def _parse_sum(self, latex: str, source: SourceLocation) -> MathIRExpr:
        pass
    
    def _parse_set_builder(self, latex: str, source: SourceLocation) -> MathIRExpr:
        pass
    
    def _parse_binop(self, latex: str, source: SourceLocation) -> MathIRExpr:
        pass


# ============================================================================
# IR → LEAN TRANSLATOR
# ============================================================================

class IRToLean:
    """
    Translate MathIR to Lean 4 code.
    
    Handles:
    - Proper Unicode operators
    - Implicit arguments
    - Type class instances
    - Tactics for proofs
    """
    
    def translate(self, ir: MathIRExpr) -> str:
        """Translate IR expression to Lean code"""
        return ir.to_lean()
    
    def translate_theorem(self, name: str, statement: MathIRExpr, proof: Optional[MathIRExpr] = None) -> str:
        """Translate a complete theorem"""
        lean_statement = statement.to_lean()
        
        if proof:
            lean_proof = proof.to_lean()
            return f"theorem {name} : {lean_statement} := by\n  {lean_proof}"
        else:
            return f"theorem {name} : {lean_statement} := by\n  sorry"


# ============================================================================
# DEMONSTRATION & VALIDATION
# ============================================================================

def demonstrate_ir():
    """Demonstrate the IR with examples"""
    
    print("=" * 80)
    print("MATHEMATICAL IR: LaTeX → IR → Lean")
    print("=" * 80)
    print()
    
    print("Based on literature:")
    print("  - Kamareddine et al. (2004): MathLang")
    print("  - OpenMath / Content MathML")
    print("  - Ranta (1994): Grammatical Framework")
    print("  - Naproche (Creutz et al. 2021)")
    print()
    
    examples = [
        {
            'name': 'Universal Quantification',
            'latex': r'\forall n : \mathbb{N}, n \geq 0',
            'natural': 'for all natural numbers n, n ≥ 0',
        },
        {
            'name': 'Subscript Notation',
            'latex': r'x_i',
            'natural': 'x subscript i',
        },
        {
            'name': 'Function Application',
            'latex': r'f(x)',
            'natural': 'f of x',
        },
        {
            'name': 'Summation',
            'latex': r'\sum_{i=0}^{n} i',
            'natural': 'sum from i=0 to n of i',
        },
    ]
    
    print("-" * 80)
    print("EXAMPLES")
    print("-" * 80)
    print()
    
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex['name']}")
        print(f"   LaTeX:   {ex['latex']}")
        print(f"   Natural: {ex['natural']}")
        print()
    
    # Manual IR construction examples
    print("=" * 80)
    print("MANUAL IR CONSTRUCTION EXAMPLES")
    print("=" * 80)
    print()
    
    # Example 1: ∀ n : Nat, n ≥ 0
    print("1. Universal Quantification: ∀ n : Nat, n ≥ 0")
    n_var = IRVar("n")
    nat_type = IRVar("Nat")
    zero = IRConst(0, MathIRSort.NAT)
    body = IRBinOp("≥", n_var, zero)
    forall_expr = IRPi("n", nat_type, body)
    
    print(f"   IR: {forall_expr.to_json()}")
    print(f"   Lean: {forall_expr.to_lean()}")
    print()
    
    # Example 2: x_i > 0
    print("2. Subscript: x_i > 0")
    x = IRVar("x")
    i = IRVar("i")
    x_i = IRSubscript(x, [i], interpretation="indexed")
    zero = IRConst(0, MathIRSort.REAL)
    subscript_expr = IRBinOp(">", x_i, zero)
    
    print(f"   IR: {subscript_expr.to_json()}")
    print(f"   Lean: {subscript_expr.to_lean()}")
    print()
    
    # Example 3: λ x. x + 1
    print("3. Lambda: λ x. x + 1")
    x = IRVar("x")
    one = IRConst(1, MathIRSort.NAT)
    x_plus_1 = IRBinOp("+", x, one)
    lambda_expr = IRLambda("x", IRVar("Nat"), x_plus_1)
    
    print(f"   IR: {lambda_expr.to_json()}")
    print(f"   Lean: {lambda_expr.to_lean()}")
    print()
    
    # Example 4: {x : Nat | x > 0}
    print("4. Set Builder: {x : Nat | x > 0}")
    x = IRVar("x")
    zero = IRConst(0, MathIRSort.NAT)
    predicate = IRBinOp(">", x, zero)
    set_expr = IRSetBuilder("x", IRVar("Nat"), predicate)
    
    print(f"   IR: {set_expr.to_json()}")
    print(f"   Lean: {set_expr.to_lean()}")
    print()
    
    # Example 5: if P then Q
    print("5. Implication: P → Q")
    p = IRVar("P")
    q = IRVar("Q")
    impl_expr = IRBinOp("→", p, q)
    
    print(f"   IR: {impl_expr.to_json()}")
    print(f"   Lean: {impl_expr.to_lean()}")
    print()
    
    print("=" * 80)
    print("IR BENEFITS")
    print("=" * 80)
    print()
    print("1. Type-checkable: Catch errors before Lean compilation")
    print("2. Reusable: Same IR → Lean, Coq, Isabelle, Z3")
    print("3. Debuggable: Human-readable JSON representation")
    print("4. Compositional: Build complex expressions from simple ones")
    print("5. Provenance: Track source locations for error reporting")
    print("6. Canonical: Multiple LaTeX/NL → same IR → same Lean")
    print()


if __name__ == '__main__':
    demonstrate_ir()
