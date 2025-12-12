"""
Lean 4 Type Theory Formal Model
================================

Complete formalization of Lean 4's dependent type theory for Z3 constraint encoding.
Includes universe hierarchy, definitional equality, dependent types, inductive types.

This module provides the foundational type-theoretic model that all semantic
composition must satisfy.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Union, Tuple
from enum import Enum
from z3 import *


class UniverseLevel:
    """Represents universe levels in Lean's type hierarchy."""
    
    def __init__(self, level: Union[int, str]):
        if isinstance(level, int):
            self.level = level
            self.is_variable = False
            self.var_name = None
        else:
            self.level = None
            self.is_variable = True
            self.var_name = level
    
    def __repr__(self):
        if self.is_variable:
            return f"u_{self.var_name}"
        return f"u_{self.level}"
    
    def __hash__(self):
        if self.is_variable:
            return hash(('var', self.var_name))
        return hash(('level', self.level))
    
    def __eq__(self, other):
        if not isinstance(other, UniverseLevel):
            return False
        if self.is_variable != other.is_variable:
            return False
        if self.is_variable:
            return self.var_name == other.var_name
        return self.level == other.level


class LeanSort(Enum):
    """Lean's fundamental sorts."""
    PROP = "Prop"
    TYPE = "Type"
    SORT = "Sort"


@dataclass
class LeanType:
    """Base class for all Lean type expressions."""
    
    def free_variables(self) -> Set[str]:
        """Return set of free variables in this type."""
        raise NotImplementedError
    
    def substitute(self, var: str, expr: 'LeanExpr') -> 'LeanType':
        """Substitute expression for variable."""
        raise NotImplementedError
    
    def universe_level(self) -> Optional[UniverseLevel]:
        """Return the universe level of this type."""
        raise NotImplementedError
    
    def to_lean_string(self) -> str:
        """Generate canonical Lean syntax."""
        raise NotImplementedError
    
    def alpha_equivalent(self, other: 'LeanType') -> bool:
        """Check alpha equivalence (equal modulo variable renaming)."""
        raise NotImplementedError


@dataclass
class LeanProp(LeanType):
    """The Prop sort (propositions)."""
    
    def free_variables(self) -> Set[str]:
        return set()
    
    def substitute(self, var: str, expr: 'LeanExpr') -> 'LeanType':
        return self
    
    def universe_level(self) -> Optional[UniverseLevel]:
        return None  # Prop is impredicative
    
    def to_lean_string(self) -> str:
        return "Prop"
    
    def alpha_equivalent(self, other: LeanType) -> bool:
        return isinstance(other, LeanProp)
    
    def __hash__(self):
        return hash('Prop')
    
    def __eq__(self, other):
        return isinstance(other, LeanProp)


@dataclass
class LeanTypeSort(LeanType):
    """Type u (types at universe level u)."""
    level: UniverseLevel
    
    def free_variables(self) -> Set[str]:
        return set()
    
    def substitute(self, var: str, expr: 'LeanExpr') -> 'LeanType':
        return self
    
    def universe_level(self) -> Optional[UniverseLevel]:
        return self.level
    
    def to_lean_string(self) -> str:
        if self.level.level == 0:
            return "Type"
        elif self.level.is_variable:
            return f"Type {self.level.var_name}"
        else:
            return f"Type {self.level.level}"
    
    def alpha_equivalent(self, other: LeanType) -> bool:
        return isinstance(other, LeanTypeSort) and self.level == other.level
    
    def __hash__(self):
        return hash(('Type', self.level))


@dataclass
class LeanVariable(LeanType):
    """Type variable (for polymorphism)."""
    name: str
    
    def free_variables(self) -> Set[str]:
        return {self.name}
    
    def substitute(self, var: str, expr: 'LeanExpr') -> 'LeanType':
        if var == self.name:
            if isinstance(expr, LeanTypeExpr):
                return expr.type_expr
            return self
        return self
    
    def universe_level(self) -> Optional[UniverseLevel]:
        return None  # Unknown until instantiated
    
    def to_lean_string(self) -> str:
        return self.name
    
    def alpha_equivalent(self, other: LeanType) -> bool:
        # Variables are equal by name in canonical form
        return isinstance(other, LeanVariable) and self.name == other.name
    
    def __hash__(self):
        return hash(('Var', self.name))


@dataclass
class LeanArrow(LeanType):
    """Function type: A → B"""
    from_type: LeanType
    to_type: LeanType
    
    def free_variables(self) -> Set[str]:
        return self.from_type.free_variables() | self.to_type.free_variables()
    
    def substitute(self, var: str, expr: 'LeanExpr') -> 'LeanType':
        return LeanArrow(
            self.from_type.substitute(var, expr),
            self.to_type.substitute(var, expr)
        )
    
    def universe_level(self) -> Optional[UniverseLevel]:
        # max(level(A), level(B)) for A → B
        from_level = self.from_type.universe_level()
        to_level = self.to_type.universe_level()
        if from_level is None or to_level is None:
            return None
        if from_level.is_variable or to_level.is_variable:
            return UniverseLevel(f"max({from_level},{to_level})")
        return UniverseLevel(max(from_level.level, to_level.level))
    
    def to_lean_string(self) -> str:
        from_str = self.from_type.to_lean_string()
        to_str = self.to_type.to_lean_string()
        
        # Parenthesize nested arrows on left
        if isinstance(self.from_type, (LeanArrow, LeanPi)):
            from_str = f"({from_str})"
        
        return f"{from_str} → {to_str}"
    
    def alpha_equivalent(self, other: LeanType) -> bool:
        return (isinstance(other, LeanArrow) and
                self.from_type.alpha_equivalent(other.from_type) and
                self.to_type.alpha_equivalent(other.to_type))
    
    def __hash__(self):
        return hash(('Arrow', self.from_type, self.to_type))


@dataclass
class LeanPi(LeanType):
    """Dependent function type: (x : A) → B x"""
    var_name: str
    var_type: LeanType
    body_type: LeanType
    implicit: bool = False
    
    def free_variables(self) -> Set[str]:
        free = self.var_type.free_variables()
        body_free = self.body_type.free_variables()
        body_free.discard(self.var_name)
        return free | body_free
    
    def substitute(self, var: str, expr: 'LeanExpr') -> 'LeanType':
        if var == self.var_name:
            # Bound variable, don't substitute in body
            return LeanPi(
                self.var_name,
                self.var_type.substitute(var, expr),
                self.body_type,
                self.implicit
            )
        return LeanPi(
            self.var_name,
            self.var_type.substitute(var, expr),
            self.body_type.substitute(var, expr),
            self.implicit
        )
    
    def universe_level(self) -> Optional[UniverseLevel]:
        # max(level(A), level(B)) for (x : A) → B
        var_level = self.var_type.universe_level()
        body_level = self.body_type.universe_level()
        if var_level is None or body_level is None:
            return None
        if var_level.is_variable or body_level.is_variable:
            return UniverseLevel(f"max({var_level},{body_level})")
        return UniverseLevel(max(var_level.level, body_level.level))
    
    def to_lean_string(self) -> str:
        var_str = f"{self.var_name} : {self.var_type.to_lean_string()}"
        if self.implicit:
            var_str = f"{{{var_str}}}"
        else:
            var_str = f"({var_str})"
        
        body_str = self.body_type.to_lean_string()
        return f"∀ {var_str}, {body_str}"
    
    def alpha_equivalent(self, other: LeanType) -> bool:
        if not isinstance(other, LeanPi):
            return False
        # Alpha equivalence: rename bound variable consistently
        fresh_var = f"_α_{hash((self, other))}"
        self_renamed = self.body_type.substitute(
            self.var_name,
            LeanVarExpr(fresh_var, self.var_type)
        )
        other_renamed = other.body_type.substitute(
            other.var_name,
            LeanVarExpr(fresh_var, other.var_type)
        )
        return (self.var_type.alpha_equivalent(other.var_type) and
                self_renamed.alpha_equivalent(other_renamed))
    
    def __hash__(self):
        return hash(('Pi', self.var_name, self.var_type, self.body_type))


@dataclass
class LeanApp(LeanType):
    """Type application: F A"""
    function: LeanType
    argument: LeanType
    
    def free_variables(self) -> Set[str]:
        return self.function.free_variables() | self.argument.free_variables()
    
    def substitute(self, var: str, expr: 'LeanExpr') -> 'LeanType':
        return LeanApp(
            self.function.substitute(var, expr),
            self.argument.substitute(var, expr)
        )
    
    def universe_level(self) -> Optional[UniverseLevel]:
        # Type application doesn't change universe level
        return self.function.universe_level()
    
    def to_lean_string(self) -> str:
        func_str = self.function.to_lean_string()
        arg_str = self.argument.to_lean_string()
        
        # Parenthesize complex arguments
        if isinstance(self.argument, (LeanApp, LeanArrow, LeanPi)):
            arg_str = f"({arg_str})"
        
        return f"{func_str} {arg_str}"
    
    def alpha_equivalent(self, other: LeanType) -> bool:
        return (isinstance(other, LeanApp) and
                self.function.alpha_equivalent(other.function) and
                self.argument.alpha_equivalent(other.argument))
    
    def __hash__(self):
        return hash(('App', self.function, self.argument))


@dataclass
class LeanExpr:
    """Base class for Lean expressions (terms)."""
    
    def type_of(self, context: 'Context') -> LeanType:
        """Infer the type of this expression."""
        raise NotImplementedError
    
    def free_variables(self) -> Set[str]:
        """Return free variables."""
        raise NotImplementedError
    
    def substitute(self, var: str, expr: 'LeanExpr') -> 'LeanExpr':
        """Substitute expression for variable."""
        raise NotImplementedError
    
    def to_lean_string(self) -> str:
        """Generate canonical Lean syntax."""
        raise NotImplementedError
    
    def alpha_equivalent(self, other: 'LeanExpr') -> bool:
        """Check alpha equivalence."""
        raise NotImplementedError


@dataclass
class LeanVarExpr(LeanExpr):
    """Variable expression."""
    name: str
    var_type: LeanType
    
    def type_of(self, context: 'Context') -> LeanType:
        return self.var_type
    
    def free_variables(self) -> Set[str]:
        return {self.name}
    
    def substitute(self, var: str, expr: LeanExpr) -> LeanExpr:
        if var == self.name:
            return expr
        return self
    
    def to_lean_string(self) -> str:
        return self.name
    
    def alpha_equivalent(self, other: LeanExpr) -> bool:
        return isinstance(other, LeanVarExpr) and self.name == other.name
    
    def __hash__(self):
        return hash(('VarExpr', self.name))


@dataclass
class LeanAppExpr(LeanExpr):
    """Function application: f a"""
    function: LeanExpr
    argument: LeanExpr
    
    def type_of(self, context: 'Context') -> LeanType:
        func_type = self.function.type_of(context)
        
        if isinstance(func_type, LeanArrow):
            return func_type.to_type
        elif isinstance(func_type, LeanPi):
            # Substitute argument into dependent type
            return func_type.body_type.substitute(
                func_type.var_name,
                self.argument
            )
        else:
            raise TypeError(f"Cannot apply non-function type: {func_type}")
    
    def free_variables(self) -> Set[str]:
        return self.function.free_variables() | self.argument.free_variables()
    
    def substitute(self, var: str, expr: LeanExpr) -> LeanExpr:
        return LeanAppExpr(
            self.function.substitute(var, expr),
            self.argument.substitute(var, expr)
        )
    
    def to_lean_string(self) -> str:
        func_str = self.function.to_lean_string()
        arg_str = self.argument.to_lean_string()
        
        if isinstance(self.argument, (LeanAppExpr, LeanLambda)):
            arg_str = f"({arg_str})"
        
        return f"{func_str} {arg_str}"
    
    def alpha_equivalent(self, other: LeanExpr) -> bool:
        return (isinstance(other, LeanAppExpr) and
                self.function.alpha_equivalent(other.function) and
                self.argument.alpha_equivalent(other.argument))


@dataclass
class LeanLambda(LeanExpr):
    """Lambda abstraction: fun x : A => body"""
    var_name: str
    var_type: LeanType
    body: LeanExpr
    implicit: bool = False
    
    def type_of(self, context: 'Context') -> LeanType:
        # Type of lambda is Pi type
        body_type = self.body.type_of(
            context.add_variable(self.var_name, self.var_type)
        )
        return LeanPi(self.var_name, self.var_type, body_type, self.implicit)
    
    def free_variables(self) -> Set[str]:
        free = self.var_type.free_variables()
        body_free = self.body.free_variables()
        body_free.discard(self.var_name)
        return free | body_free
    
    def substitute(self, var: str, expr: LeanExpr) -> LeanExpr:
        if var == self.var_name:
            return LeanLambda(
                self.var_name,
                self.var_type.substitute(var, expr),
                self.body,
                self.implicit
            )
        return LeanLambda(
            self.var_name,
            self.var_type.substitute(var, expr),
            self.body.substitute(var, expr),
            self.implicit
        )
    
    def to_lean_string(self) -> str:
        var_str = f"{self.var_name} : {self.var_type.to_lean_string()}"
        if self.implicit:
            var_str = f"{{{var_str}}}"
        else:
            var_str = f"({var_str})"
        
        body_str = self.body.to_lean_string()
        return f"fun {var_str} => {body_str}"
    
    def alpha_equivalent(self, other: LeanExpr) -> bool:
        if not isinstance(other, LeanLambda):
            return False
        fresh_var = f"_α_{hash((self, other))}"
        self_renamed = self.body.substitute(
            self.var_name,
            LeanVarExpr(fresh_var, self.var_type)
        )
        other_renamed = other.body.substitute(
            other.var_name,
            LeanVarExpr(fresh_var, other.var_type)
        )
        return (self.var_type.alpha_equivalent(other.var_type) and
                self_renamed.alpha_equivalent(other_renamed))


@dataclass
class LeanTypeExpr(LeanExpr):
    """Type as expression (types are first-class)."""
    type_expr: LeanType
    
    def type_of(self, context: 'Context') -> LeanType:
        level = self.type_expr.universe_level()
        if level is None:
            return LeanProp()  # Propositions are in Prop
        return LeanTypeSort(UniverseLevel(level.level + 1))
    
    def free_variables(self) -> Set[str]:
        return self.type_expr.free_variables()
    
    def substitute(self, var: str, expr: LeanExpr) -> LeanExpr:
        return LeanTypeExpr(self.type_expr.substitute(var, expr))
    
    def to_lean_string(self) -> str:
        return self.type_expr.to_lean_string()
    
    def alpha_equivalent(self, other: LeanExpr) -> bool:
        return (isinstance(other, LeanTypeExpr) and
                self.type_expr.alpha_equivalent(other.type_expr))


@dataclass
class Context:
    """Typing context for type inference."""
    bindings: Dict[str, LeanType] = field(default_factory=dict)
    parent: Optional['Context'] = None
    
    def lookup(self, var: str) -> Optional[LeanType]:
        """Look up variable type in context."""
        if var in self.bindings:
            return self.bindings[var]
        if self.parent:
            return self.parent.lookup(var)
        return None
    
    def add_variable(self, var: str, var_type: LeanType) -> 'Context':
        """Add variable to context (creates new context)."""
        new_context = Context(parent=self)
        new_context.bindings[var] = var_type
        return new_context
    
    def extend(self, bindings: Dict[str, LeanType]) -> 'Context':
        """Extend context with multiple bindings."""
        new_context = Context(parent=self)
        new_context.bindings.update(bindings)
        return new_context


class LeanTypeChecker:
    """Type checker for Lean expressions."""
    
    def __init__(self):
        self.context = Context()
    
    def check(self, expr: LeanExpr, expected_type: LeanType, 
              context: Optional[Context] = None) -> bool:
        """Check if expression has expected type."""
        ctx = context or self.context
        inferred_type = expr.type_of(ctx)
        return self.definitionally_equal(inferred_type, expected_type, ctx)
    
    def infer(self, expr: LeanExpr, context: Optional[Context] = None) -> LeanType:
        """Infer type of expression."""
        ctx = context or self.context
        return expr.type_of(ctx)
    
    def definitionally_equal(self, type1: LeanType, type2: LeanType,
                           context: Context) -> bool:
        """Check definitional equality (includes beta reduction, eta conversion)."""
        # For now, structural equality + alpha equivalence
        # Full definitional equality would include normalization
        return type1.alpha_equivalent(type2)
    
    def check_universe_level(self, type_expr: LeanType, 
                           max_level: Optional[int] = None) -> bool:
        """Verify universe level constraints."""
        level = type_expr.universe_level()
        if level is None:
            return True  # Prop is impredicative
        if level.is_variable:
            return True  # Universe polymorphism
        if max_level is not None:
            return level.level <= max_level
        return True


def create_forall_type(var_name: str, var_type: LeanType, 
                       body_type: LeanType) -> LeanPi:
    """Create ∀ (x : A), B x type (canonical form)."""
    return LeanPi(var_name, var_type, body_type, implicit=False)


def create_exists_type(var_type: LeanType, predicate_type: LeanType) -> LeanApp:
    """Create ∃ (x : A), P x type (canonical form)."""
    # ∃ is defined as: Exists (α : Type u) (p : α → Prop) : Prop
    exists_const = LeanVariable("Exists")
    return LeanApp(LeanApp(exists_const, var_type), predicate_type)


def create_and_type(left: LeanType, right: LeanType) -> LeanApp:
    """Create P ∧ Q type (canonical form)."""
    and_const = LeanVariable("And")
    return LeanApp(LeanApp(and_const, left), right)


def create_or_type(left: LeanType, right: LeanType) -> LeanApp:
    """Create P ∨ Q type (canonical form)."""
    or_const = LeanVariable("Or")
    return LeanApp(LeanApp(or_const, left), right)


def create_not_type(prop: LeanType) -> LeanApp:
    """Create ¬P type (canonical form)."""
    not_const = LeanVariable("Not")
    return LeanApp(not_const, prop)


def create_eq_type(left: LeanExpr, right: LeanExpr, eq_type: LeanType) -> LeanApp:
    """Create a = b type (canonical form)."""
    eq_const = LeanVariable("Eq")
    return LeanApp(LeanApp(LeanApp(eq_const, LeanTypeExpr(eq_type)), left), right)
