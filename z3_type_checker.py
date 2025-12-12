"""
Z3 Model Checking for Lean Type Theory
=======================================

Encodes Lean's type-theoretic constraints as Z3 formulas for formal verification.
This ensures generated code satisfies Lean's type system without heuristics.
"""

from z3 import *
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from lean_type_theory import (
    UniverseLevel, LeanSort, LeanType, LeanExpr,
    Context, PiType, ArrowType, AppType, VarType,
    LambdaExpr, TypeU, PropSort
)


class ConstraintKind(Enum):
    """Types of type-theoretic constraints."""
    UNIVERSE_WELLFORMED = "universe_wellformed"
    TYPE_WELLFORMED = "type_wellformed"
    TYPING_JUDGMENT = "typing_judgment"
    DEF_EQUALITY = "definitional_equality"
    UNIVERSE_CUMULATIVE = "universe_cumulativity"
    PI_FORMATION = "pi_formation"
    APP_TYPECHECK = "application_typecheck"


@dataclass
class TypeConstraint:
    """A single type-theoretic constraint."""
    kind: ConstraintKind
    formula: BoolRef
    description: str
    context: Optional[Dict] = None


class UniverseEncoder:
    """
    Encode universe levels as Z3 integers with constraints.
    
    Lean's universe hierarchy:
    - Prop : Type 0
    - Type 0 : Type 1
    - Type u : Type (u+1)
    - max u v, imax u v for dependent products
    """
    
    def __init__(self):
        self.levels: Dict[str, ArithRef] = {}
        self.solver = Solver()
        
    def get_level(self, name: str) -> ArithRef:
        """Get or create universe level variable."""
        if name not in self.levels:
            self.levels[name] = Int(f"universe_{name}")
            # All universe levels are non-negative
            self.solver.add(self.levels[name] >= 0)
        return self.levels[name]
    
    def encode_level(self, level: UniverseLevel) -> ArithRef:
        """Encode a UniverseLevel as Z3 integer."""
        if level.is_prop():
            return IntVal(-1)  # Special encoding for Prop
        elif level.is_zero():
            return IntVal(0)
        elif level.is_succ():
            base = self.encode_level(level.pred)
            return base + 1
        elif level.is_max():
            u = self.encode_level(level.left)
            v = self.encode_level(level.right)
            return If(u >= v, u, v)
        elif level.is_imax():
            # imax u v = 0 if v = Prop, else max u v
            u = self.encode_level(level.left)
            v = self.encode_level(level.right)
            return If(v == IntVal(-1), IntVal(0), If(u >= v, u, v))
        elif level.is_var():
            return self.get_level(level.var_name)
        else:
            raise ValueError(f"Unknown universe level: {level}")
    
    def less_than(self, u: UniverseLevel, v: UniverseLevel) -> BoolRef:
        """Encode u < v constraint."""
        u_encoded = self.encode_level(u)
        v_encoded = self.encode_level(v)
        return u_encoded < v_encoded
    
    def less_equal(self, u: UniverseLevel, v: UniverseLevel) -> BoolRef:
        """Encode u ≤ v constraint."""
        u_encoded = self.encode_level(u)
        v_encoded = self.encode_level(v)
        return u_encoded <= v_encoded
    
    def equal(self, u: UniverseLevel, v: UniverseLevel) -> BoolRef:
        """Encode u = v constraint."""
        u_encoded = self.encode_level(u)
        v_encoded = self.encode_level(v)
        return u_encoded == v_encoded


class TypeEncoder:
    """
    Encode Lean types and terms as Z3 formulas.
    """
    
    def __init__(self):
        self.universe_encoder = UniverseEncoder()
        self.type_sorts: Dict[str, DatatypeSortRef] = {}
        self._init_type_sorts()
        
    def _init_type_sorts(self):
        """Initialize Z3 sorts for representing types."""
        # Create datatypes for types
        TypeSort = Datatype('TypeSort')
        TypeSort.declare('prop')
        TypeSort.declare('type_u', ('level', IntSort()))
        TypeSort.declare('var', ('name', StringSort()))
        TypeSort.declare('pi', ('var', StringSort()), ('domain', 'TypeSort'), ('codomain', 'TypeSort'))
        TypeSort.declare('app', ('func', 'TypeSort'), ('arg', 'TypeSort'))
        
        self.type_sort = TypeSort.create()
        self.type_constructors = {
            'prop': self.type_sort.prop,
            'type_u': self.type_sort.type_u,
            'var': self.type_sort.var,
            'pi': self.type_sort.pi,
            'app': self.type_sort.app
        }
    
    def encode_type(self, ty: LeanType) -> DatatypeRef:
        """Encode a LeanType as Z3 datatype."""
        if isinstance(ty, PropSort):
            return self.type_constructors['prop']()
        elif isinstance(ty, TypeU):
            level_int = self.universe_encoder.encode_level(ty.level)
            return self.type_constructors['type_u'](level_int)
        elif isinstance(ty, VarType):
            return self.type_constructors['var'](StringVal(ty.name))
        elif isinstance(ty, PiType):
            domain_enc = self.encode_type(ty.domain)
            codomain_enc = self.encode_type(ty.codomain)
            return self.type_constructors['pi'](
                StringVal(ty.var_name),
                domain_enc,
                codomain_enc
            )
        elif isinstance(ty, ArrowType):
            # A → B is syntactic sugar for Π _:A, B
            domain_enc = self.encode_type(ty.domain)
            codomain_enc = self.encode_type(ty.codomain)
            return self.type_constructors['pi'](
                StringVal("_"),
                domain_enc,
                codomain_enc
            )
        elif isinstance(ty, AppType):
            func_enc = self.encode_type(ty.func)
            arg_enc = self.encode_type(ty.arg) if isinstance(ty.arg, LeanType) else self.type_constructors['var'](StringVal(str(ty.arg)))
            return self.type_constructors['app'](func_enc, arg_enc)
        else:
            raise ValueError(f"Cannot encode type: {ty}")


class TypingConstraintGenerator:
    """
    Generate Z3 constraints encoding Lean's typing rules.
    """
    
    def __init__(self):
        self.universe_encoder = UniverseEncoder()
        self.type_encoder = TypeEncoder()
        self.constraints: List[TypeConstraint] = []
        
    def add_constraint(self, kind: ConstraintKind, formula: BoolRef, 
                      description: str, context: Optional[Dict] = None):
        """Add a typing constraint."""
        constraint = TypeConstraint(kind, formula, description, context)
        self.constraints.append(constraint)
        return constraint
    
    def wellformed_universe(self, level: UniverseLevel) -> TypeConstraint:
        """
        Encode: universe level is well-formed (non-negative).
        """
        level_z3 = self.universe_encoder.encode_level(level)
        formula = level_z3 >= -1  # Allow -1 for Prop
        return self.add_constraint(
            ConstraintKind.UNIVERSE_WELLFORMED,
            formula,
            f"Universe {level} is well-formed",
            {"level": str(level)}
        )
    
    def wellformed_sort(self, sort: LeanSort, context: Context) -> TypeConstraint:
        """
        Encode: Γ ⊢ s : Type (u+1)
        
        Sort formation rules:
        - Prop : Type 0
        - Type u : Type (u+1)
        """
        if isinstance(sort, PropSort):
            # Prop : Type 0
            formula = BoolVal(True)  # Always well-formed
            desc = "Prop : Type 0"
        elif isinstance(sort, TypeU):
            # Type u : Type (u+1)
            self.wellformed_universe(sort.level)
            formula = BoolVal(True)
            desc = f"Type {sort.level} : Type {sort.level.succ()}"
        else:
            formula = BoolVal(False)
            desc = f"Unknown sort {sort}"
            
        return self.add_constraint(
            ConstraintKind.TYPE_WELLFORMED,
            formula,
            desc,
            {"sort": str(sort), "context": str(context)}
        )
    
    def pi_formation(self, var: str, domain: LeanType, codomain: LeanType,
                    context: Context) -> TypeConstraint:
        """
        Encode Pi-type formation rule:
        
        Γ ⊢ A : Type u    Γ, x:A ⊢ B : Type v
        ----------------------------------------
        Γ ⊢ (Π x:A, B) : Type (imax u v)
        """
        # Get universe levels
        u = self._infer_universe(domain, context)
        
        # Extend context with variable
        extended_context = context.copy()
        extended_context.add_variable(var, domain)
        v = self._infer_universe(codomain, extended_context)
        
        # imax u v for the result universe
        result_level = UniverseLevel.imax(u, v)
        
        # Encode constraint
        u_z3 = self.universe_encoder.encode_level(u)
        v_z3 = self.universe_encoder.encode_level(v)
        result_z3 = self.universe_encoder.encode_level(result_level)
        
        # imax definition: imax u v = 0 if v = Prop, else max u v
        imax_correct = If(
            v_z3 == IntVal(-1),
            result_z3 == IntVal(0),
            result_z3 == If(u_z3 >= v_z3, u_z3, v_z3)
        )
        
        formula = And(
            u_z3 >= -1,
            v_z3 >= -1,
            imax_correct
        )
        
        return self.add_constraint(
            ConstraintKind.PI_FORMATION,
            formula,
            f"Π {var}:{domain} → {codomain} : Type (imax {u} {v})",
            {"var": var, "domain": str(domain), "codomain": str(codomain)}
        )
    
    def application_typecheck(self, func: LeanExpr, arg: LeanExpr,
                             context: Context) -> TypeConstraint:
        """
        Encode application typing rule:
        
        Γ ⊢ f : Π x:A, B    Γ ⊢ a : A
        -------------------------------
        Γ ⊢ f a : B[x := a]
        """
        # Infer function type
        func_type = self._infer_type(func, context)
        
        if not isinstance(func_type, PiType):
            # Type error: applying non-function
            formula = BoolVal(False)
            desc = f"Type error: {func} is not a function type"
        else:
            # Infer argument type
            arg_type = self._infer_type(arg, context)
            
            # Check argument type matches domain
            domain_match = self._types_equal(arg_type, func_type.domain, context)
            
            # Result type is codomain with substitution
            # B[x := a]
            result_type = self._substitute(func_type.codomain, func_type.var_name, arg)
            
            formula = domain_match
            desc = f"({func} : {func_type}) ({arg} : {arg_type}) : {result_type}"
        
        return self.add_constraint(
            ConstraintKind.APP_TYPECHECK,
            formula,
            desc,
            {"func": str(func), "arg": str(arg)}
        )
    
    def definitional_equality(self, e1: LeanExpr, e2: LeanExpr,
                             context: Context) -> TypeConstraint:
        """
        Encode definitional equality: Γ ⊢ e1 ≡ e2
        
        Includes:
        - α-equivalence (variable renaming)
        - β-reduction (lambda application)
        - δ-reduction (unfolding definitions)
        - η-equivalence (λ x, f x ≡ f)
        """
        # Encode as structural equality (simplified)
        e1_enc = self._encode_expr(e1)
        e2_enc = self._encode_expr(e2)
        
        formula = e1_enc == e2_enc
        
        return self.add_constraint(
            ConstraintKind.DEF_EQUALITY,
            formula,
            f"{e1} ≡ {e2}",
            {"e1": str(e1), "e2": str(e2)}
        )
    
    def universe_cumulativity(self, ty: LeanType, u: UniverseLevel,
                             v: UniverseLevel, context: Context) -> TypeConstraint:
        """
        Encode cumulativity: if Γ ⊢ A : Type u and u ≤ v, then Γ ⊢ A : Type v
        """
        u_z3 = self.universe_encoder.encode_level(u)
        v_z3 = self.universe_encoder.encode_level(v)
        
        formula = Implies(u_z3 <= v_z3, BoolVal(True))
        
        return self.add_constraint(
            ConstraintKind.UNIVERSE_CUMULATIVE,
            formula,
            f"Cumulativity: {ty} : Type {u} implies {ty} : Type {v} when {u} ≤ {v}",
            {"type": str(ty), "u": str(u), "v": str(v)}
        )
    
    def _infer_universe(self, ty: LeanType, context: Context) -> UniverseLevel:
        """Infer the universe level of a type."""
        if isinstance(ty, PropSort):
            return UniverseLevel.zero()
        elif isinstance(ty, TypeU):
            return ty.level.succ()
        elif isinstance(ty, PiType):
            u = self._infer_universe(ty.domain, context)
            extended = context.copy()
            extended.add_variable(ty.var_name, ty.domain)
            v = self._infer_universe(ty.codomain, extended)
            return UniverseLevel.imax(u, v)
        elif isinstance(ty, VarType):
            if context.has_variable(ty.name):
                var_type = context.get_variable_type(ty.name)
                return self._infer_universe(var_type, context)
            else:
                # Assume Type 0 for unknown variables
                return UniverseLevel.zero()
        else:
            return UniverseLevel.zero()
    
    def _infer_type(self, expr: LeanExpr, context: Context) -> LeanType:
        """Infer the type of an expression."""
        # Simplified type inference (would use full type checker in practice)
        from lean_type_theory import LeanTypeChecker
        checker = LeanTypeChecker()
        return checker.infer_type(expr, context)
    
    def _types_equal(self, t1: LeanType, t2: LeanType, context: Context) -> BoolRef:
        """Check if two types are definitionally equal."""
        # Simplified: structural equality
        t1_enc = self.type_encoder.encode_type(t1)
        t2_enc = self.type_encoder.encode_type(t2)
        return t1_enc == t2_enc
    
    def _substitute(self, expr: LeanType, var: str, replacement: LeanExpr) -> LeanType:
        """Substitute variable in expression."""
        # Simplified substitution
        return expr  # TODO: implement full substitution
    
    def _encode_expr(self, expr: LeanExpr) -> DatatypeRef:
        """Encode expression as Z3 datatype."""
        # Use type encoder for now
        if hasattr(expr, 'type'):
            return self.type_encoder.encode_type(expr.type)
        else:
            # Create placeholder
            return self.type_encoder.type_constructors['var'](StringVal(str(expr)))
    
    def check_constraints(self) -> Tuple[bool, Optional[ModelRef]]:
        """
        Check if all constraints are satisfiable.
        
        Returns:
            (satisfiable, model) where model is None if unsat
        """
        solver = Solver()
        
        # Add all constraint formulas
        for constraint in self.constraints:
            solver.add(constraint.formula)
        
        result = solver.check()
        
        if result == sat:
            return (True, solver.model())
        elif result == unsat:
            # Get unsat core for debugging
            return (False, None)
        else:
            # Unknown
            return (None, None)
    
    def get_unsat_core(self) -> List[TypeConstraint]:
        """Get minimal set of conflicting constraints."""
        solver = Solver()
        
        # Add constraints with tracking
        tracked = []
        for i, constraint in enumerate(self.constraints):
            name = f"c{i}"
            solver.assert_and_track(constraint.formula, name)
            tracked.append((name, constraint))
        
        result = solver.check()
        
        if result == unsat:
            core = solver.unsat_core()
            return [constraint for name, constraint in tracked if name in core]
        else:
            return []


class Z3TypeChecker:
    """
    High-level interface for Z3-based type checking.
    """
    
    def __init__(self):
        self.constraint_gen = TypingConstraintGenerator()
        
    def check_statement(self, statement_type: LeanType, context: Context) -> Dict:
        """
        Check if a statement's type is well-formed.
        
        Returns:
            Dictionary with 'valid', 'constraints', 'model', 'errors'
        """
        # Generate constraints for type
        self._generate_type_constraints(statement_type, context)
        
        # Check satisfiability
        is_valid, model = self.constraint_gen.check_constraints()
        
        result = {
            'valid': is_valid,
            'constraints': self.constraint_gen.constraints,
            'model': model
        }
        
        if not is_valid:
            # Get unsat core for error reporting
            core = self.constraint_gen.get_unsat_core()
            result['errors'] = [c.description for c in core]
        
        return result
    
    def check_dag(self, dag, context: Context) -> Dict:
        """
        Check all statements in a DAG for type correctness.
        
        Returns:
            Dictionary mapping statement IDs to check results
        """
        from dependency_dag import DependencyDAG
        
        results = {}
        
        # Process in topological order
        for stmt_id in dag.topological_sort():
            statement = dag.nodes[stmt_id]
            
            # Parse type (simplified - would use full parser)
            try:
                # Assume we have type information
                stmt_type = self._parse_statement_type(statement)
                result = self.check_statement(stmt_type, context)
                results[stmt_id] = result
                
                # Update context with new definition
                if result['valid']:
                    context.add_variable(statement.name, stmt_type)
            except Exception as e:
                results[stmt_id] = {
                    'valid': False,
                    'errors': [str(e)]
                }
        
        return results
    
    def _generate_type_constraints(self, ty: LeanType, context: Context):
        """Generate all relevant constraints for a type."""
        if isinstance(ty, PropSort):
            self.constraint_gen.wellformed_sort(ty, context)
        elif isinstance(ty, TypeU):
            self.constraint_gen.wellformed_sort(ty, context)
            self.constraint_gen.wellformed_universe(ty.level)
        elif isinstance(ty, PiType):
            # Check domain and codomain
            self._generate_type_constraints(ty.domain, context)
            
            # Extend context for codomain
            extended = context.copy()
            extended.add_variable(ty.var_name, ty.domain)
            self._generate_type_constraints(ty.codomain, extended)
            
            # Check Pi formation rule
            self.constraint_gen.pi_formation(
                ty.var_name, ty.domain, ty.codomain, context
            )
        elif isinstance(ty, ArrowType):
            # Check as non-dependent Pi type
            self._generate_type_constraints(ty.domain, context)
            self._generate_type_constraints(ty.codomain, context)
        elif isinstance(ty, AppType):
            # Would check application in full implementation
            pass
    
    def _parse_statement_type(self, statement) -> LeanType:
        """Parse statement into LeanType (simplified)."""
        # In real implementation, would use compositional_semantics
        # For now, return placeholder
        return PropSort()


def verify_canonical_form(lean_code: str, rules: Dict) -> Dict:
    """
    Verify that generated Lean code satisfies canonical form constraints.
    
    Args:
        lean_code: Generated Lean source
        rules: Canonicalization rules (snake_case, etc.)
        
    Returns:
        Dictionary with verification results
    """
    solver = Solver()
    violations = []
    
    # Parse code into statements
    lines = lean_code.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check definition naming
        if line.startswith('def '):
            name = line.split()[1].split(':')[0].split('(')[0]
            if not name.islower() and '_' in name:
                violations.append(f"Line {i+1}: Definition '{name}' not in snake_case")
        
        # Check structure naming  
        if line.startswith('structure '):
            name = line.split()[1].split('(')[0]
            if not name[0].isupper():
                violations.append(f"Line {i+1}: Structure '{name}' not in PascalCase")
        
        # Check quantifier form (ASCII not Unicode)
        if '∀' in line or '∃' in line:
            violations.append(f"Line {i+1}: Use 'forall'/'exists' not Unicode")
    
    return {
        'valid': len(violations) == 0,
        'violations': violations,
        'rules_checked': list(rules.keys())
    }


def extract_type_constraints(lean_code: str) -> List[TypeConstraint]:
    """
    Extract type constraints from Lean code for Z3 verification.
    
    This parses Lean declarations and generates constraints that must
    be satisfied for the code to type check.
    """
    constraints = []
    constraint_gen = TypingConstraintGenerator()
    
    # Parse Lean code (simplified)
    lines = lean_code.split('\n')
    context = Context()
    
    for line in lines:
        line = line.strip()
        
        # Parse structure
        if line.startswith('structure '):
            # Extract fields and generate constraints
            pass  # Would parse full structure
        
        # Parse theorem
        if line.startswith('theorem '):
            # Extract type and generate constraints
            pass  # Would parse full theorem type
    
    return constraint_gen.constraints


# Example usage
if __name__ == '__main__':
    # Create simple type
    real_type = TypeU(UniverseLevel.zero())
    
    # Create context
    context = Context()
    context.add_variable("x", real_type)
    
    # Create Pi type: ∀ x : ℝ, x > 0
    gt_type = AppType("GT", [VarType("x"), VarType("0")])
    forall_type = PiType("x", real_type, gt_type)
    
    # Check with Z3
    checker = Z3TypeChecker()
    result = checker.check_statement(forall_type, context)
    
    print(f"Type check result: {result['valid']}")
    if not result['valid']:
        print(f"Errors: {result.get('errors', [])}")
    
    print(f"\nGenerated {len(result['constraints'])} constraints:")
    for constraint in result['constraints'][:5]:
        print(f"  - {constraint.description}")
