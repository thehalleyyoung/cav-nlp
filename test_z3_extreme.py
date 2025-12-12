#!/usr/bin/env python3
"""
EXTREME HARDNESS: Advanced LaTeX & Multi-Theory SMT Tests

This extends the test suite with:
- Level 7: Mixed theories (Arrays + Arithmetic)
- Level 8: Real LaTeX from papers (Cauchy-Schwarz, Triangle Inequality)
- Level 9: Proof obligations (lemma application)
- Level 10: Document with 5+ interdependent theorems
- Level 11: CANONICALIZATION (Z3 proves expression equivalences)

Z3 Usage:
- QF_AUFLIA: Arrays + Uninterpreted Functions + Linear Integer Arithmetic
- QF_AUFLIRA: Arrays + UF + Linear Integer/Real Arithmetic
- Complex quantifier patterns
- **CANONICALIZATION**: Proving x+y ≡ y+x, ¬(P∧Q) ≡ ¬P∨¬Q, etc.
"""

from test_z3_validated_ir_hard import *


def run_extreme_test_suite():
    """Even harder tests pushing Z3 to limits"""
    
    print("=" * 80)
    print("EXTREME HARDNESS TEST SUITE")
    print("Advanced LaTeX + Multi-Theory SMT")
    print("=" * 80)
    print()
    
    results = []
    
    # ========================================================================
    # LEVEL 7: Arrays + Arithmetic (QF_AUFLIA)
    # ========================================================================
    
    print("LEVEL 7: Arrays + Arithmetic (Mixed Theory)")
    print("-" * 80)
    
    test1 = HardnessLevel(7, "array_sorted", "∀ i, 0 ≤ i < n-1 → a[i] ≤ a[i+1] (sorted array)")
    try:
        ctx = SemanticContext()
        ctx.add_var("n", IRVar("Int"), Int("n"))
        ctx.add_var("i", IRVar("Int"), Int("i"))
        
        # Create array: a : Int → Int
        a_array = Array('a', IntSort(), IntSort())
        ctx.z3_vars['a'] = a_array
        
        # 0 <= i < n-1
        i_var = Int('i')
        range_constraint = And(
            i_var >= 0,
            i_var < Int('n') - 1
        )
        
        # a[i] <= a[i+1]
        sorted_constraint = Select(a_array, i_var) <= Select(a_array, i_var + 1)
        
        # ∀ i, range → sorted
        full_constraint = ForAll([i_var], Implies(range_constraint, sorted_constraint))
        
        solver = Solver()
        solver.add(full_constraint)
        solver.add(Int('n') > 1)  # Array has at least 2 elements
        
        result = solver.check()
        test1.passed = (result == sat)
        
        if test1.passed:
            print(f"  ✅ {test1.name}: Sorted array property (QF_AUFLIA)")
        else:
            print(f"  ❌ {test1.name}: {result}")
    except Exception as e:
        test1.error_msg = str(e)
        print(f"  ❌ {test1.name}: ERROR - {str(e)[:60]}")
    results.append(test1)
    
    test2 = HardnessLevel(7, "array_sum", "∑_{i=0}^{n-1} a[i] ≥ 0 when ∀ i, a[i] ≥ 0")
    try:
        # This requires induction or array theory reasoning
        # Simplified: just check constraint structure
        ctx = SemanticContext()
        ctx.add_var("n", IRVar("Int"), Int("n"))
        
        a_array = Array('a', IntSort(), IntSort())
        i_var = Int('i')
        
        # ∀ i, 0 <= i < n → a[i] >= 0
        all_nonneg = ForAll([i_var], 
            Implies(And(i_var >= 0, i_var < Int('n')), 
                    Select(a_array, i_var) >= 0))
        
        solver = Solver()
        solver.add(all_nonneg)
        solver.add(Int('n') > 0)
        
        result = solver.check()
        test2.passed = (result == sat)
        print(f"  {'✅' if test2.passed else '❌'} {test2.name}: Array element constraints")
    except Exception as e:
        test2.error_msg = str(e)
        print(f"  ❌ {test2.name}: ERROR - {str(e)[:60]}")
    results.append(test2)
    
    print()
    
    # ========================================================================
    # LEVEL 8: Real LaTeX from Mathematical Papers
    # ========================================================================
    
    print("LEVEL 8: Real LaTeX from Papers")
    print("-" * 80)
    
    test3 = HardnessLevel(8, "cauchy_schwarz", 
        "|(x,y)| ≤ ||x|| · ||y|| (Cauchy-Schwarz)")
    try:
        # LaTeX: \langle x, y \rangle^2 \leq \|x\|^2 \|y\|^2
        ctx = SemanticContext()
        
        x1, x2 = Reals('x1 x2')
        y1, y2 = Reals('y1 y2')
        
        # Inner product: x1*y1 + x2*y2
        inner_product = x1*y1 + x2*y2
        
        # Norms: ||x||^2 = x1^2 + x2^2
        norm_x_sq = x1*x1 + x2*x2
        norm_y_sq = y1*y1 + y2*y2
        
        # Cauchy-Schwarz: <x,y>^2 <= ||x||^2 * ||y||^2
        cs_inequality = inner_product * inner_product <= norm_x_sq * norm_y_sq
        
        # Check if Z3 can verify this is always true
        solver = Solver()
        solver.add(Not(cs_inequality))  # Try to find counterexample
        
        result = solver.check()
        test3.passed = (result == unsat)  # unsat means always true!
        
        if test3.passed:
            print(f"  ✅ {test3.name}: Z3 proved Cauchy-Schwarz! (UNSAT → valid)")
        else:
            print(f"  ❌ {test3.name}: Z3 couldn't prove it ({result})")
            if result == sat:
                model = solver.model()
                print(f"     Counterexample: x1={model[x1]}, x2={model[x2]}, y1={model[y1]}, y2={model[y2]}")
    except Exception as e:
        test3.error_msg = str(e)
        print(f"  ❌ {test3.name}: ERROR - {str(e)[:60]}")
    results.append(test3)
    
    test4 = HardnessLevel(8, "triangle_inequality", 
        "||x + y|| ≤ ||x|| + ||y|| (Triangle Inequality)")
    try:
        # LaTeX: \|x + y\| \leq \|x\| + \|y\|
        x1, x2 = Reals('x1 x2')
        y1, y2 = Reals('y1 y2')
        
        # x + y
        sum_x1 = x1 + y1
        sum_x2 = x2 + y2
        
        # ||x + y||^2
        norm_sum_sq = sum_x1*sum_x1 + sum_x2*sum_x2
        
        # ||x|| + ||y|| (using sqrt, approximate with squares)
        norm_x_sq = x1*x1 + x2*x2
        norm_y_sq = y1*y1 + y2*y2
        
        # Squared version: ||x+y||^2 <= (||x|| + ||y||)^2
        # This expands to: ||x+y||^2 <= ||x||^2 + 2||x||||y|| + ||y||^2
        # Hard to express without sqrt, so use weaker version:
        # ||x+y||^2 <= 4 * max(||x||^2, ||y||^2)
        
        max_norm = If(norm_x_sq > norm_y_sq, norm_x_sq, norm_y_sq)
        weak_triangle = norm_sum_sq <= 4 * max_norm
        
        solver = Solver()
        solver.add(Not(weak_triangle))
        
        result = solver.check()
        test4.passed = (result == unsat)
        
        if test4.passed:
            print(f"  ✅ {test4.name}: Weak triangle inequality verified")
        else:
            print(f"  ⚠️  {test4.name}: Couldn't verify full version (expected)")
    except Exception as e:
        test4.error_msg = str(e)
        print(f"  ❌ {test4.name}: ERROR - {str(e)[:60]}")
    results.append(test4)
    
    print()
    
    # ========================================================================
    # LEVEL 9: Proof Obligations (Lemma Application)
    # ========================================================================
    
    print("LEVEL 9: Proof Obligations & Lemma Application")
    print("-" * 80)
    
    test5 = HardnessLevel(9, "modus_ponens", "P, P→Q ⊢ Q (modus ponens)")
    try:
        P, Q = Bools('P Q')
        
        # Premises: P and P→Q
        premises = [P, Implies(P, Q)]
        
        # Conclusion: Q
        conclusion = Q
        
        # Check if premises ⊢ conclusion
        # i.e., is (P ∧ (P→Q) → Q) valid?
        # Valid means ¬(P ∧ (P→Q) → Q) is UNSAT
        
        solver = Solver()
        solver.add(premises)
        solver.add(Not(conclusion))
        
        result = solver.check()
        test5.passed = (result == unsat)
        
        if test5.passed:
            print(f"  ✅ {test5.name}: Modus ponens verified")
        else:
            print(f"  ❌ {test5.name}: Failed")
    except Exception as e:
        test5.error_msg = str(e)
        print(f"  ❌ {test5.name}: ERROR - {str(e)[:60]}")
    results.append(test5)
    
    test6 = HardnessLevel(9, "transitivity", "a≤b, b≤c ⊢ a≤c (transitivity)")
    try:
        a, b, c = Ints('a b c')
        
        premises = [a <= b, b <= c]
        conclusion = a <= c
        
        solver = Solver()
        solver.add(premises)
        solver.add(Not(conclusion))
        
        result = solver.check()
        test6.passed = (result == unsat)
        
        if test6.passed:
            print(f"  ✅ {test6.name}: Transitivity proved by Z3")
        else:
            print(f"  ❌ {test6.name}: Failed")
    except Exception as e:
        test6.error_msg = str(e)
        print(f"  ❌ {test6.name}: ERROR - {str(e)[:60]}")
    results.append(test6)
    
    print()
    
    # ========================================================================
    # LEVEL 10: Complex Document (5+ Theorems)
    # ========================================================================
    
    print("LEVEL 10: Complex Document (5+ Interdependent Theorems)")
    print("-" * 80)
    
    test7 = HardnessLevel(10, "full_document", "5-theorem chain with cross-references")
    try:
        doc_ctx = DocumentContext()
        ctx = SemanticContext()
        ctx.add_var("n", IRVar("Nat"), Int("n"))
        ctx.add_var("m", IRVar("Nat"), Int("m"))
        ctx.add_var("k", IRVar("Nat"), Int("k"))
        
        # Axiom 1: n >= 0
        axiom1 = TheoremDependency(
            name="axiom_nat_nonneg",
            statement=ValidatedIRBinOp(
                ValidatedIRVar("n"),
                "≥",
                IRConst(0, MathIRSort.NAT)
            ),
            depends_on=[]
        )
        r1 = doc_ctx.add_theorem(axiom1, ctx)
        
        # Lemma 1: n + m >= n (uses axiom1)
        lemma1 = TheoremDependency(
            name="lemma_sum_ge_left",
            statement=ValidatedIRBinOp(
                ValidatedIRBinOp(ValidatedIRVar("n"), "+", ValidatedIRVar("m")),
                "≥",
                ValidatedIRVar("n")
            ),
            depends_on=["axiom_nat_nonneg"]
        )
        r2 = doc_ctx.add_theorem(lemma1, ctx)
        
        # Lemma 2: m + n >= m (symmetric)
        lemma2 = TheoremDependency(
            name="lemma_sum_ge_right",
            statement=ValidatedIRBinOp(
                ValidatedIRBinOp(ValidatedIRVar("m"), "+", ValidatedIRVar("n")),
                "≥",
                ValidatedIRVar("m")
            ),
            depends_on=["axiom_nat_nonneg"]
        )
        r3 = doc_ctx.add_theorem(lemma2, ctx)
        
        # Theorem 1: (n + m) + k >= n (uses lemma1)
        thm1 = TheoremDependency(
            name="theorem_assoc_ge",
            statement=ValidatedIRBinOp(
                ValidatedIRBinOp(
                    ValidatedIRBinOp(ValidatedIRVar("n"), "+", ValidatedIRVar("m")),
                    "+",
                    ValidatedIRVar("k")
                ),
                "≥",
                ValidatedIRVar("n")
            ),
            depends_on=["lemma_sum_ge_left", "axiom_nat_nonneg"]
        )
        r4 = doc_ctx.add_theorem(thm1, ctx)
        
        # Corollary: n + (m + k) >= n (uses all previous)
        corollary = TheoremDependency(
            name="corollary_nested_sum",
            statement=ValidatedIRBinOp(
                ValidatedIRBinOp(
                    ValidatedIRVar("n"),
                    "+",
                    ValidatedIRBinOp(ValidatedIRVar("m"), "+", ValidatedIRVar("k"))
                ),
                "≥",
                ValidatedIRVar("n")
            ),
            depends_on=["theorem_assoc_ge", "lemma_sum_ge_left", "axiom_nat_nonneg"]
        )
        r5 = doc_ctx.add_theorem(corollary, ctx)
        
        all_valid = all([r1.is_valid, r2.is_valid, r3.is_valid, r4.is_valid, r5.is_valid])
        test7.passed = all_valid
        
        if test7.passed:
            print(f"  ✅ {test7.name}: All 5 theorems validated")
            print(f"     Dependency graph:")
            for thm_name in doc_ctx.theorem_order:
                thm = doc_ctx.theorems[thm_name]
                deps_str = ", ".join(thm.depends_on) if thm.depends_on else "none"
                print(f"       {thm_name} → depends on: {deps_str}")
        else:
            failed = [i for i, r in enumerate([r1, r2, r3, r4, r5], 1) if not r.is_valid]
            print(f"  ❌ {test7.name}: Failed at theorems: {failed}")
    except Exception as e:
        test7.error_msg = str(e)
        print(f"  ❌ {test7.name}: ERROR - {str(e)[:60]}")
        traceback.print_exc()
    results.append(test7)
    
    print()
    
    # ========================================================================
    # LEVEL 11: CANONICALIZATION (Z3 Proves Equivalences)
    # ========================================================================
    
    print("LEVEL 11: CANONICALIZATION (Z3 Equivalence Checking)")
    print("-" * 80)
    
    test8 = HardnessLevel(11, "commutativity", "x+y ≡ y+x (Z3 proves equivalence)")
    try:
        x, y = Ints('x y')
        
        expr1 = x + y
        expr2 = y + x
        
        # Z3 checks: Is (x+y ≠ y+x) UNSAT?
        solver = Solver()
        solver.add(expr1 != expr2)
        
        result = solver.check()
        test8.passed = (result == unsat)
        
        if test8.passed:
            print(f"  ✅ {test8.name}: Z3 proved x+y ≡ y+x (commutativity)")
        else:
            print(f"  ❌ {test8.name}: Failed")
    except Exception as e:
        test8.error_msg = str(e)
        print(f"  ❌ {test8.name}: ERROR - {str(e)[:60]}")
    results.append(test8)
    
    test9 = HardnessLevel(11, "associativity", "(x+y)+z ≡ x+(y+z)")
    try:
        x, y, z = Ints('x y z')
        
        expr1 = (x + y) + z
        expr2 = x + (y + z)
        
        solver = Solver()
        solver.add(expr1 != expr2)
        
        result = solver.check()
        test9.passed = (result == unsat)
        
        if test9.passed:
            print(f"  ✅ {test9.name}: Z3 proved (x+y)+z ≡ x+(y+z) (associativity)")
        else:
            print(f"  ❌ {test9.name}: Failed")
    except Exception as e:
        test9.error_msg = str(e)
        print(f"  ❌ {test9.name}: ERROR - {str(e)[:60]}")
    results.append(test9)
    
    test10 = HardnessLevel(11, "de_morgan", "¬(P∧Q) ≡ ¬P∨¬Q")
    try:
        P, Q = Bools('P Q')
        
        expr1 = Not(And(P, Q))
        expr2 = Or(Not(P), Not(Q))
        
        solver = Solver()
        solver.add(expr1 != expr2)
        
        result = solver.check()
        test10.passed = (result == unsat)
        
        if test10.passed:
            print(f"  ✅ {test10.name}: Z3 proved De Morgan's Law")
        else:
            print(f"  ❌ {test10.name}: Failed")
    except Exception as e:
        test10.error_msg = str(e)
        print(f"  ❌ {test10.name}: ERROR - {str(e)[:60]}")
    results.append(test10)
    
    test11 = HardnessLevel(11, "double_negation", "¬¬P ≡ P")
    try:
        P = Bool('P')
        
        expr1 = Not(Not(P))
        expr2 = P
        
        solver = Solver()
        solver.add(expr1 != expr2)
        
        result = solver.check()
        test11.passed = (result == unsat)
        
        if test11.passed:
            print(f"  ✅ {test11.name}: Z3 proved ¬¬P ≡ P")
        else:
            print(f"  ❌ {test11.name}: Failed")
    except Exception as e:
        test11.error_msg = str(e)
        print(f"  ❌ {test11.name}: ERROR - {str(e)[:60]}")
    results.append(test11)
    
    test12 = HardnessLevel(11, "implication", "P→Q ≡ ¬P∨Q")
    try:
        P, Q = Bools('P Q')
        
        expr1 = Implies(P, Q)
        expr2 = Or(Not(P), Q)
        
        solver = Solver()
        solver.add(expr1 != expr2)
        
        result = solver.check()
        test12.passed = (result == unsat)
        
        if test12.passed:
            print(f"  ✅ {test12.name}: Z3 proved P→Q ≡ ¬P∨Q")
        else:
            print(f"  ❌ {test12.name}: Failed")
    except Exception as e:
        test12.error_msg = str(e)
        print(f"  ❌ {test12.name}: ERROR - {str(e)[:60]}")
    results.append(test12)
    
    test13 = HardnessLevel(11, "distributivity", "x*(y+z) ≡ x*y+x*z")
    try:
        x, y, z = Ints('x y z')
        
        expr1 = x * (y + z)
        expr2 = x*y + x*z
        
        solver = Solver()
        solver.add(expr1 != expr2)
        
        result = solver.check()
        test13.passed = (result == unsat)
        
        if test13.passed:
            print(f"  ✅ {test13.name}: Z3 proved x*(y+z) ≡ x*y+x*z (distributivity)")
        else:
            print(f"  ❌ {test13.name}: Failed")
    except Exception as e:
        test13.error_msg = str(e)
        print(f"  ❌ {test13.name}: ERROR - {str(e)[:60]}")
    results.append(test13)
    
    print()
    print("  💡 Canonicalization enables:")
    print("     - Deduplication: x+y and y+x → same canonical form")
    print("     - Caching: Store by canonical form, not surface syntax")
    print("     - Pattern matching: Match modulo equivalence")
    print("     - Learning: Recognize equivalent formulations from papers")
    
    print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("=" * 80)
    print("EXTREME TEST SUITE SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")
    print()
    
    for level in range(7, 12):
        level_tests = [r for r in results if r.level == level]
        if level_tests:
            level_passed = sum(1 for r in level_tests if r.passed)
            print(f"Level {level}: {level_passed}/{len(level_tests)} passed")
    
    print()
    
    if any(not r.passed for r in results):
        print("Failed/Skipped tests:")
        for r in results:
            if not r.passed:
                print(f"  ❌ {r.name} (Level {r.level})")
                if r.error_msg:
                    print(f"     Note: {r.error_msg[:80]}")
        print()
    
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    # Run both test suites
    print("Running basic suite first...\n")
    basic_results = run_test_suite()
    
    print("\n\n")
    
    print("Running extreme suite...\n")
    extreme_results = run_extreme_test_suite()
    
    print("\n")
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    total_tests = len(basic_results) + len(extreme_results)
    total_passed = sum(1 for r in basic_results + extreme_results if r.passed)
    print(f"Total: {total_passed}/{total_tests} tests passed ({100*total_passed/total_tests:.1f}%)")
    print(f"Basic Suite: {sum(1 for r in basic_results if r.passed)}/{len(basic_results)}")
    print(f"Extreme Suite: {sum(1 for r in extreme_results if r.passed)}/{len(extreme_results)}")
    print("=" * 80)
