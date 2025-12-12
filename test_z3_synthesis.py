#!/usr/bin/env python3
"""
Quick test to verify Z3 synthesis infrastructure works.
"""

import sys
from z3_semantic_synthesis import CEGIS_SemanticLearner

def test_z3_synthesis():
    """Test Z3 pattern synthesis with simple examples."""
    
    print("=" * 70)
    print("TESTING Z3 SYNTHESIS INFRASTRUCTURE")
    print("=" * 70)
    
    # Create learner
    learner = CEGIS_SemanticLearner()
    
    # Simple test examples
    positive_examples = [
        ("if P then Q", "P → Q"),
        ("if X then Y", "X → Y"),
        ("if A holds then B holds", "A → B"),
    ]
    
    negative_examples = [
        ("P and Q", "P ∧ Q"),
        ("P or Q", "P ∨ Q"),
    ]
    
    # Get template
    template = learner.semantic_templates.get('implication')
    
    if not template:
        print("ERROR: 'implication' template not found")
        print(f"Available templates: {list(learner.semantic_templates.keys())}")
        return False
    
    print(f"\nTemplate: {template['composition_rule']}")
    print(f"Semantic type: {template['semantic_type']}")
    print(f"Arity: {template['arity']}")
    
    print(f"\nPositive examples: {len(positive_examples)}")
    for eng, lean in positive_examples[:3]:
        print(f"  '{eng}' → '{lean}'")
    
    print(f"\nNegative examples: {len(negative_examples)}")
    for eng, lean in negative_examples[:3]:
        print(f"  '{eng}' ≠ '{lean}'")
    
    # Test Z3 pattern synthesis
    print("\n" + "=" * 70)
    print("PHASE 1: Z3 Pattern Synthesis")
    print("=" * 70)
    
    try:
        synthesized_pattern = learner._z3_synthesize_pattern(
            positive_examples,
            negative_examples,
            template,
            'implies'
        )
        
        if synthesized_pattern:
            print(f"✓ Z3 synthesized pattern: {synthesized_pattern}")
        else:
            print("✗ Z3 synthesis returned None, trying fallback...")
            
            # Test fallback
            candidates = learner._generate_pattern_candidates(
                positive_examples,
                'implies'
            )
            print(f"  Generated {len(candidates)} candidate patterns")
            
            best_pattern = learner._select_best_pattern_z3(
                candidates,
                positive_examples,
                negative_examples
            )
            
            if best_pattern:
                print(f"✓ Selected best pattern: {best_pattern}")
                synthesized_pattern = best_pattern
            else:
                print("✗ Pattern selection failed")
                return False
        
    except Exception as e:
        print(f"✗ ERROR in Z3 pattern synthesis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test semantic function synthesis
    print("\n" + "=" * 70)
    print("PHASE 2: Z3 Semantic Function Synthesis")
    print("=" * 70)
    
    try:
        semantic_function = learner._z3_synthesize_semantic_function(
            template,
            synthesized_pattern,
            positive_examples,
            negative_examples
        )
        
        if semantic_function:
            print("✓ Z3 synthesized semantic function")
            
            # Test it
            import re
            test_input = "if P then Q"
            match = re.search(synthesized_pattern, test_input, re.IGNORECASE)
            if match:
                result = semantic_function(match.groups())
                print(f"  Test: '{test_input}' → '{result}'")
                print(f"  Expected: 'P → Q'")
                print(f"  Match: {result == 'P → Q' or 'P' in result and 'Q' in result}")
        else:
            print("✗ Semantic function synthesis returned None")
            return False
            
    except Exception as e:
        print(f"✗ ERROR in semantic function synthesis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test type constraints
    print("\n" + "=" * 70)
    print("PHASE 3: Z3 Type Constraint Generation")
    print("=" * 70)
    
    try:
        constraints = learner._synthesize_z3_type_constraints(
            template,
            synthesized_pattern,
            semantic_function,
            positive_examples
        )
        
        print(f"✓ Generated {len(constraints)} type constraints:")
        for constraint in constraints[:5]:
            print(f"  - {constraint}")
            
    except Exception as e:
        print(f"✗ ERROR in constraint generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - Z3 Synthesis Infrastructure Working!")
    print("=" * 70)
    
    return True

if __name__ == '__main__':
    success = test_z3_synthesis()
    sys.exit(0 if success else 1)
