#!/usr/bin/env python3
"""
Mini CEGIS test - run one iteration to verify Z3 synthesis messages appear.
"""

import sys
from z3_semantic_synthesis import CEGIS_SemanticLearner

def test_cegis_iteration():
    """Run one CEGIS iteration with Z3 synthesis."""
    
    print("=" * 70)
    print("MINI CEGIS TEST - ONE ITERATION WITH Z3 SYNTHESIS")
    print("=" * 70)
    
    # Create learner
    learner = CEGIS_SemanticLearner()
    
    # Simple examples for implication
    examples = [
        ("if P then Q", "P → Q"),
        ("if X then Y", "X → Y"),
        ("if A then B", "A → B"),
        ("P implies Q", "P → Q"),
        ("X implies Y", "X → Y"),
    ]
    
    print(f"\nTraining examples: {len(examples)}")
    for eng, lean in examples:
        print(f"  '{eng}' → '{lean}'")
    
    print("\n" + "=" * 70)
    print("RUNNING CEGIS (max 3 iterations)")
    print("=" * 70)
    print()
    
    try:
        rules = learner.learn_from_corpus(
            examples,
            max_iterations=3,
            min_confidence=0.7
        )
        
        print("\n" + "=" * 70)
        print(f"✓ CEGIS COMPLETED - Learned {len(rules)} rules")
        print("=" * 70)
        
        for i, rule in enumerate(rules):
            print(f"\nRule {i+1}:")
            print(f"  Pattern: {rule.syntactic_pattern[:80]}...")
            print(f"  Composition: {rule.composition_type}")
            print(f"  Quality: {learner.rule_quality_scores.get(rule.rule_id, rule.quality_score):.3f}")
        
        # Check if Z3 synthesis messages appeared
        print("\n" + "=" * 70)
        print("CHECK: Were Z3 synthesis messages displayed?")
        print("Expected to see:")
        print("  - '→ Z3 pattern synthesis...'")
        print("  - '→ Z3 semantic function synthesis...'")
        print("  - '→ Z3 constraint generation...'")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_cegis_iteration()
    sys.exit(0 if success else 1)
