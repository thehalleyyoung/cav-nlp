"""
Test the learned CEGIS rules on diverse examples.
Shows what patterns work and what patterns need more training.
"""

import json
import re
from pathlib import Path

def load_learned_rules():
    """Load rules from cegis_results/learned_rules.json"""
    rules_file = Path("cegis_results/learned_rules.json")
    if not rules_file.exists():
        print(f"✗ No rules file found at {rules_file}")
        return []
    
    with open(rules_file) as f:
        rules_data = json.load(f)
    
    return rules_data

def create_test_suite():
    """Create diverse test cases covering different patterns."""
    return [
        # IF-THEN (should work - this is what was learned)
        ("if x is positive then x squared is positive", "if_then"),
        ("if n is even then n is divisible by 2", "if_then"),
        ("if f is continuous then f is measurable", "if_then"),
        
        # FORALL (probably won't work - only 5.8% of training data)
        ("for all natural numbers n, n is even or n is odd", "forall"),
        ("for every real x, x squared is non-negative", "forall"),
        ("for all integers n, n plus zero equals n", "forall"),
        
        # EXISTS (probably won't work - only 6.8% of training data)
        ("there exists a prime number greater than 100", "exists"),
        ("there exists x in reals such that x squared equals 2", "exists"),
        
        # IMPLIES (might work - 6.8% of training data, similar to if-then)
        ("P implies Q", "implies"),
        ("x equals y implies x squared equals y squared", "implies"),
        
        # CONJUNCTION (probably won't work - only 2.9% of training data)
        ("A and B", "conjunction"),
        ("f is continuous and differentiable", "conjunction"),
        
        # DISJUNCTION (won't work - 0% of training data)
        ("P or Q", "disjunction"),
        ("x is positive or x is zero or x is negative", "disjunction"),
        
        # NEGATION (won't work - 0% of training data)
        ("not P", "negation"),
        ("it is not the case that all functions are continuous", "negation"),
        
        # LET-BE (might partially work - 11.7% of training data)
        ("let x be a real number", "let_be"),
        ("let f be a continuous function", "let_be"),
        
        # FUNCTION (probably won't work - not explicitly trained)
        ("the function mapping x to x squared", "function"),
        ("f maps naturals to naturals", "function"),
    ]

def test_rules():
    """Test learned rules on diverse examples."""
    print("=" * 80)
    print("TESTING LEARNED CEGIS RULES")
    print("=" * 80)
    
    rules = load_learned_rules()
    
    if not rules:
        print("\n✗ No rules loaded. Did you run run_cegis_on_papers.py first?")
        return
    
    print(f"\nLoaded {len(rules)} learned rules:")
    for rule in rules:
        print(f"  • {rule['rule_id']} ({rule['composition_type']})")
        print(f"    Pattern: {rule['syntactic_pattern'][:60]}...")
        print(f"    Quality: {rule['quality_score']:.3f}")
        print(f"    Examples: {len(rule['example_instances'])}")
    
    # Run test suite
    print("\n" + "=" * 80)
    print("TEST SUITE RESULTS")
    print("=" * 80)
    
    test_cases = create_test_suite()
    
    results_by_pattern = {}
    matched_count = 0
    total_count = len(test_cases)
    
    for test_input, expected_pattern in test_cases:
        print(f"\nInput: {test_input}")
        print(f"Expected pattern: {expected_pattern}")
        
        matched = False
        for rule in rules:
            pattern = rule['syntactic_pattern']
            if re.search(pattern, test_input, re.IGNORECASE):
                print(f"✓ MATCHED by {rule['rule_id']} ({rule['composition_type']})")
                print(f"  Pattern: {pattern[:60]}")
                print(f"  Quality: {rule['quality_score']:.3f}")
                # Show example output from rule's examples
                if rule['example_instances']:
                    ex_eng, ex_lean = rule['example_instances'][0]
                    print(f"  Example: {ex_eng[:50]} → {ex_lean[:50]}")
                matched = True
                matched_count += 1
                break
        
        if not matched:
            print(f"✗ NO MATCH - no rule found")
        
        # Track by pattern
        if expected_pattern not in results_by_pattern:
            results_by_pattern[expected_pattern] = {'matched': 0, 'total': 0}
        results_by_pattern[expected_pattern]['total'] += 1
        if matched:
            results_by_pattern[expected_pattern]['matched'] += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nOverall: {matched_count}/{total_count} tests matched ({matched_count/total_count*100:.1f}%)")
    
    print("\nResults by pattern type:")
    for pattern, stats in sorted(results_by_pattern.items(), 
                                  key=lambda x: -x[1]['matched']/max(1, x[1]['total'])):
        matched = stats['matched']
        total = stats['total']
        pct = matched/total*100 if total > 0 else 0
        print(f"  {pattern:15s}: {matched}/{total} matched ({pct:.1f}%)")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    print("\n🔍 Why only 1 rule was learned:")
    print("  1. Training data is 62% if-then patterns (very skewed)")
    print("  2. Other patterns have too few clean examples:")
    print("     - forall: 5.8% (only 6 examples)")
    print("     - exists: 6.8% (only 7 examples)")
    print("     - conjunction: 2.9% (only 3 examples)")
    print("     - disjunction: 0% (no examples)")
    print("     - negation: 0% (no examples)")
    print("  3. Many extracted examples are noisy/incomplete from arXiv papers")
    print("  4. Quality threshold 0.5 means rule needs >50% accuracy")
    
    print("\n💡 To learn more rules:")
    print("  1. Add more diverse synthetic training examples")
    print("  2. Improve Z3 canonicalizer to extract cleaner patterns")
    print("  3. Lower quality threshold (--min-confidence 0.4)")
    print("  4. Train on more papers (currently 47)")
    print("  5. Pre-filter papers for specific patterns (e.g., more forall statements)")

if __name__ == '__main__':
    test_rules()
