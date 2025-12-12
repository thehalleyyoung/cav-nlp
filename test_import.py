#!/usr/bin/env python3
"""Quick test to verify imports work."""

import sys
from pathlib import Path

print("Testing imports...")

try:
    print("1. Testing rule_discovery_from_arxiv...")
    from rule_discovery_from_arxiv import ArxivCorpusBuilder, ArxivPaper
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("2. Testing z3_semantic_synthesis...")
    from z3_semantic_synthesis import CEGIS_SemanticLearner, EnhancedCompositionRule
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("3. Testing flexible_semantic_parsing...")
    from flexible_semantic_parsing import SemanticNormalizer
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n✓ All imports successful!")
print("\nNow try running:")
print("  python run_cegis_on_papers.py --cache-only --max-papers 50 --max-iterations 10")
