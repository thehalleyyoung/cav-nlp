#!/usr/bin/env python3
"""
Test the Z3-powered translation on the successfully extracted paper.
"""

import sys
sys.path.insert(0, '/Users/halleyyoung/Documents/cav-nlp')

from arxiv_single_paper_agent import ArxivToLeanPipeline, TheoremExtractor
from pathlib import Path

# Load the paper that was successfully processed
paper_dir = Path('/Users/halleyyoung/Documents/cav-nlp/arxiv_papers/1904.07675v1')
tex_file = paper_dir / 'SMDyck3-final.tex'

if not tex_file.exists():
    print(f"❌ Could not find {tex_file}")
    sys.exit(1)

print("Reading LaTeX file...")
with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
    latex_content = f.read()

print(f"✓ Loaded {len(latex_content)} chars of LaTeX\n")

# Extract statements
extractor = TheoremExtractor()
statements = extractor.extract_all(latex_content)

print(f"📊 Extracted {len(statements)} statements\n")

# Process first few statements with Z3 pipeline
pipeline = ArxivToLeanPipeline()

for i, stmt in enumerate(statements[:3], 1):
    print(f"{'='*60}")
    print(f"Statement {i}: {stmt['name']} ({stmt['type']})")
    print(f"{'='*60}")
    print(f"LaTeX: {stmt['statement'][:150]}...")
    print()
    
    result = pipeline._process_statement(stmt)
    
    if result['success']:
        print(f"✅ SUCCESS")
        ir_str = str(result.get('ir', 'N/A'))
        print(f"   IR: {ir_str[:100]}...")
        print(f"   Z3 Validated: {result.get('z3_validated', False)}")
        canon = result.get('canonical_form')
        print(f"   Canonical: {str(canon)[:80] if canon else 'N/A'}...")
        print(f"   Lean:")
        print(result['lean'])
    else:
        print(f"❌ FAILED: {result.get('error')}")
    
    print()
