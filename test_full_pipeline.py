#!/usr/bin/env python3
"""
Generate a fresh Lean file with the Z3-powered translation and verify it compiles.
"""

import sys
sys.path.insert(0, '/Users/halleyyoung/Documents/cav-nlp')

from arxiv_single_paper_agent import ArxivToLeanPipeline, TheoremExtractor
from pathlib import Path
import subprocess

# Load the paper
paper_dir = Path('/Users/halleyyoung/Documents/cav-nlp/arxiv_papers/1904.07675v1')
tex_file = paper_dir / 'SMDyck3-final.tex'

print("="*60)
print("Z3-Powered arXiv→Lean Pipeline Test")
print("="*60)

with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
    latex_content = f.read()

# Extract
extractor = TheoremExtractor()
statements = extractor.extract_all(latex_content)
print(f"\n✓ Extracted {len(statements)} statements")

# Process with Z3 pipeline
pipeline = ArxivToLeanPipeline()
results = []

for stmt in statements:
    result = pipeline._process_statement(stmt)
    results.append(result)

successful = sum(1 for r in results if r['success'])
z3_validated = sum(1 for r in results if r.get('z3_validated'))

print(f"✓ Processed: {successful}/{len(statements)} successful")
print(f"✓ Z3 validated: {z3_validated}/{len(statements)}")

# Generate Lean file
lean_code = pipeline._generate_lean_file(results, '1904_07675v1_z3')
lean_file = paper_dir / '1904.07675v1_z3_powered.lean'

with open(lean_file, 'w') as f:
    f.write(lean_code)

print(f"\n💾 Saved to {lean_file.name}")

# Try to verify with Lean (if available)
print(f"\n{'='*60}")
print("Lean Verification Test")
print("="*60)

# Create a minimal lakefile for this test
lakefile = paper_dir / 'lakefile.lean'
if not lakefile.exists():
    with open(lakefile, 'w') as f:
        f.write("""import Lake
open Lake DSL

package arxiv where
  precompileModules := true

require mathlib from git
  "https://github.com/leanprover-community/mathlib4"

@[default_target]
lean_lib ArxivPaper
""")
    print(f"✓ Created lakefile.lean")

# Try lake build
try:
    result = subprocess.run(
        ['lake', 'build'],
        cwd=paper_dir,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        print("✅ Lean verification PASSED!")
    else:
        print(f"⚠️  Lean build had warnings/errors:")
        if result.stderr:
            print(result.stderr[:500])
except FileNotFoundError:
    print("⚠️  lake command not found - install Lean 4 to verify")
except subprocess.TimeoutExpired:
    print("⚠️  lake build timed out")
except Exception as e:
    print(f"⚠️  Could not run lake: {e}")

print(f"\n{'='*60}")
print("Summary")
print("="*60)
print(f"Statements extracted: {len(statements)}")
print(f"Successfully translated: {successful}")
print(f"Z3 validation rate: {z3_validated/len(statements)*100:.1f}%")
print(f"Success rate: {successful/len(statements)*100:.1f}%")
print(f"Generated file: {lean_file.name}")
print("="*60)
