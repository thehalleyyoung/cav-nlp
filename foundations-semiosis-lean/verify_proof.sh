#!/bin/bash

echo "=== Verification Report for thm:ch1:category_embedding ==="
echo ""

echo "1. Checking for sorry statements..."
SORRY_COUNT=$(grep -c "sorry" Chapter01.lean || echo "0")
echo "   Sorry count: $SORRY_COUNT"
if [ "$SORRY_COUNT" -eq "0" ]; then
    echo "   ✓ PASSED: No sorry statements found"
else
    echo "   ✗ FAILED: Found $SORRY_COUNT sorry statements"
    exit 1
fi
echo ""

echo "2. Checking theorem exists..."
if grep -q "theorem categoryEmbedding" Chapter01.lean; then
    echo "   ✓ PASSED: Theorem categoryEmbedding found"
else
    echo "   ✗ FAILED: Theorem categoryEmbedding not found"
    exit 1
fi
echo ""

echo "3. Building project..."
if lake build 2>&1 | grep -q "Build completed successfully"; then
    echo "   ✓ PASSED: Build completed successfully"
else
    echo "   ✗ FAILED: Build failed"
    exit 1
fi
echo ""

echo "4. Checking metadata..."
python3 << 'PYEOF'
import json
import sys

with open('metadata/chapter-01_theorems.json', 'r') as f:
    metadata = json.load(f)

for theorem in metadata['theorems']:
    if theorem['id'] == 'thm:ch1:category_embedding':
        if theorem['proof_status'] == 'proved' and not theorem['contains_sorry'] and len(theorem['axioms_used']) == 0:
            print("   ✓ PASSED: Metadata correctly updated")
            print(f"     - Proof status: {theorem['proof_status']}")
            print(f"     - Contains sorry: {theorem['contains_sorry']}")
            print(f"     - Axioms used: {theorem['axioms_used']}")
            sys.exit(0)
        else:
            print("   ✗ FAILED: Metadata incorrect")
            print(f"     - Proof status: {theorem['proof_status']}")
            print(f"     - Contains sorry: {theorem['contains_sorry']}")
            print(f"     - Axioms used: {theorem['axioms_used']}")
            sys.exit(1)
            
print("   ✗ FAILED: Theorem not found in metadata")
sys.exit(1)
PYEOF

echo ""
echo "=== All checks passed! Theorem successfully proven. ==="
