# Running CEGIS Pipeline - Debug Guide

## Step 1: Test imports
```bash
cd /Users/halleyyoung/Documents/cav-nlp
python test_import.py
```

If this fails, check which specific import is failing.

## Step 2: Run with small dataset (fast test)
```bash
python run_cegis_on_papers.py --cache-only --max-papers 10 --max-iterations 5 --min-confidence 0.5
```

This should complete in 1-2 minutes and help identify any immediate issues.

## Step 3: Run with moderate dataset
```bash
python run_cegis_on_papers.py --cache-only --max-papers 50 --max-iterations 10
```

## Step 4: Full run on large corpus
```bash
python run_cegis_on_papers.py --cache-only --max-papers 200 --max-iterations 30
```

## Expected Output Locations

All results will be saved to `cegis_results/`:
- `training_examples.json` - Extracted (English, Lean) pairs
- `learned_rules.json` - Learned compositional rules (full data)
- `training_history.json` - CEGIS iteration logs
- `learned_rules.lean` - **NEW**: Human-readable Lean summary file

## Common Issues

### Import Error: z3
```bash
pip install z3-solver
```

### Import Error: arxiv
```bash
pip install arxiv
```

### Import Error: numpy
```bash
pip install numpy
```

### No cached papers
The script will look for `*.txt` files in `arxiv_corpus/` directory.
Check if files exist:
```bash
ls arxiv_corpus/*.txt | head -5
```

### Z3 timeout
If Z3 synthesis takes too long, the script has fallbacks but you may see:
"→ Fallback to heuristic patterns..."
This is normal and expected for complex patterns.

## Debugging Tips

1. Check Python version (needs 3.7+):
   ```bash
   python --version
   ```

2. Check if z3 works:
   ```bash
   python -c "from z3 import *; print('Z3 OK')"
   ```

3. Monitor memory usage (CEGIS can be memory-intensive):
   ```bash
   # In another terminal
   top -pid $(pgrep -f run_cegis_on_papers)
   ```

4. See detailed errors:
   The script has comprehensive error handling. Look for:
   - "✗" symbols for failures
   - Stack traces at the end
   - Counter-example analysis in CEGIS iterations
