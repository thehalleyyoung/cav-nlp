# Structure Augmentation System

## Overview

The Lean-verified foundations system now includes automatic structure augmentation to handle cases where initial definitions are too minimal to support theorems.

## The Problem

When proving theorems, we often discover that our initial structure definitions lack necessary fields, properties, or instances. Previously, this would require:
1. Manual identification of what's missing
2. Manual augmentation of structures
3. Manual search for broken axioms/theorems
4. Manual reproofing

## The Solution

Automated structure augmentation with immediate axiom reproving:

### Step 1: Detection
After a theorem fails multiple proof attempts, the system analyzes:
- Which structures are used by the theorem
- What fields/properties/instances might be missing
- Whether augmentation could enable the proof

### Step 2: Augmentation
If needed, structures are enhanced with:
- Additional fields
- Missing typeclass instances
- Helper lemmas about the structure
- Derived operations or constructors

### Step 3: Tracking
All changes are documented in `axioms_to_reprove.json`:
- What was changed and why
- Which theorem triggered the change
- Which existing axioms/theorems are affected

### Step 4: Reproving
Affected axioms are immediately reproofed:
- Many may now be provable as theorems with the enhanced structure
- If still unprovable, axiom statements are updated for compatibility
- All changes are verified to compile

### Step 5: Retry
The original theorem is attempted again with the augmented structures

## File Structure

### axioms_to_prove.json
Original file tracking axioms that need proving:
```json
{
  "axioms": [
    {
      "name": "convergence_axiom",
      "statement": "axiom convergence_axiom : ...",
      "status": "unproven",
      "confidence_level": "unprovable_theorem",
      ...
    }
  ]
}
```

### axioms_to_reprove.json (NEW)
Tracks structure changes and affected axioms:
```json
{
  "axioms": [
    {
      "name": "convergence_axiom",
      "original_statement": "axiom convergence_axiom : ∀ s : Sequence, ...",
      "needs_update": true,
      "reason": "Sequence structure augmented with metric field",
      "source_file": "Chapter03.lean",
      "chapter": 3
    }
  ],
  "structure_changes": [
    {
      "structure_name": "Sequence",
      "changes": "Added: metric : ℝ → ℝ → ℝ with MetricSpace instance",
      "timestamp": "2025-12-12T15:25:00",
      "triggered_by_theorem": "thm:ch3:cauchy_convergence",
      "justification": "Theorem requires metric to define Cauchy sequences"
    }
  ]
}
```

## Example Workflow

### Initial State
```lean
structure Sequence where
  terms : ℕ → ℝ
```

### Theorem Attempt
```lean
theorem cauchy_convergence (s : Sequence) (h : IsCauchy s) : Converges s := by
  -- FAILS: IsCauchy and Converges undefined without metric
  sorry
```

### System Analysis
"Theorem fails because Sequence lacks metric structure needed for Cauchy/convergence definitions"

### Augmentation
```lean
structure Sequence where
  terms : ℕ → ℝ
  metric : ℝ → ℝ → ℝ  -- ADDED
  metric_pos : ∀ x y, x ≠ y → metric x y > 0  -- ADDED
  metric_symm : ∀ x y, metric x y = metric y x  -- ADDED
  
instance : MetricSpace Sequence := { ... }  -- ADDED
```

### Affected Axioms Found
```lean
axiom sequence_bounded : ∀ s : Sequence, ∃ M, ∀ n, s.terms n < M
-- This axiom references Sequence, needs reproving
```

### Reproving
```lean
-- Attempted to prove as theorem with enhanced structure:
theorem sequence_bounded : ∀ s : Sequence, ∃ M, ∀ n, s.terms n < M := by
  -- May now be provable using metric properties!
  ...
```

### Retry Original Theorem
```lean
theorem cauchy_convergence (s : Sequence) (h : IsCauchy s) : Converges s := by
  -- Now can use metric, MetricSpace instance, etc.
  ...
```

## Benefits

1. **Organic Growth**: Structures only gain complexity when proven necessary
2. **Justified Additions**: Every field has a documented reason (which theorem needed it)
3. **Automatic Consistency**: Affected axioms are immediately identified and reproofed
4. **Axiom Reduction**: Enhanced structures often make axioms provable as theorems
5. **Clear History**: Complete record of how and why structures evolved

## Integration with Axiom System

The structure augmentation system works together with the axiom system:

1. **First**: Try to prove with existing structures
2. **Second**: Try different proof approaches
3. **Third**: Check if structures need augmentation
4. **Fourth**: Augment structures if justified
5. **Fifth**: Reprove affected axioms (may upgrade to theorems)
6. **Sixth**: Retry original theorem
7. **Last Resort**: Add axiom (only if augmentation didn't help)

## Preventing Misuse

Structure augmentation should only happen when:
- ✅ Multiple proof attempts have failed
- ✅ Clear analysis shows what's missing
- ✅ Addition is mathematically justified
- ✅ The field/property is actually needed by the theorem

Avoid:
- ❌ Adding fields "just in case"
- ❌ Adding unnecessary complexity
- ❌ Breaking backward compatibility without reason
- ❌ Adding fields that can be derived from existing ones

## Reporting

The final report includes a "Structure Evolution" section showing:
- Which structures were augmented
- When and why each augmentation occurred
- Which theorems triggered augmentations
- How many axioms were upgraded to theorems due to augmentations
- Complete history of the foundation's organic growth

This demonstrates that the foundation wasn't pre-designed but grew naturally from the requirements of the theorems.
