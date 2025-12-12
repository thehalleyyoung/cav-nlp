# Axiom Policy and Structure Augmentation for Lean-Verified Foundations

## Overview

The `run_lean_verified_foundations.py` script now supports:
1. Marking statements as fundamental axioms (accepted without proof)
2. Augmenting structures/definitions when they're too minimal to support proofs
3. Tracking and reproving axioms affected by structure changes

Both features come with **strict requirements** to prevent misuse.

## Structure Augmentation System

### When It's Used
After a theorem fails multiple proof attempts, the system checks if existing structures/definitions are too minimal to support the theorem.

### What Can Be Augmented
- **Structures**: Add missing fields or properties
- **Classes**: Add missing instances or typeclass constraints
- **Definitions**: Expand with additional constructors or derived operations
- **Helper lemmas**: Add supporting lemmas about existing structures

### Tracking Changes
When structures are augmented, two things happen:
1. Changes are documented in `axioms_to_reprove.json` under `structure_changes`
2. Existing axioms/theorems that reference the augmented structures are added to `axioms_to_reprove`

### Reproving Affected Axioms
- Axioms affected by structure changes are immediately reproofed
- The enhanced structure may make previously axiomatic statements provable as theorems
- If still unprovable, the axiom statement is updated to work with the new structure

### Example Workflow
```
1. Theorem X fails to prove
2. System detects: Structure S is missing field F needed for the proof
3. Structure S is augmented with field F
4. All axioms using Structure S are found and added to reprove list
5. System immediately attempts to reprove those axioms
6. Many may now be provable as theorems with the enhanced structure
7. Theorem X is attempted again with the augmented structure
```

## Two Types of Axioms

### 1. Unprovable Theorems (Default)
- **confidence_level**: `"unprovable_theorem"`
- **Behavior**: The system will immediately attempt to prove these
- **Use when**: You suspect something might be provable but are having difficulty
- **Result**: If proof fails, the theorem will be rewritten without the axiom

### 2. Fundamental Axioms (Use Sparingly!)
- **confidence_level**: `"fundamental_axiom"`
- **Behavior**: Accepted without proof attempts
- **Use ONLY when**: You can strongly argue it's inherently axiomatic
- **Requires**: Detailed philosophical justification

## Requirements for Fundamental Axioms

To mark something as a fundamental axiom, you **MUST** provide:

1. **confidence_level**: Set to `"fundamental_axiom"`
2. **philosophical_justification**: A detailed paragraph explaining WHY this should be an axiom

## Examples

### ✅ Acceptable Fundamental Axioms
- **Axiom of Choice** - Fundamental to set theory, cannot be proven from ZF
- **Law of Excluded Middle** - Foundational to classical logic vs constructive
- **Univalence Axiom** - Core principle of homotopy type theory
- **Propositional Extensionality** - Fundamental to equality of propositions

### ❌ Unacceptable as Fundamental Axioms
- Specific convergence claims ("this sequence converges")
- Domain-specific "obvious" facts ("every continuous function is measurable")
- Computational shortcuts ("this algorithm terminates in O(n) time")
- Anything that sounds like it could be proven with more work
- Theorems you're failing to prove because they might be false

## JSON Format

When adding an axiom to `axioms_to_prove.json`:

```json
{
  "name": "my_axiom_name",
  "statement": "axiom my_axiom_name : ∀ x : ℕ, P x",
  "source_theorem": "thm:ch05:main1",
  "source_chapter": 5,
  "added_timestamp": "2025-12-12T10:30:00",
  "confidence_level": "fundamental_axiom",
  "philosophical_justification": "This axiom represents the principle of... It cannot be proven from... because... It is comparable to the Axiom of Choice in that...",
  "status": "accepted_axiom"
}
```

For unprovable theorems (default):
```json
{
  "name": "my_axiom_name",
  "statement": "axiom my_axiom_name : ∀ x : ℕ, P x",
  "source_theorem": "thm:ch05:main1",
  "source_chapter": 5,
  "added_timestamp": "2025-12-12T10:30:00",
  "confidence_level": "unprovable_theorem",
  "philosophical_justification": "This seems very difficult to prove, attempting to...",
  "status": "unproven",
  "proof_strategy_hint": "Try using lemmas from Mathlib.Analysis... or consider..."
}
```

## Workflow Changes

1. **When you add an axiom during proof**:
   - You'll be prompted to choose confidence level
   - For "fundamental_axiom", you must provide strong justification
   - For "unprovable_theorem", proof attempts will be made immediately

2. **Reporting**:
   - Fundamental axioms are listed separately in summaries
   - They don't count as "unproven" problems
   - Their justifications are displayed in reports

3. **Verification**:
   - Chapters with fundamental axioms are still considered "complete"
   - The report clearly distinguishes proved theorems from accepted axioms
   - Fundamental axioms are tracked for philosophical review

## Philosophy

The goal is to build a foundation that is:
- **Maximally rigorous**: Prove everything that can be proven
- **Philosophically honest**: Accept only truly fundamental principles as axioms
- **Transparent**: Clearly document what is proven vs. accepted
- **Self-aware**: Know the difference between "hard to prove" and "inherently axiomatic"

## Warning Signs You're Misusing This Feature

❌ "I can't figure out how to prove this in Lean" → Try harder, search Mathlib deeper
❌ "This is obviously true" → Obvious things can usually be proven
❌ "This would take too long to prove" → Not a reason to axiomatize
❌ "Mathlib doesn't have this lemma" → Then prove the lemma, don't axiomatize it
❌ "I need this for my theorem to work" → Find a different proof or weaken the theorem

✅ "This is a choice principle like AC" → Valid fundamental axiom
✅ "This is about classical vs constructive logic" → Valid fundamental axiom
✅ "This defines a foundational equality principle" → Valid fundamental axiom
✅ "This is equivalent to a well-known independent axiom" → Valid fundamental axiom

## Review Process

Before accepting a fundamental axiom, ask yourself:
1. Is this comparable to Choice, ExcludedMiddle, or Univalence in fundamentality?
2. Can I write a paragraph defending this as a foundational principle?
3. Have I exhausted all proof strategies in Mathlib?
4. Would a mathematician agree this should be an axiom, not a theorem?
5. Is this about the **nature of logic/set theory/type theory** itself?

If you answer "no" to any of these, use `"unprovable_theorem"` instead.

## JSON Format: axioms_to_reprove.json

This file tracks axioms that need to be reproofed after structure augmentation:

```json
{
  "axioms": [
    {
      "name": "my_axiom_name",
      "original_statement": "axiom my_axiom_name : ∀ x : MyStruct, P x",
      "needs_update": true,
      "reason": "MyStruct was augmented with field new_field",
      "source_file": "/path/to/Chapter01.lean",
      "chapter": 1,
      "processed_timestamp": "2025-12-12T15:30:00"
    }
  ],
  "structure_changes": [
    {
      "structure_name": "MyStruct",
      "changes": "Added field: new_field : ℕ with instance Inhabited",
      "timestamp": "2025-12-12T15:25:00",
      "triggered_by_theorem": "thm:ch1:main1",
      "justification": "Theorem required this field to establish convergence"
    }
  ]
}
```

### Fields Explained
- **needs_update**: `true` if axiom still needs reproving, `false` after processing
- **reason**: Why this axiom needs to be reproofed
- **processed_timestamp**: When the axiom was successfully reproofed/updated
- **triggered_by_theorem**: Which theorem's proof attempt caused the structure augmentation

## Philosophy of Organic Growth

The structure augmentation system embodies a key principle:

**Foundations should grow organically based on what theorems need, not be pre-designed.**

Traditional approach:
1. Design complete structures upfront
2. Hope they're sufficient for all theorems
3. Get stuck when they're not

Our approach:
1. Start with minimal structures
2. Attempt to prove theorems
3. Augment structures only when proven necessary
4. Reproof affected axioms (often upgrading them to theorems)
5. Continue proving with enhanced structures

This ensures:
- No unnecessary complexity
- Every structure element is justified by a theorem that needs it
- The foundation is exactly as rich as required
- Clear documentation of why each structure element exists
