
#!/usr/bin/env python3
"""
Generate comprehensive "Foundations of X" textbooks (>1500 pages) with Lean-verified proofs.

This script orchestrates:
1. Generating novel theorems in LaTeX chapters
2. Attempting to prove each theorem in Lean 4
3. Checking if structures/definitions need augmentation to support proofs
4. Reproving affected axioms after structure changes
5. Checking novelty via literature search
6. Rewriting unprovable theorems to be provable
7. Creating human-readable interpretations
8. Iterating until chapters are complete and verified

The goal is to create truly rigorous foundations where every claim is verified
in Lean, while maintaining readability and insight.

STRUCTURE AUGMENTATION POLICY:
- If a theorem fails after multiple attempts, the system checks if existing structures
  are too minimal (missing fields, instances, or properties)
- Structures can be augmented with additional fields, instances, or helper lemmas
- When structures change, existing axioms/theorems that reference them are tracked
  in axioms_to_reprove.json and immediately reproofed
- This allows the foundation to grow organically as needed by the theorems

AXIOM POLICY:
- By default, all axioms are treated as "unprovable theorems" and attempts are made to prove them
- You can mark an axiom as a "fundamental_axiom" to accept it WITHOUT proof attempts
- BUT: Only use this if you are HIGHLY CONFIDENT and willing to ARGUE that it should be 
  a fundamental axiom due to its INHERENT NATURE (like Choice, ExcludedMiddle, Univalence)
- DO NOT use "fundamental_axiom" for theorems you're failing to prove because they might be false
- Every fundamental axiom MUST include a detailed philosophical_justification explaining why
  it deserves to be an axiom rather than a theorem
- Examples of acceptable fundamental axioms: Choice, ExcludedMiddle, Univalence, Propext
- Examples of unacceptable: specific convergence claims, domain-specific "obvious" facts,
  computational shortcuts, or anything that sounds like it could be proven with enough work
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Any

def run_copilot(prompt, explanation="Executing copilot"):
    """Safely run copilot with a prompt, avoiding shell interpretation issues."""
    print(f"Executing: {explanation}")
    result = subprocess.run(
        ['copilot', '-p', prompt, '--allow-all-tools'],
        capture_output=False,
        text=True
    )
    return result.returncode

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_lean_verified_foundations.py <topic> [proposal_file]")
        print("Example: python run_lean_verified_foundations.py probability")
        print("Example: python run_lean_verified_foundations.py probability ./proposal.md")
        print("Example: python run_lean_verified_foundations.py protocol ./foundations_protocol.tex")
        sys.exit(1)
    
    topic = sys.argv[1]
    proposal_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Read proposal if provided
    proposal_content = ""
    if proposal_file:
        # Handle both absolute and relative paths
        if not os.path.isabs(proposal_file):
            proposal_file = os.path.abspath(proposal_file)
        
        if os.path.exists(proposal_file):
            with open(proposal_file, 'r') as f:
                proposal_content = f.read()
            print(f"✓ Loaded proposal from: {proposal_file}")
        else:
            print(f"ERROR: Proposal file not found: {proposal_file}")
            print("Please check the path and try again.")
            sys.exit(1)
    
    base_dir = f"foundations-{topic}-lean"
    tex_dir = os.path.join(base_dir, "chapters")
    lean_dir = base_dir  # Lean files go in root for proper module structure
    metadata_dir = os.path.join(base_dir, "metadata")
    axioms_queue_file = os.path.join(base_dir, "axioms_to_prove.json")
    axioms_to_reprove_file = os.path.join(base_dir, "axioms_to_reprove.json")
    
    # Create directory structure
    os.makedirs(tex_dir, exist_ok=True)
    # Don't create lean_dir separately since it's the base_dir
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Initialize axioms queue if it doesn't exist
    if not os.path.exists(axioms_queue_file):
        with open(axioms_queue_file, 'w') as f:
            json.dump({"axioms": []}, f, indent=2)
    
    # Initialize axioms to reprove file if it doesn't exist
    if not os.path.exists(axioms_to_reprove_file):
        with open(axioms_to_reprove_file, 'w') as f:
            json.dump({"axioms": [], "structure_changes": []}, f, indent=2)
    
    # Setup Lean project with Mathlib
    print(f"\n{'='*80}")
    print(f"Setting up Lean 4 project with Mathlib")
    print(f"{'='*80}")
    
    lean_project_dir = base_dir
    lakefile_path = os.path.join(lean_project_dir, "lakefile.lean")
    toolchain_path = os.path.join(lean_project_dir, "lean-toolchain")
    
    if not os.path.exists(lakefile_path):
        print("Creating Lean project configuration...")
        
        # Use a stable Lean version compatible with Mathlib
        # Instead of parsing lean --version (which can be unreliable), use leanprover/lean4 format
        result = subprocess.run(['lean', '--version'], capture_output=True, text=True)
        if result.returncode == 0 and 'version' in result.stdout:
            # Parse: "Lean (version 4.25.2, ..."
            version_match = result.stdout.split('version')[1].split(',')[0].strip()
            lean_version = f"leanprover/lean4:v{version_match}"
            print(f"Detected Lean version: {lean_version}")
        else:
            # Use a stable, known-working version compatible with Mathlib
            lean_version = "leanprover/lean4:v4.17.0"
            print(f"Could not detect Lean version, using stable default: {lean_version}")
        
        # Create lean-toolchain
        with open(toolchain_path, 'w') as f:
            f.write(lean_version + '\n')
        print(f"✓ Created lean-toolchain")
        
        # Create lakefile.lean with Mathlib dependency
        lakefile_content = f"""import Lake
open Lake DSL

package «foundations_{topic}» where
  -- Settings for the package
  precompileModules := true

-- Define the main library that includes chapter modules
lean_lib «Chapters» where
  roots := #[`Chapter01, `Chapter02, `Chapter03, `Chapter04, `Chapter05,
             `Chapter06, `Chapter07, `Chapter08, `Chapter09, `Chapter10,
             `Chapter11, `Chapter12, `Chapter13, `Chapter14, `Chapter15,
             `Chapter16, `Chapter17, `Chapter18, `Chapter19, `Chapter20,
             `Chapter21, `Chapter22, `Chapter23, `Chapter24, `Chapter25,
             `Chapter26, `Chapter27, `Chapter28, `Chapter29, `Chapter30]
  globs := #[.submodules `Chapters]

@[default_target]
lean_exe «foundations_{topic}» where
  root := `Main
  supportInterpreter := true

-- Require Mathlib from git
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
"""
        with open(lakefile_path, 'w') as f:
            f.write(lakefile_content)
        print(f"✓ Created lakefile.lean with Mathlib dependency")
        
        # Create a minimal Main.lean file
        main_file = os.path.join(lean_project_dir, "Main.lean")
        with open(main_file, 'w') as f:
            f.write(f"""-- Main entry point for foundations-{topic}
-- Individual chapter files are in the lean/ directory

def main : IO Unit :=
  IO.println "Foundations of {topic} - Lean verified proofs"
""")
        print(f"✓ Created Main.lean")
        
        # Try to get pre-built Mathlib cache FIRST before anything else
        print("Attempting to download pre-built Mathlib cache (this may take several minutes)...")
        print("This is essential for using Mathlib in proofs.")
        cache_result = subprocess.run(
            ['lake', 'exe', 'cache', 'get'],
            cwd=lean_project_dir,
            capture_output=False,  # Show output to user
            text=True,
            timeout=900  # 15 minute timeout for slow connections
        )
        if cache_result.returncode == 0:
            print("\n✓ Pre-built Mathlib cache downloaded successfully")
            print("Mathlib is ready to use without building from source")
        else:
            print("\n✗ Could not download pre-built cache")
            print("\nWARNING: Building Mathlib from source with version mismatch will likely fail")
            print(f"Your Lean version: v4.25.2 (detected)")
            print("Consider using a stable Mathlib-compatible Lean version")
            print("\nProceeding with limited library access - proofs may need to be more basic")
    else:
        print("✓ Lean project already configured")
        # Check if we need to get cache
        mathlib_build_dir = os.path.join(lean_project_dir, ".lake", "packages", "mathlib", ".lake", "build")
        if not os.path.exists(mathlib_build_dir) or not any(os.scandir(mathlib_build_dir)):
            print("Mathlib not built, attempting to get cache (this may take several minutes)...")
            cache_result = subprocess.run(
                ['lake', 'exe', 'cache', 'get'],
                cwd=lean_project_dir,
                capture_output=False,  # Show output
                text=True,
                timeout=900
            )
            if cache_result.returncode == 0:
                print("\n✓ Pre-built Mathlib cache downloaded")
            else:
                print("\n✗ Could not get Mathlib cache. Proofs may have limited library access.")
    
    print(f"{'='*80}\n")
    
    print(f"="*80)
    print(f"LEAN-VERIFIED FOUNDATIONS OF {topic.upper()}")
    print(f"Goal: >1500 pages of rigorously verified mathematical content")
    if proposal_file:
        print(f"Guided by proposal: {proposal_file}")
    print(f"="*80 + "\n")
    
    # Save proposal to base directory for reference
    if proposal_content:
        os.makedirs(base_dir, exist_ok=True)
        proposal_copy = os.path.join(base_dir, "PROPOSAL.md")
        with open(proposal_copy, 'w') as f:
            f.write(proposal_content)
        print(f"Proposal saved to: {proposal_copy}\n")
    
    # Create/load progress tracking file
    progress_file = os.path.join(base_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        print(f"✓ Loaded progress from {progress_file}")
        print(f"  Last chapter worked on: {progress.get('last_chapter', 0)}")
    else:
        progress = {
            "last_chapter": 0,
            "chapters_complete": [],
            "chapters_in_progress": [],
            "start_time": datetime.now().isoformat()
        }
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    # Target ~30 chapters × ~50 pages = 1500 pages
    for chapter_num in range(1, 31):
        chapter_name = f"chapter-{chapter_num:02d}"
        # Lean module name follows Lean conventions (Chapter01, Chapter02, etc.)
        lean_module_name = f"Chapter{chapter_num:02d}"
        tex_file = os.path.join(tex_dir, f"{chapter_name}.tex")
        lean_file = os.path.join(lean_dir, f"{lean_module_name}.lean")
        meta_file = os.path.join(metadata_dir, f"{chapter_name}.json")
        
        # Update progress
        progress['last_chapter'] = chapter_num
        if chapter_num not in progress['chapters_in_progress']:
            progress['chapters_in_progress'].append(chapter_num)
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Skip if chapter is already complete and verified
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if meta.get('status') == 'complete' and meta.get('lean_verified', False):
                    # Check if all theorems are truly proved
                    all_proved = meta.get('theorems_proved', 0) == meta.get('theorems_total', 0)
                    if all_proved:
                        print(f"Chapter {chapter_num} already fully verified. Skipping.")
                        continue
                    else:
                        print(f"Chapter {chapter_num} partially complete: {meta.get('theorems_proved', 0)}/{meta.get('theorems_total', 0)} theorems proved")
                        print(f"Resuming to complete remaining theorems...")
        
        # Check if chapter tex already exists
        tex_generation_attempts = 0
        max_tex_attempts = 8
        
        while not os.path.exists(tex_file) and tex_generation_attempts < max_tex_attempts:
            if tex_generation_attempts > 0:
                print(f"\n{'='*80}")
                print(f"CHAPTER {chapter_num}: Retry {tex_generation_attempts}/{max_tex_attempts}")
                print(f"{'='*80}")
            else:
                print(f"\n{'='*80}")
                print(f"CHAPTER {chapter_num}: Initial Generation")
                print(f"{'='*80}")
            
            tex_generation_attempts += 1
            
            # Step 1: Generate initial chapter with theorems
            generation_prompt = f"""
Generate (or extend) chapter {chapter_num} for "Foundations of {topic}" as a rigorous LaTeX file.

{"PROPOSAL GUIDANCE:" if proposal_content else ""}
{proposal_content if proposal_content else ""}


































CRITICAL REQUIREMENTS:
1. Save as {tex_file}
2. State 8-15 significant theorems with clear hypotheses and conclusions
3. Each theorem should be:
   - Novel and interesting (pushing boundaries of {topic})
   - Potentially provable in Lean 4 (avoid non-constructive axioms when possible)
   - Building on previous chapters' results
   {"- Aligned with the proposal's vision and direction" if proposal_content else ""}
4. Include clear definitions and lemmas
5. Use standard mathematical notation
6. Add TODO comments indicating where proofs will go: % PROOF_TODO: <theorem_label>
7. Target ~50 pages of dense mathematical content
8. Focus on foundations that unify {topic} with all of math and computation
9. Not only that, but define the rest of math and computation **in terms of** {topic} (like how category theory redefined set theory *and* topology in terms of purely categorical language)

STRUCTURE:
- Introduction explaining chapter's role in foundations
- 3-5 sections with theorems building progressively
- Clear statement of all assumptions
- Label every theorem/lemma/definition uniquely (e.g., thm:ch{chapter_num}:main1)

Previous chapters can be found in {tex_dir} (if they exist).
{"The proposal document is at " + os.path.join(base_dir, "PROPOSAL.md") + " for reference." if proposal_content else ""}
Remember: Every claim will be proven in Lean, so be precise and constructive.
"""
        
            run_copilot(generation_prompt, f"Generating chapter {chapter_num} (attempt {tex_generation_attempts})")
        
        # Check if generation succeeded
        if not os.path.exists(tex_file):
            print(f"\n{'!'*80}")
            print(f"ERROR: Chapter {chapter_num} tex file not created after {max_tex_attempts} attempts")
            print(f"Skipping to next chapter. Fix this manually or restart from this chapter.")
            print(f"{'!'*80}\n")
            continue
        else:
            print(f"✓ Chapter {chapter_num} LaTeX file exists")
        
        print(f"\n{'='*80}")
        print(f"CHAPTER {chapter_num}: Extract/Load Theorems")
        print(f"{'='*80}")
        
        theorems_file = os.path.join(metadata_dir, f"{chapter_name}_theorems.json")
        
        # Step 2a: Extract theorem list (or load if exists)
        if os.path.exists(theorems_file):
            print(f"Theorem list already exists, loading from {theorems_file}")
        else:
            extract_prompt = f"""
Extract all theorems from {tex_file} and create a theorem list.

Create {metadata_dir}/{chapter_name}_theorems.json with:
{{
  "theorems": [
    {{
      "id": "thm:ch{chapter_num}:theorem1",
      "name": "Main Theorem 1",
      "statement_latex": "<LaTeX statement>",
      "line_number": <line in tex file>,
      "dependencies": ["<other theorem ids needed for proof>"]
    }},
    ...
  ]
}}

Read {tex_file} and identify ALL theorem environments, lemmas, propositions, and corollaries.
Record their exact LaTeX statements and labels.

IMPORTANT: Save the result to {theorems_file} so progress is persistent.
"""
            
            extract_attempts = 0
            max_extract_attempts = 2
            while not os.path.exists(theorems_file) and extract_attempts < max_extract_attempts:
                extract_attempts += 1
                run_copilot(extract_prompt, f"Extracting theorems from chapter {chapter_num} (attempt {extract_attempts})")
        
        if not os.path.exists(theorems_file):
            print(f"\n{'!'*80}")
            print(f"ERROR: Could not extract theorems from chapter {chapter_num} after {max_extract_attempts} attempts")
            print(f"Skipping to next chapter.")
            print(f"{'!'*80}\n")
            continue
        
        with open(theorems_file, 'r') as f:
            theorems_data = json.load(f)

        theorems = theorems_data.get('theorems', []) if isinstance(theorems_data, dict) else theorems_data
        if not isinstance(theorems, list):
            print(f"WARNING: Unexpected theorems format in {theorems_file}; expected list")
            theorems = []

        # Filter to only theorems that need proving (not already proved)
        theorems_to_prove = [t for t in theorems if isinstance(t, dict) and t.get('proof_status') != 'proved']
        theorems_already_proved = len(theorems) - len(theorems_to_prove)
        
        print(f"\nFound {len(theorems)} total theorems")
        print(f"  - Already proved: {theorems_already_proved}")
        print(f"  - Need to prove: {len(theorems_to_prove)}")
        
        if theorems_to_prove:
            theorems = theorems_to_prove
        else:
            print(f"All theorems already proved, skipping to verification")
        
        # Step 2b: Prove each theorem individually
        print(f"\n{'='*80}")
        print(f"CHAPTER {chapter_num}: Individual Theorem Proving")
        print(f"{'='*80}")
        
        # Initialize Lean file with imports and setup - but only if it doesn't exist
        if not os.path.exists(lean_file):
            print(f"Creating initial {lean_file} with minimal structure...")
            os.makedirs(os.path.dirname(lean_file), exist_ok=True)
            with open(lean_file, 'w') as f:
                imports = '\n'.join([f'import Chapter{i:02d}' for i in range(1, chapter_num)]) if chapter_num > 1 else '-- No previous chapters'
                f.write(f"""-- Chapter {chapter_num}: {topic.title()} Foundations
-- Auto-generated Lean 4 file
-- Mathlib is available via lake

-- Import previous chapters if they exist
{imports}

-- Module namespace
namespace {lean_module_name}

-- Theorems will be added by individual proof sessions

end {lean_module_name}
""")
            print(f"✓ Created {lean_file}")
        else:
            print(f"✓ {lean_file} already exists")
        
        # Prove each theorem in its own Copilot session
        if not theorems:
            print("No theorems to prove, skipping proof generation")
        
        for i, thm in enumerate(theorems, 1):
            thm_id = thm.get('id', f'theorem_{i}')
            thm_name = thm.get('name', f'Theorem {i}')
            thm_statement = thm.get('statement_latex', '')
            thm_deps = thm.get('dependencies', [])
            
            print(f"\n{'-'*80}")
            print(f"Proving theorem {i}/{len(theorems)}: {thm_name}")
            print(f"ID: {thm_id}")
            print(f"{'-'*80}")
            
            # Try up to 3 times to prove the theorem
            max_attempts = 3
            theorem_proved = False
            
            for attempt in range(1, max_attempts + 1):
                if theorem_proved:
                    break
                    
                attempt_suffix = f" (Attempt {attempt}/{max_attempts})" if attempt > 1 else ""
                retry_guidance = ""
                if attempt > 1:
                    retry_guidance = f"""
RETRY GUIDANCE (This is attempt {attempt}):
- Previous attempt(s) failed - try a DIFFERENT approach
- If you used axioms before, try to avoid them or use SIMPLER ones
- Search deeper in Mathlib - there may be lemmas you missed
- Consider weakening the theorem statement slightly if stuck
- Try a completely different proof strategy
"""
                
                print(f"\nProving {thm_id}{attempt_suffix}...")
                
                proof_prompt = f"""
Prove theorem {thm_id} from chapter {chapter_num} in Lean 4.{attempt_suffix}
{retry_guidance}

{"PROPOSAL CONTEXT:" if proposal_content else ""}
{proposal_content[:500] + "..." if proposal_content and len(proposal_content) > 500 else proposal_content if proposal_content else ""}

THEOREM INFORMATION:
- ID: {thm_id}
- Name: {thm_name}
- LaTeX Statement: {thm_statement}
- Dependencies: {thm_deps}
- Source: {tex_file}

CRITICAL RULES:
1. NO "sorry" OR PLACEHOLDERS ALLOWED - The proof AND any definitions you produce before it must be complete by the end of this session
2. NO INCOMPLETE PROOFS - Either the theorem is fully proven, or it's marked as failed
3. MINIMIZE AXIOM USE - Rely on axioms as little as possible:
   - FIRST: Try to prove using only Mathlib and previous theorems
   - SECOND: If stuck, search harder in Mathlib - it has thousands of results
   - THIRD: Try different proof approaches or weaken the theorem slightly
   - ONLY AS LAST RESORT: Add an axiom, but ONLY if you're highly confident it can be proven as a theorem later
   - Bad axioms: "magic" results, overly specific claims, things that look unprovable
   - Good axioms: well-known mathematical facts, things with clear proof strategies, weaker versions of standard results
4. If you DO add an axiom:
   - FIRST: Ask yourself - is this TRULY a fundamental axiom, or am I just failing to prove a theorem?
   - ONLY use confidence_level="fundamental_axiom" if you can STRONGLY ARGUE it's inherently axiomatic
   - Document each axiom added to {axioms_queue_file}
   - Add to the "axioms" array with: {{
       "name": "<axiom_name>", 
       "statement": "<lean_statement>", 
       "source_theorem": "{thm_id}", 
       "source_chapter": {chapter_num}, 
       "added_timestamp": "<ISO timestamp>",
       "confidence_level": "fundamental_axiom" OR "unprovable_theorem",
       "philosophical_justification": "<detailed argument for WHY this should be an axiom - required!>",
       "status": "accepted_axiom" (if fundamental_axiom) OR "unproven" (if unprovable_theorem),
       "proof_strategy_hint": "<how you think this could be proven>" (only if unprovable_theorem)
     }}
   - Axioms marked "fundamental_axiom" are ACCEPTED without proof attempts (use sparingly!)
   - Axioms marked "unprovable_theorem" will be immediately attempted to be proven
   - If an axiom marked "unprovable_theorem" can't be proven, your theorem will be rewritten without it
   - Examples of acceptable fundamental_axiom: Choice, ExcludedMiddle, UnivalenceAxiom
   - Examples of unacceptable fundamental_axiom: specific convergence claims, domain-specific facts

TASK:
1. Read the full theorem statement from {tex_file} (around line {thm.get('line_number', '?')})
2. Translate the theorem to Lean 4 syntax
3. Add it to {lean_file} (append to the file, before the final "end" statement)
4. Write a COMPLETE proof using Lean 4 tactics - NO "sorry" statements
5. If you need an axiom to complete the proof:
   - CRITICAL: Ask yourself if this is TRULY fundamental (like Choice) or just a theorem you're failing to prove
   - Only mark confidence_level="fundamental_axiom" if you're HIGHLY CONFIDENT and can ARGUE it's inherently axiomatic
   - Declare it as: axiom my_axiom : <statement>
   - Add it to {axioms_queue_file} with full justification (see format above)
   - Use it in your proof
6. Verify the ENTIRE FILE compiles: run from {base_dir}: `lake build {lean_module_name}`
   - This must check the COMPLETE file, not just your new theorem
   - If compilation fails ANYWHERE in the file, you must fix it
   - The entire {lean_file} must compile successfully
7. Update {theorems_file} to mark this theorem's proof status:
   - Set "lean_statement": "<the Lean theorem statement>"
   - Set "proof_status": "proved" (if complete) or "failed" (if couldn't prove)
   - Set "proof_strategy": "<brief description of proof approach>"
   - Set "axioms_used": [<list of any axiom names declared>]
   - Set "contains_sorry": false (this MUST be false - check the file!)
   - Set "error_message": "<error if failed>" (if applicable)

PROOF STRATEGIES:
- Use theorems from previous chapters (they're already imported)
- Use theorems proved earlier in this chapter (reference by name)
- ACTIVELY USE MATHLIB - it IS available and contains thousands of theorems:
  * Search for relevant lemmas with: `#check`, `exact?`, `apply?`
  * Use Mathlib tactics: simp, ring, field_simp, norm_num, linarith, omega, polyrith
  * Import specific Mathlib modules as needed (e.g., Mathlib.Data.*, Mathlib.Topology.*, Mathlib.Analysis.*)
  * Mathlib is installed via lean-toolchain and ready to use
- Try tactics: intro, apply, exact, cases, induction, rw, simp, ring, linarith, omega, etc.
- Use `library_search` or `exact?` to find Mathlib lemmas automatically
- If stuck, you MAY introduce an axiom (but document it!)
- If the theorem is fundamentally unprovable without axioms you're unwilling to add, mark as "failed"

VERIFICATION BEFORE COMPLETING:
- Search {lean_file} for "sorry" - there should be ZERO occurrences
- Ensure the ENTIRE {lean_file} compiles without errors (not just this theorem)
- Run from {base_dir}: `lake build {lean_module_name}` and verify ZERO errors in the complete file
- If ANY part of the file has errors, fix them before marking complete
- Confirm you've explored Mathlib for relevant lemmas (it IS available)
- Confirm all axioms used are documented in {axioms_queue_file}

CRITICAL: The entire Lean file must compile successfully, not just your new theorem.

IMPORTANT:
- This theorem can use results from: {', '.join(thm_deps) if thm_deps else 'no dependencies listed'}
- Make the proof as rigorous and complete as possible
- NO SHORTCUTS - every proof must be complete or the theorem fails

You have this entire Copilot session dedicated to proving just this one theorem.
Take your time and be thorough. Remember: NO SORRY OR SIMPLIFICATIONS/PLACEHOLDERS ALLOWED, and DONT ACCEPT A LACK OF MATHLIB.

CRITICAL: After completing the proof, UPDATE {theorems_file} to save your progress!
This allows the script to resume if interrupted.
"""
                
                run_copilot(proof_prompt, f"Proving {thm_id} (attempt {attempt})")
                
                # Check if theorem was successfully proved
                if os.path.exists(theorems_file):
                    with open(theorems_file, 'r') as f:
                        current_theorems_data = json.load(f)

                    current_theorems = []
                    if isinstance(current_theorems_data, dict):
                        current_theorems = current_theorems_data.get('theorems', [])
                    elif isinstance(current_theorems_data, list):
                        current_theorems = current_theorems_data

                    for t in current_theorems:
                        if isinstance(t, dict) and t.get('id') == thm_id and t.get('proof_status') == 'proved':
                            theorem_proved = True
                            print(f"✓ Theorem {thm_id} successfully proved on attempt {attempt}")
                            break
                
                if not theorem_proved and attempt < max_attempts:
                    print(f"✗ Attempt {attempt} failed. Retrying with different approach...")
            
            # Step 2b.1: If theorem still not proved, check if we need to augment structures/definitions
            if not theorem_proved:
                print(f"✗ Failed to prove {thm_id} after {max_attempts} attempts")
                print(f"\n{'-'*80}")
                print(f"Checking if structures/definitions need augmentation...")
                print(f"{'-'*80}")
                
                augmentation_prompt = f"""
Analyze why theorem {thm_id} failed to be proven and determine if existing structures/definitions are too minimal.

CONTEXT:
- Theorem: {thm_id} ({thm_name})
- LaTeX: {thm_statement}
- Lean file: {lean_file}
- Failed after {max_attempts} attempts

TASK:
1. Read {lean_file} and identify all structures, classes, and definitions used by this theorem
2. Analyze whether they are too minimal to support the theorem:
   - Missing fields/properties needed for proof?
   - Missing instances or typeclasses?
   - Insufficient axioms in the structure definition?
   - Need additional constructors or derived operations?

3. If augmentation needed, create a list of changes:
   - What structure/definition needs augmentation
   - What fields/properties/instances to add
   - Justification for each addition

4. Implement the augmentation in {lean_file}:
   - Add missing fields to structures
   - Add missing instances
   - Add helper lemmas about the structure
   - Ensure everything compiles with `lake build {lean_module_name}`

5. Track which existing axioms/theorems might break due to these changes:
   - Scan {lean_file} for axioms and theorems that reference the augmented structures
   - Add them to {axioms_to_reprove_file} with structure:
     {{
       "axioms": [
         {{
           "name": "<axiom_name>",
           "original_statement": "<old statement>",
           "needs_update": true,
           "reason": "Structure X was augmented with field Y",
           "source_file": "{lean_file}",
           "chapter": {chapter_num}
         }}
       ],
       "structure_changes": [
         {{
           "structure_name": "<name>",
           "changes": "<description of what was added>",
           "timestamp": "<ISO timestamp>",
           "triggered_by_theorem": "{thm_id}"
         }}
       ]
     }}

6. After augmentation, try to prove {thm_id} ONE MORE TIME with the enhanced structures

CRITICAL:
- Only augment if truly necessary - don't add unnecessary complexity
- Ensure all changes maintain backward compatibility where possible
- Document all changes in {axioms_to_reprove_file}
- Verify the entire file still compiles after augmentation

If no augmentation needed or augmentation doesn't help, mark theorem as failed.
"""
                
                run_copilot(augmentation_prompt, f"Checking structure augmentation for {thm_id}")
                
                # Check if theorem was proved after augmentation
                if os.path.exists(theorems_file):
                    with open(theorems_file, 'r') as f:
                        current_theorems_data = json.load(f)
                    current_theorems = current_theorems_data.get('theorems', []) if isinstance(current_theorems_data, dict) else current_theorems_data
                    for t in current_theorems:
                        if isinstance(t, dict) and t.get('id') == thm_id and t.get('proof_status') == 'proved':
                            theorem_proved = True
                            print(f"✓ Theorem {thm_id} proved after structure augmentation!")
                            break
            
            # Step 2b.2: Process axioms to reprove due to structure changes
            if os.path.exists(axioms_to_reprove_file):
                with open(axioms_to_reprove_file, 'r') as f:
                    reprove_data = json.load(f)
                
                axioms_to_reprove = [ax for ax in reprove_data.get('axioms', [])
                                    if ax.get('needs_update') and ax.get('chapter') == chapter_num]
                
                if axioms_to_reprove:
                    print(f"\n{'-'*80}")
                    print(f"Structure changes affected {len(axioms_to_reprove)} axiom(s). Reproving them now...")
                    print(f"{'-'*80}")
                    
                    for ax in axioms_to_reprove:
                        ax_name = ax.get('name')
                        print(f"\nReproving axiom: {ax_name}")
                        print(f"Reason: {ax.get('reason')}")
                        
                        reprove_prompt = f"""
Reprove axiom "{ax_name}" after structure augmentation.

CONTEXT:
- Axiom name: {ax_name}
- Original statement: {ax.get('original_statement')}
- Reason for reproving: {ax.get('reason')}
- Source file: {ax.get('source_file')}
- Structure changes: {json.dumps(reprove_data.get('structure_changes', []), indent=2)}

TASK:
1. Locate the axiom declaration in {lean_file}
2. Update the axiom statement to work with the augmented structures
3. Attempt to PROVE it as a theorem (not just update the axiom statement)
4. If provable with new structure, replace "axiom" with "theorem" and add proof
5. If still requires axiom, update the axiom statement and keep as axiom
6. Verify compilation: `lake build {lean_module_name}`
7. Update {axioms_to_reprove_file}: set "needs_update": false, add "resolved_timestamp"

CRITICAL: Try to prove it as a theorem with the enhanced structure!
The augmentation may have made this provable.
"""
                        
                        run_copilot(reprove_prompt, f"Reproving axiom {ax_name}")
                    
                    # Update the reprove file to mark these as processed
                    with open(axioms_to_reprove_file, 'r') as f:
                        updated_reprove = json.load(f)
                    
                    for ax in updated_reprove.get('axioms', []):
                        if ax.get('chapter') == chapter_num and ax.get('needs_update'):
                            ax['needs_update'] = False
                            ax['processed_timestamp'] = datetime.now().isoformat()
                    
                    with open(axioms_to_reprove_file, 'w') as f:
                        json.dump(updated_reprove, f, indent=2)
            
            # Step 2b.3: Immediately try to prove any axioms added by this theorem (but skip fundamental axioms)
            with open(axioms_queue_file, 'r') as f:
                axioms_data = json.load(f)
            
            # Get newly added axioms from this theorem
            # Skip axioms marked as "fundamental_axiom" - they are accepted without proof
            # Only attempt to prove axioms marked as "unprovable_theorem" or unspecified
            all_new_axioms = [ax for ax in axioms_data.get('axioms', []) 
                             if ax.get('source_theorem') == thm_id]
            new_axioms = [ax for ax in all_new_axioms
                         if ax.get('status') in [None, 'unproven']
                         and ax.get('confidence_level') != 'fundamental_axiom']
            fundamental_axioms = [ax for ax in all_new_axioms
                                 if ax.get('confidence_level') == 'fundamental_axiom']
            
            if fundamental_axioms:
                print(f"\n{'-'*80}")
                print(f"Theorem {thm_id} relies on {len(fundamental_axioms)} FUNDAMENTAL AXIOM(S) (accepted without proof):")
                for ax in fundamental_axioms:
                    print(f"  - {ax.get('name')}")
                    justification = ax.get('philosophical_justification', 'No justification provided')
                    print(f"    Justification: {justification[:150]}...")
                print(f"{'-'*80}")
            
            if new_axioms:
                print(f"\n{'-'*80}")
                print(f"Theorem {thm_id} added {len(new_axioms)} axiom(s) marked as potentially unprovable. Attempting to prove them now...")
                print(f"{'-'*80}")
                
                axioms_that_failed = []
                
                for ax in new_axioms:
                    axiom_name = ax.get('name')
                    axiom_statement = ax.get('statement')
                    
                    print(f"\nAttempting to prove axiom: {axiom_name}")
                    
                    axiom_proof_prompt = f"""
IMMEDIATELY try to prove the axiom "{axiom_name}" that was just added to {lean_file}.

AXIOM INFORMATION:
- Name: {axiom_name}
- Statement: {axiom_statement}
- Originally added for theorem: {thm_id}
- Chapter: {chapter_num}
- Confidence this can be proven: {ax.get('confidence', 'unknown')}
- Suggested proof strategy: {ax.get('proof_strategy_hint', 'not provided')}

CRITICAL RULES:
1. Replace "axiom {axiom_name} : ..." with "theorem {axiom_name} : ... := by ..."
2. Write a COMPLETE proof - NO "sorry" allowed
3. SEARCH MATHLIB THOROUGHLY - use exact?, apply?, library_search
4. You MAY introduce NEW axioms only if absolutely necessary (add to {axioms_queue_file})
   - But these should be even simpler/more confident than the current axiom
5. Verify compilation: `lake build {lean_module_name}` from {base_dir}
6. Update {axioms_queue_file}:
   - If proved: "status": "proved", "proved_timestamp": "<ISO timestamp>"
   - If failed: "status": "failed", "failure_reason": "<detailed explanation>"

You have this session to prove this axiom. If you cannot prove it, mark it as failed.
"""
                    
                    run_copilot(axiom_proof_prompt, f"Proving axiom {axiom_name}")
                    
                    # Check if axiom was successfully proved
                    with open(axioms_queue_file, 'r') as f:
                        updated_axioms = json.load(f)
                    
                    axiom_status = None
                    for ax_updated in updated_axioms.get('axioms', []):
                        if ax_updated.get('name') == axiom_name:
                            axiom_status = ax_updated.get('status')
                            if axiom_status != 'proved':
                                axioms_that_failed.append(ax_updated)
                            break
                    
                    if axiom_status == 'proved':
                        print(f"✓ Successfully proved axiom: {axiom_name}")
                    else:
                        print(f"✗ Failed to prove axiom: {axiom_name}")
                
                # If any axioms failed, rewrite the theorem without them
                if axioms_that_failed:
                    print(f"\n{'-'*80}")
                    print(f"Axioms failed for theorem {thm_id}. Rewriting theorem without them...")
                    print(f"{'-'*80}")
                    
                    failed_axiom_names = [ax.get('name') for ax in axioms_that_failed]
                    
                    rewrite_prompt = f"""
Theorem "{thm_id}" in {lean_file} used axiom(s) that could not be proven:
{chr(10).join(['- ' + ax.get('name') + ': ' + ax.get('failure_reason', 'Too strong to prove') for ax in axioms_that_failed])}

YOUR TASK:
Rewrite the proof of theorem "{thm_id}" WITHOUT using these axioms.

OPTIONS:
1. Find a different proof approach using only Mathlib and previous theorems
2. Weaken the theorem statement to make it provable without these axioms
3. Split the theorem into smaller, more provable pieces
4. Use alternative Mathlib lemmas that avoid these strong assumptions

CRITICAL RULES:
1. REMOVE the failed axiom declarations: {', '.join(failed_axiom_names)}
2. Rewrite theorem "{thm_id}" with a COMPLETE proof (no sorry)
3. You may add NEW, WEAKER axioms if absolutely necessary (but try to avoid it)
4. Update {theorems_file} with the new proof status and strategy
5. Verify the ENTIRE file compiles: `lake build {lean_module_name}` from {base_dir}

The theorem must be provable without relying on unprovable axioms.
"""
                    
                    run_copilot(rewrite_prompt, f"Rewriting {thm_id} without unprovable axioms")
            
            # Save progress after each theorem attempt
            print(f"Saving progress for {thm_id}...")
            if os.path.exists(theorems_file):
                with open(theorems_file, 'r') as f:
                    updated_theorems = json.load(f)
                print(f"✓ Progress saved to {theorems_file}")
            
            print(f"Completed proof attempt for {thm_id}")
        
        # Step 2c: Verify all proofs compiled
        print(f"\n{'='*80}")
        print(f"CHAPTER {chapter_num}: Verification Summary")
        print(f"{'='*80}")
        
        verify_prompt = f"""
Verify the complete Lean file {lean_file} and generate summary.

CRITICAL VERIFICATION STEPS:
1. Search {lean_file} for "sorry" - if ANY found, this is a FAILURE
2. Verify Mathlib imports are present and being used (check for import Mathlib.*)
3. Run from {base_dir}: `lake build {lean_module_name}` to check if it compiles
4. Read {theorems_file} to see proof status of each theorem
5. Check {axioms_queue_file} for any axioms that were added
6. Create {meta_file} with:
   {{
     "chapter": {chapter_num},
     "theorems_total": <count>,
     "theorems_proved": <count of theorems with proof_status="proved">,
     "theorems_failed": [<list of theorem IDs that failed as strings, e.g. ["thm:ch1:main1", "thm:ch1:main2"]>],
     "lean_verified": <true if file compiles with no errors AND no sorry>,
     "contains_sorry": <true if ANY sorry found - this makes lean_verified false>,
     "axioms_added": <list of axiom names from {axioms_queue_file} added in this chapter>,
     "timestamp": "<ISO timestamp>",
     "individual_proofs": <copy from {theorems_file}>
   }}

CRITICAL: theorems_failed MUST be a list of theorem ID strings, NOT a count!
Example: "theorems_failed": ["thm:ch1:theorem1", "thm:ch1:theorem3"]
NOT: "theorems_failed": 2

Display a summary table showing which theorems succeeded/failed.
If any "sorry" statements are found, LIST THEM ALL and mark chapter as unverified.
"""
        
        run_copilot(verify_prompt, f"Verifying chapter {chapter_num} proofs")
        
        # Step 3: Check if proofs succeeded
        if not os.path.exists(meta_file):
            print(f"WARNING: Metadata file not created for chapter {chapter_num}")
            continue
        
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        # Ensure chapter metadata notes unresolved axioms (failed/unproven)
        # Requirement: If any axioms remain unproven or failed, note this in the metadata.
        try:
            with open(axioms_queue_file, 'r') as f:
                axioms_data = json.load(f)
        except Exception as e:
            axioms_data = {"axioms": []}
            meta.setdefault('axioms_status_error', str(e))

        def _chapter_num_from_any(value: Any):
            try:
                return int(value)
            except Exception:
                return None

        all_axioms = axioms_data.get('axioms', []) if isinstance(axioms_data, dict) else []
        chapter_axioms = []
        for ax in all_axioms:
            if not isinstance(ax, dict):
                continue
            src_ch = _chapter_num_from_any(ax.get('source_chapter', ax.get('chapter')))
            if src_ch == chapter_num:
                chapter_axioms.append(ax)

        chapter_failed_axioms = [ax for ax in chapter_axioms if ax.get('status') == 'failed']
        chapter_unproven_axioms = [ax for ax in chapter_axioms if ax.get('status') not in ['proved', 'failed']]

        meta['axioms_status'] = {
            'chapter_axioms_total': len(chapter_axioms),
            'chapter_axioms_proved': len([ax for ax in chapter_axioms if ax.get('status') == 'proved']),
            'chapter_axioms_failed': [
                {
                    'name': ax.get('name'),
                    'source_theorem': ax.get('source_theorem'),
                    'failure_reason': ax.get('failure_reason'),
                }
                for ax in chapter_failed_axioms
            ],
            'chapter_axioms_unproven': [
                {
                    'name': ax.get('name'),
                    'source_theorem': ax.get('source_theorem'),
                    'status': ax.get('status'),
                }
                for ax in chapter_unproven_axioms
            ],
        }
        meta['axioms_unresolved'] = bool(chapter_failed_axioms or chapter_unproven_axioms)

        try:
            with open(meta_file, 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"WARNING: Failed to update metadata with axiom status: {e}")
        
        max_iterations = 5
        iteration = 0
        
        while not meta.get('lean_verified', False) and iteration < max_iterations:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"CHAPTER {chapter_num}: Revision Iteration {iteration}")
            print(f"Failed theorems: {meta.get('theorems_failed', [])}")
            print(f"{'='*80}")
            
            # Step 4: Revise each failed theorem individually
            failed_theorems = meta.get('theorems_failed', [])
            
            # Handle case where failed_theorems might be an int (count) instead of list
            if isinstance(failed_theorems, int):
                print(f"WARNING: theorems_failed is a count ({failed_theorems}), not a list of IDs")
                print(f"Will attempt to identify failed theorems from {theorems_file}")
                # Try to load from theorems file
                if os.path.exists(theorems_file):
                    with open(theorems_file, 'r') as f:
                        theorems_data = json.load(f)
                    theorems_list = theorems_data.get('theorems', []) if isinstance(theorems_data, dict) else theorems_data
                    failed_theorems = [t.get('id') for t in theorems_list 
                                      if isinstance(t, dict) and t.get('proof_status') in ['failed', None, 'unproven']]
                    print(f"Found {len(failed_theorems)} failed theorems: {failed_theorems}")
                else:
                    print(f"ERROR: Cannot find theorems file to identify failed theorems")
                    failed_theorems = []
            
            for failed_id in failed_theorems:
                print(f"\n{'-'*80}")
                print(f"Revising failed theorem: {failed_id}")
                print(f"{'-'*80}")
                
                revision_prompt = f"""
Revise the failed theorem {failed_id} in chapter {chapter_num}.

{"PROPOSAL CONTEXT:" if proposal_content else ""}
{proposal_content[:500] + "..." if proposal_content and len(proposal_content) > 500 else proposal_content if proposal_content else ""}

CONTEXT:
- Theorem ID: {failed_id}
- Previous proof attempt failed (see {lean_file} and {theorems_file})
- This is revision attempt {iteration} of {max_iterations}

CRITICAL RULES:
1. NO "sorry" OR PLACEHOLDERS ALLOWED in the final proof
2. MATHLIB IS AVAILABLE - use it extensively! Search for relevant lemmas.
3. You MAY add axioms if necessary (document in {axioms_queue_file})
4. The proof must be COMPLETE by the end of this session

PROCESS:
1. Read the current theorem statement from {tex_file}
2. Read the failed Lean proof attempt from {lean_file}
3. Analyze why the proof failed:
   - Missing hypotheses?
   - Conclusion too strong?
   - Needs different formulation?
   - Fundamentally unprovable as stated?

4. Choose a revision strategy:
   a. Add reasonable hypotheses to make it provable
   b. Weaken the conclusion slightly
   c. Reformulate in an equivalent but more tractable way
   d. Replace with a related but provable theorem
   
5. Update the theorem in {tex_file} with the revised version
6. Attempt to prove the revised theorem in Lean 4 - NO "sorry" allowed!
7. Update the Lean code in {lean_file}
8. If you need an axiom, add it to {axioms_queue_file} 
9. Verify the ENTIRE {lean_file} compiles (not just this theorem) AND contains no "sorry"
10. Run from {base_dir}: `lake build {lean_module_name}` and ensure ZERO errors anywhere in the file
11. Update {theorems_file} with the new proof status

CONSTRAINTS:
- Keep the theorem as strong as possible while making it provable
- Maintain mathematical interest and novelty
- Ensure the revised theorem still fits the chapter's narrative
- Don't give up easily - try multiple proof strategies first
- Remember: NO SORRY - either complete proof or marked as failed

IMPORTANT: You have this entire Copilot session to revise and prove just this one theorem.
Be creative and persistent in finding a provable formulation WITHOUT using "sorry".
"""
                
                run_copilot(revision_prompt, f"Revising failed theorem {failed_id}")
            
            # Re-verify everything
            verify_prompt = f"""
Re-verify all proofs in {lean_file} after revisions.

Run from {base_dir}: `lake build {lean_module_name}` and update {meta_file} with current status.
"""
            run_copilot(verify_prompt, f"Re-verifying chapter {chapter_num} after revisions")
            
            # Re-check metadata
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
            else:
                print(f"ERROR: Metadata file disappeared!")
                break
        
        if not meta.get('lean_verified', False):
            print(f"\nWARNING: Chapter {chapter_num} could not be fully verified after {max_iterations} iterations")
            print(f"Moving on, but this needs manual attention.")
        else:
            print(f"\n✓ Chapter {chapter_num} FULLY VERIFIED in Lean!")
        
        # Step 6: Axioms are now handled immediately after each theorem
        # No separate axiom resolution phase needed
        print(f"\n{'='*80}")
        print(f"CHAPTER {chapter_num}: Axiom Status Check")
        print(f"{'='*80}")
        
        with open(axioms_queue_file, 'r') as f:
            axioms_data = json.load(f)
        
        chapter_axioms = [ax for ax in axioms_data.get('axioms', []) 
                         if ax.get('source_chapter') == chapter_num]
        proved_axioms = [ax for ax in chapter_axioms if ax.get('status') == 'proved']
        failed_axioms = [ax for ax in chapter_axioms if ax.get('status') == 'failed']
        
        print(f"Chapter {chapter_num} axiom summary:")
        print(f"  Total axioms: {len(chapter_axioms)}")
        print(f"  Proved: {len(proved_axioms)}")
        print(f"  Failed (theorems rewritten): {len(failed_axioms)}")
        
        if failed_axioms:
            print(f"\n  Failed axioms (theorems were rewritten without them):")
            for ax in failed_axioms:
                print(f"    - {ax.get('name')} from {ax.get('source_theorem')}")
        
        # Step 7: Revise LaTeX chapter to only include proven theorems
        print(f"\n{'='*80}")
        print(f"CHAPTER {chapter_num}: Revising LaTeX to Match Proven Theorems")
        print(f"{'='*80}")
        
        latex_revision_prompt = f"""
Revise {tex_file} to only include theorems that were successfully proven in {lean_file}.

TASK:
1. Read {lean_file} to identify which theorems were successfully proven (have complete proofs, no sorry)
2. Read {theorems_file} to check proof status of each theorem
3. Review {tex_file} and:
   - KEEP all theorems that were successfully proven in Lean
   - REMOVE or mark as "conjecture" any theorems that failed to be proven
   - REMOVE any theorems whose proofs depended on axioms that failed
   - Add notes explaining why certain theorems were removed (e.g., "requires axioms beyond our framework")
4. Ensure the chapter narrative still flows well after removals
5. Update theorem numbering and cross-references as needed
6. Add a note at the beginning of the chapter: "All theorems in this chapter have been verified in Lean 4 with complete proofs."

The resulting LaTeX file should be a completely verified, rigorous mathematical text.
Every theorem statement must correspond to a proven theorem in {lean_file}.
"""
        
        run_copilot(latex_revision_prompt, f"Revising chapter {chapter_num} LaTeX to match proven theorems")
        
        # Step 7b: Generate and Run Benchmarks for Proven Theorems
        print(f"\n{'='*80}")
        print(f"CHAPTER {chapter_num}: Benchmark Generation and Validation")
        print(f"{'='*80}")
        
        benchmark_dir = os.path.join(base_dir, "benchmarks")
        os.makedirs(benchmark_dir, exist_ok=True)
        benchmark_file = os.path.join(benchmark_dir, f"chapter_{chapter_num:02d}_benchmarks.py")
        benchmark_results_file = os.path.join(benchmark_dir, f"chapter_{chapter_num:02d}_results.json")
        
        benchmark_prompt = f'''
Generate executable benchmarks for the proven theorems in chapter {chapter_num}.

CONTEXT:
- Lean file: {lean_file} (contains proven theorems)
- LaTeX file: {tex_file} (contains theorem statements and descriptions)
- Theorems metadata: {theorems_file}

TASK:
For each theorem that has been proven in Lean, create a practical benchmark that validates its usefulness:

1. **Extract Algorithmic Content**:
   - Read each proven theorem from {lean_file}
   - Identify if the theorem implies an algorithm or computational method
   - Extract the algorithmic insight (e.g., "theorem proves convergence" → "test convergence rate")

2. **Generate Test Cases**:
   - Create realistic test instances for the algorithm
   - Include edge cases and typical use cases
   - Generate baseline comparison (naive method or known algorithm)

3. **Create Benchmark Script**:
   Save to {benchmark_file} with this structure:
   ```python
   import time
   import numpy as np
   from typing import Dict, List, Tuple
   
   # Implement algorithm derived from theorem
   def algorithm_from_theorem_X(input_data):
       # Implementation based on proof in Lean
       pass
   
   # Baseline for comparison
   def baseline_algorithm(input_data):
       # Standard approach or naive implementation
       pass
   
   # Test case generators
   def generate_test_cases(n_cases=10):
       # Create diverse test instances
       pass
   
   # Benchmark runner
   def run_benchmark():
       results = {{}}
       test_cases = generate_test_cases()
       
       for i, test in enumerate(test_cases):
           # Time both algorithms
           start = time.time()
           result_new = algorithm_from_theorem_X(test)
           time_new = time.time() - start
           
           start = time.time()
           result_baseline = baseline_algorithm(test)
           time_baseline = time.time() - start
           
           # Verify correctness
           # Compare performance
           results[f'test_{{i}}'] = {{
               'time_new': time_new,
               'time_baseline': time_baseline,
               'speedup': time_baseline / time_new if time_new > 0 else float('inf'),
               'correct': verify_result(result_new, result_baseline)
           }}
       
       return results
   
   if __name__ == '__main__':
       import json
       results = run_benchmark()
       print(json.dumps(results, indent=2))
   ```

4. **Benchmark Requirements**:
   - Each proven theorem should generate 1+ benchmarks
   - Benchmarks must run in < 60 seconds total
   - Include at least 3 test cases per algorithm
   - Compare against reasonable baseline (not strawman)
   - Measure: runtime, memory, solution quality, convergence rate (as applicable)

5. **Documentation**:
   At the top of {benchmark_file}, include:
   ```python
   """
   Benchmarks for Chapter {chapter_num}: {topic}
   
   Theorems tested:
   - Theorem X (Lean: line Y): Tests convergence rate of Algorithm A
   - Theorem Z (Lean: line W): Tests optimality of Method B
   
   Expected results:
   - Algorithm A should converge in O(n) iterations (proven)
   - Method B should achieve 95% optimal (proven bound)
   """
   ```

6. **Save Results**:
   After running, save summary to {benchmark_results_file}:
   {{
     "chapter": {chapter_num},
     "benchmarks_generated": <count>,
     "theorems_tested": [<list of theorem IDs>],
     "all_tests_passed": true/false,
     "performance_summary": {{
       "average_speedup": <float>,
       "tests_faster": <count>,
       "tests_slower": <count>
     }}
   }}

CRITICAL:
- Only benchmark theorems that have "proof_status": "proved" in {theorems_file}
- Skip purely abstract theorems with no algorithmic content
- Be creative in extracting testable claims from theorems
- Include comments explaining the theorem → benchmark mapping

Create {benchmark_file} with executable Python benchmarks.
'''
        
        run_copilot(benchmark_prompt, f"Generating benchmarks for chapter {chapter_num}")
        
        # Run the benchmarks if they were created
        if os.path.exists(benchmark_file):
            print(f"\nRunning generated benchmarks...")
            benchmark_result = subprocess.run(
                ['python3.11', benchmark_file],
                cwd=benchmark_dir,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if benchmark_result.returncode == 0:
                print(f"✓ Benchmarks completed successfully")
                if benchmark_result.stdout:
                    print(f"\nBenchmark output:\n{benchmark_result.stdout[:500]}")
                    
                    # Try to parse results
                    try:
                        results = json.loads(benchmark_result.stdout)
                        
                        # Calculate summary statistics
                        speedups = [r.get('speedup', 1.0) for r in results.values() if isinstance(r, dict)]
                        if speedups:
                            avg_speedup = sum(speedups) / len(speedups)
                            faster_count = sum(1 for s in speedups if s > 1.0)
                            print(f"\nPerformance summary:")
                            print(f"  Average speedup: {avg_speedup:.2f}x")
                            print(f"  Tests faster than baseline: {faster_count}/{len(speedups)}")
                    except Exception:
                        pass
            else:
                print(f"✗ Benchmark execution failed")
                if benchmark_result.stderr:
                    print(f"Error: {benchmark_result.stderr[:500]}")
        else:
            print(f"No benchmarks generated (may be purely theoretical chapter)")
        
        # Step 8: Check novelty of theorems
        print(f"\n{'='*80}")
        print(f"CHAPTER {chapter_num}: Novelty Check")
        print(f"{'='*80}")
        
        novelty_prompt = f"""
Check the novelty of main theorems in {tex_file}.

PROCESS:
1. Identify the 3-5 most significant theorems in the chapter
2. For each theorem:
   a. Formulate a search query for Google Scholar / arXiv / MathSciNet
   b. Search for similar results in the literature
   c. Document findings in {meta_file} under "novelty_check"
   
3. Update {meta_file} with:
   {{
     ...existing fields...,
     "novelty_check": {{
       "<theorem_name>": {{
         "statement": "<brief statement>",
         "search_queries": ["<query1>", "<query2>"],
         "similar_results": ["<citation1>", "<citation2>"],
         "assessment": "novel|known|unclear",
         "notes": "<explanation>"
       }},
       ...
     }}
   }}

Use web search to check literature. Be thorough but don't spend too long on each theorem.
"""
        
        run_copilot(novelty_prompt, f"Checking novelty for chapter {chapter_num}")
        
        # Step 8: Generate human-readable interpretations
        print(f"\n{'='*80}")
        print(f"CHAPTER {chapter_num}: Human-Readable Interpretations")
        print(f"{'='*80}")
        
        interpretation_prompt = f"""
Add human-readable interpretations to {tex_file}.

For each major theorem, add a subsection or remark explaining:
1. **Intuition**: What does this theorem really say, intuitively?
2. **Significance**: Why is this result important for {topic}?
3. **Context**: How does this fit into the bigger picture?
4. **Applications**: What can we do with this theorem?
5. **Proof Idea**: High-level strategy (even though Lean proof is complete)

Format as LaTeX \\begin{{remark}}...\\end{{remark}} or subsections.
Make these interpretations:
- Accessible to graduate students
- Highlighting computational implications
- Connecting to other areas of mathematics
- Motivating the formal development

Update {tex_file} with these interpretations, maintaining the formal structure.
Also ensure the chapter has a good introduction and conclusion summarizing the key contributions.
"""
        
        run_copilot(interpretation_prompt, f"Adding interpretations to chapter {chapter_num}")
        
        # Step 9: Final verification and marking complete
        print(f"\n{'='*80}")
        print(f"CHAPTER {chapter_num}: Final Verification")
        print(f"{'='*80}")
        
        final_check_prompt = f"""
Perform final verification of chapter {chapter_num}:

CRITICAL CHECKS:
1. Verify {lean_file} compiles without errors from {base_dir}: `lake build {lean_module_name}`
2. SCAN FOR "sorry" - if ANY found, this chapter is NOT complete
3. Confirm Mathlib is being used (check imports and look for Mathlib theorems in proofs)
4. Check all theorem references are correct in {tex_file}
5. Verify LaTeX compiles: `pdflatex {tex_file}` (or similar)
6. Ensure {meta_file} is complete with all fields
7. Review {axioms_queue_file} for remaining axioms
   - Count axioms with \"status\": \"proved\" vs \"status\": \"failed\" vs \"status\": \"unproven\"
   - If any axioms remain unproven or failed, note this in the metadata
8. Update {meta_file} with:
   {{
     ...existing fields...,
     "status": "complete" (only if NO sorry and compiles),
     "page_count_estimate": <number>,
     "final_verification_timestamp": "<ISO timestamp>",
     "axioms_remaining": <count of unproven axioms from this chapter>,
     "quality_checks": {{
       "lean_compiles": true/false,
       "contains_sorry": true/false (MUST be false for complete),
       "latex_compiles": true/false,
       "references_valid": true/false,
       "novelty_checked": true/false,
       "interpretations_added": true/false
     }}
   }}

Only mark status as "complete" if ALL quality checks pass AND contains_sorry is false.
If some theorems are proved but not all, mark status as "partial" with counts.
"""
        
        run_copilot(final_check_prompt, f"Final verification of chapter {chapter_num}")
        
        # Check chapter completion status
        chapter_status = "incomplete"
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                final_meta = json.load(f)
            
            theorems_total = final_meta.get('theorems_total', 0)
            theorems_proved = final_meta.get('theorems_proved', 0)
            lean_verified = final_meta.get('lean_verified', False)
            
            if theorems_proved == 0:
                chapter_status = "no progress"
            elif theorems_proved < theorems_total:
                chapter_status = f"partial ({theorems_proved}/{theorems_total} proved)"
            elif theorems_proved == theorems_total and lean_verified:
                chapter_status = "complete"
                # Mark as complete in progress file only if ALL theorems proved
                if chapter_num in progress['chapters_in_progress']:
                    progress['chapters_in_progress'].remove(chapter_num)
                if chapter_num not in progress['chapters_complete']:
                    progress['chapters_complete'].append(chapter_num)
                progress['last_completed'] = chapter_num
            
            # Update progress regardless of completion status
            progress['last_update'] = datetime.now().isoformat()
            if 'chapter_status' not in progress:
                progress['chapter_status'] = {}
            progress['chapter_status'][str(chapter_num)] = {
                'status': chapter_status,
                'theorems_proved': theorems_proved,
                'theorems_total': theorems_total,
                'lean_verified': lean_verified
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            
            if chapter_status == "complete":
                print(f"✓ Chapter {chapter_num} marked as complete in progress tracker")
            else:
                print(f"⚠ Chapter {chapter_num} status: {chapter_status}")
        
        print(f"\n{'='*80}")
        print(f"CHAPTER {chapter_num} PROCESSING COMPLETE")
        print(f"Status: {chapter_status}")
        print(f"{'='*80}\n")
        
        # Progress update with detailed status
        completed = len(progress['chapters_complete'])
        partial = len([ch for ch in progress.get('chapter_status', {}).values() 
                      if 'partial' in str(ch.get('status', ''))])
        estimated_pages = completed * 50
        
        print(f"\nPROGRESS SUMMARY:")
        print(f"  Complete chapters: {completed}/30 (~{estimated_pages}/1500 pages)")
        print(f"  Partial chapters: {partial}")
        print(f"  Fully complete: {sorted(progress['chapters_complete'])}")
        print(f"  In progress: {sorted(progress['chapters_in_progress'])}")
        
        # Show detailed status of recent chapters
        if 'chapter_status' in progress:
            print(f"\n  Detailed status (last 5 chapters):")
            recent_chapters = sorted([int(k) for k in progress['chapter_status'].keys()])[-5:]
            for ch in recent_chapters:
                status = progress['chapter_status'][str(ch)]
                print(f"    Ch {ch}: {status['status']} - {status['theorems_proved']}/{status['theorems_total']} theorems")
        
        print(f"{'='*80}\n")
    
    # Final report
    print(f"\n{'='*80}")
    print(f"FOUNDATIONS OF {topic.upper()} - GENERATION COMPLETE")
    print(f"{'='*80}")
    
    # Check for remaining axioms across all chapters
    with open(axioms_queue_file, 'r') as f:
        axioms_data = json.load(f)
    
    all_axioms = axioms_data.get('axioms', [])
    proved_axioms = [ax for ax in all_axioms if ax.get('status') == 'proved']
    failed_axioms = [ax for ax in all_axioms if ax.get('status') == 'failed']
    fundamental_axioms = [ax for ax in all_axioms if ax.get('confidence_level') == 'fundamental_axiom']
    unproven_axioms = [ax for ax in all_axioms 
                      if ax.get('status') not in ['proved', 'failed', 'accepted_axiom']
                      and ax.get('confidence_level') != 'fundamental_axiom']
    
    print(f"\nAXIOM RESOLUTION SUMMARY:")
    print(f"  Total axioms encountered: {len(all_axioms)}")
    print(f"  Successfully proved: {len(proved_axioms)}")
    print(f"  Fundamental axioms (accepted): {len(fundamental_axioms)}")
    print(f"  Failed to prove: {len(failed_axioms)}")
    print(f"  Unproven (not attempted): {len(unproven_axioms)}")
    
    if fundamental_axioms:
        print(f"\n  FUNDAMENTAL AXIOMS (accepted without proof):")
        for ax in fundamental_axioms:
            print(f"    - {ax.get('name')} (from {ax.get('source_theorem')} in Ch{ax.get('source_chapter')})")
            justification = ax.get('philosophical_justification', 'No justification provided')
            print(f"      Justification: {justification[:150]}")
    
    if failed_axioms:
        print(f"\n  Failed axioms (these required theorem rewrites):")
        for ax in failed_axioms:
            print(f"    - {ax.get('name')} (from {ax.get('source_theorem')} in Ch{ax.get('source_chapter')})")
    
    if unproven_axioms:
        print(f"\n  Unproven axioms (need manual attention):")
        for ax in unproven_axioms:
            print(f"    - {ax.get('name')} (from {ax.get('source_theorem')} in Ch{ax.get('source_chapter')})")
    
    # Check for remaining axioms (excluding fundamental axioms which are accepted)
    
    remaining_axioms = [ax for ax in axioms_data.get('axioms', []) 
                       if ax.get('status') not in ['proved', 'accepted_axiom']
                       and ax.get('confidence_level') != 'fundamental_axiom']
    
    if remaining_axioms:
        print(f"\n{'='*80}")
        print(f"WARNING: {len(remaining_axioms)} AXIOMS STILL NEED PROOF")
        print(f"{'='*80}")
        
        for ax in remaining_axioms[:10]:  # Show first 10
            confidence = ax.get('confidence_level', 'unspecified')
            print(f"  - {ax.get('name')} (from {ax.get('source_theorem')} in chapter {ax.get('source_chapter')})")
            print(f"    Confidence level: {confidence}")
        
        if len(remaining_axioms) > 10:
            print(f"  ... and {len(remaining_axioms) - 10} more")
        
        print(f"\nThese axioms should be converted to proven theorems in subsequent passes.")
        print(f"Run additional axiom-proving sessions or strengthen earlier chapters.")
    else:
        if fundamental_axioms:
            print(f"\n✓ ALL NON-FUNDAMENTAL AXIOMS CONVERTED TO PROVEN THEOREMS!")
            print(f"  ({len(fundamental_axioms)} fundamental axioms accepted as part of the foundation)")
        else:
            print(f"\n✓ ALL AXIOMS CONVERTED TO PROVEN THEOREMS!")
    
    report_prompt = f"""
Generate a final report for "Foundations of {topic}" in {base_dir}/REPORT.md.

Include:
1. **Overview**: Total chapters, pages, theorems
2. **Verification Status**: How many theorems fully proved in Lean (NO sorry allowed)
3. **Structure Evolution**: Summary of structure augmentations
   - Parse {axioms_to_reprove_file} for structure_changes
   - Report: which structures were augmented, when, and why
   - List axioms that needed reproving due to structure changes
   - Show how the foundation evolved organically to support theorems
4. **Axiom Status**: List of axioms used, which are proven, which remain
5. **Benchmark Results**: Summary of empirical validation
   - Parse all files in {base_dir}/benchmarks/*_results.json
   - Report: chapters benchmarked, average speedups, tests passed/failed
   - Highlight theorems that led to superior algorithms
6. **Dual Verification Summary**: 
   - Formal verification: X theorems proven in Lean
   - Empirical verification: Y benchmarks showing practical superiority
   - Theory-practice coherence: Do proven bounds match measured performance?
7. **Novelty Assessment**: Summary of novel contributions
8. **Quality Metrics**: Compilation success rates, "sorry" count (should be 0), etc.
9. **Table of Contents**: List all chapters with brief descriptions
10. **Next Steps**: Suggestions for extensions, applications, or improvements
11. **Axioms Needing Proof**: Detailed list from {axioms_queue_file}

Parse all metadata files in {metadata_dir} and benchmark results to generate this report.
Also parse {axioms_to_reprove_file} for structure evolution history.
Make it comprehensive and suitable for sharing with the mathematical/ML community.
Highlight that ALL proofs are complete (no "sorry" or placeholders).
Emphasize the dual verification: formal (Lean) + empirical (benchmarks).
Emphasize how structures evolved organically to support theorems (not pre-designed).
"""
    
    run_copilot(report_prompt, "Generating final report with benchmark results")
    
    print(f"\n✓ Foundations of {topic} generation complete!")
    print(f"  LaTeX chapters: {tex_dir}")
    print(f"  Lean proofs: {lean_dir}")
    print(f"  Metadata: {metadata_dir}")
    print(f"  Final report: {base_dir}/REPORT.md")


if __name__ == "__main__":
    main()
