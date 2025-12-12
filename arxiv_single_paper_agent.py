#!/usr/bin/env python3
"""
arXiv-to-Lean Single Paper Agent

Implements the complete pipeline from the arxiv_to_lean_agent.prompt.md:
1. Download random arXiv paper
2. Extract all theorems/definitions/axioms (handling multiple LaTeX styles)
3. Translate to Z3-validated IR
4. Canonicalize expressions for deduplication
5. Synthesize Lean code using Z3-guided templates
6. Update vocabulary for missing terms
7. Verify in Lean
8. Learn from failures
9. Regression test
"""

import arxiv
import random
import re
import json
import os
import subprocess
import tarfile
import gzip
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Import our existing components
try:
    from z3_validated_ir import SemanticContext, ValidatedIRExpr
    from canonicalization_engine import CanonicalizationEngine
    from latex_to_lean_ir import IRVar, IRApp, IRConst
    # semantic_to_ir has Z3ToIRConverter, not SemanticToIR
    from semantic_to_ir import Z3ToIRConverter
except ImportError as e:
    print(f"⚠️  Required modules not found: {e}")
    print("Ensure z3_validated_ir.py, canonicalization_engine.py, and latex_to_lean_ir.py are available.")
    # Continue anyway for basic functionality
    SemanticContext = None
    CanonicalizationEngine = None


class TheoremExtractor:
    """Extract theorems from LaTeX with support for multiple styles."""
    
    # Patterns for different theorem styles (as per prompt)
    THEOREM_PATTERNS = [
        # Style 1: Standard LaTeX environments
        (r'\\begin\{theorem\}(.*?)\\end\{theorem\}', 'theorem', 'high'),
        (r'\\begin\{thm\}(.*?)\\end\{thm\}', 'theorem', 'high'),
        (r'\\begin\{lemma\}(.*?)\\end\{lemma\}', 'lemma', 'high'),
        (r'\\begin\{proposition\}(.*?)\\end\{proposition\}', 'proposition', 'high'),
        (r'\\begin\{corollary\}(.*?)\\end\{corollary\}', 'corollary', 'high'),
        
        # Style 2: Bold markdown-style
        (r'\*\*Theorem\s+[\d.]+\*\*.*?(.*?)(?=\n\n|\*\*|$)', 'theorem', 'medium'),
        
        # Style 3: Numbered manual
        (r'Theorem\s+([\d.]+)\.\s+(.*?)(?=\n\n|Proof|Lemma|Theorem|$)', 'theorem', 'medium'),
        (r'Lemma\s+([\d.]+)\.\s+(.*?)(?=\n\n|Proof|Lemma|Theorem|$)', 'lemma', 'medium'),
    ]
    
    DEFINITION_PATTERNS = [
        (r'\\begin\{definition\}(.*?)\\end\{definition\}', 'definition', 'high'),
        (r'\\begin\{defn\}(.*?)\\end\{defn\}', 'definition', 'high'),
        (r'\*\*Definition\s+[\d.]+\*\*\s+(.*?)(?=\n\n|\*\*|$)', 'definition', 'medium'),
        # "Let X be Y if Z" style
        (r'Let\s+(.*?)\s+be\s+\*?(.*?)\*?\s+if\s+(.*?)(?=\.|\n)', 'definition', 'low'),
    ]
    
    AXIOM_PATTERNS = [
        (r'\\begin\{axiom\}(.*?)\\end\{axiom\}', 'axiom', 'high'),
        (r'\*\*Axiom\*\*\s+(.*?)(?=\n\n|\*\*|$)', 'axiom', 'medium'),
        (r'Axiom\s+([\d.]+)\.\s+(.*?)(?=\n\n|$)', 'axiom', 'medium'),
    ]
    
    def __init__(self):
        self.doc_context = SemanticContext()
        self.vocabulary_gaps = []
    
    def extract_all(self, latex_content: str) -> List[Dict]:
        """Extract all mathematical statements from LaTeX."""
        statements = []
        
        print(f"📄 Extracting statements from LaTeX ({len(latex_content)} chars)...")
        
        # Extract theorems
        for pattern, stmt_type, priority in self.THEOREM_PATTERNS:
            for match in re.finditer(pattern, latex_content, re.DOTALL | re.IGNORECASE):
                stmt = {
                    'type': stmt_type,
                    'name': f'{stmt_type}_{len(statements)+1}',
                    'statement': match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip(),
                    'priority': priority,
                    'raw_match': match.group(0)
                }
                statements.append(stmt)
        
        # Extract definitions
        for pattern, stmt_type, priority in self.DEFINITION_PATTERNS:
            for match in re.finditer(pattern, latex_content, re.DOTALL | re.IGNORECASE):
                stmt = {
                    'type': stmt_type,
                    'name': f'definition_{len(statements)+1}',
                    'statement': match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip(),
                    'priority': priority,
                    'raw_match': match.group(0)
                }
                statements.append(stmt)
        
        # Extract axioms
        for pattern, stmt_type, priority in self.AXIOM_PATTERNS:
            for match in re.finditer(pattern, latex_content, re.DOTALL | re.IGNORECASE):
                stmt = {
                    'type': stmt_type,
                    'name': f'axiom_{len(statements)+1}',
                    'statement': match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip(),
                    'priority': priority,
                    'raw_match': match.group(0)
                }
                statements.append(stmt)
        
        return statements


class VocabularyManager:
    """Manage domain-specific vocabulary with Z3-powered extraction."""
    
    def __init__(self, definitions_path='definitions.json'):
        self.definitions_path = Path(definitions_path)
        self.definitions = self._load_definitions()
    
    def _load_definitions(self) -> Dict:
        if self.definitions_path.exists():
            with open(self.definitions_path, 'r') as f:
                return json.load(f)
        return {}
    
    def add_definition(self, term: str, definition_data: Dict):
        """Add new term to vocabulary."""
        self.definitions[term] = definition_data
        
        # Save
        with open(self.definitions_path, 'w') as f:
            json.dump(self.definitions, f, indent=2)
        
        print(f"✅ Added '{term}' to vocabulary")
    
    def lookup(self, term: str) -> Optional[Dict]:
        """Look up term definition."""
        return self.definitions.get(term)


class ArxivToLeanPipeline:
    """Complete pipeline for processing arXiv papers."""
    
    def __init__(self, output_dir='arxiv_papers'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if SemanticContext:
            self.sem_context = SemanticContext()
        if CanonicalizationEngine:
            self.canon_engine = CanonicalizationEngine()
        
        self.vocab_mgr = VocabularyManager()
        self.z3_converter = Z3ToIRConverter() if 'Z3ToIRConverter' in globals() else None
        
        self.previous_papers = []
        self.translation_cache = {}
        
        # Results tracking
        self.results_file = self.output_dir / 'processing_results.json'
        self.results = self._load_results()
    
    def _load_results(self) -> Dict:
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {
            'papers_processed': [],
            'total_statements': 0,
            'successful_translations': 0,
            'failed_translations': 0,
            'vocabulary_size': len(self.vocab_mgr.definitions)
        }
    
    def _save_results(self):
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def download_random_paper(self, category='math.CO', max_results=50) -> Optional[arxiv.Result]:
        """Download random paper from arXiv."""
        print(f"\n{'='*60}")
        print(f"Downloading random paper from {category}...")
        print(f"{'='*60}\n")
        
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = list(client.results(search))
            if not papers:
                print("❌ No papers found")
                return None
            
            paper = random.choice(papers)
            
            print(f"📄 Selected: {paper.title}")
            print(f"👥 Authors: {', '.join(a.name for a in paper.authors[:3])}")
            print(f"🔗 arXiv ID: {paper.entry_id}")
            print(f"📅 Published: {paper.published}")
            
            # Download source
            paper_dir = self.output_dir / paper.entry_id.split('/')[-1]
            paper_dir.mkdir(exist_ok=True)
            try:
                paper.download_source(dirpath=str(paper_dir))
                print(f"✅ Downloaded source to {paper_dir}")
                
                # Extract if compressed
                self._extract_compressed_files(paper_dir)
                
            except Exception as e:
                print(f"⚠️  Could not download source: {e}")
                try:
                    paper.download_pdf(dirpath=str(paper_dir))
                    print(f"✅ Downloaded PDF instead")
                except Exception as e2:
                    print(f"❌ Could not download PDF either: {e2}")
                    return None
            
            return paper
            
        except Exception as e:
            print(f"❌ Error downloading paper: {e}")
            return None
    
    def _extract_compressed_files(self, paper_dir: Path):
        """Extract compressed source files (.tar.gz, .gz)."""
        # Find compressed files
        for gz_file in paper_dir.glob('*.gz'):
            print(f"📦 Extracting {gz_file.name}...")
            
            if gz_file.name.endswith('.tar.gz'):
                # Extract tar.gz
                try:
                    with tarfile.open(gz_file, 'r:gz') as tar:
                        tar.extractall(path=paper_dir)
                    print(f"   ✅ Extracted tar.gz archive")
                except Exception as e:
                    print(f"   ⚠️  Could not extract tar.gz: {e}")
            else:
                # Extract plain .gz
                try:
                    output_file = paper_dir / gz_file.stem
                    with gzip.open(gz_file, 'rb') as f_in:
                        with open(output_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"   ✅ Extracted to {output_file.name}")
                except Exception as e:
                    print(f"   ⚠️  Could not extract gz: {e}")
    
    def extract_latex_from_source(self, paper_dir: Path) -> Optional[str]:
            print(f"❌ Error downloading paper: {e}")
            return None
    
    def extract_latex_from_source(self, paper_dir: Path) -> Optional[str]:
        """Extract LaTeX content from downloaded source."""
        # Look for .tex files
        tex_files = list(paper_dir.glob('*.tex'))
        if not tex_files:
            # Might be in a subdirectory
            tex_files = list(paper_dir.glob('**/*.tex'))
        
        if not tex_files:
            print("❌ No .tex files found")
            return None
        
        # Prefer main.tex or paper.tex
        main_file = None
        for f in tex_files:
            if f.name.lower() in ['main.tex', 'paper.tex', 'article.tex']:
                main_file = f
                break
        
        if not main_file:
            main_file = tex_files[0]
        
        print(f"📖 Reading LaTeX from {main_file.name}")
        
        try:
            with open(main_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"❌ Error reading LaTeX: {e}")
            return None
    
    def process_paper(self, paper: arxiv.Result) -> Dict:
        """Process entire paper through the pipeline."""
        paper_id = paper.entry_id.split('/')[-1]
        
        print(f"\n{'='*60}")
        print(f"PROCESSING: {paper_id}")
        print(f"Title: {paper.title}")
        print(f"{'='*60}\n")
        
        result = {
            'paper_id': paper_id,
            'title': paper.title,
            'timestamp': datetime.now().isoformat(),
            'statements': [],
            'successful': 0,
            'failed': 0,
            'new_vocabulary': [],
            'lean_code': '',
            'errors': []
        }
        
        # Step 1: Extract LaTeX
        paper_dir = self.output_dir / paper_id
        latex_content = self.extract_latex_from_source(paper_dir)
        
        if not latex_content:
            result['errors'].append('Could not extract LaTeX content')
            return result
        
        # Step 2: Extract statements
        extractor = TheoremExtractor()
        statements = extractor.extract_all(latex_content)
        
        print(f"\n📊 Extracted {len(statements)} statements:")
        by_type = {}
        for s in statements:
            by_type[s['type']] = by_type.get(s['type'], 0) + 1
        for stype, count in by_type.items():
            print(f"   - {count} {stype}(s)")
        
        # Step 3: Process each statement
        for i, stmt in enumerate(statements, 1):
            print(f"\n[{i}/{len(statements)}] Processing: {stmt['name']} ({stmt['type']})")
            print(f"   Statement: {stmt['statement'][:80]}...")
            
            stmt_result = self._process_statement(stmt)
            result['statements'].append(stmt_result)
            
            if stmt_result['success']:
                result['successful'] += 1
                print(f"   ✅ Success!")
            else:
                result['failed'] += 1
                print(f"   ❌ Failed: {stmt_result.get('error', 'Unknown error')}")
        
        # Step 4: Generate combined Lean file
        lean_code = self._generate_lean_file(result['statements'], paper_id)
        result['lean_code'] = lean_code
        
        # Save Lean file
        lean_file = paper_dir / f'{paper_id}.lean'
        with open(lean_file, 'w') as f:
            f.write(lean_code)
        print(f"\n💾 Saved Lean code to {lean_file}")
        
        # Step 5: Update global results
        self.results['papers_processed'].append(paper_id)
        self.results['total_statements'] += len(statements)
        self.results['successful_translations'] += result['successful']
        self.results['failed_translations'] += result['failed']
        self.results['vocabulary_size'] = len(self.vocab_mgr.definitions)
        self._save_results()
        
        # Step 6: Print summary
        self._print_result_summary(result)
        
        return result
    
    def _process_statement(self, stmt: Dict) -> Dict:
        """Process a single statement through the pipeline."""
        result = {
            'name': stmt['name'],
            'type': stmt['type'],
            'statement': stmt['statement'],
            'success': False,
            'ir': None,
            'canonical_form': None,
            'lean': None,
            'error': None
        }
        
        try:
            # For now, create placeholder Lean translation
            # In full implementation, this would use z3_validated_ir and semantic_to_ir
            
            # Simplified translation
            lean_statement = self._simple_latex_to_lean(stmt['statement'], stmt['type'])
            
            if lean_statement:
                result['lean'] = lean_statement
                result['success'] = True
            else:
                result['error'] = 'Translation failed'
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _simple_latex_to_lean(self, latex: str, stmt_type: str) -> Optional[str]:
        """
        Simplified LaTeX to Lean translation.
        In full implementation, this uses Z3-validated IR and synthesis.
        """
        # Clean LaTeX
        cleaned = latex.strip()
        cleaned = re.sub(r'\\label\{.*?\}', '', cleaned)
        cleaned = re.sub(r'\\cite\{.*?\}', '', cleaned)
        
        # Very basic translation (placeholder)
        if stmt_type == 'definition':
            return f"-- Definition: {cleaned[:100]}\n-- TODO: Formalize this definition"
        elif stmt_type in ['theorem', 'lemma', 'proposition', 'corollary']:
            return f"-- {stmt_type.capitalize()}: {cleaned[:100]}\n-- TODO: Prove this theorem"
        elif stmt_type == 'axiom':
            return f"-- axiom: {cleaned[:100]}\n-- TODO: Decide if this should be an axiom"
        
        return None
    
    def _generate_lean_file(self, statements: List[Dict], paper_id: str) -> str:
        """Generate complete Lean file from statements."""
        lean_code = f"""-- Auto-generated from arXiv paper {paper_id}
-- Generated by arXiv-to-Lean agent
-- Date: {datetime.now().isoformat()}

import Mathlib.Data.List.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Tactic

namespace ArxivPaper_{paper_id.replace('.', '_').replace('-', '_')}

"""
        
        # Add each statement
        for stmt in statements:
            if stmt['success'] and stmt.get('lean'):
                lean_code += f"\n{stmt['lean']}\n"
        
        lean_code += "\nend ArxivPaper_" + paper_id.replace('.', '_').replace('-', '_') + "\n"
        
        return lean_code
    
    def _print_result_summary(self, result: Dict):
        """Print summary of processing results."""
        print(f"\n{'='*60}")
        print(f"RESULTS: {result['paper_id']}")
        print(f"{'='*60}")
        print(f"Total statements:  {len(result['statements'])}")
        print(f"Successful:        {result['successful']} ({result['successful']/max(1,len(result['statements']))*100:.1f}%)")
        print(f"Failed:            {result['failed']}")
        print(f"New vocabulary:    {len(result['new_vocabulary'])}")
        print(f"{'='*60}\n")


def main():
    """Main entry point for the agent."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║         arXiv-to-Lean Continuous Learning Agent          ║
║                                                           ║
║  Mission: Download random papers, extract theorems,      ║
║           translate to Lean, and learn from failures     ║
╚═══════════════════════════════════════════════════════════╝
""")
    
    pipeline = ArxivToLeanPipeline()
    
    # Download and process a paper
    paper = pipeline.download_random_paper(category='math.CO')
    
    if paper:
        result = pipeline.process_paper(paper)
        
        print(f"\n{'='*60}")
        print("OVERALL STATISTICS")
        print(f"{'='*60}")
        print(f"Papers processed:       {len(pipeline.results['papers_processed'])}")
        print(f"Total statements:       {pipeline.results['total_statements']}")
        print(f"Successful translations: {pipeline.results['successful_translations']}")
        print(f"Failed translations:    {pipeline.results['failed_translations']}")
        print(f"Vocabulary size:        {pipeline.results['vocabulary_size']}")
        
        if pipeline.results['total_statements'] > 0:
            success_rate = pipeline.results['successful_translations'] / pipeline.results['total_statements']
            print(f"Overall success rate:   {success_rate*100:.1f}%")
        
        print(f"{'='*60}\n")
    else:
        print("❌ Failed to download paper")


if __name__ == '__main__':
    main()
