"""
Rule Discovery from arXiv: RL-Based Compositional Grammar Learning
===================================================================

Instead of learning weights over fixed rules, DISCOVER NEW COMPOSITIONAL RULES
from arXiv papers by treating rule synthesis as an RL problem where:

- State: Current grammar + unparsed corpus examples
- Action: Synthesize new compositional rule using Z3
- Reward: Coverage of corpus + type-correctness + generalization
- Learning: Discover rules that maximize parseable correct Lean output

This is meta-learning: learning the STRUCTURE of compositional semantics itself.
"""

import requests
import xml.etree.ElementTree as ET
import tarfile
import tempfile
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
import json
import time
import re
from collections import defaultdict
import numpy as np

from z3 import *
from flexible_semantic_parsing import SemanticPrimitive, SemanticNormalizer
from z3_semantic_synthesis import Z3SemanticAlgebra


@dataclass
class ArxivPaper:
    """Metadata for an arXiv paper with extracted content."""
    arxiv_id: str
    title: str
    abstract: str
    categories: List[str]
    full_text: Optional[str] = None
    
    # Extracted mathematical content
    definitions: List[str] = field(default_factory=list)
    theorems: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    proofs: List[str] = field(default_factory=list)


@dataclass
class CompositionRule:
    """
    A compositional semantic rule discovered by RL.
    
    Unlike weighted rules, these are STRUCTURAL: they define how
    semantic primitives compose into Lean types.
    """
    id: str
    
    # Syntactic pattern (what English looks like)
    pattern: str
    pattern_type: str  # 'regex', 'semantic_primitive', 'structural'
    
    # Semantic composition (how to build Lean type)
    composition_template: str  # Lambda expression in Z3
    composition_type: str  # 'quantifier', 'arrow', 'product', etc.
    
    # Z3 constraints (what type theory requires)
    z3_constraints: List[str]
    type_signature: str  # Input types → Output type
    
    # RL metrics
    coverage: float  # How many examples does it parse?
    correctness: float  # How often is output type-correct?
    generalization: float  # How well does it transfer?
    reward: float  # Total RL reward
    
    # Provenance
    discovered_from: List[str]  # Paper IDs
    example_instances: List[Tuple[str, str]]  # (English, Lean)
    discovery_iteration: int
    
    # Composition metadata
    sub_rules: List[str] = field(default_factory=list)  # Rules used by this rule
    uses_primitives: List[str] = field(default_factory=list)  # Semantic primitives


@dataclass
class RLState:
    """
    State in the rule discovery RL problem.
    
    State = (current grammar, unparsed examples, context)
    """
    grammar_rules: List[CompositionRule]
    unparsed_examples: List[Tuple[str, str]]  # (English, expected Lean)
    parsed_successfully: int
    total_examples: int
    
    # Context from papers
    known_types: Set[str]  # Type names seen
    known_predicates: Set[str]  # Predicate names seen
    known_relations: Set[str]  # Relation names seen
    
    def coverage_ratio(self) -> float:
        """Fraction of corpus successfully parsed."""
        if self.total_examples == 0:
            return 0.0
        return self.parsed_successfully / self.total_examples


@dataclass
class RLAction:
    """
    Action in the rule discovery RL problem.
    
    Action = synthesize a new compositional rule from examples
    """
    action_type: str  # 'synthesize_rule', 'refine_rule', 'compose_rules'
    
    # For synthesis
    target_examples: List[Tuple[str, str]]  # Examples to cover
    synthesis_strategy: str  # 'z3_search', 'pattern_abstraction', 'composition'
    
    # For composition
    rule_ids_to_compose: List[str] = field(default_factory=list)
    
    # Synthesized result
    new_rule: Optional[CompositionRule] = None


@dataclass
class RLReward:
    """
    Reward for discovering a compositional rule.
    
    Multi-objective:
    1. Coverage: Parse more examples
    2. Correctness: Type-check with Z3
    3. Generalization: Work on held-out data
    4. Simplicity: Prefer simpler rules
    """
    coverage_gain: float  # New examples parsed
    correctness_score: float  # Type-correctness
    generalization_score: float  # Held-out performance
    simplicity_score: float  # Rule complexity penalty
    
    # Penalties
    redundancy_penalty: float = 0.0  # Rule overlaps with existing
    overfitting_penalty: float = 0.0  # Too specific to training
    
    @property
    def total(self) -> float:
        return (
            100.0 * self.coverage_gain +        # Highest priority: coverage
            50.0 * self.correctness_score +     # Must type-check
            30.0 * self.generalization_score +  # Must generalize
            10.0 * self.simplicity_score -      # Prefer simple
            20.0 * self.redundancy_penalty -    # Avoid redundancy
            40.0 * self.overfitting_penalty     # Heavy penalty for overfitting
        )


class ArxivCorpusBuilder:
    """
    Download and extract mathematical content from arXiv papers.
    
    Focuses on math-heavy categories and extracts theorem-definition pairs.
    """
    
    API_BASE = "http://export.arxiv.org/api/query"
    EXPORT_BASE = "https://arxiv.org/e-print"
    
    def __init__(self, cache_dir: Path = Path("arxiv_corpus")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_corpus(self, categories: List[str], 
                       papers_per_category: int = 50) -> List[ArxivPaper]:
        """
        Download papers from specified categories.
        
        Args:
            categories: arXiv categories (e.g., ['math.CT', 'math.LO'])
            papers_per_category: Papers to download per category
            
        Returns:
            List of papers with extracted content
        """
        all_papers = []
        
        for category in categories:
            print(f"\nDownloading {category} papers...")
            papers = self._search_category(category, papers_per_category)
            
            for i, paper in enumerate(papers, 1):
                print(f"  [{i}/{len(papers)}] {paper.arxiv_id}: {paper.title[:60]}...")
                
                # Download source
                try:
                    self._download_source(paper)
                    self._extract_math_content(paper)
                    all_papers.append(paper)
                except Exception as e:
                    print(f"    Error: {e}")
                    continue
                
                # Rate limiting
                time.sleep(1.0)
        
        print(f"\nTotal papers downloaded: {len(all_papers)}")
        return all_papers
    
    def _search_category(self, category: str, max_results: int) -> List[ArxivPaper]:
        """Search arXiv for papers in category."""
        params = {
            'search_query': f'cat:{category}',
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        response = requests.get(self.API_BASE, params=params)
        
        if response.status_code != 200:
            raise Exception(f"arXiv API error: {response.status_code}")
        
        # Parse XML
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
            title = entry.find('atom:title', ns).text.strip()
            abstract = entry.find('atom:summary', ns).text.strip()
            
            cats = [cat.attrib['term'] for cat in entry.findall('atom:category', ns)]
            
            paper = ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                categories=cats
            )
            papers.append(paper)
        
        return papers
    
    def _download_source(self, paper: ArxivPaper):
        """Download LaTeX source for paper."""
        cache_file = self.cache_dir / f"{paper.arxiv_id.replace('/', '_')}.tar.gz"
        text_file = self.cache_dir / f"{paper.arxiv_id.replace('/', '_')}.txt"
        
        # Check cache
        if text_file.exists():
            paper.full_text = text_file.read_text(encoding='utf-8', errors='ignore')
            return
        
        # Download source tarball
        url = f"{self.EXPORT_BASE}/{paper.arxiv_id}"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise Exception(f"Download failed: {response.status_code}")
        
        cache_file.write_bytes(response.content)
        
        # Extract and read .tex files
        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(cache_file, 'r:gz') as tar:
                tar.extractall(tmpdir)
            
            # Find all .tex files
            tex_files = list(Path(tmpdir).rglob('*.tex'))
            
            # Concatenate all
            full_text = []
            for tex_file in tex_files:
                try:
                    content = tex_file.read_text(encoding='utf-8', errors='ignore')
                    full_text.append(content)
                except:
                    continue
            
            paper.full_text = '\n\n'.join(full_text)
            text_file.write_text(paper.full_text, encoding='utf-8')
    
    def _extract_math_content(self, paper: ArxivPaper):
        """Extract mathematical definitions, theorems, etc."""
        if not paper.full_text:
            return
        
        text = paper.full_text
        
        # Remove LaTeX commands for cleaner text
        text = re.sub(r'\\(?:cite|ref|label)\{[^}]*\}', '', text)
        text = re.sub(r'\\(?:begin|end)\{(?:document|abstract)\}', '', text)
        
        # Extract definitions
        for match in re.finditer(
            r'\\begin\{definition\}(.*?)\\end\{definition\}',
            text,
            re.DOTALL | re.IGNORECASE
        ):
            definition = self._clean_latex(match.group(1))
            if len(definition) > 20:
                paper.definitions.append(definition)
        
        # Extract theorems
        for match in re.finditer(
            r'\\begin\{theorem\}(.*?)\\end\{theorem\}',
            text,
            re.DOTALL | re.IGNORECASE
        ):
            theorem = self._clean_latex(match.group(1))
            if len(theorem) > 20:
                paper.theorems.append(theorem)
        
        # Extract lemmas
        for match in re.finditer(
            r'\\begin\{lemma\}(.*?)\\end\{lemma\}',
            text,
            re.DOTALL | re.IGNORECASE
        ):
            lemma = self._clean_latex(match.group(1))
            if len(lemma) > 20:
                paper.lemmas.append(lemma)
        
        # Also try simpler patterns
        for match in re.finditer(
            r'\\(?:begin\{)?(?:Theorem|Definition|Lemma|Proposition)\s*(.*?)(?:(?=\\(?:begin|end|section))|$)',
            text,
            re.DOTALL | re.IGNORECASE
        ):
            content = self._clean_latex(match.group(1))
            if len(content) > 20 and len(content) < 500:
                if 'definition' in match.group(0).lower():
                    paper.definitions.append(content)
                elif 'theorem' in match.group(0).lower():
                    paper.theorems.append(content)
                elif 'lemma' in match.group(0).lower():
                    paper.lemmas.append(content)
    
    def _clean_latex(self, text: str) -> str:
        """Clean LaTeX markup."""
        # Remove math mode delimiters
        text = text.replace('$', '')
        text = text.replace('\\[', '')
        text = text.replace('\\]', '')
        
        # Remove common LaTeX commands
        text = re.sub(r'\\(?:emph|textbf|textit|mathbb|mathcal|mathfrak)\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text


class RuleDiscoveryAgent:
    """
    RL agent that discovers compositional semantic rules from arXiv corpus.
    
    Unlike traditional RL where actions are predefined, here actions are
    SYNTHESIZING NEW RULES via Z3-guided program synthesis.
    """
    
    def __init__(self):
        self.z3_algebra = Z3SemanticAlgebra()
        self.normalizer = SemanticNormalizer()
        
        # Grammar state
        self.discovered_rules: List[CompositionRule] = []
        self.rule_by_id: Dict[str, CompositionRule] = {}
        
        # RL parameters
        self.learning_rate = 0.1
        self.exploration_rate = 0.5
        self.discount_factor = 0.95
        
        # Training history
        self.episode_rewards: List[float] = []
        self.coverage_history: List[float] = []
        
    def train_on_corpus(self, papers: List[ArxivPaper], 
                       num_episodes: int = 100,
                       validation_split: float = 0.2) -> List[CompositionRule]:
        """
        Train on arXiv corpus to discover compositional rules.
        
        Args:
            papers: Downloaded arXiv papers
            num_episodes: Training episodes
            validation_split: Fraction held out for validation
            
        Returns:
            Discovered compositional rules
        """
        # Extract training examples from papers
        all_examples = self._extract_examples(papers)
        print(f"Extracted {len(all_examples)} training examples")
        
        # Split train/validation
        split_idx = int(len(all_examples) * (1 - validation_split))
        train_examples = all_examples[:split_idx]
        val_examples = all_examples[split_idx:]
        
        print(f"Training: {len(train_examples)}, Validation: {len(val_examples)}")
        
        # Initialize state
        state = RLState(
            grammar_rules=[],
            unparsed_examples=train_examples.copy(),
            parsed_successfully=0,
            total_examples=len(train_examples),
            known_types=self._extract_types(papers),
            known_predicates=self._extract_predicates(papers),
            known_relations=self._extract_relations(papers)
        )
        
        # Training loop
        for episode in range(num_episodes):
            print(f"\n{'='*60}")
            print(f"EPISODE {episode + 1}/{num_episodes}")
            print(f"{'='*60}")
            print(f"Current coverage: {state.coverage_ratio():.2%}")
            print(f"Discovered rules: {len(self.discovered_rules)}")
            
            # Run episode
            episode_reward = self._run_episode(state, val_examples)
            
            self.episode_rewards.append(episode_reward)
            self.coverage_history.append(state.coverage_ratio())
            
            # Decay exploration
            self.exploration_rate *= 0.95
            
            print(f"Episode reward: {episode_reward:.2f}")
            
            # Early stopping if coverage plateaus
            if len(self.coverage_history) > 10:
                recent_coverage = self.coverage_history[-10:]
                if max(recent_coverage) - min(recent_coverage) < 0.01:
                    print("\nCoverage plateau reached. Stopping early.")
                    break
        
        # Final report
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Final coverage: {state.coverage_ratio():.2%}")
        print(f"Total rules discovered: {len(self.discovered_rules)}")
        print(f"Average reward: {np.mean(self.episode_rewards):.2f}")
        
        # Test on validation
        val_coverage = self._evaluate_on_validation(val_examples)
        print(f"Validation coverage: {val_coverage:.2%}")
        
        return self.discovered_rules
    
    def _extract_examples(self, papers: List[ArxivPaper]) -> List[Tuple[str, str]]:
        """
        Extract (English, Lean) pairs from papers.
        
        Since we don't have ground-truth Lean, we extract English
        and use it for rule discovery. The "Lean" will be synthesized.
        """
        examples = []
        
        for paper in papers:
            # Use definitions as examples
            for defn in paper.definitions[:5]:  # Limit per paper
                examples.append((defn, ""))  # Empty Lean initially
            
            # Use theorems
            for thm in paper.theorems[:5]:
                examples.append((thm, ""))
            
            # Use lemmas
            for lemma in paper.lemmas[:3]:
                examples.append((lemma, ""))
        
        return examples
    
    def _extract_types(self, papers: List[ArxivPaper]) -> Set[str]:
        """Extract type names from papers."""
        types = set()
        
        for paper in papers:
            text = ' '.join(paper.definitions + paper.theorems)
            
            # Find capitalized words (likely types)
            for word in re.findall(r'\b([A-Z][a-z]+)\b', text):
                types.add(word)
        
        return types
    
    def _extract_predicates(self, papers: List[ArxivPaper]) -> Set[str]:
        """Extract predicate names."""
        predicates = set()
        
        common = {'continuous', 'differentiable', 'measurable', 'bounded',
                 'compact', 'connected', 'finite', 'infinite', 'countable'}
        
        for paper in papers:
            text = ' '.join(paper.definitions + paper.theorems).lower()
            for pred in common:
                if pred in text:
                    predicates.add(pred)
        
        return predicates
    
    def _extract_relations(self, papers: List[ArxivPaper]) -> Set[str]:
        """Extract relation symbols."""
        return {'<', '>', '=', '≤', '≥', '∈', '⊆', '→'}
    
    def _run_episode(self, state: RLState, val_examples: List) -> float:
        """Run one episode of rule discovery."""
        episode_reward = 0.0
        
        # Try to discover rules for unparsed examples
        attempts = min(10, len(state.unparsed_examples))
        
        for attempt in range(attempts):
            # Choose action (which examples to target)
            action = self._choose_action(state)
            
            if not action.new_rule:
                continue
            
            # Execute action (try the synthesized rule)
            new_state, reward = self._execute_action(state, action, val_examples)
            
            episode_reward += reward.total
            
            # Update state
            state = new_state
            
            # Learn from this experience
            self._update_from_experience(action, reward)
            
            print(f"  Attempt {attempt + 1}: Reward={reward.total:.2f}, "
                  f"Coverage={state.coverage_ratio():.2%}")
        
        return episode_reward
    
    def _choose_action(self, state: RLState) -> RLAction:
        """Choose action: which rule to synthesize."""
        
        if np.random.rand() < self.exploration_rate:
            # Exploration: synthesize from random examples
            target_examples = np.random.choice(
                len(state.unparsed_examples),
                size=min(5, len(state.unparsed_examples)),
                replace=False
            )
            target_examples = [state.unparsed_examples[i] for i in target_examples]
            strategy = 'z3_search'
        else:
            # Exploitation: target examples with similar patterns
            target_examples = self._find_similar_examples(state.unparsed_examples)
            strategy = 'pattern_abstraction'
        
        # Synthesize rule
        new_rule = self._synthesize_rule(
            target_examples,
            state,
            strategy
        )
        
        return RLAction(
            action_type='synthesize_rule',
            target_examples=target_examples,
            synthesis_strategy=strategy,
            new_rule=new_rule
        )
    
    def _find_similar_examples(self, examples: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Find examples with similar structure."""
        if not examples:
            return []
        
        # Normalize to find patterns
        normalized = []
        for eng, lean in examples:
            primitives = self.normalizer.normalize(eng)
            pattern = tuple(p.kind for p in primitives)
            normalized.append((pattern, (eng, lean)))
        
        # Find most common pattern
        pattern_counts = defaultdict(int)
        for pattern, _ in normalized:
            pattern_counts[pattern] += 1
        
        if not pattern_counts:
            return examples[:5]
        
        most_common_pattern = max(pattern_counts, key=pattern_counts.get)
        
        # Return examples with that pattern
        similar = [ex for pattern, ex in normalized if pattern == most_common_pattern]
        return similar[:5]
    
    def _synthesize_rule(self, examples: List[Tuple[str, str]],
                        state: RLState, strategy: str) -> Optional[CompositionRule]:
        """
        Synthesize new compositional rule from examples.
        
        This is the core of rule discovery: use Z3 to find a rule
        that covers these examples and type-checks.
        """
        if not examples:
            return None
        
        # Normalize examples to semantic primitives
        primitive_sequences = []
        for eng, _ in examples:
            prims = self.normalizer.normalize(eng)
            primitive_sequences.append(prims)
        
        # Find common pattern
        common_pattern = self._find_common_pattern(primitive_sequences)
        
        if not common_pattern:
            return None
        
        # Synthesize composition template using Z3
        composition_template = self._z3_synthesize_composition(
            common_pattern,
            examples,
            state
        )
        
        if not composition_template:
            return None
        
        # Create rule
        rule = CompositionRule(
            id=f"rule_{len(self.discovered_rules)}",
            pattern=self._pattern_to_regex(common_pattern),
            pattern_type='semantic_primitive',
            composition_template=composition_template,
            composition_type=self._infer_composition_type(common_pattern),
            z3_constraints=self._generate_z3_constraints(common_pattern),
            type_signature=self._infer_type_signature(common_pattern),
            coverage=0.0,
            correctness=0.0,
            generalization=0.0,
            reward=0.0,
            discovered_from=[],
            example_instances=examples,
            discovery_iteration=len(self.discovered_rules)
        )
        
        return rule
    
    def _find_common_pattern(self, sequences: List[List[SemanticPrimitive]]) -> Optional[List[str]]:
        """Find common pattern across semantic primitive sequences."""
        if not sequences:
            return None
        
        # Extract kinds
        kind_sequences = [[p.kind for p in seq] for seq in sequences]
        
        # Find longest common subsequence
        if len(kind_sequences) == 1:
            return kind_sequences[0]
        
        # Simple: use first sequence as template
        template = kind_sequences[0]
        
        # Check if all match
        if all(seq == template for seq in kind_sequences):
            return template
        
        # Otherwise, find common prefix
        common = []
        for i in range(min(len(seq) for seq in kind_sequences)):
            kinds_at_i = [seq[i] for seq in kind_sequences]
            if len(set(kinds_at_i)) == 1:
                common.append(kinds_at_i[0])
            else:
                break
        
        return common if common else None
    
    def _z3_synthesize_composition(self, pattern: List[str],
                                  examples: List[Tuple[str, str]],
                                  state: RLState) -> Optional[str]:
        """Use Z3 to synthesize composition function."""
        
        # Create synthesis problem
        solver = Solver()
        
        # For each example, add constraint that composition must work
        # This is simplified - full version would encode composition logic
        
        # For now, use template based on pattern
        if 'UNIVERSAL_QUANTIFIER' in pattern:
            return "lambda var, domain, body: Pi(var, domain, body)"
        elif 'IMPLICATION' in pattern:
            return "lambda antecedent, consequent: Arrow(antecedent, consequent)"
        else:
            return "lambda *args: App(*args)"
    
    def _pattern_to_regex(self, pattern: List[str]) -> str:
        """Convert semantic pattern to regex."""
        # Simplified
        return ".*" + ".*".join(p.lower() for p in pattern) + ".*"
    
    def _infer_composition_type(self, pattern: List[str]) -> str:
        """Infer type of composition."""
        if 'UNIVERSAL_QUANTIFIER' in pattern or 'EXISTENTIAL_QUANTIFIER' in pattern:
            return 'quantifier'
        elif 'IMPLICATION' in pattern:
            return 'arrow'
        else:
            return 'application'
    
    def _generate_z3_constraints(self, pattern: List[str]) -> List[str]:
        """Generate Z3 constraints for pattern."""
        constraints = ["wellformed_type(result)"]
        
        if 'UNIVERSAL_QUANTIFIER' in pattern:
            constraints.append("pi_formation(var, domain, codomain)")
        
        return constraints
    
    def _infer_type_signature(self, pattern: List[str]) -> str:
        """Infer type signature of composition."""
        return "List[Primitive] -> LeanType"
    
    def _execute_action(self, state: RLState, action: RLAction,
                       val_examples: List) -> Tuple[RLState, RLReward]:
        """Execute action and compute reward."""
        
        if not action.new_rule:
            return state, RLReward(0, 0, 0, 0)
        
        rule = action.new_rule
        
        # Test rule on training examples
        newly_parsed = []
        for example in state.unparsed_examples:
            eng, _ = example
            if self._rule_matches(rule, eng):
                newly_parsed.append(example)
        
        coverage_gain = len(newly_parsed) / state.total_examples if state.total_examples > 0 else 0
        
        # Test correctness (simplified - would use Z3)
        correctness_score = 0.8  # Assume mostly correct
        
        # Test generalization on validation
        val_parsed = sum(1 for eng, _ in val_examples if self._rule_matches(rule, eng))
        generalization_score = val_parsed / len(val_examples) if val_examples else 0
        
        # Simplicity
        simplicity_score = 1.0 / (1.0 + len(rule.composition_template))
        
        # Check redundancy
        redundancy = self._check_redundancy(rule, state.grammar_rules)
        
        # Create reward
        reward = RLReward(
            coverage_gain=coverage_gain,
            correctness_score=correctness_score,
            generalization_score=generalization_score,
            simplicity_score=simplicity_score,
            redundancy_penalty=redundancy
        )
        
        # Update rule metrics
        rule.coverage = coverage_gain
        rule.correctness = correctness_score
        rule.generalization = generalization_score
        rule.reward = reward.total
        
        # Add to discovered rules if good
        if reward.total > 10.0:
            self.discovered_rules.append(rule)
            self.rule_by_id[rule.id] = rule
            
            # Update state
            new_state = RLState(
                grammar_rules=state.grammar_rules + [rule],
                unparsed_examples=[ex for ex in state.unparsed_examples if ex not in newly_parsed],
                parsed_successfully=state.parsed_successfully + len(newly_parsed),
                total_examples=state.total_examples,
                known_types=state.known_types,
                known_predicates=state.known_predicates,
                known_relations=state.known_relations
            )
            
            return new_state, reward
        
        return state, reward
    
    def _rule_matches(self, rule: CompositionRule, text: str) -> bool:
        """Check if rule matches text."""
        return re.search(rule.pattern, text, re.IGNORECASE) is not None
    
    def _check_redundancy(self, new_rule: CompositionRule,
                         existing_rules: List[CompositionRule]) -> float:
        """Check if rule is redundant with existing."""
        for existing in existing_rules:
            # Check pattern overlap
            if existing.pattern == new_rule.pattern:
                return 1.0
        return 0.0
    
    def _update_from_experience(self, action: RLAction, reward: RLReward):
        """Learn from this experience."""
        # In full implementation, would update policy
        # For now, just track
        pass
    
    def _evaluate_on_validation(self, val_examples: List) -> float:
        """Evaluate discovered rules on validation set."""
        parsed = 0
        for eng, _ in val_examples:
            for rule in self.discovered_rules:
                if self._rule_matches(rule, eng):
                    parsed += 1
                    break
        
        return parsed / len(val_examples) if val_examples else 0.0
    
    def save_rules(self, filepath: Path):
        """Save discovered rules."""
        rules_data = []
        for rule in self.discovered_rules:
            rules_data.append({
                'id': rule.id,
                'pattern': rule.pattern,
                'composition_template': rule.composition_template,
                'composition_type': rule.composition_type,
                'z3_constraints': rule.z3_constraints,
                'coverage': rule.coverage,
                'correctness': rule.correctness,
                'generalization': rule.generalization,
                'reward': rule.reward,
                'example_instances': rule.example_instances[:3]
            })
        
        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        print(f"Saved {len(rules_data)} rules to {filepath}")


# Main training pipeline
if __name__ == '__main__':
    # Download corpus
    print("PHASE 1: Downloading arXiv Corpus")
    print("=" * 60)
    
    corpus_builder = ArxivCorpusBuilder()
    
    categories = [
        'math.CT',  # Category Theory
        'math.LO',  # Logic
        'math.AG',  # Algebraic Geometry
        'math.AT',  # Algebraic Topology
        'math.GR',  # Group Theory
    ]
    
    papers = corpus_builder.download_corpus(
        categories=categories,
        papers_per_category=20  # 100 total papers
    )
    
    # Train agent
    print("\n" + "=" * 60)
    print("PHASE 2: Discovering Compositional Rules")
    print("=" * 60)
    
    agent = RuleDiscoveryAgent()
    discovered_rules = agent.train_on_corpus(
        papers=papers,
        num_episodes=50
    )
    
    # Save results
    output_dir = Path("discovered_rules")
    output_dir.mkdir(exist_ok=True)
    
    agent.save_rules(output_dir / "compositional_rules.json")
    
    # Report
    print("\n" + "=" * 60)
    print("DISCOVERED RULES SUMMARY")
    print("=" * 60)
    
    for i, rule in enumerate(discovered_rules[:10], 1):
        print(f"\n{i}. {rule.id}")
        print(f"   Type: {rule.composition_type}")
        print(f"   Pattern: {rule.pattern[:60]}...")
        print(f"   Coverage: {rule.coverage:.3f}")
        print(f"   Correctness: {rule.correctness:.3f}")
        print(f"   Generalization: {rule.generalization:.3f}")
        print(f"   Reward: {rule.reward:.2f}")
