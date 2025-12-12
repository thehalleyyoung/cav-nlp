"""
Dependency DAG Extraction
=========================

Extract complete dependency graph from mathematical papers.
Identifies definitions, theorems, structures and their dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
from enum import Enum
import re
from collections import defaultdict


class StatementKind(Enum):
    """Types of mathematical statements."""
    DEFINITION = "definition"
    THEOREM = "theorem"
    LEMMA = "lemma"
    COROLLARY = "corollary"
    PROPOSITION = "proposition"
    STRUCTURE = "structure"
    AXIOM = "axiom"
    EXAMPLE = "example"


@dataclass
class Statement:
    """A mathematical statement from a paper."""
    id: str  # Unique identifier
    kind: StatementKind
    name: Optional[str]  # Name if given (e.g., "Cauchy-Schwarz")
    number: Optional[str]  # Number in paper (e.g., "2.1", "Theorem 3")
    text: str  # Full text of statement
    proof_text: Optional[str] = None  # Proof if available
    location: Tuple[int, int] = (0, 0)  # Character span in document
    
    # Extracted structure
    variables: List[str] = field(default_factory=list)
    types_mentioned: List[str] = field(default_factory=list)
    concepts_used: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Statement) and self.id == other.id


@dataclass
class Dependency:
    """Dependency edge in the DAG."""
    from_id: str  # Statement that depends
    to_id: str  # Statement being depended on
    kind: str  # Type of dependency: 'explicit', 'implicit', 'type'
    evidence: Optional[str] = None  # Text evidence for dependency
    
    def __hash__(self):
        return hash((self.from_id, self.to_id, self.kind))


class DependencyDAG:
    """Directed acyclic graph of mathematical statement dependencies."""
    
    def __init__(self):
        self.nodes: Dict[str, Statement] = {}
        self.edges: List[Dependency] = []
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
    
    def add_node(self, statement: Statement) -> None:
        """Add statement to DAG."""
        self.nodes[statement.id] = statement
        if statement.id not in self.adjacency:
            self.adjacency[statement.id] = set()
        if statement.id not in self.reverse_adjacency:
            self.reverse_adjacency[statement.id] = set()
    
    def add_edge(self, dependency: Dependency) -> None:
        """Add dependency edge to DAG."""
        if dependency.from_id not in self.nodes:
            raise ValueError(f"Unknown node: {dependency.from_id}")
        if dependency.to_id not in self.nodes:
            raise ValueError(f"Unknown node: {dependency.to_id}")
        
        self.edges.append(dependency)
        self.adjacency[dependency.from_id].add(dependency.to_id)
        self.reverse_adjacency[dependency.to_id].add(dependency.from_id)
    
    def get_dependencies(self, stmt_id: str) -> Set[str]:
        """Get all statements that stmt_id depends on."""
        return self.adjacency.get(stmt_id, set())
    
    def get_dependents(self, stmt_id: str) -> Set[str]:
        """Get all statements that depend on stmt_id."""
        return self.reverse_adjacency.get(stmt_id, set())
    
    def is_acyclic(self) -> bool:
        """Check if DAG is acyclic."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.adjacency[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                if has_cycle(node):
                    return False
        
        return True
    
    def find_cycle(self) -> Optional[List[str]]:
        """Find a cycle in the graph if one exists."""
        visited = set()
        rec_stack = set()
        path = []
        
        def find_cycle_dfs(node: str) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.adjacency[node]:
                if neighbor not in visited:
                    cycle = find_cycle_dfs(neighbor)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            path.pop()
            return None
        
        for node in self.nodes:
            if node not in visited:
                cycle = find_cycle_dfs(node)
                if cycle:
                    return cycle
        
        return None
    
    def topological_sort(self) -> List[str]:
        """
        Return topological ordering of statements.
        Statements with no dependencies come first.
        """
        if not self.is_acyclic():
            raise ValueError("Cannot topologically sort a cyclic graph")
        
        in_degree = {node: 0 for node in self.nodes}
        for node in self.nodes:
            for dep in self.adjacency[node]:
                in_degree[dep] += 1
        
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort queue to prefer original paper order
            queue.sort(key=lambda n: self.nodes[n].location[0])
            node = queue.pop(0)
            result.append(node)
            
            for dependent in self.reverse_adjacency[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        
        return result
    
    def get_roots(self) -> List[str]:
        """Get nodes with no dependencies (roots of DAG)."""
        return [node_id for node_id in self.nodes 
                if len(self.adjacency[node_id]) == 0]
    
    def get_leaves(self) -> List[str]:
        """Get nodes with no dependents (leaves of DAG)."""
        return [node_id for node_id in self.nodes
                if len(self.reverse_adjacency[node_id]) == 0]
    
    def transitive_dependencies(self, stmt_id: str) -> Set[str]:
        """Get all transitive dependencies of a statement."""
        deps = set()
        visited = set()
        
        def dfs(node: str):
            if node in visited:
                return
            visited.add(node)
            for dep in self.adjacency[node]:
                deps.add(dep)
                dfs(dep)
        
        dfs(stmt_id)
        return deps


class PaperStructureExtractor:
    """Extract dependency DAG from mathematical paper."""
    
    def __init__(self):
        self.statement_patterns = self._build_statement_patterns()
        self.reference_patterns = self._build_reference_patterns()
        self.concept_patterns = {}
    
    def _build_statement_patterns(self) -> Dict[StatementKind, List[re.Pattern]]:
        """Build regex patterns for identifying statements."""
        return {
            StatementKind.DEFINITION: [
                re.compile(r'Definition\s+(\d+(?:\.\d+)*)\s*(?:\(([^)]+)\))?\s*[:.]\s*(.+?)(?=(?:Definition|Theorem|Lemma|Proposition|Proof|$))', re.DOTALL | re.IGNORECASE),
                re.compile(r'Define\s+(.+?)(?=(?:Definition|Theorem|Lemma|Proposition|Proof|$))', re.DOTALL | re.IGNORECASE),
            ],
            StatementKind.THEOREM: [
                re.compile(r'Theorem\s+(\d+(?:\.\d+)*)\s*(?:\(([^)]+)\))?\s*[:.]\s*(.+?)(?=(?:Proof|Definition|Theorem|Lemma|$))', re.DOTALL | re.IGNORECASE),
            ],
            StatementKind.LEMMA: [
                re.compile(r'Lemma\s+(\d+(?:\.\d+)*)\s*(?:\(([^)]+)\))?\s*[:.]\s*(.+?)(?=(?:Proof|Definition|Theorem|Lemma|$))', re.DOTALL | re.IGNORECASE),
            ],
            StatementKind.PROPOSITION: [
                re.compile(r'Proposition\s+(\d+(?:\.\d+)*)\s*(?:\(([^)]+)\))?\s*[:.]\s*(.+?)(?=(?:Proof|Definition|Theorem|Lemma|$))', re.DOTALL | re.IGNORECASE),
            ],
            StatementKind.STRUCTURE: [
                re.compile(r'(?:A|An)\s+([A-Z][a-zA-Z]+)\s+is\s+(?:a|an)\s+(.+?)(?:together\s+with|with)\s+(.+?)(?=(?:Definition|Theorem|Lemma|$))', re.DOTALL | re.IGNORECASE),
            ],
        }
    
    def _build_reference_patterns(self) -> List[re.Pattern]:
        """Build patterns for detecting references to other statements."""
        return [
            re.compile(r'\bby\s+(Definition|Theorem|Lemma|Proposition|Corollary)\s+(\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'\bfrom\s+(Definition|Theorem|Lemma|Proposition)\s+(\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'\busing\s+(Definition|Theorem|Lemma|Proposition)\s+(\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'\bapplying\s+(Definition|Theorem|Lemma|Proposition)\s+(\d+(?:\.\d+)*)', re.IGNORECASE),
            re.compile(r'\b(Definition|Theorem|Lemma|Proposition)\s+(\d+(?:\.\d+)*)', re.IGNORECASE),
        ]
    
    def extract_dag(self, paper_text: str) -> DependencyDAG:
        """
        Extract complete dependency DAG from paper text.
        
        Args:
            paper_text: Full text of mathematical paper
            
        Returns:
            DependencyDAG with all statements and dependencies
        """
        dag = DependencyDAG()
        
        # Phase 1: Extract all statements
        statements = self.extract_statements(paper_text)
        for stmt in statements:
            dag.add_node(stmt)
        
        # Phase 2: Extract dependencies
        for stmt in statements:
            dependencies = self.extract_dependencies(stmt, statements, paper_text)
            for dep in dependencies:
                dag.add_edge(dep)
        
        # Phase 3: Add implicit type dependencies
        self._add_type_dependencies(dag)
        
        return dag
    
    def extract_statements(self, paper_text: str) -> List[Statement]:
        """Extract all mathematical statements from paper."""
        statements = []
        statement_id = 0
        
        for kind, patterns in self.statement_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(paper_text):
                    # Extract components
                    if kind == StatementKind.STRUCTURE:
                        number = None
                        name = match.group(1)
                        text = match.group(0)
                    else:
                        number = match.group(1) if match.lastindex >= 1 else None
                        name = match.group(2) if match.lastindex >= 2 else None
                        text = match.group(3) if match.lastindex >= 3 else match.group(0)
                    
                    stmt_id = f"{kind.value}_{statement_id}"
                    statement_id += 1
                    
                    stmt = Statement(
                        id=stmt_id,
                        kind=kind,
                        name=name,
                        number=number,
                        text=text.strip(),
                        location=match.span()
                    )
                    
                    # Extract structure
                    self._analyze_statement_structure(stmt, paper_text)
                    
                    statements.append(stmt)
        
        # Sort by location in paper
        statements.sort(key=lambda s: s.location[0])
        
        return statements
    
    def _analyze_statement_structure(self, stmt: Statement, paper_text: str) -> None:
        """Analyze internal structure of statement."""
        
        # Extract variable names (single letters, Greek letters)
        var_pattern = re.compile(r'\b([a-z]|[α-ωΑ-Ω])\b')
        stmt.variables = list(set(var_pattern.findall(stmt.text)))
        
        # Extract type names (capitalized words)
        type_pattern = re.compile(r'\b([A-Z][a-zA-Z]+)\b')
        stmt.types_mentioned = list(set(type_pattern.findall(stmt.text)))
        
        # Extract concepts (lowercase multi-word phrases that might be definitions)
        concept_pattern = re.compile(r'\b([a-z]+(?:\s+[a-z]+)*)\b')
        potential_concepts = concept_pattern.findall(stmt.text.lower())
        # Filter to meaningful concepts (length > 1 word or special math terms)
        stmt.concepts_used = [c for c in set(potential_concepts) 
                             if len(c.split()) > 1 or c in ['group', 'space', 'metric', 
                                                              'continuous', 'convergent']]
    
    def extract_dependencies(self, stmt: Statement, all_statements: List[Statement],
                           paper_text: str) -> List[Dependency]:
        """Extract dependencies for a statement."""
        dependencies = []
        
        # 1. EXPLICIT REFERENCES
        # Look for references in statement text and proof
        full_text = stmt.text
        if stmt.proof_text:
            full_text += " " + stmt.proof_text
        
        for pattern in self.reference_patterns:
            for match in pattern.finditer(full_text):
                ref_kind = match.group(1).lower()
                ref_number = match.group(2)
                
                # Find referenced statement
                for other_stmt in all_statements:
                    if other_stmt.number == ref_number and ref_kind in other_stmt.kind.value:
                        dep = Dependency(
                            from_id=stmt.id,
                            to_id=other_stmt.id,
                            kind='explicit',
                            evidence=match.group(0)
                        )
                        dependencies.append(dep)
                        break
        
        # 2. IMPLICIT TYPE DEPENDENCIES
        # Statement uses types defined earlier
        for type_name in stmt.types_mentioned:
            for other_stmt in all_statements:
                if other_stmt.location[0] >= stmt.location[0]:
                    break  # Only look at earlier statements
                
                # Check if other statement defines this type
                if (other_stmt.kind == StatementKind.STRUCTURE or
                    other_stmt.kind == StatementKind.DEFINITION):
                    if type_name.lower() in other_stmt.text.lower():
                        dep = Dependency(
                            from_id=stmt.id,
                            to_id=other_stmt.id,
                            kind='type',
                            evidence=f"Uses type {type_name}"
                        )
                        dependencies.append(dep)
        
        # 3. IMPLICIT CONCEPT DEPENDENCIES
        # Statement uses concepts defined earlier
        for concept in stmt.concepts_used:
            for other_stmt in all_statements:
                if other_stmt.location[0] >= stmt.location[0]:
                    break
                
                if other_stmt.kind == StatementKind.DEFINITION:
                    if concept in other_stmt.text.lower():
                        dep = Dependency(
                            from_id=stmt.id,
                            to_id=other_stmt.id,
                            kind='implicit',
                            evidence=f"Uses concept '{concept}'"
                        )
                        dependencies.append(dep)
        
        # Remove duplicates
        seen = set()
        unique_deps = []
        for dep in dependencies:
            key = (dep.from_id, dep.to_id)
            if key not in seen:
                seen.add(key)
                unique_deps.append(dep)
        
        return unique_deps
    
    def _add_type_dependencies(self, dag: DependencyDAG) -> None:
        """Add implicit type dependencies based on variable usage."""
        
        # Build map of types to their defining statements
        type_definitions = {}
        for stmt_id, stmt in dag.nodes.items():
            if stmt.kind in [StatementKind.STRUCTURE, StatementKind.DEFINITION]:
                for type_name in stmt.types_mentioned:
                    if type_name not in type_definitions:
                        type_definitions[type_name] = stmt_id
        
        # Add dependencies for statements using these types
        for stmt_id, stmt in dag.nodes.items():
            for type_name in stmt.types_mentioned:
                if type_name in type_definitions:
                    def_id = type_definitions[type_name]
                    if def_id != stmt_id and def_id not in dag.get_dependencies(stmt_id):
                        # Add implicit type dependency
                        dep = Dependency(
                            from_id=stmt_id,
                            to_id=def_id,
                            kind='type',
                            evidence=f"Implicit use of type {type_name}"
                        )
                        dag.add_edge(dep)


def visualize_dag(dag: DependencyDAG, output_file: Optional[str] = None) -> str:
    """
    Generate a text visualization of the DAG.
    
    Args:
        dag: The dependency DAG to visualize
        output_file: Optional file to write visualization to
        
    Returns:
        String representation of DAG
    """
    lines = []
    lines.append("DEPENDENCY DAG")
    lines.append("=" * 60)
    lines.append("")
    
    # Show nodes in topological order
    try:
        topo_order = dag.topological_sort()
    except ValueError as e:
        lines.append(f"ERROR: {e}")
        cycle = dag.find_cycle()
        if cycle:
            lines.append(f"Cycle detected: {' → '.join(cycle)}")
        return "\n".join(lines)
    
    lines.append(f"Total statements: {len(dag.nodes)}")
    lines.append(f"Total dependencies: {len(dag.edges)}")
    lines.append(f"Is acyclic: {dag.is_acyclic()}")
    lines.append("")
    
    lines.append("TOPOLOGICAL ORDER:")
    lines.append("-" * 60)
    
    for i, stmt_id in enumerate(topo_order, 1):
        stmt = dag.nodes[stmt_id]
        deps = dag.get_dependencies(stmt_id)
        
        lines.append(f"{i}. [{stmt.kind.value.upper()}] {stmt.id}")
        if stmt.name:
            lines.append(f"   Name: {stmt.name}")
        if stmt.number:
            lines.append(f"   Number: {stmt.number}")
        lines.append(f"   Text: {stmt.text[:100]}...")
        
        if deps:
            lines.append(f"   Dependencies: {', '.join(deps)}")
        else:
            lines.append("   Dependencies: None (root)")
        
        lines.append("")
    
    # Show dependency edges
    lines.append("DEPENDENCY EDGES:")
    lines.append("-" * 60)
    
    for edge in dag.edges:
        from_stmt = dag.nodes[edge.from_id]
        to_stmt = dag.nodes[edge.to_id]
        lines.append(f"{edge.from_id} → {edge.to_id} [{edge.kind}]")
        if edge.evidence:
            lines.append(f"  Evidence: {edge.evidence}")
    
    result = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(result)
    
    return result
