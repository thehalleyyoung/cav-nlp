#!/usr/bin/env python3
"""
Analyze uncovered examples and categorize by linguistic phenomena.
Maps each category to relevant papers from formal linguistics/philosophy of math.
Validates that proposed solutions produce compilable Z3 definitions.
"""

import json
import re
from collections import Counter, defaultdict
from z3 import *

# Load data
examples = json.load(open('cegis_results/training_examples.json'))
rules = json.load(open('cegis_results/learned_rules.json'))

# Find uncovered examples
covered = set()
for rule in rules:
    pattern = rule['syntactic_pattern']
    for i, ex in enumerate(examples):
        try:
            if re.search(pattern, ex['english'], re.IGNORECASE):
                covered.add(i)
        except:
            pass

uncovered = [ex for i, ex in enumerate(examples) if i not in covered]

# Z3 Validation Functions
def validate_z3_compilation(code_snippet: str, category: str) -> tuple[bool, str]:
    """
    Validate that a Z3 code snippet compiles without errors.
    Returns (success, error_message).
    """
    try:
        # Create a fresh solver for each test
        s = Solver()
        
        # Execute the Z3 code in a controlled namespace
        namespace = {
            'Bool': Bool, 'Int': Int, 'Real': Real, 'String': String,
            'Const': Const, 'Function': Function, 'DeclareSort': DeclareSort,
            'ForAll': ForAll, 'Exists': Exists, 'And': And, 'Or': Or, 'Not': Not,
            'Implies': Implies, 'If': If, 'Solver': Solver, 'sat': sat, 'unsat': unsat,
            's': s, 'IntSort': IntSort, 'BoolSort': BoolSort, 'RealSort': RealSort,
            'Array': Array, 'ArraySort': ArraySort, 'Store': Store, 'Select': Select,
        }
        
        # Try to execute the code
        exec(code_snippet, namespace)
        
        # Try to check satisfiability (validates the constraints are well-formed)
        result = s.check()
        
        return True, f"✓ Compiles and checks ({result})"
    
    except SyntaxError as e:
        return False, f"✗ Syntax error: {e}"
    except Exception as e:
        return False, f"✗ Runtime error: {e}"


def test_category_solution(category_name: str, test_cases: list) -> dict:
    """
    Test proposed solutions for a category with actual Z3 compilation.
    Returns statistics on success/failure.
    """
    results = {'success': 0, 'failure': 0, 'errors': []}
    
    for test_case in test_cases:
        success, msg = validate_z3_compilation(test_case['code'], category_name)
        if success:
            results['success'] += 1
        else:
            results['failure'] += 1
            results['errors'].append({
                'test': test_case.get('description', 'unnamed'),
                'error': msg,
                'code': test_case['code']
            })
    
    return results

# Categorize by linguistic phenomena with paper references
categories = {}

# Category 1: Let statements (declarative binding)
let_examples = []
for ex in uncovered:
    if re.match(r'let\s+\w+\s+be', ex['english'], re.IGNORECASE):
        let_examples.append(ex)
categories['let_statements'] = {
    'examples': let_examples,
    'papers': [
        'Ganesalingam (2013) Ch. 4: Definitional mode vs assertional mode',
        'Ranta (1994): Dependent types for variable binding in natural language',
        'Sundholm (1986): Proof theory and meaning (constructive definitions)',
    ],
    'description': 'Variable declarations and type ascriptions',
    'z3_tests': [
        {
            'description': 'Let X be a natural number',
            'code': '''
# Declare X as Int with constraint
X = Int('X')
s.add(X >= 0)  # Natural number constraint
'''
        },
        {
            'description': 'Let P be integral (domain)',
            'code': '''
# Declare domain sort and P
Domain = DeclareSort('Domain')
P = Const('P', Domain)
integral = Function('integral', Domain, BoolSort())
s.add(integral(P))
'''
        },
        {
            'description': 'Let f be a function from reals to reals',
            'code': '''
# Function declaration with type
f = Function('f', RealSort(), RealSort())
x = Real('x')
# f is well-defined on all reals (implicit in Z3)
'''
        }
    ]
}

# Category 2: Anaphora resolution
anaphora_examples = []
for ex in uncovered:
    if re.search(r'\b(their|its|such that|this|these|those|the former|the latter)\b', ex['english']):
        anaphora_examples.append(ex)
categories['anaphora'] = {
    'examples': anaphora_examples,
    'papers': [
        'Ganesalingam (2013) Ch. 6: Anaphora in mathematical discourse',
        'Kamp & Reyle (1993): Discourse Representation Theory (DRT)',
        'Groenendijk & Stokhof (1991): Dynamic predicate logic',
        'Asher (1993): Reference to Abstract Objects in Discourse',
    ],
    'description': 'Pronouns and discourse referents requiring context resolution',
    'z3_tests': [
        {
            'description': 'Y and Z have property P, their union has P',
            'code': '''
# Declare sets and property
Set = DeclareSort('Set')
Y = Const('Y', Set)
Z = Const('Z', Set)
P = Function('P', Set, BoolSort())
union = Function('union', Set, Set, Set)

# Y and Z have P
s.add(P(Y))
s.add(P(Z))

# Their union = union(Y, Z)
their_union = union(Y, Z)
s.add(P(their_union))
'''
        },
        {
            'description': 'Such that binding',
            'code': '''
# W such that w satisfies v
World = DeclareSort('World')
w = Const('w', World)
v = Const('v', World)
satisfies = Function('satisfies', World, World, BoolSort())

# Existential with constraint
W = Const('W', World)
s.add(satisfies(w, v))
'''
        }
    ]
}

# Category 3: Discourse structure
discourse_examples = []
for ex in uncovered:
    if re.match(r'(Assume|Then|Thus|Therefore|Hence|Moreover|Furthermore|Consequently)', ex['english'], re.IGNORECASE):
        discourse_examples.append(ex)
categories['discourse_structure'] = {
    'examples': discourse_examples,
    'papers': [
        'Asher & Lascarides (2003): Logics of Conversation',
        'Mann & Thompson (1988): Rhetorical Structure Theory',
        'Ganesalingam (2013) Ch. 8: Discourse structure in mathematical texts',
        'Webber et al. (2012): Discourse relations in mathematical proofs',
    ],
    'description': 'Discourse markers indicating logical flow and argumentation structure'
}

# Category 4: Metalanguage
metalang_examples = []
for ex in uncovered:
    if re.search(r'(Theorem|Corollary|Lemma|Definition|Proposition|Axiom)\s+\d', ex['english']):
        metalang_examples.append(ex)
categories['metalanguage'] = {
    'examples': metalang_examples,
    'papers': [
        'Ganesalingam (2013) Ch. 3: Metalanguage vs object language',
        'Tarski (1956): The concept of truth in formalized languages',
        'Kohlhase et al. (2011): OMDoc - semantic markup for mathematics',
        'Cramer et al. (2009): Naproche - handling theorem references',
    ],
    'description': 'References to theorems, lemmas, and other mathematical objects'
}

# Category 5: Ellipsis
ellipsis_examples = []
for ex in uncovered:
    if (ex['english'].endswith('has a') or ex['english'].endswith('holds for') or 
        ex['english'].endswith('implies') or ex['english'].endswith('then') or
        ex['english'].endswith(', ') or len(ex['english'].split()) < 5):
        ellipsis_examples.append(ex)
categories['ellipsis'] = {
    'examples': ellipsis_examples,
    'papers': [
        'Merchant (2001): The Syntax of Silence (sluicing and ellipsis)',
        'Dalrymple et al. (1991): Ellipsis and higher-order unification',
        'Ganesalingam (2013) Ch. 5: Implicit content in mathematical language',
        'Crabbe (2004): Implicit content in mathematical texts',
    ],
    'description': 'Omitted or implicit content requiring reconstruction',
    'z3_tests': [
        {
            'description': 'X has a [property from context]',
            'code': '''
# Reconstruction: X has a base (from context)
X = DeclareSort('X')
x = Const('x', X)
has_base = Function('has_base', X, BoolSort())
s.add(has_base(x))
'''
        },
        {
            'description': 'Let P be integral [domain]',
            'code': '''
# Type reconstruction
Domain = DeclareSort('Domain')
P = Const('P', Domain)
integral_domain = Function('integral_domain', Domain, BoolSort())
s.add(integral_domain(P))
'''
        }
    ]
}

# Category 6: Abbreviations and domain-specific notation
abbrev_examples = []
for ex in uncovered:
    if re.search(r'\b(GCH|SPL|CLS|wFn|SAT|Nt|HnT|ZFC|CH)\b', ex['english']):
        abbrev_examples.append(ex)
categories['abbreviations'] = {
    'examples': abbrev_examples,
    'papers': [
        'Ganesalingam (2013) Ch. 2: Abbreviation mechanisms in mathematics',
        'De Bruijn (1994): Mathematical Vernacular',
        'Kohlhase (2006): OMDoc abbreviation definitions',
        'Farmer (2004): Theory interpretation and abbreviations',
    ],
    'description': 'Domain-specific abbreviations requiring expansion'
}

# Category 7: Complex quantification
complex_quant_examples = []
for ex in uncovered:
    if (('for all' in ex['english'].lower() or 'there exists' in ex['english'].lower()) and
        ('such that' in ex['english'].lower() or ',' in ex['english'])):
        complex_quant_examples.append(ex)
categories['complex_quantification'] = {
    'examples': complex_quant_examples,
    'papers': [
        'Ganesalingam (2013) Ch. 5: Binder scope and quantifier raising',
        'Barwise & Cooper (1981): Generalized quantifiers and natural language',
        'May (1977): The Grammar of Quantification (PhD thesis)',
        'Ranta (1994) Ch. 3: Dependent quantification in type theory',
    ],
    'description': 'Nested quantifiers with complex scoping',
    'z3_tests': [
        {
            'description': 'For all n, there exists m > n',
            'code': '''
n = Int('n')
m = Int('m')
# ∀n. ∃m. m > n
s.add(ForAll([n], Exists([m], m > n)))
'''
        },
        {
            'description': 'There exists sequence in P such that property holds',
            'code': '''
P = DeclareSort('P')
Sequence = DeclareSort('Sequence')
seq = Const('seq', Sequence)
in_P = Function('in_P', Sequence, P, BoolSort())
property = Function('property', Sequence, BoolSort())

p = Const('p', P)
# ∃seq. in_P(seq, p) ∧ property(seq)
s.add(Exists([seq], And(in_P(seq, p), property(seq))))
'''
        },
        {
            'description': 'For all n in N satisfies: for all k, condition',
            'code': '''
n = Int('n')
k = Int('n')
satisfies = Function('satisfies', IntSort(), BoolSort())
condition = Function('condition', IntSort(), IntSort(), BoolSort())

# ∀n. (n >= 0 ∧ satisfies(n)) → (∀k. condition(n, k))
s.add(ForAll([n], 
    Implies(And(n >= 0, satisfies(n)),
            ForAll([k], condition(n, k)))))
'''
        }
    ]
}

# Category 8: Presupposition
presup_examples = []
for ex in uncovered:
    if re.search(r'the (unique|only|first|last|greatest|least)', ex['english']):
        presup_examples.append(ex)
categories['presupposition'] = {
    'examples': presup_examples,
    'papers': [
        'Russell (1905): On Denoting (definite descriptions)',
        'Strawson (1950): On Referring (presupposition vs assertion)',
        'Heim (1982): The Semantics of Definite and Indefinite Noun Phrases',
        'Ganesalingam (2013) Ch. 4: Definiteness in mathematics',
    ],
    'description': 'Definite descriptions presupposing existence and uniqueness'
}

# Category 9: Mathematical notation parsing
notation_examples = []
for ex in uncovered:
    if re.search(r'[_\^]\s*\{|\\<|\\>', ex['english']):
        notation_examples.append(ex)
categories['mathematical_notation'] = {
    'examples': notation_examples,
    'papers': [
        'Ganesalingam & Gowers (2017): Automatic problem solving with notation',
        'Kohlhase (2000): OpenMath and MathML',
        'Miller & Pfenning (1992): Higher-order unification for notation',
        'Kamareddine et al. (2004): Computerizing mathematical text',
    ],
    'description': 'Subscripts, superscripts, and special mathematical symbols',
    'z3_tests': [
        {
            'description': 'Subscript notation: x_n',
            'code': '''
# Array indexing for subscripts
x = Array('x', IntSort(), RealSort())
n = Int('n')
x_n = Select(x, n)  # x[n] or x_n
'''
        },
        {
            'description': 'Superscript: 2^n',
            'code': '''
# Power function
power = Function('power', IntSort(), IntSort(), IntSort())
n = Int('n')
result = power(2, n)
# Could also use arithmetic
# result = 2 ** n  # But Z3 Int doesn't support **
'''
        },
        {
            'description': 'Mixed sub/superscript: w_{C}^n',
            'code': '''
# Nested indexing: w indexed by C, then raised to n
W = ArraySort(IntSort(), ArraySort(IntSort(), IntSort()))
w = Array('w', IntSort(), ArraySort(IntSort(), IntSort()))
C = Int('C')
n = Int('n')
# w[C][n] represents w_{C}^{n}
w_C = Select(w, C)
w_C_n = Select(w_C, n)
'''
        }
    ]
}

# Category 10: Coordination ambiguity
coord_examples = []
for ex in uncovered:
    if ex['english'].count(' and ') >= 2 or (ex['english'].count(',') >= 2 and ' and ' in ex['english']):
        coord_examples.append(ex)
categories['coordination'] = {
    'examples': coord_examples,
    'papers': [
        'Steedman (2000) Ch. 8: Coordination in CCG',
        'Partee & Rooth (1983): Generalized conjunction and type ambiguity',
        'Ganesalingam (2013) Ch. 5.4: Coordination scope in mathematics',
        'Dowty (1988): Type raising, functional composition, and coordination',
    ],
    'description': 'Multiple conjuncts with ambiguous attachment',
    'z3_tests': [
        {
            'description': 'A and B and C',
            'code': '''
A = Bool('A')
B = Bool('B')
C = Bool('C')
# Left-associative: (A ∧ B) ∧ C
s.add(And(And(A, B), C))
# Or right-associative: A ∧ (B ∧ C)
# s.add(And(A, And(B, C)))
# Both equivalent in Z3
'''
        },
        {
            'description': 'if P, Q, and R, then S',
            'code': '''
P = Bool('P')
Q = Bool('Q')
R = Bool('R')
S = Bool('S')
# (P ∧ Q ∧ R) → S
s.add(Implies(And(P, Q, R), S))
'''
        }
    ]
}

# Print analysis
print('=' * 80)
print('LINGUISTIC ANALYSIS OF UNCOVERED EXAMPLES')
print('=' * 80)
print(f'\nTotal examples: {len(examples)}')
print(f'Covered by learned rules: {len(covered)} ({len(covered)/len(examples)*100:.1f}%)')
print(f'Uncovered: {len(uncovered)} ({len(uncovered)/len(examples)*100:.1f}%)')

print('\n' + '=' * 80)
print('CATEGORIZATION BY LINGUISTIC PHENOMENON')
print('=' * 80)

# Track validation results
validation_summary = {}

for cat_name, cat_data in sorted(categories.items(), key=lambda x: -len(x[1]['examples'])):
    examples_list = cat_data['examples']
    print(f'\n{cat_name.upper().replace("_", " ")}')
    print('-' * 80)
    print(f'Count: {len(examples_list)} examples')
    print(f'Description: {cat_data["description"]}')
    
    # Z3 Validation
    if 'z3_tests' in cat_data:
        print(f'\n📊 Z3 VALIDATION ({len(cat_data["z3_tests"])} tests):')
        test_results = test_category_solution(cat_name, cat_data['z3_tests'])
        validation_summary[cat_name] = test_results
        
        if test_results['success'] > 0:
            print(f'   ✅ {test_results["success"]} tests passed')
        if test_results['failure'] > 0:
            print(f'   ❌ {test_results["failure"]} tests failed')
            for error in test_results['errors'][:2]:  # Show first 2 errors
                print(f'      • {error["test"]}: {error["error"]}')
    else:
        print(f'\n⚠️  No Z3 validation tests defined yet')
    
    print(f'\nRelevant Papers:')
    for paper in cat_data['papers']:
        print(f'  • {paper}')
    
    print(f'\nSample Examples (showing 5):')
    for i, ex in enumerate(examples_list[:5], 1):
        eng = ex['english'][:100]
        lean = ex['lean'][:100]
        print(f'  {i}. {eng}...')
        print(f'     → {lean}...')
        print()

# Summary statistics
total_categorized = sum(len(cat['examples']) for cat in categories.values())
print('=' * 80)
print('SUMMARY')
print('=' * 80)
print(f'Total categorized: {total_categorized}')
print(f'Overlap (some examples in multiple categories): {total_categorized - len(set(id(ex) for cat in categories.values() for ex in cat["examples"]))}')
print(f'\nTop 5 categories by count:')
for cat_name, cat_data in sorted(categories.items(), key=lambda x: -len(x[1]['examples']))[:5]:
    print(f'  {len(cat_data["examples"]):4d}  {cat_name}')

# Z3 Validation Summary
print('\n' + '=' * 80)
print('Z3 VALIDATION SUMMARY')
print('=' * 80)
total_tests = sum(r['success'] + r['failure'] for r in validation_summary.values())
total_passed = sum(r['success'] for r in validation_summary.values())
total_failed = sum(r['failure'] for r in validation_summary.values())

print(f'\nTotal Z3 tests: {total_tests}')
print(f'Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)')
print(f'Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)')

print(f'\nBy category:')
for cat_name, results in sorted(validation_summary.items(), key=lambda x: -x[1]['success']):
    total = results['success'] + results['failure']
    status = '✓' if results['failure'] == 0 else '⚠'
    print(f'  {status} {cat_name:30s}: {results["success"]}/{total} passed')

if total_failed > 0:
    print(f'\n⚠️  {total_failed} tests failed - need to fix Z3 encodings')
else:
    print(f'\n✅ All Z3 validations passed! Proposed solutions are compilable.')

print('\n' + '=' * 80)
print('RECOMMENDATIONS')
print('=' * 80)
print('''
To improve coverage, integrate the following frameworks:

1. ANAPHORA RESOLUTION (Kamp & Reyle 1993)
   - Implement Discourse Representation Structures (DRS)
   - Track discourse referents across sentences
   - Resolve pronouns to antecedents

2. DISCOURSE STRUCTURE (Asher & Lascarides 2003)
   - Parse discourse markers (Assume, Then, Thus)
   - Build rhetorical structure trees
   - Connect propositions via discourse relations

3. ELLIPSIS RECONSTRUCTION (Merchant 2001)
   - Detect elliptical constructions
   - Reconstruct omitted content from context
   - Use higher-order unification

4. ABBREVIATION EXPANSION (De Bruijn 1994)
   - Build glossary of domain abbreviations
   - Expand GCH → Generalized Continuum Hypothesis
   - Learn abbreviations from corpus

5. PRESUPPOSITION PROJECTION (Heim 1982)
   - Handle definite descriptions (the unique X)
   - Check existence and uniqueness presuppositions
   - Use Z3 to verify presupposition satisfaction

6. NOTATION PARSING (Kamareddine et al. 2004)
   - Separate presentational from content MathML
   - Parse subscripts/superscripts correctly
   - Normalize notation variants
''')
