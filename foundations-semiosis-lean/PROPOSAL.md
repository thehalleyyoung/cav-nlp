## 1. Manifesto: *Mathematics as Semiosis*

### 1.1 Slogan

The central claim of **Semiotic Mathematics** is:

> **Mathematics is semiosis.**
> Every mathematical object is constituted by the way it participates in a web of *sign → object → interpretant* relationships. What we call “structure” is the stable pattern of these meaning-preserving transformations.

Instead of taking *set*, *type*, *space*, or *process* as primitive, we take:

* **Signs** (syntactic, diagrammatic, symbolic artifacts),
* **Objects** (the things those signs are about), and
* **Interpretants** (further signs/structures in which meanings are grasped, compared, or re-expressed)

as the basic ingredients of mathematics, with emphasis on **how they transform**.

Classical foundations treat meaning as parasitic on objects: sets first, then formulas that “talk about them.” Semiotic Mathematics inverts this: *objects are those invariant cores which emerge from the global pattern of sign–object–interpretant relations*.

---

### 1.2 Why “semiosis” and why now?

Modern mathematics sits on an enormous stack of **representations of representations**:

* Formal theories and their models;
* Programs, type systems, proofs, and semantics;
* Diagrams, string calculi, and graphical notations;
* Dualities and correspondences where “the same” thing appears in wildly different guises.

All of these are **semiotic phenomena**: they are about *how signs stand for things* and *how different sign systems can faithfully translate into each other*.

We already have powerful fragmentary formalisms:

* **Model theory** (syntax vs models),
* **Institutions** (signatures, sentences, models, satisfaction),
* **Category theory and toposes** (internal languages and external objects),
* **Denotational semantics** (programs, meanings, and full abstraction),
* **Duality theories** (Stone, Gelfand, Pontryagin, etc.),

but we treat them as separate tools. Semiosis is the unifying perspective:

> A logical theory, a programming language, a coordinate chart on a manifold, and a duality theorem are *all* instances of the same primitive phenomenon:
> **sign systems with meaning functors into worlds of objects, and translations that preserve that meaning.**

Semiotic Mathematics proposes to **elevate that phenomenon to the foundational level**, on par with “everything is a set” or “everything is a homotopy type.”

---

### 1.3 The basic picture

A *semiotic system* has three layers:

1. **Signs** – formulas, diagrams, programs, terms, coordinate expressions, proofs…
2. **Objects** – structures, models, states, spaces, processes…
3. **Interpretants** – semantic profiles of signs: truth assignments, behaviors, types, spectra, sheaves, representations.

The **act of semiosis** is the passage:

[
\text{sign} \quad \xrightarrow{\text{denotation}} \quad \text{object} \quad \xrightarrow{\text{re-expression}} \quad \text{interpretant}
]

The fundamental structure of mathematics is then:

* The **categories of signs** and **objects**,
* The **meaning functor** that explains how signs “apply” to objects,
* The **space of interpretants** where meanings are compared, composed, and internalized,
* The **translations** (functors) between sign systems and between object worlds that preserve semiosis.

Under this view:

* A theorem is a statement that certain semiosis diagrams commute.
* A duality theorem is an equivalence between two semiotic systems with different signs but “the same” objects/interpretants.
* An equivalence of semantics (operational vs denotational, geometric vs algebraic) is an equivalence of interpretants.

---

### 1.4 Relationship to existing foundations

**Set theory**: sets are one particular kind of object; membership is one kind of sign–object relation. Semiotic Mathematics doesn’t deny sets; it explains them as *one semiotic regime among many*, where signs are formulas in ZFC and interpretants are truth values.

**Category theory**: categories are natural habitats for semiotic structure (signs as objects/morphisms, diagrams as composite signs). Semiotic Mathematics extends categorical foundations by making **meaning functors** and **translation functors** first-class citizens. Many categorical gadgets (adjunction, equivalence, topos, Lawvere theory) become recognizable as semiotic phenomena.

**Homotopy Type Theory / Univalent foundations**: HoTT treats equality as paths and types as spaces. Semiotic Mathematics is orthogonal: it treats *reference and interpretation* as primitive, not identity. A future synthesis could study “paths between interpretations” rather than just between points.

**Optimal transport / games / protocols**: these give “everything is geometry,” “everything is interaction,” “everything is process.” Semiosis sits in a different quadrant:

* It is not about cost or flow (OT),
* Not about strategic interaction (games),
* Not about message-passing behavior (protocols),

but about **meaning and translation**, in a precise, compositional sense. OT/games/protocols can be *reinterpreted* semiotically (their formalisms are particular sign systems with semantics), but semiotic foundations do not pivot on them.

---

### 1.5 Program sketch

Semiotic Mathematics is both a **reframing** and a **research program**. Some core goals:

1. **Unified notion of semiotic structure.**
   Define a mathematical object that simultaneously generalizes:

   * model-theoretic institutions,
   * denotational semantics of programming languages,
   * internal language–object correspondences in toposes,
   * representation and duality theories.

2. **Representation theorems as semiotic dualities.**
   Show that many classical dualities (Stone, Gelfand, Hochster, etc.) are instances of a single “semiotic duality” pattern: a category of objects is reconstructed from structured categories of signs and their meaning functors.

3. **Higher semiosis.**
   Treat higher-order interpretants (types of types, logics of logics, meta-languages) as iterated semiosis in a 2-category or ∞-categorical setting.

4. **Semiosis in computation.**
   Recast programming languages, type systems, proof systems, and model checking as specific semiotic structures. Soundness, completeness, and full abstraction become concrete properties of semiosis diagrams.

5. **Semiosis under resource and uncertainty.**
   Extend the value of meaning from simple truth values to probabilities, costs, or epistemic states. This connects to information theory, learning, and LLM/oracle-style computation *without* making them the foundation.

The long-term ambition is that “math is semiosis” becomes as natural a slogan as “math is the science of structures,” but with sharper tools for reasoning about languages, representations, and translations—exactly what modern math and CS lean on most.

---

## 2. Core formalism: Definitions

We’ll set up a minimal but expressive formalism. It’s deliberately close to institutions and categorical semantics, but with an explicit “interpretant” layer.

### 2.1 Value category

Fix a **value category** (\mathcal{V}) that encodes “degrees of meaning/validity.”

For a first pass, you can take:

* (\mathcal{V} = \mathbf{2} = {\bot,\top}) with (\bot \le \top), for classical truth values,

or more generally:

* (\mathcal{V}) a complete Heyting algebra (to model intuitionistic truth / entailment),
* or a complete poset (for more general “grades” of satisfaction).

We will treat (\mathcal{V}) as a posetal category (at most one morphism between any two objects, capturing (\le)).

### 2.2 Semiotic structure

**Definition (Semiotic structure).**
A **semiotic structure** (S) consists of:

* A small category of **signs** (\mathbf{Sig}),
* A category of **objects** (\mathbf{Obj}),
* A value category (\mathcal{V}) as above,
* A **meaning functor** (a generalized satisfaction relation)
  [
  \llbracket - , - \rrbracket : \mathbf{Sig}^{op} \times \mathbf{Obj} \to \mathcal{V}
  ]
  which is functorial in both arguments.

For a sign (s \in \mathbf{Sig}) and an object (A \in \mathbf{Obj}), the value
(\llbracket s, A \rrbracket \in \mathcal{V}) is “the meaning of sign (s) at object (A)” (e.g. truth of a formula in a model, satisfaction of a specification by a program, etc.).

Functoriality means:

* If (f: s' \to s) in (\mathbf{Sig}), then
  [
  \llbracket s, A \rrbracket \le \llbracket s', A \rrbracket
  ]
  (contravariant direction encodes logical consequence / weakening).
* If (g: A \to B) in (\mathbf{Obj}), then
  [
  \llbracket s, A \rrbracket \le \llbracket s, B \rrbracket
  ]
  (covariance captures how structure-preserving maps respect meanings).

Formally, those conditions are expressed as naturality of (\llbracket - , - \rrbracket) as a functor.

From this, we can **currify**:

* For each fixed sign (s), we get its **interpretant** (semantic profile)
  [
  \llbracket s, - \rrbracket : \mathbf{Obj}^{op} \to \mathcal{V},
  ]
  a functor assigning to each object its “meaning value” for (s).
* This defines a functor
  [
  M : \mathbf{Sig} \to [\mathbf{Obj}^{op}, \mathcal{V}], \quad s \mapsto \llbracket s, - \rrbracket,
  ]
  sending signs to their interpretants.
  The functor category ([\mathbf{Obj}^{op}, \mathcal{V}]) is then the **category of interpretants**.

So the Peircean triad is encoded as:

* sign (s \in \mathbf{Sig})
* object (A \in \mathbf{Obj})
* interpretant (M(s) = \llbracket s, - \rrbracket \in [\mathbf{Obj}^{op},\mathcal{V}])

and the act of semiosis is the passage (s \mapsto M(s)).

---

### 2.3 Morphisms of semiotic structures

**Definition (Semiotic morphism).**
Given semiotic structures
[
S_i = (\mathbf{Sig}_i, \mathbf{Obj}_i, \mathcal{V}_i, \llbracket-,-\rrbracket_i), \quad i=1,2,
]
a **semiotic morphism** (F: S_1 \to S_2) consists of:

* A functor on signs (F_{\text{sig}} : \mathbf{Sig}_1 \to \mathbf{Sig}_2),
* A functor on objects (F_{\text{obj}} : \mathbf{Obj}_1 \to \mathbf{Obj}_2),
* A monotone map on values (F_{\mathcal{V}} : \mathcal{V}_1 \to \mathcal{V}_2),

such that satisfaction is preserved in the sense that the diagram
[
\mathbf{Sig}_1^{op} \times \mathbf{Obj}_1
\xrightarrow{\llbracket-,-\rrbracket_1}
\mathcal{V}*1
\xrightarrow{F*{\mathcal{V}}}
\mathcal{V}_2
]
is naturally less than or equal to
[
\mathbf{Sig}*1^{op} \times \mathbf{Obj}*1
\xrightarrow{F*{\text{sig}}^{op} \times F*{\text{obj}}}
\mathbf{Sig}_2^{op} \times \mathbf{Obj}_2
\xrightarrow{\llbracket-,-\rrbracket_2}
\mathcal{V}_2.
]

Intuitively: translating signs and objects along (F) does not *lose* satisfaction.

This gives a category (or 2-category) of semiotic structures and their meaning-preserving translations.

---

### 2.4 Examples (very brief)

* **Tarskian model theory.**

  * (\mathbf{Sig}): syntactic category of formulas in a fixed language (or signatures + sentences).
  * (\mathbf{Obj}): category of structures and homomorphisms.
  * (\mathcal{V} = \mathbf{2}) (truth values).
  * (\llbracket \varphi, M \rrbracket = \top) iff (M \models \varphi).

* **Simple PL semantics.**

  * (\mathbf{Sig}): judgments or typing derivations.
  * (\mathbf{Obj}): semantic domains / operational states.
  * (\mathcal{V}): truth values or refined observables (e.g. sets of possible outcomes).
  * (\llbracket \text{judgment}, \text{model} \rrbracket) = whether the judgment holds in that model.

More elaborate examples will tie in toposes, Lawvere theories, etc., but this suffices for the central theorem.

---

## 3. A central theorem to aim for: Semiotic Representation (Yoneda-style)

The core “math is semiosis” theorem you want is:

> **Objects are determined, up to isomorphism, by how all signs apply to them.**
> Equivalently, the category of objects embeds as a full subcategory of the category of interpretants.

This is a *Yoneda-like* result: just as an object in a category is determined by all morphisms into/from it, a semiotic object is determined by the values of all signs at that object.

### 3.1 Statement (informal then formal)

**Informal version.**
Let (S = (\mathbf{Sig}, \mathbf{Obj}, \mathcal{V}, \llbracket-,-\rrbracket)) be a semiotic structure such that:

1. Different objects are separated by signs: if two objects agree on all sign-values, they are isomorphic.
2. Morphisms between objects are determined by how they transform sign-values.

Then the assignment
[
A \mapsto \llbracket -, A \rrbracket : \mathbf{Sig}^{op} \to \mathcal{V}
]
embeds (\mathbf{Obj}) fully faithfully into the functor category ([\mathbf{Sig}^{op}, \mathcal{V}]).

The essential image can be characterized by simple axioms (e.g. preserving some limits/colimits, a definability condition, etc.). So:

> **The category of objects is equivalent to a category of semantic functors on the sign category.**

This turns the “Peircean triangle” into a strict equivalence: *objects are precisely those interpretants that arise from some semiotic system*.

---

### 3.2 Statement (more precise)

Fix a semiotic structure
[
S = (\mathbf{Sig}, \mathbf{Obj}, \mathcal{V}, \llbracket - , - \rrbracket)
]
where:

* (\mathbf{Sig}) is small,
* (\mathbf{Obj}) is locally small,
* (\mathcal{V}) is a complete posetal category (all small limits/colimits exist).

Define the **semiotic evaluation functor**
[
E_S : \mathbf{Obj} \to [\mathbf{Sig}^{op}, \mathcal{V}], \quad
E_S(A)(s) = \llbracket s, A \rrbracket.
]

We impose two “separation” and “definability” conditions:

1. (**Sign-separation of objects**)
   For any (A,B \in \mathbf{Obj}), if
   [
   \llbracket s, A \rrbracket = \llbracket s, B \rrbracket
   \quad \text{for all } s \in \mathbf{Sig},
   ]
   then (A \cong B) in (\mathbf{Obj}).

2. (**Definability of arrows**)
   For any (A,B \in \mathbf{Obj}), every natural transformation
   [
   \tau : E_S(A) \Rightarrow E_S(B)
   ]
   that is **locally definable by signs** (more on this in a moment) arises uniquely as
   [
   \tau_s = \llbracket s, f \rrbracket : \llbracket s, A \rrbracket \to \llbracket s, B \rrbracket
   ]
   for some morphism (f : A \to B) in (\mathbf{Obj}).

The “locally definable by signs” condition is the technical heart: you restrict to those natural transformations that arise from the intended semantics (e.g. respecting logical consequence, structure of (\mathbf{Sig}), etc.), analogously to how in Stone/Gelfand duality you restrict to certain algebraic/continuous maps.

> **Theorem (Semiotic Representation / Semiotic Yoneda, schematic).**
> Under assumptions (1) and (2), the functor
> [
> E_S : \mathbf{Obj} \to [\mathbf{Sig}^{op}, \mathcal{V}]
> ]
> is fully faithful, and its essential image is precisely the full subcategory
> [
> \mathbf{Int}_S \subseteq [\mathbf{Sig}^{op}, \mathcal{V}]
> ]
> of **semantic functors**, i.e. those functors satisfying the same algebraic/limit-preserving conditions that each (E_S(A)) does.
>
> Thus (\mathbf{Obj} \simeq \mathbf{Int}_S), and every object is (up to canonical isomorphism) determined by its interpretant (E_S(A)).

In more down-to-earth terms:

> Two objects are the same if and only if every sign has the same value on them; and every “admissible relationship” between their semantic profiles comes from an actual morphism of objects.

This theorem is the precise mathematical incarnation of **“math is semiosis”**:

* The “world of objects” is not primitive; it is **recovered as a category of meaning-functors out of the world of signs.**
* Many classical dualities should be realizable as specializations where (\mathbf{Sig}) is a logic, (\mathbf{Obj}) is a class of structures, and (\mathcal{V}) is Booleans, reals, etc.

---

### 3.3 Why this is a good central theorem to chase

1. It is the exact analogue of Yoneda, Stone, Gelfand, etc., but **driven by sign–object semiosis** instead of hom-sets.
2. It gives a common pattern under which:

   * “Models of a theory are determined by which sentences they satisfy,”
   * “A compact Hausdorff space is determined by its algebra of continuous functions,”
   * “A C(^*)-algebra is determined by its spectrum,”
     are all seen as instances of **“objects are reconstructible from interpretants.”**
3. It creates a clean separation between:

   * the **semiotic core** (the quadruple ((\mathbf{Sig},\mathbf{Obj},\mathcal{V},\llbracket-,-\rrbracket))), and
   * the **representation layer** ((\mathbf{Int}_S)), where objects live as functors.

Your research program then becomes:

* Pin down natural **axioms on (\mathbf{Sig},\mathcal{V})** and the meaning functor that guarantee the theorem.
* Work through **major examples** (classical model theory, algebraic geometry, toposes, PL semantics) and show how known dualities and completeness theorems fit the pattern.
* Explore **higher or resource-sensitive versions** (probabilistic (\mathcal{V}), metric-valued (\mathcal{V}), enriched categories).
