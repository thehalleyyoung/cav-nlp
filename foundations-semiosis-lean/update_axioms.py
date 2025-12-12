import json
from datetime import datetime

# Load the current axioms file
with open('axioms_to_prove.json', 'r') as f:
    data = json.load(f)

# Define the new axioms for Stone duality
new_axioms = [
    {
        "name": "StoneSpace",
        "statement": "axiom StoneSpace : Type",
        "source_theorem": "thm:ch1:stone_duality",
        "source_chapter": 1,
        "added_timestamp": datetime.now().isoformat(),
        "status": "unproven",
        "confidence": "high",
        "proof_strategy_hint": "StoneSpace should be defined as a structure containing a topological space that is compact, Hausdorff, and totally disconnected. In Lean 4, this would be: structure StoneSpace where carrier : Type* [topology : TopologicalSpace carrier] [compact : CompactSpace carrier] [hausdorff : T2Space carrier] [totallyDisconnected : TotallyDisconnectedSpace carrier]. This is a standard topological definition. The proof would require importing Mathlib topology modules and constructing the structure with the appropriate type class instances."
    },
    {
        "name": "StoneSpace.category",
        "statement": "axiom StoneSpace.category : Category StoneSpace",
        "source_theorem": "thm:ch1:stone_duality",
        "source_chapter": 1,
        "added_timestamp": datetime.now().isoformat(),
        "status": "unproven",
        "confidence": "high",
        "proof_strategy_hint": "The category structure on Stone spaces has continuous maps as morphisms. This should be definable using Mathlib's ContinuousMap type: instance : Category StoneSpace where Hom X Y := ContinuousMap X.carrier Y.carrier; id X := ContinuousMap.id X.carrier; comp f g := ContinuousMap.comp g f. This is standard - Stone spaces with continuous maps form a concrete category. The proof requires showing composition is associative and identity laws hold, which follow from properties of continuous functions in Mathlib."
    },
    {
        "name": "spectrumFunctor",
        "statement": "axiom spectrumFunctor : Functor BoolAlgᵒᵖ StoneSpace",
        "source_theorem": "thm:ch1:stone_duality",
        "source_chapter": 1,
        "added_timestamp": datetime.now().isoformat(),
        "status": "unproven",
        "confidence": "high",
        "proof_strategy_hint": "The spectrum functor maps each Boolean algebra B to Spec(B), the space of ultrafilters (or prime ideals) on B with the Stone topology. For a Boolean algebra homomorphism h: B → B', the induced map Spec(h): Spec(B') → Spec(B) sends an ultrafilter P on B' to the ultrafilter h^{-1}(P) on B. This is a classical construction in Stone duality. To prove as a theorem: (1) Define ultrafilters/prime ideals on Boolean algebras, (2) Define the Stone topology on the set of ultrafilters (basic opens are {P | b ∈ P}), (3) Prove this space is Stone (compact, Hausdorff, totally disconnected), (4) Show morphisms are continuous and functorial. This requires significant formalization of Boolean algebra theory and Stone topology."
    },
    {
        "name": "clopenAlgebraFunctor",
        "statement": "axiom clopenAlgebraFunctor : Functor StoneSpace BoolAlgᵒᵖ",
        "source_theorem": "thm:ch1:stone_duality",
        "source_chapter": 1,
        "added_timestamp": datetime.now().isoformat(),
        "status": "unproven",
        "confidence": "high",
        "proof_strategy_hint": "The clopen algebra functor maps each Stone space X to Clop(X), the Boolean algebra of clopen (closed and open) subsets under union, intersection, and complementation. For a continuous map f: X → Y, the induced Boolean algebra homomorphism Clop(f): Clop(Y) → Clop(X) sends a clopen set U ⊆ Y to its preimage f^{-1}(U) ⊆ X. To prove as a theorem: (1) Show clopen sets form a Boolean algebra (closed under boolean operations), (2) Show continuous maps preserve clopenness (preimages of clopen sets are clopen), (3) Verify functoriality (preserves identities and composition). This is standard topology: in totally disconnected spaces, clopen sets separate points and form a complete Boolean algebra."
    },
    {
        "name": "stoneUnitIso",
        "statement": "axiom stoneUnitIso : 𝟭 BoolAlgᵒᵖ ≅ spectrumFunctor ⋙ clopenAlgebraFunctor",
        "source_theorem": "thm:ch1:stone_duality",
        "source_chapter": 1,
        "added_timestamp": datetime.now().isoformat(),
        "status": "unproven",
        "confidence": "high",
        "proof_strategy_hint": "This states that for any Boolean algebra B, there is a natural isomorphism B ≅ Clop(Spec(B)). The forward map sends each element b ∈ B to the clopen set {P ∈ Spec(B) | b ∈ P} (the set of ultrafilters containing b). The inverse map sends a clopen set U ⊆ Spec(B) to the unique element b such that U = {P | b ∈ P}. To prove: (1) Show the forward map is a Boolean algebra homomorphism, (2) Show it's bijective using Stone's representation theorem, (3) Verify naturality in B. This is one half of the classical Stone duality theorem - Boolean algebras embed into the clopen algebras of their spectra. The proof requires Stone's representation theorem: every Boolean algebra is isomorphic to a field of sets."
    },
    {
        "name": "stoneCounitIso",
        "statement": "axiom stoneCounitIso : clopenAlgebraFunctor ⋙ spectrumFunctor ≅ 𝟭 StoneSpace",
        "source_theorem": "thm:ch1:stone_duality",
        "source_chapter": 1,
        "added_timestamp": datetime.now().isoformat(),
        "status": "unproven",
        "confidence": "high",
        "proof_strategy_hint": "This states that for any Stone space X, there is a natural homeomorphism X ≅ Spec(Clop(X)). The forward map sends each point x ∈ X to the ultrafilter {U ∈ Clop(X) | x ∈ U} (the set of clopen sets containing x). The inverse map sends an ultrafilter P on Clop(X) to the unique point in ∩{U ∈ P}, which is non-empty by compactness. To prove: (1) Show the forward map is well-defined (each point determines an ultrafilter), (2) Show it's bijective using compactness and total disconnectedness, (3) Show it's a homeomorphism (continuous with continuous inverse), (4) Verify naturality in X. This is the other half of Stone duality - Stone spaces embed into the spectra of their clopen algebras. The proof uses the key fact that in a Stone space, clopen sets separate points and generate the topology."
    },
    {
        "name": "stoneTriangle",
        "statement": "axiom stoneTriangle : ∀ (B : BoolAlgᵒᵖ), spectrumFunctor.map (stoneUnitIso.hom.app B) ≫ stoneCounitIso.hom.app (spectrumFunctor.obj B) = 𝟙 (spectrumFunctor.obj B)",
        "source_theorem": "thm:ch1:stone_duality",
        "source_chapter": 1,
        "added_timestamp": datetime.now().isoformat(),
        "status": "unproven",
        "confidence": "high",
        "proof_strategy_hint": "This is the triangle identity for the adjoint equivalence. It states that composing the unit and counit in a specific way yields the identity. For a Boolean algebra B, we start with Spec(B), apply the unit to get a map B → Clop(Spec(B)), then apply the spectrum functor to get Spec(Clop(Spec(B))), then apply the counit to get back to Spec(B). The axiom states this composition is the identity. To prove: Trace through the definitions of the unit and counit. The unit sends b ∈ B to {P ∈ Spec(B) | b ∈ P}. Applying Spec gives a map from Spec(Clop(Spec(B))) to Spec(B). The counit identifies Spec(Clop(Spec(B))) with Spec(B). The composition should be definitionally the identity by the way the maps are constructed. This is a coherence condition that follows from the explicit constructions in the proof of Stone duality."
    }
]

# Check if axioms already exist and update, or add new ones
existing_names = {ax['name'] for ax in data['axioms']}
for new_ax in new_axioms:
    if new_ax['name'] not in existing_names:
        data['axioms'].append(new_ax)
        print(f"Added axiom: {new_ax['name']}")
    else:
        print(f"Axiom already exists: {new_ax['name']}")

# Save back to file
with open('axioms_to_prove.json', 'w') as f:
    json.dump(data, f, indent=2)

print("\nAxioms file updated successfully!")
