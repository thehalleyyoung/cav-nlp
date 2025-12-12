import Mathlib.CategoryTheory.Category.Basic
import Mathlib.CategoryTheory.Yoneda
import Mathlib.Order.CompleteLattice
import Mathlib.Data.Set.Lattice

-- Check what structures are available
variable (C : Type u) [Category.{v} C]

-- Set α has a complete lattice structure for any α
example (α : Type*) : CompleteLattice (Set α) := inferInstance

-- The Yoneda embedding goes to Type v
-- Hom(X, A) is a Type v
#check @CategoryTheory.Functor.yoneda C _ 

-- Can we use Type v as our lattice? No, Type doesn't have a lattice structure.
-- But we can use Set (Type v) = Type v → Prop

