import Mathlib.Topology.Category.Stonean.Basic
import Mathlib.Order.Category.BoolAlg
import Mathlib.CategoryTheory.Equivalence

-- Stonean is exactly Stone spaces in Mathlib
#check Stonean
#print Stonean

-- Check the structure
example : ∀ (X : Stonean), CompactSpace X.toCompHaus := by intro X; infer_instance
