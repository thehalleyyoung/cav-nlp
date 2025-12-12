import Mathlib.Topology.Category.Stonean.Basic

#check Stonean
#print Stonean

open CategoryTheory

-- Check that Stonean has what we need
example (X : Stonean) : TopologicalSpace X := inferInstance
example (X : Stonean) : CompactSpace X := inferInstance
example (X : Stonean) : T2Space X := inferInstance
example (X : Stonean) : TotallyDisconnectedSpace X := inferInstance

-- Check the category structure
#check (inferInstance : Category Stonean)
