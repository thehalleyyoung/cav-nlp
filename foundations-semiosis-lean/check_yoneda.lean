import Mathlib.CategoryTheory.Yoneda
import Chapter01

open CategoryTheory

-- Check what's available
#check @yoneda
#check @yoneda_fullyFaithful  
#check @Functor.preimageIso
#check @yonedaEquiv

-- Try to use them
variable (C : Type*) [Category C]

example : yoneda.FullyFaithful := yoneda_fullyFaithful

example (A B : C) (iso : yoneda.obj A ≅ yoneda.obj B) : A ≅ B :=
  Functor.preimageIso yoneda iso
