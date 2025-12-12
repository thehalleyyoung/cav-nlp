import Lake
open Lake DSL

package «foundations_semiosis» where
  -- Settings for the package
  precompileModules := true

-- Define the main library that includes chapter modules
lean_lib «Chapters» where
  roots := #[`Chapter01, `Chapter02, `Chapter03, `Chapter04, `Chapter05,
             `Chapter06, `Chapter07, `Chapter08, `Chapter09, `Chapter10,
             `Chapter11, `Chapter12, `Chapter13, `Chapter14, `Chapter15,
             `Chapter16, `Chapter17, `Chapter18, `Chapter19, `Chapter20,
             `Chapter21, `Chapter22, `Chapter23, `Chapter24, `Chapter25,
             `Chapter26, `Chapter27, `Chapter28, `Chapter29, `Chapter30]
  globs := #[.submodules `Chapters]

@[default_target]
lean_exe «foundations_semiosis» where
  root := `Main
  supportInterpreter := true

-- Require Mathlib from git
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
