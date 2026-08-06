# LiteLightGBM extraction-only refactor plan

## Status

**Complete.** The shared core, binning, and tree definitions have been extracted
into the three implementation modules under `src/lite_lightgbm_dep/` described below. `src.lite_lightgbm`
remains the stable public façade and preserves the documented re-exports and
estimator behavior. The Stage 1 regression foundation is persistent in
`tests/test_lite_lightgbm_refactor.py` and is run with the unittest discovery
command specified below.

## Purpose

`src/lite_lightgbm.py` currently contains 2,691 lines covering shared numerical
definitions, binning, histogram trees, estimator validation, training, and prediction.
OPT3 through OPT7 will make the histogram and tree sections larger. This refactor splits
the implementation into cohesive files while keeping `src.lite_lightgbm` as the only
supported public import surface.

This is an **extraction-only refactor**. Its success criterion is unchanged behavior,
not cleaner algorithms or faster training. Implement it after OPT2 and before OPT3.

An implementation agent should read these files completely before editing:

1. `src/lite_lightgbm.md` — algorithm and behavioral contract;
2. `src/lite_lightgbm_docs.md` — public API and usage reference;
3. this file — file ownership, move order, and verification requirements; and
4. `src/lite_lightgbm.py` — source being extracted.

## Non-goals

Do not combine any of the following work with this refactor:

- OPT3 feature pre-filtering or local histogram layouts;
- OPT4 validated internal hot paths;
- OPT5 vectorized flattened histogram aggregation;
- OPT6 histogram subtraction;
- OPT7 bounded histogram caching;
- validation deduplication;
- changes to exception types or messages;
- changes to floating-point accumulation order;
- changes to RNG calls, sampling, tie-breaking, or tree growth;
- new estimator parameters or learned attributes;
- API cleanup, renaming, or removal;
- formatting or rewriting unrelated code; or
- adding LightGBM or scikit-learn imports.

Move existing definitions with the smallest possible textual edits. If a function is
awkward or repetitive, leave it that way until the optimization specifically responsible
for it is implemented.

## Target layout

The completed refactor has one public façade plus three implementation modules (and a
package marker):

```text
src/
├── lite_lightgbm.py             public façade and LiteLightGBM estimator
└── lite_lightgbm_dep/
    ├── __init__.py              implementation package marker
    ├── core.py                  shared configuration and numerical definitions
    ├── binning.py               bin mapper and sparse bin transformation
    └── tree.py                  histograms, splits, trees, and tree traversal
```

The dependency package keeps the three extracted modules separate from the façade. Project code,
notebooks, reports, and users must continue importing from `src.lite_lightgbm`.

Do not replace `lite_lightgbm.py` with a directory package. Keeping the existing public
file avoids changing imports, documentation links, and the location expected by the
project.

## Dependency graph

Dependencies must remain one-way. In this diagram, `A -> B` means that `B` is allowed to
import from `A`:

```text
lite_lightgbm_dep.core -> lite_lightgbm_dep.binning -> lite_lightgbm_dep.tree
          |                                                  |
          +--------------------------------------------------+

lite_lightgbm_dep.core ---+
lite_lightgbm_dep.binning +---> lite_lightgbm
lite_lightgbm_dep.tree ----+
```

More precisely:

- `lite_lightgbm_dep/core.py` imports no LiteLightGBM sibling module.
- `lite_lightgbm_dep/binning.py` may import only from `lite_lightgbm_dep/core.py`.
- `lite_lightgbm_dep/tree.py` may import from `lite_lightgbm_dep/core.py` and
  `lite_lightgbm_dep/binning.py`.
- `lite_lightgbm.py` imports and re-exports definitions from all three dependency modules.
- No dependency module may import from `lite_lightgbm.py`.

Violating the final rule creates a circular import because the façade must import every
dependency definition before it can define the estimator.

Use package-relative imports:

```python
from .lite_lightgbm_dep.core import ...
```

The supported application import remains:

```python
from src.lite_lightgbm import LiteLightGBM
```

Running `src/lite_lightgbm.py` directly as a script and importing it as an unrelated
top-level module after manually placing `src/` on `sys.path` are not supported usage.

## Exact symbol ownership

### `src/lite_lightgbm_dep/core.py`

Move these definitions here without changing their behavior:

- `Matrix`;
- `ClassWeight`;
- `EPSILON`;
- `LiteLightGBMConfig`;
- `sigmoid`;
- `soft_threshold`; and
- `binary_gradients_hessians`.

Expected imports:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
```

`Matrix` needs the SciPy import. `LiteLightGBMConfig` needs `ClassWeight`. The numerical
helpers need NumPy. This module must not import estimator, binning, or tree definitions.

### `src/lite_lightgbm_dep/binning.py`

Move these definitions here:

- `BinMapper`;
- `BinnedDataset`;
- `_find_bin_boundaries`;
- `fit_bin_mapper`;
- `_encoded_bin_dtype`; and
- `transform_bins`.

Expected imports include:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .core import Matrix, LiteLightGBMConfig
```

Keep all mapper validation, sparse canonicalization, OPT1 boundary selection, OPT2
compact dtype selection, encoding, and dtype checks inside this file. Do not move
binning validation into core.

`BinnedDataset.shape` remains unchanged. Both `BinMapper` and `BinnedDataset` must be
defined only once; the façade re-exports the same class objects rather than defining
duplicates.

### `src/lite_lightgbm_dep/tree.py`

Move these definitions here:

- `Histogram`;
- `SplitInfo`;
- `TreeNode`;
- `DecisionTree`;
- `build_histogram`;
- `find_best_split`;
- `partition_rows`;
- `fit_tree`; and
- `predict_tree_raw`.

Also move the nested validation helpers with their owning functions. They are local
implementation details, not new module-level helpers.

Expected imports include:

```python
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from .binning import BinnedDataset, BinMapper
from .core import EPSILON, LiteLightGBMConfig, soft_threshold
```

The SciPy import is required because `partition_rows` and `predict_tree_raw` validate the
CSR/CSC storage types held by `BinnedDataset`. Add other imports only when the moved code
actually needs them.

Keep direct child histogram construction unchanged. OPT6, not this refactor, introduces
histogram subtraction.

### `src/lite_lightgbm.py`

Keep these responsibilities in the public façade:

- the short module docstring;
- `LiteLightGBM`;
- estimator constructor parameters;
- `get_params` and `set_params`;
- `__sklearn_tags__`;
- estimator-level validation;
- class/sample weighting;
- RNG setup and row/feature sampling;
- the boosting loop;
- fitted-state publication; and
- public prediction methods.

Expected direct imports include:

```python
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy.sparse as sp
```

It also imports the symbols it uses or re-exports from the three dependency modules.

## Stable façade and re-exports

Every documented name currently reachable from `src.lite_lightgbm` must remain reachable
after extraction. Import these names into the façade at module scope:

```text
Matrix
ClassWeight
EPSILON
LiteLightGBMConfig
BinMapper
BinnedDataset
Histogram
SplitInfo
TreeNode
DecisionTree
sigmoid
soft_threshold
binary_gradients_hessians
fit_bin_mapper
transform_bins
build_histogram
find_best_split
partition_rows
fit_tree
predict_tree_raw
```

Also import `_find_bin_boundaries` and `_encoded_bin_dtype` into the façade. They remain
private helpers, while the façade continues to re-export them for focused checks and
optimization work.

Do not create wrapper functions for re-exported helpers. A wrapper changes introspection,
tracebacks, identity, and sometimes pickling. Import the original object directly:

```python
from .lite_lightgbm_dep.binning import fit_bin_mapper
```

Do not add `__all__` during this refactor. The monolithic module has no `__all__`, so
adding one would change wildcard-import behavior and would be unrelated API cleanup.

## Dataclass identity and persistence

Never leave a copied dataclass definition in `lite_lightgbm.py`. For example, this must
be true:

```python
from src import lite_lightgbm
from src.lite_lightgbm_dep import binning

assert lite_lightgbm.BinMapper is binning.BinMapper
```

The same requirement applies to every moved class. Duplicate class definitions can look
identical while breaking `isinstance`, equality expectations, and deserialization.

Moving a class changes the module recorded by new pickle files. This is acceptable for
the private representation as long as all extracted files ship with the project. Keep
the façade aliases so older pickles that refer to names such as
`src.lite_lightgbm.BinMapper` can still resolve them. Do not mutate class `__module__`
attributes to disguise their new location.

Before refactoring, create a small estimator pickle with the standard-library `pickle`
module. After refactoring, confirm that it loads and reproduces predictions. Also test a
new post-refactor pickle round trip. Persistence must not import joblib, LightGBM, or
scikit-learn inside the implementation.

## Required implementation sequence

Do not move all definitions in one edit. Use these stages so failures have a small search
area.

### Stage 1: establish regression tests before moving code

Create `tests/test_lite_lightgbm_refactor.py` using `unittest`, NumPy, SciPy, and the
standard library only. If the repository adopts another test convention before this
refactor starts, follow that convention without importing scikit-learn.

The tests must run successfully against the current monolithic file before extraction:

```powershell
uv run python -m unittest discover -s tests -p "test_lite_lightgbm*.py"
```

At minimum, freeze these behaviors:

- constructor parameter names, defaults, and `get_params` output;
- dense/CSR/CSC mapper parity;
- OPT1 deterministic cut points and tie behavior;
- OPT2 dtype thresholds and encoded maximum values;
- dense/CSR/CSC transformed storage parity;
- malformed stored-zero rejection;
- histogram values for a hand-checkable leaf;
- deterministic split tie-breaking;
- row partition order;
- one complete seeded tree;
- a small seeded multi-tree estimator;
- raw scores, probabilities, labels, and feature importances;
- empty prediction batches;
- calling `set_params` after fit does not change existing predictions; and
- import succeeds while `sklearn` and `lightgbm` imports are blocked.

For tree equality, compare every `TreeNode` field, node order, and
`DecisionTree.feature_indices`. Use exact equality where the extraction does not change
calculation order. The refactor must not require widening tolerances.

### Stage 2: extract core definitions

1. Create `lite_lightgbm_dep/core.py` with a short module docstring.
2. Move the exact core symbols listed above.
3. Import and re-export them from `lite_lightgbm.py`.
4. Remove the original definitions from the façade.
5. Remove only imports made unused by this stage.
6. Run the complete refactor test file.

Verify class identity and function identity between the façade and dependency core module.

### Stage 3: extract binning

1. Create `lite_lightgbm_dep/binning.py`.
2. Move the two dataclasses and four binning functions as one coherent block.
3. Import their core dependencies through relative imports.
4. Import and re-export the moved objects from the façade.
5. Remove their original definitions from the façade.
6. Run the complete tests, including every OPT1 and OPT2 boundary case.

Do not rewrite validation or change temporary dtypes while moving the code.

### Stage 4: extract trees

1. Create `lite_lightgbm_dep/tree.py`.
2. Move the four tree dataclasses and five tree functions.
3. Keep nested helpers inside the same owning functions.
4. Import only core and binning dependencies.
5. Re-export the moved objects from the façade.
6. Remove their original definitions from the façade.
7. Run the complete tests.

Pay particular attention to `Any`, `heapq`, `field`, `EPSILON`, `soft_threshold`,
`BinMapper`, and `BinnedDataset`; these are easy imports to omit during extraction.

### Stage 5: clean the façade

After all moves pass:

1. remove imports no longer used by the estimator or re-export surface;
2. keep the public module docstring short and point to the documentation and contract;
3. confirm that `LiteLightGBM.__module__` remains `src.lite_lightgbm`;
4. confirm all documented imports still resolve;
5. do not introduce forwarding wrappers or an `__all__`; and
6. update the refactor status in the three Markdown files from planned to complete.

### Stage 6: persistence and project smoke checks

Run both old-pickle loading and new-pickle round-trip tests. Then load the supplied
20,000 by 5,000 sparse representation and fit a deliberately small model, such as one
or two shallow trees. This is an import/integration smoke test, not a full performance
benchmark.

The refactor must be complete before starting OPT3.

## Verification matrix

| Area | Required check | Equality requirement |
|---|---|---|
| Imports | All documented names import from `src.lite_lightgbm` | Exact availability |
| Dependencies | Block `sklearn` and `lightgbm`, then import | Must succeed |
| Signatures | Compare `inspect.signature` before and after | Exact |
| Configuration | Defaults and `get_params` keys/values | Exact |
| Binning | Dense/CSR/CSC mapper metadata | Exact |
| OPT2 encoding | Data dtype and decoded sparse coordinates | Exact |
| Histograms | Gradients, Hessians, counts | Exact |
| Splits | Gain, feature, threshold, child statistics | Exact |
| Trees | Node list, fields, sampled features | Exact |
| Estimator | Seeded fitted attributes and predictions | Exact |
| Sampling | Repeated seeded models | Exact |
| Sparse behavior | No conversion of project sparse input to dense | Structural check |
| Persistence | Old load and new round trip | Exact predictions |
| Formatting | `git diff --check` | No errors |

Because this refactor only moves definitions, a changed floating-point result is a
failure. Do not explain it away as harmless numerical drift.

## Public usage after refactoring

User code does not change:

```python
from src.lite_lightgbm import LiteLightGBM

model = LiteLightGBM(random_state=42)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
```

Documented development helpers also remain available from the façade:

```python
from src.lite_lightgbm import fit_bin_mapper, fit_tree
```

Do not teach users to import from `lite_lightgbm_dep.core`,
`lite_lightgbm_dep.binning`, or `lite_lightgbm_dep.tree`. Those implementation locations
are free to change during later optimizations.

## Ownership of later optimizations

The split is designed so later work has an obvious home:

| Work | Primary file after refactor |
|---|---|
| OPT3 active-feature calculation | `lite_lightgbm_dep/binning.py` and estimator façade |
| OPT3 histogram layout | `lite_lightgbm_dep/tree.py` |
| OPT4 trusted tree kernels | `lite_lightgbm_dep/tree.py` |
| OPT5 flattened histogram aggregation | `lite_lightgbm_dep/tree.py` |
| OPT6 histogram subtraction | `lite_lightgbm_dep/tree.py` |
| OPT7 histogram cache | `lite_lightgbm_dep/tree.py` |
| Boosting or estimator API work | `lite_lightgbm.py` |

If `lite_lightgbm_dep/tree.py` later becomes difficult to navigate after OPT3-OPT7, profile
and review it before considering a separate histogram module. Do not create that fifth
file speculatively during this extraction.

## Common failure modes

- **Circular import:** a dependency module imports the façade. Fix dependency direction;
  do not hide the cycle with local imports.
- **Duplicate dataclass:** a definition remains in both old and new files. Remove the
  façade copy and re-export the dependency object.
- **Missing façade name:** internal tests pass but an old import fails. Check every name
  in the stable re-export list.
- **Changed exception:** validation was rewritten while moving it. Restore the original
  code and defer cleanup to OPT4.
- **Changed tree:** imports or code order were altered beyond extraction. Compare the
  first differing node and restore the original calculation order.
- **Unsigned underflow:** OPT2 consumers subtract before widening. Preserve the existing
  signed conversion exactly.
- **Broken pickle:** old class names are no longer present in the façade. Restore the
  re-export alias.
- **Hidden scikit-learn dependency:** a convenience base class or validation helper was
  introduced. Remove it and keep local validation.
- **Mixed optimization:** local layouts, subtraction, caching, or fast paths appear in
  the extraction diff. Revert those portions and implement them under their own plans.

## Completion criteria

The refactor is complete only when:

- all four target files exist and follow the dependency graph;
- `lite_lightgbm.py` remains the stable import façade and owns `LiteLightGBM`;
- every listed name remains importable from `src.lite_lightgbm`;
- no moved class or function is defined twice;
- the pre-refactor regression tests pass without relaxed assertions;
- dense and sparse seeded models remain exactly identical to the pre-refactor oracle;
- old and new pickle checks pass;
- imports succeed with LightGBM and scikit-learn blocked;
- no project sparse matrix is densified;
- the implementation still uses only NumPy, SciPy, and the standard library;
- `git diff --check` reports no errors; and
- documentation status is changed from planned to complete only after all gates pass.
