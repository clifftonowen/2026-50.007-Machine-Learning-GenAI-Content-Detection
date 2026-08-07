# LiteLightGBM optimization 2: compact bin storage

## Completion status

Completed. `transform_bins` now stores encoded bin values in the smallest safe
unsigned dtype, while mapper metadata and sparse structural indices remain
unchanged. Sparse structural indices may be signed `int32` or `int64`, depending
on SciPy and matrix shape.

## Objective

Store encoded bins in the smallest safe unsigned integer dtype. The current
`transform_bins` implementation previously stored sparse bin data as `int64`, even though
the default configuration has at most 255 bins per feature. Compact storage
reduces the memory footprint and memory bandwidth of the binned CSR/CSC views.

This is primarily a memory optimization. The measured transform time is already
small, so do not claim a large training-time improvement from this step alone.
The implementation must not use scikit-learn.

## Encoding invariant

Sparse storage intentionally uses:

```text
encoded_value = bin_id + 1
```

SciPy's implicit zero therefore means "use this feature's default bin," while a
stored value is always positive. If a feature has `n_bins` bins, the largest
encoded value is `n_bins`, not `n_bins - 1`. Dtype selection must use that
encoded maximum.

`BinMapper.default_bins`, `BinMapper.n_bins`, and `BinMapper.bin_offsets` remain
signed `int64`. Sparse matrix indices and indptr arrays are structural indices,
not bin values, are unchanged by this optimization, and may be signed `int32` or
`int64` depending on SciPy and matrix shape.

## Dtype selection

Add one private helper such as:

```python
def _encoded_bin_dtype(n_bins: np.ndarray) -> np.dtype:
    """Return the smallest unsigned dtype for validated bin counts."""
    ...
```

The caller validates and normalizes `mapper.n_bins` to a signed `int64` array
before calling this helper. It is a pure dtype selector: let
`largest_encoded_bin = max(n_bins)` and select:

| Largest encoded bin | Dtype |
|---:|---|
| `<= 255` | `np.uint8` |
| `<= 65_535` | `np.uint16` |
| `<= 4_294_967_295` | `np.uint32` |
| otherwise | `np.uint64` |

Reject negative, zero, non-integral, non-finite, or unrepresentable metadata
through the existing mapper validation before dtype selection. Do not silently
wrap values during conversion.

## Changes to `transform_bins`

Determine the encoded dtype once after mapper validation.

For sparse input:

- `np.searchsorted` may calculate bin IDs using its normal integer result;
- apply the comparison with the `int64` default bin before narrowing;
- cast only the kept `bin_id + 1` values to the selected unsigned dtype;
- create the CSC matrix with that dtype;
- derive the CSR matrix from the CSC matrix; and
- after all `sum_duplicates`, `eliminate_zeros`, `sort_indices`, and `tocsr`
  calls, verify that both `csc.data.dtype` and `csr.data.dtype` are preserved.

For dense input, the temporary bin matrix uses the selected unsigned dtype. When
constructing one-based stored values, first widen the selected entries to signed
`int64`, add one, check the encoded bound, and only then cast to the selected
unsigned dtype. Never add one directly in an unsigned dtype such as `uint8`.

All consumers must treat `csr.data` and `csc.data` as encoded unsigned values.
When subtracting one, first widen to a signed type:

```python
bin_ids = np.asarray(encoded_values, dtype=np.int64) - np.int64(1)
```

Never subtract directly from an unsigned array, because an invalid stored zero
would underflow instead of becoming `-1` and being rejected. Audit at least:

- `build_histogram`;
- `partition_rows`;
- `predict_tree_raw`;
- any validation helper that inspects encoded bin values.

The logical bins, mapper, tree structures, and predictions must remain
unchanged.

## SciPy considerations

CSR and CSC matrices may choose `int32` or `int64` for indices depending on
shape. Tests should assert only the sparse **data** dtype for this optimization.
Do not depend on the index dtype and do not recreate an entire sparse matrix
solely to force its indices smaller.

Calling `sum_duplicates`, `eliminate_zeros`, `sort_indices`, and `tocsr` must not
change the unsigned data dtype. Check both sparse views only after all of these
canonicalization calls have completed.

## Verification

Tests must exercise dtype boundaries directly with hand-built valid mappers:

- 1 and 255 bins use `uint8`;
- 256 and 65,535 bins use `uint16`;
- 65,536 bins use `uint32`;
- an encoded value equal to the dtype maximum round-trips correctly;
- a stored zero is still rejected by histogram, partition, and prediction
  validation paths;
- dense, CSR, and CSC forms yield logically identical binned matrices;
- constant features and an all-default matrix produce empty sparse data arrays
  with the selected dtype.

For ordinary fixtures, compare the previous `int64` representation and the new
representation after widening both sparse data arrays to `int64`. CSR/CSC
`indptr`, indices, shape, and decoded values must match exactly. Fit the same
seeded estimator through both paths and require identical tree metadata and raw
predictions.

## Benchmark

On the supplied 5,000-feature matrix, report:

- `binned.csr.data.dtype` and `binned.csc.data.dtype`;
- `data.nbytes` for each sparse view before and after;
- total sparse storage bytes, including data, indices, and indptr;
- transform time and one-tree time as informational measurements.

With `max_bin=255`, bin data should fall from eight bytes per stored value to
one byte. Total matrix memory will fall by less than 8x because sparse indices
and pointers are unchanged and both CSR and CSC views are retained.

## Acceptance criteria

- The smallest safe unsigned dtype is selected at every boundary.
- No encoded value wraps, underflows, or changes meaning.
- Decoded CSR and CSC matrices are exactly equivalent to the old `int64`
  representation.
- Seeded model structure and predictions remain unchanged.
- Sparse inputs remain sparse, and the module still depends only on NumPy,
  SciPy, and the standard library.
