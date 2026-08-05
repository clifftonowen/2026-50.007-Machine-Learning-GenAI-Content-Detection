"""Stage 3 - is 'test-likeness' a better transfer proxy than document length?

The repo grades every candidate on length-banded folds. That measures generalisation
across length, and the actual gap on this task is that train and test come from
disjoint corpora. Length is a proxy for domain, and nobody has checked how good a
proxy it is.

This stage builds folds banded by a train-vs-test discriminator's score instead, and
reports the same LightGBM baseline under both. Two things are worth knowing:

- If the shift bands drop the score further than the length bands do, they are
  catching domain structure the length bands miss, and any candidate aimed at the
  domain gap should be graded on them too.
- If the two agree, the length protocol was already doing the job and the extra
  machinery can be dropped - which is also a useful answer, and a cheap one.

Diagnostic only. Nothing is selected on these folds; the ledger and the 0.0084 noise
floor are calibrated against the length protocol.
"""

import numpy as np

from experiments.common import *


def run(env):
    y_dev = env["y_dev"]

    log_stage("stage 3: shift-banded folds vs length-banded folds")
    s = shift_scores()[env["dev_idx"]]
    folds_shift, labels = shift_folds(n_bands=3)
    print(f"{len(folds_shift)} shift bands from discriminator scores "
          f"[{s.min():.2f}, {s.max():.2f}]", flush=True)
    for b in np.unique(labels):
        m = labels == b
        print(f"  band {b}: {m.sum():5d} rows, machine share {y_dev[m].mean():.3f}, "
              f"mean test-likeness {s[m].mean():+.3f}", flush=True)

    def fold_scores(_folds):
        out = []
        for i, (tr, te) in enumerate(folds_shift):
            m = build_lgbm({}).fit(env["X_dev"][tr], y_dev[tr])
            out.append(evaluation.macro_f1(y_dev[te], m.predict(env["X_dev"][te])))
            print(f"  band {i} held out: {out[-1]:.4f}", flush=True)
        return out

    rec = manual_trial("shiftband_baseline", {"bands": 3, "model": "lgbm_defaults"},
                       fold_scores, protocol="shift")
    shift_mean = rec["mean"]

    print(f"\nLightGBM defaults, same model, three ways of grouping the folds:")
    print(f"  length 3-band  {BASE3.mean():.4f}   (the established protocol)")
    print(f"  length 5-band  {BASE5.mean():.4f}")
    print(f"  shift 3-band   {shift_mean:.4f}   (banded by test-likeness)")
    delta = shift_mean - BASE3.mean()
    print(f"\nshift minus length: {delta:+.4f}")
    if delta < -NOISE_FLOOR:
        print("Shift bands are harder than length bands by more than the noise floor - "
              "they catch domain structure length misses. Grade domain-adaptation "
              "candidates on both.")
    elif abs(delta) <= NOISE_FLOOR:
        print("The two protocols agree within the noise floor. Length bands were "
              "already an adequate proxy; no reason to add a third protocol.")
    else:
        print("Shift bands are EASIER than length bands, so they are not a stricter "
              "transfer test. Length stays the protocol of record.")
    return rec


if __name__ == "__main__":
    run(ENV)
