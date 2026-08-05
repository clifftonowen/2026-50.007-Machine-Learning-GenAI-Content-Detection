"""Finish the night: the ensemble, the pseudo-label artifact check, and submissions.

Separate from run_night.py because the first pass died in stage 5 when XGBoost asked
for 12.4 GB, and everything before it is cached and should not be recomputed.
"""

import time
import traceback

from experiments.common import *


def guard(name, fn, *args, **kwargs):
    t = time.time()
    try:
        out = fn(*args, **kwargs)
        print(f"\n### {name} finished in {(time.time() - t) / 60:.0f} min", flush=True)
        return out
    except Exception:
        print(f"\n### {name} FAILED after {(time.time() - t) / 60:.0f} min", flush=True)
        traceback.print_exc()
        return None


def main():
    from experiments import stage5_stack, stage6_submit, stage11_pseudomore

    stack = guard("stage 5 (ensemble)", stage5_stack.run, ENV)
    guard("stage 11 (pseudo-label artifact check)", stage11_pseudomore.run, ENV)

    log_stage("writing submissions")
    guard("submission: lgbm defaults", stage6_submit.submit_lgbm_defaults, ENV)
    guard("submission: seed bag", stage6_submit.submit_seedbag, ENV)

    # Only blend if the ensemble stage actually produced a shortlist. Selection is by
    # lowest optimism gap among combiners that at least match the best single member -
    # round 4's finding was that the gap, not the held-out AUC, predicted transfer.
    if stack is not None and len(stack["shortlist"]):
        row = stack["shortlist"].iloc[0]
        guard("submission: best ensemble", stage6_submit.submit_ensemble,
              ENV, row["lane"], row["config"], stack["members"])
    else:
        print("No combiner matched the best single member - nothing to submit from "
              "the ensemble lane.", flush=True)

    log_stage("done")


if __name__ == "__main__":
    main()
