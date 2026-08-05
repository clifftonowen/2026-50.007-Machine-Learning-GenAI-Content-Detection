"""Overnight driver - runs the remaining stages in one process.

Order is by expected value on the domain gap, not by stage number: the transductive
work (stage 4) is the only lever large enough to move 0.80 toward 0.85, so it gets
compute before the model bake-off (stage 2). Every stage caches its trials, so a crash
or a kill costs only the trial in flight, and rerunning this script resumes.

Each stage is wrapped so one failure cannot take the night down with it.
"""

import time
import traceback

from experiments.common import *

# Ordered by expected value, not by number. Stage 4 is the only lever big enough to
# move 0.80 toward 0.85; stage 5 is the one with a proven track record on this task;
# stage 2 is the widest net but the least likely to pay, so it goes last where a
# time overrun costs the least.
# stage 7 (short-document features) was dropped once the length check showed band 0 is
# only 11% of the test set and test documents skew long - the headroom there is about a
# third of what the equal-weighted band table suggested. stage 10 replaces it: the
# length shift is the one distribution gap we can see and measure.
# stage 2 (more model families) is cut too: tonight's evidence says the model family is
# not what is holding the score back, and it was the most expensive item left.
STAGES = ["stage10", "stage4", "stage5"]


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
    done = {}
    if "stage3" in STAGES:
        from experiments import stage3_shiftcv
        done["stage3"] = guard("stage 3 (shift-band diagnostic)", stage3_shiftcv.run, ENV)
    if "stage8" in STAGES:
        from experiments import stage8_threshold
        done["stage8"] = guard("stage 8 (cutoff vs share thresholding)",
                               stage8_threshold.run, ENV)
    if "stage9" in STAGES:
        from experiments import stage9_bandcal
        done["stage9"] = guard("stage 9 (band 0: hard or unseen?)",
                               stage9_bandcal.run, ENV)
    if "stage7" in STAGES:
        from experiments import stage7_shortdocs
        done["stage7"] = guard("stage 7 (short-document bottleneck)",
                               stage7_shortdocs.run, ENV)
    if "stage10" in STAGES:
        from experiments import stage10_lengthmatch
        done["stage10"] = guard("stage 10 (length-matched weighting)",
                                stage10_lengthmatch.run, ENV)
    if "stage4" in STAGES:
        from experiments import stage4_transductive
        done["stage4"] = guard("stage 4 (transductive)", stage4_transductive.run, ENV)
    if "stage5" in STAGES:
        from experiments import stage5_stack
        done["stage5"] = guard("stage 5 (ensemble)", stage5_stack.run, ENV)
    if "stage2" in STAGES:
        from experiments import stage2_models
        done["stage2"] = guard("stage 2 (model families)", stage2_models.run, ENV)

    log_stage("night complete")
    for name, out in done.items():
        print(f"  {name}: {'ok' if out is not None else 'failed'}", flush=True)
    return done


if __name__ == "__main__":
    main()
