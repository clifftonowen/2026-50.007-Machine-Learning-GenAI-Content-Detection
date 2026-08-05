"""Stage 7 - the short-document bottleneck.

The baseline's five length bands do not fail evenly:

    band 0 (shortest)  0.6266
    band 1             0.7915
    band 2             0.8766
    band 3             0.9073
    band 4 (longest)   0.8364

The average everyone quotes, 0.8077, is dragged down by one band. Lifting band 0 from
0.63 to 0.80 would move the mean to about 0.84 on its own - more headroom than any
model swap has offered so far.

Two plausible causes, and they call for different fixes:

1. The style features need length to mean anything. MATTR uses a 100-token window,
   Yule's K and the burstiness moments need enough sentences to have a distribution.
   On a 200-character document these are mostly noise, and noise the model has learned
   to trust on longer documents.
2. The model simply spends its capacity where the data is, and short documents are a
   minority of the training rows.

Cause 1 predicts that dropping the length-fragile blocks helps band 0. Cause 2 predicts
that upweighting short training rows helps band 0. Both are five fits apiece, so we can
just ask.
"""

import numpy as np

from experiments.common import *


def band_report(env):
    """Class balance and length range per band, to rule out the boring explanation."""
    lens = np.array([len(t) for t in env["dev_texts"]])
    labels = clustering.length_groups(env["dev_texts"], n_groups=5)
    print("band   n     chars            machine share   baseline F1")
    for b in np.unique(labels):
        m = labels == b
        print(f"  {b}  {m.sum():5d}  {lens[m].min():6d}-{lens[m].max():7d}   "
              f"{env['y_dev'][m].mean():.3f}          {BASE5[b]:.4f}", flush=True)


def run(env):
    X_dev, y_dev, folds5 = env["X_dev"], env["y_dev"], env["folds5"]
    results = {}

    log_stage("stage 7: why band 0 fails")
    band_report(env)

    # --- fix 1: drop the blocks that need length to be meaningful ------------
    # F_diversity is MATTR/Yule's K/hapax; E_length carries the burstiness moments.
    # Both are estimated from too little text in band 0. The ablation in notebook 14
    # found F to be the single most valuable block overall - which is consistent with
    # it being excellent on long documents and misleading on short ones.
    log_stage("stage 7a: drop length-fragile feature blocks")
    KEEP_SETS = {
        "no_F": ["A_function_words", "B_punctuation", "C_casing", "D_structure",
                 "E_length", "H_char_ngrams", "I_word_ngrams"],
        "no_EF": ["A_function_words", "B_punctuation", "C_casing", "D_structure",
                  "H_char_ngrams", "I_word_ngrams"],
    }
    for tag, blocks in KEEP_SETS.items():
        Xb = tf.stack(env["built"], blocks)[0][env["dev_idx"]]
        rec = night_trial(f"blocks_{tag}", {"blocks": tag}, build_lgbm({}), X=Xb)
        report(rec, f"blocks {tag} ({Xb.shape[1]} cols)")
        print(f"    per band: {np.round(rec['scores'], 4).tolist()}", flush=True)
        results[tag] = rec

    # --- fix 2: pay more attention to short documents ------------------------
    log_stage("stage 7b: upweight short training rows")
    lens = np.array([len(t) for t in env["dev_texts"]], dtype=float)
    rank = np.argsort(np.argsort(lens)) / len(lens)      # 0 = shortest, 1 = longest
    for power in (1.0, 2.0):
        w = (1.0 + (1.0 - rank)) ** power
        w = w / w.mean()

        def fold_scores(folds, w=w):
            out = []
            for tr, te in folds:
                m = build_lgbm({}).fit(X_dev[tr], y_dev[tr], sample_weight=w[tr])
                out.append(evaluation.macro_f1(y_dev[te], m.predict(X_dev[te])))
            return out

        rec = manual_trial(f"shortweight{power:g}", {"power": power}, fold_scores)
        report(rec, f"short-document upweighting (power {power:g})")
        print(f"    per band: {np.round(rec['scores'], 4).tolist()}", flush=True)
        results[f"shortweight{power:g}"] = rec

    log_stage("stage 7 summary - band 0 is what matters here")
    print(f"  {'baseline':28s} band0 {BASE5[0]:.4f}  mean {BASE5.mean():.4f}")
    for name, rec in results.items():
        s = np.array(rec["scores"])
        print(f"  {name:28s} band0 {s[0]:.4f}  mean {rec['mean']:.4f}  "
              f"paired {rec['paired_mean']:+.4f}", flush=True)
    return results


if __name__ == "__main__":
    run(ENV)
