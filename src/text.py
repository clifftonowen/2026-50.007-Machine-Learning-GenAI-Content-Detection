"""Raw-text loading and the id-group split, for the round-5 feature work.

`src/data.py` loads the supplied TF-IDF matrices and deliberately never touches the
raw text. This module is the counterpart: it loads `train.csv` / `test.csv`, which
hold genuine natural text with original casing, punctuation, newlines and markdown.
Nothing in the project read those files before round 5 except one length `describe()`
in 01_eda section 3.

Two things beyond loading live here because both are properties of the *ids* rather
than of any model, and both are needed by more than one notebook:

**The id-group split.** The 6,999 test rows are 1,999 UUID-style ids and 5,000
numeric-style ids, and they are not the same population. Measured on the raw text, the
numeric group has median 1,862 characters against the UUID group's 1,189, markdown
`**` at 2.52 occurrences per document against 0.44, and the word "reviewer" in 15.8%
of documents against 0.2%. The COLING paper explains why: the shared task's test split
is drawn from CUDRT, IELTS, NLPeer, PeerSum and MixSet, none of which appear in
training, and NLPeer and PeerSum are academic peer reviews. The UUID rows match the
training distribution on every statistic checked. Treating the two groups as one
population, in particular giving them a single predicted class balance, throws that
away.

**The train/test duplicate overlap.** Ten texts appear verbatim in both files. Their
labels are therefore known for free on the test side. They are matched on exact text,
never on id, because the id sets are disjoint by construction.
"""

import numpy as np
import pandas as pd

from . import paths


def _read(path, columns):
    """Read one raw CSV, forcing `id` to string.

    The `id` column mixes UUID strings with digit-only strings. Letting pandas infer
    the dtype coerces the digit-only ids to integers, which silently reorders and
    reformats them and misaligns every downstream submission.
    """
    df = pd.read_csv(path, dtype={"id": str})
    missing = [c for c in columns if c not in df.columns]
    assert not missing, f"{path.name} is missing {missing}; found {list(df.columns)}"
    assert df[columns].notna().all().all(), f"{path.name} has nulls in {columns}"
    return df


def load_train_text():
    """Load the labelled training text.

    Returns
    -------
    ids : ndarray of str, shape (20000,)
    texts : ndarray of object, shape (20000,)
        Raw text, uncleaned. Casing, punctuation and newlines are intact.
    y : ndarray of int64, shape (20000,)
        1 = machine-generated, 0 = human-authored.
    """
    df = _read(paths.TRAIN_CSV, ["id", "text", "label"])
    return (df["id"].to_numpy(), df["text"].to_numpy(),
            df["label"].to_numpy(dtype=np.int64))


def load_test_text():
    """Load the unlabelled holdout text.

    Returns
    -------
    ids : ndarray of str, shape (6999,)
    texts : ndarray of object, shape (6999,)
    """
    df = _read(paths.TEST_CSV, ["id", "text"])
    return df["id"].to_numpy(), df["text"].to_numpy()


def id_group(ids):
    """Label each id as "numeric" or "uuid".

    The split is exact rather than heuristic: an id is either entirely digits or it is
    not. Train is 100% uuid; test is 5,000 numeric and 1,999 uuid.

    Parameters
    ----------
    ids : array-like of str

    Returns
    -------
    ndarray of str, shape (n,)
        Values "numeric" or "uuid".
    """
    ids = np.asarray(ids, dtype=object)
    numeric = np.array([str(i).isdigit() for i in ids])
    return np.where(numeric, "numeric", "uuid")


def group_masks(ids):
    """Boolean masks for the two id groups, for per-group thresholding.

    Parameters
    ----------
    ids : array-like of str

    Returns
    -------
    dict
        {"uuid": ndarray of bool, "numeric": ndarray of bool}. The two are disjoint
        and cover every row.
    """
    groups = id_group(ids)
    masks = {"uuid": groups == "uuid", "numeric": groups == "numeric"}
    assert masks["uuid"].sum() + masks["numeric"].sum() == len(groups)
    return masks


def find_train_test_duplicates(train_ids, train_texts, train_y, test_ids, test_texts):
    """Test rows whose text appears verbatim in train, so their label is known.

    Matched on exact text rather than on id: the id sets are disjoint, so an id join
    finds nothing. Whitespace is not normalised, because two documents that differ
    only in whitespace are not the same document for these features.

    A duplicated train text carrying two different labels would make its test-side
    label ambiguous, so those are dropped rather than guessed at. As measured on the
    current data there are ten matches, all label 0, five in each id group.

    Parameters
    ----------
    train_ids, train_texts : array-like, shape (n_train,)
    train_y : array-like of int, shape (n_train,)
    test_ids, test_texts : array-like, shape (n_test,)

    Returns
    -------
    pd.DataFrame
        Columns: test_id, train_id, label, group. Empty (with those columns) if no
        unambiguous match exists.
    """
    train = pd.DataFrame({"id": np.asarray(train_ids), "text": np.asarray(train_texts),
                          "label": np.asarray(train_y)})

    # Keep only texts whose label is unambiguous across all their train occurrences.
    per_text = train.groupby("text")["label"].agg(["nunique", "first"])
    unambiguous = per_text[per_text["nunique"] == 1]["first"]
    first_id = train.drop_duplicates("text").set_index("text")["id"]

    test = pd.DataFrame({"id": np.asarray(test_ids), "text": np.asarray(test_texts)})
    hit = test[test["text"].isin(unambiguous.index)].copy()

    cols = ["test_id", "train_id", "label", "group"]
    if hit.empty:
        return pd.DataFrame(columns=cols)

    hit["train_id"] = hit["text"].map(first_id)
    hit["label"] = hit["text"].map(unambiguous).astype(int)
    hit["group"] = id_group(hit["id"].to_numpy())
    return (hit.rename(columns={"id": "test_id"})[cols]
            .sort_values("test_id").reset_index(drop=True))


def text_summary(texts, y=None):
    """Cheap per-group text statistics, for sanity-checking a feature build.

    Deliberately duplicates a handful of the numbers that `text_features` computes, by
    a different and much simpler route. If the two disagree, the feature builder is
    wrong. Used by 12_text_features section 2.

    Parameters
    ----------
    texts : array-like of str
    y : array-like of int, optional
        If given, the table is split by label instead of pooled.

    Returns
    -------
    pd.DataFrame
        Columns: n, char_mean, char_median, word_mean, newlines_mean, bold_mean,
        upper_frac, punct_frac. One row per group.
    """
    texts = np.asarray(texts, dtype=object)

    def stats(subset):
        chars = np.array([len(t) for t in subset], dtype=float)
        words = np.array([len(t.split()) for t in subset], dtype=float)
        return {
            "n": len(subset),
            "char_mean": chars.mean(),
            "char_median": float(np.median(chars)),
            "word_mean": words.mean(),
            "newlines_mean": np.mean([t.count("\n") for t in subset]),
            "bold_mean": np.mean([t.count("**") for t in subset]),
            "upper_frac": np.mean([sum(c.isupper() for c in t) / max(len(t), 1)
                                   for t in subset]),
            "punct_frac": np.mean([sum(not c.isalnum() and not c.isspace() for c in t)
                                   / max(len(t), 1) for t in subset]),
        }

    if y is None:
        return pd.DataFrame([{"group": "all", **stats(texts)}]).set_index("group")

    y = np.asarray(y)
    rows = [{"group": f"label={lab}", **stats(texts[y == lab])}
            for lab in np.unique(y)]
    return pd.DataFrame(rows).set_index("group")
