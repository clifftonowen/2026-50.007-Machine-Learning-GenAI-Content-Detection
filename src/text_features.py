"""Nine families of feature built from raw text, for round 5.

**Why this module exists.** The supplied 5,000-dimension TF-IDF matrix is lemmatized
with stop words removed, so it encodes *content words only*. The COLING paper shows
the test set is drawn from entirely different source corpora than train (CUDRT, IELTS,
NLPeer, PeerSum and MixSet against HC3, M4GT and MAGE), which makes content the part
of the signal least likely to transfer. Function words, punctuation, orthography and
layout are the classic domain-invariant style markers for machine-text detection, and
the course preprocessing removed exactly those. This module puts them back.

The blocks are kept separate rather than concatenated up front so that
`14_ablation.ipynb` can drop one at a time and measure what each is worth. Note that
value is measured under leave-one-cluster-out CV, not the standard 5-fold: standard CV
rewards features that memorise the training domains, which is the criterion that has
produced non-transferring gains in every previous round.

    A  function words   ~318 sklearn stop-word rates, the deleted signal
    B  punctuation      per-1000-character symbol rates
    C  casing           uppercase, ALL-CAPS, Titlecase, internal capitals
    D  structure        newlines, line lengths, bullets, markdown, hard wrapping
    E  length           counts, word and sentence length moments, burstiness
    F  diversity        type-token ratio, hapax rate, Yule's K, MATTR
    G  readability      Flesch reading ease and Flesch-Kincaid grade
    H  char n-grams     char_wb 2-5, TF-IDF
    I  word n-grams     word 1-2 WITH stop words kept, TF-IDF

**Leakage discipline.** Blocks B-G are per-document functions and cannot leak. Block A
uses a *pinned* vocabulary, so it is not fitted at all. Blocks H and I are the only
fitted transforms, and their `build_*` functions take train and test separately and
fit on train alone. Nothing here is ever fitted on the combined corpus.

**No new dependencies.** Everything is numpy, scipy and sklearn. The syllable counter
in block G is hand-rolled rather than pulled from `textstat`; see `_count_syllables`
for what that approximation costs.
"""

import re

import numpy as np
import pandas as pd
from collections import Counter
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from . import paths

RANDOM_STATE = 42

DENSE_BLOCKS = ("B_punctuation", "C_casing", "D_structure", "E_length",
                "F_diversity", "G_readability",
                # Added in round 6. Notebook 14's ablation found F_diversity the single
                # most valuable family for held-out-domain transfer (dropping it cost
                # 0.0510 grouped Macro F1, 1.7x what 20,000 char n-grams cost) off five
                # columns, and E_burstiness / E_sent_len_std ranked second and third on
                # permutation importance. J and K build out those two ideas rather than
                # editing F and E, so every number in ablation_results.csv stays valid
                # and the new measures can be ablated as their own families.
                "J_diversity_ext", "K_variability")
FITTED_BLOCKS = ("H_char_ngrams", "I_word_ngrams")
ALL_BLOCKS = ("A_function_words",) + DENSE_BLOCKS + FITTED_BLOCKS

# Symbols counted by block B. Ordered so the feature names are stable across runs.
_PUNCT = list(".,;:!?'\"()[]{}<>-_/\\|@#$%^&*+=~`")

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENT_RE = re.compile(r"[.!?]+(?:\s|$)")
_VOWEL_RUN_RE = re.compile(r"[aeiouy]+")
_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_HEADER_RE = re.compile(r"^\s*#{1,6}\s+")
# A newline followed by a lowercase letter means the line break is not a sentence or
# paragraph boundary, i.e. the document is hard-wrapped at a fixed column. This is the
# arXiv-abstract signature and it separates human scientific text sharply.
_HARDWRAP_RE = re.compile(r"\n[a-z]")


def _safe_div(a, b):
    """a / b with 0 where b is 0, so an empty document cannot produce nan or inf."""
    return a / b if b else 0.0


def _count_syllables(word):
    """Approximate English syllable count by counting vowel runs.

    Deliberately crude, and the crudeness is the price of not adding `textstat` or a
    pronunciation dictionary as a dependency. It counts maximal runs of `aeiouy` and
    then removes one for a silent trailing "e". It over-counts diaereses ("naive") and
    under-counts some vowel-run-spanning syllables ("fire"), but the error is roughly
    constant across documents, which is all block G needs: Flesch is only ever compared
    between documents here, never reported as an absolute reading level.

    Parameters
    ----------
    word : str

    Returns
    -------
    int
        At least 1 for any non-empty word.
    """
    word = word.lower()
    n = len(_VOWEL_RUN_RE.findall(word))
    if word.endswith("e") and n > 1 and not word.endswith(("le", "ee", "ye", "ae")):
        n -= 1
    return max(n, 1)


def _mattr(tokens, window=100):
    """Moving-average type-token ratio.

    Plain TTR falls monotonically with document length, so on a corpus whose documents
    range from 6 to 2,950 words it measures length far more than it measures
    vocabulary richness. MATTR averages TTR over a fixed-size sliding window and is
    therefore comparable across lengths. Both are returned by block F so the ablation
    can show this rather than the module asserting it.

    Parameters
    ----------
    tokens : list of str
    window : int, default 100

    Returns
    -------
    float
        Plain TTR when the document is shorter than one window.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    return float(np.mean([len(set(tokens[i:i + window])) / window
                          for i in range(n - window + 1)]))


def _yules_k(tokens):
    """Yule's K, a length-robust measure of vocabulary repetitiveness.

    Higher means more repetitive. Unlike TTR it is close to invariant in document
    length by construction, which is why it is worth carrying alongside MATTR.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    freqs = pd.Series(tokens).value_counts().to_numpy()
    spectrum = pd.Series(freqs).value_counts()  # how many types occur i times
    total = float(sum((i ** 2) * v for i, v in spectrum.items()))
    return 1e4 * (total - n) / (n ** 2)


def _mtld_one_direction(tokens, threshold=0.72):
    """MTLD factor count in one direction. Helper for `_mtld`."""
    factors, types, n = 0.0, set(), 0
    for tok in tokens:
        types.add(tok)
        n += 1
        ttr = len(types) / n
        if ttr <= threshold:
            factors += 1.0
            types, n = set(), 0
    if n > 0:
        ttr = len(types) / n
        # Partial trailing factor, scaled by how far it got toward the threshold.
        denom = 1.0 - threshold
        factors += (1.0 - ttr) / denom if denom > 0 else 0.0
    return factors


def _mtld(tokens, threshold=0.72):
    """Measure of Textual Lexical Diversity, forward and backward averaged.

    The standard length-robust diversity measure. Where TTR falls with length and MATTR
    fixes a window, MTLD measures how many tokens it takes for diversity to decay to a
    threshold, which is a property of the writing rather than of the sample size. Added
    in round 6 because block F's five columns were the strongest transfer family in
    notebooks 14's ablation and MTLD is the obvious missing member.

    Parameters
    ----------
    tokens : list of str
    threshold : float, default 0.72
        The conventional value from McCarthy and Jarvis.

    Returns
    -------
    float
        0.0 for an empty document. Never inf: a document that never crosses the
        threshold yields one partial factor, so the divisor is bounded below.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    fwd = _mtld_one_direction(tokens, threshold)
    bwd = _mtld_one_direction(tokens[::-1], threshold)
    scores = [n / f for f in (fwd, bwd) if f > 0]
    return float(np.mean(scores)) if scores else float(n)


def _entropy(values):
    """Shannon entropy in nats of the empirical distribution of `values`.

    Used for the length distributions in block K: two documents can share a mean and a
    standard deviation while one alternates between two lengths and the other spreads
    smoothly, and entropy separates them.
    """
    if len(values) == 0:
        return 0.0
    counts = np.asarray(list(Counter(values).values()), dtype=np.float64)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _ngram_stats(tokens, n):
    """(distinct rate, max repeat rate) over `n`-grams of `tokens`.

    Distinct-n is a direct measure of local repetition and one of the better-known
    decoded-text markers: sampling with a low temperature reuses phrasing, which shows
    up here long before it shows up in a unigram statistic.

    Returns (0.0, 0.0) when the document is shorter than one n-gram.
    """
    total = len(tokens) - n + 1
    if total <= 0:
        return 0.0, 0.0
    grams = Counter(tuple(tokens[i:i + n]) for i in range(total))
    return len(grams) / total, max(grams.values()) / total


def _moments(x):
    """(skew, excess kurtosis) computed directly, returning 0.0 on a constant array.

    scipy.stats would warn and return nan when the standard deviation is 0, which
    happens for any single-sentence document.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return 0.0, 0.0
    sd = x.std()
    if sd <= 1e-12:
        return 0.0, 0.0
    z = (x - x.mean()) / sd
    return float((z ** 3).mean()), float((z ** 4).mean() - 3.0)


def _acf1(x):
    """Lag-1 autocorrelation, 0.0 when undefined.

    Asks whether a long sentence tends to follow a long sentence. Burstiness measures
    how *much* sentence length varies; this measures whether the variation is
    structured or independent, which is a different question and not implied by it.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    denom = float((x ** 2).sum())
    if denom <= 1e-12:
        return 0.0
    return float((x[:-1] * x[1:]).sum() / denom)


def _longest_similar_run(x, tol=0.25):
    """Longest run of consecutive values within `tol` relative distance of each other.

    A long run of similarly-sized sentences is the concrete shape of the steady rhythm
    that burstiness only summarises.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return float(len(x))
    best = run = 1
    for a, b in zip(x[:-1], x[1:]):
        ref = max(abs(a), 1e-9)
        run = run + 1 if abs(b - a) / ref <= tol else 1
        best = max(best, run)
    return float(best)


def _document_features(text):
    """Every dense block for one document, as a flat dict.

    Computed in a single pass over the text so 27,000 documents are not scanned once
    per block. `dense_features` slices the result back into blocks B-G.
    """
    text = "" if text is None else str(text)
    n_char = len(text)
    words = _WORD_RE.findall(text)
    n_word = len(words)
    lower_words = [w.lower() for w in words]
    lines = text.split("\n")
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    n_sent = len(sentences)
    word_lens = np.array([len(w) for w in words], dtype=float) if words else np.zeros(1)
    sent_words = (np.array([len(_WORD_RE.findall(s)) for s in sentences], dtype=float)
                  if sentences else np.zeros(1))
    para = [p for p in text.split("\n\n") if p.strip()]

    f = {}

    # --- B: punctuation, per 1000 characters ------------------------------------
    for ch in _PUNCT:
        f[f"B_punct_{ord(ch)}"] = 1000.0 * _safe_div(text.count(ch), n_char)
    f["B_punct_total"] = 1000.0 * _safe_div(
        sum(1 for c in text if not c.isalnum() and not c.isspace()), n_char)
    f["B_digit_rate"] = 1000.0 * _safe_div(sum(c.isdigit() for c in text), n_char)
    f["B_bold_md"] = text.count("**")
    f["B_header_md"] = text.count("##")
    f["B_rule_md"] = text.count("---")

    # --- C: casing ---------------------------------------------------------------
    f["C_upper_frac"] = _safe_div(sum(c.isupper() for c in text), n_char)
    f["C_allcaps_word_frac"] = _safe_div(
        sum(1 for w in words if len(w) > 1 and w.isupper()), n_word)
    f["C_titlecase_word_frac"] = _safe_div(
        sum(1 for w in words if w[:1].isupper() and w[1:].islower()), n_word)
    f["C_internal_cap_frac"] = _safe_div(
        sum(1 for w in words if any(c.isupper() for c in w[1:])), n_word)
    f["C_lower_start_sent_frac"] = _safe_div(
        sum(1 for s in sentences if s.strip()[:1].islower()), n_sent)

    # --- D: structure and layout --------------------------------------------------
    line_lens = np.array([len(ln) for ln in lines], dtype=float)
    f["D_n_lines"] = len(lines)
    f["D_newline_rate"] = 1000.0 * _safe_div(text.count("\n"), n_char)
    f["D_blank_lines"] = sum(1 for ln in lines if not ln.strip())
    f["D_line_len_mean"] = float(line_lens.mean())
    f["D_line_len_max"] = float(line_lens.max())
    f["D_line_len_std"] = float(line_lens.std())
    f["D_bullet_frac"] = _safe_div(sum(1 for ln in lines if _BULLET_RE.match(ln)),
                                   len(lines))
    f["D_header_frac"] = _safe_div(sum(1 for ln in lines if _HEADER_RE.match(ln)),
                                   len(lines))
    f["D_hardwrap"] = len(_HARDWRAP_RE.findall(text))
    f["D_n_paragraphs"] = len(para)

    # --- E: length and burstiness -------------------------------------------------
    f["E_log_char"] = float(np.log1p(n_char))
    f["E_log_word"] = float(np.log1p(n_word))
    f["E_log_sent"] = float(np.log1p(n_sent))
    f["E_word_len_mean"] = float(word_lens.mean())
    f["E_word_len_std"] = float(word_lens.std())
    f["E_sent_len_mean"] = float(sent_words.mean())
    f["E_sent_len_std"] = float(sent_words.std())
    # Burstiness: relative variability of sentence length. Human writing mixes long and
    # short sentences; decoded text tends to a steadier rhythm, so this is one of the
    # better-established single markers in the MGT literature.
    f["E_burstiness"] = _safe_div(float(sent_words.std()), float(sent_words.mean()))
    f["E_para_len_mean"] = float(np.mean([len(p) for p in para])) if para else 0.0
    f["E_chars_per_word"] = _safe_div(n_char, n_word)

    # --- F: lexical diversity ------------------------------------------------------
    n_type = len(set(lower_words))
    counts = pd.Series(lower_words).value_counts() if lower_words else pd.Series(dtype=int)
    f["F_ttr"] = _safe_div(n_type, n_word)
    f["F_hapax_frac"] = _safe_div(int((counts == 1).sum()), n_type)
    f["F_dis_frac"] = _safe_div(int((counts == 2).sum()), n_type)
    f["F_yules_k"] = _yules_k(lower_words)
    f["F_mattr_100"] = _mattr(lower_words, window=100)

    # --- G: readability ------------------------------------------------------------
    n_syl = sum(_count_syllables(w) for w in words) if words else 0
    wps = _safe_div(n_word, n_sent)
    spw = _safe_div(n_syl, n_word)
    f["G_flesch_reading_ease"] = 206.835 - 1.015 * wps - 84.6 * spw
    f["G_flesch_kincaid_grade"] = 0.39 * wps + 11.8 * spw - 15.59
    f["G_syllables_per_word"] = spw
    f["G_words_per_sentence"] = wps
    f["G_polysyllable_frac"] = _safe_div(
        sum(1 for w in words if _count_syllables(w) >= 3), n_word)

    # --- J: extended lexical diversity ---------------------------------------------
    # Block F returned the largest transfer value of any family in notebook 14 off five
    # columns. These are the length-robust measures F was missing, plus the local
    # repetition rates that no unigram statistic can see.
    f["J_mattr_25"] = _mattr(lower_words, window=25)
    f["J_mattr_50"] = _mattr(lower_words, window=50)
    f["J_mattr_200"] = _mattr(lower_words, window=200)
    f["J_mtld"] = _mtld(lower_words)
    log_n = float(np.log(n_word)) if n_word > 1 else 0.0
    log_v = float(np.log(n_type)) if n_type > 1 else 0.0
    f["J_herdan_c"] = _safe_div(log_v, log_n)
    f["J_maas"] = _safe_div(log_n - log_v, log_n ** 2)
    # Simpson's D: probability that two tokens drawn without replacement match.
    f["J_simpson_d"] = _safe_div(
        float((counts * (counts - 1)).sum()) if len(counts) else 0.0,
        n_word * (n_word - 1))
    n_hapax = int((counts == 1).sum()) if len(counts) else 0
    # Honore's R blows up as the hapax fraction approaches 1, which is every very short
    # document, so the denominator is floored rather than left to produce inf.
    f["J_honore_r"] = _safe_div(100.0 * log_n,
                                max(1.0 - _safe_div(n_hapax, n_type), 1e-3))
    d2, r2 = _ngram_stats(lower_words, 2)
    d3, r3 = _ngram_stats(lower_words, 3)
    f["J_distinct_2"] = d2
    f["J_distinct_3"] = d3
    f["J_max_trigram_repeat"] = r3
    content = [w for w in lower_words if w not in ENGLISH_STOP_WORDS]
    stops = [w for w in lower_words if w in ENGLISH_STOP_WORDS]
    f["J_content_ttr"] = _safe_div(len(set(content)), len(content))
    f["J_stopword_ttr"] = _safe_div(len(set(stops)), len(stops))
    half = n_word // 2
    if half >= 1:
        first = _safe_div(len(set(lower_words[:half])), half)
        second = _safe_div(len(set(lower_words[half:])), n_word - half)
        f["J_ttr_half_ratio"] = _safe_div(first, second)
    else:
        f["J_ttr_half_ratio"] = 0.0

    # --- K: rhythm and variability --------------------------------------------------
    # E_burstiness is std/mean of sentence length, one summary of a distribution worth
    # several. E_burstiness and E_sent_len_std were the second and third most important
    # dense features on a held-out cluster in notebook 14 section 6.
    sl = sent_words
    sl_mean, sl_std = float(sl.mean()), float(sl.std())
    # sent_words is np.zeros(1) when the document has no sentence terminator, so it is
    # never empty and the percentiles below are always defined.
    q1 = float(np.percentile(sl, 25))
    q3 = float(np.percentile(sl, 75))
    f["K_sent_len_iqr"] = q3 - q1
    f["K_sent_len_iqr_ratio"] = _safe_div(q3 - q1, float(np.median(sl)))
    f["K_sent_len_mad"] = float(np.abs(sl - np.median(sl)).mean())
    f["K_sent_len_entropy"] = _entropy([int(v) for v in sl])
    f["K_sent_len_acf1"] = _acf1(sl)
    skew, kurt = _moments(sl)
    f["K_sent_len_skew"] = skew
    f["K_sent_len_kurt"] = kurt
    f["K_frac_within_1sd"] = (
        float(np.mean(np.abs(sl - sl_mean) <= sl_std)) if sl_std > 0 else 1.0)
    f["K_longest_run_similar"] = _longest_similar_run(sl)
    para_lens = np.array([len(p) for p in para], dtype=float) if para else np.zeros(1)
    f["K_para_len_std"] = float(para_lens.std())
    f["K_para_len_cv"] = _safe_div(float(para_lens.std()), float(para_lens.mean()))
    f["K_word_len_entropy"] = _entropy([int(v) for v in word_lens])

    return f


def dense_features(texts):
    """Blocks B to G for a corpus, in one pass.

    Parameters
    ----------
    texts : array-like of str, shape (n_samples,)

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        float64, guaranteed finite.
    names : list of str
        Column names, each prefixed with its block letter.
    """
    rows = [_document_features(t) for t in texts]
    frame = pd.DataFrame(rows)
    # Column order follows the block order, not dict insertion, so a build is
    # reproducible even if _document_features is reordered later.
    names = sorted(frame.columns, key=lambda c: (c[0], c))
    X = frame[names].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.isfinite(X).all(), "dense features produced non-finite values"
    return X, list(names)


def function_word_features(texts):
    """Block A: rates of ~318 English stop words, normalised by document word count.

    These are the words the course preprocessing removed. In authorship attribution
    they are the standard feature basis precisely because they are content-free: an
    author's rate of "the", "of" and "would" is stable across what they write about,
    which is what makes the feature survive a domain shift.

    The vocabulary is **pinned** to sklearn's `ENGLISH_STOP_WORDS`, so this transform
    is not fitted and cannot leak. `token_pattern` is widened from sklearn's default,
    which requires two or more characters and would otherwise silently drop "a" and
    "i" from a stop-word list that contains them.

    Parameters
    ----------
    texts : array-like of str

    Returns
    -------
    X : sparse CSR, shape (n_samples, n_stop_words)
    names : list of str
    """
    vocab = sorted(ENGLISH_STOP_WORDS)
    vec = CountVectorizer(vocabulary=vocab, lowercase=True,
                          token_pattern=r"(?u)\b\w+\b")
    counts = vec.transform(texts).astype(np.float64)
    n_word = np.array([max(len(_WORD_RE.findall(str(t))), 1) for t in texts],
                      dtype=np.float64)
    X = sparse.csr_matrix(counts.multiply(1.0 / n_word[:, None]))
    return X, [f"A_fw_{w}" for w in vocab]


def char_ngram_features(train_texts, test_texts, *, max_features=20000, min_df=3):
    """Block H: character n-grams, 2 to 5, word-boundary aware.

    `char_wb` keeps n-grams inside word boundaries and pads them, so the features are
    morphological (suffixes, prefixes, contractions) plus punctuation context, without
    needing a tokenizer, lemmatizer or POS tagger. This is the block most likely to
    carry on its own, and also the one whose transfer is most worth checking, since
    long character n-grams can memorise domain vocabulary as effectively as word
    features can.

    Fitted on `train_texts` only.

    Returns
    -------
    Xtr, Xte : sparse CSR
    names : list of str
    """
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True,
                          min_df=min_df, max_features=max_features,
                          sublinear_tf=True, dtype=np.float32)
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    return Xtr, Xte, [f"H_char_{t}" for t in vec.get_feature_names_out()]


def word_ngram_features(train_texts, test_texts, *, max_features=20000, min_df=5):
    """Block I: word unigrams and bigrams, stop words KEPT, no lemmatization.

    The direct control against the supplied features. Same family of representation,
    same corpus, but retaining exactly what the course pipeline threw away. If block I
    beats the supplied matrix, the loss came from the preprocessing rather than from
    the representation, and that is a clean result for the report either way.

    Fitted on `train_texts` only.

    Returns
    -------
    Xtr, Xte : sparse CSR
    names : list of str
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words=None,
                          min_df=min_df, max_features=max_features,
                          sublinear_tf=True, dtype=np.float32)
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    return Xtr, Xte, [f"I_word_{t}" for t in vec.get_feature_names_out()]


def build_blocks(train_texts, test_texts, blocks=ALL_BLOCKS):
    """Build every requested block and return them keyed by name.

    Parameters
    ----------
    train_texts, test_texts : array-like of str
    blocks : sequence of str, default all nine

    Returns
    -------
    dict
        name -> {"train": CSR, "test": CSR, "names": list of str}. Dense blocks are
        returned as CSR too, so downstream code has one code path; they are small
        enough that the overhead does not matter.
    """
    unknown = [b for b in blocks if b not in ALL_BLOCKS]
    assert not unknown, f"unknown blocks {unknown}, expected from {ALL_BLOCKS}"
    out = {}

    wanted_dense = [b for b in blocks if b in DENSE_BLOCKS]
    if wanted_dense:
        # One pass over each corpus covers all of B-G; slice it per block afterwards.
        Xtr, names = dense_features(train_texts)
        Xte, names_te = dense_features(test_texts)
        assert names == names_te, "train and test dense feature names diverged"
        for block in wanted_dense:
            cols = [i for i, n in enumerate(names) if n.startswith(block[0] + "_")]
            out[block] = {"train": sparse.csr_matrix(Xtr[:, cols]),
                          "test": sparse.csr_matrix(Xte[:, cols]),
                          "names": [names[i] for i in cols]}

    if "A_function_words" in blocks:
        Xtr, names = function_word_features(train_texts)
        Xte, _ = function_word_features(test_texts)
        out["A_function_words"] = {"train": Xtr, "test": Xte, "names": names}

    if "H_char_ngrams" in blocks:
        Xtr, Xte, names = char_ngram_features(train_texts, test_texts)
        out["H_char_ngrams"] = {"train": Xtr, "test": Xte, "names": names}

    if "I_word_ngrams" in blocks:
        Xtr, Xte, names = word_ngram_features(train_texts, test_texts)
        out["I_word_ngrams"] = {"train": Xtr, "test": Xte, "names": names}

    return out


def stack(built, blocks):
    """Horizontally concatenate several built blocks into one matrix.

    Parameters
    ----------
    built : dict
        As returned by `build_blocks`.
    blocks : sequence of str
        Which blocks to include, and in what order.

    Returns
    -------
    Xtr, Xte : sparse CSR
    names : list of str
    """
    blocks = list(blocks)
    assert blocks, "no blocks selected"
    missing = [b for b in blocks if b not in built]
    assert not missing, f"blocks not built: {missing}"
    Xtr = sparse.hstack([built[b]["train"] for b in blocks], format="csr")
    Xte = sparse.hstack([built[b]["test"] for b in blocks], format="csr")
    names = [n for b in blocks for n in built[b]["names"]]
    assert Xtr.shape[1] == len(names) == Xte.shape[1]
    return Xtr.astype(np.float32), Xte.astype(np.float32), names


def cache_path(block):
    """Where one built block is cached: data/processed/textfeat_<block>.npz."""
    paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    return paths.DATA_PROCESSED / f"textfeat_{block}.npz"


def save_block(block, entry):
    """Persist one built block. Mirrors the CSR cache pattern in `src/data.py`.

    Feature names are stored as a fixed-width unicode array rather than an object
    array, so neither saving nor loading needs `allow_pickle`. Loading a pickle
    executes arbitrary code, and these caches are passed between teammates through
    git; a plain string array removes that from the threat model entirely.
    """
    path = cache_path(block)
    np.savez_compressed(
        path,
        train_data=entry["train"].data, train_indices=entry["train"].indices,
        train_indptr=entry["train"].indptr,
        train_shape=np.asarray(entry["train"].shape, dtype=np.int64),
        test_data=entry["test"].data, test_indices=entry["test"].indices,
        test_indptr=entry["test"].indptr,
        test_shape=np.asarray(entry["test"].shape, dtype=np.int64),
        names=np.asarray(entry["names"], dtype=np.str_))
    return path


def load_block(block):
    """Load one cached block, or raise if it has not been built yet.

    Returns
    -------
    dict
        {"train": CSR, "test": CSR, "names": list of str}
    """
    path = cache_path(block)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not built yet - run 12_text_features section 3 first")
    # allow_pickle stays at its safe default of False; nothing here is pickled.
    z = np.load(path)
    train = sparse.csr_matrix((z["train_data"], z["train_indices"], z["train_indptr"]),
                              shape=tuple(z["train_shape"]))
    test = sparse.csr_matrix((z["test_data"], z["test_indices"], z["test_indptr"]),
                             shape=tuple(z["test_shape"]))
    return {"train": train, "test": test, "names": [str(n) for n in z["names"]]}


def load_blocks(blocks=ALL_BLOCKS):
    """Load several cached blocks at once."""
    return {b: load_block(b) for b in blocks}
