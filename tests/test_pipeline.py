import re
from pathlib import Path

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest

from metrics import best_threshold, evaluate

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
PROC_S = ROOT / "data" / "processed_skab"

need_cmapss = pytest.mark.skipif(not (PROC / "test.npz").exists(), reason="run 04_preprocessing.py first")
need_skab = pytest.mark.skipif(not (PROC_S / "test.npz").exists(), reason="run 07_skab_preprocessing.py first")


def load(folder):
    return {n: np.load(folder / f"{n}.npz") for n in ("train", "val", "test")}


# ---------- metrics (T1-T4) ----------
def test_perfect_scores_give_perfect_metrics():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    m = evaluate(y, s, best_threshold(y, s))
    assert m["f1"] == 1.0 and m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0


def test_inverted_scores_give_zero_auc():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert evaluate(y, s, 0.5)["roc_auc"] == 0.0


def test_best_threshold_separates_clean_classes():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    thr = best_threshold(y, s)
    assert 0.2 < thr <= 0.8


def test_evaluate_returns_all_metrics():
    m = evaluate(np.array([0, 1, 0, 1]), np.array([0.2, 0.7, 0.4, 0.6]), 0.5)
    assert set(m) == {"precision", "recall", "f1", "roc_auc", "pr_auc"}


# ---------- windowing on toy data (T5-T6) ----------
def test_window_shape_and_content():
    X = np.arange(100).reshape(50, 2)
    W = sliding_window_view(X, 30, axis=0)            # (windows, sensors, 30)
    assert W.shape == (21, 2, 30)
    assert np.array_equal(W[0, :, -1], X[29])         # first window ends at row 29
    assert np.array_equal(W[-1, :, -1], X[-1])        # last window ends at last row


def test_window_label_is_last_row():
    labels = np.r_[np.zeros(40, int), np.ones(10, int)]
    y = labels[30 - 1:]
    assert len(y) == 21 and y[-1] == 1 and y[0] == 0


# ---------- C-MAPSS leakage and structure (T7-T12) ----------
@need_cmapss
def test_cmapss_no_engine_overlap():
    d = load(PROC)
    tr, va, te = (set(np.unique(d[n]["engine"])) for n in ("train", "val", "test"))
    assert not tr & va and not tr & te and not va & te


@need_cmapss
def test_cmapss_split_sizes():
    d = load(PROC)
    assert [len(np.unique(d[n]["engine"])) for n in ("train", "val", "test")] == [70, 15, 15]


@need_cmapss
def test_cmapss_window_shape():
    d = load(PROC)
    for n in d:
        assert d[n]["X"].shape[1:] == (30, 15)
        assert len(d[n]["X"]) == len(d[n]["y"]) == len(d[n]["engine"])


@need_cmapss
def test_cmapss_labels_are_monotonic_per_engine():
    # RUL <= 30 gives 31 anomalous cycles at the end of each engine's life
    te = np.load(PROC / "test.npz")
    for e in np.unique(te["engine"]):
        y = te["y"][te["engine"] == e]
        assert np.all(np.diff(y) >= 0)                # healthy first, then anomalous
        assert y.sum() == 31


@need_cmapss
def test_cmapss_scaler_fitted_on_healthy_train():
    tr = np.load(PROC / "train.npz")
    h = tr["X"][tr["y"] == 0]
    assert np.all(np.abs(h.mean(axis=(0, 1))) < 0.1)
    assert np.all(np.abs(h.std(axis=(0, 1)) - 1) < 0.15)


@need_cmapss
def test_cmapss_no_nan_or_inf():
    for n, d in load(PROC).items():
        assert np.isfinite(d["X"]).all()


# ---------- SKAB (T13-T14) ----------
@need_skab
def test_skab_train_has_no_anomalies_and_8_features():
    tr = np.load(PROC_S / "train.npz")
    assert tr["y"].sum() == 0
    assert tr["X"].shape[1:] == (30, 8)


@need_skab
def test_skab_val_test_have_anomalies():
    d = load(PROC_S)
    assert d["val"]["y"].sum() > 0 and d["test"]["y"].sum() > 0


# ---------- threshold discipline (T15) ----------
def test_no_script_picks_threshold_on_test():
    bad = re.compile(r"best_threshold\(\s*(te\b|yte\b|te\[)")
    offenders = [p.name for p in (ROOT / "src").glob("*.py") if bad.search(p.read_text())]
    assert offenders == []


# ---------- reproducibility (T16-T17) ----------
def test_isolation_forest_same_seed_same_scores():
    X = np.random.default_rng(0).normal(size=(200, 5))
    a = IsolationForest(random_state=1).fit(X).score_samples(X)
    b = IsolationForest(random_state=1).fit(X).score_samples(X)
    assert np.array_equal(a, b)


def test_torch_same_seed_same_init():
    torch = pytest.importorskip("torch")

    def make():
        torch.manual_seed(7)
        return torch.nn.Linear(4, 3).weight.detach().clone()

    assert torch.equal(make(), make())