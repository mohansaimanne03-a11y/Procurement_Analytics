#!/usr/bin/env python3
"""KB-PPV: knowledge-based, uncertainty-aware purchase-price-variance (PPV) prediction
and cost-saving action optimisation for procurement analytics.

Pipeline (Figure 1 of the paper)
    1. Knowledge layer  - monotone domain priors + action-eligibility rules
    2. Predictor        - monotone-constrained gradient boosting for PPV (%)
    3. Uncertainty      - conformalised quantile regression (CQR) intervals
    4. Optimiser        - multiple-choice knapsack (exact DP) over buyer hours
    5. Explanations     - occlusion-based driver attribution per PO line

Usage
    python kbppv.py                    # full experiment (10 independent replicates)
    python kbppv.py --reps 3           # quicker run
    python kbppv.py --data lines.csv   # your own PO-line data (schema in README)
    python kbppv.py --selftest         # fast correctness checks
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# 1. Knowledge layer                                                          #
# --------------------------------------------------------------------------- #
CATEGORIES = ["Electronics", "Mechanical", "Raw materials", "Packaging", "Services", "Software/SoW"]
BASE_COLS = ["order_value", "quantity", "lead_time_days", "commodity_volatility", "supplier_otd",
             "months_since_negotiation", "num_quotes", "expedite", "single_source",
             "contract_coverage", "npi", "non_usd"]

# Domain priors: sign of the partial effect of each driver on PPV.
# +1 = a higher value never lowers expected PPV; -1 = never raises it.
MONOTONE = {"commodity_volatility": +1, "expedite": +1, "single_source": +1,
            "months_since_negotiation": +1, "num_quotes": -1, "contract_coverage": -1,
            "supplier_otd": -1}
BINARY = {"expedite", "single_source", "contract_coverage"}

# Buyer actions: (review hours per line, share of overspend recovered).
ACTIONS = {"renegotiate": (6, 0.30), "competitive_rfq": (14, 0.50), "price_lock": (10, 0.35)}
HOURS = np.array([h for h, _ in ACTIONS.values()])
RECOVERY = np.array([r for _, r in ACTIONS.values()])


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Model features: log-scaled size variables, flags, one-hot category."""
    X = df[BASE_COLS].astype(float).copy()
    X["order_value"], X["quantity"] = np.log1p(X["order_value"]), np.log1p(X["quantity"])
    X = X.rename(columns={"order_value": "log_order_value", "quantity": "log_quantity"})
    for c in CATEGORIES:
        X[f"cat_{c}"] = (df["category"] == c).astype(float)
    return X


def monotone_vector(columns) -> list[int]:
    return [MONOTONE.get(c, 0) for c in columns]


def eligibility(df: pd.DataFrame, rules: bool = True) -> np.ndarray:
    """Knowledge rules: which actions are admissible for each PO line (n x 3)."""
    if not rules:
        return np.ones((len(df), 3), bool)
    reneg = np.ones(len(df), bool)
    rfq = ((df["single_source"] == 1) | (df["num_quotes"] <= 2)).to_numpy()
    lock = ((df["contract_coverage"] == 0) & (df["commodity_volatility"] > 0.8)).to_numpy()
    return np.column_stack([reneg, rfq, lock])


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def make_synthetic(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Controlled PO-line benchmark with a known data-generating process.

    PPV(%) = (actual - standard) / standard * 100.  Volatility, expediting, single
    sourcing and stale contracts raise PPV; competition, contract coverage and
    on-time delivery lower it.  Noise is heavy-tailed (Student-t, df=5) and
    heteroscedastic in volatility.  Ground truth is known, so decision quality can be
    scored against an oracle.
    """
    rng = np.random.default_rng(seed)
    cat = rng.choice(len(CATEGORIES), n, p=[.22, .20, .18, .14, .14, .12])
    base = np.array([1.0, 0.6, 2.2, 0.3, 0.1, -0.2])[cat]
    vol = np.clip(rng.gamma(2.0, 0.4, n) * np.array([.7, .5, 1.2, .35, .15, .15])[cat], 0, 3)
    value = rng.lognormal(9.3, 1.2, n)
    qty = np.maximum(1, rng.lognormal(4.0, 1.3, n).round())
    lead = np.clip(rng.normal(35, 16, n), 3, 120).round()
    expedite = rng.random(n) < 0.04 + 0.5 / (1 + np.exp((lead - 15) / 5))
    single = rng.random(n) < 0.22 + 0.12 * (cat == 0) + 0.08 * (cat == 1)
    quotes = np.clip(np.where(single, 1, 1 + rng.poisson(1.4, n)), 1, 6)
    contract = rng.random(n) < 0.60 - 0.25 * single
    otd = np.clip(rng.beta(9, 2, n), 0.5, 1.0)
    months = rng.integers(0, 37, n)
    npi = rng.random(n) < 0.15
    non_usd = rng.random(n) < 0.30
    mean = (base + 2.6 * vol * (1 - 0.6 * contract) - 0.8 * contract + 3.0 * expedite
            + 1.8 * single * (1 + 0.5 * vol) - 0.7 * (quotes - 1) + 0.05 * months * (1 + vol)
            - 5.0 * (otd - 0.85) + 1.2 * npi + 0.8 * non_usd * vol - 0.25 * np.log(value / 1e4))
    noise = (0.8 + 1.6 * vol) * rng.standard_t(5, n) / np.sqrt(5 / 3)
    return pd.DataFrame({
        "category": np.array(CATEGORIES)[cat], "order_value": value.round(2), "quantity": qty,
        "lead_time_days": lead, "commodity_volatility": vol.round(3), "supplier_otd": otd.round(3),
        "months_since_negotiation": months, "num_quotes": quotes, "expedite": expedite.astype(int),
        "single_source": single.astype(int), "contract_coverage": contract.astype(int),
        "npi": npi.astype(int), "non_usd": non_usd.astype(int),
        "ppv_pct": np.clip(mean + noise, -15, 40).round(3)})


def load_data(path: str | None, n: int, seed: int) -> pd.DataFrame:
    if path is None:
        return make_synthetic(n, seed)
    df = pd.read_csv(path)
    missing = set(BASE_COLS + ["category", "ppv_pct"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    unknown = set(df["category"]) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown categories {sorted(unknown)}; edit CATEGORIES in kbppv.py")
    return df.dropna(subset=BASE_COLS + ["category", "ppv_pct"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2-3. Predictor and conformalised uncertainty                                #
# --------------------------------------------------------------------------- #
def gbm(seed: int, cst=None, **kw) -> HGB:
    return HGB(learning_rate=0.06, max_iter=400, max_leaf_nodes=15, min_samples_leaf=20,
               l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
               n_iter_no_change=25, monotonic_cst=cst, random_state=seed, **kw)


MODELS = ["Mean baseline", "Ridge regression", "Random forest", "Gradient boosting", "KB-PPV (monotone GB)"]


def make_models(columns, seed: int) -> dict:
    return {
        MODELS[0]: DummyRegressor(),
        MODELS[1]: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))),
        MODELS[2]: RandomForestRegressor(200, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        MODELS[3]: gbm(seed),
        MODELS[4]: gbm(seed, monotone_vector(columns))}


class CQR:
    """Conformalised quantile regression (Romano et al., 2019) with monotone priors."""

    def __init__(self, columns, alpha: float = 0.10, seed: int = 0):
        self.cst, self.alpha, self.seed = monotone_vector(columns), alpha, seed

    def fit(self, X, y):
        Xp, Xc, yp, yc = train_test_split(X, y, test_size=0.25, random_state=self.seed)
        a = self.alpha
        self.lo = gbm(self.seed, self.cst, loss="quantile", quantile=a / 2).fit(Xp, yp)
        self.hi = gbm(self.seed, self.cst, loss="quantile", quantile=1 - a / 2).fit(Xp, yp)
        score = np.maximum(self.lo.predict(Xc) - yc, yc - self.hi.predict(Xc))
        k = int(np.ceil((1 - a) * (len(yc) + 1)))
        self.q = np.sort(score)[min(k, len(score)) - 1]
        return self

    def predict(self, X):
        lo, hi = self.lo.predict(X), self.hi.predict(X)
        return np.minimum(lo, hi) - self.q, np.maximum(lo, hi) + self.q


def violation_rate(model, X: pd.DataFrame) -> float:
    """Share of (line, constrained driver) pairs where moving the driver from its
    favourable to its unfavourable extreme moves the prediction against domain knowledge."""
    bad = tot = 0
    for col, sign in MONOTONE.items():
        lo_v, hi_v = (0.0, 1.0) if col in BINARY else (X[col].quantile(.1), X[col].quantile(.9))
        lo, hi = X.copy(), X.copy()
        lo[col], hi[col] = lo_v, hi_v
        diff = model.predict(hi) - model.predict(lo)
        bad += int((sign * diff < -1e-9).sum())
        tot += len(X)
    return 100 * bad / tot


# --------------------------------------------------------------------------- #
# 4. Optimiser: multiple-choice knapsack                                      #
# --------------------------------------------------------------------------- #
def action_values(overspend_pct, value, elig) -> np.ndarray:
    """Savings (n x 3) of each admissible action: recovered share of the overspend."""
    base = np.maximum(overspend_pct, 0)[:, None] / 100 * value[:, None]
    return base * RECOVERY[None, :] * elig


def solve_mckp(values: np.ndarray, capacity: int) -> np.ndarray:
    """Exact DP: at most one action per line, maximise value within `capacity` review
    hours.  Returns choice per line (0 = none, a = action index + 1)."""
    n = len(values)
    best = np.zeros(capacity + 1)
    choice = np.zeros((n, capacity + 1), dtype=np.int8)
    for i in np.flatnonzero(values.max(axis=1) > 0):
        new = best.copy()
        for a, h in enumerate(HOURS):
            if values[i, a] > 0 and h <= capacity:
                cand = best[: capacity + 1 - h] + values[i, a]
                better = cand > new[h:]
                new[h:] = np.where(better, cand, new[h:])
                choice[i, h:][better] = a + 1
        best = new
    sel, c = np.zeros(n, dtype=int), capacity
    for i in range(n - 1, -1, -1):
        a = choice[i, c]
        if a:
            sel[i], c = a, c - HOURS[a - 1]
    return sel


def top_k_renegotiate(score: np.ndarray, capacity: int) -> np.ndarray:
    """Greedy baseline: renegotiate the highest-scored lines until hours run out."""
    sel = np.zeros(len(score), dtype=int)
    sel[np.argsort(-score)[: capacity // HOURS[0]]] = 1
    return sel


def realised(sel: np.ndarray, true_vals: np.ndarray) -> tuple[float, float]:
    """(realised savings, share of actioned lines that recovered nothing)."""
    idx = np.flatnonzero(sel)
    got = true_vals[idx, sel[idx] - 1]
    return float(got.sum()), float(100 * np.mean(got <= 0)) if len(idx) else 0.0


# --------------------------------------------------------------------------- #
# 5. Explanations                                                             #
# --------------------------------------------------------------------------- #
def explain(model, X_ref: pd.DataFrame, x: pd.Series, top: int = 3) -> str:
    """Occlusion attribution: change in predicted PPV when a driver is reset to its
    reference median (category dummies are occluded together)."""
    med = X_ref.median()
    base = float(model.predict(x.to_frame().T)[0])
    groups = {c: [c] for c in X_ref.columns if not c.startswith("cat_")}
    groups["category"] = [c for c in X_ref.columns if c.startswith("cat_")]
    effect = {}
    for g, cols in groups.items():
        z = x.copy()
        z[cols] = med[cols]
        effect[g] = base - float(model.predict(z.to_frame().T)[0])
    drivers = sorted(effect.items(), key=lambda kv: -abs(kv[1]))[:top]
    return f"predicted PPV {base:+.2f}%; main drivers: " + ", ".join(f"{g} ({d:+.2f} pp)" for g, d in drivers)


# --------------------------------------------------------------------------- #
# Experiment: one replicate = one independent synthetic dataset               #
# --------------------------------------------------------------------------- #
POLICIES = ["Random lines", "Largest spend first", "Gradient boosting + greedy", "KB-PPV + greedy",
            "KB-PPV + MCKP, no eligibility rules", "Ridge + MCKP", "KB-PPV + MCKP", "KB-PPV + MCKP + interval gate",
            "Oracle (true PPV) + MCKP"]
FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def run_replicate(df: pd.DataFrame, seed: int, n_random: int = 20) -> dict:
    X, y = design_matrix(df), df["ppv_pct"].to_numpy()
    n, cols = len(df), X.columns
    rng = np.random.default_rng(seed)

    # (a) 5-fold CV predictions, intervals and metrics
    oof = {m: np.zeros(n) for m in MODELS}
    lo, hi = np.zeros(n), np.zeros(n)
    cv = {m: [] for m in MODELS}
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        for name, model in make_models(cols, seed).items():
            p = model.fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            oof[name][te] = p
            cv[name].append((mean_absolute_error(y[te], p), mean_squared_error(y[te], p) ** .5, r2_score(y[te], p)))
        lo[te], hi[te] = CQR(cols, 0.10, seed).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    res = {"cv": {m: np.mean(v, axis=0) for m, v in cv.items()},
           "coverage": float(np.mean((y >= lo) & (y <= hi))), "width": float(np.mean(hi - lo))}

    # (b) stress tests: market-shock extrapolation and small-sample regime
    vol = df["commodity_volatility"].to_numpy()
    tr, te = np.flatnonzero(vol <= np.quantile(vol, .70)), np.flatnonzero(vol > np.quantile(vol, .90))
    res["shift"] = {m: mean_absolute_error(y[te], mod.fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
                    for m, mod in make_models(cols, seed).items()}
    c_lo, c_hi = CQR(cols, 0.10, seed).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    res["shift_coverage"] = float(np.mean((y[te] >= c_lo) & (y[te] <= c_hi)))
    perm = rng.permutation(n)
    tr, te = perm[:300], perm[300:]
    res["small"] = {m: mean_absolute_error(y[te], mod.fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
                    for m, mod in make_models(cols, seed).items()}

    # (c) domain-consistency of each fitted model
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=seed)
    res["viol"] = {m: violation_rate(mod.fit(Xtr, ytr), Xte) for m, mod in make_models(cols, seed).items()}

    # (d) decision experiment at several capacities
    value, elig = df["order_value"].to_numpy(), eligibility(df)
    true_vals = action_values(y, value, elig)
    kb_vals = action_values(oof[MODELS[4]], value, elig)
    free_vals = action_values(oof[MODELS[4]], value, eligibility(df, rules=False))
    gate_vals = kb_vals * (lo > 0)[:, None]
    exp_reneg = lambda p: action_values(p, value, elig)[:, 0]
    rows = []
    for f in FRACTIONS:
        cap = int(f * n * HOURS[0])
        sels = {
            "Largest spend first": top_k_renegotiate(value, cap),
            "Gradient boosting + greedy": top_k_renegotiate(exp_reneg(oof[MODELS[3]]), cap),
            "KB-PPV + greedy": top_k_renegotiate(exp_reneg(oof[MODELS[4]]), cap),
            "KB-PPV + MCKP, no eligibility rules": solve_mckp(free_vals, cap),
            "Ridge + MCKP": solve_mckp(action_values(oof[MODELS[1]], value, elig), cap),
            "KB-PPV + MCKP": solve_mckp(kb_vals, cap),
            "KB-PPV + MCKP + interval gate": solve_mckp(gate_vals, cap),
            "Oracle (true PPV) + MCKP": solve_mckp(true_vals, cap)}
        oracle = realised(sels["Oracle (true PPV) + MCKP"], true_vals)[0]
        out = {k: realised(s, true_vals) + (int(np.count_nonzero(s)),) for k, s in sels.items()}
        rnd = [realised(top_k_renegotiate(rng.random(n), cap), true_vals) for _ in range(n_random)]
        out["Random lines"] = (np.mean([r[0] for r in rnd]), np.mean([r[1] for r in rnd]), cap // HOURS[0])
        for k, (sav, waste, lines) in out.items():
            rows.append(dict(fraction=f, policy=k, savings=sav, pct_oracle=100 * sav / oracle,
                             per_hour=sav / cap, lines=lines, waste=waste))
    res["decision"] = pd.DataFrame(rows)
    return res


# --------------------------------------------------------------------------- #
# Aggregation, tables, figures                                                #
# --------------------------------------------------------------------------- #
def ms(a, d=2) -> str:
    a = np.asarray(a, float)
    return f"{a.mean():.{d}f} ± {a.std(ddof=1):.{d}f}"


def aggregate(reps: list[dict], frac: float = 0.20):
    t1 = pd.DataFrame({
        "MAE (pp)": [ms([r["cv"][m][0] for r in reps], 3) for m in MODELS],
        "RMSE (pp)": [ms([r["cv"][m][1] for r in reps], 3) for m in MODELS],
        "R²": [ms([r["cv"][m][2] for r in reps], 3) for m in MODELS],
        "MAE, volatility shock (pp)": [ms([r["shift"][m] for r in reps], 2) for m in MODELS],
        "MAE, n = 300 (pp)": [ms([r["small"][m] for r in reps], 2) for m in MODELS],
        "Prior violations (%)": [ms([r["viol"][m] for r in reps], 1) for m in MODELS]}, index=MODELS)
    dec = pd.concat([r["decision"].assign(rep=i) for i, r in enumerate(reps)])
    d = dec[np.isclose(dec.fraction, frac)]
    g = d.groupby("policy")
    t2 = pd.DataFrame({
        "Savings (k$)": g["savings"].apply(lambda s: ms(s / 1e3, 1)),
        "% of oracle": g["pct_oracle"].apply(lambda s: ms(s, 1)),
        "Savings per hour ($)": g["per_hour"].apply(lambda s: ms(s, 0)),
        "Lines actioned": g["lines"].apply(lambda s: f"{s.mean():.0f}"),
        "Ineffective actions (%)": g["waste"].apply(lambda s: ms(s, 1))}).loc[POLICIES]
    p = lambda a, b: wilcoxon(a, b).pvalue
    pick = lambda pol, col: d[d.policy == pol].sort_values("rep")[col].to_numpy()
    tests = {
        "MAE: KB-PPV vs gradient boosting (CV)": p([r["cv"][MODELS[4]][0] for r in reps], [r["cv"][MODELS[3]][0] for r in reps]),
        "MAE: KB-PPV vs gradient boosting (shock)": p([r["shift"][MODELS[4]] for r in reps], [r["shift"][MODELS[3]] for r in reps]),
        "MAE: KB-PPV vs gradient boosting (n=300)": p([r["small"][MODELS[4]] for r in reps], [r["small"][MODELS[3]] for r in reps]),
        "Savings: KB-PPV+MCKP vs GB+greedy": p(pick("KB-PPV + MCKP", "savings"), pick("Gradient boosting + greedy", "savings")),
        "Savings: MCKP vs MCKP without eligibility": p(pick("KB-PPV + MCKP", "savings"), pick("KB-PPV + MCKP, no eligibility rules", "savings")),
        "Savings: MCKP vs MCKP + gate": p(pick("KB-PPV + MCKP", "savings"), pick("KB-PPV + MCKP + interval gate", "savings")),
        "Ineffective: MCKP vs MCKP + gate": p(pick("KB-PPV + MCKP", "waste"), pick("KB-PPV + MCKP + interval gate", "waste"))}
    extra = {"coverage": ms([r["coverage"] for r in reps], 3), "width": ms([r["width"] for r in reps], 2),
             "shift_coverage": ms([r["shift_coverage"] for r in reps], 3)}
    return t1, t2, tests, extra, dec


C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "violet": "#6250d6", "gray": "#8a8984"}


def plot_framework(out: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 10), ax.set_ylim(0, 4.6), ax.axis("off")
    boxes = [(0.1, 2.9, 2.5, 1.3, "(1) PO-line data\nspend, supplier, market,\nlead time, contract", "#eef3fb"),
             (3.75, 2.9, 2.5, 1.3, "(2) Knowledge layer\nmonotone priors +\naction-eligibility rules", "#fdf0e8"),
             (7.4, 2.9, 2.5, 1.3, "(3) Predictor\nmonotone gradient\nboosting: PPV (%)", "#e6f6ef"),
             (7.4, 0.4, 2.5, 1.3, "(4) Uncertainty\nconformal quantile\nregression (90%)", "#e6f6ef"),
             (3.75, 0.4, 2.5, 1.3, "(5) Optimiser\nmultiple-choice knapsack\nunder hour budget", "#efedfb"),
             (0.1, 0.4, 2.5, 1.3, "(6) Outputs\naction plan + driver\nexplanations per line", "#eef3fb")]
    for x, y, w, h, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=c, ec="#55544f", lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5, color="#0b0b0b")
    for a, b in [((2.65, 3.55), (3.7, 3.55)), ((6.3, 3.55), (7.35, 3.55)), ((8.65, 2.85), (8.65, 1.75)),
                 ((7.35, 1.05), (6.3, 1.05)), ((3.7, 1.05), (2.65, 1.05)), ((5.0, 2.85), (5.0, 1.75))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
    ax.text(5.12, 2.3, "admissible actions", fontsize=7, style="italic", color="#52514e")
    ax.text(8.77, 2.3, "PPV forecast", fontsize=7, style="italic", color="#52514e")
    ax.text(6.85, 1.2, "interval\ngate", fontsize=7, style="italic", color="#52514e", ha="center", va="bottom")
    fig.savefig(out / "Figure_1.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / "Figure_1.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_decisions(out: Path, dec: pd.DataFrame):
    import matplotlib.pyplot as plt
    spec = [("Oracle (true PPV) + MCKP", C["gray"], "-", None, "Oracle (100%)"),
            ("KB-PPV + MCKP", C["blue"], "-", "o", "KB-PPV + MCKP"),
            ("KB-PPV + MCKP + interval gate", C["violet"], "--", "s", "KB-PPV + MCKP + interval gate"),
            ("KB-PPV + MCKP, no eligibility rules", C["aqua"], "-.", "^", "MCKP without eligibility rules"),
            ("Gradient boosting + greedy", C["orange"], "--", "D", "Gradient boosting + greedy"),
            ("Largest spend first", "#52514e", ":", "v", "Largest spend first")]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for pol, col, ls, mk, lab in spec:
        s = dec[dec.policy == pol].groupby("fraction")["pct_oracle"].agg(["mean", "std"])
        x = s.index * 100
        if mk is None:
            ax.axhline(100, color=col, lw=1, ls="-")
            continue
        ax.fill_between(x, s["mean"] - s["std"], s["mean"] + s["std"], color=col, alpha=0.15, lw=0)
        ax.plot(x, s["mean"], color=col, ls=ls, marker=mk, ms=4.5, lw=1.8, label=lab,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.text(40.6, 100.6, "Oracle", fontsize=7.5, color=C["gray"], ha="right", va="bottom")
    ax.set_xlabel("Buyer capacity (% of the hours needed to renegotiate every line)", fontsize=9)
    ax.set_ylabel("Realised savings (% of oracle)", fontsize=9)
    ax.set_ylim(50, 104), ax.set_xlim(4, 41)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e4df", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(out / "Figure_2.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / "Figure_2.pdf", bbox_inches="tight")
    plt.close(fig)


def run(args):
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    reps, first = [], None
    for r in range(args.reps if args.data is None else 1):
        df = load_data(args.data, args.n, args.seed + r)
        if r == 0:
            first = df
            if args.data is None:
                df.to_csv(out / "synthetic_po_lines.csv", index=False)
        reps.append(run_replicate(df, args.seed + r))
        print(f"replicate {r + 1} done", flush=True)
    t1, t2, tests, extra, dec = aggregate(reps)

    X, y = design_matrix(first), first["ppv_pct"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    kb = make_models(X.columns, args.seed)[MODELS[4]].fit(Xtr, ytr)
    pi = permutation_importance(kb, Xte, yte, scoring="neg_mean_absolute_error", n_repeats=20, random_state=args.seed)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(5)
    example = explain(kb, Xtr, Xte.iloc[int(np.argmax(kb.predict(Xte)))])

    plot_framework(out)
    plot_decisions(out, dec)
    txt = ("## Table 1. Predictive performance and domain consistency\n\n" + t1.to_markdown()
           + "\n\n## Table 2. Decisions at 20% buyer capacity\n\n" + t2.to_markdown()
           + "\n\n## Wilcoxon signed-rank tests (paired over replicates)\n\n"
           + "\n".join(f"- {k}: p = {v:.4g}" for k, v in tests.items())
           + "\n\n## Decision curve: mean % of oracle by buyer capacity\n\n"
           + dec.pivot_table(index="policy", columns="fraction", values="pct_oracle").loc[POLICIES].round(1).to_markdown()
           + f"\n\n## Other\n\n- CQR 90% coverage {extra['coverage']}, width {extra['width']} pp; "
             f"coverage under volatility shock {extra['shift_coverage']}\n"
           + "- Top permutation importances (MAE increase, pp): "
           + ", ".join(f"{k} {v:.2f}" for k, v in imp.items()) + f"\n- Example explanation: {example}\n")
    (out / "tables.md").write_text(txt)
    print(txt)
    print(f"Artefacts written to {out}/")


# --------------------------------------------------------------------------- #
# Self-tests                                                                  #
# --------------------------------------------------------------------------- #
def selftest():
    rng = np.random.default_rng(0)
    vals = rng.random((7, 3)) * (rng.random((7, 3)) > 0.3)
    cap, best = 30, 0.0
    for combo in itertools.product(range(4), repeat=7):  # brute force
        if sum(HOURS[a - 1] for a in combo if a) <= cap:
            best = max(best, sum(vals[i, a - 1] for i, a in enumerate(combo) if a))
    assert abs(realised(solve_mckp(vals, cap), vals)[0] - best) < 1e-9, "MCKP is not optimal"
    sel = solve_mckp(rng.random((300, 3)), 200)
    assert sum(HOURS[a - 1] for a in sel if a) <= 200, "capacity exceeded"
    df = make_synthetic(1500, 1)
    X, y = design_matrix(df), df.ppv_pct.to_numpy()
    model = make_models(X.columns, 0)[MODELS[4]].fit(X, y)
    assert violation_rate(model, X) == 0.0, "monotone priors violated"
    assert violation_rate(make_models(X.columns, 0)[MODELS[2]].fit(X, y), X) > 0, "violation metric is inert"
    lo, hi = CQR(X.columns, 0.1, 0).fit(X.iloc[:1000], y[:1000]).predict(X.iloc[1000:])
    cov = np.mean((y[1000:] >= lo) & (y[1000:] <= hi))
    assert 0.82 <= cov <= 0.97, f"coverage {cov:.3f} far from 0.90"
    assert (eligibility(df)[:, 0]).all() and eligibility(df, rules=False).all()
    print(f"selftest OK (MCKP optimal, capacity respected, monotone priors hold, CQR coverage {cov:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV of PO lines (see README); default: synthetic benchmark")
    ap.add_argument("--n", type=int, default=4000, help="synthetic sample size per replicate")
    ap.add_argument("--reps", type=int, default=10, help="independent synthetic replicates")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a)
