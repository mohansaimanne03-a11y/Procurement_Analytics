# KB-PPV: Data-Driven Optimization for Procurement Analytics

Code, data and results for the short communication *"Data-Driven Optimization for Procurement Analytics:
Predicting Purchase Price Variance and Cost-Saving Opportunities"* (submitted to *Knowledge-Based Systems*).

KB-PPV couples (1) a **knowledge layer** (monotone domain priors and action-eligibility rules), (2) a
**monotone gradient-boosting predictor** of purchase price variance (PPV, %), (3) **conformalised quantile
regression** intervals, (4) an exact **multiple-choice knapsack** optimiser that allocates limited buyer hours to
renegotiation, competitive RFQ or price-lock actions, and (5) occlusion-based **explanations** per PO line.

## Reproduce

```bash
pip install -r requirements.txt
python kbppv.py --selftest   # ~10 s: MCKP optimality vs brute force, capacity, monotonicity, CQR coverage
python kbppv.py              # ~12 min on one CPU core: 10 independent replicates, tables, figures
```

Everything is seeded (`--seed 7`); `results/` holds the outputs the paper reports:

| File | Content |
|---|---|
| `results/tables.md` | Tables 1-2, Wilcoxon tests, coverage, example explanation |
| `results/Figure_1.*`, `Figure_2.*` | Framework diagram and savings-vs-capacity curves (PNG 600 dpi + vector PDF) |
| `results/synthetic_po_lines.csv` | Replicate 1 of the synthetic benchmark (4,000 PO lines) |

## Use your own data

```bash
python kbppv.py --data my_po_lines.csv --out my_results
```

One row per PO line with columns `category` (one of the six in `CATEGORIES`), `order_value`, `quantity`,
`lead_time_days`, `commodity_volatility`, `supplier_otd`, `months_since_negotiation`, `num_quotes`, `expedite`,
`single_source`, `contract_coverage`, `npi`, `non_usd` (0/1 flags) and the target `ppv_pct = (actual - standard) / standard * 100`.
Edit `MONOTONE` (driver priors), `ACTIONS` (hours and recovery rates) and `eligibility()` (rules) to match your policies.
Note: with real data the oracle comparison is unavailable (true action outcomes are unobserved); prefer back-testing
against realised savings.

## Scope and honest limits

The benchmark is synthetic with a known data-generating process, so decision quality can be scored against an
oracle; it demonstrates the method, not field performance. The savings model (recovered share per action) is an
assumption. Conformal coverage holds under exchangeability and degrades under severe market shocks (see paper).

## Licence

MIT. Cite the paper if you use the code.
