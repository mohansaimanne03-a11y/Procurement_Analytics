## Table 1. Predictive performance and domain consistency

|                      | MAE (pp)      | RMSE (pp)     | R²             | MAE, volatility shock (pp)   | MAE, n = 300 (pp)   | Prior violations (%)   |
|:---------------------|:--------------|:--------------|:---------------|:-----------------------------|:--------------------|:-----------------------|
| Mean baseline        | 2.724 ± 0.032 | 3.574 ± 0.048 | -0.001 ± 0.001 | 5.98 ± 0.21                  | 2.74 ± 0.04         | 0.0 ± 0.0              |
| Ridge regression     | 1.193 ± 0.017 | 1.782 ± 0.043 | 0.750 ± 0.009  | 2.79 ± 0.12                  | 1.25 ± 0.03         | 0.0 ± 0.0              |
| Random forest        | 1.329 ± 0.017 | 1.926 ± 0.037 | 0.709 ± 0.009  | 4.43 ± 0.14                  | 1.57 ± 0.04         | 0.6 ± 0.3              |
| Gradient boosting    | 1.224 ± 0.014 | 1.813 ± 0.040 | 0.742 ± 0.008  | 4.12 ± 0.12                  | 1.48 ± 0.04         | 0.1 ± 0.1              |
| KB-PPV (monotone GB) | 1.229 ± 0.016 | 1.806 ± 0.041 | 0.744 ± 0.009  | 4.04 ± 0.15                  | 1.45 ± 0.04         | 0.0 ± 0.0              |

## Table 2. Decisions at 20% buyer capacity

| policy                              | Savings (k$)   | % of oracle   | Savings per hour ($)   |   Lines actioned | Ineffective actions (%)   |
|:------------------------------------|:---------------|:--------------|:-----------------------|-----------------:|:--------------------------|
| Random lines                        | 158.0 ± 6.2    | 21.5 ± 0.7    | 33 ± 1                 |              800 | 18.8 ± 0.5                |
| Largest spend first                 | 475.9 ± 31.0   | 64.8 ± 0.9    | 99 ± 6                 |              800 | 23.8 ± 2.2                |
| Gradient boosting + greedy          | 574.7 ± 33.2   | 78.3 ± 1.0    | 120 ± 7                |              800 | 2.8 ± 0.7                 |
| KB-PPV + greedy                     | 575.1 ± 32.9   | 78.3 ± 1.1    | 120 ± 7                |              800 | 3.0 ± 0.7                 |
| KB-PPV + MCKP, no eligibility rules | 641.1 ± 48.3   | 87.3 ± 1.7    | 134 ± 10               |              515 | 7.9 ± 1.2                 |
| Ridge + MCKP                        | 696.9 ± 46.2   | 94.9 ± 0.4    | 145 ± 10               |              538 | 2.2 ± 0.6                 |
| KB-PPV + MCKP                       | 696.4 ± 48.3   | 94.8 ± 0.6    | 145 ± 10               |              536 | 2.6 ± 0.7                 |
| KB-PPV + MCKP + interval gate       | 640.3 ± 49.0   | 87.1 ± 2.2    | 133 ± 10               |              516 | 0.8 ± 0.3                 |
| Oracle (true PPV) + MCKP            | 734.4 ± 48.7   | 100.0 ± 0.0   | 153 ± 10               |              533 | 0.0 ± 0.0                 |

## Wilcoxon signed-rank tests (paired over replicates)

- MAE: KB-PPV vs gradient boosting (CV): p = 0.08398
- MAE: KB-PPV vs gradient boosting (shock): p = 0.009766
- MAE: KB-PPV vs gradient boosting (n=300): p = 0.003906
- Savings: KB-PPV+MCKP vs GB+greedy: p = 0.001953
- Savings: MCKP vs MCKP without eligibility: p = 0.001953
- Savings: MCKP vs MCKP + gate: p = 0.001953
- Ineffective: MCKP vs MCKP + gate: p = 0.001953

## Decision curve: mean % of oracle by buyer capacity

| policy                              |   0.05 |   0.1 |   0.15 |   0.2 |   0.3 |   0.4 |
|:------------------------------------|-------:|------:|-------:|------:|------:|------:|
| Random lines                        |   10.2 |  14.6 |   18.1 |  21.5 |  27.7 |  33.4 |
| Largest spend first                 |   60.2 |  62.6 |   63.8 |  64.8 |  66.2 |  67.2 |
| Gradient boosting + greedy          |   83   |  80.7 |   79.4 |  78.3 |  76.3 |  74.5 |
| KB-PPV + greedy                     |   82.9 |  80.8 |   79.6 |  78.3 |  76.3 |  74.6 |
| KB-PPV + MCKP, no eligibility rules |   87.1 |  87.6 |   87.3 |  87.3 |  87.1 |  87.1 |
| Ridge + MCKP                        |   93   |  94.1 |   94.4 |  94.9 |  95.5 |  95.9 |
| KB-PPV + MCKP                       |   92.8 |  93.7 |   94.1 |  94.8 |  95.2 |  95.7 |
| KB-PPV + MCKP + interval gate       |   87   |  87.1 |   87.3 |  87.1 |  86.8 |  86.3 |
| Oracle (true PPV) + MCKP            |  100   | 100   |  100   | 100   | 100   | 100   |

## Other

- CQR 90% coverage 0.897 ± 0.003, width 5.62 ± 0.06 pp; coverage under volatility shock 0.398 ± 0.022
- Top permutation importances (MAE increase, pp): commodity_volatility 0.79, single_source 0.52, months_since_negotiation 0.34, expedite 0.33, contract_coverage 0.30
- Example explanation: predicted PPV +15.20%; main drivers: commodity_volatility (+6.39 pp), expedite (+3.78 pp), contract_coverage (+3.12 pp)
