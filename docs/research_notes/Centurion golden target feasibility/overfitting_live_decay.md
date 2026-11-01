# How backtests decay live and what anti-overfitting practice actually works (Centurion, Indian swing/positional)

Evidence grades: [LIVE] real-money or live-product record. [ACAD-OOS] academic with out-of-sample test. [BT] backtest or simulation only. [OPINION] practitioner or author judgement. [CALC] my own closed-form hand calculation (no code was run, per the brief). Provenance flags: (secondary) means read through a summary, not the paper; (excerpt) means seen only in a search excerpt.

Verification limit: most primary PDFs could not be read here (the fetch tool returned undecoded binary), so many numbers come through secondary summaries by CXO Advisory, David Bailey's blog, or trade press. The shared web-search budget also ran out mid-task, so several named sources were never retrieved. They are listed under Gaps and are not cited as findings.

## How much do backtests decay live, and what is a defensible haircut by number of trials?

### Takeaway
Roughly half of a backtest Sharpe is lost in the disciplined case and about three quarters in the marketed-product case, with a best case of 20-33% for liquid, published, simple signals. Centurion's own 1.16 to 0.60 drop (48%) is ordinary. The haircut is nonlinear in the t-statistic and N, so a flat percentage is the wrong tool. The defensible central live expectation from the walk-forward is Sharpe about 0.4-0.5.

### Cited Findings
- [LIVE] Suhonen, Lennkh, Perez (JPM 2017): 215 alternative-beta strategies from 15 banks, 5 asset classes, 11 strategy types. Median Sharpe 1.20 backtest versus 0.31 live, a 73% deterioration. Average backtest 10.7 years, average live 4.6 years. Only 18 of 215 matched or beat their backtest; 65 of 215 had a negative live Sharpe. Targeted alphas fell by about half live. Deterioration was at least 50% in every asset class. The commentary says the 50% derating practice "is not conservative enough" — [CXO Advisory](https://www.cxoadvisory.com/big-ideas/live-performance-of-alternative-beta-products/) (secondary).
- [LIVE] Same study, other cuts: median haircut 72% (Bailey's rounding), 80% for equity strategies, 62% on a 3-year pre/post window excluding 2008-09, and about 3,000 strategies in the original database reduced to 215 unique ones. The most complex strategies had a haircut 30 percentage points larger than the simplest — [Bailey, Mathematical Investor](https://mathinvestor.org/2017/08/backtest-overfitting-in-smart-beta-investments/) (secondary). CXO words the complexity effect as "about 30% stronger", a small wording conflict.
- [ACAD-OOS, forward simulation rather than real capital] Wiecki et al 2016, 888 Quantopian algorithms with at least 6 months out of sample, out-of-sample period June 2015 to February 2016. Backtest Sharpe predicted out-of-sample Sharpe with R² below 0.025 — [Quantopian archive](https://quantopian-archive.netlify.app/forum/threads/q-paper-all-that-glitters-is-not-gold-comparing-backtest-and-out-of-sample-performance-on-a-large-cohort-of-trading-algorithms).
- [ACAD-OOS] Same study: most individual metrics had R² 0.01-0.02. Annual return had a slightly negative in-sample/out-of-sample correlation. Information ratio, alpha and Calmar had no predictability. Risk metrics did far better: volatility R² 0.67, max drawdown R² 0.34. More backtest days used meant a larger in-sample/out-of-sample gap. ML on backtest features reached R² 0.17 on hold-out, and its top-10 portfolio made 1.8 net out of sample versus 0.7 for the top-10 by in-sample Sharpe — [CXO Advisory](https://www.cxoadvisory.com/big-ideas/in-sample-vs-out-of-sample-performance-of-888-trading-strategies) (secondary).
- [ACAD-OOS] Falck, Rej, Thesmar (Quantitative Finance 2022): 72 published anomalies. Mean in-sample Sharpe 0.98 (1.15 for the 60 with in-sample Sharpe above 0.3). The out-of-sample-on-in-sample slope is 0.57, about 43% decay; median discount ratio 0.55. In the most liquid US pools decay was only 33% (top 1,000) and 20% (top 500). International pools: raw discount ratio 0.11, about 0.41 after a simple size adjustment. Year of publication alone gives R² 0.30, with decay about 5 percentage points worse per later publication year — [Falck et al](https://ar5iv.labs.arxiv.org/html/2105.01380).
- [ACAD-OOS] McLean and Pontiff (JF 2016), 97 predictors: 58% post-publication decay as cited by [Falck et al](https://ar5iv.labs.arxiv.org/html/2105.01380). A working-paper version covers 95 characteristics, with an upper-bound decline from statistical bias of 25% and a post-publication decline of 56% — [Hedge Fund Alpha repost](https://hedgefundalpha.com/?p=521337). Earlier drafts report about 35% and about 32% — [MoneySense](https://moneysense.ca/?p=36813), [Counterpoint copy](https://counterpointfunds.com/wp-content/uploads/2017/07/PredictabilityMcleanPontiff.pdf) (both excerpts). The brief's 26% was not confirmed from the published text; the working paper's 25% is close, and the split matters: only the first number estimates pure data mining, the rest is attributed to informed trading.
- [LIVE] Song, Huang, Xiang, "The Smart Beta Mirage" (JFQA 2023): CAPM alpha of 2.77% a year (t = 4.59) on paper over an average 13 years before ETF listing, then -0.44% a year over an average 6 years after listing. The authors say variation in factor premia, strategic timing and diminishing returns to scale do not explain the decline — [JFQA paper](https://resolve.cambridge.org/core/services/aop-cambridge-core/content/view/35CA1CEA091A485DDA960BB2D9930BC9/S0022109023000674a.pdf/smart_beta_mirage.pdf) (excerpt). The authors attribute it to "data mining in backtests" — [InvestmentNews](https://www.investmentnews.com/industry-news/news/smart-beta-funds-dont-live-up-backtesting-196778).
- [LIVE] Vanguard (Dickson et al, "Joined at the Hip", 2012): new index funds beat the broad US market by 10.29% annualized in the 60 months before launch and trailed it by 0.99% annualized in the 60 months after — [Bailey](https://mathinvestor.org/?p=1583). Trade-press excerpts report 10.31% and -0.93% on 370 indices, about half of 1,000-plus indices under six months old at ETF launch, and 87% outperforming in back-tested periods versus 51% after launch. Those pieces returned 403 on direct fetch, so the exact page for each number is unconfirmed: treat as excerpts from [ThinkAdvisor](https://www.thinkadvisor.com/2012/07/23/etf-buyers-beware-of-back-filled-indexes/), [BenefitsPro](https://benefitspro.com/2012/07/23/hypothetical-performance-data-skews-future-etf-per) and [PlanAdviser](https://www.planadviser.com/vanguard-principal-advises-against-rearview-mirror-investing/). Note the 10.29/-0.99 versus 10.31/-0.93 conflict; I could not read the Vanguard PDF to settle it.
- [LIVE] Brightman, Li, Liu: ETF providers predominantly launch funds whose indices are outperforming, and strategies with strong prior three-year excess returns then behave like an average investor's portfolio after launch — [Hedge Fund Alpha repost](https://hedgefundalpha.com/?p=1063516) (partial, paywalled). No magnitudes were retrievable.
- [LIVE] Li and West 2017 (Research Affiliates): alpha is significantly higher in backtest than in the live record, and does not survive over live index histories — [RA paper](https://originpreprod.researchaffiliates.com/content/dam/ra/publications/pdf/626-live-from-newport-beach-its-smart-beta.pdf) (excerpt).
- [LIVE] Chang Liu, arXiv April 2026: 1,726 commercially distributed strategies from ten institutions. Pro-forma performance transfers only partly into live trading, and much less against peer benchmarks. Backtests mainly reflect the common factor regime before launch, and launches after strong factor conditions deteriorate more — [arXiv abstract](https://arxiv.org/abs/2604.18821). Preprint, abstract only.
- [OPINION, method] Harvey and Liu, "Backtesting" (JPM 2015): the 50% rule is "a serious mistake". The haircut is nonlinear: the highest Sharpes are penalized moderately and marginal ones heavily. Worked example: a single-test t of 3.0 falls to about 2.0 after multiple testing — [Man Group summary](https://www.man.com/insights/backtesting). The exact table was not retrievable.
- [BT, theorem] Bailey et al, "Pseudo-Mathematics" (Notices AMS 2014): with 5 years of daily data, no more than 45 variations should be tried, or a skill-less strategy can be expected to show an in-sample Sharpe of 1. The authors call MinBTL necessary but not sufficient — [Bailey slides](https://www.carmamaths.org/meetings/mabef/talks/mabef-David_Bailey-1.pdf), [Bailey et al](https://carmamaths.org/jon/backtest.pdf) (excerpts).

### Inferences
- [CALC] Best-of-N noise. Using the False Strategy Theorem form E[max z] = (1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(Ne)), a skill-less Sharpe from N independent trials over T years is about z/√T. This reproduces Bailey's rule: z² is about 4.9 years at N = 45.

| N (independent) | E[max z] | Noise Sharpe, 13y | Noise Sharpe, 18y | MinBTL for Sharpe 1.0 (years) | MinBTL for Sharpe 0.6 (years) |
|---|---|---|---|---|---|
| 10 | 1.58 | 0.44 | 0.37 | 2.5 | 6.9 |
| 30 | 2.07 | 0.57 | 0.49 | 4.3 | 11.9 |
| 61 | 2.35 | 0.65 | 0.55 | 5.5 | 15.3 |
| 100 | 2.53 | 0.70 | 0.60 | 6.4 | 17.8 |
| 200 | 2.76 | 0.77 | 0.65 | 7.6 | 21.2 |
| 1,000 | 3.25 | 0.90 | 0.77 | 10.6 | 29.3 |
| 7,000 | 3.77 | 1.05 | 0.89 | 14.2 | 39.5 |

- The sharp read: 18 years at Sharpe 0.60 matches the expected best of about 100 independent skill-less trials. So the walk-forward result is only meaningful if effective N is well under 100, or if the walk-forward was genuinely free of configuration selection.
- [CALC] Bonferroni haircut, illustrating nonlinearity. For in-sample Sharpe 1.16 over 13 years (t = 4.18): N = 10 gives 14%, 61 gives 25%, 100 gives 29%, 200 gives 34%, 1,000 gives 48%, 7,000 gives 70%. For walk-forward Sharpe 0.60 over 18 years (t = 2.55): N = 5 gives 24%, 10 gives 37%, 20 gives 51%, 61 gives 83%, and about 93 or more gives 100%. Bonferroni is conservative for correlated trials and treats the walk-forward as a pure selection, so these are upper bounds.
- [OPINION] Rule-of-thumb Sharpe haircut. Mapping N to these settings is my judgement, anchored on the literature in the last column.

| Research setting | Effective N | Sharpe haircut | Anchor |
|---|---|---|---|
| Single pre-stated hypothesis, liquid universe, simple rule | 1-5 | 20-35% | Falck liquid pools 20-33% |
| Disciplined family with an economic prior | 10-50 | 40-50% | Falck 43-45% |
| Broad search with a registry | 100-500 | 50-65% | McLean-Pontiff 58%, Suhonen 3-year cut 62% |
| Heavy search, complex model, or marketing incentive | 1,000+ | 70-85%, or treat as zero until proven | Suhonen 73%, equities 80%, Wiecki R² near 0 |

- [CALC] CAGR rule. Apply the haircut to excess return, not total CAGR: CAGR_live is about rf + (1-h)(CAGR_bt - rf) at constant volatility. For a long-only book, haircut only the alpha, since market beta survives. Illustrative rf of 6.5% is my assumption; replace it with the scorecard's rf. Centurion's raw CAGR falls 44% (23.1 to 13.0), but excess falls about 61% (16.6 to 6.5 points), while Sharpe falls 48%.
- Centurion's walk-forward adds 2008-12, a period the in-sample lacks, so 1.16 to 0.60 mixes decay with a regime change. It is not a clean haircut.
- [OPINION] Forward estimate. A further 20-35% from 0.60 gives live Sharpe about 0.39-0.48, and CAGR about 10.7-11.7% on the illustrative rf. Project memory says live-like metrics are "WF + D2 haircut". If 0.60 already includes D2, do not double-charge the implementation part; the residual selection part still applies. This needs checking by someone allowed to read the scorecard.
- [OPINION] Keep 0.31 from Suhonen as the stress case, not the base case: those products had a sales incentive and non-public trial counts.

### Gaps
- Huij and Kyosev on simulated ETF histories: not found in one search; the second search was blocked. No claim made.
- Harvey and Liu's numerical haircut table and "Lucky Factors": not retrieved. My Bonferroni figures stand in.
- Brightman-Li-Liu magnitudes: paywalled.
- McLean-Pontiff published 26% and 97-predictor abstract not read directly; the 58% comes through Falck's citation, and the 25%/56% through a working-paper repost.
- Live decay evidence for Indian systematic strategies: none found. The Nifty200 Momentum 30 ETFs are the nearest proxy (see the execution section) and their records are too short to compute a haircut.

## Which tools work and what are their limits?

### Takeaway
No tool certifies a strategy. They only reject bad ones or bound selection bias, and all of them depend on an honest trial count and a fixed trial set. CPCV beats walk-forward as a research filter in a synthetic comparison, while walk-forward remains the realistic simulation of how you would trade, so run both for different jobs. PBO and DSR are diagnostics, not gates to optimise against.

### Cited Findings
- [BT, synthetic] Arian, Norouzi, Seco (Knowledge-Based Systems 2024, DOI 10.1016/j.knosys.2024.112477): CPCV outperforms K-fold, purged K-fold and especially walk-forward, with lower PBO and a better DSR test statistic. Walk-forward shows notable shortcomings in false discovery prevention, with greater temporal variability and weaker stationarity. The paper introduces Bagged CPCV and Adaptive CPCV — [SSRN copy on Scribd](https://www.scribd.com/document/725401650/SSRN-id4778909) (excerpt).
- [OPINION] A 2025 paper notes walk-forward remains the industry standard for realistic trading despite CPCV's better showing — [arXiv 2512.12924](https://arxiv.org/pdf/2512.12924) (excerpt).
- [OPINION] López de Prado on walk-forward: repeating the fit-backtest loop until a good result appears is overfitting, and it does not matter if the backtest is a walk-forward out-of-sample — [Oxford Law blog](https://blogs.law.ox.ac.uk/business-law-blog/blog/2018/03/machine-learning-funds-and-investment-malpractice) (excerpt).
- [OPINION, method] Purging removes training observations whose labels overlap test-set labels; K-fold fails on non-IID financial data because of leakage — [Quantpedia on López de Prado](https://quantpedia.com/why-machine-learning-funds-fail/). The embargo is sized to the longest feature lookback, and walk-forward test windows are not fully independent, so should be read qualitatively rather than pooled — [MQL5 article](https://www.mql5.com/en/articles/21603) (practitioner).
- [BT, theorem] DSR (Bailey and López de Prado 2014) adjusts Sharpe for selection across trials and non-normality. The False Strategy Theorem gives the expected maximum Sharpe of unskilled trials, which depends on N and the cross-sectional variance of trial Sharpes. Effective N comes from clustering, with ONC (López de Prado and Lewis 2019) or hierarchical clustering on the return correlation matrix — [Wikipedia, Deflated Sharpe ratio](https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio) (excerpt; page returned 404 on fetch).
- [BT] PBO via CSCV: estimates how likely a backtest-selected strategy is a poor out-of-sample performer, shown to give reasonable estimates on several examples — [Bailey et al](https://carmamaths.org/jon/backtest2.pdf) (excerpt). That is a demonstration on examples, not a general validation.
- [OPINION] PBO limits: "CSCV does not fix overfitting"; a high PBO says the selection process is unreliable, not how to fix it; thresholds are informal rules of thumb — [MQL5](https://www.mql5.com/en/articles/21603). There is no universal threshold, and CSCV's reshuffling weakens temporal order, so it complements forward confirmation — [LuxAlgo](https://www.luxalgo.com/library/concept/probability-of-backtest-overfitting/) (vendor, low authority).
- [ACAD-OOS, method] White's Reality Check loses power, potentially toward zero, as poor and irrelevant rules are added (Hansen 2005). SPA is more powerful and less sensitive to them. Romano-Wolf Step-RC stays conservative, and Hsu, Hsu, Kuan's stepwise SPA is more powerful — [Hsu et al 2010](https://repository.hku.hk/handle/10722/141768) (excerpt).
- [ACAD-OOS] Sullivan, Timmermann, White: technical rules lost predictive power for major US indices after the mid-1980s; Hsu et al find the evidence weakens after ETFs are introduced — [Springer 2023](https://link.springer.com/10.1007/s11408-023-00433-2), [Hsu et al](https://repository.hku.hk/handle/10722/141768) (excerpts).
- [ACAD-OOS] Bajgrowicz and Scaillet (JFE 2012), DJIA 1897-2011: FDR selects more rules than family-wise tests, allowing diversification against model uncertainty, but an investor could never have picked the future best rules ex ante, and low transaction costs wiped out in-sample performance — [EconPapers listing](https://econpapers.repec.org/RePEc:eee:jfinec:v:106:y:2012:i:3:p:473-491) (excerpt).
- [ACAD-OOS] Harvey, Liu, Zhu (RFS 2016): a new factor needs t above 3.0 — [NBER](https://www.nber.org/papers/w20592). Chen: a t-hurdle near 3.0 already implies FDR of 5% or less, and most claimed findings are likely true — [arXiv 2206.15365](https://arxiv.org/pdf/2206.15365) (excerpt). Raised hurdles are hard to justify because failed results are unobserved, so they are weakly identified, while empirical Bayes and FDR are strongly identified — [Chen arXiv abstract](https://arxiv.org/abs/2204.10275v4).
- [ACAD-OOS] Jensen, Kelly, Pedersen (JF 2023): most factors replicate, cluster into 13 themes, work out of sample across 93 countries, and evidence is strengthened by the number of factors observed — [NBER](https://www.nber.org/papers/w28432).
- [ACAD-OOS] Falck: the drop-data outlier sensitivity of a factor's in-sample Sharpe predicts decay (R² 0.14), and the in-sample t-stat dummy does not (R² 0.01) — [Falck et al](https://ar5iv.labs.arxiv.org/html/2105.01380). This is the closest retrieved evidence that fragility tests work.

### Inferences
| Tool | Job | Main limit | Use at Centurion |
|---|---|---|---|
| Walk-forward | One realistic path, operational realism | Single path; selection on it re-contaminates it | Reporting and live-like estimate |
| CPCV | Distribution of paths for model research | Synthetic evidence only; needs purge and embargo sized to holding period and lookback | Research filter |
| DSR/PSR | Bound selection bias on one Sharpe | N and trial variance are inputs you choose | Report a range across N, not a point |
| PBO/CSCV | Is the selection process stable? | Depends on trial set; shuffles time | Process diagnostic on the actual selection metric |
| RC, SPA, Romano-Wolf | Family-wise test against a benchmark | RC dilutes with junk rules | SPA preferred if one is run |
| FDR, empirical Bayes | What share of the registry is false? | Assumes reasonable dependence handling | Well suited: a private registry sees failed trials, which removes Chen's identification problem |
| Fragility and plateau tests | Is the result one outlier or one parameter? | Thresholds informal | Cheap and supported by Falck |

- A private registry is better placed than academia: Chen's objection (unobserved failures) does not apply to Centurion, so a registry-level FDR estimate is feasible.
- [OPINION] PBO is gameable. Adding clearly bad trials should lower PBO because the selected strategy then ranks high out of sample against junk, the same dilution mechanism that hurts White's Reality Check. Fix the trial set before computing it.
- [OPINION] Use fragility tests as vetoes. They have evidence behind them and cost little.

### Gaps
- Exact Arian numbers: seen only in search excerpts; the publisher page and a bohrium listing did not load, so no PBO or DSR values are available and the direction of the finding is all that can be cited.
- Monte Carlo permutation tests, synthetic data, block bootstrap and direct parameter-plateau studies: no source retrieved, so no finding. These are not evidence of failure, just unresearched here.
- Peer-reviewed critique of CSCV under autocorrelation or sensitivity to partition count: none found.
- PSR's 0.95 convention was seen only in passing; I did not verify the primary statement.

## How should trial counts be estimated honestly, and what do Centurion's numbers imply?

### Takeaway
Effective N should come from clustering all logged runs, and should be reported as a range with the registry as a floor, because design choices made from memory are never logged. PBO 29.2% and DSR 0.939 are acceptable but not strong: DSR misses 0.95 at N = 61, and the verdict is sensitive to N. The 51.2% raw PBO means Centurion has no evidence of selection skill on raw returns.

### Cited Findings
- [BT, method] Trials overlap, so effective N is estimated by clustering. ONC is a modified K-means producing subgroups with high intra-cluster and low inter-cluster correlation, and its output supplies both effective N and the variance of Sharpes across clusters — [Wikipedia, Deflated Sharpe ratio](https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio) (excerpt).
- [ACAD-OOS] More backtesting produces a larger in-sample/out-of-sample gap — [Quantopian archive](https://quantopian-archive.netlify.app/forum/threads/q-paper-all-that-glitters-is-not-gold-comparing-backtest-and-out-of-sample-performance-on-a-large-cohort-of-trading-algorithms).
- [OPINION] Arnott, Harvey, Markowitz: count how many variables and combinations were tried, and check whether the data can justify added complexity — [Alpha Architect summary](https://hedgefundalpha.com/news/a-protocol-to-prevent-quants-gone-wild/).
- [ACAD-OOS, method] Lo 2002: the IID Sharpe standard error is the square root of (1 + SR²/2)/T in per-period units, and positive serial correlation can overstate annualized Sharpe, by more than 65% in some hedge fund cases — [Lo](https://traders.studentorg.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf), [CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios) (excerpts).

### Inferences
- [OPINION] Best practice for 7,000+ runs:
  1. Log every run, including abandoned and failed ones.
  2. Cluster on return correlations (ONC or hierarchical), using the same frequency for all trials.
  3. Use cluster-representative Sharpes for the trial variance, not raw run Sharpes.
  4. Count a new trial whenever someone re-runs on the same data after seeing results.
  5. Add implicit trials: universe choices, cost-model changes, gate thresholds and features picked from prior knowledge of what worked in India. These are selection too, and correlation clustering cannot see them.
  6. Report DSR at a low, central and high N.
- [CALC] Sensitivity of DSR 0.939 (z = 1.546). The threshold SR0 scales with E[max z] relative to N = 61: +17% at N = 200, +38% at N = 1,000, +60% at N = 7,000. I could not read which Sharpe and SE entered the DSR, so two cases are shown with SE = 1/√T, ignoring AC widening and higher moments.

| N | Case A: Sharpe 1.16, 13y | Case B: Sharpe 0.60, 18y |
|---|---|---|
| 30 | 0.97 | 0.95 |
| 61 | 0.939 (given) | 0.939 (given) |
| 100 | 0.91 | 0.93 |
| 200 | 0.86 | 0.915 |
| 1,000 | 0.70 | 0.88 |
| 7,000 | 0.48 | 0.83 |

- In Case A, DSR crosses 0.95 at about N = 47. That is a change of only about 14 effective trials, so 0.939 versus 0.95 should not be read as pass or fail.
- [CALC, formula from memory, unverified] If N_eff is about ρ̄ + (1-ρ̄)M, then N = 61 from M = 7,000 implies an average pairwise correlation of about 0.991. That is plausible within one parameter sweep and implausible across distinct strategy families. An average of 0.95 would give N about 351, and 0.90 about 701. Check that 61 reflects clusters across families, and that M counts everything.
- PBO 29.2%: in about 7 of 10 combinatorial splits the in-sample best stayed above the out-of-sample median. It is not a 71% chance the strategy is real.
- PBO 51.2% on raw: selecting on raw returns is a coin flip, which fits a common market factor dominating every trial's ranks. Only the excess-return claim is supported. PBO must match the selection metric actually used; if configurations were chosen on raw CAGR or Sharpe, 51.2% is the relevant number, not 29.2%. Which benchmark defines "excess" also matters, since a beta mismatch leaks into the ranks.
- Overall: consistent with a real but modest edge, fully consistent with live Sharpe about 0.4-0.5, and not proof of anything above that.

### Gaps
- Bailey and López de Prado's exact effective-N formula: unverified, from memory.
- How the registry's N = 61 and the AC widening were computed: not read, as centurion_core is off limits.
- No published standard for counting implicit (undocumented human) trials was found.

## Does 60-90 sessions of paper trading have statistical power, and what are better forward tests?

### Takeaway
No. Separating Sharpe 1.0 from 0.5 takes about 6,200 sessions (25 years). A 60-session test has about 8% power at a 5% false-positive rate, so it is an operations and behaviour check only. Real forward evidence comes from measures with high signal, such as fills, slippage, volatility and turnover, on staged real capital, with sequential monitoring for failure.

### Cited Findings
- [ACAD-OOS, method] Lo's standard error — [Lo 2002](https://traders.studentorg.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf).
- [BT, method] Philips, Yashchin, Stein (JPM 2003) CUSUM on information ratio, good 0.5 versus bad 0: about 84 months between false alarms and about 41 months to detect the switch — [R implementation docs](https://rdrr.io/github/chindhanai/GSoC2017/src/R/cusumActMgr.R). The abstract cites about 40 months and describes the method as provably optimal for a given false-alarm rate — [IBM listing](https://researcher.ibm.com/publications/using-statistical-process-control-to-monitor-active-managers) (excerpt).
- [BT, theory] Rej, Seager, Bouchaud: for a 10-year process, the 5% drawdown length is about 2.14/SR² and the 5% depth about 1.50/SR in annual-volatility units. At Sharpe 0.5 there is a 5% chance of a drawdown of 7 years or more. A breach of the 5% bound can mean bad luck, an overestimated Sharpe, or fat tails, and the test cannot separate the first two — [Rej et al](https://ar5iv.arxiv.org/html/1707.01457).
- [ACAD-OOS] Volatility and drawdown persist far better than return metrics: R² 0.67 and 0.34 versus 0.01-0.02 — [CXO on Wiecki](https://www.cxoadvisory.com/big-ideas/in-sample-vs-out-of-sample-performance-of-888-trading-strategies).
- [BT] Carver, random-data simulation: adding a moving-average filter to a profitable equity curve generally lowers return and return/drawdown, and faster filters do worse. It helps only with positive autocorrelation and low costs; with 0.03 SR units a year of cost he suggests autocorrelation of at least 0.2 before considering it — [qoppac](https://qoppac.blogspot.com/2015/11/random-data-evaluating-trading-equity.html).
- [OPINION] Arnott et al: don't tweak a model based on live results — [Alpha Architect summary](https://hedgefundalpha.com/news/a-protocol-to-prevent-quants-gone-wild/).

### Inferences
- [CALC] With daily data, the per-period SR²/2 term is negligible, so the SE of annualized Sharpe is about 1/√T. Sessions needed = 252 × ((z_α + z_β)/ΔSR)².

| Test | One-sided 5%, power 80% | One-sided 20%, power 80% |
|---|---|---|
| Sharpe 1.0 vs 0.5 | 24.7y, about 6,235 sessions | 11.3y, about 2,860 |
| Sharpe 1.0 vs 0 | 6.2y, about 1,560 | 2.8y, about 715 |
| Sharpe 0.6 vs 0 | 17.2y, about 4,330 | 7.9y, about 1,985 |
| Sharpe 0.6 vs 0.3 | 68.7y, about 17,320 | 31.5y, about 7,940 |

- [CALC] Short-window precision: SE of annualized Sharpe is 2.9 at 30 sessions, 2.05 at 60, 1.67 at 90. Testing 0.5 against 1.0 at 5% false positive: power about 8.1% at 60 sessions and 8.9% at 90. Probability of a losing 60-session window: 31% at Sharpe 1.0, 38% at 0.6, 50% at zero. Positive autocorrelation shrinks effective sample further, by about (1-ρ)/(1+ρ).
- [CALC] Bayesian update, prior Sharpe mean 0.5, SD 0.3: weight on the data is 2% at 60 sessions, 8% at one year, 31% at five years. After a full year of live results, the prior still carries 92%, so the prior must be stated honestly.
- [CALC] SPRT, Wald approximations. Sharpe 1.0 vs 0.5 at α 5%, β 20%: expected about 15 years if 1.0, 11 years if 0.5. Sharpe 0.6 vs 0: about 10.6 and 7.5 years; at α = β = 20%, about 4.6 years. Sequential testing cuts expected time by roughly 40-57% against the fixed-sample design, but cannot turn years into weeks.
- [CALC] CUSUM scaling: detection time scales about 1/Δ², so a fall from 0.6 to 0 should take about 28 months (41 × (0.5/0.6)²) at Philips' 7-year false-alarm rate. Drawdown bounds from Rej: at Sharpe 0.6, length about 5.9 years and depth about 2.5 annual vols; these are too slow and too wide to act as short-window tests.
- [CALC] Tests with real power in 60 sessions: realized volatility has a sampling error of about 9-14% (fat tails at the upper end). Slippage has SE = s/√n, so with an illustrative s of 60 bp and 100 fills, SE is 6 bp and a 15 bp model error shows clearly. At an illustrative 10 full-book round trips a year, 15 bp per side is 3% a year, around half the assumed excess return. Slippage matters more than any Sharpe test, and paper fills cannot reveal it.
- [OPINION] Better forward ladder, in rising order of realism:
  1. Dry runs and paper: operations only. Reconcile signals, orders, positions, corporate actions.
  2. Shadow: backtest engine re-run on the same days as live; target is zero unexplained position difference.
  3. Canary capital: real orders at the smallest size the whole-share constraint permits. This is the first test that measures fills, rejects and GTT behaviour.
  4. Ramp on process evidence, not P&L.
- [OPINION] Ramp proposal, no direct published evidence: canary 5-10% for 3-6 months; 25% for 6-12 months; 50% to 24 months; full after that. Advance only if slippage is within the model's bound, volatility ratio is 0.8-1.25, turnover and holding period match, and reconciliation has no breaks. Do not advance or retreat on P&L in the first year. Scale down on a CUSUM alarm or a Rej 5% breach, not on a short loss. Treat the Carver result as a warning against equity-curve overlays.
- The 60-session and 5-dry-run gates are worth keeping, relabelled "operational readiness". They should not be read as confirming a Sharpe. Passing them adds almost no evidence on edge.
- [OPINION] Add one gate the current chain lacks: a canary stage with slippage measurement between paper and full capital.

### Gaps
- No empirical study found comparing canary capital against paper trading, or setting capital ramp schedules.
- Evans on mutual fund incubation, directly relevant, could not be searched.
- Practitioner sequential-test case studies on Sharpe were not found. Carver's own view on years needed was not retrieved beyond comments that long backtests are needed.

## What is the execution reality for live swing systems in India?

### Takeaway
GTT is not a guaranteed stop: it lives on the broker's side until triggered, then sends a limit order that a gap can skip, and an unfilled order simply dies at session end. Modelled stop fills are therefore optimistic for gap events. Impact is small at retail sizes in large caps but climbs in steps down the market-cap ladder, and current small-cap impact data was not found.

### Cited Findings
- [LIVE, broker terms] Zerodha GTT: no exchange order exists until the trigger is met; Zerodha stores the condition and sends a limit order at trigger. Triggers track LTP. A gap through the trigger may fill at the limit or leave the order unfilled. "This GTT order does not assure execution". Unfilled limit orders from a breached trigger are cancelled at the end of that session; triggers are one-time and must be re-placed. Margin and holdings are checked only at trigger time. Sell orders can be rejected without eDIS/POA/DDPI. Trigger cut-offs are 3:15 PM for F&O underlyings and 3:30 PM for other equities. Triggers must be at least 0.25% from market for stocks above Rs. 50. Zerodha bears no liability for non-execution — [Zerodha GTT terms](https://zerodha.com/tos/gtt).
- [LIVE, broker terms] Corporate actions: Zerodha may cancel a GTT at its discretion on the ex-date with a day's notice, and trailing stop-losses are cancelled automatically — [Zerodha GTT terms](https://zerodha.com/tos/gtt).
- [LIVE, broker doc] Zerodha's example: a stock opening at Rs 90 against a trigger of Rs 95 and limit of Rs 93 places a Rs 93 sell, which fills only if price recovers. Reasons for triggered-but-unfilled: limit mismatch, missing authorisation, insufficient funds, no volume at the limit, or a circuit-band breach — [Zerodha support](https://support.zerodha.com/category/trading-and-markets/gtt/articles/why-did-my-gtt-order-trigger-but-was-not-executed), [GTT article](https://support.zerodha.com/category/trading-and-markets/gtt/articles/what-is-the-good-till-triggered-gtt-feature) (excerpt).
- [LIVE, anecdote] A user reported a GTT stop trigger hit with the exit not completed, forcing a manual square-off, on a two-leg F&O GTT, so transfer to equity is limited — [Kite forum](https://kite.trade/forum/discussion/15838/gtt-stop-loss-not-executing-correctly-drreddy26febfut-trigger-hit-but-exit-not-completed).
- [LIVE, dated] Impact cost, March 2019: Nifty 50 weighted average 0.02% at Rs 50 lakh, Nifty Next 50 0.04% at Rs 25 lakh. Impact rises in steps as free-float cap falls, and most Next 50 stocks have at least twice Nifty 50 impact. The author warns midcap and small-cap funds share the liquidity risk — [freefincal](https://freefincal.com/?p=28799).
- [LIVE, very dated] Nifty impact cost fell from 0.12% in 2002 to 0.08% in 2005 at Rs 5 million, and Nifty Junior from 0.41% to 0.16% at Rs 2.5 million — [Economic Survey 2005-06](https://indiabudget.gov.in/budget_archive/es2005-06/chapt2006/chap43.pdf) (excerpt).
- [OPINION, low authority] Glossary claims of "several percentage points" for mid and small caps at equivalent order sizes, illustrative only, with no data behind them — [equitiesindia](https://equitiesindia.com/glossary/impact-cost) (excerpt).
- [ACAD-OOS] Lalwani and Meshram used a survivorship-bias-free Indian sample, 1994-2019 — [EconPapers](https://econpapers.repec.org/RePEc:taf:applec:v:54:y:2022:i:16:p:1814-1828). It shows the standard is achievable for India.
- [LIVE] Nifty200 Momentum 30, launched 25 August 2020 with backtested history from 2005 — [Substack](https://curiousinvestinginsights.substack.com/p/momentum-or-mayhem-unpacking-the). NSE whitepaper reports 18.6% CAGR against 12.7% for Nifty 200, with volatility 23.4% versus 22.3% — [NSE whitepaper](https://www.niftyindices.com/docs/default-source/indices/nifty200-momentum-30/nifty200_momentum_30_index_whitepaper_sep_20.pdf) (excerpt). A correction to March 2025 saw about -27% versus -12% for Nifty 50 — [Substack](https://curiousinvestinginsights.substack.com/p/momentum-or-mayhem-unpacking-the). All pre-launch history is backtested — [freefincal](https://freefincal.com/nifty200-momentum-30-index-a-new-strategy-index-from-the-nse/).

### Inferences
- [OPINION] GTT model for backtests: for a stop-out on a gap, assume the fill is the open or worse, and allow a defined rate of non-fill. Use a wider sell limit below trigger to fill more often at a worse price. No Zerodha market-order fallback was seen for GTT: treat that as unverified. Also add an end-of-day check for fired-but-unfilled GTTs and a corporate-action check, since both silently remove protection. This is a real source of unmodelled tail loss for a stop-reliant book.
- [CALC] Whole-share rounding: rounding error per position is up to ±P/(2V) for price P and position value V. At V = Rs 25,000, a Rs 5,000 stock has ±10% size error and stocks above V are unbuyable. Live weights therefore drift from the backtest, and high-priced names are dropped. The backtest should run with the same whole-share rule at the actual capital.
- [OPINION] At small capital Centurion's orders are far below book depth for liquid names, so spread and open-auction price matter more than impact. Order-size impact becomes the issue only in small caps or as capital scales.
- Canary-stage fill data is the only way to get Centurion-specific slippage, since no published Indian mid and small-cap table was found.

### Gaps
- AMO and pre-open call-auction fill quality, timing and rules: no source retrieved. No claim made.
- Current Indian mid and small-cap impact or slippage data: not found. NSE publishes monthly impact cost; not retrieved.
- Size of survivorship bias in Indian backtests, vendor error rates, Kite historical data handling of delisted names, and partial-fill statistics: nothing retrieved.
- Crowding or capacity data for Indian momentum funds.

## Which live failure modes are documented, and which checks catch which?

### Takeaway
The documented failures fall into selection, regime, costs, data and operations. No single check covers them. Pre-launch regime tailwind is the least-appreciated one: it survives every statistical test because it is in the data.

### Cited Findings
- [OPINION] López de Prado's pitfalls include the Sisyphus paradigm, integer differentiation, inefficient sampling, wrong labelling, non-IID sample weighting, CV leakage (remedy purging), and backtest overfitting (remedy DSR). Quantpedia's version lists 7; the final paper lists 10, and the rest were not retrievable — [Quantpedia](https://quantpedia.com/why-machine-learning-funds-fail/). He recommends de-emphasising backtesting and testing across many train/test splits — [CXO summary](https://www.cxoadvisory.com/big-ideas/10-steps-to-becoming-a-better-quant) (excerpt).
- [OPINION] Arnott, Harvey, Markowitz's seven points: research motivation with an ex-ante economic hypothesis and who is on the other side; multiple-testing accounting; honest data and sample choice; cross-validation with the reminder that true out-of-sample is live trading and fresh data; model dynamics with no tweaking on live results; simplicity; research culture rewarding process over results. Their illustration is a "magical" strategy using ticker letters that passes in-sample and out-of-sample validation — [Alpha Architect summary](https://hedgefundalpha.com/news/a-protocol-to-prevent-quants-gone-wild/).
- [ACAD-OOS] Regime and crowding: later-published factors decay more, about 5 points per year; arbitrage variables add only marginally once publication year and overfitting are included (R² 0.47 with, 0.45 without). Log market cap of the short leg alone has R² 0.11 and Amihud liquidity 0.06 — [Falck et al](https://ar5iv.labs.arxiv.org/html/2105.01380).
- [LIVE] Regime tailwind: strategies launched after unusually strong factor conditions deteriorate more — [Liu 2026](https://arxiv.org/abs/2604.18821). Falls in premia, timing and scale do not explain smart-beta decay — [Song et al](https://resolve.cambridge.org/core/services/aop-cambridge-core/content/view/35CA1CEA091A485DDA960BB2D9930BC9/S0022109023000674a.pdf/smart_beta_mirage.pdf) (excerpt). These two lean differently, so treat the split as unsettled.
- [ACAD-OOS] Cost underestimation: low transaction costs alone removed all in-sample performance — [Bajgrowicz and Scaillet](https://econpapers.repec.org/RePEc:eee:jfinec:v:106:y:2012:i:3:p:473-491) (excerpt).

### Inferences
| Failure mode | Best check | Grade of evidence for the check |
|---|---|---|
| Data snooping | Honest registry, DSR range, PBO, FDR | Theorem plus live support (Wiecki) |
| Look-ahead and leakage | Purge, embargo, point-in-time audit, shadow versus backtest on the same days | Method, opinion |
| Cost underestimation | Canary fills, cost stress at 2x | Academic plus inference |
| Pre-launch regime tailwind | Peer or benchmark excess, sub-period and regime splits, launch-timing discount | Live (Liu), preprint |
| Crowding | Fragility, liquidity limits, simple-factor exposure check | Weak |
| Regime change | Regime-conditional stats, CUSUM | Method |
| Operational outage or broker failure | Dry runs, alerting, manual fallback | Opinion; 5 dry runs fit here |
| Capacity | Participation limits, liquid-universe test | Falck liquid-pool result, indirect |
| Vendor data and corporate actions | Reconcile against exchange data; corporate-action checks | Opinion |

- Centurion's in-sample window 2013-25 sits in a strong Indian equity period, and Liu suggests a discount when launching after a strong regime. Whether current conditions qualify needs factor data I did not retrieve.
- Project memory records real operational incidents on the paper cron (GitHub cache egress refusal), which supports treating dry runs as operational evidence rather than statistical evidence.

### Gaps
- Items 8-10 of López de Prado's list not retrieved.
- Documented outage cases (Knight Capital, NSE outage in 2021): not retrieved, so not cited.
- Quant-quake and factor-crowding literature was not retrieved.

## How simple should the model be, and does ML add out-of-sample Sharpe at swing horizons?

### Takeaway
Simple wins where evidence is live: complex strategies decay more. Combine variants instead of selecting one, and shrink means hard. ML has strong academic results at monthly horizons in the US, but its star "complexity" claim is contested, and no India evidence with net Sharpe at swing horizon was found. ML is not established for Centurion.

### Cited Findings
- [LIVE] Complexity: the most complex strategies have a haircut 30 percentage points larger — [Bailey on Suhonen](https://mathinvestor.org/2017/08/backtest-overfitting-in-smart-beta-investments/).
- [ACAD-OOS] Number of operations to compute a signal predicts decay: coefficient -0.49, R² 0.11 — [Falck et al](https://ar5iv.labs.arxiv.org/html/2105.01380).
- [OPINION] Keep models as simple as possible; when data are limited, economic foundations become more important — [Alpha Architect summary](https://hedgefundalpha.com/news/a-protocol-to-prevent-quants-gone-wild/).
- [BT, random data] Carver: with 1 year of data, naive estimates gave Sharpe 1.36 in sample and 0.58 out of sample. Means need more shrinkage than correlations, and less data needs more shrinkage. With Sharpes of 0, 0.5 and 1.0 over 20 years, best mean shrinkage is 33-66%. He warns real degradation may be larger — [qoppac](https://qoppac.blogspot.com/2015/11/random-data-random-wanderings-in.html).
- [OPINION] Carver uses all available data for Sharpes and designs handcrafting to account for uncertainty; the page gives no out-of-sample result — [qoppac](https://qoppac.blogspot.com/2018/12/portfolio-construction-through_14.html).
- [ACAD-OOS] FDR selects more rules, allowing diversification against model uncertainty, while no method picked future winners ex ante — [Bajgrowicz and Scaillet](https://econpapers.repec.org/RePEc:eee:jfinec:v:106:y:2012:i:3:p:473-491) (excerpt).
- [ACAD-OOS] Gu, Kelly, Xiu: a neural network market-timing strategy on the S&P 500 made out-of-sample Sharpe 0.77 versus 0.51 buy and hold — [NBER w25398](https://www.nber.org/papers/w25398.pdf) (excerpt). A value-weighted long-short made 1.35 — [Lextech summary](https://www.lextechinstitute.ch/?p=2505) (excerpt). This is monthly horizon, US, and gross of realistic implementation.
- [ACAD-OOS] International extension: equal-weighted Sharpe above 2 and value-weighted above 1 in most markets, India not named — [CAIA](https://caia.org/node/5944) (excerpt; page returned 404 on fetch).
- [ACAD-OOS] Kelly, Malamud, Zhou (JF 2024): theory and US evidence that complex models with parameters exceeding observations beat simple ones in market return prediction — [EconPapers](https://econpapers.repec.org/RePEc:bla:jfinan:v:79:y:2024:i:1:p:459-503).
- [ACAD-OOS, critique] Nagel (NBER 34104, 2025): with short windows the method reduces to a recency-weighted average of past returns, essentially volatility-timed momentum. It worked historically rather than extracting a signal, and performs poorly on artificial data where returns reverse — [EconPapers](https://econpapers.repec.org/RePEc:nbr:nberwo:34104).
- [OPINION, secondary] At least six papers challenge the result, with more than 10,000 parameters on one year of data. Berk called it "virtually useless"; Buncic cited design errors. Kelly replied that small models under-fit, and called the preference for small models "a learned bias" — [Bloomberg repost](https://climateerinvest.blogspot.com/2025/08/aqrs-hard-to-believe-study-spurs-clash.html).
- [ACAD-OOS] India: Lalwani and Meshram (Applied Economics 2022), 35 characteristics, 1994-2019, survivorship-free. 14 predictors have t above 3, ML beats OLS on one-month-ahead accuracy, and strategies earn "significant returns". The listing gives no Sharpe, and full text is paywalled — [EconPapers](https://econpapers.repec.org/RePEc:taf:applec:v:54:y:2022:i:16:p:1814-1828).
- [ACAD-OOS, short window] ML as a meta-selector on backtest features: top-10 out-of-sample 1.8 versus 0.7 — [CXO on Wiecki](https://www.cxoadvisory.com/big-ideas/in-sample-vs-out-of-sample-performance-of-888-trading-strategies). Over about 8 months only, and one cohort.

### Inferences
- [CALC] Holding all N variants removes the z_N selection premium: selection bias is about σ_SR × z_N for the best variant, and zero for the equal-weighted set. The cost is a lower headline Sharpe, and the benefit is that the reported number is closer to honest. Parameter ensembles therefore trade an illusory gain for a real reduction in haircut.
- [OPINION] Parsimony rule: few parameters each with a stated economic reason; ensemble over a plateau of values; apply heavy shrinkage to Sharpe estimates in allocation (33-66% as a starting point, per Carver). Prefer equal or handcrafted weights across streams.
- [OPINION] ML at Centurion should earn its place as a challenger stream, not replace the core: monthly-horizon US results do not carry over to swing horizons in India without net-of-cost evidence, and Nagel's critique warns against treating an ML result as a new signal when it may be momentum. Test any ML signal for incremental alpha after momentum and volatility exposures.
- Falck's short-leg and market-cap findings are for long-short anomalies, and transfer to a long-only Indian book only loosely.
- Practical limit: 18 years of data cannot discriminate between candidate complexity levels at Sharpe 0.6, per the power table above. Arnott et al make the same point qualitatively.

### Gaps
- No India study with net-of-cost out-of-sample Sharpe at swing horizon was found; absence reflects a cut-off search as well as a likely thin literature.
- Avramov, Cheng, Metzker and Azevedo, Hoegner, Velikov on ML net of costs and restrictions: searches blocked, no claim made. They are the most relevant missing items.
- Hanauer and Kalsbach on emerging markets: only a reference listing seen.
- Kelly's written reply: reported secondarily only.
- Quantified parameters-versus-OOS-Sharpe evidence and direct parameter-ensemble studies: not retrieved.
