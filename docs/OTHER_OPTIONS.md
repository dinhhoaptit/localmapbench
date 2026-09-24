# Broader venue options (low fee / hybrid preferred)

Context: LSSP desk-rejected on novelty; SoftwareX too costly; CompStat SI-only; *Journal of Systems and Software* is SE (poor fit). Goal: protocol/software or computational-stats home with **$0 or modest fees**.

Fit scores are rough (10 = strong match). Fee = typical author cost if you **avoid optional OA**.

---

## Tier A — best next options ($0)

| Journal | Model | Fee | Fit | Notes |
|---------|-------|-----|----:|-------|
| **JOSS** (*Journal of Open Source Software*) | Diamond OA | **$0** | **8/10** | Software *is* the paper. Needs installable package, docs, tests. Fastest free route. Short article. |
| **JSS** (*Journal of Statistical Software*) | Diamond OA | **$0** | **7.5/10** | Best free prestige in statistical software. Rewrite around API/design; demote long MC grids. Long queues; high packaging bar. |

**Recommendation if fee must be ~0:** start packaging for **JOSS**; keep **JSS** as upgrade path after a solid release.

---

## Tier B — hybrid stats / data-analysis ($0 subscription)

| Journal | Model | Fee | Fit | Notes |
|---------|-------|-----|----:|-------|
| **Journal of Applied Statistics** (T&F) | Hybrid (Open Select) | **$0** sub | **6/10** | Applied methods OK; still may want clearer “new method” than matched protocol alone. Regular submissions. |
| **Statistical Papers** (Springer) | Hybrid | **$0** sub | **6.5/10** | Explicitly publishes **reports on statistical software**; regular submissions. Better CompStat-like home without SI-only policy. |
| **Advances in Data Analysis and Classification** (Springer) | Hybrid | **$0** sub | **5.5/10** | Data analysis / classification focus; protocol paper is a stretch unless reframed toward structured feature representations. |
| **Statistical Analysis and Data Mining** (Wiley / ASA) | Hybrid-ish | **$0** sub typical | **5/10** | Wants innovative analytic techniques with rigor; “expected regime winners” critique may return. |

These keep a longer scholarly article, but the **LSSP novelty critique can reappear** unless the claim is clearly computational protocol + open software.

---

## Tier C — domain / neighboring ($0 or modest)

| Journal | Model | Fee | Fit | Notes |
|---------|-------|-----|----:|-------|
| **Journal of Chemometrics** (Wiley) | Hybrid | **$0** sub | **6.5/10** | Strong if you **lead with spectra / PLS / Tecator / NIR** and treat unordered UCI as secondary. Narrower audience than pure stats. |
| **Software Impacts** (Elsevier) | Gold OA | ~**USD 770** | **7/10** | Short software note; fee smaller than SoftwareX (~$1560). |
| **ACM TOMS** | Full OA from 2026 | ~**USD 950–1450** (2026 subsidized; $0 if ACM Open institution) | **6/10** | Mathematical software bar is high; needs polished library + algorithmic contribution, not only experiment scripts. |

---

## Tier D — usually skip (cost, scope, or history)

| Journal | Why skip |
|---------|----------|
| SoftwareX | ~USD 1560 APC |
| PeerJ Computer Science | ~USD 2155 APC; CS framing weak for this stats protocol |
| Journal of Open Research Software | ~£800 APC |
| Journal of Systems and Software | Software **engineering** — out of scope |
| Computational Statistics (Springer) | **Special issues only** |
| Statistics and Computing | Mandatory APC path / board instability |
| LSSP / CSDA / JSCS | Already rejected |
| JCGS | Novelty bar too high for current claim |

---

## How to choose

```text
Need $0 + software product first?     → JOSS
Need $0 + max stats-software prestige? → JSS (after packaging)
Want longer applied-stats article, $0? → Journal of Applied Statistics
Have chemometrics angle to emphasize? → Journal of Chemometrics
Can pay ~$770?                       → Software Impacts
```

---

## Suggested decision tree

1. **Default:** package `localmapbench` → submit **JOSS** ($0).  
2. **If you want a full journal article without APC:** rewrite for **Statistical Papers** (software-report lane) or **Journal of Applied Statistics**; chemometrics-centered rewrite for **Journal of Chemometrics**.  
3. **If packaging becomes excellent:** consider **JSS** next (or instead of JOSS).

Do not invest in CompStat or *Journal of Systems and Software* unless policy/scope changes.
