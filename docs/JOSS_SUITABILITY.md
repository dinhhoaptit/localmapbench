# Suitability vs *Journal of Open Source Software* (JOSS)

Assessment date: 2026-09-24. Based on JOSS submitting guide and review checklist.

## Overall rating

| Lens | Score | Verdict |
|------|------:|---------|
| **Topical aims fit** (research software with clear scientific use) | **8 / 10** | Good: nested-CV benchmarking of feature maps for local regression is in-scope research software |
| **Current LSSP manuscript as JOSS paper** | **2 / 10** | Wrong format and wrong focus (methods/MC results paper) |
| **Current `software/` assets as JOSS package** | **3 / 10** | Useful scripts, but not a packaged, documented, tested open project |
| **Immediate submission readiness** | **2 / 10** | Not ready; see blockers below |
| **Feasibility after full JOSS prep** | **7 / 10** | Achievable if packaging + open development timeline are met |

**Bottom line:** JOSS is a **feasible and cost-free venue**, but the present Monte Carlo article must be **replaced** by a short `paper.md`, and the code must become a **real open package** with months of public history—not a zip of experiment scripts.

---

## What JOSS wants (relevant aims)

- Open-source **research software** (not notebooks / pretrained models alone).
- Obvious research application; meaningful contribution (not a one-off script or thin wrapper).
- Paper must **not** focus on new research results from using the software.
- Short Markdown paper (`paper.md`, typically ~750–1750 words) with required sections.
- Public Git repo (clone/browse/issues without registration), OSI license, installable package, docs, tests.
- **Pre-review gates (strict):**  
  1. Public development history **> 6 months** with iterative commits (not a dump)  
  2. Demonstrated research use/impact (not only aspirational)  
  3. Good open-source practices (LICENSE, releases, docs, tests/CI, CONTRIBUTING, etc.)  
  4. Feature-complete, language-standard packaging

Fee: **$0**.

---

## Alignment (strengths)

| Criterion | Match |
|-----------|--------|
| Research software | Yes — evaluation protocol + feature maps + diagnostic for local regression |
| Research application | Yes — Monte Carlo validation + public tabular/spectral examples already exist |
| Not SE / not CompStat-SI | Yes — JOSS accepts regular software submissions |
| Fee | Yes — free |
| Can cite prior LSSP-related work as impact | Possible — rejected/submitted methods work can support “used in research,” if disclosed carefully |

---

## Gaps / risks (must fix)

| Issue | Severity |
|-------|----------|
| No public GitHub history ≥ 6 months yet | **Critical** — will desk-reject if published and submitted immediately |
| Code is flat scripts, not `pip`-installable package | High |
| No LICENSE / README / CONTRIBUTING / CODE_OF_CONDUCT | High |
| No automated tests / CI | High |
| No tagged releases / Zenodo DOI yet | High |
| Current manuscript focuses on MC findings (JOSS forbids research-results focus) | High |
| “Minor utility / one-off analysis toolkit” perception | Medium — must show reusable API + design vs related tools |
| Solo author open-practice signals | Medium — need changelog, releases, docs, tests to compensate for no multi-author PRs |

---

## Manuscript vs JOSS paper

| LSSP article | JOSS `paper.md` |
|--------------|-----------------|
| Long methods + MC results | Short software note |
| Nested CV comparison as scientific claim | Software solves matched evaluation / confounding of map vs \(k\) |
| Many figures/tables of RMSE | At most light illustration; no result-heavy paper |
| LaTeX journal article | Markdown + BibTeX in the software repo |

Do **not** submit `manuscript_with_authors.pdf` to JOSS.
