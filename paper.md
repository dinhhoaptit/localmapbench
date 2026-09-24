---
title: "localmapbench: Nested Cross-Validation Benchmarking of Feature Maps for Local Regression"
tags:
  - Python
  - statistics
  - local regression
  - nested cross-validation
  - feature preprocessing
  - benchmarking
authors:
  - name: Hoa Dinh Nguyen
    orcid: 0009-0004-1708-9248
    corresponding: true
    affiliation: "1"
affiliations:
  - name: Posts and Telecommunications Institute of Technology, Hanoi, Vietnam
    index: 1
date: 24 September 2026
bibliography: paper.bib
---

# Summary

`localmapbench` is a Python toolkit for comparing feature preprocessors in local (neighborhood) regression under a **matched nested cross-validation** protocol. Neighborhood size $k$ and the feature map jointly determine predictive accuracy. Rankings that fix $k$ can confound representation quality with smoothing. This package evaluates candidate maps—standardized raw features, binary and soft block maps, sliding filters, pairwise channels, partial least squares (PLS), and sliced inverse regression (SIR)—with shared outer folds, train-only map fitting, and a data-driven one-standard-error largest-$k$ rule. A five-regime synthetic ladder serves as a **positive-control validation suite**, and a train-only suitability diagnostic screens when patterned maps merit full outer evaluation.

# Statement of need

Applied researchers often try ordered block aggregations, sliding filters, or supervised projections before local regression on tabular or spectral data. Without a common protocol, apparent gains may come from unequal neighborhood tuning rather than a better geometry. Existing libraries provide pieces of this pipeline (cross-validation utilities, PLS/SIR, local regressors) but do not offer a single, leakage-aware workflow for comparing heterogeneous feature maps under nested selection of $k$. `localmapbench` fills that gap for statisticians and domain scientists who need reproducible, matched comparisons and a lightweight screen before expensive outer-fold studies.

# State of the field

General ML toolkits such as scikit-learn [@pedregosa2011scikit] implement cross-validation, PLS, and nearest-neighbor estimators, but users must manually nest map fitting inside outer folds and choose $k$ consistently across maps. Packages focused on nested CV with feature selection (e.g., domain-specific nestedcv tools) target high-dimensional omics workflows rather than local-regression geometry and ordered-feature maps. Seriation utilities reorder variables [@hahsler2008seriation] but do not evaluate predictive maps under opt-$k$ local heads. Building on these ecosystems, `localmapbench` contributes a dedicated protocol layer: map adapters, 1-SE largest-$k$ selection, positive-control regimes, and a suitability diagnostic—rather than reinventing base estimators.

# Software design

The design separates (i) **feature maps** fit on training folds only, (ii) **neighborhood selection** via repeated inner CV with a 1-SE largest-$k$ rule [@hastie2009esl], and (iii) **outer assessment** on held-out folds. Positive-control regimes encode structural assumptions so that a correctly matched map should dominate; shuffled-order and fixed-$k$ ablations (available in research scripts) stress-test the protocol. The public API exposes `make_regime`, `evaluate_maps_nested`, `select_optimum_k`, and `suitability_diagnostic` for short scripts, while lower-level modules retain the research implementations of masks, sliding filters, and projections. This split keeps quickstarts small without discarding the fuller experimental stack under `software/`.

# Research impact statement

The protocol and validation suite underpin a Monte Carlo study of feature preprocessors for local regression (manuscript previously submitted to *Communications in Statistics—Simulation and Computation*). Frozen regime and sensitivity tables accompany the research scripts, providing reproducible evidence that the software recovers designed rankings under matched nested CV. The package is intended for reuse whenever local regression is paired with competing tabular/spectral feature maps, including chemometric and engineering regression settings.

# AI usage disclosure

Generative AI tools were used to assist with repository packaging, documentation drafting, and editorial reorganization of existing research code into the `localmapbench` package layout. Core statistical algorithms and experimental designs originate from the author’s research implementations. All AI-assisted text and code were reviewed, edited, and validated by the author, who made the core design decisions.

# Acknowledgements

No funding was obtained for this software. The author thanks users of the forthcoming repository issues for feedback.

# References
