# Project Heimdall Market Research and SOTA Landscape

Version: v1.0  
Date: April 15, 2026  
Source synthesis: `C:\Users\zen\Downloads\deep-research-report.md`

## Purpose
This document captures external market and state-of-the-art research relevant to Project Heimdall.
It is a companion to `src/docs/RESEARCH_PAPER.md` and is focused on:

- external landscape and competitive direction,
- dataset and benchmark ecosystem,
- technology options we should consider,
- practical decisions for our roadmap.

## What Was Reviewed
The external review covered:

- classical geo-localization paradigms (retrieval, Earth-cell classification, coordinate regression),
- modern image-location alignment models (GeoCLIP/StreetCLIP family and remote-sensing variants),
- uncertainty and reliability methods (calibration, selective prediction, conformal prediction),
- aerial-specific detection and retrieval context (DOTA, OBB toolchains),
- benchmark governance and leakage risk controls (spatial split discipline, anti-duplicate checks).

## External Landscape Summary
Current market/research practice converges on hybrid systems, not single-model solutions.
The strongest systems combine:

1. large-scale retrieval over geo-referenced corpora,
2. multiple evidence sources (visual, metadata, cross-source agreement),
3. explicit uncertainty and calibration controls,
4. strict benchmark governance to avoid leakage-driven false gains.

For Heimdall, this is directionally aligned with our architecture:

- multi-provider candidate generation,
- probabilistic fusion with uncertainty outputs,
- benchmark and regression governance.

## Methods Landscape (High Level)
Primary families and implications:

1. Retrieval-first systems:
- Usually strongest practical baseline when index quality is high.
- Scale well with ANN infrastructure.
- Sensitive to index leakage and duplicate contamination.

2. Earth-partition classification systems:
- Effective at global scale with huge data.
- Expensive to train and less natural for local multi-modal uncertainty.

3. Image-location alignment foundation models:
- Strong frontier direction.
- Domain mismatch risk for aerial imagery if using only generic vision-language backbones.
- Remote-sensing-adapted encoders are promising.

4. Reasoning-oriented VLM systems:
- Useful for explainability and chain-of-thought style localization narratives.
- Not yet a direct replacement for retrieval+fusion precision workloads.

## Dataset and Benchmark Ecosystem
Most relevant benchmark/data directions for us:

1. Global street-level generalization:
- OSV-5M style datasets with strict spatial train/test separation.

2. Landmark/web-scale retrieval:
- GLDv2 and related historical benchmark families.

3. Cross-view robustness:
- VIGOR/CVACT/University-1652 style settings.

4. Aerial and overhead domains:
- SpaceNet-family data for geospatial imagery.
- DOTA for oriented aerial detection signals.

## Market Tooling and Infra Signals
Stable engineering choices in the broader ecosystem:

- ANN/vector search: Faiss and ScaNN class tooling.
- Oriented detection research: MMRotate ecosystem.
- Production-leaning OBB workflows: Ultralytics OBB stacks.
- Geospatial ML datasets/components: TorchGeo ecosystem.

These confirm our infra assumptions around index-centric retrieval and OBB-driven aerial signal extraction.

## Implications for Heimdall
### What Is Already Strong
- Multi-provider retrieval and probabilistic fusion.
- Configurable retrieval controls (diversity/locality/consensus/tta).
- Uncertainty-aware outputs and confidence tiering.
- Benchmark governance and reproducibility discipline.

### Main Gaps to Close
- Domain-adapted remote-sensing retrieval backbones are not yet first-class default.
- Localizability/abstention is not yet a dedicated upstream module.
- Region-level coverage guarantees (conformal-style credible regions) are not yet integrated.
- Global and cross-view leakage-safe benchmark breadth remains limited.

## Prioritized Directions (Current)
1. Domain-adapted retrieval backbones (highest priority):
- Evaluate remote-sensing-native encoders against current CLIP/SigLIP path on the same realistic splits.

2. Selective prediction/localizability:
- Add predict-vs-abstain gating to reduce catastrophic confident failures.

3. Spatial uncertainty with guarantees:
- Add conformal-style credible regions over candidate clusters.

4. Expansion of leakage-safe evaluation:
- Add broader global and cross-view suites with anti-duplicate controls.

## Approaches We Will Consider
The following approaches remain active candidates:

1. Remote-sensing-native embedding backbones as primary retrieval features.
2. Localizability scoring as a gating signal in confidence policy.
3. Conformal prediction for region-level uncertainty guarantees.
4. Rank-based multi-index source fusion where benchmarks support it.
5. Additional diversity reranking (MMR/DPP style) for multi-hypothesis coverage.

## Decision Snapshot From Current Cycle
From the current engineering cycle:

- We implemented `retrieval_source_fusion_mode` with:
  - `weighted_score` (default),
  - `rrf` (experimental).
- Controlled benchmark on realistic split (`n=180`) showed `rrf` underperformed `weighted_score` in current Paris profile.
- Decision:
  - keep `weighted_score` as default,
  - keep `rrf` available for future multi-index/global experiments.

## Practical Next Experiments
Shortlist for next implementation rounds:

1. Backbone benchmark round:
- add remote-sensing candidate encoders,
- run same realistic split metrics (`within_1km_pct`, `within_2km_pct`, `median_km`, `p90_km`),
- promote only if gains hold on leakage-safe settings.

2. Localizability gate prototype:
- start with thresholded policy from fusion diagnostics,
- then move to learned gate if needed.

3. Conformal uncertainty prototype:
- calibrate on held-out set and output region coverage diagnostics.

## Relationship to Other Documents
- Technical narrative and experiments: `src/docs/RESEARCH_PAPER.md`
- Geolocation architecture and knobs: `src/docs/GEO_TECH.md`
- Reproducibility process: `src/docs/REPRODUCIBILITY.md`
- Engineering timeline and validation artifacts: `PROGRESS.md`

