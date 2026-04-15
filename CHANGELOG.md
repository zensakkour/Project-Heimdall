# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Open-source governance and contribution scaffolding
- Retrieval tuning rank objectives for close-range optimization (`within_1km_pct`, `within_5km_pct`, `within_10km_pct`)
- Retrieval tuning support for explicit `within_2km_pct` optimization target.
- Multi-index retrieval source-fusion mode (`retrieval_source_fusion_mode`) with `weighted_score` (default) and `rrf` options.
- Retrieval consensus top-1 refinement controls (`retrieval_consensus_top_n`, `retrieval_consensus_radius_km`, `retrieval_consensus_score_power`) across runtime config and retrieval provider.
- Lab random-sample geo evaluation flow (`Run Random Samples`) with random seed runs, per-sample distance diagnostics, and accuracy summary bands.
- UI/server endpoints for random sample scoring: `POST /eval/geo/random/start`, `GET /eval/geo/random/status`.

### Changed

- Retuned realistic single-index retrieval profile (`runs/bench_cfg/cfg_realistic_single.json`) to improve close-range accuracy on the realistic `n=180` benchmark split.
- Updated Paris retrieval profile to enable consensus refinement, improving realistic split metrics (`within_1km_pct`: `5.00` -> `10.00`, `median_km`: `11.50` -> `9.77` on `n=180`).
- Upgraded retrieval consensus center estimation to adaptive centroid/weighted-geo-median selection for stronger local outlier robustness in top-1 refinement.
- Improved operator globe visual presentation with stronger atmosphere/fog styling, candidate link lines, and glow/halo layers for geolocation readability.
- Replaced MIT terms with a non-commercial license requiring a separate commercial agreement for paid/company use.

## Notes

- `CHANGELOG.md` is for release-oriented summaries.
- Detailed engineering chronology lives in `PROGRESS.md` (append-only).
- Research-method details live in `src/docs/RESEARCH_PAPER.md`.
