# Local Data Layout

This repo intentionally keeps `data/` out of Git. The folder is a local cache for datasets, model weights, indices, and manual test images.

## Best Images To Use In `/analysis/`

Use these when you just want to upload a photo and test the site:

| Purpose | Recommended path | Notes |
| --- | --- | --- |
| Clean manual upload set | `data/analysis_tests/paris_street/images/` | Curated local copies from the real Paris street corpus. Use these first in `/analysis/`. |
| Current realistic Paris street photo | `data/paris_realistic_v1/street_combined/images/mapillary__1000629290469563.jpg` | Best match for the new street-to-aerial research direction. |
| More current Paris street photos | `data/paris_realistic_v1/street_combined/images/` | Use any `.jpg` here for realistic Paris input. |
| Old aerial benchmark chip | `data/spacenet_paris_test/chips/` | These are aerial chips, useful for old SpaceNet-style testing but not normal street-photo product usage. |

Default UI profile for normal Paris testing: `Paris (SpaceNet)`.

## Keep For Current Work

These are still useful for current model/data work:

| Path | Why keep |
| --- | --- |
| `data/paris_realistic_v1/street_combined/` | Current `40,000` all-source street-image corpus used as street/query images. |
| `data/paris_realistic_v1_combined/aerial/` | Current `40,000` IGN aerial crop corpus used as reference images. |
| `data/paris_realistic_v1_combined/splits_strict/` | Current leakage-buffered split; this is the benchmark split to trust. |
| `data/paris_realistic_v1_combined/indices/` | Current realistic aerial index. |
| `data/geo_index/` | Retrieval indices used by existing configs and older Paris benchmarks. |
| `data/models/` | Local model cache, especially GeoSpot. |
| `data/spacenet_paris_test/chips/` and `metadata.csv` | Older canonical SpaceNet Paris evaluation data used by lab tools and historical comparisons. |
| `data/analysis_tests/paris_street/` | Small manual upload set copied from the real Paris street corpus. |

## Cleanup Candidates

Do not delete blindly; move to an archive drive first if you are not sure.

### Usually Safe To Remove After Final Combined Data Exists

These are generated recovery/checkpoint folders. They are not needed for normal UI use or current combined strict training/eval once `data/paris_realistic_v1_combined/` exists.

| Path | Approx. local size | Why it can go |
| --- | ---: | --- |
| `data/paris_realistic_v1_combined_chunkpairs/` | `16.9 GB` | Intermediate chunk-pair output duplicated by final combined dataset. Removed locally. |
| `data/paris_realistic_v1_chunkpairs/` | `4.2 GB` | First-pass chunk-pair output duplicated by final v1/final combined outputs. Removed locally. |
| `data/paris_realistic_v1_combined_chunkmeta/` | small | Intermediate metadata chunks. Removed locally. |
| `data/paris_realistic_v1_chunkmeta/` | small | Intermediate metadata chunks. Removed locally. |
| `data/paris_realistic_smoke/` | small | Smoke-test dataset only. Removed locally if not debugging ingestion. |

### Large Raw Archives / Rebuild Sources

These are useful only if you want to rebuild derived data from scratch.

| Path | Approx. local size | Keep only if |
| --- | ---: | --- |
| `data/spacenet_paris/PS-RGB/` | `84.6 GB` | You want to re-chip SpaceNet from raw TIFs locally. Current code usually needs `chips/` and metadata, not raw TIFs. |
| `data/spacenet_paris_test/*.tar.gz` | large | You want local SpaceNet test archives after extraction. |
| `data/dota` and `data/dota_v1/` | `~3.8 GB` together | Removed locally for the Paris-focused workflow. Recreate only if detector dataset evaluation is needed later. |

### Possible But More Aggressive

Only remove these if you accept losing source-specific raw street folders and old baseline checkpoints:

| Path | Why it may be removable |
| --- | --- |
| `data/paris_realistic_v1/street_mapillary/` | Duplicated by `street_combined/` for current model use, but useful for source-specific provenance. |
| `data/paris_realistic_v1/street_panoramax/` | Duplicated by `street_combined/` for current model use, but useful for source-specific provenance. |
| `data/paris_realistic_v1/aerial/` | First `10,000` Panoramax -> IGN aerial checkpoint; useful for old baseline comparisons. |
| old `data/paris_realistic_v1/splits_*` except `splits_strict/` | Older split experiments; not used for final claims. |

## Suggested Conservative Cleanup

This removes obvious intermediate cache/checkpoint data while preserving the current realistic dataset and historical benchmark inputs:

```powershell
Remove-Item -LiteralPath data/paris_realistic_v1_combined_chunkpairs -Recurse
Remove-Item -LiteralPath data/paris_realistic_v1_chunkpairs -Recurse
Remove-Item -LiteralPath data/paris_realistic_v1_combined_chunkmeta -Recurse
Remove-Item -LiteralPath data/paris_realistic_v1_chunkmeta -Recurse
```

Potential savings: about `21 GB`.

## Suggested Big Cleanup

Only do this if you are okay re-downloading/rebuilding raw SpaceNet/DOTA data later:

```powershell
Remove-Item -LiteralPath data/spacenet_paris/PS-RGB -Recurse
Remove-Item -LiteralPath data/spacenet_paris_test/SN3_roads_test_public_AOI_3_Paris.tar.gz -Force
Remove-Item -LiteralPath data/spacenet_paris_test/AOI_3_Paris_Test_public.tar.gz -Force
```

Potential savings: over `95 GB`, depending on local archive sizes.
