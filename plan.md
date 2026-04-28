# Branch Plan: `tech/structure-analysis-cues-v2`

## Status
- Branch purpose: research and prototype stronger scene-analysis cues for geolocation and place matching.
- Completion policy: this branch does not need to complete every downstream idea now; it needs to compare options cleanly and decide what is worth keeping.

## Problem
The current stack already benefits from weighted hard-negative retrieval adaptation and a first structure-aware reranker. That is still a shallow scene-layout layer. We need a stronger plan for cues such as:
- sun and shadow direction
- building corners and junction geometry
- roof and block footprint shape
- street-scene building layout for non-overhead photos

## Direction
We will improve in stages instead of jumping straight to a single heavy architecture.

### Stage 1: Compare Remote-Sensing Backbones
Goal:
- Compare the current CLIP-based retrieval path against remote-sensing-specialized alternatives before adding more handcrafted logic.

Candidates to compare:
- current OpenAI CLIP baseline
- RemoteCLIP
- GeoRSCLIP

Decision gate:
- Keep only backbones that improve leakage-safe benchmark quality on the same splits and with the same evaluation protocol.

Why first:
- If the embedding backbone is weak, extra shadow/corner logic becomes a patch over the wrong foundation.

### Stage 2: Choose the Winning Backbone and Refit Retrieval Adaptation
Goal:
- Re-run projection / hard-negative adaptation on the selected backbone instead of carrying old tuning assumptions forward.

Work:
- rebuild or route indices for the winning backbone
- retrain the weighted projection / adaptation path if the backbone changes
- benchmark again on the realistic split

Decision gate:
- Only keep the backbone swap if it still wins after adaptation and not just in raw retrieval.

### Stage 3: Expand Scene-Structure Analysis
Goal:
- Add richer analysis cues than the current first-pass structure rerank.

Cue families to test:
- stronger sun-shadow cues:
  - shadow direction consistency
  - shadow elongation / confidence
  - sun-facing vs shadow-facing mass balance
- building-corner cues:
  - junction density
  - corner spatial distribution
  - dominant corner angle families
- footprint / block-layout cues:
  - coarse building mass layout
  - orientation consistency
  - grid / block regularity

Decision gate:
- Keep only cues that add measurable value on realistic evaluation, not just visual plausibility.

### Stage 4: Add Building-Footprint / Segmentation Assistance
Goal:
- Move from coarse image statistics toward explicit building structure when needed.

Potential path:
- remote-sensing building / footprint segmentation support
- derive corners, polygon orientation, compactness, and layout descriptors from the segmented structure

Why:
- Corners and shadows are more reliable when tied to extracted structure rather than only raw gradients.

### Stage 5: Move Toward Geometry / BEV / 3D
Goal:
- Progress toward geometry-aware matching once retrieval and structure signals are benchmarked.

Important scope rule:
- This is not only for ground-to-satellite geolocation.
- Ground-to-satellite geolocation is one target feature.
- Another target is street-photo and general in-the-wild photo localization, where geometry can help match building corners, facade layout, viewpoint, and street structure.

Potential geometry directions:
- BEV-style normalization where viewpoint mismatch is severe
- 3D or pseudo-3D reasoning for building/facade structure
- geometry-aware reranking using corner correspondences and layout consistency

Decision gate:
- Only move heavy geometry into the core path if the product direction clearly benefits from broader photo localization, not just overhead imagery.

## Benchmarks and Evidence Rules
- Use leakage-safe evaluation for all improvement claims.
- Record every real comparison in `PROGRESS.md`.
- Update `README.md` and `src/docs/RESEARCH_PAPER.md` when branch conclusions change.
- Prefer compare -> choose -> deepen over adding multiple unvalidated systems at once.

## Immediate Next Actions
1. Benchmark RemoteCLIP against the current CLIP baseline on the canonical realistic split.
2. Benchmark GeoRSCLIP against the same split and same settings.
3. Choose one of:
   - keep current CLIP
   - switch to RemoteCLIP
   - switch to GeoRSCLIP
4. Only after choosing, extend the structure-analysis rerank with stronger shadow and corner-layout signals.
