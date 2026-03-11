# Algorithm Refinement & Comprehensive Validation Design

**Date:** 2026-03-09
**Project:** TVSEF/CWSF 2026 — Automating Asteroid Detection

## Problem Statement

The algorithm detects 2024 RX69 (ASP0090) at 80% confidence but misses ASP0091-0093 (fainter objects in the same field). Calibration constants are inaccurate, and we lack systematic cross-referencing against known MPC submissions.

## Design

### 1. Calibration Corrections
- PIXEL_SCALE: 0.25 -> 0.258 arcsec/pixel (Pan-STARRS confirmed)
- FRAME_INTERVAL_MINUTES: 30.0 -> auto-detect from FITS MJD-OBS headers (~16.3 min for this dataset)
- Affects: velocity conversion, motion rate reporting, FWHM thresholds

### 2. Detection Sensitivity
- detection_sigma for real data: 8.0 -> 5.0 (matches SNR_THRESHOLD)
- ASP0091-0093 have SNR 6.2-8.8; sigma=8 prevents extraction
- Column artifact filters + 10-criteria validation handle increased noise

### 3. MPC Cross-Referencing
- Parse MPCReport.txt for ASP0090-0093, K14X22V, Q7555 positions
- Match algorithm detections within 10px tolerance
- Annotate candidates with MPC designations

### 4. Reporting Function
- Per-candidate: confidence %, pixel coords, motion rate, magnitude, criteria breakdown
- Per-field: detection count, FP count, cross-reference matches
- Summary: detection matrix, completeness, false positive rate

### 5. Full Validation
- All 21 IASC fields
- Known object recovery verification
- Sensitivity analysis (faintest object detected)

## Success Criteria
- Detect ASP0090 (2024 RX69) at >= 70% confidence
- Detect ASP0091-0093 if they exist in extractable data
- Detect K14X22V and Q7555 (bright known asteroids)
- False positive rate < 1 per field on negative fields
- No regression on synthetic tests (3/3, 0 FPs)
