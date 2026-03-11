# Algorithm Improvements — Research & Audit Synthesis

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 6 bugs and implement 4 quality improvements identified by the research/audit phase, improving detection reliability and reducing false positives for shift-and-stack candidates.

**Architecture:** The algorithm is a single-file pipeline (`asteroid_detector.py`, 3313 lines). Changes are surgical edits to specific functions — no new files needed. The key architectural improvement is routing shift-and-stack candidates through the same artifact filters that standard candidates already use.

**Tech Stack:** Python 3, numpy, scipy, matplotlib, astropy (no new dependencies)

---

## Context: Research & Audit Findings

### Bugs Found (audit)
1. **Global variable mutation** (L2140-2149): `FRAME_INTERVAL_MINUTES` is mutated via `global`, which persists between calls in batch processing (e.g., `run_full_validation.py` processing 22 fields). If field 1 sets it to 16.1 and field 2's FITS headers fail, field 2 uses 16.1 instead of 16.3 default.
2. **Shift-and-stack candidates bypass all 5 artifact filters** (L2370-2401): They're appended AFTER the filters at L2240-2365.
3. **Criterion 8 always passes** (L1344): Catalogue check is hardcoded `True`. Not useful.
4. **Shift-and-stack confidence uses incompatible scale** (L2073): `min(100, best_snr*10)` vs weighted 9-criteria scoring. An SNR=5 detection gets conf=50 despite passing all criteria it could.

### Quality Improvements (research)
5. **Graduated confidence scoring**: Binary pass/fail per criterion loses information. Professional surveys use continuous scoring.
6. **PSF validation for shift-and-stack**: Shift-and-stack candidates get `fwhm=TYPICAL_FWHM` and `fit_rms=0.1` hardcoded (L2064-2065) instead of measured values.
7. **Per-frame SNR for shift-and-stack sources**: Uses single-pixel `peak / noise` (L2049) instead of aperture photometry, making per-frame SNR misleadingly low.
8. **Cosmic ray injection** (synthetic mode): Only injected in frame 1. Minor — synthetic mode works fine for demos.

### Not Implementing (scope/risk)
- Finer velocity grid for RH39 recovery: RH39 has SNR ~0.5 per frame with persistent source overlap; no grid refinement will help. Already confirmed below automated detection threshold.
- GPU acceleration (KBMOD-style): Not needed for 4-frame IASC data.
- Online catalogue queries: Science fair demo must work offline.
- HelioLinC heliocentric linking: Requires orbit fitting, beyond scope.

---

### Task 1: Fix Global Variable Mutation Bug

**Files:**
- Modify: `asteroid_detector.py:2139-2149` (run_detection_pipeline)

**Step 1: Write the fix**

Replace the `global FRAME_INTERVAL_MINUTES` pattern with a local variable that doesn't leak between calls.

Current code (L2139-2149):
```python
    global FRAME_INTERVAL_MINUTES
    actual_frame_interval = FRAME_INTERVAL_MINUTES  # default
    if frame_mjds is not None and len(frame_mjds) >= 2:
        intervals = []
        for i in range(1, len(frame_mjds)):
            dt_days = frame_mjds[i] - frame_mjds[i - 1]
            intervals.append(dt_days * 24.0 * 60.0)
        actual_frame_interval = np.mean(intervals)
        FRAME_INTERVAL_MINUTES = actual_frame_interval
```

New code:
```python
    actual_frame_interval = FRAME_INTERVAL_MINUTES  # default
    if frame_mjds is not None and len(frame_mjds) >= 2:
        intervals = []
        for i in range(1, len(frame_mjds)):
            dt_days = frame_mjds[i] - frame_mjds[i - 1]
            intervals.append(dt_days * 24.0 * 60.0)
        actual_frame_interval = np.mean(intervals)
```

Then ensure `actual_frame_interval` is used everywhere downstream instead of the global. Check: it's already passed to `shift_and_stack_search(... actual_frame_interval ...)` at L2378 and used for velocity conversion at L2079.

**Step 2: Verify no other references to the global mutation**

Search for all uses of `FRAME_INTERVAL_MINUTES` after line 2149 in the pipeline function to confirm `actual_frame_interval` is used everywhere.

**Step 3: Run validation to verify no regression**

Run: `python asteroid_detector.py --validate`
Expected: Synthetic test passes (3/3 detections, 0 FPs)

---

### Task 2: Route Shift-and-Stack Candidates Through Artifact Filters

**Files:**
- Modify: `asteroid_detector.py:2370-2401` (run_detection_pipeline, Step 4)

**Step 1: Refactor artifact filter application into a helper function**

Extract Steps 3b-3e (L2240-2365) into a function `apply_artifact_filters(candidates, frames, is_real_data, verbose)` that returns filtered candidates. This allows reuse for both standard and shift-and-stack candidates.

**Step 2: Apply artifact filters to shift-and-stack candidates**

After shift-and-stack candidates are generated (L2376-2379), run them through the same artifact filter function before appending to the main candidates list.

Current flow:
```
Step 3: validate_tracklet -> candidates
Step 3b-3e: artifact filters on candidates
Step 4: shift_and_stack_search -> append to candidates (BYPASSES filters)
```

New flow:
```
Step 3: validate_tracklet -> candidates
Step 3b-3e: artifact filters on candidates
Step 4: shift_and_stack_search -> artifact filters on stack_candidates -> append
```

**Step 3: Run validation**

Run: `python asteroid_detector.py --validate`
Expected: Synthetic test still passes. Run on one real field to check shift-and-stack candidates are filtered.

---

### Task 3: Unify Confidence Scoring for Shift-and-Stack Candidates

**Files:**
- Modify: `asteroid_detector.py:2027-2098` (shift_and_stack_search, tracklet conversion section)
- Modify: `asteroid_detector.py:2370-2401` (run_detection_pipeline, Step 4)

**Step 1: Run validate_tracklet on shift-and-stack candidates**

After shift-and-stack candidates are converted to Tracklet objects (L2027-2098), pass them through `validate_tracklet()` with `frames=subtracted_frames` for proper PSF measurement and 9-criteria scoring. This replaces the ad-hoc `min(100, best_snr*10)` confidence at L2073.

In `run_detection_pipeline`, after `shift_and_stack_search()` returns, iterate over each tracklet and call:
```python
validate_tracklet(sc, frames=subtracted_frames, field_fwhm=field_fwhm, is_real_data=is_real_data)
```

Keep the tracklet's `detection_method = "shift_and_stack"` marker. The validate_tracklet function will now assign proper confidence scores and criteria results.

**Step 2: Remove hardcoded confidence in shift_and_stack_search**

Remove L2073: `tracklet.confidence_score = min(100, best_snr * 10)` — this will be set by `validate_tracklet` instead.

**Step 3: Threshold shift-and-stack candidates**

Only add shift-and-stack candidates that pass `tracklet.is_candidate` (confidence >= 70 + hard gates). This ensures consistent quality between standard and shift-and-stack detections.

Note: Some shift-and-stack candidates may fail criterion 2 (per-frame SNR < 5) because they're sub-threshold objects. That's expected — the shift-and-stack SNR boost is what makes them detectable. For shift-and-stack candidates, criterion 2 should use the STACKED SNR (stored in the detection) rather than per-frame SNR. Implement this by checking `detection_method == "shift_and_stack"` in validate_tracklet and using the stacked SNR.

**Step 4: Run validation**

Run: `python asteroid_detector.py --validate`
Then run on XY75_p00 to verify RX69 still detected.

---

### Task 4: Implement Graduated Confidence Scoring

**Files:**
- Modify: `asteroid_detector.py:1421-1458` (validate_tracklet, confidence calculation)

**Step 1: Replace binary scoring with graduated scoring**

Current: Each criterion contributes its full weight if passed, 0 if failed.

New: Each criterion contributes a FRACTION of its weight based on how well it passes:
- Criterion 2 (SNR): `min(1.0, (mean_snr - 3.0) / (SNR_THRESHOLD - 3.0))` — partial credit for SNR between 3 and 5
- Criterion 3 (PSF fit): `max(0, 1.0 - mean_fit_rms / FIT_RMS_THRESHOLD)` — better fit = more credit
- Criterion 5 (linearity): `max(0, 1.0 - max_residual / LINEARITY_THRESHOLD)` — less deviation = more credit
- Criterion 6 (velocity): `max(0, 1.0 - max_variation / VELOCITY_VARIATION_MAX)` — less variation = more credit
- Criterion 7 (magnitude): `max(0, 1.0 - mag_range / MAGNITUDE_VARIATION_MAX)` — less range = more credit
- Criteria 1, 4, 8, 9: Keep binary (these are fundamentally pass/fail)

This means a tracklet with SNR=4.5 (just below threshold) gets partial credit instead of 0.

**Step 2: Update confidence calculation**

```python
graduated_scores = {}
for k, v in criteria.items():
    if k == '2_snr':
        graduated_scores[k] = min(1.0, max(0, (mean_snr - 3.0) / (SNR_THRESHOLD - 3.0))) if not snr_pass else 1.0
    elif k == '3_psf_fit':
        graduated_scores[k] = max(0, 1.0 - mean_fit_rms / FIT_RMS_THRESHOLD) if not psf_pass else 1.0
    elif k == '5_linear_motion':
        graduated_scores[k] = max(0, 1.0 - max_residual / LINEARITY_THRESHOLD) if not linearity_pass else 1.0
    elif k == '6_constant_velocity':
        graduated_scores[k] = max(0, 1.0 - max_variation / VELOCITY_VARIATION_MAX) if not velocity_pass else 1.0
    elif k == '7_stable_magnitude':
        graduated_scores[k] = max(0, 1.0 - mag_range / MAGNITUDE_VARIATION_MAX) if not magnitude_pass else 1.0
    else:
        graduated_scores[k] = 1.0 if v['passed'] else 0.0

earned = sum(weights[k] * graduated_scores[k] for k in criteria)
confidence = (earned / total_weight) * 100
```

**Step 3: Run validation**

Run: `python asteroid_detector.py --validate`
Expected: Confidence scores may shift slightly. Synthetic test should still pass.
Run on XY75_p00 to verify RX69 confidence changes (should stay >= 70%).

---

### Task 5: Improve Criterion 8 (Catalogue Check)

**Files:**
- Modify: `asteroid_detector.py:1339-1350` (validate_tracklet, criterion 8)

**Step 1: Make criterion 8 honest**

Instead of pretending it passes, mark it as "not applicable" with 0 weight when offline:

```python
# When running offline (no catalogue access), mark as N/A
# and remove its weight from the total rather than giving free points
catalogue_pass = True  # N/A in offline mode
catalogue_applicable = False
criteria['8_not_known'] = {
    'passed': catalogue_pass,
    'value': 'N/A (offline mode)',
    'threshold': 'No match in MPC/SkyBot',
    'note': 'Catalogue check: not available offline — excluded from scoring',
    'applicable': False
}
```

Update the confidence calculation to exclude inapplicable criteria:

```python
total_weight = sum(weights[k] for k in criteria if criteria[k].get('applicable', True))
earned = sum(weights[k] * graduated_scores[k] for k in criteria if criteria[k].get('applicable', True))
```

This makes the confidence score based on 95 points (not 100), giving more honest differentiation.

**Step 2: Run validation**

Expected: Confidence scores slightly adjusted (95-point scale instead of 100). All existing candidates should still pass.

---

### Task 6: Fix Shift-and-Stack PSF and SNR Measurements

**Files:**
- Modify: `asteroid_detector.py:2027-2098` (shift_and_stack_search, tracklet conversion)

**Step 1: Replace hardcoded PSF values with measurements**

Current (L2064-2065):
```python
snr=snr, fwhm=TYPICAL_FWHM,
fit_rms=0.1, frame_index=i,
```

These should be measured using the existing `measure_psf()` function:
```python
# Measure PSF at predicted position (if within bounds)
fwhm_meas, rms_meas = measure_psf(subtracted_frames[i], float(sx), float(sy), sigma_psf)
```

Where `sigma_psf = TYPICAL_FWHM_PIX / 2.355` (same as used in validate_tracklet deferred PSF).

Note: For very faint shift-and-stack sources, PSF measurement may be noisy. The `validate_tracklet` deferred PSF fitting will override these anyway, but having reasonable initial values is better than hardcoded ones.

**Step 2: Use aperture photometry for per-frame SNR**

Current (L2048-2049): `snr = peak / local_noise if local_noise > 0 else 0`

Better: use aperture photometry (consistent with what the stacker uses):
```python
# Aperture photometry SNR (same method as stacker Stage 2)
ap_r = 4
ann_in, ann_out = 8, 15
# [aperture photometry code similar to L1728-1744]
```

This gives more meaningful per-frame SNR values that are comparable to the stacked SNR.

**Step 3: Run validation**

Run on XY75_p00 to verify RX69 shift-and-stack candidate (if any) has measured PSF values.

---

### Task 7: Run Full Validation on All 22 Fields

**Files:**
- Run: `run_full_validation.py`

**Step 1: Run full validation**

```bash
cd citizen-science-asteroid-detection
python run_full_validation.py
```

**Step 2: Compare results to previous baseline**

Previous baseline: 46 candidates (13 INTERESTING, 17 PLAUSIBLE, 1 UNCERTAIN, 5 POSSIBLE_ARTIFACT, 10 LIKELY_ARTIFACT), ~2.1 FP/field, 0 MPC matches.

Compare:
- Total candidate count (should decrease slightly due to shift-and-stack filtering)
- Confidence score distribution (should be more differentiated with graduated scoring)
- RX69 detection in XY75_p00 (must remain >= 70% confidence)
- False positive rate (target: < 2.1 per field, ideally lower)

**Step 3: Run candidate analysis**

```bash
python analyze_candidates.py
```

Review updated classifications.

---

## Execution Order

Tasks 1-6 are code changes (implement sequentially to avoid conflicts).
Task 7 is the final validation run.

Priority order: 1 (bug fix, trivial) → 2 (critical bug) → 3 (critical quality) → 4 (quality) → 5 (quality) → 6 (quality) → 7 (validation)
