"""Real-data validation: RH39 on XY26_p10 (regression test)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import asteroid_detector as ad

# XY26_p10 FITS files (RH39 field)
fits_files = [
    r"C:/Users/astro/Downloads/Astrometrica/Data/3rd Sep/moving object found/ps2-20240903_8_XY26_p10/XY26_p10/o60556h0328o.782882.ch.714012.XY26.p10.fits",
    r"C:/Users/astro/Downloads/Astrometrica/Data/3rd Sep/moving object found/ps2-20240903_8_XY26_p10/XY26_p10/o60556h0345o.782899.ch.714029.XY26.p10.fits",
    r"C:/Users/astro/Downloads/Astrometrica/Data/3rd Sep/moving object found/ps2-20240903_8_XY26_p10/XY26_p10/o60556h0362o.782916.ch.714046.XY26.p10.fits",
    r"C:/Users/astro/Downloads/Astrometrica/Data/3rd Sep/moving object found/ps2-20240903_8_XY26_p10/XY26_p10/o60556h0379o.782933.ch.714063.XY26.p10.fits",
]

print("=" * 60)
print("REGRESSION TEST: XY26_p10 (RH39 field)")
print("=" * 60)

for f in fits_files:
    if not os.path.exists(f):
        print(f"MISSING: {f}")
        sys.exit(1)
print(f"All {len(fits_files)} FITS files found.")

data = ad.load_fits_frames(fits_files)
frames = data['frames']
mjds = data['mjds']
print(f"Frames: {len(frames)}, shape: {frames[0].shape}")

print("\nRunning detection pipeline...")
t0 = time.time()
result = ad.run_detection_pipeline(frames, verbose=False, detection_sigma=5.0, frame_mjds=mjds)
total_time = time.time() - t0

print(f"\nRESULTS ({total_time:.1f}s):")
print(f"  Candidates: {len(result.candidates)}")
print(f"  Stack candidates: {len(result.stack_candidates)}")
print(f"  Deep candidates: {len(result.deep_candidates)}")

# Check for RH39 at position (1623, 733) in frame 0
RH39_X, RH39_Y = 1623, 733
RH39_TOLERANCE = 50  # px

found_rh39 = False
best_match = None
best_dist = float('inf')

for label, dets in [("candidates", result.candidates),
                     ("stack_candidates", result.stack_candidates),
                     ("deep_candidates", result.deep_candidates)]:
    for det in dets:
        if det.sources:
            sx = det.sources[0].x
            sy = det.sources[0].y
            dist = ((sx - RH39_X)**2 + (sy - RH39_Y)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_match = (label, det, dist)

if best_match and best_match[2] < RH39_TOLERANCE:
    label, det, dist = best_match
    found_rh39 = True
    print(f"\n  RH39 FOUND in {label}!")
    print(f"    Position: ({det.sources[0].x:.0f}, {det.sources[0].y:.0f})")
    print(f"    Distance from expected: {dist:.1f}px")
    print(f"    Confidence: {det.confidence_score*100:.1f}%")
    print(f"    Method: {det.detection_method}")
    print(f"    Velocity: {det.velocity_arcsec_min:.3f}\"/min")
    if det.confidence_score >= 0.80:
        print("    STATUS: PASS (>=80% confidence)")
    else:
        print(f"    STATUS: MARGINAL ({det.confidence_score*100:.1f}% < 80%)")
else:
    print(f"\n  RH39 NOT FOUND within {RH39_TOLERANCE}px of ({RH39_X},{RH39_Y})")
    if best_match:
        print(f"  Closest: {best_match[0]} at dist={best_match[2]:.1f}px")
    print("  STATUS: FAIL")

# Show deep candidate details
if result.deep_candidates:
    print(f"\n  Deep candidates ({len(result.deep_candidates)}):")
    for i, dc in enumerate(result.deep_candidates[:10]):
        src = dc.sources[0] if dc.sources else None
        x = src.x if src else '?'
        y = src.y if src else '?'
        print(f"    [{i}] pos=({x},{y}) vel={dc.velocity_arcsec_min:.3f}\"/min "
              f"conf={dc.confidence_score*100:.1f}%")
