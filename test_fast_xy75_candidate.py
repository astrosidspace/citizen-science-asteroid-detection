"""Regression test: RX69 on XY75_p00 using FAST algorithm."""
import sys, os, time, glob
sys.path.insert(0, os.path.dirname(__file__))
import asteroid_detector_fast as ad

# XY75_p00 FITS files (RX69 confirmed detection field)
base = r"C:/Users/astro/Downloads/Astrometrica/Data/7th Sep"
p00_dir = os.path.join(base, "ps2-20240907_10_XY75_p00", "XY75_p00")
if not os.path.exists(p00_dir):
    for d in glob.glob(os.path.join(base, "*XY75*p00*")):
        if os.path.isdir(d):
            p00_dir = d
            for sub in glob.glob(os.path.join(d, "*")):
                if os.path.isdir(sub):
                    p00_dir = sub
                    break
            break

fits_files = sorted(glob.glob(os.path.join(p00_dir, "*.fits")))
if not fits_files:
    fits_files = sorted(glob.glob(os.path.join(base, "**/*XY75*p00*.fits"), recursive=True))

print("=" * 60)
print("FAST REGRESSION TEST: XY75_p00 (RX69 confirmed field)")
print("=" * 60)

if not fits_files:
    print(f"No FITS files found for XY75_p00 in {base}")
    sys.exit(1)

for f in fits_files:
    print(f"  {os.path.basename(f)}")

data = ad.load_fits_frames(fits_files)
frames = data['frames']
mjds = data['mjds']

print(f"\nRunning FAST detection pipeline...")
t0 = time.time()
result = ad.run_detection_pipeline(frames, verbose=True, detection_sigma=5.0, frame_mjds=mjds)
total_time = time.time() - t0

print(f"\nRESULTS ({total_time:.1f}s):")
print(f"  Candidates: {len(result.candidates)}")
print(f"  Stack candidates: {len(result.stack_candidates)}")
print(f"  Deep candidates: {len(result.deep_candidates)}")

# RX69 known position on XY75_p00: (627, 590)
RX69_X, RX69_Y = 627, 590
RX69_TOLERANCE = 50

found = False
best_match = None
best_dist = float('inf')

for label, dets in [("candidates", result.candidates),
                     ("stack_candidates", result.stack_candidates),
                     ("deep_candidates", result.deep_candidates)]:
    for det in dets:
        if det.sources:
            sx = det.sources[0].x
            sy = det.sources[0].y
            dist = ((sx - RX69_X)**2 + (sy - RX69_Y)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_match = (label, det, dist)

if best_match and best_match[2] < RX69_TOLERANCE:
    label, det, dist = best_match
    found = True
    print(f"\n  RX69 FOUND in {label}!")
    print(f"    Position: ({det.sources[0].x:.0f}, {det.sources[0].y:.0f})")
    print(f"    Distance from expected: {dist:.1f}px")
    print(f"    Confidence: {det.confidence_score:.1f}%")
    print(f"    Method: {det.detection_method}")
    if det.confidence_score >= 80.0:
        print("    STATUS: PASS (>=80%)")
    else:
        print(f"    STATUS: MARGINAL ({det.confidence_score:.1f}% < 80%)")
else:
    print(f"\n  RX69 NOT FOUND within {RX69_TOLERANCE}px of ({RX69_X},{RX69_Y})")
    if best_match:
        print(f"  Closest match in {best_match[0]} at dist={best_match[2]:.1f}px")
    print("  STATUS: FAIL")

print(f"\n  TIMING: {total_time:.1f}s (target: <60s)")
