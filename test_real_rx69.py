"""Real-data validation: RX69 on XY75_p11 with adaptive coarse factor."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import asteroid_detector as ad

# XY75_p11 FITS files (RX69 field)
fits_files = [
    r"C:/Users/astro/Downloads/Astrometrica/Data/7th Sep/ps2-20240907_10_XY75_p11/XY75_p11/o60560h0263o.785232.ch.716162.XY75.p11.fits",
    r"C:/Users/astro/Downloads/Astrometrica/Data/7th Sep/ps2-20240907_10_XY75_p11/XY75_p11/o60560h0280o.785249.ch.716179.XY75.p11.fits",
    r"C:/Users/astro/Downloads/Astrometrica/Data/7th Sep/ps2-20240907_10_XY75_p11/XY75_p11/o60560h0297o.785266.ch.716196.XY75.p11.fits",
    r"C:/Users/astro/Downloads/Astrometrica/Data/7th Sep/ps2-20240907_10_XY75_p11/XY75_p11/o60560h0314o.785283.ch.716213.XY75.p11.fits",
]

mpc_file = r"C:/Users/astro/Downloads/Astrometrica/Data/MPCReport.txt"

print("=" * 60)
print("REAL DATA VALIDATION: XY75_p11 (RX69 field)")
print("=" * 60)

# Check files exist
for f in fits_files:
    if not os.path.exists(f):
        print(f"MISSING: {f}")
        sys.exit(1)
print(f"All {len(fits_files)} FITS files found.")

# Load frames
print("\nLoading frames...")
data = ad.load_fits_frames(fits_files)
frames = data['frames']
mjds = data['mjds']
print(f"  Frames: {len(frames)}, shape: {frames[0].shape}")
print(f"  MJD range: {mjds[0]:.6f} - {mjds[-1]:.6f}")

# Run full pipeline
print("\nRunning detection pipeline (sigma=5.0)...")
t0 = time.time()
result = ad.run_detection_pipeline(frames, verbose=True, detection_sigma=5.0, frame_mjds=mjds)
total_time = time.time() - t0

print(f"\n{'=' * 60}")
print(f"RESULTS (total {total_time:.1f}s)")
print(f"{'=' * 60}")
print(f"  Standard tracklets: {len(result.tracklets)}")
print(f"  Candidates: {len(result.candidates)}")
print(f"  Stack candidates: {len(result.stack_candidates)}")
print(f"  Deep candidates: {len(result.deep_candidates)}")
print(f"  Processing time: {result.processing_time_seconds:.1f}s")

# Show deep candidates
if result.deep_candidates:
    print(f"\n  Deep candidates detail:")
    for i, dc in enumerate(result.deep_candidates):
        src = dc.sources[0] if dc.sources else None
        x = src.x if src else '?'
        y = src.y if src else '?'
        print(f"    [{i}] pos=({x},{y}) vel={dc.velocity_arcsec_min:.3f}\"/min "
              f"conf={dc.confidence_score:.3f} method={dc.detection_method}")

# Cross-reference with MPC
if os.path.exists(mpc_file):
    print(f"\nCross-referencing with MPC report...")
    mpc_objects = ad.parse_mpc_report(mpc_file)
    print(f"  MPC objects: {list(mpc_objects.keys())}")

    wcs_header = data['wcs_headers'][0] if data.get('wcs_headers') else None

    # Check all detection lists
    all_detections = []
    for label, dets in [("candidates", result.candidates),
                         ("stack_candidates", result.stack_candidates),
                         ("deep_candidates", result.deep_candidates)]:
        if dets:
            matches = ad.cross_reference_candidates(dets, mpc_objects, wcs_header)
            if matches:
                print(f"\n  MPC MATCHES in {label}:")
                for m in matches:
                    print(f"    {m.get('designation','?')} matched at "
                          f"({m.get('candidate_x','?')},{m.get('candidate_y','?')}) "
                          f"sep={m.get('separation_arcsec','?'):.1f}\"")
                all_detections.extend(matches)

    if not all_detections:
        print("  No MPC matches found in any detection list.")
else:
    print(f"  MPC file not found: {mpc_file}")

print(f"\nDone.")
