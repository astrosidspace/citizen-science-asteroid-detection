#!/usr/bin/env python
"""
Quick test: Can TCFP recover 2024 RH39 from XY26_p10?

Expected: RH39 at approx (1623, 732), velocity ~(+26, -41) px/frame
MPC designation: K24R39H, magnitude ~19.9 R
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import asteroid_detector as ad

# Load the 4 FITS frames for XY26_p10
data_dir = r"C:\Users\astro\Downloads\Astrometrica\Data\3rd Sep\moving object found\ps2-20240903_8_XY26_p10\XY26_p10"
fits_files = sorted(glob.glob(os.path.join(data_dir, "*.fits")))

print(f"Found {len(fits_files)} FITS files")

# Load frames
result = ad.load_fits_frames(fits_files)
frames = result['frames']
mjds = result['mjds']
wcs_headers = result['wcs_headers']

print(f"Frames: {len(frames)}, shape={frames[0].shape}")

# Run detection pipeline
print("\n" + "="*70)
print("RUNNING DETECTION PIPELINE")
print("="*70)
det_result = ad.run_detection_pipeline(
    frames, verbose=True, detection_sigma=5.0, frame_mjds=mjds)

# Access DetectionResult fields
main_cands = det_result.candidates      # List[Tracklet]
sas_cands = det_result.stack_candidates  # List[Tracklet]

print(f"\n{'='*70}")
print(f"RESULTS: {len(main_cands)} main + {len(sas_cands)} shift-and-stack")
print(f"{'='*70}")

# Check ALL candidates for RH39
rh39_x, rh39_y = 1623, 732
MATCH_RADIUS = 80

print(f"\nSearching for RH39 near ({rh39_x}, {rh39_y}) within {MATCH_RADIUS}px:")

found = False
for label, cand_list in [("MAIN", main_cands), ("SAS", sas_cands)]:
    for i, t in enumerate(cand_list):
        # Tracklet: sources[0].x, sources[0].y
        if t.sources:
            cx = t.sources[0].x
            cy = t.sources[0].y
        else:
            continue
        dx = cx - rh39_x
        dy = cy - rh39_y
        dist = (dx**2 + dy**2)**0.5
        if dist < MATCH_RADIUS:
            found = True
            print(f"  *** {label} MATCH #{i}: pos=({cx:.0f},{cy:.0f}) "
                  f"dist={dist:.1f}px "
                  f"vel=({t.velocity_x:.1f},{t.velocity_y:.1f}) "
                  f"v={t.velocity_arcsec_min:.3f}\"/min "
                  f"mag={t.mean_magnitude:.1f} "
                  f"conf={t.confidence_score:.0f}% "
                  f"method={t.detection_method}")

if not found:
    print("  *** RH39 NOT FOUND in final candidates ***")

# Print all candidates
print(f"\nAll candidates:")
for label, cand_list in [("MAIN", main_cands), ("SAS", sas_cands)]:
    for i, t in enumerate(cand_list):
        if t.sources:
            cx, cy = t.sources[0].x, t.sources[0].y
        else:
            cx, cy = 0, 0
        print(f"  [{label} #{i}]: pos=({cx:.0f},{cy:.0f}) "
              f"vel=({t.velocity_x:.1f},{t.velocity_y:.1f}) "
              f"v={t.velocity_arcsec_min:.3f}\"/min "
              f"mag={t.mean_magnitude:.1f} "
              f"conf={t.confidence_score:.0f}% "
              f"method={t.detection_method}")

# MPC cross-reference
mpc_file = os.path.join(data_dir, "COD F52.txt")
if os.path.exists(mpc_file):
    mpc_obs = ad.parse_mpc_report(mpc_file)
    print(f"\nMPC objects in report: {list(mpc_obs.keys())}")
    all_cands = main_cands + sas_cands
    if all_cands and wcs_headers:
        matches = ad.cross_reference_candidates(
            all_cands, mpc_obs, wcs_headers[0])
        print(f"MPC matches: {len(matches)}")
        for m in matches:
            print(f"  {m}")
