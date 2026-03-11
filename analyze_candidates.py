#!/usr/bin/env python3
"""
Analyze all remaining candidates from the validation run.
Classify by motion characteristics and compute RA/Dec positions.
Generate a comprehensive findings report.
"""
import os
import sys
import json
import glob
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asteroid_detector import (
    load_fits_frames, pixel_to_radec, ra_dec_to_string,
    parse_mpc_report, PIXEL_SCALE, FRAME_INTERVAL_MINUTES
)

DATA_ROOT = "C:/Users/astro/Downloads/Astrometrica/Data"
MPC_REPORT = os.path.join(DATA_ROOT, "MPCReport.txt")

# Load validation results
results_path = os.path.join(os.path.dirname(__file__), "validation_results.json")
with open(results_path) as f:
    all_results = json.load(f)

# Parse MPC report for cross-reference
mpc_objects = {}
if os.path.exists(MPC_REPORT):
    mpc_objects = parse_mpc_report(MPC_REPORT)

# Collect all field paths for WCS lookups
def find_field_path(field_name):
    """Find the data directory for a given field name."""
    for root, dirs, files in os.walk(DATA_ROOT):
        if os.path.basename(root) == field_name:
            fits_files = sorted(glob.glob(os.path.join(root, "*.fits")))
            if len(fits_files) == 4:
                return root, fits_files
    return None, None


def classify_candidate(cd):
    """Classify a candidate based on motion characteristics."""
    vx = cd['vx']
    vy = cd['vy']
    xs = cd['x_spread']
    vel = cd['velocity_arcsec_min']
    mag = cd['magnitude']
    x = cd['x']
    y = cd['y']

    # Compute motion angle (degrees from horizontal)
    if abs(vx) > 0.01 or abs(vy) > 0.01:
        angle = math.degrees(math.atan2(abs(vy), abs(vx)))
    else:
        angle = 0

    # Classification rules
    flags = []

    # 1. Primarily horizontal motion (vy ≈ 0) with large x-spread
    if abs(vy) < 2.0 and xs > 40:
        flags.append("HORIZ_ARTIFACT")

    # 2. Near known artifact positions but not caught by filters
    if 577 <= x <= 627 or 1859 <= x <= 1909:
        flags.append("NEAR_COLUMN")
    if 1818 <= y <= 1858:
        flags.append("NEAR_ROW")

    # 3. Very large x-spread (suggesting artifact, not point source motion)
    if xs > 100:
        flags.append("HUGE_XSPREAD")

    # 4. Velocity in plausible asteroid range
    # MBA: 0.1-0.5 arcsec/min, NEO: 0.5-2.0 arcsec/min
    if 0.1 <= vel <= 0.5:
        flags.append("MBA_VELOCITY")
    elif 0.5 < vel <= 2.0:
        flags.append("NEO_VELOCITY")
    elif vel > 2.0:
        flags.append("TOO_FAST")

    # 5. Consistent diagonal motion (both vx and vy significant)
    if abs(vx) > 3 and abs(vy) > 3:
        flags.append("DIAGONAL")
    elif abs(vx) > 3 and abs(vy) <= 3:
        flags.append("MOSTLY_HORIZ")
    elif abs(vy) > 3 and abs(vx) <= 3:
        flags.append("MOSTLY_VERT")

    # 6. Small x-spread with significant motion — more asteroid-like
    if xs < 30 and vel > 0.2:
        flags.append("COMPACT_MOTION")

    # Assign a category
    if "HORIZ_ARTIFACT" in flags or "HUGE_XSPREAD" in flags:
        category = "LIKELY_ARTIFACT"
    elif "NEAR_COLUMN" in flags or "NEAR_ROW" in flags:
        category = "POSSIBLE_ARTIFACT"
    elif "COMPACT_MOTION" in flags and ("MBA_VELOCITY" in flags or "NEO_VELOCITY" in flags):
        category = "INTERESTING"
    elif ("MBA_VELOCITY" in flags or "NEO_VELOCITY" in flags) and "DIAGONAL" in flags:
        category = "PLAUSIBLE"
    else:
        category = "UNCERTAIN"

    return category, flags, angle


# ====================================================================
# ANALYZE ALL CANDIDATES
# ====================================================================
print("=" * 80)
print("  COMPREHENSIVE CANDIDATE ANALYSIS")
print("  Post-filtering: Row/Column/Edge artifact removal applied")
print("=" * 80)
print()

total_candidates = 0
categories = {"INTERESTING": [], "PLAUSIBLE": [], "POSSIBLE_ARTIFACT": [],
              "LIKELY_ARTIFACT": [], "UNCERTAIN": []}

for r in all_results:
    field = r['field']
    for i, cd in enumerate(r.get('candidate_details', [])):
        total_candidates += 1
        category, flags, angle = classify_candidate(cd)
        entry = {
            'field': field,
            'field_category': r['category'],
            'idx': i,
            'x': cd['x'], 'y': cd['y'],
            'confidence': cd['confidence'],
            'velocity': cd['velocity_arcsec_min'],
            'magnitude': cd['magnitude'],
            'vx': cd['vx'], 'vy': cd['vy'],
            'x_spread': cd['x_spread'],
            'angle': angle,
            'flags': flags,
            'category': category,
            'ra': None, 'dec': None,
        }
        categories[category].append(entry)

print(f"  Total candidates analyzed: {total_candidates}")
print()
for cat, entries in categories.items():
    print(f"  {cat}: {len(entries)}")
print()

# ====================================================================
# RESOLVE RA/DEC FOR ALL CANDIDATES
# ====================================================================
print("=" * 80)
print("  RESOLVING RA/DEC COORDINATES")
print("=" * 80)

# Cache WCS headers per field
wcs_cache = {}
for r in all_results:
    field = r['field']
    if field in wcs_cache or not r.get('candidate_details'):
        continue
    path, files = find_field_path(field)
    if path and files:
        data = load_fits_frames(files)
        if data and data['wcs_headers'] and data['wcs_headers'][0]:
            wcs_cache[field] = data['wcs_headers'][0]
            print(f"  {field}: WCS loaded")
        else:
            print(f"  {field}: No WCS available")
    else:
        print(f"  {field}: Field not found")

print()

# Resolve coordinates
for cat_entries in categories.values():
    for entry in cat_entries:
        if entry['field'] in wcs_cache:
            wcs_hdr = wcs_cache[entry['field']]
            ra, dec = pixel_to_radec(entry['x'], entry['y'], wcs_hdr)
            if ra is not None:
                entry['ra'] = ra
                entry['dec'] = dec

# ====================================================================
# DETAILED REPORT
# ====================================================================
print("=" * 80)
print("  DETAILED FINDINGS REPORT")
print("=" * 80)

# 1. INTERESTING candidates (most asteroid-like)
print(f"\n  {'='*70}")
print(f"  CATEGORY: INTERESTING ({len(categories['INTERESTING'])} candidates)")
print(f"  Compact motion at plausible asteroid velocity")
print(f"  {'='*70}")
for e in sorted(categories['INTERESTING'], key=lambda x: -x['confidence']):
    ra_str = ra_dec_to_string(e['ra'], e['dec']) if e['ra'] else "N/A"
    print(f"\n  [{e['field']}] Candidate at ({e['x']:.0f}, {e['y']:.0f})")
    print(f"    RA/Dec: {ra_str}")
    print(f"    Confidence: {e['confidence']:.0f}%")
    print(f"    Velocity: {e['velocity']:.3f} arcsec/min")
    print(f"    Magnitude: {e['magnitude']:.1f}")
    print(f"    Motion: vx={e['vx']:.2f} vy={e['vy']:.2f} px/frame "
          f"(angle={e['angle']:.0f}°)")
    print(f"    X-spread: {e['x_spread']:.1f}px")
    print(f"    Flags: {', '.join(e['flags'])}")
    print(f"    Field category: {e['field_category']}")

# 2. PLAUSIBLE candidates
print(f"\n  {'='*70}")
print(f"  CATEGORY: PLAUSIBLE ({len(categories['PLAUSIBLE'])} candidates)")
print(f"  Asteroid-range velocity with diagonal motion")
print(f"  {'='*70}")
for e in sorted(categories['PLAUSIBLE'], key=lambda x: -x['confidence']):
    ra_str = ra_dec_to_string(e['ra'], e['dec']) if e['ra'] else "N/A"
    print(f"\n  [{e['field']}] ({e['x']:.0f}, {e['y']:.0f})")
    print(f"    RA/Dec: {ra_str}  Conf={e['confidence']:.0f}%  "
          f"Vel={e['velocity']:.3f}\"/m  Mag={e['magnitude']:.1f}")
    print(f"    Motion: vx={e['vx']:.2f} vy={e['vy']:.2f} xs={e['x_spread']:.1f}px  "
          f"Flags: {', '.join(e['flags'])}")

# 3. UNCERTAIN
print(f"\n  {'='*70}")
print(f"  CATEGORY: UNCERTAIN ({len(categories['UNCERTAIN'])} candidates)")
print(f"  {'='*70}")
for e in sorted(categories['UNCERTAIN'], key=lambda x: -x['confidence']):
    ra_str = ra_dec_to_string(e['ra'], e['dec']) if e['ra'] else "N/A"
    print(f"  [{e['field']}] ({e['x']:.0f}, {e['y']:.0f})  "
          f"RA/Dec: {ra_str}  Conf={e['confidence']:.0f}%  "
          f"Vel={e['velocity']:.3f}\"/m  Mag={e['magnitude']:.1f}  "
          f"xs={e['x_spread']:.1f}  Flags: {', '.join(e['flags'])}")

# 4. POSSIBLE_ARTIFACT
print(f"\n  {'='*70}")
print(f"  CATEGORY: POSSIBLE_ARTIFACT ({len(categories['POSSIBLE_ARTIFACT'])} candidates)")
print(f"  {'='*70}")
for e in categories['POSSIBLE_ARTIFACT']:
    print(f"  [{e['field']}] ({e['x']:.0f}, {e['y']:.0f})  "
          f"Conf={e['confidence']:.0f}%  Vel={e['velocity']:.3f}\"/m  "
          f"xs={e['x_spread']:.1f}  Flags: {', '.join(e['flags'])}")

# 5. LIKELY_ARTIFACT
print(f"\n  {'='*70}")
print(f"  CATEGORY: LIKELY_ARTIFACT ({len(categories['LIKELY_ARTIFACT'])} candidates)")
print(f"  {'='*70}")
for e in categories['LIKELY_ARTIFACT']:
    print(f"  [{e['field']}] ({e['x']:.0f}, {e['y']:.0f})  "
          f"Conf={e['confidence']:.0f}%  Vel={e['velocity']:.3f}\"/m  "
          f"xs={e['x_spread']:.1f}  Flags: {', '.join(e['flags'])}")

# ====================================================================
# SCIENCE FAIR SUMMARY
# ====================================================================
print(f"\n\n{'='*80}")
print(f"  SCIENCE FAIR FINDINGS SUMMARY")
print(f"{'='*80}")

interesting = categories['INTERESTING']
plausible = categories['PLAUSIBLE']

print(f"""
  ALGORITHM PERFORMANCE:
  - Processed: 22 IASC campaign fields (88 Pan-STARRS images)
  - Initial tracklets formed: ~600+ (after source linking)
  - After column cluster filter: ~83 candidates
  - After row/column/edge filters: {total_candidates} candidates

  CANDIDATE CLASSIFICATION:
  - INTERESTING (compact motion, asteroid velocity): {len(interesting)}
  - PLAUSIBLE (diagonal motion, asteroid velocity): {len(plausible)}
  - UNCERTAIN (ambiguous characteristics): {len(categories['UNCERTAIN'])}
  - POSSIBLE ARTIFACT (near known artifact positions): {len(categories['POSSIBLE_ARTIFACT'])}
  - LIKELY ARTIFACT (horizontal motion/huge spread): {len(categories['LIKELY_ARTIFACT'])}

  KNOWN ASTEROID STATUS:
  - MPC Report contains 6 objects, all in field XY75_p11
  - ASP0090 (2024 RX69): mag 21.1-21.2, SNR < 3 in all frames
  - All MPC objects are below the algorithm's detection threshold (sigma=5)
  - These objects were originally found via manual blink comparison
    (Astrometrica), where human pattern recognition can detect
    motion at SNR 2-3 that automated extraction cannot

  KEY FINDING:
  - The algorithm successfully identifies candidate regions that
    warrant human inspection, reducing ~10,000+ sources per field
    to a small handful of candidates
  - False positive rate is now {total_candidates}/{22} = ~{total_candidates/22:.1f} per field
  - {len(interesting) + len(plausible)} candidates have motion characteristics
    consistent with real asteroid detections
""")

# Save detailed analysis
output_path = os.path.join(os.path.dirname(__file__), "candidate_analysis.json")
all_entries = []
for cat_entries in categories.values():
    all_entries.extend(cat_entries)
with open(output_path, 'w') as f:
    json.dump(all_entries, f, indent=2, default=str)
print(f"  Detailed analysis saved to: {output_path}")
