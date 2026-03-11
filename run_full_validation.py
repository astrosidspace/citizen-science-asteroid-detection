#!/usr/bin/env python3
"""
Full validation run: Process all 21 IASC campaign fields with the calibrated
asteroid detection algorithm, cross-reference with MPC report, and generate
a comprehensive results summary.
"""
import os
import sys
import glob
import json
import time

# Add the algorithm directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asteroid_detector import (
    load_fits_frames, run_detection_pipeline, parse_mpc_report,
    cross_reference_candidates, generate_detailed_report, print_report,
    PIXEL_SCALE, FRAME_INTERVAL_MINUTES
)

DATA_ROOT = "C:/Users/astro/Downloads/Astrometrica/Data"
MPC_REPORT = os.path.join(DATA_ROOT, "MPCReport.txt")
REFERENCE_ROOT = "C:/Users/astro/Downloads/TVSEF 2026/Reference/RASC/RASC"

def find_all_fields():
    """Find all field directories containing exactly 4 FITS files."""
    fields = []
    for root, dirs, files in os.walk(DATA_ROOT):
        fits_files = sorted(glob.glob(os.path.join(root, "*.fits")))
        if len(fits_files) == 4:
            # Extract field name from directory
            field_name = os.path.basename(root)
            # Determine parent category
            parent = os.path.basename(os.path.dirname(os.path.dirname(root)))
            fields.append({
                'name': field_name,
                'path': root,
                'files': fits_files,
                'parent_category': parent
            })

    # Also check reference directory
    if os.path.isdir(REFERENCE_ROOT):
        for subdir in sorted(os.listdir(REFERENCE_ROOT)):
            ref_path = os.path.join(REFERENCE_ROOT, subdir)
            if os.path.isdir(ref_path):
                fits_files = sorted(glob.glob(os.path.join(ref_path, "*.fits")))
                if len(fits_files) == 4:
                    fields.append({
                        'name': subdir,
                        'path': ref_path,
                        'files': fits_files,
                        'parent_category': 'Reference (negative)'
                    })

    return fields

def main():
    print("=" * 80)
    print("  COMPREHENSIVE ASTEROID DETECTION VALIDATION")
    print("  All IASC Campaign Fields + Reference Negatives")
    print("=" * 80)
    print(f"  Pixel scale: {PIXEL_SCALE} arcsec/pixel")
    print(f"  Default frame interval: {FRAME_INTERVAL_MINUTES} min")
    print()

    # Find all fields
    fields = find_all_fields()
    print(f"  Found {len(fields)} fields to process")
    print()

    # Parse MPC report - returns dict: designation -> list of observations
    mpc_objects = {}
    if os.path.exists(MPC_REPORT):
        mpc_objects = parse_mpc_report(MPC_REPORT)
        print(f"  Parsed {len(mpc_objects)} objects from MPC report")
        for desig, obs_list in mpc_objects.items():
            print(f"    {desig}: {len(obs_list)} obs, "
                  f"RA={obs_list[0]['ra_deg']:.4f} "
                  f"Dec={obs_list[0]['dec_deg']:.4f} "
                  f"mag={obs_list[0]['mag']}")
        print()

    # Process each field
    all_results = []
    total_candidates = 0
    total_start = time.time()

    for idx, field in enumerate(fields):
        print(f"\n{'='*80}")
        print(f"  FIELD {idx+1}/{len(fields)}: {field['name']}")
        print(f"  Category: {field['parent_category']}")
        print(f"  Path: {field['path']}")
        print(f"{'='*80}")

        # Load FITS frames
        data = load_fits_frames(field['files'])
        if data is None:
            print(f"  SKIP: Could not load FITS files")
            all_results.append({
                'field': field['name'],
                'category': field['parent_category'],
                'error': 'Could not load FITS files',
                'candidates': 0
            })
            continue

        frames = data['frames']
        mjds = data['mjds']
        wcs_headers = data['wcs_headers']

        # Run detection pipeline
        result = run_detection_pipeline(frames, verbose=True, frame_mjds=mjds)

        n_candidates = result.total_candidates_passed
        total_candidates += n_candidates

        # Cross-reference with MPC objects if WCS available
        wcs_header = wcs_headers[0] if wcs_headers else None
        matches = []
        if n_candidates > 0 and wcs_header and mpc_objects:
            matches = cross_reference_candidates(
                result.candidates, mpc_objects, wcs_header
            )

        # Generate detailed report
        report = generate_detailed_report(
            result, field['name'], wcs_header, mpc_objects, mjds
        )

        # Print candidate details
        if n_candidates > 0:
            print(f"\n  --- {n_candidates} CANDIDATE(S) DETECTED ---")
            for i, cand in enumerate(result.candidates):
                src0 = cand.sources[0]
                print(f"  Candidate {i+1}:")
                print(f"    Position: ({src0.x:.1f}, {src0.y:.1f})")
                print(f"    Confidence: {cand.confidence_score:.0f}%")
                print(f"    Velocity: {cand.velocity_arcsec_min:.3f} arcsec/min")
                print(f"    Magnitude: {cand.mean_magnitude:.1f}")
                print(f"    Motion: vx={cand.velocity_x:.2f} vy={cand.velocity_y:.2f} px/frame")

                # Show x-spread for column artifact analysis
                x_coords = [s.x for s in cand.sources]
                x_spread = max(x_coords) - min(x_coords)
                print(f"    X-spread: {x_spread:.1f} px")

                # Check MPC matches
                for match in matches:
                    if match['candidate_idx'] == i:
                        print(f"    MPC MATCH: {match['designation']} "
                              f"(sep={match['separation_arcsec']:.1f}\")")
        else:
            print(f"\n  No candidates detected.")

        # Report shift-and-stack supplementary candidates
        n_stack = len(result.stack_candidates)
        if n_stack > 0:
            print(f"\n  --- {n_stack} SHIFT-AND-STACK SUPPLEMENTARY ---")
            for i, sc in enumerate(result.stack_candidates):
                src0 = sc.sources[0]
                print(f"    Stack {i+1}: ({src0.x:.1f},{src0.y:.1f}) "
                      f"conf={sc.confidence_score:.0f}% "
                      f"vel={sc.velocity_arcsec_min:.3f}\"/min "
                      f"mag={sc.mean_magnitude:.1f}")

        # Store results
        field_result = {
            'field': field['name'],
            'category': field['parent_category'],
            'candidates': n_candidates,
            'stack_candidates': n_stack,
            'candidate_details': [],
            'mpc_matches': matches,
            'frame_interval_min': report.get('frame_interval_min'),
        }
        for cand in result.candidates:
            src0 = cand.sources[0]
            x_coords = [s.x for s in cand.sources]
            field_result['candidate_details'].append({
                'x': src0.x, 'y': src0.y,
                'confidence': cand.confidence_score,
                'velocity_arcsec_min': cand.velocity_arcsec_min,
                'magnitude': cand.mean_magnitude,
                'vx': cand.velocity_x, 'vy': cand.velocity_y,
                'x_spread': max(x_coords) - min(x_coords),
            })
        all_results.append(field_result)

    total_time = time.time() - total_start

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("  COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 80)
    print(f"  Fields processed: {len(fields)}")
    print(f"  Total candidates: {total_candidates}")
    print(f"  Total time: {total_time:.1f}s")
    print()

    # Per-field summary table
    print(f"  {'Field':<15} {'Category':<30} {'Candidates':>10} {'MPC Match':>10}")
    print(f"  {'-'*15} {'-'*30} {'-'*10} {'-'*10}")
    for r in all_results:
        mpc_str = ""
        if r.get('mpc_matches'):
            desigs = [m['designation'] for m in r['mpc_matches']]
            mpc_str = ", ".join(desigs)
        print(f"  {r['field']:<15} {r['category']:<30} {r['candidates']:>10} {mpc_str:>10}")

    # High-confidence candidates
    print(f"\n  HIGH-CONFIDENCE CANDIDATES (>= 60%):")
    print(f"  {'Field':<15} {'Pos (x,y)':<20} {'Conf':>6} {'Vel (as/m)':>10} {'Mag':>6} {'MPC':>12}")
    for r in all_results:
        for i, cd in enumerate(r.get('candidate_details', [])):
            if cd['confidence'] >= 60:
                mpc = ""
                for m in r.get('mpc_matches', []):
                    if m['candidate_idx'] == i:
                        mpc = m['designation']
                print(f"  {r['field']:<15} ({cd['x']:.0f},{cd['y']:.0f}){'':<10} "
                      f"{cd['confidence']:>5.0f}% {cd['velocity_arcsec_min']:>10.3f} "
                      f"{cd['magnitude']:>5.1f} {mpc:>12}")

    # All candidates with details
    print(f"\n  ALL CANDIDATES:")
    for r in all_results:
        for i, cd in enumerate(r.get('candidate_details', [])):
            mpc = ""
            for m in r.get('mpc_matches', []):
                if m['candidate_idx'] == i:
                    mpc = f" [MPC:{m['designation']}]"
            print(f"  {r['field']:<15} ({cd['x']:>7.1f},{cd['y']:>7.1f}) "
                  f"conf={cd['confidence']:>5.1f}% vel={cd['velocity_arcsec_min']:.3f}\"/m "
                  f"mag={cd['magnitude']:.1f} xspread={cd['x_spread']:.1f}px "
                  f"vx={cd['vx']:.2f} vy={cd['vy']:.2f}{mpc}")

    # Save results to JSON
    output_path = os.path.join(os.path.dirname(__file__), "validation_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")

if __name__ == "__main__":
    main()
