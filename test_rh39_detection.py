#!/usr/bin/env python3
"""
Diagnostic: test if the shift-and-stack pipeline detects 2024 RH39.
Simulates Phase 1 coarse search + velocity-uniqueness filter on XY26_p10.
"""
import os, sys, glob, math, numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asteroid_detector import (load_fits_frames, estimate_background,
                                PIXEL_SCALE, FRAME_INTERVAL_MINUTES)

DATA_DIR = "C:/Users/astro/Downloads/Astrometrica/Data/3rd Sep/moving object found/ps2-20240903_8_XY26_p10/XY26_p10"
fits_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.fits")))
print(f"Loading {len(fits_files)} FITS files from XY26_p10...")
data = load_fits_frames(fits_files)
frames = data['frames']
n = len(frames)
h, w = frames[0].shape
print(f"Image: {h}x{w}, {n} frames")

# Background subtraction
print("Computing background subtraction...")
subtracted = []
noise_maps = []
for i, frame in enumerate(frames):
    bg, noise = estimate_background(frame)
    subtracted.append(frame - bg)
    noise_maps.append(noise)
print("Done.")

from scipy.ndimage import gaussian_filter, maximum_filter, binary_dilation

# Build difference images
reference = np.median(np.array(subtracted), axis=0)
diff_frames = [subtracted[i] - reference for i in range(n)]
mean_noise = np.median(np.array(noise_maps), axis=0)
psf_sigma = 3.0

# Star mask
ref_noise_med = np.median(mean_noise)
bright_threshold = 5.0 * ref_noise_med
star_mask = np.abs(reference) > bright_threshold
star_mask = binary_dilation(star_mask, iterations=8)
for col_center in [602, 1884]:
    col_lo = max(0, col_center - 40)
    col_hi = min(w, col_center + 40)
    star_mask[:, col_lo:col_hi] = True
print(f"Star mask: {100*np.mean(star_mask):.1f}% of pixels masked")

# PSF-integrated reference for Stage 3 filter
smoothed_ref = gaussian_filter(reference, sigma=psf_sigma)
sr_unmasked = smoothed_ref[~star_mask]
sr_med = float(np.median(sr_unmasked))
sr_mad = float(np.median(np.abs(sr_unmasked - sr_med)))
sr_noise = sr_mad * 1.4826
if sr_noise <= 0:
    sr_noise = float(np.std(sr_unmasked))
print(f"Smoothed ref noise: {sr_noise:.6f}, median: {sr_med:.6f}")

# Integer shift-and-add
def shift_add(diffs, ivx, ivy, n, h, w):
    stack = np.zeros((h, w), dtype=np.float64)
    for i in range(n):
        dy, dx = -ivy * i, -ivx * i
        sy0, sy1 = max(0, dy), min(h, h + dy)
        sx0, sx1 = max(0, dx), min(w, w + dx)
        dy0, dy1 = max(0, -dy), min(h, h - dy)
        dx0, dx1 = max(0, -dx), min(w, w - dx)
        if sy1 > sy0 and sx1 > sx0:
            stack[dy0:dy1, dx0:dx1] += diffs[i][sy0:sy1, sx0:sx1]
    stack /= n
    return stack

# ======================================================================
# TARGETED SEARCH near expected velocity
# ======================================================================
print("\n=== Targeted search at velocities near 2024 RH39 ===")
print(f"Expected position: ~(1623, 732), velocity ~(+26, -41) px/frame")
print()

ap_radius = 4
ann_inner = 8
ann_outer = 15

for ivx in range(16, 40, 8):
    for ivy in range(-48, -24, 8):
        stack = shift_add(diff_frames, ivx, ivy, n, h, w)
        stack[star_mask] = 0.0
        smoothed = gaussian_filter(stack, sigma=psf_sigma)
        smoothed[star_mask] = 0.0

        unmasked = smoothed[~star_mask]
        med_val = np.median(unmasked)
        mad = np.median(np.abs(unmasked - med_val))
        robust_noise = mad * 1.4826

        crude_snr = (smoothed - med_val) / robust_noise
        crude_snr[star_mask] = 0.0

        # Find peak in the expected region
        region = crude_snr[680:780, 1580:1680]
        if region.size == 0:
            continue
        pk_idx = np.unravel_index(np.argmax(region), region.shape)
        pk_y = 680 + pk_idx[0]
        pk_x = 1580 + pk_idx[1]
        pk_crude = crude_snr[pk_y, pk_x]

        # Aperture photometry on UN-SMOOTHED stack
        cy, cx = pk_y, pk_x
        cut_r = ann_outer + 2
        cut_y0 = max(0, cy - cut_r)
        cut_y1 = min(h, cy + cut_r + 1)
        cut_x0 = max(0, cx - cut_r)
        cut_x1 = min(w, cx + cut_r + 1)
        cutout = stack[cut_y0:cut_y1, cut_x0:cut_x1]
        mask_cut = star_mask[cut_y0:cut_y1, cut_x0:cut_x1]
        YY, XX = np.ogrid[0:cutout.shape[0], 0:cutout.shape[1]]
        cy_l, cx_l = cy - cut_y0, cx - cut_x0
        RR = np.sqrt((XX - cx_l)**2 + (YY - cy_l)**2)
        aperture_sel = (RR <= ap_radius) & (~mask_cut)
        annulus_sel = (RR > ann_inner) & (RR <= ann_outer) & (~mask_cut)
        n_ap = int(np.sum(aperture_sel))
        n_ann = int(np.sum(annulus_sel))

        if n_ap >= 10 and n_ann >= 20:
            ap_flux = float(np.sum(cutout[aperture_sel]))
            sky_med = float(np.median(cutout[annulus_sel]))
            sky_std = float(np.std(cutout[annulus_sel]))
            net_flux = ap_flux - sky_med * n_ap
            ap_snr = net_flux / (sky_std * math.sqrt(n_ap)) if sky_std > 0 else 0
        else:
            ap_snr = 0

        status = "*** DETECTED ***" if ap_snr >= 5.0 else ""
        print(f"  v=({ivx:+3d},{ivy:+3d})  peak@({pk_x},{pk_y})  "
              f"crude_SNR={pk_crude:.1f}  aperture_SNR={ap_snr:.1f}  "
              f"{status}")

# ======================================================================
# FULL COARSE SEARCH + VELOCITY-UNIQUENESS FILTER
# ======================================================================
print("\n=== Full coarse search (all velocities, step=8) ===")
print("Phase 1: collecting all detections with aperture SNR >= 5.0...")
print("         (crude_snr threshold = 4.0, crude_snr cap = 50)")

all_detections = []
n_tested = 0
PEAK_FIND_SIGMA = 4.0   # raised from 3.0 to reduce noise peaks
AP_THRESHOLD = 5.0

for ivx in range(-48, 52, 8):
    for ivy in range(-48, 52, 8):
        if abs(ivx) < 4 and abs(ivy) < 4:
            continue
        n_tested += 1

        stack = shift_add(diff_frames, ivx, ivy, n, h, w)
        stack[star_mask] = 0.0
        smoothed = gaussian_filter(stack, sigma=psf_sigma)
        smoothed[star_mask] = 0.0

        unmasked = smoothed[~star_mask]
        med_val = np.median(unmasked)
        mad = np.median(np.abs(unmasked - med_val))
        robust_noise = mad * 1.4826
        if robust_noise <= 0:
            continue

        crude_snr = (smoothed - med_val) / robust_noise
        crude_snr[star_mask] = 0.0

        # Suppress bright residuals BEFORE local max search
        crude_snr[crude_snr > 50] = 0.0

        # Find local maxima above threshold
        local_max = maximum_filter(crude_snr, size=15)
        peaks = (crude_snr == local_max) & (crude_snr >= PEAK_FIND_SIGMA)
        pys, pxs = np.where(peaks)
        if len(pys) == 0:
            continue

        # Edge rejection
        margin = 60
        peak_snrs = crude_snr[pys, pxs]
        keep = ((pxs >= margin) & (pxs < w-margin) &
                (pys >= margin) & (pys < h-margin))
        pys, pxs, peak_snrs = pys[keep], pxs[keep], peak_snrs[keep]
        if len(peak_snrs) == 0:
            continue

        for py, px, csnr in zip(pys, pxs, peak_snrs):
            cy, cx = int(py), int(px)
            cut_r = ann_outer + 2
            cut_y0, cut_y1 = max(0, cy-cut_r), min(h, cy+cut_r+1)
            cut_x0, cut_x1 = max(0, cx-cut_r), min(w, cx+cut_r+1)
            cutout = stack[cut_y0:cut_y1, cut_x0:cut_x1]
            mask_cut = star_mask[cut_y0:cut_y1, cut_x0:cut_x1]
            YY, XX = np.ogrid[0:cutout.shape[0], 0:cutout.shape[1]]
            cy_l, cx_l = cy - cut_y0, cx - cut_x0
            RR = np.sqrt((XX - cx_l)**2 + (YY - cy_l)**2)
            ap_sel = (RR <= ap_radius) & (~mask_cut)
            ann_sel = (RR > ann_inner) & (RR <= ann_outer) & (~mask_cut)
            n_ap = int(np.sum(ap_sel))
            n_ann = int(np.sum(ann_sel))
            if n_ap < 10 or n_ann < 20:
                continue
            af = float(np.sum(cutout[ap_sel]))
            sm = float(np.median(cutout[ann_sel]))
            ss = float(np.std(cutout[ann_sel]))
            if ss <= 0:
                continue
            nf = af - sm * n_ap
            asnr = nf / (ss * math.sqrt(n_ap))
            if asnr >= AP_THRESHOLD:
                all_detections.append((ivx, ivy, px, py, csnr, asnr))

print(f"  Tested {n_tested} velocity vectors")
print(f"  Found {len(all_detections)} raw detections with aperture SNR >= {AP_THRESHOLD}")

# ======================================================================
# VELOCITY-UNIQUENESS FILTER
# ======================================================================
print("\n=== Velocity filter (min=2, max=8 velocity hits per cell) ===")

MAX_VELOCITY_HITS = 8
MIN_VELOCITY_HITS = 2
VEL_CELL = 40
VEL_HALF = VEL_CELL // 2

rejected = set()
unconfirmed = set(range(len(all_detections)))

for grid_offset in [0, VEL_HALF]:
    cell_data = defaultdict(lambda: {'idx': [], 'vel': set()})
    for idx, (vx, vy, x, y, csnr, asnr) in enumerate(all_detections):
        key = ((int(x) + grid_offset) // VEL_CELL,
               (int(y) + grid_offset) // VEL_CELL)
        cell_data[key]['idx'].append(idx)
        cell_data[key]['vel'].add((int(vx), int(vy)))

    for key, data in cell_data.items():
        n_vel = len(data['vel'])
        if n_vel > MAX_VELOCITY_HITS:
            for idx in data['idx']:
                rejected.add(idx)
        if n_vel >= MIN_VELOCITY_HITS:
            for idx in data['idx']:
                unconfirmed.discard(idx)

rejected |= unconfirmed

filtered = [all_detections[i] for i in range(len(all_detections))
            if i not in rejected]

n_too_many = len(rejected - unconfirmed)
n_too_few = len(unconfirmed)
print(f"  Raw: {len(all_detections)} -> Filtered: {len(filtered)} "
      f"({n_too_many} fixed-source, {n_too_few} single-velocity rejected)")

# Best detection per spatial cell
best_per_cell = {}
for det in filtered:
    vx, vy, x, y, csnr, asnr = det
    cell = (int(x) // VEL_CELL, int(y) // VEL_CELL)
    if cell not in best_per_cell or asnr > best_per_cell[cell][5]:
        best_per_cell[cell] = det

print(f"  Unique spatial positions: {len(best_per_cell)}")

# ======================================================================
# PER-FRAME CONSISTENCY CHECK
# ======================================================================
# A real asteroid is visible in MULTIPLE frames when shifted to the
# correct velocity. A fixed-source residual is only visible in frame 0
# (the other frames sample empty sky at the shifted positions).
#
# For each candidate, measure aperture flux in each diff_frame at the
# position predicted by the trial velocity, using a large aperture
# (r=7) to tolerate coarse velocity position errors of 4-7 pixels.
print("\n=== Per-frame consistency scoring ===")

CONSISTENCY_AP_R = 7  # large aperture to capture flux with coarse velocity errors

scored_candidates = []
for det in best_per_cell.values():
    ivx, ivy, px, py, csnr, asnr = det

    frame_fluxes = []
    for i in range(n):
        # Position in diff_frames[i] where this candidate's signal should be
        src_x = int(round(px + ivx * i))
        src_y = int(round(py + ivy * i))

        if (CONSISTENCY_AP_R < src_x < w - CONSISTENCY_AP_R and
                CONSISTENCY_AP_R < src_y < h - CONSISTENCY_AP_R):
            # Circular aperture sum in diff_frames[i]
            cut_r = CONSISTENCY_AP_R + 1
            cut = diff_frames[i][src_y - cut_r:src_y + cut_r + 1,
                                 src_x - cut_r:src_x + cut_r + 1]
            mask_cut = star_mask[src_y - cut_r:src_y + cut_r + 1,
                                 src_x - cut_r:src_x + cut_r + 1]
            YY, XX = np.ogrid[0:cut.shape[0], 0:cut.shape[1]]
            cy_l, cx_l = cut_r, cut_r
            RR = np.sqrt((XX - cx_l)**2 + (YY - cy_l)**2)
            ap = (RR <= CONSISTENCY_AP_R) & (~mask_cut)
            if np.sum(ap) > 20:
                frame_fluxes.append(float(np.sum(cut[ap])))
            else:
                frame_fluxes.append(0.0)
        else:
            frame_fluxes.append(0.0)

    # Compute dominance: fraction of total positive flux in the peak frame
    pos_fluxes = [max(0, f) for f in frame_fluxes]
    total_pos = sum(pos_fluxes)
    if total_pos > 0:
        dominance = max(pos_fluxes) / total_pos
    else:
        dominance = 1.0

    # Consistency score: low dominance = multi-frame signal (asteroid-like)
    consistency = 1.0 - dominance
    score = asnr * consistency

    scored_candidates.append((ivx, ivy, px, py, csnr, asnr,
                              dominance, consistency, score))

# Sort by consistency-weighted score
unique_positions = sorted(scored_candidates, key=lambda d: -d[8])
print(f"  Scored {len(unique_positions)} candidates")

# ======================================================================
# RESULTS
# ======================================================================
print(f"\n=== Top 20 candidates (sorted by consistency-weighted score) ===")
for i, (ivx, ivy, px, py, csnr, asnr, dom, cons, score) in enumerate(unique_positions[:20]):
    dist = math.sqrt((px - 1623)**2 + (py - 732)**2)
    match = "<-- 2024 RH39!" if dist < 50 else ""
    vel = math.sqrt(ivx**2 + ivy**2) * PIXEL_SCALE / FRAME_INTERVAL_MINUTES
    print(f"  #{i+1:2d}  pos=({px},{py})  v=({ivx:+3d},{ivy:+3d}) ({vel:.3f}\"/min)  "
          f"ap={asnr:.1f}  dom={dom:.2f}  score={score:.1f}  {match}")

# Check if RH39 is in the results
rh39_in_top = [d for d in unique_positions[:20]
               if math.sqrt((d[2] - 1623)**2 + (d[3] - 732)**2) < 50]
if rh39_in_top:
    rank = next(i+1 for i, d in enumerate(unique_positions[:20])
                if math.sqrt((d[2] - 1623)**2 + (d[3] - 732)**2) < 50)
    d = rh39_in_top[0]
    print(f"\n*** 2024 RH39 FOUND at rank #{rank} ***")
    print(f"    aperture_SNR={d[5]:.1f}  dominance={d[6]:.2f}  score={d[8]:.1f}")
else:
    rh39_all = [d for d in unique_positions
                if math.sqrt((d[2] - 1623)**2 + (d[3] - 732)**2) < 50]
    if rh39_all:
        rank = next(i+1 for i, d in enumerate(unique_positions)
                    if math.sqrt((d[2] - 1623)**2 + (d[3] - 732)**2) < 50)
        d = rh39_all[0]
        print(f"\n*** 2024 RH39 found at rank #{rank} (not in top 20) ***")
        print(f"    aperture_SNR={d[5]:.1f}  dominance={d[6]:.2f}  score={d[8]:.1f}")
    else:
        print(f"\n*** 2024 RH39 NOT found in filtered results ***")
