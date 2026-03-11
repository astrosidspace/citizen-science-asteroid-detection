#!/usr/bin/env python3
"""Debug: what happens at v=(24,-40) in the full search pipeline?"""
import os, sys, glob, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asteroid_detector import load_fits_frames, estimate_background, PIXEL_SCALE, FRAME_INTERVAL_MINUTES

DATA_DIR = "C:/Users/astro/Downloads/Astrometrica/Data/3rd Sep/moving object found/ps2-20240903_8_XY26_p10/XY26_p10"
fits_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.fits")))
data = load_fits_frames(fits_files)
frames = data['frames']
n, h, w = len(frames), frames[0].shape[0], frames[0].shape[1]

subtracted, noise_maps = [], []
for f in frames:
    bg, noise = estimate_background(f)
    subtracted.append(f - bg)
    noise_maps.append(noise)

from scipy.ndimage import gaussian_filter, maximum_filter, binary_dilation

reference = np.median(np.array(subtracted), axis=0)
diff_frames = [subtracted[i] - reference for i in range(n)]
mean_noise = np.median(np.array(noise_maps), axis=0)
psf_sigma = 3.0

ref_noise_med = np.median(mean_noise)
star_mask = np.abs(reference) > (5.0 * ref_noise_med)
star_mask = binary_dilation(star_mask, iterations=8)
for col_center in [602, 1884]:
    star_mask[:, max(0, col_center-25):min(w, col_center+25)] = True

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

# Test at v=(24, -40) — where RH39 should be found
ivx, ivy = 24, -40
print(f"\n=== Debugging v=({ivx},{ivy}) ===")
stack = shift_add(diff_frames, ivx, ivy, n, h, w)
stack[star_mask] = 0.0
smoothed = gaussian_filter(stack, sigma=psf_sigma)
smoothed[star_mask] = 0.0

unmasked = smoothed[~star_mask]
med_val = np.median(unmasked)
mad = np.median(np.abs(unmasked - med_val))
robust_noise = mad * 1.4826
print(f"Robust noise: {robust_noise:.4f}, median: {med_val:.4f}")

crude_snr = (smoothed - med_val) / robust_noise
crude_snr[star_mask] = 0.0

# Check at expected asteroid position
ex, ey = 1623, 732
print(f"\nAt expected position ({ex}, {ey}):")
print(f"  stack value: {stack[ey, ex]:.4f}")
print(f"  smoothed value: {smoothed[ey, ex]:.4f}")
print(f"  crude SNR (before clamp): {crude_snr[ey, ex]:.2f}")
print(f"  star_mask: {star_mask[ey, ex]}")

# Check neighborhood for higher SNR pixels
print(f"\nCrude SNR in 30x30 region around ({ex},{ey}):")
region = crude_snr[ey-15:ey+15, ex-15:ex+15]
print(f"  Max: {np.max(region):.2f} at relative pos {np.unravel_index(np.argmax(region), region.shape)}")
print(f"  Mean: {np.mean(region):.2f}")
print(f"  Pixels > 50: {np.sum(region > 50)}")
print(f"  Pixels > 10: {np.sum(region > 10)}")
print(f"  Pixels > 5: {np.sum(region > 5)}")

# After clamping
crude_snr_clamped = crude_snr.copy()
crude_snr_clamped[crude_snr_clamped > 50] = 0.0

# Check local maxima
local_max = maximum_filter(crude_snr_clamped, size=15)
print(f"\nAfter clamping:")
print(f"  crude_snr at ({ex},{ey}): {crude_snr_clamped[ey, ex]:.2f}")
print(f"  local_max at ({ex},{ey}): {local_max[ey, ex]:.2f}")
print(f"  Is local max: {crude_snr_clamped[ey, ex] == local_max[ey, ex]}")
print(f"  Exceeds threshold 3.0: {crude_snr_clamped[ey, ex] >= 3.0}")

# Find ALL peaks near the expected position
peaks = (crude_snr_clamped == local_max) & (crude_snr_clamped >= 3.0)
pys, pxs = np.where(peaks)
near = [(px, py, crude_snr_clamped[py, px])
        for py, px in zip(pys, pxs)
        if abs(px - ex) < 50 and abs(py - ey) < 50]
print(f"\nPeaks within 50px of ({ex},{ey}): {len(near)}")
for px, py, snr in sorted(near, key=lambda x: -x[2]):
    print(f"  ({px}, {py}) crude_snr = {snr:.2f}")

# Also check: what IS the local maximum at the asteroid position?
print(f"\nIn 15x15 window around ({ex},{ey}):")
win_snr = crude_snr_clamped[ey-7:ey+8, ex-7:ex+8]
pk = np.unravel_index(np.argmax(win_snr), win_snr.shape)
pk_y, pk_x = ey - 7 + pk[0], ex - 7 + pk[1]
print(f"  Highest pixel: ({pk_x}, {pk_y}) = {win_snr[pk[0], pk[1]]:.2f}")
print(f"  This is the local_max value: {local_max[ey, ex]:.2f}")

# Do aperture photometry at the peak position
ap_r, ann_in, ann_out = 4, 8, 15
cy, cx = ey, ex  # Use expected position
cut_r = ann_out + 2
cut_y0, cut_y1 = max(0, cy-cut_r), min(h, cy+cut_r+1)
cut_x0, cut_x1 = max(0, cx-cut_r), min(w, cx+cut_r+1)
cutout = stack[cut_y0:cut_y1, cut_x0:cut_x1]
mask_cut = star_mask[cut_y0:cut_y1, cut_x0:cut_x1]
YY, XX = np.ogrid[0:cutout.shape[0], 0:cutout.shape[1]]
cy_l, cx_l = cy-cut_y0, cx-cut_x0
RR = np.sqrt((XX - cx_l)**2 + (YY - cy_l)**2)
ap_sel = (RR <= ap_r) & (~mask_cut)
ann_sel = (RR > ann_in) & (RR <= ann_out) & (~mask_cut)
af = float(np.sum(cutout[ap_sel]))
sm = float(np.median(cutout[ann_sel]))
ss = float(np.std(cutout[ann_sel]))
n_ap = int(np.sum(ap_sel))
nf = af - sm * n_ap
asnr = nf / (ss * math.sqrt(n_ap)) if ss > 0 else 0
print(f"\nAperture photometry at ({ex},{ey}) on un-smoothed stack:")
print(f"  aperture flux: {af:.1f}, sky_median: {sm:.4f}, sky_std: {ss:.4f}")
print(f"  n_ap: {n_ap}, net_flux: {nf:.1f}")
print(f"  aperture SNR: {asnr:.1f}")
