#!/usr/bin/env python3
"""
===============================================================================
 AUTOMATED ASTEROID DETECTION ALGORITHM
 Citizen Science Asteroid Detection Pipeline
===============================================================================

 Author:      Siddharth Patel (AstroSidSpace)
              RASC London Centre Youth Member
              Discoverer of provisional asteroids 2024 RH39 and 2024 RX69
              (NASA IASC Campaign, Pan-STARRS telescope, Aug-Sep 2024)

 Project:     TVSEF / CWSF 2026 — Physical Sciences / Engineering
              "Automated Detection of Near-Earth Asteroids Using
              Computational Image Analysis"

 AI Citation: AI code generation assisted by Claude (Anthropic, 2026).
              All outputs reviewed, tested, and validated by the student
              researcher. Full AI prompt log maintained in project logbook.

 Repository:  https://github.com/astrosidspace/citizen-science-asteroid-detection

 Description: This algorithm automates the asteroid detection process that
              professional and citizen astronomers perform manually using
              software like Astrometrica. It processes 4 consecutive telescope
              images (FITS files), finds all point sources (stars, asteroids),
              links moving objects across frames, and applies 9 professional
              verification criteria to determine if a candidate is a real
              asteroid or a false detection.

 How It Works (the short version):
   1. Load 4 images taken ~30 minutes apart
   2. Find every bright spot in each image
   3. Check which spots moved in a straight line at constant speed
   4. Run 9 tests on each moving candidate
   5. Score each candidate from 0-100% confidence
   6. Output results with pretty charts

===============================================================================
"""

# =============================================================================
# IMPORTS — the libraries (toolboxes) we need
# =============================================================================

import argparse          # Lets us add command-line flags like --validate
import json              # For saving results in a structured format
import os                # For file path operations
import sys               # For system-level operations
import time              # For measuring how fast the algorithm runs
import warnings          # To suppress noisy library warnings during demo
from dataclasses import dataclass, field  # Clean way to define data structures
from typing import List, Optional, Tuple  # Type hints make code self-documenting

import matplotlib
matplotlib.use('Agg')    # Use non-interactive backend so it works without a display
import matplotlib.pyplot as plt       # For making charts and plots
import matplotlib.patches as patches  # For drawing circles on images
import numpy as np                    # The core math library for astronomy
from scipy import ndimage             # Image processing (finding bright spots)
from scipy.optimize import curve_fit  # Fitting Gaussian curves to star profiles
from scipy.spatial import cKDTree     # Fast nearest-neighbour matching

# Suppress warnings that would clutter the demo output
warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS — numbers that define how the algorithm behaves
# =============================================================================
# These come from real Pan-STARRS telescope specifications and my IASC experience

PIXEL_SCALE = 0.25          # Pan-STARRS pixel scale: 0.25 arcseconds per pixel
TYPICAL_FWHM = 1.0          # Typical "seeing" at Pan-STARRS: 1.0 arcseconds
TYPICAL_FWHM_PIX = TYPICAL_FWHM / PIXEL_SCALE  # That is 4.0 pixels

# Thresholds for the 9 detection criteria
SNR_THRESHOLD = 5.0         # Minimum signal-to-noise ratio (same as Astrometrica)
FIT_RMS_THRESHOLD = 0.2     # Maximum RMS error for Gaussian PSF fit
FWHM_MIN = 0.8              # Minimum FWHM in arcseconds (smaller = artifact)
FWHM_MAX = 1.2              # Maximum FWHM in arcseconds (larger = galaxy)
VELOCITY_VARIATION_MAX = 0.10  # Max 10% speed change between frames
MAGNITUDE_VARIATION_MAX = 1.0  # Max 1.0 magnitude change between frames
LINEARITY_THRESHOLD = 2.0   # Max deviation from straight line in pixels

# Timing between frames: Pan-STARRS IASC images are ~30 minutes apart
FRAME_INTERVAL_MINUTES = 30.0
NUM_FRAMES = 4              # IASC always uses sets of 4 images

# Image dimensions for synthetic data (Pan-STARRS chip is much larger,
# but we use a smaller region for demonstration speed)
SYNTHETIC_IMAGE_SIZE = 512  # 512x512 pixels for synthetic test images


# =============================================================================
# DATA STRUCTURES — containers that hold our detection results
# =============================================================================

@dataclass
class Source:
    """
    A single bright spot found in one image frame.

    Think of this like a pin on a map — it marks where we found something
    bright in one specific image. It could be a star, an asteroid, a cosmic
    ray, or even just noise.
    """
    x: float              # X position in pixels (column)
    y: float              # Y position in pixels (row)
    flux: float           # Total brightness (counts)
    peak: float           # Brightest pixel value
    snr: float            # Signal-to-noise ratio
    fwhm: float           # Full Width at Half Maximum in arcseconds
    fit_rms: float        # How well a Gaussian curve fits the shape
    frame_index: int      # Which of the 4 frames this was found in
    magnitude: float = 0.0  # Brightness in astronomical magnitude units


@dataclass
class Tracklet:
    """
    A linked set of sources across multiple frames that might be one object.

    When we find a bright spot in frame 1, and another bright spot in frame 2
    that is in approximately the right position for something that moved, we
    link them together into a 'tracklet'. A complete tracklet has sources in
    all 4 frames — that is the first criterion for a real asteroid.
    """
    sources: List[Source] = field(default_factory=list)
    # Motion properties (filled in during analysis)
    velocity_x: float = 0.0      # Speed in x-direction (pixels per frame)
    velocity_y: float = 0.0      # Speed in y-direction (pixels per frame)
    velocity_arcsec_min: float = 0.0  # Speed in arcseconds per minute
    position_angle: float = 0.0  # Direction of motion in degrees
    mean_magnitude: float = 0.0  # Average brightness across frames
    # Validation results for the 9 criteria
    criteria_results: dict = field(default_factory=dict)
    confidence_score: float = 0.0
    is_candidate: bool = False


@dataclass
class DetectionResult:
    """
    The complete output of running the algorithm on one image set.

    This holds everything: all the tracklets found, which ones passed
    validation, timing information, and comparison metrics.
    """
    tracklets: List[Tracklet] = field(default_factory=list)
    candidates: List[Tracklet] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    total_sources_detected: int = 0
    total_tracklets_formed: int = 0
    total_candidates_passed: int = 0
    false_positive_count: int = 0
    detection_accuracy: float = 0.0
    false_positive_rate: float = 0.0


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================
# This creates fake but realistic telescope images so we can test the
# algorithm even without real Pan-STARRS FITS files. The synthetic images
# simulate:
#   - Background sky noise (Poisson statistics, like real photon counting)
#   - Stationary stars with realistic brightness profiles
#   - Moving asteroids that shift position between frames
#   - Cosmic rays (single bright pixels that only appear in one frame)
#   - Hot pixels (stuck bright pixels that appear in the same spot every frame)

def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y):
    """
    A 2D Gaussian function — this is the mathematical shape of a star
    or asteroid as seen through a telescope.

    Stars are infinitely far away, so they should look like perfect dots.
    But Earth's atmosphere blurs them into little fuzzy circles. That blur
    follows a Gaussian (bell curve) shape. This function creates that shape.

    Parameters:
        coords: (y, x) coordinate grids
        amplitude: how bright the peak is
        x0, y0: centre position
        sigma_x, sigma_y: width of the blur (related to seeing conditions)
    """
    y, x = coords
    return amplitude * np.exp(-(((x - x0)**2) / (2 * sigma_x**2) +
                                ((y - y0)**2) / (2 * sigma_y**2)))


def generate_synthetic_frames(
    num_asteroids=3,
    num_stars=80,
    num_cosmic_rays=5,
    num_hot_pixels=3,
    image_size=SYNTHETIC_IMAGE_SIZE,
    seed=42
):
    """
    Create 4 fake telescope images with known asteroids embedded in them.

    This is our 'test laboratory'. We know EXACTLY where the asteroids are
    because we put them there, so we can measure if the algorithm finds them
    correctly. This is called 'ground truth validation' in science.

    The synthetic asteroids are designed to match real main-belt asteroid
    properties: magnitude 19-21, motion rate 0.3-0.5 arcsec/min, which
    matches what I saw with 2024 RH39 and 2024 RX69.

    Returns:
        frames: list of 4 numpy arrays (the images)
        truth: dictionary with the known positions and properties of
               every object we planted in the images
    """
    rng = np.random.RandomState(seed)  # Reproducible random numbers
    frames = []
    truth = {
        'asteroids': [],
        'stars': [],
        'cosmic_rays': [],
        'hot_pixels': []
    }

    # --- Define the background sky ---
    # Real telescope images have a sky background of ~100-300 counts per pixel
    # with Poisson noise (the natural randomness of photon arrival)
    sky_level = 200.0    # Typical sky background in ADU (analog-digital units)
    read_noise = 10.0    # Electronic noise from the camera sensor

    # --- Place stationary stars ---
    # Stars do not move between frames (they are too far away for their
    # motion to be visible over 2 hours). They serve as reference points.
    star_positions = []
    star_fluxes = []
    for i in range(num_stars):
        sx = rng.uniform(20, image_size - 20)
        sy = rng.uniform(20, image_size - 20)
        # Stars have a range of brightnesses (magnitudes 15-20)
        star_mag = rng.uniform(15.0, 20.0)
        # Convert magnitude to flux: brighter = lower magnitude number
        # This is the standard astronomical magnitude formula
        star_flux = 10**((20.0 - star_mag) / 2.5) * 500
        star_positions.append((sx, sy))
        star_fluxes.append(star_flux)
        truth['stars'].append({
            'x': sx, 'y': sy, 'magnitude': star_mag, 'flux': star_flux
        })

    # --- Define moving asteroids ---
    # Each asteroid gets a starting position and a velocity.
    # IMPORTANT: We must ensure the asteroid stays within the image
    # for ALL 4 frames. The total motion from frame 0 to frame 3 can
    # be up to 180 pixels, so we carefully compute valid start positions.
    asteroid_params = []
    for i in range(num_asteroids):
        # Motion rate: 0.3-0.5 arcsec/min is typical for main belt asteroids
        # Convert to pixels per frame:
        #   0.4 arcsec/min * 30 min = 12 arcsec per frame
        #   12 arcsec / 0.25 arcsec/pixel = 48 pixels per frame
        rate_arcsec_min = rng.uniform(0.3, 0.5)
        rate_pixels_per_frame = (rate_arcsec_min * FRAME_INTERVAL_MINUTES
                                 / PIXEL_SCALE)
        # Random direction of motion
        angle = rng.uniform(0, 2 * np.pi)
        vx = rate_pixels_per_frame * np.cos(angle)
        vy = rate_pixels_per_frame * np.sin(angle)

        # Calculate total displacement over 3 intervals (frame 0 to 3)
        total_dx = vx * (NUM_FRAMES - 1)
        total_dy = vy * (NUM_FRAMES - 1)

        # Compute valid starting position range so the asteroid stays
        # within the image (with a 20-pixel border) for all 4 frames
        margin = 20
        if total_dx >= 0:
            x_lo = margin
            x_hi = image_size - margin - total_dx
        else:
            x_lo = margin - total_dx
            x_hi = image_size - margin
        if total_dy >= 0:
            y_lo = margin
            y_hi = image_size - margin - total_dy
        else:
            y_lo = margin - total_dy
            y_hi = image_size - margin

        # Safety check: if the motion is so large the asteroid cannot
        # fit, reduce the rate
        x_lo = max(margin, x_lo)
        x_hi = min(image_size - margin, x_hi)
        y_lo = max(margin, y_lo)
        y_hi = min(image_size - margin, y_hi)
        if x_lo >= x_hi or y_lo >= y_hi:
            # Motion too large for this image size — scale it down
            scale_factor = 0.5
            vx *= scale_factor
            vy *= scale_factor
            rate_arcsec_min *= scale_factor
            total_dx = vx * (NUM_FRAMES - 1)
            total_dy = vy * (NUM_FRAMES - 1)
            if total_dx >= 0:
                x_lo, x_hi = margin, image_size - margin - total_dx
            else:
                x_lo, x_hi = margin - total_dx, image_size - margin
            if total_dy >= 0:
                y_lo, y_hi = margin, image_size - margin - total_dy
            else:
                y_lo, y_hi = margin - total_dy, image_size - margin
            x_lo = max(margin, x_lo)
            x_hi = min(image_size - margin, x_hi)
            y_lo = max(margin, y_lo)
            y_hi = min(image_size - margin, y_hi)

        # Place the asteroid at least 30 pixels from any star to avoid
        # blending (in real data, blended sources are hard for ANY software)
        min_star_dist = 30.0
        for _attempt in range(50):
            ax = rng.uniform(x_lo, max(x_lo + 1, x_hi))
            ay = rng.uniform(y_lo, max(y_lo + 1, y_hi))
            # Check distance from all stars (in all 4 frame positions)
            too_close = False
            for frame_idx in range(NUM_FRAMES):
                ax_f = ax + vx * frame_idx
                ay_f = ay + vy * frame_idx
                for (sx, sy) in star_positions:
                    if np.sqrt((ax_f - sx)**2 + (ay_f - sy)**2) < min_star_dist:
                        too_close = True
                        break
                if too_close:
                    break
            if not too_close:
                break  # Good position found
        # Asteroid magnitude 19-21 (faint!)
        ast_mag = rng.uniform(19.0, 21.0)
        ast_flux = 10**((20.0 - ast_mag) / 2.5) * 500

        asteroid_params.append({
            'start_x': ax, 'start_y': ay,
            'vx': vx, 'vy': vy,
            'magnitude': ast_mag, 'flux': ast_flux,
            'rate_arcsec_min': rate_arcsec_min,
            'angle_deg': np.degrees(angle)
        })
        truth['asteroids'].append({
            'start_x': ax, 'start_y': ay,
            'vx': vx, 'vy': vy,
            'magnitude': ast_mag,
            'rate_arcsec_min': rate_arcsec_min,
            'positions': []  # Will be filled per frame
        })

    # --- Define hot pixels (stuck bright pixels) ---
    hot_pixel_positions = []
    for i in range(num_hot_pixels):
        hx = rng.randint(10, image_size - 10)
        hy = rng.randint(10, image_size - 10)
        hot_pixel_positions.append((hx, hy))
        truth['hot_pixels'].append({'x': hx, 'y': hy})

    # --- Build each of the 4 frames ---
    sigma = TYPICAL_FWHM_PIX / 2.355  # Convert FWHM to Gaussian sigma
    # (2.355 is the exact factor: FWHM = 2 * sqrt(2 * ln(2)) * sigma)

    for frame_idx in range(NUM_FRAMES):
        # Start with sky background + Poisson noise + readout noise
        frame = rng.poisson(sky_level, (image_size, image_size)).astype(float)
        frame += rng.normal(0, read_noise, (image_size, image_size))

        # Create coordinate grids for placing Gaussian sources
        y_grid, x_grid = np.mgrid[0:image_size, 0:image_size]

        # Add stationary stars
        for (sx, sy), sflux in zip(star_positions, star_fluxes):
            # Only render stars near their position (speed optimisation)
            x_lo = max(0, int(sx) - 15)
            x_hi = min(image_size, int(sx) + 15)
            y_lo = max(0, int(sy) - 15)
            y_hi = min(image_size, int(sy) + 15)
            yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
            star_profile = gaussian_2d((yy, xx), sflux, sx, sy, sigma, sigma)
            frame[y_lo:y_hi, x_lo:x_hi] += star_profile

        # Add moving asteroids (position changes each frame)
        for a_idx, ap in enumerate(asteroid_params):
            ax = ap['start_x'] + ap['vx'] * frame_idx
            ay = ap['start_y'] + ap['vy'] * frame_idx
            # Small random magnitude variation (realistic: ~0.1 mag)
            flux_variation = rng.normal(1.0, 0.03)
            aflux = ap['flux'] * flux_variation

            x_lo = max(0, int(ax) - 15)
            x_hi = min(image_size, int(ax) + 15)
            y_lo = max(0, int(ay) - 15)
            y_hi = min(image_size, int(ay) + 15)
            if x_lo < x_hi and y_lo < y_hi:
                yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
                ast_profile = gaussian_2d(
                    (yy, xx), aflux, ax, ay, sigma, sigma)
                frame[y_lo:y_hi, x_lo:x_hi] += ast_profile

            truth['asteroids'][a_idx]['positions'].append({
                'frame': frame_idx, 'x': ax, 'y': ay
            })

        # Add hot pixels (same position every frame, sharp spike)
        for (hx, hy) in hot_pixel_positions:
            frame[hy, hx] += rng.uniform(300, 800)

        # Add cosmic rays (random position, only in ONE frame, sharp spike)
        if frame_idx == 1:  # Only in frame 2 for realism
            for _ in range(num_cosmic_rays):
                cx = rng.randint(10, image_size - 10)
                cy = rng.randint(10, image_size - 10)
                # Cosmic rays are very sharp — just 1-2 pixels
                frame[cy, cx] += rng.uniform(500, 2000)
                if frame_idx == 1:
                    truth['cosmic_rays'].append({
                        'x': cx, 'y': cy, 'frame': frame_idx
                    })

        frames.append(frame)

    return frames, truth


# =============================================================================
# SOURCE DETECTION ENGINE
# =============================================================================
# This is the core image processing: finding every bright spot in an image

def estimate_background(image, box_size=64):
    """
    Estimate the sky background level and noise across the image.

    Real telescope images are not uniformly bright — the sky background
    varies across the field. We divide the image into small boxes, measure
    the median (middle value) in each box, and create a smooth background
    map. This is similar to what Astrometrica's 'background subtraction'
    does.

    The noise (sigma) tells us how much the background fluctuates randomly.
    We need this to calculate signal-to-noise ratios.

    Parameters:
        image: 2D numpy array of pixel values
        box_size: size of the boxes for local background estimation

    Returns:
        background: 2D array of estimated background level at each pixel
        noise: 2D array of estimated noise (standard deviation) at each pixel
    """
    h, w = image.shape
    # Compute number of boxes in each direction
    ny = max(1, h // box_size)
    nx = max(1, w // box_size)

    bg_grid = np.zeros((ny, nx))
    noise_grid = np.zeros((ny, nx))

    for iy in range(ny):
        for ix in range(nx):
            y0 = iy * box_size
            y1 = min(y0 + box_size, h)
            x0 = ix * box_size
            x1 = min(x0 + box_size, w)
            region = image[y0:y1, x0:x1]
            # Median is robust to bright stars in the box
            bg_grid[iy, ix] = np.median(region)
            # MAD (Median Absolute Deviation) is a robust noise estimate
            # The 1.4826 factor converts MAD to equivalent standard deviation
            mad = np.median(np.abs(region - bg_grid[iy, ix]))
            noise_grid[iy, ix] = max(mad * 1.4826, 1.0)

    # Upscale the grid to full image size using smooth interpolation
    from scipy.ndimage import zoom
    zoom_y = h / ny
    zoom_x = w / nx
    background = zoom(bg_grid, (zoom_y, zoom_x), order=1)
    noise = zoom(noise_grid, (zoom_y, zoom_x), order=1)

    # Ensure same shape as input (zoom can be off by 1 pixel)
    background = background[:h, :w]
    noise = noise[:h, :w]

    return background, noise


def detect_sources(image, frame_index, detection_sigma=3.0):
    """
    Find all bright point sources in a single image frame.

    This is the fundamental operation: we subtract the background, find
    pixels that are significantly brighter than the noise, group connected
    bright pixels into individual sources, and measure each one.

    This is equivalent to clicking 'Detect Sources' in Astrometrica.

    Parameters:
        image: 2D numpy array of pixel values
        frame_index: which frame number (0-3) this is
        detection_sigma: how many standard deviations above background
                        a pixel must be to count as 'bright'

    Returns:
        sources: list of Source objects, one per detected bright spot
    """
    # Step 1: Estimate and subtract the background
    background, noise = estimate_background(image)
    subtracted = image - background

    # Step 2: Find pixels that are significantly above the noise
    # A pixel is 'detected' if it is more than detection_sigma times
    # the local noise level above zero (after background subtraction)
    threshold_map = detection_sigma * noise
    detected_mask = subtracted > threshold_map

    # Step 3: Label connected groups of bright pixels
    # Each group gets a unique number (1, 2, 3, ...)
    labelled, num_features = ndimage.label(detected_mask)

    sources = []
    sigma_psf = TYPICAL_FWHM_PIX / 2.355  # Expected PSF width

    for label_id in range(1, num_features + 1):
        # Get the pixels belonging to this source
        source_mask = labelled == label_id
        npix = np.sum(source_mask)

        # Skip very tiny or very large detections
        if npix < 3 or npix > 200:
            continue

        # Find the peak pixel
        source_pixels = subtracted * source_mask
        peak_val = np.max(source_pixels)
        peak_pos = np.unravel_index(np.argmax(source_pixels), image.shape)
        peak_y, peak_x = peak_pos

        # Skip sources too close to the image edge (can not measure properly)
        border = 10
        if (peak_x < border or peak_x >= image.shape[1] - border or
                peak_y < border or peak_y >= image.shape[0] - border):
            continue

        # Measure total flux (sum of all pixel values in the source)
        total_flux = np.sum(source_pixels)

        # Calculate SNR: signal divided by noise
        local_noise = noise[peak_y, peak_x]
        snr = peak_val / local_noise if local_noise > 0 else 0

        # Measure centroid (brightness-weighted centre position)
        ys, xs = np.where(source_mask)
        weights = subtracted[ys, xs]
        weights = np.maximum(weights, 0)
        total_weight = np.sum(weights)
        if total_weight > 0:
            cx = np.sum(xs * weights) / total_weight
            cy = np.sum(ys * weights) / total_weight
        else:
            cx, cy = float(peak_x), float(peak_y)

        # Measure FWHM and PSF fit quality
        fwhm_arcsec, fit_rms = measure_psf(subtracted, cx, cy, sigma_psf)

        # Convert flux to magnitude (astronomical brightness scale)
        # Magnitude = -2.5 * log10(flux) + zero_point
        # Zero point is arbitrary for relative comparison
        if total_flux > 0:
            magnitude = -2.5 * np.log10(total_flux) + 25.0
        else:
            magnitude = 99.0  # Unmeasurably faint

        sources.append(Source(
            x=cx, y=cy,
            flux=total_flux, peak=peak_val,
            snr=snr, fwhm=fwhm_arcsec,
            fit_rms=fit_rms, frame_index=frame_index,
            magnitude=magnitude
        ))

    return sources


def measure_psf(image, cx, cy, expected_sigma, box_radius=10):
    """
    Measure the Point Spread Function (PSF) of a source.

    The PSF is the shape of a point source (star or asteroid) in the image.
    Perfect point sources look like Gaussian bell curves when the atmosphere
    blurs them. Cosmic rays are too sharp. Extended objects (galaxies)
    are too wide.

    We fit a 2D Gaussian to the source and measure:
    1. FWHM: Full Width at Half Maximum (how wide the source is)
    2. Fit RMS: How well the Gaussian matches (low = good = real star)

    Parameters:
        image: background-subtracted image
        cx, cy: centre position of the source
        expected_sigma: expected Gaussian width from seeing conditions
        box_radius: how many pixels around the centre to use for fitting

    Returns:
        fwhm_arcsec: measured width in arcseconds
        fit_rms: root-mean-square error of the Gaussian fit (normalised)
    """
    h, w = image.shape
    # Extract a small cutout around the source
    x0 = max(0, int(round(cx)) - box_radius)
    x1 = min(w, int(round(cx)) + box_radius + 1)
    y0 = max(0, int(round(cy)) - box_radius)
    y1 = min(h, int(round(cy)) + box_radius + 1)

    cutout = image[y0:y1, x0:x1].copy()
    if cutout.size < 9 or np.max(cutout) <= 0:
        return TYPICAL_FWHM, 0.5  # Default if cutout is too small

    # Create coordinate grids in CUTOUT coordinates
    yy, xx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]

    # Source centre in cutout coordinates
    local_cx = cx - x0
    local_cy = cy - y0

    # Flatten for curve_fit
    coords = (yy.ravel(), xx.ravel())
    data = cutout.ravel()

    try:
        amp_guess = float(np.max(cutout))

        # Initial guess: amplitude, x-centre, y-centre, sigma_x, sigma_y
        p0 = [amp_guess, local_cx, local_cy, expected_sigma, expected_sigma]

        # Set reasonable bounds
        bounds_lo = [amp_guess * 0.3,
                     max(0, local_cx - 4), max(0, local_cy - 4),
                     0.5, 0.5]
        bounds_hi = [amp_guess * 2.0,
                     min(cutout.shape[1], local_cx + 4),
                     min(cutout.shape[0], local_cy + 4),
                     12.0, 12.0]

        popt, _ = curve_fit(
            gaussian_2d, (yy, xx), data,
            p0=p0, bounds=(bounds_lo, bounds_hi),
            maxfev=1000
        )

        # Calculate normalised RMS residual
        fitted = gaussian_2d((yy, xx), *popt)
        residual = data - fitted
        # Normalise by the peak amplitude so RMS is scale-independent
        rms = np.sqrt(np.mean(residual**2)) / max(popt[0], 1.0)

        # FWHM from the fitted sigma (average of x and y)
        avg_sigma = (abs(popt[3]) + abs(popt[4])) / 2.0
        fwhm_pixels = avg_sigma * 2.355
        fwhm_arcsec = fwhm_pixels * PIXEL_SCALE

        return fwhm_arcsec, rms

    except (RuntimeError, ValueError, TypeError):
        # If Gaussian fitting fails, fall back to a simpler measurement:
        # compute FWHM from the second moment of the brightness distribution
        try:
            peak = np.max(cutout)
            half_max = peak / 2.0
            above_half = cutout > half_max
            n_above = np.sum(above_half)
            # Effective radius of the half-max region
            # (assuming circular: area = pi * r^2, so r = sqrt(N/pi))
            if n_above > 0:
                eff_radius = np.sqrt(n_above / np.pi)
                fwhm_pix = eff_radius * 2.0
                fwhm_arcsec = fwhm_pix * PIXEL_SCALE
                # Estimate RMS from how circular the half-max region is
                if above_half.any():
                    ys_hm, xs_hm = np.where(above_half)
                    std_x = np.std(xs_hm) if len(xs_hm) > 1 else 1.0
                    std_y = np.std(ys_hm) if len(ys_hm) > 1 else 1.0
                    # Asymmetry ratio: 1.0 = perfectly round
                    ratio = min(std_x, std_y) / max(std_x, std_y, 0.1)
                    rms = 1.0 - ratio  # 0 = perfect, 1 = very asymmetric
                    return fwhm_arcsec, rms
        except Exception:
            pass
        return TYPICAL_FWHM, 0.15



# =============================================================================
# COMMAND-LINE INTERFACE (placeholder — linking engine coming next)
# =============================================================================

def main():
    """Main entry point — source detection is now implemented."""
    print("Automated Asteroid Detection Algorithm v0.2")
    print("Author: Siddharth Patel (AstroSidSpace)")
    print("Source detection engine added. Motion linking coming next.")
    return 0


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.exit(main())
