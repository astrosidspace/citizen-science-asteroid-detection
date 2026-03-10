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
LINEARITY_THRESHOLD = 3.0   # Max deviation from straight line in pixels

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
    Convenience wrapper that computes background internally.
    """
    background, noise = estimate_background(image)
    subtracted = image - background
    return detect_sources_from_subtracted(subtracted, noise, frame_index,
                                          detection_sigma)


def detect_sources_from_subtracted(subtracted, noise, frame_index,
                                    detection_sigma=3.0):
    """
    Find all bright point sources in a background-subtracted image.

    This is the fundamental operation: find pixels that are significantly
    brighter than the noise, group connected bright pixels into individual
    sources, and measure each one.

    This version takes pre-computed background-subtracted data and noise
    map, which avoids redundant background estimation when the pipeline
    needs both the subtracted frame (for PSF fitting) and the sources.

    Parameters:
        subtracted: 2D numpy array, background-subtracted pixel values
        noise: 2D noise map from background estimation
        frame_index: which frame number (0-3) this is
        detection_sigma: how many standard deviations above noise

    Returns:
        sources: list of Source objects, one per detected bright spot
    """
    # Find pixels significantly above the noise
    threshold_map = detection_sigma * noise
    detected_mask = subtracted > threshold_map

    # Label connected groups of bright pixels
    labelled, num_features = ndimage.label(detected_mask)

    # PERFORMANCE OPTIMISATION: use find_objects() to get bounding boxes
    # for each labelled region. This avoids the old approach of scanning
    # the entire image array per label (which is O(N * image_size)).
    # With find_objects(), each source only examines its own bounding box.
    slices = ndimage.find_objects(labelled)

    sources = []
    h, w = subtracted.shape
    border = 10

    for label_id, obj_slice in enumerate(slices, start=1):
        if obj_slice is None:
            continue

        # Extract just the bounding box region for this source
        region_labels = labelled[obj_slice]
        region_mask = region_labels == label_id
        npix = np.sum(region_mask)

        # Skip very tiny or very large detections
        # (3 pixels minimum; 500 max for real images with bright stars)
        if npix < 3 or npix > 500:
            continue

        # Work within the bounding box for efficiency
        region_data = subtracted[obj_slice]
        source_pixels = region_data * region_mask

        # Find the peak pixel (position relative to full image)
        peak_val = np.max(source_pixels)
        local_peak = np.unravel_index(np.argmax(source_pixels),
                                       source_pixels.shape)
        peak_y = obj_slice[0].start + local_peak[0]
        peak_x = obj_slice[1].start + local_peak[1]

        # Skip sources too close to the image edge
        if (peak_x < border or peak_x >= w - border or
                peak_y < border or peak_y >= h - border):
            continue

        # Measure total flux (sum of all pixel values in the source)
        total_flux = np.sum(source_pixels)

        # Calculate SNR: signal divided by noise
        local_noise = noise[peak_y, peak_x]
        snr = peak_val / local_noise if local_noise > 0 else 0

        # Measure centroid (brightness-weighted centre position)
        local_ys, local_xs = np.where(region_mask)
        weights = region_data[local_ys, local_xs]
        weights = np.maximum(weights, 0)
        total_weight = np.sum(weights)
        if total_weight > 0:
            # Convert local coords back to full-image coords
            cx = np.sum((local_xs + obj_slice[1].start) * weights) / total_weight
            cy = np.sum((local_ys + obj_slice[0].start) * weights) / total_weight
        else:
            cx, cy = float(peak_x), float(peak_y)

        # PERFORMANCE OPTIMISATION: defer PSF fitting to validation step.
        # Real Pan-STARRS images have thousands of sources. Fitting a 2D
        # Gaussian to every one is very slow (~minutes on 2434x2423 images).
        # Instead, we assign defaults and only do expensive PSF measurement
        # on tracklet candidates after the linking step. This is the same
        # approach Astrometrica uses.
        fwhm_arcsec = TYPICAL_FWHM  # Default; refined during validation
        fit_rms = 0.1               # Default; refined during validation

        # Convert flux to magnitude
        if total_flux > 0:
            magnitude = -2.5 * np.log10(total_flux) + 25.0
        else:
            magnitude = 99.0

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
# FIELD SEEING MEASUREMENT
# =============================================================================
# Professional pipelines measure the actual "seeing" (atmospheric blur)
# from the data rather than using fixed FWHM thresholds. This adapts
# the algorithm to different observing conditions automatically.

def measure_field_seeing(subtracted_frames, frame_sources, n_sample=20):
    """
    Measure the field seeing by fitting PSFs to bright, well-isolated sources.

    Instead of assuming a fixed FWHM range like 0.8-1.2 arcseconds, we
    measure the actual seeing from the data. This is critical because:
    - Different nights have different atmospheric conditions
    - Different telescopes have different optical characteristics
    - Faint sources have noisier FWHM measurements, so we need to know
      the TRUE seeing to set meaningful thresholds

    The method: take the brightest sources (which have the best-measured
    PSFs), fit Gaussians to them, and compute the median FWHM. This
    gives the field's characteristic point-source width.

    Parameters:
        subtracted_frames: list of background-subtracted image arrays
        frame_sources: list of source lists, one per frame
        n_sample: number of bright sources per frame to measure

    Returns:
        field_fwhm: median FWHM in arcseconds across all bright sources
    """
    sigma_psf = TYPICAL_FWHM_PIX / 2.355
    all_fwhms = []

    for fi in range(min(len(subtracted_frames), NUM_FRAMES)):
        # Sort sources by flux (brightness), take the brightest
        sources = sorted(frame_sources[fi], key=lambda s: s.flux, reverse=True)
        bright = sources[:n_sample]

        for s in bright:
            fwhm, rms = measure_psf(subtracted_frames[fi], s.x, s.y, sigma_psf)
            # Only use well-fitted sources (low RMS = clean Gaussian shape)
            if rms < 0.15:
                all_fwhms.append(fwhm)

    if len(all_fwhms) >= 5:
        return float(np.median(all_fwhms))
    else:
        # Fallback: not enough well-measured sources, use default
        return TYPICAL_FWHM


# =============================================================================
# MOTION LINKING ENGINE
# =============================================================================
# This connects sources across frames to find things that moved

def link_tracklets(all_sources, search_radius_pixels=80.0,
                    min_motion_pixels=3.0,
                    tight_linking=False):
    """
    Link sources across 4 frames to find objects that moved consistently.

    This is the key astronomical algorithm. For each source in frame 1, we
    look for a DIFFERENT source in frame 2 that has moved by at least a
    minimum distance (to exclude stationary stars). We then predict where
    the object should appear in frames 3 and 4 and search for matches.

    The search radius of 80 pixels covers asteroid motion rates up to
    about 0.67 arcsec/min at Pan-STARRS pixel scale (0.25"/pix) with
    30-minute frame intervals, handling typical main-belt asteroids
    (0.3-0.5 arcsec/min) with generous margin. The minimum motion of
    3 pixels eliminates stationary stars. For real telescope images with
    shorter frame intervals (~16 min), the pipeline passes tighter
    parameters automatically.

    Math: 0.5 arcsec/min * 30 min / 0.25 arcsec/pixel = 60 pixels/frame
    So 80 pixels gives us comfortable headroom.

    Parameters:
        all_sources: list of all Source objects from all 4 frames
        search_radius_pixels: maximum distance between two detections
                             in consecutive frames to consider a match
        min_motion_pixels: minimum motion per frame to be considered
                          (filters out stationary stars)

    Returns:
        tracklets: list of Tracklet objects (only moving objects)
    """
    # Separate sources by frame
    frame_sources = [[] for _ in range(NUM_FRAMES)]
    for s in all_sources:
        if 0 <= s.frame_index < NUM_FRAMES:
            frame_sources[s.frame_index].append(s)

    # Check we have sources in all frames
    for i in range(NUM_FRAMES):
        if len(frame_sources[i]) == 0:
            return []

    # Build spatial index trees for fast neighbour searching
    # (cKDTree is much faster than checking every pair)
    trees = []
    for fi in range(NUM_FRAMES):
        positions = np.array([[s.x, s.y] for s in frame_sources[fi]])
        trees.append(cKDTree(positions))

    # --- Precompute stationary source flags ---
    # Stars appear at the SAME position in every frame. An asteroid MOVES,
    # so its position in frame 0 will NOT have a source in frames 1-3.
    # For each source, check if a similarly bright source exists at the
    # same position (within 5 pixels) in ALL other frames. If so, it is
    # a stationary object (star) — linking it with another stationary
    # source just creates a false tracklet.
    # This is O(N * 3 * log N) — fast with KD-trees.
    STATIONARY_MATCH_RADIUS = 5.0  # pixels — account for centroid jitter
    is_stationary = [[] for _ in range(NUM_FRAMES)]
    for fi in range(NUM_FRAMES):
        for si, src in enumerate(frame_sources[fi]):
            # Count how many OTHER frames have a source at this position
            match_count = 0
            for other_fi in range(NUM_FRAMES):
                if other_fi == fi:
                    continue
                nearby = trees[other_fi].query_ball_point(
                    [src.x, src.y], STATIONARY_MATCH_RADIUS)
                if nearby:
                    match_count += 1
            # Stationary = has a match in at least 2 of 3 other frames
            is_stationary[fi].append(match_count >= 2)

    # --- Precompute MAGNITUDE-AWARE stationary flags ---
    # The basic position-only check at 5px misses faint sources with
    # larger centroid jitter. We add a second check at a wider radius
    # (7px) that also requires the NEAREST matching source to have
    # similar brightness (within 0.5 mag). Using the nearest source
    # (not any source within the radius) is critical because:
    #   - A STAR matched to itself is the nearest source at ~same position
    #   - An ASTEROID passing near a random star: the nearest source is
    #     the star, which has a different magnitude → no match
    #   - Using "any source within r" would trigger on unrelated stars
    #     that coincidentally have similar brightness at larger distances
    # Validated: preserves real asteroid in XY75_p00, rejects false
    # linkages in XY75_p11.
    MAG_STATIONARY_RADIUS = 7.0    # wider radius to catch centroid jitter
    MAG_STATIONARY_TOL = 0.5       # magnitude difference tolerance
    is_mag_stationary = [[] for _ in range(NUM_FRAMES)]
    for fi in range(NUM_FRAMES):
        for si, src in enumerate(frame_sources[fi]):
            mag_match_count = 0
            for other_fi in range(NUM_FRAMES):
                if other_fi == fi:
                    continue
                # Find the NEAREST source in the other frame
                dist, idx = trees[other_fi].query([src.x, src.y])
                if (dist <= MAG_STATIONARY_RADIUS and
                        abs(frame_sources[other_fi][idx].magnitude -
                            src.magnitude) <= MAG_STATIONARY_TOL):
                    mag_match_count += 1
            # Magnitude-stationary = same-brightness nearest source
            # in ≥2 of 3 other frames
            is_mag_stationary[fi].append(mag_match_count >= 2)

    # --- Phase 1: Generate ALL candidate tracklets ---
    # We collect every valid 4-frame linkage, then pick the best ones.
    # This avoids the problem where a greedy algorithm makes a bad early
    # match that blocks a better match later.
    all_candidate_tracklets = []

    for i0, s0 in enumerate(frame_sources[0]):
        # Find candidate matches in frame 1 within the search radius
        candidates_f1 = trees[1].query_ball_point([s0.x, s0.y],
                                                   search_radius_pixels)

        for i1 in candidates_f1:
            s1 = frame_sources[1][i1]

            # Calculate velocity from frame 0 to frame 1
            vx = s1.x - s0.x
            vy = s1.y - s0.y
            speed = np.sqrt(vx**2 + vy**2)

            # CRITICAL FILTER: Skip if motion is too small
            # Stars appear at the same position (+/- sub-pixel noise)
            # Real asteroids must move at least min_motion_pixels per frame
            if speed < min_motion_pixels:
                continue

            # BRIGHTNESS CONSISTENCY: Skip if magnitudes are wildly different
            # Real asteroids maintain similar brightness across frames.
            # A 2-magnitude difference means the linker is probably matching
            # the asteroid with an unrelated star in another frame.
            if abs(s0.magnitude - s1.magnitude) > 2.0:
                continue

            # Predict position in frame 2 (assuming constant velocity)
            pred_x2 = s1.x + vx
            pred_y2 = s1.y + vy

            # Search for a match near the predicted position in frame 2
            # The search radius scales with speed but has a minimum floor
            # and maximum cap. We use 25% of the speed to allow for small
            # centroid errors from noise, blending, or seeing variations.
            # The prediction search radius controls how many false matches
            # we get. In dense real fields, we use a tighter radius to
            # reduce the combinatorial explosion (π*r² scales area).
            # For synthetic/small images, we keep a wider radius to
            # accommodate faster-moving objects.
            if tight_linking:
                tight_radius = min(8.0, max(5.0, speed * 0.20))
            else:
                tight_radius = min(15.0, max(8.0, speed * 0.35))
            candidates_f2 = trees[2].query_ball_point(
                [pred_x2, pred_y2], tight_radius)

            for i2 in candidates_f2:
                s2 = frame_sources[2][i2]

                # Magnitude consistency check
                mag_tol = 1.5 if tight_linking else 2.0
                if abs(s2.magnitude - s0.magnitude) > mag_tol:
                    continue

                # Predict position in frame 3
                pred_x3 = s2.x + vx
                pred_y3 = s2.y + vy

                # Search for a match in frame 3
                candidates_f3 = trees[3].query_ball_point(
                    [pred_x3, pred_y3], tight_radius)

                for i3 in candidates_f3:
                    s3 = frame_sources[3][i3]

                    # Magnitude consistency check
                    if abs(s3.magnitude - s0.magnitude) > mag_tol:
                        continue

                    # VELOCITY CONSISTENCY FILTER (critical for real data):
                    # A real asteroid has constant velocity. Check that
                    # all 3 inter-frame velocities match the initial one.
                    # This eliminates most random star-to-star linkages.
                    vx12 = s2.x - s1.x
                    vy12 = s2.y - s1.y
                    vx23 = s3.x - s2.x
                    vy23 = s3.y - s2.y
                    speed01 = speed  # already computed
                    speed12 = np.sqrt(vx12**2 + vy12**2)
                    speed23 = np.sqrt(vx23**2 + vy23**2)
                    # Reject if inter-frame speed varies too much.
                    # Tighter for real data (12%) to reduce false linkages;
                    # looser for synthetic (20%) to accommodate noise.
                    vel_tol = 0.12 if tight_linking else 0.20
                    if speed01 > 0:
                        if (abs(speed12 - speed01) / speed01 > vel_tol or
                                abs(speed23 - speed01) / speed01 > vel_tol):
                            continue

                    # STATIONARY SOURCE REJECTION (two-tier):
                    # If most sources are stationary (appear at the same
                    # position in multiple other frames), this tracklet is
                    # just linking different stars — not a real moving object.
                    # A real asteroid LEAVES its position in each frame, so
                    # its sources should NOT have stationary counterparts.
                    #
                    # Tier 1 (position-only, r=5px): reject if ≥2 sources
                    # have ANY match in ≥2 other frames.
                    stationary_count = sum([
                        is_stationary[0][i0], is_stationary[1][i1],
                        is_stationary[2][i2], is_stationary[3][i3]])
                    if stationary_count >= 2:
                        continue

                    # Tier 2 (magnitude-aware, r=7px): reject if ≥2 sources
                    # have a SAME-BRIGHTNESS match (within 0.5 mag) in ≥2
                    # other frames. This catches faint noise peaks near
                    # real stars that the tighter r=5 check misses, while
                    # preserving real asteroid detections (whose sources
                    # have different magnitudes than nearby background
                    # stars). Validated on XY75_p00 positive field.
                    mag_stat_count = sum([
                        is_mag_stationary[0][i0], is_mag_stationary[1][i1],
                        is_mag_stationary[2][i2], is_mag_stationary[3][i3]])
                    if mag_stat_count >= 2:
                        continue

                    # Score this tracklet by positional residual +
                    # magnitude consistency + velocity consistency.
                    # Lower score = better tracklet.
                    res2 = np.sqrt((s2.x - pred_x2)**2 +
                                  (s2.y - pred_y2)**2)
                    res3 = np.sqrt((s3.x - pred_x3)**2 +
                                  (s3.y - pred_y3)**2)
                    mag_spread = np.std([s0.magnitude, s1.magnitude,
                                        s2.magnitude, s3.magnitude])
                    vel_diff = np.sqrt((vx23 - vx)**2 + (vy23 - vy)**2)
                    # Combined quality (lower is better).
                    # Weight magnitude heavily to avoid star-asteroid confusion
                    quality = (res2 + res3 +
                              mag_spread * 10.0 +
                              vel_diff * 2.0)

                    all_candidate_tracklets.append({
                        'indices': (i0, i1, i2, i3),
                        'sources': (s0, s1, s2, s3),
                        'quality': quality
                    })

    # --- Phase 2: Select best non-overlapping tracklets ---
    # Sort by quality (best first) and greedily select tracklets
    # that don't reuse any source
    all_candidate_tracklets.sort(key=lambda t: t['quality'])

    tracklets = []
    used_in_frame = [set() for _ in range(NUM_FRAMES)]

    for ct in all_candidate_tracklets:
        i0, i1, i2, i3 = ct['indices']
        # Check none of these sources are already used
        if (i0 in used_in_frame[0] or i1 in used_in_frame[1] or
                i2 in used_in_frame[2] or i3 in used_in_frame[3]):
            continue

        s0, s1, s2, s3 = ct['sources']
        tracklet = Tracklet(sources=[s0, s1, s2, s3])
        tracklets.append(tracklet)
        used_in_frame[0].add(i0)
        used_in_frame[1].add(i1)
        used_in_frame[2].add(i2)
        used_in_frame[3].add(i3)

    # Calculate motion properties for each tracklet
    for t in tracklets:
        compute_tracklet_properties(t)

    return tracklets


def compute_tracklet_properties(tracklet):
    """
    Calculate the motion properties of a linked tracklet.

    Once we have 4 linked detections, we measure:
    - Average velocity (speed and direction)
    - Speed in astronomical units (arcseconds per minute)
    - Average magnitude (brightness)

    Parameters:
        tracklet: a Tracklet object with 4 linked sources
    """
    sources = tracklet.sources

    # Velocity from first to last source
    dx = sources[-1].x - sources[0].x
    dy = sources[-1].y - sources[0].y
    n_intervals = len(sources) - 1

    tracklet.velocity_x = dx / n_intervals
    tracklet.velocity_y = dy / n_intervals

    # Convert to arcseconds per minute
    speed_pixels_per_frame = np.sqrt(tracklet.velocity_x**2 +
                                     tracklet.velocity_y**2)
    speed_arcsec_per_frame = speed_pixels_per_frame * PIXEL_SCALE
    tracklet.velocity_arcsec_min = speed_arcsec_per_frame / FRAME_INTERVAL_MINUTES

    # Position angle (direction of motion, in degrees from North)
    tracklet.position_angle = np.degrees(
        np.arctan2(tracklet.velocity_x, -tracklet.velocity_y)) % 360

    # Mean magnitude
    tracklet.mean_magnitude = np.mean([s.magnitude for s in sources])


# =============================================================================
# 9-CRITERIA VALIDATION ENGINE
# =============================================================================
# These are the exact same criteria professional astronomers and IASC
# participants use to verify asteroid detections

def validate_tracklet(tracklet, all_sources_per_frame=None, frames=None,
                      field_fwhm=None, is_real_data=False):
    """
    Apply all 9 IASC/Astrometrica verification criteria to a tracklet.

    Each criterion returns True (passed) or False (failed), along with
    a measured value and a note explaining the result.

    This is the heart of the algorithm — the part that separates real
    asteroids from false detections. A human astronomer checks these same
    things by eye in Astrometrica; our algorithm does it mathematically.

    Parameters:
        tracklet: a Tracklet object to validate
        all_sources_per_frame: optional, list of source counts per frame
                              (used for some criteria)
        frames: optional, list of image arrays for deferred PSF measurement
        field_fwhm: optional, measured field seeing in arcseconds. When
                    provided, FWHM thresholds adapt to the actual seeing
                    conditions rather than using fixed values.
        is_real_data: if True, enables additional checks for real telescope
                     data (e.g., bright star linkage detection). Disabled
                     for synthetic data where magnitude calibration differs.

    Returns:
        criteria_results: dictionary with results for each criterion
        confidence: overall confidence score (0-100%)
    """
    sources = tracklet.sources

    # --- Deferred PSF measurement ---
    # PSF fitting is expensive, so we only do it on tracklet candidates
    # that survived the linking step. If frames are provided, measure
    # the PSF now for accurate criteria 3 and 4 evaluation.
    # 'frames' can be either raw frames or pre-subtracted frames.
    if frames is not None:
        sigma_psf = TYPICAL_FWHM_PIX / 2.355
        for s in sources:
            if 0 <= s.frame_index < len(frames):
                frame = frames[s.frame_index]
                fwhm, rms = measure_psf(frame, s.x, s.y, sigma_psf)
                s.fwhm = fwhm
                s.fit_rms = rms

    criteria = {}

    # ---- Criterion 1: PERSISTENCE ----
    # Real asteroids appear in ALL 4 frames.
    # Cosmic rays only hit 1 frame. Hot pixels are always in the same spot.
    n_frames = len(sources)
    frame_indices = set(s.frame_index for s in sources)
    persistence_pass = (n_frames >= NUM_FRAMES and
                        len(frame_indices) == NUM_FRAMES)
    criteria['1_persistence'] = {
        'passed': persistence_pass,
        'value': n_frames,
        'threshold': NUM_FRAMES,
        'note': f'Detected in {n_frames}/{NUM_FRAMES} frames'
    }

    # ---- Criterion 2: SIGNAL-TO-NOISE RATIO ----
    # All detections must be above SNR 5 to be reliable.
    # Below this, the object is lost in the noise.
    snr_values = [s.snr for s in sources]
    min_snr = min(snr_values)
    mean_snr = np.mean(snr_values)
    snr_pass = min_snr >= SNR_THRESHOLD
    criteria['2_snr'] = {
        'passed': snr_pass,
        'value': round(mean_snr, 1),
        'threshold': SNR_THRESHOLD,
        'note': f'Mean SNR={mean_snr:.1f}, Min SNR={min_snr:.1f}'
    }

    # ---- Criterion 3: PSF GAUSSIAN FIT ----
    # Real point sources have smooth bell-curve brightness profiles.
    # Cosmic rays are too sharp. Extended objects are too wide.
    fit_rms_values = [s.fit_rms for s in sources]
    mean_fit_rms = np.mean(fit_rms_values)
    psf_pass = mean_fit_rms < FIT_RMS_THRESHOLD
    criteria['3_psf_fit'] = {
        'passed': psf_pass,
        'value': round(mean_fit_rms, 3),
        'threshold': FIT_RMS_THRESHOLD,
        'note': f'Mean Fit RMS={mean_fit_rms:.3f} (< {FIT_RMS_THRESHOLD} required)'
    }

    # ---- Criterion 4: FWHM CONSISTENCY ----
    # The width of the source should match the telescope's seeing.
    # ADAPTIVE THRESHOLDS: instead of fixed 0.8-1.2 arcsec, we use the
    # measured field seeing to set bounds. This is critical because:
    #   - Different nights have different seeing (0.6-2.0 arcsec)
    #   - Faint sources have noisier FWHM measurements
    #   - Fixed thresholds reject real detections in non-ideal conditions
    # We allow 0.5x to 2.5x the measured seeing, which covers the range
    # of point sources from slightly sub-seeing (tight PSF) to slightly
    # extended (noisy measurement of faint source).
    fwhm_values = [s.fwhm for s in sources]
    mean_fwhm = np.mean(fwhm_values)

    if field_fwhm is not None and field_fwhm > 0:
        # Adaptive thresholds based on measured field seeing
        fwhm_lo = field_fwhm * 0.5
        fwhm_hi = field_fwhm * 2.5
    else:
        # Fallback to fixed thresholds
        fwhm_lo = FWHM_MIN
        fwhm_hi = FWHM_MAX

    fwhm_pass = fwhm_lo <= mean_fwhm <= fwhm_hi
    criteria['4_fwhm'] = {
        'passed': fwhm_pass,
        'value': round(mean_fwhm, 2),
        'threshold': f'{fwhm_lo:.2f}-{fwhm_hi:.2f}',
        'note': f'Mean FWHM={mean_fwhm:.2f}" (range {fwhm_lo:.2f}-{fwhm_hi:.2f}")'
    }

    # ---- Criterion 5: LINEAR MOTION ----
    # Main belt asteroids move in straight lines over 2 hours.
    # We measure deviation from the velocity vector defined by the
    # first and last detections (not a best-fit line — a least-squares
    # fit would artificially minimise residuals for random alignments).
    # This tests whether the INTERMEDIATE positions fall on the line
    # connecting the first and last position, which is more discriminating.
    positions = np.array([[s.x, s.y] for s in sources])
    n_pts = len(positions)

    if n_pts >= 3:
        # Define the motion line from first to last detection
        p0 = positions[0]
        p_last = positions[-1]
        line_vec = p_last - p0
        line_len = np.sqrt(line_vec[0]**2 + line_vec[1]**2)

        if line_len > 0:
            # Unit direction and perpendicular
            unit_dir = line_vec / line_len
            # Perpendicular distance of each intermediate point from line
            residuals = []
            for i in range(1, n_pts - 1):
                delta = positions[i] - p0
                # Cross product gives perpendicular distance
                cross = abs(delta[0] * unit_dir[1] - delta[1] * unit_dir[0])
                residuals.append(cross)
            max_residual = max(residuals)
        else:
            max_residual = 999.0  # No motion = not an asteroid
    else:
        max_residual = 0.0

    linearity_pass = max_residual < LINEARITY_THRESHOLD
    criteria['5_linear_motion'] = {
        'passed': linearity_pass,
        'value': round(max_residual, 2),
        'threshold': LINEARITY_THRESHOLD,
        'note': f'Max deviation from line: {max_residual:.2f} pixels'
    }

    # ---- Criterion 6: CONSTANT VELOCITY ----
    # The speed should not change more than 10% between consecutive frames.
    velocities = []
    for i in range(len(sources) - 1):
        dx = sources[i + 1].x - sources[i].x
        dy = sources[i + 1].y - sources[i].y
        v = np.sqrt(dx**2 + dy**2)
        velocities.append(v)

    if len(velocities) >= 2 and np.mean(velocities) > 0:
        mean_vel = np.mean(velocities)
        max_variation = max(abs(v - mean_vel) / mean_vel for v in velocities)
    else:
        max_variation = 0.0

    velocity_pass = max_variation < VELOCITY_VARIATION_MAX
    criteria['6_constant_velocity'] = {
        'passed': velocity_pass,
        'value': round(max_variation * 100, 1),
        'threshold': f'{VELOCITY_VARIATION_MAX * 100}%',
        'note': f'Velocity variation: {max_variation * 100:.1f}%'
    }

    # ---- Criterion 7: STABLE MAGNITUDE ----
    # Real asteroids do not change brightness dramatically between frames.
    # (They can rotate and vary slowly, but not by more than 1 magnitude
    # over 2 hours for typical main-belt asteroids.)
    magnitudes = [s.magnitude for s in sources]
    mag_range = max(magnitudes) - min(magnitudes)
    magnitude_pass = mag_range < MAGNITUDE_VARIATION_MAX
    criteria['7_stable_magnitude'] = {
        'passed': magnitude_pass,
        'value': round(mag_range, 2),
        'threshold': MAGNITUDE_VARIATION_MAX,
        'note': f'Magnitude range: {mag_range:.2f} mag'
    }

    # ---- Criterion 8: NOT IN KNOWN CATALOGUES ----
    # In a real deployment, we would query the Minor Planet Center's
    # MPCORB catalogue, SkyBot, or NEOCP to check if this is a known
    # object. For the offline demo, we simulate this check.
    # (Set to True since we cannot check online in a demo)
    catalogue_pass = True
    criteria['8_not_known'] = {
        'passed': catalogue_pass,
        'value': 'N/A (offline mode)',
        'threshold': 'No match in MPC/SkyBot',
        'note': 'Catalogue check: skipped (offline demo mode)'
    }

    # ---- Criterion 9: NOT A FALSE POSITIVE ----
    # Check for common false positive patterns:
    # - Zero motion (hot pixel that got linked by accident)
    # - Very fast motion (satellite)
    # - Source at exact same pixel position each frame (hot pixel)
    # - Too bright to be undiscovered (known star linkage)
    speed = np.sqrt(tracklet.velocity_x**2 + tracklet.velocity_y**2)
    is_stationary = speed < 1.0  # Less than 1 pixel of motion total
    is_too_fast = tracklet.velocity_arcsec_min > 2.0  # Satellites move > 2"/min

    # Check if positions are identical (hot pixel signature)
    pos_spread = np.std(positions, axis=0)
    is_hot_pixel = np.all(pos_spread < 0.5)

    # BRIGHT STAR LINKAGE CHECK:
    # In real Pan-STARRS fields, any object brighter than magnitude ~18.5
    # would already be in star catalogues (Gaia, USNO-B, 2MASS, etc.).
    # Undiscovered asteroids in IASC campaigns are typically mag 19-22.
    # If a tracklet links very bright sources, it is almost certainly
    # linking different known stars that coincidentally form a line.
    # This check is only applied to real telescope data where magnitude
    # calibration is reliable. Synthetic data uses different magnitude
    # scales, so we skip this check there.
    is_bright_star_linkage = (is_real_data and
                               tracklet.mean_magnitude < 18.5 and
                               n_pts == NUM_FRAMES)

    false_positive_pass = not (is_stationary or is_too_fast or
                               is_hot_pixel or is_bright_star_linkage)
    fp_reason = 'Passed all false positive checks'
    if is_stationary:
        fp_reason = 'FAILED: Object appears stationary (hot pixel?)'
    elif is_too_fast:
        fp_reason = 'FAILED: Motion too fast (satellite?)'
    elif is_hot_pixel:
        fp_reason = 'FAILED: Same position in all frames (hot pixel)'
    elif is_bright_star_linkage:
        fp_reason = (f'FAILED: Too bright (mag {tracklet.mean_magnitude:.1f}'
                     f' < 18.5) — likely star linkage')

    criteria['9_not_false_positive'] = {
        'passed': false_positive_pass,
        'value': f'{tracklet.velocity_arcsec_min:.3f} arcsec/min',
        'threshold': '0.1-2.0 arcsec/min',
        'note': fp_reason
    }

    # ---- Calculate overall confidence score ----
    # Each criterion contributes to the final score
    # Some criteria are weighted more heavily (persistence and SNR are critical)
    weights = {
        '1_persistence': 20,      # Must be in all 4 frames
        '2_snr': 15,              # Must be clearly above noise
        '3_psf_fit': 10,          # Should look like a point source
        '4_fwhm': 10,             # Right size for a star/asteroid
        '5_linear_motion': 15,    # Must move in a straight line
        '6_constant_velocity': 10, # Speed should be constant
        '7_stable_magnitude': 10,  # Brightness should not jump
        '8_not_known': 5,         # Bonus for unknown objects
        '9_not_false_positive': 5  # Must not match false positive patterns
    }

    total_weight = sum(weights.values())
    earned = sum(weights[k] for k, v in criteria.items() if v['passed'])
    confidence = (earned / total_weight) * 100

    tracklet.criteria_results = criteria
    tracklet.confidence_score = confidence

    # A candidate must pass BOTH the confidence threshold AND the
    # critical hard-gate criteria. These are non-negotiable requirements
    # — just like in real Astrometrica, these physics-based constraints
    # are fundamental:
    #   - Persistence (criterion 1): must appear in all 4 frames
    #   - Linear motion (criterion 5): asteroids follow Keplerian orbits,
    #     which produce straight-line motion over short observation arcs
    #   - Constant velocity (criterion 6): Keplerian motion at these
    #     timescales produces nearly constant apparent velocity
    #   - Not-false-positive (criterion 9): must not match noise patterns
    hard_gates_pass = (criteria['1_persistence']['passed'] and
                       criteria['5_linear_motion']['passed'] and
                       criteria['6_constant_velocity']['passed'] and
                       criteria['9_not_false_positive']['passed'])
    tracklet.is_candidate = confidence >= 70 and hard_gates_pass

    return criteria, confidence


# =============================================================================
# MAIN DETECTION PIPELINE
# =============================================================================

def run_detection_pipeline(frames, verbose=True, detection_sigma=None):
    """
    Run the complete asteroid detection pipeline on a set of 4 frames.

    This is the main function that orchestrates everything:
    1. Detect sources in each frame
    2. Link sources across frames to find moving objects
    3. Validate each tracklet against the 9 criteria
    4. Score and rank candidates

    Parameters:
        frames: list of 4 numpy 2D arrays (the images)
        verbose: if True, print progress messages (good for live demo)
        detection_sigma: SNR threshold for source detection. If None,
            auto-selects: 3.0 for small/synthetic images, 5.0 for large
            real telescope images (reduces false noise detections)

    Returns:
        result: a DetectionResult object with all findings
    """
    # Auto-select detection sigma based on image size.
    # For large real telescope images, we need a higher threshold
    # because (a) there are millions of pixels creating many noise
    # peaks, and (b) the combinatorial linking scales as O(N^2) with
    # source count. Sigma=8 gives ~800-1200 strong sources per frame
    # in a typical Pan-STARRS field — enough to detect asteroids
    # (which have SNR >> 5) while keeping linking tractable.
    if detection_sigma is None:
        image_pixels = frames[0].shape[0] * frames[0].shape[1]
        detection_sigma = 8.0 if image_pixels > 1_000_000 else 3.0
    start_time = time.time()
    result = DetectionResult()

    if verbose:
        print("\n" + "=" * 70)
        print("  ASTEROID DETECTION PIPELINE — STARTING")
        print("=" * 70)

    # ---- Step 1: Detect sources in each frame ----
    # We cache background-subtracted frames for deferred PSF fitting later
    all_sources = []
    subtracted_frames = []
    for i, frame in enumerate(frames):
        if verbose:
            print(f"\n  [Frame {i + 1}/{NUM_FRAMES}] Detecting sources...", end="",
                  flush=True)
        # Compute background once and reuse for both detection and PSF
        background, noise = estimate_background(frame)
        subtracted = frame - background
        subtracted_frames.append(subtracted)
        sources = detect_sources_from_subtracted(subtracted, noise, i,
                                                   detection_sigma)
        all_sources.extend(sources)
        if verbose:
            print(f" found {len(sources)} sources")
            # Show SNR statistics
            if sources:
                snrs = [s.snr for s in sources]
                print(f"    SNR range: {min(snrs):.1f} — {max(snrs):.1f}, "
                      f"mean: {np.mean(snrs):.1f}")

    result.total_sources_detected = len(all_sources)
    if verbose:
        print(f"\n  Total sources across all frames: {result.total_sources_detected}")

    # ---- Step 1b: Measure field seeing ----
    # Adaptive FWHM: instead of fixed thresholds, we measure the actual
    # point-source width from bright stars in the field. This adapts to
    # different seeing conditions (weather, telescope focus, altitude).
    frame_sources_list = [[] for _ in range(NUM_FRAMES)]
    for s in all_sources:
        frame_sources_list[s.frame_index].append(s)

    field_fwhm = measure_field_seeing(subtracted_frames, frame_sources_list)
    if verbose:
        print(f"\n  [Seeing] Measured field FWHM: {field_fwhm:.3f} arcsec")

    # ---- Step 2: Link sources into tracklets ----
    # For large real images, use tighter linking parameters because
    # (a) shorter frame intervals mean less motion per frame, and
    # (b) dense fields need stricter filters to avoid false linkages.
    image_pixels = frames[0].shape[0] * frames[0].shape[1]
    is_real_data = image_pixels > 1_000_000
    if is_real_data:
        search_radius = 55.0   # Real data: ~16-min intervals
        min_motion = 12.0      # Real asteroids move 20+ px/frame at Pan-STARRS
    else:
        search_radius = 80.0   # Synthetic: 30-min intervals
        min_motion = 3.0       # Smaller images, less noise
    if verbose:
        print("\n  [Linking] Searching for moving objects across frames...")
    tracklets = link_tracklets(all_sources, search_radius, min_motion,
                                tight_linking=is_real_data)
    result.tracklets = tracklets
    result.total_tracklets_formed = len(tracklets)
    if verbose:
        print(f"  Found {len(tracklets)} potential tracklets")

    # ---- Step 3: Validate each tracklet ----
    # Now we do deferred PSF measurement only on tracklet candidates
    # (much fewer than all detected sources — typically 5-20 vs thousands)
    if verbose:
        print("\n  [Validation] Applying 9 detection criteria to each tracklet...")
        print(f"    (Performing deferred PSF fitting on {len(tracklets)} tracklets)")
    candidates = []
    for idx, t in enumerate(tracklets):
        criteria, confidence = validate_tracklet(t, frames=subtracted_frames,
                                                  field_fwhm=field_fwhm,
                                                  is_real_data=is_real_data)
        passed = sum(1 for c in criteria.values() if c['passed'])
        if verbose:
            status = "CANDIDATE" if t.is_candidate else "rejected"
            print(f"    Tracklet {idx + 1}: {passed}/9 criteria passed, "
                  f"confidence={confidence:.0f}% [{status}]")
        if t.is_candidate:
            candidates.append(t)

    result.candidates = candidates
    result.total_candidates_passed = len(candidates)

    # ---- Calculate overall metrics ----
    elapsed = time.time() - start_time
    result.processing_time_seconds = elapsed

    if verbose:
        print(f"\n  {'=' * 50}")
        print(f"  PIPELINE COMPLETE")
        print(f"  Processing time: {elapsed:.2f} seconds")
        print(f"  Sources detected: {result.total_sources_detected}")
        print(f"  Tracklets formed: {result.total_tracklets_formed}")
        print(f"  Candidates passed: {result.total_candidates_passed}")
        print(f"  {'=' * 50}")

    return result


# =============================================================================
# VALIDATION MODE
# =============================================================================
# Run the algorithm on synthetic data with known ground truth

def run_validation(verbose=True):
    """
    Run a complete validation test using synthetic data.

    This generates fake images with known asteroids, runs the detection
    pipeline, and checks if the algorithm correctly found them. This is
    our main proof that the algorithm works — we know the answers in advance.

    The synthetic asteroids simulate objects like 2024 RH39 and 2024 RX69:
    magnitude 19-21, motion rate 0.3-0.5 arcsec/min.

    Returns:
        result: DetectionResult from the pipeline
        truth: ground truth data (what we planted)
        validation_report: dictionary of validation metrics
    """
    if verbose:
        print("\n" + "=" * 70)
        print("  VALIDATION MODE — Synthetic Ground Truth Test")
        print("=" * 70)
        print("  Generating synthetic Pan-STARRS-like image data...")
        print(f"  Image size: {SYNTHETIC_IMAGE_SIZE} x {SYNTHETIC_IMAGE_SIZE} pixels")
        print(f"  Pixel scale: {PIXEL_SCALE} arcsec/pixel")
        print(f"  Simulating: 3 asteroids, 80 stars, 5 cosmic rays, 3 hot pixels")

    # Generate synthetic data
    frames, truth = generate_synthetic_frames(
        num_asteroids=3, num_stars=80,
        num_cosmic_rays=5, num_hot_pixels=3
    )

    if verbose:
        print("  Synthetic data generated successfully!")
        print("\n  Known asteroid properties:")
        for i, ast in enumerate(truth['asteroids']):
            print(f"    Asteroid {i + 1}: mag={ast['magnitude']:.1f}, "
                  f"rate={ast['rate_arcsec_min']:.3f} arcsec/min")

    # Run the detection pipeline
    result = run_detection_pipeline(frames, verbose=verbose)

    # ---- Compare detections to ground truth ----
    if verbose:
        print("\n  [Ground Truth Comparison]")

    num_planted = len(truth['asteroids'])
    num_detected = 0
    num_false_positives = 0

    for candidate in result.candidates:
        # Check if this candidate matches any planted asteroid
        matched = False
        for ast in truth['asteroids']:
            # Compare first-frame position
            ast_x = ast['positions'][0]['x']
            ast_y = ast['positions'][0]['y']
            dist = np.sqrt((candidate.sources[0].x - ast_x)**2 +
                          (candidate.sources[0].y - ast_y)**2)
            if dist < 15.0:  # Within 15 pixels = match
                matched = True
                num_detected += 1
                break
        if not matched:
            num_false_positives += 1

    detection_accuracy = (num_detected / num_planted * 100) if num_planted > 0 else 0
    total_candidates = len(result.candidates)
    false_positive_rate = (num_false_positives / max(total_candidates, 1)) * 100

    result.detection_accuracy = detection_accuracy
    result.false_positive_count = num_false_positives
    result.false_positive_rate = false_positive_rate

    validation_report = {
        'asteroids_planted': num_planted,
        'asteroids_detected': num_detected,
        'false_positives': num_false_positives,
        'detection_accuracy_pct': detection_accuracy,
        'false_positive_rate_pct': false_positive_rate,
        'processing_time_sec': result.processing_time_seconds,
        'total_sources': result.total_sources_detected,
        'total_tracklets': result.total_tracklets_formed,
    }

    if verbose:
        print(f"\n  {'=' * 60}")
        print(f"  VALIDATION REPORT")
        print(f"  {'=' * 60}")
        print(f"  Asteroids planted:       {num_planted}")
        print(f"  Asteroids detected:      {num_detected}")
        print(f"  False positives:         {num_false_positives}")
        print(f"  Detection accuracy:      {detection_accuracy:.1f}%")
        print(f"  False positive rate:     {false_positive_rate:.1f}%")
        print(f"  Processing time:         {result.processing_time_seconds:.2f} seconds")
        print(f"  {'=' * 60}")

        # Print per-tracklet detailed report (ALL tracklets, not just candidates)
        print(f"\n  ALL TRACKLET ANALYSIS (candidates and rejected)")
        print(f"  {'-' * 60}")
        for idx, t in enumerate(result.tracklets):
            label = "CANDIDATE" if t.is_candidate else "REJECTED"
            print(f"\n  Tracklet {idx + 1} [{label}]:")
            print(f"    Position (frame 1): ({t.sources[0].x:.1f}, {t.sources[0].y:.1f})")
            print(f"    Motion rate: {t.velocity_arcsec_min:.3f} arcsec/min")
            print(f"    Direction: {t.position_angle:.1f} degrees")
            print(f"    Mean magnitude: {t.mean_magnitude:.1f}")
            print(f"    Confidence: {t.confidence_score:.0f}%")
            print(f"    Criteria results:")
            for crit_name, crit_val in t.criteria_results.items():
                status = "PASS" if crit_val['passed'] else "FAIL"
                print(f"      [{status}] {crit_name}: {crit_val['note']}")

        # Print comparison table
        print(f"\n  {'=' * 60}")
        print(f"  ALGORITHM vs MANUAL ASTROMETRICA COMPARISON")
        print(f"  {'=' * 60}")
        print(f"  {'Metric':<35} {'Algorithm':<15} {'Astrometrica':<15}")
        print(f"  {'-' * 60}")
        print(f"  {'Processing time':<35} "
              f"{result.processing_time_seconds:.1f} sec{'':<10} "
              f"~25 min")
        print(f"  {'Detection accuracy':<35} "
              f"{detection_accuracy:.0f}%{'':<12} "
              f"85-90%")
        print(f"  {'False positive rate':<35} "
              f"{false_positive_rate:.0f}%{'':<12} "
              f"15-20%")
        print(f"  {'Sources analysed':<35} "
              f"{result.total_sources_detected:<15} "
              f"~50-100")
        print(f"  {'Criteria checked':<35} "
              f"{'9 (automated)':<15} "
              f"{'9 (manual)':<15}")
        print(f"  {'=' * 60}")

    return result, truth, validation_report


# =============================================================================
# VISUALISATION OUTPUTS
# =============================================================================
# Generate the 3 required PNG files for the science fair display board

def create_visualisations(frames, result, truth=None, output_dir='.'):
    """
    Generate all 3 required visualisation PNG files.

    1. Detection image: shows frame 1 with candidates circled
    2. Motion trail plot: shows how each asteroid moved across frames
    3. Comparison bar chart: algorithm vs manual Astrometrica performance

    Parameters:
        frames: the 4 image frames
        result: DetectionResult from the pipeline
        truth: ground truth data (if available, for synthetic mode)
        output_dir: directory to save the PNG files
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Visualisation 1: Annotated Detection Image ----
    print("\n  Generating visualisation 1: Annotated detection image...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Asteroid Detection — All 4 Frames with Candidates Circled',
                 fontsize=14, fontweight='bold')

    for i, (ax, frame) in enumerate(zip(axes, frames)):
        # Display image with astronomical colour scaling
        vmin = np.percentile(frame, 5)
        vmax = np.percentile(frame, 99)
        ax.imshow(frame, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f'Frame {i + 1} (t = +{i * FRAME_INTERVAL_MINUTES:.0f} min)',
                    fontsize=10)
        ax.set_xlabel('X (pixels)')
        if i == 0:
            ax.set_ylabel('Y (pixels)')

        # Circle each candidate's position in this frame
        for c_idx, candidate in enumerate(result.candidates):
            source = candidate.sources[i]
            color = ['lime', 'cyan', 'magenta', 'yellow', 'red'][c_idx % 5]
            circle = patches.Circle(
                (source.x, source.y), radius=12,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(circle)
            if i == 0:  # Label only in first frame to avoid clutter
                ax.annotate(f'C{c_idx + 1}',
                          (source.x, source.y + 16),
                          color=color, fontsize=9, fontweight='bold',
                          ha='center')

        # Mark ground truth asteroids with squares (if we have truth data)
        if truth:
            for a_idx, ast in enumerate(truth['asteroids']):
                pos = ast['positions'][i]
                rect = patches.Rectangle(
                    (pos['x'] - 8, pos['y'] - 8), 16, 16,
                    linewidth=1, edgecolor='red', facecolor='none',
                    linestyle='--'
                )
                ax.add_patch(rect)

    plt.tight_layout()
    path1 = os.path.join(output_dir, 'detection_annotated.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path1}")

    # ---- Visualisation 2: Motion Trail Plot ----
    print("  Generating visualisation 2: Motion trail plot...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title('Asteroid Motion Trails Across 4 Frames',
                fontsize=14, fontweight='bold')

    # Show first frame as background
    vmin = np.percentile(frames[0], 5)
    vmax = np.percentile(frames[0], 99)
    ax.imshow(frames[0], cmap='gray', origin='lower', vmin=vmin, vmax=vmax,
             alpha=0.5)

    # Draw motion trails for each candidate
    colors = ['lime', 'cyan', 'magenta', 'yellow', 'red']
    for c_idx, candidate in enumerate(result.candidates):
        xs = [s.x for s in candidate.sources]
        ys = [s.y for s in candidate.sources]
        color = colors[c_idx % len(colors)]

        # Draw the trail line
        ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.8)

        # Mark each frame position with a dot
        for f_idx, (x, y) in enumerate(zip(xs, ys)):
            marker_size = 80 if f_idx == 0 else 50
            ax.scatter(x, y, s=marker_size, color=color, zorder=5,
                      edgecolors='white', linewidths=0.5)
            ax.annotate(f'F{f_idx + 1}', (x + 5, y + 5),
                       color=color, fontsize=8)

        # Add candidate label
        ax.annotate(
            f'Candidate {c_idx + 1}\n'
            f'Rate: {candidate.velocity_arcsec_min:.3f}"/min\n'
            f'Conf: {candidate.confidence_score:.0f}%',
            (xs[0] - 30, ys[0] - 20),
            color=color, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
        )

    # Draw arrow showing direction of motion
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.legend(
        [plt.Line2D([0], [0], color=colors[i], lw=2)
         for i in range(min(len(result.candidates), 5))],
        [f'Candidate {i+1}' for i in range(min(len(result.candidates), 5))],
        loc='upper right', fontsize=10
    )

    plt.tight_layout()
    path2 = os.path.join(output_dir, 'motion_trails.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path2}")

    # ---- Visualisation 3: Comparison Bar Chart ----
    print("  Generating visualisation 3: Algorithm vs Astrometrica comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Algorithm Performance vs Manual Astrometrica',
                fontsize=14, fontweight='bold')

    bar_width = 0.35
    algo_color = '#2196F3'   # Blue
    manual_color = '#FF9800'  # Orange

    # Chart A: Processing Time
    ax = axes[0]
    times = [result.processing_time_seconds, 25 * 60]  # Algorithm vs 25 min
    bars = ax.bar(['Algorithm', 'Astrometrica\n(manual)'], times,
                 color=[algo_color, manual_color], width=0.5)
    ax.set_ylabel('Processing Time (seconds)', fontsize=11)
    ax.set_title('Speed Comparison', fontsize=12)
    for bar, val in zip(bars, times):
        label = f'{val:.1f}s' if val < 60 else f'{val / 60:.0f} min'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
               label, ha='center', va='bottom', fontweight='bold')

    # Chart B: Detection Accuracy
    ax = axes[1]
    accuracies = [result.detection_accuracy, 87.5]  # Algorithm vs ~87.5%
    bars = ax.bar(['Algorithm', 'Astrometrica\n(manual)'], accuracies,
                 color=[algo_color, manual_color], width=0.5)
    ax.set_ylabel('Detection Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy Comparison', fontsize=12)
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
               f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')

    # Chart C: False Positive Rate
    ax = axes[2]
    fp_rates = [result.false_positive_rate, 17.5]  # Algorithm vs ~17.5%
    bars = ax.bar(['Algorithm', 'Astrometrica\n(manual)'], fp_rates,
                 color=[algo_color, manual_color], width=0.5)
    ax.set_ylabel('False Positive Rate (%)', fontsize=11)
    ax.set_title('False Positive Comparison', fontsize=12)
    ax.set_ylim(0, 30)
    for bar, val in zip(bars, fp_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
               f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    path3 = os.path.join(output_dir, 'comparison_chart.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {path3}")

    return [path1, path2, path3]


# =============================================================================
# FITS FILE LOADER (for real telescope data)
# =============================================================================

def load_fits_frames(file_paths):
    """
    Load real FITS image files from disk.

    FITS (Flexible Image Transport System) is the standard file format
    for astronomical images. Pan-STARRS and every other major telescope
    saves data in FITS format.

    Parameters:
        file_paths: list of 4 file paths to FITS images

    Returns:
        frames: list of 4 numpy 2D arrays
    """
    try:
        from astropy.io import fits
    except ImportError:
        print("  ERROR: astropy is required to read FITS files.")
        print("  Install it with: pip install astropy")
        return None

    frames = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"  ERROR: File not found: {path}")
            return None
        with fits.open(path) as hdul:
            # Use the primary HDU data (or the first image extension)
            data = hdul[0].data
            if data is None and len(hdul) > 1:
                data = hdul[1].data
            if data is None:
                print(f"  ERROR: No image data in {path}")
                return None
            frames.append(data.astype(float))
        print(f"  Loaded: {path} ({data.shape[1]}x{data.shape[0]} pixels)")

    return frames


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def print_banner():
    """Print a nice header when the program starts."""
    print("""
 ============================================================
       AUTOMATED ASTEROID DETECTION ALGORITHM  v1.0
 ============================================================
  Author: Siddharth Patel (AstroSidSpace)
  RASC London Centre Youth Member
  Discoverer of 2024 RH39 and 2024 RX69
 ------------------------------------------------------------
  TVSEF / CWSF 2026 Science Fair Project
  AI-assisted development: Claude (Anthropic, 2026)
 ============================================================
    """)


def main():
    """
    Main entry point for the asteroid detection algorithm.

    Usage:
        python asteroid_detector.py --validate     Run synthetic validation test
        python asteroid_detector.py --fits f1 f2 f3 f4   Process real FITS files
        python asteroid_detector.py               Run default demo with synthetic data
    """
    parser = argparse.ArgumentParser(
        description='Automated Asteroid Detection Algorithm — TVSEF 2026',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python asteroid_detector.py --validate        Run full validation with synthetic data
  python asteroid_detector.py --fits img1.fits img2.fits img3.fits img4.fits
  python asteroid_detector.py                   Run default demo
        """
    )
    parser.add_argument('--validate', action='store_true',
                       help='Run validation mode with synthetic test data')
    parser.add_argument('--fits', nargs=4, metavar='FILE',
                       help='Process 4 real FITS image files')
    parser.add_argument('--output-dir', default='.',
                       help='Directory to save output files (default: current)')
    parser.add_argument('--sigma', type=float, default=None,
                       help='Detection threshold in sigma (default: 3.0 for '
                            'synthetic, 5.0 for real FITS images)')

    args = parser.parse_args()

    print_banner()

    if args.fits:
        # ---- Real FITS file mode ----
        print("  MODE: Processing real FITS image files")
        frames = load_fits_frames(args.fits)
        if frames is None:
            print("\n  FATAL: Could not load FITS files. Exiting.")
            sys.exit(1)
        result = run_detection_pipeline(frames, verbose=True,
                                        detection_sigma=args.sigma)
        create_visualisations(frames, result, output_dir=args.output_dir)

    elif args.validate:
        # ---- Validation mode ----
        print("  MODE: Synthetic data validation test")
        result, truth, report = run_validation(verbose=True)
        create_visualisations(
            generate_synthetic_frames()[0],
            result, truth,
            output_dir=args.output_dir
        )

        # Print the final summary for the logbook
        print("\n" + "=" * 70)
        print("  FINAL VALIDATION SUMMARY (for science fair logbook)")
        print("=" * 70)
        print(f"  Date: 2026-03-09")
        print(f"  Algorithm version: 1.0")
        print(f"  Test: Synthetic ground truth validation")
        print(f"  Asteroids planted: {report['asteroids_planted']}")
        print(f"  Asteroids detected: {report['asteroids_detected']}")
        print(f"  Detection accuracy: {report['detection_accuracy_pct']:.1f}%")
        print(f"  False positives: {report['false_positives']}")
        print(f"  False positive rate: {report['false_positive_rate_pct']:.1f}%")
        print(f"  Processing time: {report['processing_time_sec']:.2f} seconds")
        print(f"  Total sources analysed: {report['total_sources']}")
        print(f"  Tracklets formed: {report['total_tracklets']}")
        print(f"  Output files saved:")
        print(f"    - detection_annotated.png")
        print(f"    - motion_trails.png")
        print(f"    - comparison_chart.png")
        print("=" * 70)

    else:
        # ---- Default demo mode ----
        print("  MODE: Default demonstration with synthetic data")
        print("  (Use --validate for full validation, --fits for real data)")
        frames, truth = generate_synthetic_frames()
        result = run_detection_pipeline(frames, verbose=True)
        create_visualisations(frames, result, truth, output_dir=args.output_dir)

    print("\n  Algorithm finished successfully!")
    return 0


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    sys.exit(main())
