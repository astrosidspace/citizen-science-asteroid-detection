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




# =============================================================================
# COMMAND-LINE INTERFACE (placeholder — will be expanded in later commits)
# =============================================================================

def main():
    """Main entry point — algorithm components will be added iteratively."""
    print("Automated Asteroid Detection Algorithm v0.1")
    print("Author: Siddharth Patel (AstroSidSpace)")
    print("Structure defined. Detection engine coming next.")
    return 0


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.exit(main())
