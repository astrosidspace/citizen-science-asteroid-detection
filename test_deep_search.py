"""Tests for Phase 1b Deep Search: Optimal Image Coaddition + Bayesian Scoring."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))


def _make_gaussian_psf(fwhm_pix=4.0, size=21):
    """Create a normalized 2D Gaussian PSF kernel."""
    sigma = fwhm_pix / 2.355
    half = size // 2
    y, x = np.ogrid[-half:half+1, -half:half+1]
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return psf / psf.sum()


def _make_synthetic_frame(h=500, w=500, n_stars=50, bg_level=1000,
                          bg_noise=15.0, fwhm_pix=4.0, seed=42):
    """Create a synthetic frame with stars and Gaussian noise."""
    rng = np.random.RandomState(seed)
    frame = rng.normal(bg_level, bg_noise, (h, w)).astype(np.float64)
    psf = _make_gaussian_psf(fwhm_pix)
    ph, pw = psf.shape
    for _ in range(n_stars):
        sy = rng.randint(ph, h - ph)
        sx = rng.randint(pw, w - pw)
        flux = rng.uniform(500, 50000)
        frame[sy - ph//2:sy + ph//2 + 1,
              sx - pw//2:sx + pw//2 + 1] += psf * flux
    return frame


def test_estimate_frame_psf_returns_valid_kernel():
    """PSF estimation must return a normalized, positive, symmetric kernel."""
    import asteroid_detector as ad
    frame = _make_synthetic_frame(n_stars=80, fwhm_pix=4.0)
    noise = np.full_like(frame, 15.0)
    psf_kernel, psf_fwhm = ad._estimate_frame_psf(frame, noise)
    assert psf_kernel.ndim == 2
    assert psf_kernel.shape[0] % 2 == 1
    assert psf_kernel.shape[1] % 2 == 1
    assert abs(psf_kernel.sum() - 1.0) < 0.01
    assert 2.0 < psf_fwhm < 8.0, f"FWHM {psf_fwhm} out of range"
    cy, cx = psf_kernel.shape[0] // 2, psf_kernel.shape[1] // 2
    assert psf_kernel[cy, cx] == psf_kernel.max()


def test_estimate_frame_psf_handles_noisy_frame():
    """PSF estimation should still work with very noisy data."""
    import asteroid_detector as ad
    frame = _make_synthetic_frame(n_stars=30, bg_noise=50.0, fwhm_pix=5.0)
    noise = np.full_like(frame, 50.0)
    psf_kernel, psf_fwhm = ad._estimate_frame_psf(frame, noise)
    assert psf_kernel.ndim == 2
    assert psf_kernel.sum() > 0.9
    assert 1.5 < psf_fwhm < 12.0


def test_matched_filter_stack_recovers_faint_source():
    """Matched-filter stacking at correct velocity recovers faint injected source."""
    import asteroid_detector as ad
    h, w, n_frames = 500, 500, 4
    bg_noise = 15.0
    fwhm_pix = 4.0
    psf = _make_gaussian_psf(fwhm_pix)
    ph, pw = psf.shape
    asteroid_flux = 400.0  # per-frame SNR ~1.5, stacked SNR ~7
    x0, y0 = 250.0, 250.0
    vx, vy = 6.0, -4.0

    rng = np.random.RandomState(123)
    frames = []
    noise_maps = []
    for i in range(n_frames):
        frame = rng.normal(1000, bg_noise, (h, w)).astype(np.float64)
        ax = int(round(x0 + vx * i))
        ay = int(round(y0 + vy * i))
        frame[ay - ph//2:ay + ph//2 + 1, ax - pw//2:ax + pw//2 + 1] += psf * asteroid_flux
        frames.append(frame)
        noise_maps.append(np.full((h, w), bg_noise))

    reference = np.median(np.array(frames), axis=0)
    diff_frames = [f - reference for f in frames]
    star_mask = np.zeros((h, w), dtype=bool)
    psf_kernel = _make_gaussian_psf(fwhm_pix, size=21)
    noise_var = [nm**2 for nm in noise_maps]

    # Pre-filter with PSF matched filter (now done externally)
    from scipy.signal import fftconvolve
    psf_flipped = psf_kernel[::-1, ::-1]
    filtered_frames = [fftconvolve(df, psf_flipped, mode='same') for df in diff_frames]

    score_image = ad._matched_filter_stack(filtered_frames, psf_kernel, noise_var, star_mask, vx=vx, vy=vy, n_frames=n_frames)
    peak_y, peak_x = np.unravel_index(np.argmax(score_image), score_image.shape)
    assert abs(peak_x - x0) < 10, f"Peak X {peak_x} too far from {x0}"
    assert abs(peak_y - y0) < 10, f"Peak Y {peak_y} too far from {y0}"
    peak_snr = score_image[peak_y, peak_x]
    assert peak_snr > 3.0, f"Peak SNR {peak_snr:.2f} below 3.0"


def test_matched_filter_wrong_velocity_no_peak():
    """At WRONG velocity, matched filter should NOT recover the source."""
    import asteroid_detector as ad
    h, w, n_frames = 500, 500, 4
    bg_noise = 15.0
    fwhm_pix = 4.0
    psf = _make_gaussian_psf(fwhm_pix)
    ph, pw = psf.shape
    asteroid_flux = 400.0
    x0, y0 = 250.0, 250.0
    vx_true, vy_true = 6.0, -4.0

    rng = np.random.RandomState(123)
    frames = []
    noise_maps = []
    for i in range(n_frames):
        frame = rng.normal(1000, bg_noise, (h, w)).astype(np.float64)
        ax = int(round(x0 + vx_true * i))
        ay = int(round(y0 + vy_true * i))
        frame[ay - ph//2:ay + ph//2 + 1, ax - pw//2:ax + pw//2 + 1] += psf * asteroid_flux
        frames.append(frame)
        noise_maps.append(np.full((h, w), bg_noise))

    reference = np.median(np.array(frames), axis=0)
    diff_frames = [f - reference for f in frames]
    star_mask = np.zeros((h, w), dtype=bool)
    psf_kernel = _make_gaussian_psf(fwhm_pix, size=21)
    noise_var = [nm**2 for nm in noise_maps]

    # Pre-filter with PSF matched filter (now done externally)
    from scipy.signal import fftconvolve
    psf_flipped = psf_kernel[::-1, ::-1]
    filtered_frames = [fftconvolve(df, psf_flipped, mode='same') for df in diff_frames]

    score_wrong = ad._matched_filter_stack(filtered_frames, psf_kernel, noise_var, star_mask, vx=-4.0, vy=-6.0, n_frames=n_frames)
    score_right = ad._matched_filter_stack(filtered_frames, psf_kernel, noise_var, star_mask, vx=vx_true, vy=vy_true, n_frames=n_frames)
    wrong_peak = score_wrong.max()
    right_peak = score_right.max()
    assert right_peak > wrong_peak * 1.1, f"Correct peak {right_peak:.2f} not sufficiently above wrong {wrong_peak:.2f}"


def test_bcs_high_confidence_real_asteroid():
    """Good SNR + consistent flux + good PSF + MBA velocity → high score."""
    import asteroid_detector as ad
    score = ad._bayesian_candidate_score(
        stacked_snr=5.0,
        frame_fluxes=[100, 105, 98, 102],
        frame_noises=[15, 15, 15, 15],
        measured_fwhm=4.0,
        expected_fwhm=4.0,
        ellipticity=0.1,
        velocity_arcsec_min=0.5,
    )
    assert score > 0.70, f"Real asteroid BCS score {score:.3f} below 0.70"


def test_bcs_low_score_for_noise_spike():
    """Marginal SNR + inconsistent flux + bad PSF + fast velocity → low score."""
    import asteroid_detector as ad
    score = ad._bayesian_candidate_score(
        stacked_snr=3.6,
        frame_fluxes=[200, 5, 10, 180],
        frame_noises=[15, 15, 15, 15],
        measured_fwhm=8.0,
        expected_fwhm=4.0,
        ellipticity=0.5,
        velocity_arcsec_min=3.5,
    )
    assert score < 0.30, f"Noise spike BCS score {score:.3f} should be below 0.30"


def test_bcs_returns_float_between_0_and_1():
    """BCS score must always be a valid probability."""
    import asteroid_detector as ad
    for snr in [3.5, 5.0, 10.0, 50.0]:
        score = ad._bayesian_candidate_score(
            stacked_snr=snr,
            frame_fluxes=[100, 100, 100, 100],
            frame_noises=[15, 15, 15, 15],
            measured_fwhm=4.0,
            expected_fwhm=4.0,
            ellipticity=0.1,
            velocity_arcsec_min=0.5,
        )
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] range for SNR={snr}"


def test_deep_search_finds_faint_asteroid_missed_by_standard():
    """
    End-to-end: inject faint asteroid (per-frame SNR ~2.5) that sigma=5 misses.
    Phase 1b deep search should recover it.
    """
    import asteroid_detector as ad
    h, w, n_frames = 500, 500, 4
    bg_noise = 15.0
    fwhm_pix = 4.0
    psf = _make_gaussian_psf(fwhm_pix)
    ph, pw = psf.shape

    asteroid_flux = 400.0
    x0, y0 = 250.0, 250.0
    vx, vy = 8.0, -4.0  # exceeds DEEP_MIN_VELOCITY_PX=6

    rng = np.random.RandomState(456)
    frames = []
    subtracted = []
    noise_maps = []
    for i in range(n_frames):
        frame = rng.normal(1000, bg_noise, (h, w)).astype(np.float64)
        ax = int(round(x0 + vx * i))
        ay = int(round(y0 + vy * i))
        frame[ay - ph//2:ay + ph//2 + 1, ax - pw//2:ax + pw//2 + 1] += psf * asteroid_flux
        frames.append(frame)
        subtracted.append(frame - 1000)
        noise_maps.append(np.full((h, w), bg_noise))

    deep_detections = ad._deep_search_phase(
        frames, subtracted, noise_maps,
        frame_interval_minutes=16.3,
        verbose=False
    )

    found = False
    for det in deep_detections:
        dx = abs(det['x'] - x0)
        dy = abs(det['y'] - y0)
        if dx < 15 and dy < 15:
            found = True
            assert det['bcs_probability'] > 0.3, \
                f"BCS probability {det['bcs_probability']:.3f} too low"
            break

    assert found, (f"Deep search did not find asteroid at ({x0},{y0}). "
                   f"Got {len(deep_detections)} detections: "
                   f"{[(d['x'], d['y']) for d in deep_detections[:5]]}")


if __name__ == "__main__":
    test_estimate_frame_psf_returns_valid_kernel()
    print("PASS: test_estimate_frame_psf_returns_valid_kernel")
    test_estimate_frame_psf_handles_noisy_frame()
    print("PASS: test_estimate_frame_psf_handles_noisy_frame")
    test_matched_filter_stack_recovers_faint_source()
    print("PASS: test_matched_filter_stack_recovers_faint_source")
    test_matched_filter_wrong_velocity_no_peak()
    print("PASS: test_matched_filter_wrong_velocity_no_peak")
    test_bcs_high_confidence_real_asteroid()
    print("PASS: test_bcs_high_confidence_real_asteroid")
    test_bcs_low_score_for_noise_spike()
    print("PASS: test_bcs_low_score_for_noise_spike")
    test_bcs_returns_float_between_0_and_1()
    print("PASS: test_bcs_returns_float_between_0_and_1")
    test_deep_search_finds_faint_asteroid_missed_by_standard()
    print("PASS: test_deep_search_finds_faint_asteroid_missed_by_standard")
    print("All 8 tests passed!")
