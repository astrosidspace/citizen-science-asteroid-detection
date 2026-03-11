"""Profile pipeline to find speed bottlenecks."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import asteroid_detector as ad

# Use RH39 field (XY26_p10) for profiling
fits_files = [
    r"C:/Users/astro/Downloads/Astrometrica/Data/3rd Sep/moving object found/ps2-20240903_8_XY26_p10/XY26_p10/o60556h0328o.782882.ch.714012.XY26.p10.fits",
    r"C:/Users/astro/Downloads/Astrometrica/Data/3rd Sep/moving object found/ps2-20240903_8_XY26_p10/XY26_p10/o60556h0345o.782899.ch.714029.XY26.p10.fits",
    r"C:/Users/astro/Downloads/Astrometrica/Data/3rd Sep/moving object found/ps2-20240903_8_XY26_p10/XY26_p10/o60556h0362o.782916.ch.714046.XY26.p10.fits",
    r"C:/Users/astro/Downloads/Astrometrica/Data/3rd Sep/moving object found/ps2-20240903_8_XY26_p10/XY26_p10/o60556h0379o.782933.ch.714063.XY26.p10.fits",
]

data = ad.load_fits_frames(fits_files)
frames = data['frames']
mjds = data['mjds']

print("Running pipeline with verbose=True to capture phase timing...")
result = ad.run_detection_pipeline(frames, verbose=True, detection_sigma=5.0, frame_mjds=mjds)
print(f"\nTotal: {result.processing_time_seconds:.1f}s")
