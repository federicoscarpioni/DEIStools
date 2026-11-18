from deistools.processing.frequency_band import get_band_bins

def test_get_band_bins_even():
    ke = 100
    n_bins = 20
    n_samples = 400
    band = get_band_bins(ke, n_bins,n_samples)
    assert band.size == n_bins

def test_get_band_bins_odd():
    ke = 100
    n_bins = 21
    n_samples = 420
    band = get_band_bins(ke, n_bins,n_samples)
    assert band.size == n_bins