from crome.bands import ALPHAEARTH_BANDS, alphaearth_band_names, validate_alphaearth_bands


def test_alphaearth_bands_are_complete_and_ordered() -> None:
    bands = alphaearth_band_names()
    assert len(bands) == 64
    assert bands[0] == "A00"
    assert bands[-1] == "A63"
    assert bands == ALPHAEARTH_BANDS


def test_validate_alphaearth_bands_rejects_partial_band_lists() -> None:
    try:
        validate_alphaearth_bands(("A00", "A01"))
    except ValueError as exc:
        assert "full AlphaEarth band set" in str(exc)
    else:
        raise AssertionError("Expected ValueError for partial AlphaEarth bands.")
