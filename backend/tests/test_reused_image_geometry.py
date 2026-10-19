import numpy as np
import pytest


def test_cached_geometry_matches_original_coordinates_and_face_pixels():
    from app.initial_observation import _pixel_grid
    from app.piece_classifier import _piece_face_mask
    for height, width in ((29, 31), (48, 48), (61, 59)):
        yy, xx = _pixel_grid(height, width)
        original_y, original_x = np.mgrid[:height, :width]
        np.testing.assert_array_equal(yy, original_y)
        np.testing.assert_array_equal(xx, original_x)
        expected = ((original_x - (width - 1) / 2) ** 2 + (original_y - (height - 1) / 2) ** 2
                    < (min(height, width) * .35) ** 2)
        np.testing.assert_array_equal(_piece_face_mask(height, width), expected)
        assert _pixel_grid(height, width)[0] is yy
        with pytest.raises(ValueError):
            yy[0, 0] = 5
        with pytest.raises(ValueError):
            _piece_face_mask(height, width)[0, 0] = False
