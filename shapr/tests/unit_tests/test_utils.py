from shapeae import import_image
#import pytest

def test_import_image():
    _image = import_image("shapr/tests/test_data/images/A_cell_clusters000604.tif")
    assert _image.shape == (64, 64)
