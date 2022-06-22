"""Convert volume TIF to X3D file for web-based visualisation."""

import argparse
import os

from shapr.utils import import_image

from skimage.filters import threshold_otsu

from tqdm import tqdm

from vedo import Plotter
from vedo import Volume


def to_x3d(name, image_data):
    """Convert image data to X3D file.

    Parameters
    ----------
    name : str
        Name to use for the converted file

    image_data : np.array
        Raw image data
    """
    plt = Plotter(size=(600, 600), bg='GhostWhite')
    volume = Volume(image_data).isosurface()

    plt.show(volume, axes=1, viewup='z')
    plt.export(f'/tmp/{name}.x3d', binary=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'FILE',
        type=str,
        nargs='+',
        help='Input file(s)'
    )

    args = parser.parse_args()

    for filename in tqdm(args.FILE, desc='File'):
        image_data = import_image(filename).squeeze()

        to_x3d(os.path.splitext(
            filename)[0], image_data
        )
