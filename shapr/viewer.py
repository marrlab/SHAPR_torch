"""Viewer for volumetric result images."""

import argparse

from vedo import Volume

from vedo.applications import IsosurfaceBrowser
from vedo.applications import RayCastPlotter

from shapr.utils import import_image

from skimage.filters import threshold_otsu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', type=str, help='Input filename')

    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=['ray', 'iso'],
        default='iso',
        help='Determine visualisation type'
    )

    parser.add_argument(
        '-t', '--threshold',
        action='store_true',
        help='If set, determines binarisation threshold automatically'
    )

    args = parser.parse_args()

    image_data = import_image(args.INPUT).squeeze()

    if args.threshold:
        thres = threshold_otsu(image_data)
        image_data = image_data > thres
        image_data = image_data.astype(float)
        image_data *= 1.0

    volume = Volume(image_data)

    if args.mode == 'ray':
        plotter = RayCastPlotter(volume)
    else:
        plotter = IsosurfaceBrowser(volume, c='gold')

    plotter.show(axes=7).close()
