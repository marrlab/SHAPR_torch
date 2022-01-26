"""Viewer for volumetric result images."""

import argparse

from vedo import Volume

from vedo.applications import IsosurfaceBrowser
from vedo.applications import RayCastPlotter

from shapr.utils import import_image


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

    args = parser.parse_args()

    filename = args.INPUT
    image_data = import_image(filename)

    volume = Volume(image_data)

    if args.mode == 'ray':
        plotter = RayCastPlotter(volume)
    else:
        plotter = IsosurfaceBrowser(volume, c='gold')

    plotter.show(axes=7).close()
