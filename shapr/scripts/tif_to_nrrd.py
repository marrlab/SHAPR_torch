"""Convert volumetric TIF to NRRD file for better visualisation."""

import argparse
import nrrd
import os

from shapr.utils import import_image
from tqdm import tqdm


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

        out_name = os.path.splitext(filename)[0] + '.nrrd'
        nrrd.write(out_name, image_data)
