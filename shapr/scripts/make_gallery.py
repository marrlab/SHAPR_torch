"""Create gallery of volumetric result images."""

import argparse
import glob
import gc
import math
import os

import matplotlib.pyplot as plt

from vedo import Volume
from vedo import Plotter
from vedo import settings

from shapr.utils import import_image

from skimage.filters import threshold_otsu


def OffscreenIsosurfaceBrowser(
    volume,
    c=None,
    alpha=1,
    lego=False,
    cmap='hot',
    pos=None,
    delayed=False
):
    vp = settings.plotter_instance
    if not vp:
        vp = Plotter(
            axes=4, bg='w', title="Isosurface Browser",
            offscreen=True
        )

    # TODO: Super hacky, but it works...
    vp.camera.SetViewUp(-0.137737, 0.561213, 0.81613)
    vp.camera.SetPosition(30.6638, -88.3588, 130.57)

    scrange = volume.scalarRange()
    threshold = (scrange[1] - scrange[0]) / 3.0 + scrange[0]

    if lego:
        sliderpos = ((0.79, 0.035), (0.975, 0.035))
        slidertitle = ""
        showval = False
        mesh = volume.legosurface(vmin=threshold, cmap=cmap).alpha(alpha)
        mesh.addScalarBar(horizontal=True)
    else:
        sliderpos = 4
        slidertitle = "threshold"
        showval = True
        mesh = volume.isosurface(threshold)
        mesh.color(c).alpha(alpha)

    if pos is not None:
        sliderpos = pos

    vp.actors = [mesh] + vp.actors

    ############################## threshold slider
    bacts = dict()
    def sliderThres(widget, event):

        prevact = vp.actors[0]
        wval =  widget.GetRepresentation().GetValue()
        wval_2 = precision(wval, 2)
        if wval_2 in bacts.keys():  # reusing the already available mesh
            mesh = bacts[wval_2]
        else:                       # else generate it
            if lego:
                mesh = volume.legosurface(vmin=wval, cmap=cmap)
            else:
                mesh = volume.isosurface(threshold=wval).color(c).alpha(alpha)
            bacts.update({wval_2: mesh}) # store it

        vp.renderer.RemoveActor(prevact)
        vp.renderer.AddActor(mesh)
        vp.actors[0] = mesh

    dr = scrange[1] - scrange[0]
    vp.addSlider2D( sliderThres,
                    scrange[0] + 0.02 * dr,
                    scrange[1] - 0.02 * dr,
                    value=threshold,
                    pos=sliderpos,
                    title=slidertitle,
                    showValue=showval,
                    delayed=delayed,
                    )
    return vp


def OffscreenRayCastPlotter(volume):
    vp = settings.plotter_instance
    if not vp:
        vp = Plotter(axes=4, bg='bb')

    volumeProperty = volume.GetProperty()
    img = volume.imagedata()

    if volume.dimensions()[2]<3:
        print("Error in raycaster: not enough depth", volume.dimensions())
        return vp
    # printc("GPU Ray-casting tool", c="b", invert=1)

    smin, smax = img.GetScalarRange()

    x0alpha = smin + (smax - smin) * 0.25
    x1alpha = smin + (smax - smin) * 0.5
    x2alpha = smin + (smax - smin) * 1.0

    ############################## color map slider
    # Create transfer mapping scalar value to color
    cmaps = ["jet",
            "viridis",
            "bone",
            "hot",
            "plasma",
            "winter",
            "cool",
            "gist_earth",
            "coolwarm",
            "tab10",
            ]
    cols_cmaps = []
    for cm in cmaps:
        cols = colorMap(range(0, 21), cm, 0, 20)  # sample 20 colors
        cols_cmaps.append(cols)
    Ncols = len(cmaps)
    csl = (0.9, 0.9, 0.9)
    if sum(getColor(vp.renderer.GetBackground())) > 1.5:
        csl = (0.1, 0.1, 0.1)

    def sliderColorMap(widget, event):
        sliderRep = widget.GetRepresentation()
        k = int(sliderRep.GetValue())
        sliderRep.SetTitleText(cmaps[k])
        volume.color(cmaps[k])

    w1 = vp.addSlider2D(
        sliderColorMap,
        0,
        Ncols - 1,
        value=0,
        showValue=0,
        title=cmaps[0],
        c=csl,
        pos=[(0.8, 0.05), (0.965, 0.05)],
    )
    w1.GetRepresentation().SetTitleHeight(0.018)

    ############################## alpha sliders
    # Create transfer mapping scalar value to opacity
    opacityTransferFunction = volumeProperty.GetScalarOpacity()

    def setOTF():
        opacityTransferFunction.RemoveAllPoints()
        opacityTransferFunction.AddPoint(smin, 0.0)
        opacityTransferFunction.AddPoint(smin + (smax - smin) * 0.1, 0.0)
        opacityTransferFunction.AddPoint(x0alpha, _alphaslider0)
        opacityTransferFunction.AddPoint(x1alpha, _alphaslider1)
        opacityTransferFunction.AddPoint(x2alpha, _alphaslider2)

    setOTF()

    def sliderA0(widget, event):
        global _alphaslider0
        _alphaslider0 = widget.GetRepresentation().GetValue()
        setOTF()

    vp.addSlider2D(sliderA0, 0, 1,
                    value=_alphaslider0,
                    pos=[(0.84, 0.1), (0.84, 0.26)],
                    c=csl, showValue=0)

    def sliderA1(widget, event):
        global _alphaslider1
        _alphaslider1 = widget.GetRepresentation().GetValue()
        setOTF()

    vp.addSlider2D(sliderA1, 0, 1,
                    value=_alphaslider1,
                    pos=[(0.89, 0.1), (0.89, 0.26)],
                    c=csl, showValue=0)

    def sliderA2(widget, event):
        global _alphaslider2
        _alphaslider2 = widget.GetRepresentation().GetValue()
        setOTF()

    w2 = vp.addSlider2D(sliderA2, 0, 1,
                        value=_alphaslider2,
                        pos=[(0.96, 0.1), (0.96, 0.26)],
                        c=csl, showValue=0,
                        title="Opacity levels")
    w2.GetRepresentation().SetTitleHeight(0.016)

    # add a button
    def buttonfuncMode():
        s = volume.mode()
        snew = (s + 1) % 2
        volume.mode(snew)
        bum.switch()

    bum = vp.addButton(
        buttonfuncMode,
        pos=(0.7, 0.035),
        states=["composite", "max proj."],
        c=["bb", "gray"],
        bc=["gray", "bb"],  # colors of states
        font="",
        size=16,
        bold=0,
        italic=False,
    )
    bum.status(volume.mode())

    # add histogram of scalar
    plot = CornerHistogram(volume,
        bins=25, logscale=1, c=(.7,.7,.7), bg=(.7,.7,.7), pos=(0.78, 0.065),
        lines=True, dots=False,
        nmax=3.1415e+06, # subsample otherwise is too slow
    )

    plot.GetPosition2Coordinate().SetValue(0.197, 0.20, 0)
    plot.GetXAxisActor2D().SetFontFactor(0.7)
    plot.GetProperty().SetOpacity(0.5)
    vp.add([plot, volume])
    return vp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('DIRECTORY', type=str, help='Input directory')

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

    filenames = sorted(glob.glob(os.path.join(args.DIRECTORY, '*.tif')))

    n_rows = int(math.ceil(math.sqrt(len(filenames)) + 0.5))
    fig, axes = plt.subplots(n_rows, n_rows, figsize=(8, 8))

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_visible(False)

    for filename, ax in zip(filenames, axes.ravel()):
        image_data = import_image(filename).squeeze()

        if args.threshold:
            thres = threshold_otsu(image_data)
            image_data = image_data > thres
            image_data = image_data.astype(float)
            image_data *= 1.0

        volume = Volume(image_data)

        if args.mode == 'ray':
            plotter = OffscreenRayCastPlotter(volume)
        else:
            plotter = OffscreenIsosurfaceBrowser(volume, c='gold')

        plotter.show()
        image = plotter.screenshot(returnNumpy=True)
        plotter.close()


        ax.set_visible(True)
        ax.imshow(image)

    plt.tight_layout(pad=0.0)
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()
