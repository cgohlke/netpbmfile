Read and write Netpbm files
===========================

Netpbmfile is a Python library to read and write files in the Netpbm format
as specified at http://netpbm.sourceforge.net/doc/.

The following Netpbm formats are supported: PBM (bi-level), PGM (grayscale),
PPM (color), PAM (arbitrary), XV thumbnail (RGB332, read-only).
Also reads Portable FloatMap formats: PF (float32 RGB) and
Pf (float32 grayscale).

No gamma correction is performed. Only one image per file is supported.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2018.10.18

Requirements
------------
* `CPython 2.7 or 3.5+ <https://www.python.org>`_
* `Numpy 1.13 <https://www.numpy.org>`_
* `Matplotlib 2.2 <https://www.matplotlib.org>`_ (optional for plotting)

Revisions
---------
2018.10.18
    Move netpbmfile.py into netpbmfile package.
2018.02.18
    Support Portable FloatMaps.
    Style fixes.
2016.02.24
    Use 'fromdata' classmethod to initialize from data.
    Support 'with' statement.
    Scale RGB images to maxval for display.
    Make keyword arguments explicit.
    Support numpy 1.10.

Examples
--------
>>> im1 = numpy.array([[0, 1], [65534, 65535]], dtype='uint16')
>>> imsave('_tmp.pgm', im1)
>>> im2 = imread('_tmp.pgm')
>>> assert numpy.all(im1 == im2)
