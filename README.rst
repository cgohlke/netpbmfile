Read and write Netpbm files
===========================

Netpbmfile is a Python library to read and write image files in the Netpbm
format as specified at http://netpbm.sourceforge.net/doc/.

The following Netpbm and Portable FloatMap formats are supported:

- PBM (bi-level)
- PGM (grayscale)
- PPM (color)
- PAM (arbitrary)
- XV thumbnail (RGB332, read-only)
- Pf (float32 grayscale, read-only)
- PF (float32 RGB, read-only)
- PF4 (float32 RGBA, read-only)
- PGX (signed grayscale, read-only)

No gamma correction is performed. Only one image per file is supported.

The PGX format is specified in ITU-T Rec. T.803.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2022.9.12

Requirements
------------

This release has been tested with the following requirements and dependencies
(other versions may work):

- `CPython 3.8.10, 3.9.13, 3.10.7, 3.11.0rc2 <https://www.python.org>`_
- `NumPy 1.22.4 <https://pypi.org/project/numpy/>`_
- `Matplotlib 3.5.3 <https://pypi.org/project/matplotlib/>`_  (optional)

Revisions
---------

2022.9.12

- Allow space after token value in PAM.
- Update metadata.

2022.2.2

- Add type hints.
- Support reading PF4 RGBA FloatMaps.
- Drop support for Python 3.7 and numpy < 1.19 (NEP29).

2021.6.6

- Fix unclosed file warnings.
- Support reading PGX JPEG2000 reference images.

2020.10.18

- Disallow comments after last value in PNM headers.

2020.9.18

- Remove support for Python 3.6 (NEP 29).
- Support os.PathLike file names.

2020.1.1

- Fix reading tightly packed P1 format and ASCII data with inline comments.
- Remove support for Python 2.7 and 3.5.
- Update copyright.

2018.10.18

- Move netpbmfile.py into netpbmfile package.

2018.02.18

- Support reading Portable FloatMaps.
- Style fixes.

2016.02.24

- Use fromdata classmethod to initialize from data.
- Support with statement.
- Scale RGB images to maxval for display.
- Make keyword arguments explicit.
- Support numpy 1.10.

Examples
--------

Save a numpy array to a Netpbm file in grayscale format:

>>> data = numpy.array([[0, 1], [65534, 65535]], dtype='uint16')
>>> imwrite('_tmp.pgm', data)

Read the image data from a Netpbm file as numpy array:

>>> image = imread('_tmp.pgm')
>>> assert numpy.all(image == data)

Access meta and image data in a Netpbm file:

>>> with NetpbmFile('_tmp.pgm') as pgm:
...     pgm.axes
...     pgm.shape
...     pgm.dtype
...     pgm.maxval
...     pgm.magicnum
...     image = pgm.asarray()
'YX'
(2, 2)
dtype('>u2')
65535
b'P5'

View the image stored in a Netpbm file from a command line::

    python -m netpbmfile _tmp.pgm
