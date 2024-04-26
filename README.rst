Read and write Netpbm files
===========================

Netpbmfile is a Python library to read and write image files in the Netpbm
or related formats:

- PBM (Portable Bit Map): P1 (text) and P4 (binary)
- PGM (Portable Gray Map): P2 (text) and P5 (binary)
- PPM (Portable Pixel Map): P3 (text) and P6 (binary)
- PNM (Portable Any Map): shorthand for PBM, PGM, and PPM collectively
- PAM (Portable Arbitrary Map): P7, bilevel, gray, and rgb
- PGX (Portable Graymap Signed): PG, signed grayscale
- PFM (Portable Float Map): Pf (gray), PF (rgb), and PF4 (rgba), read-only
- XV thumbnail: P7 332 (rgb332), read-only

The Netpbm formats are specified at https://netpbm.sourceforge.net/doc/.

The PGX format is specified in ITU-T Rec. T.803.

No gamma correction or scaling is performed.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2024.4.24

Quickstart
----------

Install the netpbmfile package and all dependencies from the
`Python Package Index <https://pypi.org/project/netpbmfile/>`_::

    python -m pip install -U netpbmfile[all]

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/netpbmfile>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.9, 3.12.3
- `NumPy <https://pypi.org/project/numpy/>`_ 1.26.4

Revisions
---------

2024.4.24

- Support NumPy 2.

2023.8.30

- Fix linting issues.
- Add py.typed marker.

2023.6.15

- Drop support for Python 3.8 and numpy < 1.21 (NEP29).
- Improve type hints.

2023.1.1

- Several breaking changes:
- Rename magicnum to magicnumber (breaking).
- Rename tupltypes to tupltype (breaking).
- Change magicnumber and header properties to str (breaking).
- Replace pam parameter with magicnumber (breaking).
- Move byteorder parameter from NetpbmFile.asarray to NetpbmFile (breaking).
- Fix shape and axes properties for multi-image files.
- Add maxval and tupltype parameters to NetpbmFile.fromdata and imwrite.
- Add option to write comment to PNM and PAM files.
- Support writing PGX and text formats.
- Add Google style docstrings.
- Add unittests.

2022.10.25

- Read multi-image files.
- Fix reading ASCII formats with trailing comments.
- Fix writing maxval=1, depth=1 binary images.
- Use tifffile.imshow for multi-image arrays if installed.
- Change tupltypes to bytes according to specification (breaking).

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

Write a numpy array to a Netpbm file in grayscale binary format:

>>> data = numpy.array([[0, 1], [65534, 65535]], dtype=numpy.uint16)
>>> imwrite('_tmp.pgm', data)

Read the image data from a Netpbm file as numpy array:

>>> image = imread('_tmp.pgm')
>>> numpy.testing.assert_equal(image, data)

Access meta and image data in a Netpbm file:

>>> with NetpbmFile('_tmp.pgm') as pgm:
...     pgm.magicnumber
...     pgm.axes
...     pgm.shape
...     pgm.dtype
...     pgm.maxval
...     pgm.asarray().tolist()
'P5'
'YX'
(2, 2)
dtype('>u2')
65535
[[0, 1], [65534, 65535]]

View the image and metadata in the Netpbm file from the command line::

    $ python -m netpbmfile _tmp.pgm
