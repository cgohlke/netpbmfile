# netpbmfile.py

# Copyright (c) 2011-2020, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read and write Netpbm files.

Netpbmfile is a Python library to read and write image files in the Netpbm
format as specified at http://netpbm.sourceforge.net/doc/.

The following Netpbm and Portable FloatMap formats are supported:

* PBM (bi-level)
* PGM (grayscale)
* PPM (color)
* PAM (arbitrary)
* XV thumbnail (RGB332, read-only)
* PF (float32 RGB, read-only)
* Pf (float32 grayscale, read-only)

No gamma correction is performed. Only one image per file is supported.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:License: BSD 3-Clause

:Version: 2020.10.18

Requirements
------------
* `CPython >= 3.7 <https://www.python.org>`_
* `Numpy 1.15 <https://www.numpy.org>`_
* `Matplotlib 3.2 <https://www.matplotlib.org>`_ (optional for plotting)

Revisions
---------
2020.10.18
    Disallow comments after last value in PNM headers.
2020.9.18
    Remove support for Python 3.6 (NEP 29).
    Support os.PathLike file names.
2020.1.1
    Fix reading tightly packed P1 format and ASCII data with inline comments.
    Remove support for Python 2.7 and 3.5.
    Update copyright.
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

"""

__version__ = '2020.10.18'

__all__ = ('imread', 'imwrite', 'imsave', 'NetpbmFile')

import sys
import os
import re
import math
import warnings
from copy import deepcopy

import numpy


def imread(filename, byteorder=None):
    """Return image data from Netpbm file as numpy array."""
    with NetpbmFile(filename) as netpbm:
        image = netpbm.asarray(byteorder=byteorder)
    return image


def imwrite(filename, data, maxval=None, pam=False):
    """Write image data to Netpbm file."""
    NetpbmFile.fromdata(data, maxval=maxval).write(filename, pam=pam)


imsave = imwrite


class NetpbmFile:
    """Read and write Netpbm PAM, PBM, PGM, PPM, and PF files."""

    MAGIC_NUMBER = {
        b'P1': b'BLACKANDWHITE',
        b'P2': b'GRAYSCALE',
        b'P3': b'RGB',
        b'P4': b'BLACKANDWHITE',
        b'P5': b'GRAYSCALE',
        b'P6': b'RGB',
        b'P7 332': b'RGB',
        b'P7': b'RGB_ALPHA',
        b'PF': b'RGB_FLOAT',
        b'Pf': b'GRAYSCALE_FLOAT',
    }

    def __init__(self, filename):
        """Initialize instance from filename or open file."""
        self.header = b''
        self.width = 0
        self.height = 0
        self.depth = 0
        self.maxval = 0
        self.tupltypes = []
        self.byteorder = '>'
        self.filename = ''
        self._data = None

        if filename is None:
            return
        if hasattr(filename, 'seek'):
            self._fh = filename
        else:
            self._fh = open(filename, 'rb')
            self.filename = os.fspath(filename)

        self._fh.seek(0)
        data = self._fh.read(4096)
        if len(data) < 7 or data[:2] not in NetpbmFile.MAGIC_NUMBER:
            raise ValueError(f'Not a Netpbm file:\n{data[:32]}')
        if data[:2] in b'PFPf':
            self._read_pf_header(data)
        else:
            try:
                self._read_pam_header(data)
            except Exception:
                try:
                    self._read_pnm_header(data)
                except Exception as exc:
                    raise ValueError(
                        f'Not a Netpbm file:\n{data[:32]}'
                    ) from exc

        if self.magicnum in b'PFPf':
            dtype = self.byteorder + 'f4'
        elif self.maxval < 256:
            dtype = 'u1'
        else:
            dtype = self.byteorder + 'u2'
        self.dtype = numpy.dtype(dtype)

        depth = 1 if self.magicnum == b'P7 332' else self.depth
        # TODO: multi-image shape
        if depth > 1:
            self.shape = (self.height, self.width, depth)
            self.axes = 'YXS'
        else:
            self.shape = (self.height, self.width)
            self.axes = 'YX'

    @classmethod
    def fromdata(cls, data, maxval=None):
        """Initialize instance from numpy array."""
        data = numpy.array(data, ndmin=2, copy=True)
        if data.dtype.kind not in 'uib':
            raise ValueError(f'not an integer type: {data.dtype}')
        if data.dtype.kind == 'i' and numpy.min(data) < 0:
            raise ValueError(f'data out of range: {numpy.min(data)}')
        if maxval is None:
            maxval = numpy.max(data)
            maxval = 255 if maxval < 256 else 65535
        if maxval < 0 or maxval > 65535:
            raise ValueError(f'data out of range: {maxval}')
        data = data.astype('u1' if maxval < 256 else '>u2')

        self = cls(None)
        self._data = data
        if data.ndim > 2 and data.shape[-1] in (3, 4):
            self.depth = data.shape[-1]
            self.width = data.shape[-2]
            self.height = data.shape[-3]
            self.shape = (self.height, self.width, self.depth)
            self.axes = 'YXS'
            self.magicnum = b'P7' if self.depth == 4 else b'P6'
        else:
            self.depth = 1
            self.width = data.shape[-1]
            self.height = data.shape[-2]
            self.shape = (self.height, self.width)
            self.axes = 'YX'
            self.magicnum = b'P5' if maxval > 1 else b'P4'
        self.maxval = maxval
        self.tupltypes = [NetpbmFile.MAGIC_NUMBER[self.magicnum]]
        self.header = self._header()
        self.dtype = data.dtype
        self._fh = None
        return self

    def asarray(self, copy=True, cache=False, byteorder=None):
        """Return image data from file as numpy array."""
        data = self._data
        if data is None:
            data = self._read_data(self._fh, byteorder=byteorder)
            if cache:
                self._data = data
            else:
                return data
        return deepcopy(data) if copy else data

    def write(self, filename, pam=False):
        """Write instance to file."""
        if hasattr(filename, 'seek'):
            self._tofile(filename, pam=pam)
        else:
            with open(filename, 'wb') as fh:
                self._tofile(fh, pam=pam)

    def close(self):
        """Close open file."""
        if self.filename and self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __str__(self):
        """Return information about Netpbm file."""
        magicnum = NetpbmFile.MAGIC_NUMBER[self.magicnum].decode('ascii')
        return '\n '.join(
            (
                self.__class__.__name__,
                os.path.normpath(os.path.normcase(self.filename)),
                f'type: {magicnum}',
                f'axes: {self.axes}',
                'shape: {}'.format(', '.join(str(i) for i in self.shape)),
                f'dtype: {self.dtype}',
            )
        )

    def _read_pam_header(self, data):
        """Read PAM header and initialize instance."""
        regroups = re.search(
            br'(^P7[\n\r]+(?:(?:[\n\r]+)|(?:#.*)|'
            br'(HEIGHT\s+\d+)|(WIDTH\s+\d+)|(DEPTH\s+\d+)|(MAXVAL\s+\d+)|'
            br'(?:TUPLTYPE\s+\w+))*ENDHDR\n)',
            data,
        ).groups()
        self.header = regroups[0]
        self.magicnum = b'P7'
        for group in regroups[1:]:
            key, value = group.split()
            setattr(self, key.decode('ascii').lower(), int(value))
        matches = re.findall(br'(TUPLTYPE\s+\w+)', self.header)
        self.tupltypes = [s.split(None, 1)[1] for s in matches]

    def _read_pnm_header(self, data):
        """Read PNM header and initialize instance."""
        bpm = data[1:2] in b'14'
        regroups = re.search(
            b''.join(
                (
                    br'(^(P[123456]|P7 332)\s+(?:#.*[\r\n])*',
                    br'\s*(\d+)\s+(?:#.*[\r\n])*',
                    br'\s*(\d+)\s+(?:#.*[\r\n])*' * (not bpm),
                    # in disagreement with the netpbm doc pages, the netpbm
                    # man pages only allow a single whitespace character after
                    # the last value
                    br'\s*(\d+)\s)',
                )
            ),
            data,
        ).groups()
        regroups = regroups + (1,) * bpm
        self.header = regroups[0]
        self.magicnum = regroups[1]
        self.width = int(regroups[2])
        self.height = int(regroups[3])
        self.maxval = int(regroups[4])
        self.depth = 3 if self.magicnum in b'P3P6P7 332' else 1
        self.tupltypes = [NetpbmFile.MAGIC_NUMBER[self.magicnum]]

    def _read_pf_header(self, data):
        """Read PF header and initialize instance."""
        # there are no comments in these files
        regroups = re.search(
            br'(^(P[Ff])\s+(?:#.*[\r\n])*'
            br'\s*(\d+)\s+(?:#.*[\r\n])*'
            br'\s*(\d+)\s+(?:#.*[\r\n])*'
            br'\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
            br'\s*[\n])',
            data,
        ).groups()
        self.header = regroups[0]
        self.magicnum = regroups[1]
        self.width = int(regroups[2])
        self.height = int(regroups[3])
        self.scale = abs(float(regroups[4]))
        self.byteorder = '<' if float(regroups[4]) < 0 else '>'
        self.depth = 3 if self.magicnum == b'PF' else 1

    def _read_data(self, fh, byteorder=None):
        """Return image data from open file as numpy array."""
        fh.seek(len(self.header))

        if byteorder is None:
            byteorder = self.byteorder
        if self.magicnum in b'PFPf':
            dtype = byteorder + 'f4'
        elif self.maxval < 256:
            dtype = 'u1'
        else:
            dtype = byteorder + 'u2'
        dtype = numpy.dtype(dtype)
        depth = 1 if self.magicnum == b'P7 332' else self.depth
        shape = [-1, self.height, self.width, depth]

        if self.magicnum in b'P1P2P3':
            data = fh.read()
            if self.magicnum == b'P1' and data[1] != b' ':
                data = [bytes([i]) for i in data if i == 48 or i == 49]
            else:
                data = [i for i in data.split() if i.isdigit()]
            size = numpy.prod(shape[1:], dtype='int64')
            data = numpy.array(data[:size], dtype)
            data = data.reshape(shape)
        else:
            bilevel = self.maxval == 1 and self.magicnum not in b'PFPf'
            if bilevel:
                shape[2] = int(math.ceil(self.width / 8))
            size = numpy.prod(shape[1:], dtype='int64') * dtype.itemsize
            data = fh.read(size)
            data = numpy.frombuffer(data, dtype).reshape(shape)
            if bilevel:
                data = numpy.unpackbits(data, axis=-2)[:, :, : self.width, :]

        if data.shape[0] < 2:
            data = data.reshape(data.shape[1:])
        if data.shape[-1] < 2:
            data = data.reshape(data.shape[:-1])
        if self.magicnum == b'P7 332':
            rgb332 = numpy.array(list(numpy.ndindex(8, 8, 4)), numpy.uint8)
            rgb332 *= numpy.array([36, 36, 85], numpy.uint8)
            data = numpy.take(rgb332, data, axis=0)
        return data

    def _tofile(self, fh, pam=False):
        """Write Netpbm file."""
        fh.seek(0)
        fh.write(self._header(pam))
        data = self.asarray(copy=False)
        if self.maxval == 1:
            data = numpy.packbits(data, axis=-1)
        data.tofile(fh)

    def _header(self, pam=False):
        """Return file header as byte string."""
        if pam or self.magicnum == b'P7':
            header = '\n'.join(
                (
                    'P7',
                    f'HEIGHT {self.height}',
                    f'WIDTH {self.width}',
                    f'DEPTH {self.depth}',
                    f'MAXVAL {self.maxval}',
                    '\n'.join(
                        f"TUPLTYPE {i.decode('ascii')}" for i in self.tupltypes
                    ),
                    'ENDHDR\n',
                )
            )
        elif self.maxval == 1:
            header = f'P4 {self.width} {self.height}\n'
        elif self.depth == 1:
            header = f'P5 {self.width} {self.height} {self.maxval}\n'
        else:
            header = f'P6 {self.width} {self.height} {self.maxval}\n'
        return header.encode('ascii')


def main(argv=None, test=False):
    """Command line usage main function.

    Show images specified on command line or all images in  directory.

    """
    import os
    from glob import glob
    from matplotlib import pyplot

    if argv is None:
        argv = sys.argv

    if len(argv) > 1 and '--doctest' in argv:
        import doctest

        doctest.testmod()
        return

    if len(argv) == 1:
        files = glob('*.p*m')
    elif '*' in argv[1]:
        files = glob(argv[1])
    elif os.path.isdir(argv[1]):
        files = glob(f'{argv[1]}/*.p*m')
    else:
        files = argv[1:]

    for fname in files:
        try:
            with NetpbmFile(fname) as pam:
                print(pam)
                img = pam.asarray(copy=False)
                if test:  # enable for testing
                    pam.write('_tmp.pgm.out', pam=True)
                    img2 = imread('_tmp.pgm.out')
                    assert numpy.all(img == img2)
                    imsave('_tmp.pgm.out', img)
                    img2 = imread('_tmp.pgm.out')
                    assert numpy.all(img == img2)
                print()
        except ValueError as exc:
            # raise  # enable for debugging
            print(fname, exc)
            continue

        cmap = 'gray' if (pam.maxval is None or pam.maxval > 1) else 'binary'
        dtype = img.dtype
        shape = img.shape
        if img.ndim > 3 or (img.ndim > 2 and img.shape[-1] not in (3, 4)):
            warnings.warn('displaying first image only')
            img = img[0]
        if img.shape[-1] in (3, 4) and pam.maxval != 255:
            warnings.warn('converting RGB image for display')
            maxval = numpy.max(img) if pam.maxval is None else pam.maxval
            maxval = float(maxval)
            if maxval > 0.0:
                img = img / maxval
            else:
                img = img.copy()
            img *= 255
            numpy.rint(img, out=img)
            numpy.clip(img, 0, 255, out=img)
            img = img.astype('uint8')

        pyplot.imshow(img, cmap, interpolation='nearest')
        pyplot.title(
            f"{os.path.split(fname)[-1]} "
            f"{pam.magicnum.decode('ascii')} {shape} "
            f"{dtype}"
        )
        pyplot.show()


if __name__ == '__main__':
    sys.exit(main())
