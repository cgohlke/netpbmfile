# -*- coding: utf-8 -*-
# netpbmfile.py

# Copyright (c) 2011-2019, Christoph Gohlke
# Copyright (c) 2011-2019, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
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

:Version: 2019.1.1

Requirements
------------
* `CPython 2.7 or 3.5+ <https://www.python.org>`_
* `Numpy 1.13 <https://www.numpy.org>`_
* `Matplotlib 2.2 <https://www.matplotlib.org>`_ (optional for plotting)

Revisions
---------
2019.1.1
    Update copyright year.
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

"""

from __future__ import division, print_function

__version__ = '2019.1.1'
__docformat__ = 'restructuredtext en'
__all__ = 'imread', 'imsave', 'NetpbmFile'

import sys
import re
import math
import warnings
from copy import deepcopy

import numpy


def imread(filename, copy=True, cache=False, byteorder=None):
    """Return image data from Netpbm file as numpy array.

    `args` and `kwargs` are arguments to NetpbmFile.asarray().

    Examples
    --------
    >>> image = imread('_tmp.pgm')

    """
    with NetpbmFile(filename) as netpbm:
        image = netpbm.asarray(copy=copy, cache=cache, byteorder=byteorder)
    return image


def imsave(filename, data, maxval=None, pam=False):
    """Write image data to Netpbm file.

    Examples
    --------
    >>> image = numpy.array([[0, 1], [65534, 65535]], dtype='uint16')
    >>> imsave('_tmp.pgm', image)

    """
    NetpbmFile.fromdata(data, maxval=maxval).write(filename, pam=pam)


class NetpbmFile(object):
    """Read and write Netpbm PAM, PBM, PGM, PPM, files."""

    _types = {b'P1': b'BLACKANDWHITE', b'P2': b'GRAYSCALE', b'P3': b'RGB',
              b'P4': b'BLACKANDWHITE', b'P5': b'GRAYSCALE', b'P6': b'RGB',
              b'P7 332': b'RGB', b'P7': b'RGB_ALPHA', b'PF': None, b'Pf': None}

    def __init__(self, filename):
        """Initialize instance from filename or open file."""
        for attr in ('header', 'magicnum', 'width', 'height', 'maxval',
                     'depth', 'dtype', 'tupltypes', 'byteorder', '_filename',
                     '_fh', '_data'):
            setattr(self, attr, None)
        if filename is None:
            return
        if hasattr(filename, 'seek'):
            self._fh = filename
        else:
            self._fh = open(filename, 'rb')
            self._filename = filename

        self._fh.seek(0)
        data = self._fh.read(4096)
        if len(data) < 7 or data[:2] not in self._types:
            raise ValueError('Not a Netpbm file:\n%s' % data[:32])
        if data[:2] in (b'PF', b'Pf'):
            self._read_pf_header(data)
        else:
            try:
                self._read_pam_header(data)
            except Exception:
                try:
                    self._read_pnm_header(data)
                except Exception:
                    raise ValueError('Not a Netpbm file:\n%s' % data[:32])

    @classmethod
    def fromdata(cls, data, maxval=None):
        """Initialize instance from numpy array."""
        data = numpy.array(data, ndmin=2, copy=True)
        if data.dtype.kind not in 'uib':
            raise ValueError('not an integer type: %s' % data.dtype)
        if data.dtype.kind == 'i' and numpy.min(data) < 0:
            raise ValueError('data out of range: %i' % numpy.min(data))
        if maxval is None:
            maxval = numpy.max(data)
            maxval = 255 if maxval < 256 else 65535
        if maxval < 0 or maxval > 65535:
            raise ValueError('data out of range: %i' % maxval)
        data = data.astype('u1' if maxval < 256 else '>u2')

        self = cls(None)
        self._data = data
        if data.ndim > 2 and data.shape[-1] in (3, 4):
            self.depth = data.shape[-1]
            self.width = data.shape[-2]
            self.height = data.shape[-3]
            self.magicnum = b'P7' if self.depth == 4 else b'P6'
        else:
            self.depth = 1
            self.width = data.shape[-1]
            self.height = data.shape[-2]
            self.magicnum = b'P5' if maxval > 1 else b'P4'
        self.maxval = maxval
        self.tupltypes = [self._types[self.magicnum]]
        self.header = self._header()
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
        if self._filename and self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __str__(self):
        """Return information about instance."""
        return unicode(self.header)

    def _read_pam_header(self, data):
        """Read PAM header and initialize instance."""
        regroups = re.search(
            br'(^P7[\n\r]+(?:(?:[\n\r]+)|(?:#.*)|'
            br'(HEIGHT\s+\d+)|(WIDTH\s+\d+)|(DEPTH\s+\d+)|(MAXVAL\s+\d+)|'
            br'(?:TUPLTYPE\s+\w+))*ENDHDR\n)', data).groups()
        self.header = regroups[0]
        self.magicnum = b'P7'
        for group in regroups[1:]:
            key, value = group.split()
            setattr(self, unicode(key).lower(), int(value))
        matches = re.findall(br'(TUPLTYPE\s+\w+)', self.header)
        self.tupltypes = [s.split(None, 1)[1] for s in matches]
        self.byteorder = '>'

    def _read_pnm_header(self, data):
        """Read PNM header and initialize instance."""
        bpm = data[1:2] in b'14'
        regroups = re.search(b''.join((
            br'(^(P[123456]|P7 332)\s+(?:#.*[\r\n])*',
            br'\s*(\d+)\s+(?:#.*[\r\n])*',
            br'\s*(\d+)\s+(?:#.*[\r\n])*' * (not bpm),
            br'\s*(\d+)\s(?:\s*#.*[\r\n]\s)*)')), data).groups() + (1, ) * bpm
        self.header = regroups[0]
        self.magicnum = regroups[1]
        self.width = int(regroups[2])
        self.height = int(regroups[3])
        self.maxval = int(regroups[4])
        self.depth = 3 if self.magicnum in b'P3P6P7 332' else 1
        self.tupltypes = [self._types[self.magicnum]]
        self.byteorder = '>'

    def _read_pf_header(self, data):
        """Read PF header and initialize instance."""
        regroups = re.search(
            br'(^(P[Ff])\s+(?:#.*[\r\n])*'
            br'\s*(\d+)\s+(?:#.*[\r\n])*'
            br'\s*(\d+)\s+(?:#.*[\r\n])*'
            br'\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s+(?:#.*[\r\n])*)',
            data).groups()
        self.header = regroups[0]
        self.magicnum = regroups[1]
        self.width = int(regroups[2])
        self.height = int(regroups[3])
        self.scale = abs(float(regroups[4]))
        self.byteorder = '<' if float(regroups[4]) < 0 else '>'
        self.depth = 3 if self.magicnum == b'PF' else 1
        self.tupltypes = None

    def _read_data(self, fh, byteorder=None):
        """Return image data from open file as numpy array."""
        fh.seek(len(self.header))
        data = fh.read()
        if byteorder is None:
            byteorder = self.byteorder
        if self.magicnum in (b'PF', b'Pf'):
            dtype = byteorder + 'f4'
        elif self.maxval < 256:
            dtype = 'u1'
        else:
            dtype = byteorder + 'u2'
        depth = 1 if self.magicnum == b'P7 332' else self.depth
        shape = [-1, self.height, self.width, depth]
        size = numpy.prod(shape[1:], dtype='int64')
        if self.magicnum in (b'PF', b'Pf'):
            size *= numpy.dtype(dtype).itemsize
            data = numpy.frombuffer(data[:size], dtype).reshape(shape)
        elif self.magicnum in b'P1P2P3':
            data = numpy.array(data.split(None, size)[:size], dtype)
            data = data.reshape(shape)
        elif self.maxval == 1:
            shape[2] = int(math.ceil(self.width / 8))
            data = numpy.frombuffer(data, dtype).reshape(shape)
            data = numpy.unpackbits(data, axis=-2)[:, :, :self.width, :]
        else:
            size *= numpy.dtype(dtype).itemsize
            data = numpy.frombuffer(data[:size], dtype).reshape(shape)
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
            header = '\n'.join((
                'P7',
                'HEIGHT %i' % self.height,
                'WIDTH %i' % self.width,
                'DEPTH %i' % self.depth,
                'MAXVAL %i' % self.maxval,
                '\n'.join('TUPLTYPE %s' % unicode(i) for i in self.tupltypes),
                'ENDHDR\n'))
        elif self.maxval == 1:
            header = 'P4 %i %i\n' % (self.width, self.height)
        elif self.depth == 1:
            header = 'P5 %i %i %i\n' % (self.width, self.height, self.maxval)
        else:
            header = 'P6 %i %i %i\n' % (self.width, self.height, self.maxval)
        if sys.version_info[0] > 2:
            header = bytes(header, 'ascii')
        return header


def main(argv=None):
    """Command line usage main function.

    Show images specified on command line or all images in current directory.

    """
    from glob import glob
    from matplotlib import pyplot

    if argv is None:
        argv = sys.argv

    if len(argv) > 1 and 'doctest' in argv:
        import doctest
        doctest.testmod()
        return

    files = argv[1:] if len(argv) > 1 else glob('*.p*m')
    for fname in files:
        try:
            with NetpbmFile(fname) as pam:
                img = pam.asarray(copy=False)
                if False:  # enable for testing
                    pam.write('_tmp.pgm.out', pam=True)
                    img2 = imread('_tmp.pgm.out')
                    assert numpy.all(img == img2)
                    imsave('_tmp.pgm.out', img)
                    img2 = imread('_tmp.pgm.out')
                    assert numpy.all(img == img2)
        except ValueError as e:
            # raise  # enable for debugging
            print(fname, e)
            continue

        cmap = 'gray' if (pam.maxval is None or pam.maxval > 1) else 'binary'
        shape = img.shape
        if img.ndim > 3 or (img.ndim > 2 and img.shape[-1] not in (3, 4)):
            warnings.warn('displaying first image only')
            img = img[0]
        if img.shape[-1] in (3, 4) and pam.maxval != 255:
            warnings.warn('converting RGB image for display')

            maxval = numpy.max(img) if pam.maxval is None else pam.maxval
            img = img / float(maxval)
            img *= 255
            numpy.rint(img, out=img)
            numpy.clip(img, 0, 255, out=img)
            img = img.astype('uint8')

        pyplot.imshow(img, cmap, interpolation='nearest')
        pyplot.title('"%s %s %s %s' % (fname, unicode(pam.magicnum),
                                       shape, img.dtype))
        pyplot.show()


if sys.version_info[0] > 2:
    basestring = str

    def unicode(x):
        return str(x, 'ascii')

if __name__ == '__main__':
    sys.exit(main())
