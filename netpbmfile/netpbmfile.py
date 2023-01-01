# netpbmfile.py

# Copyright (c) 2011-2023, Christoph Gohlke
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
or related formats:

- PBM (Portable Bit Map): P1 (text) and P4 (binary)
- PGM (Portable Gray Map): P2 (text) and P5 (binary)
- PPM (Portable Pixel Map): P3 (text) and P6 (binary)
- PNM (Portable Any Map): shorthand for PBM, PGM, and PPM collectively
- PAM (Portable Arbitrary Map): P7, bilevel, gray, and rgb
- PGX (Portable Graymap Signed): PG, signed grayscale
- PFM (Portable Float Map): Pf (gray), PF (rgb), and PF4 (rgba), read-only
- XV thumbnail: P7 332 (rgb332), read-only

The Netpbm formats are specified at http://netpbm.sourceforge.net/doc/.

The PGX format is specified in ITU-T Rec. T.803.

No gamma correction or scaling is performed.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2023.1.1

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

This release has been tested with the following requirements and dependencies
(other versions may work):

- `CPython 3.8.10, 3.9.13, 3.10.9, 3.11.1 <https://www.python.org>`_
- `NumPy 1.23.5 <https://pypi.org/project/numpy/>`_

Revisions
---------

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

"""

from __future__ import annotations

__version__ = '2023.1.1'

__all__ = ['imread', 'imwrite', 'imsave', 'NetpbmFile']

import sys
import os
import re
import math
import warnings

import numpy


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import BinaryIO, Iterable, Literal, Union

    try:
        from numpy.typing import ArrayLike
    except ImportError:
        # numpy < 1.20
        from numpy import ndarray as ArrayLike

    PathLike = Union[str, os.PathLike]
    ByteOrder = Union[Literal['>'], Literal['<']]
    MagicNumber = Union[
        Literal['P1'],
        Literal['P2'],
        Literal['P3'],
        Literal['P4'],
        Literal['P5'],
        Literal['P6'],
        Literal['P7'],
        Literal['PG'],
        Literal['Pf'],
        Literal['PF'],
        Literal['PF4'],
        Literal['P7 332'],
    ]


def imread(
    file: PathLike | BinaryIO, /, *, byteorder: ByteOrder | None = None
) -> numpy.ndarray:
    """Return image data from Netpbm file.

    Parameters:
        file:
            Name of file or open binary file to read.
        byteorder:
            Byte order of image data in file.
            By default, all formats are big-endian except for PFM, which
            encodes the byte order in the file header.

    """
    with NetpbmFile(file, byteorder=byteorder) as netpbm:
        image = netpbm.asarray()
    return image


def imwrite(
    file: PathLike | BinaryIO,
    data: ArrayLike,
    /,
    *,
    magicnumber: MagicNumber | None = None,
    maxval: int | None = None,
    tupltype: str | None = None,
    byteorder: ByteOrder | None = None,
    comment: str | None = None,
) -> None:
    """Write image data to Netpbm file.

    Parameters:
        file:
            Name of file or open binary file to write.
        data:
            Image data to write.
        magicnumber:
            Netpbm format to write.
            By default, this is determined from the data shape, dtype,
            and maxval.
            Writing PFM and XV formats is not supported.
        maxval:
            Maximum value of image samples.
            By default, this is determined from the data.
        tupltype:
            Kind of PAM image.
            By default, this is determined from the magicnumber and data
            shape and dtype.
        byteorder:
            Byte order for 16-bit image data in P5 and P6 formats.
            By default, the byte order is '>'.
            Other formats are written in the byte order of the data array.
        comment:
            Single line ASCII string to write to PNM and PAM formats.
            Maximum 66 characters.

    """
    with NetpbmFile.fromdata(
        data, maxval=maxval, magicnumber=magicnumber, tupltype=tupltype
    ) as netpbm:
        netpbm.write(
            file, magicnumber=magicnumber, byteorder=byteorder, comment=comment
        )


imsave = imwrite


class NetpbmFile:
    """Read and write Netpbm and related files.

    Parameters:
        file:
            Name of file or open binary file to read.
        byteorder:
            Byte order of image data in file.
            By default, all formats are big endian except for PFM, which
            encodes the byte order in the file header.

    """

    header: str
    """Netpbm header starting with magicnumber."""

    magicnumber: MagicNumber
    """ID determining Netpbm type."""

    frames: int
    """Number of frames in image."""

    height: int
    """Number of rows in image."""

    width: int
    """Number of columns in image."""

    depth: int
    """Number of samples in image."""

    maxval: int
    """Maximum value of image samples."""

    scale: float
    """Factor to scale image values in PFM formats."""

    byteorder: ByteOrder
    """Byte order of binary image data."""

    filename: str
    """File name."""

    tupltype: str
    """Kind of PAM image."""

    dtype: numpy.dtype
    """Data type of image array."""

    dataoffset: int
    """Position of image data in file."""

    _data: numpy.ndarray | None
    _fh: BinaryIO | None

    MAGIC_NUMBER: dict[str, str] = {
        'P1': 'BLACKANDWHITE',
        'P2': 'GRAYSCALE',
        'P3': 'RGB',
        'P4': 'BLACKANDWHITE',
        'P5': 'GRAYSCALE',
        'P6': 'RGB',
        'P7 332': 'RGB',
        'P7': 'ARBITRARY',
        'Pf': 'GRAYSCALE_FLOAT',
        'PF': 'RGB_FLOAT',
        'PF4': 'RGB_ALPHA_FLOAT',
        'PG': 'GRAYSCALE',
    }

    def __init__(
        self,
        file: PathLike | BinaryIO | None,
        /,
        *,
        byteorder: ByteOrder | None = None,
    ) -> None:
        # initialize instance from filename or open file
        self.header = ''
        self.magicnumber = 'P7'
        self.frames = 1
        self.width = 0
        self.height = 0
        self.depth = 0
        self.maxval = 0
        self.scale = 0.0
        self.tupltype = ''
        self.byteorder = '>'
        self.filename = ''
        self.dtype = numpy.dtype('u1')
        self.dataoffset = 0
        self._data = None
        self._fh = None

        if file is None:
            return

        if isinstance(file, (str, os.PathLike)):
            self._fh = open(file, 'rb')
            self.filename = os.fspath(file)
        else:
            self._fh = file

        try:
            self._fh.seek(0)
            data = self._fh.read(4096)
            if (
                len(data) < 7
                or not data[:2].isascii()
                or data[:2].decode('ascii') not in NetpbmFile.MAGIC_NUMBER
            ):
                raise ValueError(f'not a Netpbm file:\n  {data[:16]!r}')
            if data[:2] in b'PFPf':
                self._read_pf_header(data)
            elif data[:2] == b'PG':
                self._read_pg_header(data)
            else:
                try:
                    self._read_pam_header(data)
                except Exception:
                    try:
                        self._read_pnm_header(data)
                    except Exception as exc:
                        raise ValueError(
                            f'not a Netpbm file:\n{data[:16]!r}'
                        ) from exc
        except Exception:
            self._fh.close()
            raise

        if byteorder is not None:
            self.byteorder = byteorder

        if self.magicnumber in 'P1 P4':
            dtype = 'bool_'
        elif self.magicnumber in 'PF4 Pf':
            dtype = self.byteorder + 'f4'
        elif self.magicnumber == 'PG':
            dtype = self.byteorder + self.dtype.char
        elif self.maxval < 256:
            dtype = 'u1'
        elif self.maxval < 65536:
            dtype = self.byteorder + 'u2'
        elif self.maxval < 2**32:
            dtype = self.byteorder + 'u4'
        else:
            raise ValueError(f'maxval {self.maxval} out of range')

        self.dtype = numpy.dtype(dtype)

        if self.magicnumber in 'P1 P2 P3':
            self.frames = 1
        else:
            bytecount = self._fh.seek(0, 2) - len(self.header)
            shape = [
                self.height,
                int(math.ceil(self.width / 8))
                if self.magicnumber == 'P4'
                else self.width,
                self.depth,
                self.dtype.itemsize,
            ]
            self.frames = max(1, bytecount // product(shape))

    @classmethod
    def fromdata(
        cls,
        data: ArrayLike,
        /,
        *,
        magicnumber: MagicNumber | None = None,
        maxval: int | None = None,
        tupltype: str | None = None,
    ) -> NetpbmFile:
        """Initialize instance from numpy array.

        Parameters:
            data:
                Image data.
            magicnumber:
                ID determining Netpbm type.
                By default, this is determined from the data shape, dtype,
                and maxval.
            maxval:
                Maximum value of image samples.
                By default, this is determined from the data.
            tupltype:
                Kind of PAM image.
                By default, this is determined from the magicnumber and data
                shape and dtype.

        """
        data = numpy.array(data, ndmin=2, copy=True)
        if data.dtype.kind not in 'uib':
            # TODO: support PF, Pf, PF4
            raise ValueError(f'dtype {data.dtype!r} not supported')

        issigned = data.dtype.kind == 'i' and numpy.min(data) < 0
        if issigned:
            if magicnumber is None:
                magicnumber = 'PG'
            elif magicnumber != 'PG':
                raise ValueError(
                    f'invalid dtype {data.dtype!r} for '
                    f'magicnumber {magicnumber!r}'
                )

        if maxval is None:
            if issigned:
                maxval = numpy.max(numpy.abs(data))
            else:
                maxval = numpy.max(data)
            if maxval == 1:
                maxval = 1
            else:
                maxval = max(
                    255, int(2 ** math.ceil(math.log2(maxval + 1)) - 1)
                )
        if not 0 < maxval < 2**32:
            # allow maxval > 65535
            raise ValueError(f'maxval {maxval} of range')

        self = cls(None)

        if magicnumber is None:
            if data.ndim > 2 and data.shape[-1] in (3, 4):
                # rgba
                self.depth = data.shape[-1]
                self.width = data.shape[-2]
                self.height = data.shape[-3]
                magicnumber = 'P6' if self.depth == 3 else 'P7'
            else:
                # bilevel or gray
                self.depth = 1
                self.width = data.shape[-1]
                self.height = data.shape[-2]
                magicnumber = 'P5' if maxval > 1 else 'P4'
        elif magicnumber == 'P7':
            if data.ndim > 2 and data.shape[-1] <= 4:
                # only allow up to 4 samples
                self.depth = data.shape[-1]
                self.width = data.shape[-2]
                self.height = data.shape[-3]
            else:
                self.depth = 1
                self.width = data.shape[-1]
                self.height = data.shape[-2]
        elif magicnumber in 'P3 P6':
            # rgb
            if data.ndim < 3 or data.shape[-1] != 3:
                raise ValueError(
                    f'invalid magicnumber {magicnumber!r} '
                    f'for shape {data.shape}'
                )
            self.depth = data.shape[-1]
            self.width = data.shape[-2]
            self.height = data.shape[-3]
        elif magicnumber in 'P1 P2 P4 P5 PG':
            # bilevel or gray
            if magicnumber in 'P1 P4' and maxval != 1:
                raise ValueError(
                    f'invalid magicnumber {magicnumber!r} for maxval {maxval}'
                )
            if magicnumber == 'PG':
                cls.byteorder = '<' if data.dtype.byteorder in '<|=' else '>'
            self.depth = 1
            self.width = data.shape[-1]
            self.height = data.shape[-2]
        else:
            raise ValueError(f'invalid magicnumber {magicnumber}')

        if magicnumber == 'PG' and data.dtype.kind == 'i':
            self._data = data.astype(
                'i1'
                if maxval < 128
                else (
                    cls.byteorder + 'i2'
                    if maxval < 32768
                    else cls.byteorder + 'i4'
                )
            )
        elif magicnumber in 'P1 P4':
            self._data = data.astype('bool')
        else:
            self._data = data.astype(
                'u1' if maxval < 256 else ('>u2' if maxval < 65536 else '>u4')
            )

        self.frames = max(
            1, product(data.shape) // (self.height * self.width * self.depth)
        )
        self.magicnumber = magicnumber
        self.maxval = maxval
        self.dtype = self._data.dtype
        self.header = self._header()

        if tupltype is not None:
            assert tupltype is not None
            self.tupltype = tupltype
        elif magicnumber != 'P7':
            self.tupltype = NetpbmFile.MAGIC_NUMBER[self.magicnumber]
        elif self.maxval == 1 and self.depth in (1, 2):
            self.tupltype = 'BLACKANDWHITE'
            if self.depth == 2:
                self.tupltype += '_ALPHA'
        else:
            self.tupltype = {
                1: 'GRAYSCALE',
                2: 'GRAYSCALE_ALPHA',
                3: 'RGB',
                4: 'RGB_ALPHA',
            }.get(self.depth, 'ARBITRARY')

        return self

    def asarray(
        self,
        *,
        copy: bool = True,
        cache: bool = False,
    ) -> numpy.ndarray:
        """Return image array.

        Parameters:
            copy:
                Return a copy of image array.
            cache:
                Keep a copy of image data after reading from file.

        """
        data = self._data
        if data is None:
            assert self._fh is not None
            data = self._read_data(self._fh)
            if cache:
                self._data = data
            else:
                return data
        return numpy.copy(data) if copy else data

    def write(
        self,
        file: PathLike | BinaryIO,
        /,
        *,
        magicnumber: MagicNumber | None = None,
        byteorder: ByteOrder | None = None,
        comment: str | None = None,
    ) -> None:
        """Write instance to file.

        Parameters:
            file:
                Name of file or open binary file to write.
            magicnumber:
                Netpbm format to write.
                By default, the instance's magicnumber is used.
                PFM and XV formats are not supported.
            byteorder:
                Byte order for 16-bit image data in P5 and P6 formats.
                By default, the byte order is '>'.
                Other formats are written in the byte order of the data array.
            comment:
                Single line ASCII string to write to PNM and PAM formats.
                Maximum 66 characters.

        """
        if isinstance(file, (str, os.PathLike)):
            with open(file, 'wb') as fh:
                self._tofile(
                    fh,
                    magicnumber=magicnumber,
                    byteorder=byteorder,
                    comment=comment,
                )
        else:
            assert hasattr(file, 'seek')
            self._tofile(
                file,
                magicnumber=magicnumber,
                byteorder=byteorder,
                comment=comment,
            )

    def close(self) -> None:
        """Close open file."""
        if self.filename and self._fh is not None:
            self._fh.close()
            self._fh = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of image array."""
        shape = [self.height, self.width]
        if self.depth > 1:
            shape += [self.depth]
        if self.frames > 1:
            shape = [self.frames] + shape
        return tuple(shape)

    @property
    def axes(self) -> str:
        """Axes of image array."""
        axes = 'YX'
        if self.depth > 1:
            axes += 'S'
        if self.frames > 1:
            axes = 'I' + axes
        return axes

    def _read_pam_header(self, data: bytes, /) -> None:
        """Read PAM header and initialize instance."""
        match = re.search(
            br'(^P7[\n\r]+(?:(?:[\n\r]+)|(?:#.*)|'
            br'(HEIGHT\s+\d+ *)|(WIDTH\s+\d+ *)|(DEPTH\s+\d+ *)|'
            br'(MAXVAL\s+\d+ *)|(?:TUPLTYPE\s+\w+))*ENDHDR[\n\r])',
            data,
        )
        if match is None:
            raise ValueError('invalid PAM header')
        regroups = match.groups()
        self.dataoffset = len(regroups[0])
        self.header = regroups[0].decode(errors='ignore')
        self.magicnumber = 'P7'
        for group in regroups[1:]:
            key, value = group.split()
            setattr(self, key.decode('ascii').lower(), int(value))
        matches = re.findall(r'(TUPLTYPE\s+\w+)', self.header)
        self.tupltype = ' '.join(s.split(None, 1)[1] for s in matches)

    def _read_pnm_header(self, data: bytes, /) -> None:
        """Read PNM header and initialize instance."""
        bpm = data[1:2] in b'14'
        match = re.search(
            b''.join(
                (
                    br'(^(P[123456]|P7 332)\s+(?:#.*[\r\n])*',
                    br'\s*(\d+)\s+(?:#.*[\r\n])*',
                    br'\s*(\d+)\s+(?:#.*[\r\n])*' * (not bpm),
                    # in disagreement with the netpbm doc pages, the netpbm
                    # man pages only allow a single whitespace character after
                    # the last value.
                    # See also https://stackoverflow.com/a/7369986/453463
                    br'\s*(\d+)\s)',
                )
            ),
            data,
        )
        if match is None:
            raise ValueError('invalid PNM header')
        regroups = match.groups()
        regroups = regroups + (1,) * bpm
        self.dataoffset = len(regroups[0])
        self.header = regroups[0].decode(errors='ignore')
        self.magicnumber = regroups[1].decode()  # type: ignore
        self.width = int(regroups[2])
        self.height = int(regroups[3])
        self.maxval = int(regroups[4])
        self.depth = 3 if self.magicnumber in 'P3 P6 P7 332' else 1
        self.tupltype = NetpbmFile.MAGIC_NUMBER[self.magicnumber]

    def _read_pf_header(self, data: bytes, /) -> None:
        """Read PF header and initialize instance."""
        # there are no comments in these files
        match = re.search(
            br'(^(PF|PF4|Pf)\s+(?:#.*[\r\n])*'
            br'\s*(\d+)\s+(?:#.*[\r\n])*'
            br'\s*(\d+)\s+(?:#.*[\r\n])*'
            br'\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)'
            br'\s*[\r\n])',
            data,
        )
        if match is None:
            raise ValueError('invalid PF header')
        regroups = match.groups()
        self.dataoffset = len(regroups[0])
        self.header = regroups[0].decode(errors='ignore')
        self.magicnumber = regroups[1].decode()  # type: ignore
        self.width = int(regroups[2])
        self.height = int(regroups[3])
        self.scale = abs(float(regroups[4]))
        self.byteorder = '<' if float(regroups[4]) < 0 else '>'
        if self.magicnumber == 'PF4':
            self.depth = 4
        elif self.magicnumber == 'PF':
            self.depth = 3
        elif self.magicnumber == 'Pf':
            self.depth = 1
        else:
            raise ValueError(f'invalid magicnumber {self.magicnumber!r}')

    def _read_pg_header(self, data: bytes, /) -> None:
        """Read PG header and initialize instance."""
        match = re.search(
            br'(^(PG)[ ]+'
            br'(LM|ML)?[ ]*'
            br'([-+])?[ ]*([0-9]+)[ ]+'
            br'([0-9]+)[ ]+'
            br'([0-9]+)[ ]*[\r?\n])',
            data,
        )
        if match is None:
            raise ValueError('invalid PG header')
        regroups = match.groups()
        self.dataoffset = len(regroups[0])
        self.header = regroups[0].decode(errors='ignore')
        self.magicnumber = regroups[1].decode()  # type: ignore
        self.byteorder = '>' if (regroups[2] == b'ML') else '<'
        signed = regroups[3] == b'-'
        bitdepth = int(regroups[4])
        self.width = int(regroups[5])
        self.height = int(regroups[6])
        self.depth = 1
        self.maxval = 2 ** (bitdepth - 1 if signed else bitdepth) - 1
        if bitdepth <= 8:
            self.dtype = numpy.dtype('i1' if signed else 'u1')
        elif bitdepth <= 16:
            self.dtype = numpy.dtype(
                self.byteorder + ('i2' if signed else 'u2')
            )
        elif bitdepth <= 32:
            self.dtype = numpy.dtype(
                self.byteorder + ('i4' if signed else 'u4')
            )
        else:
            raise ValueError(f'bitdepth {bitdepth} out of range')

    def _read_data(self, fh: BinaryIO) -> numpy.ndarray:
        """Return image data from open file."""
        fh.seek(self.dataoffset)

        bilevel = self.magicnumber in 'P1 P4'
        dtype = self.dtype if not bilevel else numpy.dtype('u1')
        depth = self.depth if self.magicnumber != 'P7 332' else 1
        shape = [-1, self.height, self.width, depth]
        rawdata = fh.read()

        if self.magicnumber in 'P1 P2 P3':
            if bilevel and rawdata.strip()[1:2] in b'01':
                datalist = [
                    bytes([i])
                    for line in rawdata.splitlines()
                    for i in line.split(b'#')[0]
                    if i == 48 or i == 49
                ]
            else:
                datalist = [
                    i
                    for line in rawdata.splitlines()
                    for i in line.split(b'#')[0].split()
                    if i.isdigit()
                ]
            size = product(shape[1:])
            size *= max(1, len(datalist) // size)
            data = numpy.array(datalist[:size], dtype).reshape(shape)
        else:
            if bilevel:
                shape[2] = int(math.ceil(self.width / 8))
            size = product(shape[1:]) * dtype.itemsize
            size *= max(1, len(rawdata) // size)
            data = numpy.frombuffer(rawdata[:size], dtype).reshape(shape)
            if bilevel:
                data = numpy.unpackbits(data, axis=-2)[:, :, : self.width, :]

        if bilevel:
            data = data.astype('bool')

        if data.shape[0] < 2:
            data = data.reshape(data.shape[1:])
        if data.shape[-1] < 2:
            data = data.reshape(data.shape[:-1])
        if self.magicnumber == 'P7 332':
            rgb332 = numpy.array(list(numpy.ndindex(8, 8, 4)), 'u1')
            rgb332 *= numpy.array([36, 36, 85], 'u1')
            data = numpy.take(rgb332, data, axis=0)
        return data

    def _tofile(
        self,
        fh: BinaryIO,
        /,
        *,
        magicnumber: MagicNumber | None = None,
        byteorder: ByteOrder | None = None,
        comment: str | None = None,
    ) -> None:
        """Write header and image data to file."""
        if magicnumber is None:
            magicnumber = self.magicnumber
        if magicnumber not in NetpbmFile.MAGIC_NUMBER:
            raise ValueError(f'invalid magicnumber {magicnumber!r}')

        fh.seek(0)
        fh.write(
            self._header(magicnumber=magicnumber, comment=comment).encode(
                'ascii'
            )
        )

        if self._data is None:
            data = self.asarray(copy=False)
        else:
            data = self._data

        # data type/shape verification done in fromdata() and _header()
        if magicnumber in 'P1':
            assert self.maxval == 1
            assert self.depth == 1
            assert data.dtype.kind == 'b'
            if self.frames > 1:
                log_warning('writing non-compliant multi-image file')
            # one line per sample
            numpy.savetxt(fh, data.reshape(-1), fmt='%i')
        elif magicnumber == 'P2':
            if self.maxval > 65535:
                log_warning('writing non-compliant maxval {self.maxval}')
            if self.frames > 1:
                log_warning('writing non-compliant multi-image file')
            assert self.depth == 1
            assert data.dtype.kind in 'iu'
            # one line per sample
            numpy.savetxt(fh, data.reshape(-1), fmt='%i')
        elif magicnumber == 'P3':
            if self.maxval > 65535:
                log_warning('writing non-compliant maxval {self.maxval}')
            if self.frames > 1:
                log_warning('writing non-compliant multi-image file')
            assert self.depth == 3
            assert data.dtype.kind in 'iu'
            # one line per sample
            numpy.savetxt(fh, data.reshape((-1, 3)), fmt='%i')
        elif magicnumber == 'P4':
            assert self.maxval == 1
            assert self.depth == 1
            assert data.dtype.kind == 'b'
            data = numpy.packbits(data, axis=-1)
            fh.write(data.tobytes())
        elif magicnumber == 'PG':
            assert data.dtype.kind in 'iu'
            fh.write(data.tobytes())
        else:
            # P5, P6, P7
            assert data.dtype.kind in 'iub'
            if data.itemsize > 1:
                if byteorder is None:
                    byteorder = '>'
                data = numpy.asarray(
                    data, dtype=data.dtype.newbyteorder(byteorder)
                )
            fh.write(data.tobytes())

    def _header(
        self,
        *,
        magicnumber: MagicNumber | None = None,
        comment: str | None = None,
    ) -> str:
        """Return file header."""
        if magicnumber is None:
            magicnumber = self.magicnumber
        if comment is None:
            comment = ''  # f'written by netpbmfile {__version__}'
        if comment:
            comment = comment.split('\n')[0].strip().encode('ascii').decode()
        if comment:
            comment = f'\n# {comment[:66]}\n'
        else:
            comment = ' '
        if magicnumber.startswith('P7'):
            if self.maxval < 1 or self.dtype.kind not in 'bu':
                raise ValueError(
                    f'data not compatible with {magicnumber!r} format'
                )
            return '\n'.join(
                (
                    f'P7{comment[:-1]}',
                    f'HEIGHT {self.height}',
                    f'WIDTH {self.width}',
                    f'DEPTH {self.depth}',
                    f'MAXVAL {self.maxval}',
                    '\n'.join(
                        f'TUPLTYPE {tt}' for tt in self.tupltype.split(' ')
                    ),
                    'ENDHDR\n',
                )
            )
        if magicnumber == 'PG':
            if self.dtype.kind not in 'iu':
                raise ValueError(
                    f'data not compatible with {magicnumber!r} format'
                )
            bitdepth = int(math.ceil(math.log2(self.maxval + 1)))
            if self.dtype.kind == 'i':
                bitdepth += 1
            return ''.join(
                (
                    'PG ',  # do not allow comments
                    'ML ' if self.byteorder == '>' else 'LM',
                    '-' if self.dtype.kind == 'i' else '',
                    f'{bitdepth} ',
                    f'{self.width} ' f'{self.height}\n',
                )
            )
        if magicnumber in 'P1 P4':
            if self.maxval != 1 or self.depth != 1 or self.dtype.kind != 'b':
                raise ValueError(
                    f'data not compatible with {magicnumber!r} format'
                )
            return f'{magicnumber}{comment}{self.width} {self.height}\n'
        if magicnumber in 'P2 P5':
            if self.depth != 1 or self.dtype.kind not in 'ui':
                raise ValueError(
                    f'data not compatible with {magicnumber!r} format'
                )
            return (
                f'{magicnumber}{comment}'
                f'{self.width} {self.height} {self.maxval}\n'
            )
        if magicnumber in 'P3 P6':
            if self.depth != 3 or self.dtype.kind not in 'ui':
                raise ValueError(
                    f'data not compatible with {magicnumber!r} format'
                )
            return (
                f'{magicnumber}{comment}'
                f'{self.width} {self.height} {self.maxval}\n'
            )
        raise ValueError(f'writing {magicnumber!r} format not supported')

    def __enter__(self) -> NetpbmFile:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self) -> str:
        if self.filename:
            arg = f'{os.path.split(os.path.normcase(self.filename))[-1]!r}'
        elif self._fh is not None:
            arg = str(type(self._fh).__name__)
        else:
            arg = ''
        return f'<{self.__class__.__name__}({arg})>'

    def __str__(self) -> str:
        return indent(
            repr(self),
            f'magicnumber: {self.magicnumber}',
            f'axes: {self.axes}',
            f'shape: {self.shape}',
            f'dtype: {self.dtype}',
            # f'byteorder: {self.byteorder}',
            f'scale: {self.scale}'
            if self.magicnumber in 'PF4 Pf'
            else f'maxval: {self.maxval}',
        )


def product(iterable: Iterable[int], /) -> int:
    """Return product of sequence of numbers."""
    prod = 1
    for i in iterable:
        prod *= i
    return prod


def indent(*args) -> str:
    """Return joined string representations of objects with indented lines."""
    text = '\n'.join(str(arg) for arg in args)
    return '\n'.join(
        ('  ' + line if line else line) for line in text.splitlines() if line
    )[2:]


def log_warning(msg, *args, **kwargs):
    """Log message with level WARNING."""
    import logging

    logging.getLogger('netpbmfile').warning(msg, *args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    """Command line usage main function.

    Show images specified on command line or all images in  directory.

    """
    from glob import glob
    from matplotlib import pyplot

    try:
        import tifffile
    except ImportError:
        tifffile = None

    if argv is None:
        argv = sys.argv

    if len(argv) > 1 and '--doctest' in argv:
        import doctest

        doctest.testmod()
        return 0

    if len(argv) == 1:
        files = glob('*.p*')
    elif '*' in argv[1]:
        files = glob(argv[1])
    elif os.path.isdir(argv[1]):
        files = glob(f'{argv[1]}/*.p*')
    else:
        files = argv[1:]

    for fname in files:
        try:
            with NetpbmFile(fname) as pam:
                print(pam)
                img = pam.asarray(copy=False)
                print()
        except ValueError as exc:
            # raise  # enable for debugging
            print(fname, exc)
            continue

        cmap = 'binary' if pam.maxval == 1 else 'gray'
        dtype = img.dtype
        shape = img.shape
        title = f'{os.path.split(fname)[-1]} {pam.magicnumber} {shape} {dtype}'

        multiimage = img.ndim > 3 or (
            img.ndim > 2 and img.shape[-1] not in (3, 4)
        )
        if tifffile is None or not multiimage:
            if img.ndim > 3 or (img.ndim > 2 and img.shape[-1] not in (3, 4)):
                warnings.warn('displaying first image only')
                img = img[0]
            if img.shape[-1] in (3, 4) and pam.maxval != 255:
                warnings.warn('converting RGB image for display')
                maxval = float(
                    numpy.max(img) if pam.maxval is None else pam.maxval
                )
                if maxval > 0.0:
                    img = img / maxval
                else:
                    img = img.copy()
                img *= 255
                numpy.rint(img, out=img)
                numpy.clip(img, 0, 255, out=img)
                img = img.astype('uint8')
            pyplot.imshow(img, cmap, interpolation='nearest')
            pyplot.title(title)
            pyplot.show()
        else:
            photometric = 'RGB' if img.shape[-1] in (3, 4) else 'minisblack'
            tifffile.imshow(
                img, photometric=photometric, cmap=cmap, title=title, show=True
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
