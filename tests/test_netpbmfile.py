# test_netpbmfile.py

# Copyright (c) 2011-2023, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unittests for the netpbmfile package.

:Version: 2023.8.30

"""

import hashlib
import io
import os
import sys
import tempfile

import numpy
import pytest
from numpy.testing import assert_array_equal

import netpbmfile
from netpbmfile import NetpbmFile, imread, imwrite  # noqa

TEST_DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(TEST_DIR, '_tmp')

if not os.path.exists(TEMP_DIR):
    TEMP_DIR = tempfile.gettempdir()


class TempFileName:
    """Temporary file name context manager."""

    def __init__(self, name=None, ext='', remove=False):
        self.remove = remove or TEMP_DIR == tempfile.gettempdir()
        if not name:
            with tempfile.NamedTemporaryFile(prefix='test_') as fh:
                self.name = fh.named
        else:
            self.name = os.path.join(TEMP_DIR, f'test_{name}{ext}')

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        if self.remove:
            try:
                os.remove(self.name)
            except Exception:
                pass


def md5(a):
    """Return MD5 of numpy array data."""
    return hashlib.md5(numpy.ascontiguousarray(a).data).digest()


def idfn(val):
    """Replace checksum bytes test IDs with 'md5'."""
    return 'md5' if isinstance(val, bytes) else None


# (4, 32, 31, 4) uint16, 12 bit
RGBA64 = numpy.load(os.path.join(TEST_DIR, 'data.npy'))
RGBA32 = (RGBA64 >> 4).astype(numpy.uint8)
RGB48 = RGBA64[..., :3]
RGB24 = RGBA32[..., :3]
GRAY16 = RGBA64[..., 0]
GRAY8 = RGBA32[..., 0]
BILEVEL = GRAY8 > 128

FILES = [
    (
        'P1.pbm',
        'P1',
        'bool',
        'YX',
        (10, 6),
        1,
        b'"\xd0\xd3A\xd64`\x8c\x19\xb8\xce\x12N\xd9\x1aN',
    ),
    (
        'P1_1.pbm',
        'P1',
        'bool',
        'YX',
        (10, 6),
        1,
        b'"\xd0\xd3A\xd64`\x8c\x19\xb8\xce\x12N\xd9\x1aN',
    ),
    (
        'P2.pgm',
        'P2',
        'u1',
        'YX',
        (7, 24),
        15,
        b'T|\xf5e+\x8c\xf5\x92\x92\xa8a\xe9\x83=#\xc6',
    ),
    (
        'P2_multi.pgm',
        'P2',
        '>u2',
        'IYX',
        (2, 8, 8),  # not standard
        4095,
        b'xb9\xea\x14\x989X\xa8\xa2\rZ\xeb0\xe1\x89',
    ),
    (
        'P3.ppm',
        'P3',
        'u1',
        'YXS',
        (2, 3, 3),
        255,
        b'+\xbc7\x89qw\xf9\xf4Yq\x95\xa4\xebi\x98\xe3',
    ),
    (
        'P3_uint1.ppm',
        'P3',
        'u1',
        'YXS',
        (2, 3, 3),
        1,
        b'\xb7M\xc2f\x1es{\xb4L\xf6\x8ep\xe0.f\x98',
    ),
    (
        'P3_uint1_2.ppm',
        'P3',
        'u1',
        'YXS',
        (2, 3, 3),
        1,
        b'\xb7M\xc2f\x1es{\xb4L\xf6\x8ep\xe0.f\x98',
    ),
    (
        'P4.pbm',
        'P4',
        'bool',
        'YX',
        (16, 14),
        1,
        b'>`\xfbN3<\x8c\xc6a\x8b\x96[\xb1\x12\xc6\xa4',
    ),
    (
        'P4_multi.pbm',
        'P4',
        'bool',
        'IYX',
        (2, 342, 512),
        1,
        b'Z\xa2\xdc\xa8\xf5\x1d\xabT\xf1\xa3\x022\xc9\x9cg\xfb',
    ),
    (
        'P5.pgm',  # file contains non-ASCII character in header
        'P5',
        '>u2',
        'YX',
        (128, 128),
        8191,  # 13-bit
        b'\x95\xc5<\xb8X\xcf2\xcd\xe9\xaaw\x03\xe5$X\xf2',
    ),
    (
        'P5_1px.pgm',
        'P5',
        'u1',
        'YX',
        (1, 1),
        255,
        b"@:\xe0\x91\xd3\xbej\xcf\x11\x81\x14\x85'\xf1\xe0\xae",
    ),
    (
        'P5_uint32.pgm',
        'P5',
        '>u4',  # not standard
        'YX',
        (2, 2),
        4294967295,
        b'B\xd91a\x16\x05T\xdf\x96\xd3\xa3\x82\x88\xdeW\x97',
    ),
    (
        'P5_multi.pgm',
        'P5',
        'u1',
        'IYX',
        (3, 38, 38),
        255,
        b'\xb2\xe5\x8d\xc5\xfb\xe5\xcc\x81\xae1\xa9\x03\xe5l\xa2R',
    ),
    (
        'P6.ppm',
        'P6',
        '>u2',
        'YXS',
        (333, 440, 3),
        1953,
        b'&\xf5x\x84\xdd:\xdbJ_c6D\x85\xd5;\xd2',
    ),
    (
        'P6_maxval.ppm',
        'P6',
        'u1',
        'YXS',
        (320, 240, 3),
        32,
        b'.\xea\x11\xf1\x88\x94\x05\xd1\xf9\xdf\t\xcd\xcb;zB',
    ),
    (
        'P7_332.pam',
        'P7 332',
        'u1',
        'YXS',
        (46, 70, 3),
        255,
        b'\xfc;8\x91[\x10\xb3\x88)\nL\x9cE\xfa\nI',
    ),
    (
        'P7_rgba.pam',
        'P7',
        'u1',
        'YXS',
        (2, 4, 4),
        255,
        b'\xb5a\x07R\xce\xdd\xcd\x13\x97&0\x84\x0b\xc6\x9d4',
    ),
    (
        'PF.pfm',
        'PF',
        '<f4',
        'YXS',
        (768, 512, 3),
        1.0,
        b'\xc5\xe0\xbeK\xd5g,\x8b\x9b\xe3Ix3\xdb1\xbd',
    ),
    (
        'PFbe.pfm',
        'PF',
        '>f4',
        'YXS',
        (240, 240, 3),
        1.0,
        b'\xbaZIG\x1e\xb6\x0cE\x83\xc0i\x04\xd9\xe8$\r',
    ),
    (
        'Pf_gray.pfm',
        'Pf',
        '<f4',
        'YX',
        (2016, 2940),
        0.003922,
        b'-#\xe4\x8a\xc1\x97\x99-\x04\xac\x8fm\xa5G\x16;',
    ),
    (
        'PF4.pfm',
        'PF4',
        '<f4',
        'YXS',
        (1024, 768, 4),
        1.0,
        b'\x1b\xed\xca\x99\xc4\xc4\xf8m\xe5.\xe3I4\xdfs\xb0',
    ),
    (
        'PG_int8.pgx',
        'PG',
        'i1',
        'YX',
        (8, 8),
        127,
        b'R\x06 \xde\x89\xe2 \xf9\xb5\x85\x0c\xc9|\xbf\xf4l',
    ),
    (
        'PG_int12.pgx',
        'PG',
        '>i2',
        'YX',
        (8, 8),
        2047,
        b'O\xe10Y\x8dG\xf1|\x19\xa7\xc4\x93\xb4\xce\x0c\xf1',
    ),
    (
        'PG_int16.pgx',
        'PG',
        '>i2',
        'YX',
        (8, 8),
        32767,
        b'O\xe10Y\x8dG\xf1|\x19\xa7\xc4\x93\xb4\xce\x0c\xf1',
    ),
    (
        'foo.pgm',
        'P5',
        '<u2',  # wrong byteorder
        'YX',
        (480, 640),
        65535,
        b'\xd2I&\xe6\xe1\xa0\x93\x03\x80^M\xdd\x02\xb5\xb5\x17',
    ),
]


@pytest.mark.skipif(__doc__ is None, reason='__doc__ is None')
def test_version():
    """Assert netpbmfile versions match docstrings."""
    ver = ':Version: ' + netpbmfile.__version__
    assert ver in __doc__
    assert ver in netpbmfile.__doc__


@pytest.mark.parametrize(
    'name, magicnumber',
    [
        ('bilevel', 'p1'),
        ('bilevel', 'p4'),
        ('gray8', 'p2'),
        ('gray8', 'p5'),
        ('gray16', 'p2'),
        ('gray16', 'p5'),
        ('rgb24', 'p3'),
        ('rgb24', 'p6'),
        ('rgb48', 'p3'),
        ('rgb48', 'p6'),
    ],
)
@pytest.mark.parametrize('multi', [False, True])
@pytest.mark.parametrize('pam', [False, True])
def test_roundtrip(name, magicnumber, multi, pam):
    """Test writing and reading PNM and PAM formats."""
    comment = 'written by netpbmfile.py'
    data = globals()[name.upper()]
    if pam:
        magicnumber = 'P7'
        ext = 'pam'
        if name == 'bilevel':
            data = data.astype('u1')
    else:
        ext = 'pnm'
    fname = f'{name}{"x4" if multi else ""}.{magicnumber}.{ext}'
    with TempFileName(fname) as temp:
        magicnumber = magicnumber.upper()
        if not multi:
            data = data[0]
        imwrite(temp, data, magicnumber=magicnumber, comment=comment)
        with NetpbmFile(temp) as fh:
            if fh.magicnumber in 'P1 P2 P3':
                assert fh.shape == data.shape[1:] if multi else data.shape
            else:
                assert fh.shape == data.shape
            assert fh.dtype.str[1:] == data.dtype.str[1:]
            assert fh.byteorder == '>'
            assert comment in fh.header
            assert_array_equal(fh.asarray(), data)
            if magicnumber not in 'P4 P5 P6':
                return
            # export PNM as PAM
            with TempFileName(fname + '.pam') as fn:
                fh.write(fn, magicnumber='P7', comment=comment)
                with NetpbmFile(fn) as fh2:
                    assert fh2.magicnumber == 'P7'
                    assert comment in fh2.header
                    assert_array_equal(fh.asarray(), data)


@pytest.mark.parametrize(
    'fname, magicnumber, dtype, axes, shape, maxval, hash', FILES, ids=idfn
)
def test_file(fname, magicnumber, dtype, axes, shape, maxval, hash):
    """Verify files can be read and rewritten."""
    filepath = os.path.join(TEST_DIR, fname)
    if not os.path.exists(filepath):
        pytest.skip(f'{fname} not found')

    byteorder = '<' if dtype[:2] == '<u' else None
    multitext = magicnumber in 'P1 P2 P3' and axes[0] == 'I'
    if multitext:
        axes = axes[1:]
        shape = shape[1:]

    with NetpbmFile(filepath, byteorder=byteorder) as fh:  # type: ignore
        assert str(fh)
        assert fh.magicnumber == magicnumber
        # assert fh.tupltype == tupltype
        tupltype = fh.tupltype
        assert fh.axes == axes
        assert fh.shape == shape
        assert fh.dtype == dtype
        if 'f' in dtype:
            assert fh.scale == maxval
            assert fh.byteorder == dtype[0]
        elif byteorder is not None:
            assert fh.maxval == maxval
            assert fh.byteorder == byteorder
        else:
            assert fh.maxval == maxval
            assert fh.byteorder == '>'
        data = fh.asarray()
        if not multitext:
            assert data.shape == shape
        assert md5(data) == hash

    if magicnumber in 'PF4 Pf P7 332':
        return

    # rewrite
    with TempFileName(fname) as temp:
        imwrite(
            temp,
            data,
            magicnumber=magicnumber,
            maxval=maxval,
            tupltype=tupltype,
        )
        with NetpbmFile(temp) as fh:
            assert fh.magicnumber == magicnumber
            assert fh.axes == axes
            assert fh.shape == shape
            assert fh.maxval == maxval
            assert fh.tupltype == tupltype
            assert fh.byteorder == '>'
            assert_array_equal(fh.asarray(cache=True), data)
            fh.close()
            # overwrite
            fh.write(temp)
        with NetpbmFile(temp) as fh:
            assert fh.magicnumber == magicnumber
            assert fh.axes == axes
            assert fh.shape == shape
            assert fh.maxval == maxval
            assert fh.tupltype == tupltype
            assert fh.byteorder == '>'
            assert_array_equal(fh.asarray(cache=True), data)

    if magicnumber in 'P7 PG':
        return

    # rewrite as text/binary into BytesIO
    if magicnumber in 'P1 P2 P3 P4 P5 P6':
        magicnumber = {
            'P1': 'P4',
            'P2': 'P5',
            'P3': 'P6',
            'P4': 'P1',
            'P5': 'P2',
            'P6': 'P3',
        }[magicnumber]
        temp = io.BytesIO()
        imwrite(temp, data, magicnumber=magicnumber, maxval=maxval)
        with NetpbmFile(temp) as fh:
            str(fh)
            assert fh.magicnumber == magicnumber
            assert fh.maxval == maxval
            if fh.axes[0] != 'I' and axes[0] != 'I':
                assert fh.axes == axes
                assert fh.shape == shape
            assert fh.byteorder == '>'
            assert_array_equal(fh.asarray(), data)
        temp.close()

    # rewrite as P7
    magicnumber = 'P7'
    with TempFileName(fname + '.pam') as temp:
        imwrite(
            temp,
            data,
            magicnumber=magicnumber,
            maxval=maxval,
            tupltype=tupltype,
        )
        with NetpbmFile(temp) as fh:
            assert fh.magicnumber == magicnumber
            assert fh.maxval == maxval
            assert fh.tupltype == tupltype
            assert fh.byteorder == '>'
            if not multitext:
                assert fh.axes == axes
                assert fh.shape == shape
            assert_array_equal(fh.asarray(), data)


def test_byteorder():
    """Test writing little-endian P5. Read and write open file handle."""
    data = GRAY16
    with TempFileName('gray16_le.p5.pgm') as temp:
        with open(temp, 'wb') as filehandle:
            imwrite(temp, data, byteorder='<')
        with open(temp, 'rb') as filehandle:
            with NetpbmFile(filehandle, byteorder='<') as fh:
                str(fh)
                assert fh.magicnumber == 'P5'
                assert fh.maxval == 4095
                assert fh.shape == data.shape
                assert fh.dtype == data.dtype
                assert fh.byteorder == '<'
                assert_array_equal(fh.asarray(), data)
            with NetpbmFile(filehandle) as fh:
                assert fh.dtype != data.dtype


def test_non_netpbm():
    """Test non-netpbm formats."""
    with pytest.raises(ValueError):
        imread(os.path.join(TEST_DIR, 'data.npy'))

    with pytest.raises(ValueError):
        imread(io.BytesIO(b'P7      '))


def test_lfdfiles():
    """Test lfdfiles package."""
    try:
        from lfdfiles import LfdFile
    except ImportError as exc:
        pytest.skip(exc.msg)

    fname = os.path.join(TEST_DIR, 'P4_multi.pbm')
    data = imread(fname)
    with LfdFile(fname) as lfd:
        assert_array_equal(data, lfd.asarray())


if __name__ == '__main__':
    import warnings

    # warnings.simplefilter('always')
    warnings.filterwarnings('ignore', category=ImportWarning)
    argv = sys.argv
    argv.append('--cov-report=html')
    argv.append('--cov=netpbmfile')
    argv.append('--verbose')
    sys.exit(pytest.main(argv))
