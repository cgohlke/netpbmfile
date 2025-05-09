Revisions
---------

2025.5.8

- Remove doctest command line option.

2025.1.1

- Improve type hints.
- Drop support for Python 3.9, support Python 3.13.

2024.5.24

- Fix docstring examples not correctly rendered on GitHub.

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

- Drop support for Python 3.6 (NEP 29).
- Support os.PathLike file names.

2020.1.1

- Fix reading tightly packed P1 format and ASCII data with inline comments.
- Drop support for Python 2.7 and 3.5.
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
