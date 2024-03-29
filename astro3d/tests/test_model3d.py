"""Test the core operations of model3d"""

from filecmp import cmp
from os import environ
from pathlib import Path
import pytest

from astro3d.core.model3d import Model3D
from astro3d.core.model3d import read_stellar_table

pytestmark = pytest.mark.skipif(
    environ.get('ASTRO3D_TESTDATA') is None,
    reason=(
        'Test requires environmental ASTRO3D_TESTDATA'
        ' pointing to the test data set.'
    )
)


@pytest.mark.usefixtures('jail')
def test_remove_stars():
    """Test star removal handling
    """

    # Get the data
    data_path = Path(environ['ASTRO3D_TESTDATA'])

    model = Model3D.from_fits(data_path / 'ngc3344_crop.fits')
    model.read_all_masks(str(data_path / 'features' / 'ngc3344_remove_star_*.fits'))

    # Create and save the model
    model.make(
        intensity=True, textures=True, double_sided=False, spiral_galaxy=False
    )
    model.write_stl('model3d', split_model=False)

    # Check against truth
    truth_path = data_path / 'truth' / 'model3d_make_remove_stars' / 'model3d.stl'
    assert cmp('model3d.stl', truth_path)


def test_catalog_read(caplog):
    """test catalog reading

    Parameters
    ----------
    caplog: log from the test, provided by the pytest
    `caplog` fixture
    """

    # Get the data
    data_path = Path(environ['ASTRO3D_TESTDATA'])
    catalog = data_path / 'special_features' / 'catalog_uppercase_names.txt'

    # Execute
    table = read_stellar_table(str(catalog), 'stars')

    # Ensure no warnings
    assert 'Cannot find required column names' not in caplog.text

    # Ensure other table characteristics
    assert len(table.colnames) >= 3
    assert set(('xcentroid', 'ycentroid', 'flux')).issubset(table.colnames)


def test_catalog_read_badnames(caplog):
    """test catalog reading

    Parameters
    ----------
    caplog: log from the test, provided by the pytest
    `caplog` fixture
    """

    # Get the data
    data_path = Path(environ['ASTRO3D_TESTDATA'])
    catalog = data_path / 'special_features' / 'catalog_bad_names.txt'

    # Execute
    table = read_stellar_table(str(catalog), 'stars')

    # Ensure warnings
    assert 'Cannot find required column names' in caplog.text

    # Ensure other table characteristics
    assert len(table.colnames) >= 2
    assert set(('xcentroid', 'ycentroid')).issubset(table.colnames)


@pytest.mark.usefixtures('jail')
@pytest.mark.parametrize(
    'id, make_kwargs, use_bulge_mask',
    [
        ('full',                 {'spiral_galaxy': True,  'compress_bulge': True},  True),
        ('none',                 {'spiral_galaxy': False, 'compress_bulge': False}, True),
        ('spiral_only',          {'spiral_galaxy': True,  'compress_bulge': False}, True),
        ('compress_only',        {'spiral_galaxy': False, 'compress_bulge': True},  True),
        ('full_nomask',          {'spiral_galaxy': True,  'compress_bulge': True},  False),
        ('none_nomask',          {'spiral_galaxy': False, 'compress_bulge': False}, False),
        ('spiral_only_nomask',   {'spiral_galaxy': True,  'compress_bulge': False}, False),
        ('compress_only_nomask', {'spiral_galaxy': False, 'compress_bulge': True},  False),
    ]
)
def test_bulge_handling(id, make_kwargs, use_bulge_mask, caplog):
    """Test bulge/spiral model handling

    Parameters
    ----------
    id: str
        Test identifier used to pick input and truth data.

    make_kwargs: dict
        The `Astro3d.make` keyword arguments.

    use_bulge_mask: bool
        Use a bulge mask.

    caplog: fixture
        The magical `pytest.caplog` fixture that encapsulates the log output.
    """

    # Get the data
    data_path = Path(environ['ASTRO3D_TESTDATA'])

    model = Model3D.from_fits(data_path / 'ngc3344_crop.fits')
    if use_bulge_mask:
        model.read_all_masks(str(data_path / 'features' / 'ngc3344_bulge.fits'))

    # Create and save the model
    model.make(
        intensity=True, textures=True, double_sided=False, **make_kwargs
    )
    model.write_stl('model3d', split_model=False)

    # Check against truth
    if use_bulge_mask:
        truth_path = data_path / 'truth' / 'model3d_make_bulge' / id / 'model3d.stl'
    else:
        truth_path = data_path / 'truth' / 'model3d_make_bulge' / 'none_nomask' / 'model3d.stl'
    assert cmp('model3d.stl', truth_path)

    if not use_bulge_mask and id is not 'none_nomask':
        assert 'A "bulge" mask must be defined.' in caplog.text


@pytest.mark.usefixtures('jail')
@pytest.mark.parametrize(
    'id, make_kwargs',
    [
        ('full',      {'intensity': True,  'textures': True,  'double_sided': True,  'spiral_galaxy': True}),
        ('nospiral',  {'intensity': True,  'textures': True,  'double_sided': True,  'spiral_galaxy': False}),
        ('nospiralnocompress',
         {'intensity': True,  'textures': True,  'double_sided': True,  'spiral_galaxy': False, 'compress_bulge': False}
        ),
        ('nodouble',  {'intensity': True,  'textures': True,  'double_sided': False, 'spiral_galaxy': False}),
        ('intensity', {'intensity': True,  'textures': False, 'double_sided': False, 'spiral_galaxy': False}),
        ('texture',   {'intensity': False, 'textures': True,  'double_sided': False, 'spiral_galaxy': False}),
    ]
)
def test_make(id, make_kwargs):
    """Test a full run of a spiral model"""

    # Get the data
    data_path = Path(environ['ASTRO3D_TESTDATA'])

    model = Model3D.from_fits(data_path / 'ngc3344_crop.fits')
    model.read_all_masks(str(data_path / 'features' / '*.fits'))

    # Create and save the model
    model.make(**make_kwargs)
    model.write_stl('model3d', split_model=False)

    # Check for file existence
    assert cmp('model3d.stl', data_path / 'truth' / 'model3d_make' / id / 'model3d.stl')
