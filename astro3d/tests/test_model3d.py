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


@pytest.mark.xfail(
    reason='See issue #7',
    run=False
)
@pytest.mark.usefixtures('jail')
def test_replace_stars():
    """Test a full run of a spiral model"""

    # Get the data
    data_path = Path(environ['ASTRO3D_TESTDATA'])

    model = Model3D.from_fits(data_path / 'ngc3344_crop.fits')
    model.read_all_masks(str(data_path / 'features' / 'ngc3344_bulge.fits'))
    model.read_all_masks(str(data_path / 'special_features' / 'issue7_remove_star.fits'))

    # Create and save the model
    model.make(
        intensity=True, textures=True, double_sided=False, spiral_galaxy=True
    )
    model.write_stl('model3d', split_model=False)

    # Check against truth
    assert cmp('model3d.stl', data_path / 'truth' / 'model3d_issue7_remove_star' / 'model3d.stl')


@pytest.mark.usefixtures('jail')
def test_make_bulge():
    """Test a full run of a spiral model"""

    # Get the data
    data_path = Path(environ['ASTRO3D_TESTDATA'])

    model = Model3D.from_fits(data_path / 'ngc3344_crop.fits')
    model.read_all_masks(str(data_path / 'features' / 'ngc3344_bulge.fits'))

    # Create and save the model
    model.make(
        intensity=True, textures=True, double_sided=False, spiral_galaxy=True
    )
    model.write_stl('model3d', split_model=False)

    # Check against truth
    assert cmp('model3d.stl', data_path / 'truth' / 'model3d_make_bulge' / 'model3d.stl')


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
