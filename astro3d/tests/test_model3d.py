"""Test the core operations of model3d"""

from filecmp import cmp
from os import environ
from pathlib import Path
import pytest

from astro3d.core.model3d import Model3D


@pytest.mark.usefixtures('jail')
@pytest.mark.skipif(
    environ.get('ASTRO3D_TESTDATA') is None,
    reason=(
        'Test requires environmental ASTRO3D_TESTDATA'
        ' pointing to the test data set.'
    )
)
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
@pytest.mark.skipif(
    environ.get('ASTRO3D_TESTDATA') is None,
    reason=(
        'Test requires environmental ASTRO3D_TESTDATA'
        ' pointing to the test data set.'
    )
)
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
