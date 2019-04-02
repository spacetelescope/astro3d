"""Text catalog tests"""


def test_import_text_catalog():
    """Test the the catalog is importable"""
    import astro3d.util.text_catalog


def test_basic_info():
    """Ensure the text catalog has some basic information"""
    from astro3d.util.text_catalog import TEXT_CATALOG

    basic_keys = set(['instructions_default', 'shape_editor'])
    assert basic_keys.issubset(TEXT_CATALOG.keys())
