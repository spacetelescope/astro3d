"""Load the text catalog

The text catalog is a dictionary of all text, such as instructions,
that are displayed by the application.
"""
from pathlib import Path
import yaml

# If the text catalog has not been loaded, load now
try:
    TEXT_CATALOG
except NameError:
    _catalog_path = Path(__file__).parents[1] / 'data' / 'text_catalog.yaml'
    with open(_catalog_path, 'r') as fh:
        TEXT_CATALOG = yaml.safe_load(fh)
