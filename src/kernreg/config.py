"""General configuration for kernreg."""
from pathlib import Path

# Obtain the root directory of the package.
# Do not import kernreg which creates a circular import.
ROOT_DIR = Path(__file__).parent.parent.parent
PACKAGE_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_DIR = ROOT_DIR / "tests"
TEST_RESOURCES = ROOT_DIR / "tests" / "resources"
RESOURCES_DIR = PACKAGE_DIR / "example_data"
