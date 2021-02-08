"""Package-wide test fixtures."""
import os
from typing import Any, Generator

import pytest

from kernreg.config import ROOT_DIR


@pytest.fixture(scope="function")
def change_test_dir(request: Any) -> Generator:
    """Switches to package directory."""
    # Change to package directory
    os.chdir(ROOT_DIR / "src")

    # Run the test
    yield

    # Change back to the calling directory to avoid side-effects
    os.chdir(request.config.invocation_dir)
