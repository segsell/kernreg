"""This is the entry-point to the kernreg package.

Include only imports which should be available using

.. code-block::

    import kernreg as kr

    kr.<func>
"""
from kernreg.bandwidth_selection import get_bandwidth_rsc  # noqa: F401
from kernreg.smooth import locpoly  # noqa: F401
from kernreg.utils import get_example_data, sort_by_x  # noqa: F401
from kernreg.visualize import plot  # noqa: F401

__version__ = "0.0.5"
