"""Sky Shine - drone light show design and swarm control.

Two halves, one show format between them:

  design   geometry -> formation -> assign -> trajectory -> safety
  control  fleet -> driver (sim / guided stream / native show mode)

The safety contract is in safety.py and is the only thing that may
authorise a show for flight. Nothing else in this package is allowed to
mark a show as flyable.
"""

__version__ = "1.0.0"

from .show import (
    Formation,
    Show,
    ShowLimits,
    Timeline,
    load_show,
    save_show,
)

__all__ = [
    "Formation",
    "Show",
    "ShowLimits",
    "Timeline",
    "load_show",
    "save_show",
    "__version__",
]
