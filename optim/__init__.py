"""
:mod:`torch.optim` is a package implementing various optimization algorithms.
Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from . import lr_scheduler
from .adam import Adam
from .optimizer import Optimizer
from .sgd import SGD

del adam
del sgd
del optimizer
