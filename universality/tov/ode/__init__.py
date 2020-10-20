"""a module that houses TOV solvers. We provide several formulations of the TOV equations for cross-checking
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

### by default, we use the log(enthalpy per rest mass) formalism
from .logenthalpy import (integrate, MACRO_COLS)
