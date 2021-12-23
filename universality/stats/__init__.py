"""a module that houses basic statistics we use to quantify our results based on large monte carlo integrals
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from .montecarlo import *
from .samples import *
#from .information import * ### relies on universality.kde.kde, which relies on stats.montecarlo and stats.samples
from .kde import *
