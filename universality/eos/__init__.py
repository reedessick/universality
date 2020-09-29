"""a module that houses a few proposed EoS from the literature
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

from pkg_resources import (resource_listdir, resource_filename)

#-------------------------------------------------

eospaths = dict((path[:-4], resource_filename(__name__, path)) for path in resource_listdir(__name__,'') if path[-3:]=='csv')
DEFAULT_CRUST_EOS = 'sly'
