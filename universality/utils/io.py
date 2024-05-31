"""a module for basic I/O operations
"""
__author__ = "Reed Essick (reed.essick@gmail.com)"

#-------------------------------------------------

import os
import shutil

import numpy as np

from . import utils

#-------------------------------------------------

ALPHANUMERIC = 'a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9'.split()
DEFAULT_MAX_NUM_SAMPLES = np.infty

#-------------------------------------------------
# basic utilities for manipulating existing sapmles
#-------------------------------------------------

def column2logcolumn(name):
    return 'log(%s)'%name

def check_columns(present, required, logcolumns=[]):
    required = [column2logcolumn(column) if column in logcolumns else column for column in required]
    for column in required:
        assert column in present, 'required column=%s is missing!'%column

def load(inpath, columns=[], logcolumns=[], max_num_samples=DEFAULT_MAX_NUM_SAMPLES):
    data = []

    ### make sure we load only a single copy of each column; preserve the order as much as possible
    unique = []
    for col in columns:
        if col not in unique:
            unique.append(col)
    columns = unique

    with open(inpath, 'r') as obj:
        cols = [_.strip() for _ in obj.readline().strip().split(',')]
        if columns:
            check_columns(cols, columns)
        else:
            columns = cols

        inds = [cols.index(col) for col in columns]

        count = 0
        for line in obj:
            if line[0]=='#':
                continue
            if count >= max_num_samples:
                break
            fields = line.strip().split(',')
            ans = [] ### downselect to what we actually want and cast to float
            for ind in inds:
                try: ### try casting everything to a float
                    ans.append(float(fields[ind]))
                except ValueError: ### could not cast to a float
                    ans.append(fields[ind])
            data.append(ans)
            count += 1

    data = np.array(data) ### cast as an array

    cols = [] ### figure out column names and map to logs as requested
    for i, col in enumerate(columns):
        if col in logcolumns:
            data[:,i] = np.log(data[:,i])
            cols.append(column2logcolumn(col))
        else:
            cols.append(col)

    if not len(data): ### make sure data has the correct shape
        data = data.reshape((0, len(cols)))

    return data, cols #, dict((strind, col) for col, strind in strmap.items())

def load_weights(*args, **kwargs):
    """loads and returns weights from multiple columns via  delegation to load_logweights
    normalizes the weights while it's at it
    """
    normalize = kwargs.pop('normalize', True)
    return utils.exp_weights(load_logweights(*args, **kwargs), normalize=normalize)

def load_logweights(inpath, weight_columns, logweightcolumns=[], invweightcolumns=[], max_num_samples=DEFAULT_MAX_NUM_SAMPLES):
    """loads and returns logweights from multiple columns
    """
    data, columns = load(inpath, columns=weight_columns, max_num_samples=max_num_samples) ### load the raw data

    for i, column in enumerate(columns): ### iterate through columns, transforming as necessary
        if column in logweightcolumns:
            if column in invweightcolumns:
                data[:,i] *= -1

        else:
            if column in invweightcolumns:
                data[:,i] = 1./data[:,i]

            data[:,i] = np.log(data[:,i])

    # multiply weights across all samples, which is the same as adding the logs
    return np.sum(data, axis=1)

def write(path, data, cols, footer='', tmpdir='.', tmpdigits=6, verbose=False):
    """a standardized writing function that does an atomic write
    """
    path = os.path.abspath(path) ### make sure have the full path

    tmppath = os.path.join(tmpdir, "."+os.path.basename(path)+'-'+''.join(np.random.choice(ALPHANUMERIC, size=tmpdigits))) ### save to a temporary file
    if verbose:
        print('writing to temporary file location: '+tmppath)
    np.savetxt(tmppath, data, header=','.join(cols), footer=footer, comments='', delimiter=',')

    if verbose:
        print('moving to final location: '+path)
    shutil.move(tmppath, path) ### move to the permanent location
