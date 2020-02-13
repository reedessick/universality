"""a module that will help us write SUBs and DAGs for distributed workflows
"""
__author__ = "reed.essick@gmail.com"

#-------------------------------------------------

import os
from distutils.spawn import find_executable
from getpass import getuser

from . import eos

#-------------------------------------------------

DEFAULT_UNIVERSE = 'vanilla'

#--- workflow defaults

DEFAULT_START = 0
DEFAULT_NUM_PER_DIRECTORY = 1000
DEFAULT_OUTDIR = '.'
DEFAULT_TAG = ''

#-------------------------------------------------

### utilities for parsing config files

def eval_kwargs(**kwargs):
    """a utility to help when parsing classads from INI files
    """
    for key, value in kwargs.items():
        try:
            value = eval(value)
        except (SyntaxError, NameError):
            value = value.strip().split()
            try:
                value = [eval(_) for _ in value]
            except (SyntaxError, NameError):
                pass
            finally:
                if len(value)==1:
                    value = value[0]
        kwargs[key] = value
    return kwargs

def classads2dict(condor_classads):
    """a utility to help when parsing classads from INI files
    """
    return dict((condor_classads[2*i], str(condor_classads[2*i+1])) for i in xrange(len(condor_classads)/2))

#-------------------------------------------------

### SUB files for different (expensive) tasks

def write_sub(path, **classads):
    """standardize how sub files are written
    """
    with open(path, 'w') as file_obj:
        file_obj.write('\n'.join(' = '.join(item) for item in classads.items())+'\nqueue 1')
    return path

#---

def which(exe):
    """standardize finding full paths to exes
    """
    path = find_executable(exe)
    if path is None:
        raise NameError('%s not found! Please check your PATH and other environmental variables'%exe)
    return path

def path2logdir(path):
    """standardize where logs are written
    """
    logdir = os.path.join(os.path.abspath(os.path.dirname(path)), 'log')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir

def basic_classads(exe, logdir, **classads):
    """add common classads
    """
    classads['universe'] = classads.get('universe', DEFAULT_UNIVERSE)
    classads['getenv'] = classads.get('getenv', True)
    assert 'accounting_group' in classads, 'must specify an accounting_group'
    classads['accounting_group_user'] = getuser()

    classads['log'] = os.path.join(logdir, 'condor-draw-gpr_$(JOB)-$(Cluster)-$(Process).log')
    classads['error'] = os.path.join(logdir, 'condor-draw-gpr_$(JOB)-$(Cluster)-$(Process).err')
    classads['output'] = os.path.join(logdir, 'condor-draw-gpr_$(JOB)-$(Cluster)-$(Process).out')

    classads['executable'] = exe

    return classads

def tag2opt(tag):
    return '--tag '+tag if tag else ''

def argdict2str(**argdict):
    tmp = '%s="%s"'
    return ' '.join(tmp%item for item in argdict.items())

#------------------------

def sub_draw_gpr(path, **classads):
    """write the sub file for draw-gpr into path
    """
    classads = basic_classads(which('draw-gpr'), path2logdir(path), **classads)
    classads['arguments'] = '$(hdf5path) --start $(start) --num-draws $(num_draws) --num-per-directory $(num_per_directory) --output-dir $(outdir) $(tag) --verbose'
    write_sub(path, **classads)

DEFAULT_START = 0
DEFAULT_OUTDIR = '.'
DEFAULT_TAG = ''
def args_draw_gpr(
        hdf5path,
        start=DEFAULT_START,
        num_per_directory=DEFAULT_NUM_PER_DIRECTORY,
        outdir=DEFAULT_OUTDIR,
        tag=DEFAULT_TAG,
    ):
    """format the ARGS string for draw-gpr nodes in a DAG
    NOTE: we fix num_draws==num_per_directory, so the condor workflows will only be able to generate total numbers of samples that are integer multiples of "num_per_directory"
    """
    args = dict()

    args['hdf5path'] = hdf5path
    args['start'] = '%d'%start
    args['num_draws'] = args['num_per_directory'] = '%d'%num_per_directory
    args['outdir'] = os.path.abspath(outdir)
    args['tag'] = tag2opt(tag)

    return argdict2str(**args)

#---

def sub_integrate_phi(path, **classads):
    """write the sub file for integrate-phi into path
    """
    classads = basic_classads(which('integrate-phi'), path2logdir(path), **classads)
    classads['arguments'] = '$(phipaths) $(reference_pressurec2) $(outpaths) $(sigma_logpressurec2) $(stitch_below_reference_pressure) --curst-eos $(crust_eos) --verbose'
    write_sub(path, **classads)

def args_integrate_phi(
        reference_pressurec2,
        sigma_logpressurec2,
        crust=eos.DEFAULT_CRUST_EOS,
        stitch_below_reference_pressure=True,
        start=DEFAULT_START,
        num_per_directory=DEFAULT_NUM_PER_DIRECTORY,
        outdir=DEFAULT_OUTDIR,
        tag=DEFAULT_TAG,
    ):
    """format the ARGS string for integrate-phi nodes in a DAG
    NOTE: this is based on the logic within args_draw_gpr and will integrate every EOS generated in the corresponding draw-gpr node
    """
    args = dict()

    phipaths = [] ### FIXME: set this up based on start, num_per_directory, outdir, and tag
    outpaths = [] ### FIXME

    raise NotImplementedError

    args['phipaths'] = phipaths
    args['outpaths'] = oupaths
    args['reference_pressurec2'] = "%.9e"%reference_pressurec2
    args['sigma_logpressurec2'] = "%.9e"%sigma_logpressurec2
    args['stitch_below_reference_pressure'] = '--stitch-below-reference-pressure' if stitch_below_reference_pressure else ''
    args['crust'] = crust

    return argdict2str(**args)

#---

def sub_draw_samples(path, **classads):
    """write the sub file for draw-samples into path
    """
    classads = basic_classads(which('draw-samples'), path2logdir(path), **classads)
    classads['arguments'] = ''

    raise NotImplementedError('''set up args for draw-samples''')

    write_sub(path, **classads)

def args_draw_samples(*args, **kwargs):
    """format the ARGS string for draw-samples nodes in a DAG
    """
    args = dict()

    raise NotImplementedError('''set up args for draw-samples''')

    return argdict2str(**args)

#---

def sub_weigh_samples(path, **classads):
    """write the sub file for weigh-samples into path
    """
    classads = basic_classads(which('weigh-samples'), path2logdir(path), **classads)
    classads['arguments'] = '$(source) $(target) $(output) $(columns) $(logcolumns) $(weight_column) $(weight_column_is_log) $(invert_weight_column) $(column_range) $(reflect) $(prune) $(output_weight_column) $(do_not_log_output_weights) $(bandwidth) --verbose'
    write_sub(path, **classads)

def args_weigh_samples(*args, **kwargs):
    """format the ARGS string for weigh-samples nodes in a DAG
    """
    args = dict()

    '''
    args['source'] = "%s" 
    target
    output
    columns
    logcolumns
    weight_column
    weight_column_is_log
    invert_weight_column
    column_range
    reflect
    prune
    output_weight_column
    do_not_log_output_weights
    bandwidth
'''

    raise NotImplementedError('''
usage: weigh-samples [-h] [-v] [--logcolumn LOGCOLUMN]
                     [--weight-column WEIGHT_COLUMN]
                     [--weight-column-is-log WEIGHT_COLUMN_IS_LOG]
                     [--invert-weight-column INVERT_WEIGHT_COLUMN]
                     [-r COLUMN_RANGE COLUMN_RANGE COLUMN_RANGE] [--reflect]
                     [--prune] [--output-weight-column OUTPUT_WEIGHT_COLUMN]
                     [--do-not-log-output-weights] [-b BANDWIDTH]
                     [--num-proc NUM_PROC]
                     source target output columns [columns ...]

a script that computes the associated weights for target_samples.csv based on
the distribution within source_samples.csv.

positional arguments:
  source
  target
  output
  columns

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose
  --logcolumn LOGCOLUMN
                        convert the values read in for this column to natural
                        log. Can be repeated to specify multiple columns.
                        DEFAULT=[]
  --weight-column WEIGHT_COLUMN
                        if provided, thie numerical values from this column
                        will be used as weights in the KDE
  --weight-column-is-log WEIGHT_COLUMN_IS_LOG
                        if supplied, interpret the values in weight_column as
                        log(weight), meaning we exponentiate them before using
                        them in the KDE
  --invert-weight-column INVERT_WEIGHT_COLUMN
                        After extracting the weights from source_samples.csv,
                        this will compute the KDE using the inverse of those
                        values; e.g.: weight by the inverse of the prior for a
                        set of posterior samples so the effective sampling is
                        with respect to the likelihood. The inversion is done
                        after exponentiation when --weight-column-is-log is
                        supplied.
  -r COLUMN_RANGE COLUMN_RANGE COLUMN_RANGE, --column-range COLUMN_RANGE COLUMN_RANGE COLUMN_RANGE
                        specify the ranges used in corner.corner (eg.:
                        "--column-range column minimum maximum"). Can specify
                        ranges for multiple columns by repeating this option.
                        DEFAULT will use the minimum and maximum observed
                        sample points.
  --reflect             reflect the points about their boundaries within the
                        KDE
  --prune               throw away samples that live outside the specified
                        ranges
  --output-weight-column OUTPUT_WEIGHT_COLUMN
                        the name of the new weight-column in the output file.
                        **BE CAREFUL!** You should make sure this is
                        consistent with whether or not you specified --do-not-
                        log-output-weight! DEFAULT=logweight
  --do-not-log-output-weights
                        record the raw weights instead of the log(weight) in
                        the output CVS. **BE CAREFUL!** You should make sure
                        this is consistent with the name specified by
                        --output-weight-column.
  -b BANDWIDTH, --bandwidth BANDWIDTH
                        the bandwidth (standard deviation) used within the
                        Gaussian KDE over whitened data. DEFAULT=0.03
  --num-proc NUM_PROC   number of processes for parallelized computation of
                        logkde. DEFAULT=15
''')

    return argdict2str(**args)

#---

def sub_marginalize_samples(path, **classads):
    """write the sub file for marginalize-samples into path
    """
    classads = basic_classads(which('marginalize-samples'), path2logdir(path), **classads)
    classads['arguments'] = '$(samples) $(columns) $(weight_column) $(weight_column_is_log) --output-path $(outpath) --verbose'
    write_sub(path, **classads)

def args_marginalize_samples(*args, **kwargs):
    """format the ARGS string for marginalize-samples nodes in a DAG
    """
    args = dict()

    raise NotImplementedError('''
usage: marginalize-samples [-h] [--weight-column WEIGHT_COLUMN]
                           [--weight-column-is-log WEIGHT_COLUMN_IS_LOG]
                           [--max-num-samples MAX_NUM_SAMPLES] [-v]
                           [-o OUTPUT_PATH]
                           samples columns [columns ...]

marginalize over all weights associated with combinations of columns, creating
a new file with marginalized weights within it

positional arguments:
  samples
  columns               columns used to define unique sets. We will
                        marginalize over anything not specified here

optional arguments:
  -h, --help            show this help message and exit
  --weight-column WEIGHT_COLUMN
  --weight-column-is-log WEIGHT_COLUMN_IS_LOG
  --max-num-samples MAX_NUM_SAMPLES
  -v, --verbose
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        print to this file if specified. Otherwise we print to
                        STDOUT.
''')

    return argdict2str(**args)

#---

def sub_collate_samples(path, **classads):
    """write the sub file for collate-samples into path
    """
    classads = basic_classads(which('collate-samples'), path2logdir(path), **classads)
    classads['arguments'] = '$(reference_column) $(outpath) $(samples) $(column_map) --Verbose'
    write_sub(path, **classads)

def args_collate_samples(*args, **kwargs):
    """format the ARGS string for collate-samples nodes in a DAG
    """
    args = dict()

    raise NotImplementedError('''set up args for collate-samples
usage: collate-samples [-h] [-s SAMPLES [SAMPLES ...]]
                       [--column-map COLUMN_MAP COLUMN_MAP COLUMN_MAP] [-v]
                       [-V]
                       reference_column output

copy columns from multiple sample CSV into a target CSV

positional arguments:
  reference_column      the column that is used to match rows in samples and
                        target. This must be present in both files. It must
                        also have only a single row in source for each value
                        of this column.
  output                if this file already exists, we require
                        reference_column to exist in that file and then map
                        all the --samples into those rows. Otherwise, we
                        create a new file that only containes the columns from
                        the samples

optional arguments:
  -h, --help            show this help message and exit
  -s SAMPLES [SAMPLES ...], --samples SAMPLES [SAMPLES ...]
                        eg: "--samples path column1 column2 ...". If no
                        columns are supplied, we copy all of them
  --column-map COLUMN_MAP COLUMN_MAP COLUMN_MAP
                        map the column names from one of --samples into a new
                        name in the output file. Useful if several of the
                        --samples have the same column names. eg: "--column-
                        map path old_column new_column"
  -v, --verbose
  -V, --Verbose
''')

    return argdict2str(**args)

#-------------------------------------------------

### DAG writing logic

def dag_draw_and_integrate(*args, **kwargs):
    """generate realizations of synthetic EOS via draw-gpr and integrate-phi
    """
    raise NotImplementedError

def dag_weigh_eos(*args, **kwargs):
    """weigh reailizations of syntehtic EOS via draw-samples, weigh-samples, marginalize-samples, and collate-samples
    """
    raise NotImplementedError

def dag_draw_and_weigh_eos(*args, **kwargs):
    """generate realizations of synthetic EOS via draw-gpr and integrate-phi and then weigh reailizations of syntehtic EOS via draw-samples, weigh-samples, marginalize-samples, and collate-samples
    """
    raise NotImplementedError
