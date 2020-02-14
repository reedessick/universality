"""a module that will help us write SUBs and DAGs for distributed workflows
"""
__author__ = "reed.essick@gmail.com"

#-------------------------------------------------

import os
from distutils.spawn import find_executable
from getpass import getuser

from . import utils
from . import eos

#-------------------------------------------------

DEFAULT_UNIVERSE = 'vanilla'

#--- workflow defaults

DEFAULT_START = 0
DEFAULT_NUM_PER_DIRECTORY = utils.DEFAULT_NUM_PER_DIRECTORY
DEFAULT_OUTDIR = utils.DEFAULT_OUTDIR
DEFAULT_TAG = utils.DEFAULT_TAG

#-------------------------------------------------

### utilities for parsing config files

#def eval_kwargs(**kwargs):
#    """a utility to help when parsing classads from INI files
#    """
#    for key, value in kwargs.items():
#        try:
#            value = eval(value)
#        except (SyntaxError, NameError):
#            value = value.strip().split()
#            try:
#                value = [eval(_) for _ in value]
#            except (SyntaxError, NameError):
#                pass
#            finally:
#                if len(value)==1:
#                    value = value[0]
#        kwargs[key] = value
#    return kwargs
#
#def classads2dict(condor_classads):
#    """a utility to help when parsing classads from INI files
#    """
#    return dict((condor_classads[2*i], str(condor_classads[2*i+1])) for i in xrange(len(condor_classads)/2))

#-------------------------------------------------

def samples_path(prefix='samples', directory=DEFAULT_OUTDIR, tag=DEFAULT_TAG):
    if tag:
        tag = "_"+tag
    return os.path.join(directory, '%s%s.csv'%(prefix, tag))

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

def args2str(**argdict):
    tmp = '%s="%s"'
    return ' '.join(tmp%item for item in argdict.items())

#------------------------

def sub_draw_gpr(path, **classads):
    """write the sub file for draw-gpr into path
    """
    classads = basic_classads(which('draw-gpr'), path2logdir(path), **classads)
    classads['arguments'] = '$(hdf5path) --start $(start) --num-draws $(num_draws) --num-per-directory $(num_per_directory) --output-dir $(outdir) $(tag) --verbose'
    write_sub(path, **classads)

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

    return args2str(**args)

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

    ### predict filenames from draw-gpr
    tag = '_'+tag if tag else ''
    directory = utils.draw2directory(outdir=outdir, num_per_directory=num_per_directory)
    phi = utils.draw2path(outdir=directory, prefix='draw-gpr', tag=tag)
    eos = utils.draw2path(outdir=directory, prefix='draw-eos', tag=tag)

    phipaths = []
    outpaths = []
    for i in range(start, start+num_per_directory):
        fmt = utils.draw2fmt(i, num_per_directory=num_per_directory)
        phipath = phi%fmt
        eospath = eos%fmt
        phipaths.append(phipath)
        outpaths.append((phipath, eospath))

    args['phipaths'] = ' '.join(phipaths)
    args['outpaths'] = ' '.join('--outpath %s %s'%tup in zip(phipaths, oupaths))
    args['reference_pressurec2'] = "%.9e"%reference_pressurec2
    args['sigma_logpressurec2'] = "%.9e"%sigma_logpressurec2
    args['stitch_below_reference_pressure'] = '--stitch-below-reference-pressure' if stitch_below_reference_pressure else ''
    args['crust'] = crust

    return args2str(**args)

#---

def sub_draw_samples(path, **classads):
    """write the sub file for draw-samples into path
    """
    classads = basic_classads(which('draw-samples'), path2logdir(path), **classads)
    classads['arguments'] = ''

    raise NotImplementedError('''set up args for draw-samples''')

    write_sub(path, **classads)

def args_draw_samples(
        directory=DEFAULT_OUTDIR,
        tag=DEFAULT_TAG,
        *args,
        **kwargs
    ):
    """format the ARGS string for draw-samples nodes in a DAG
    """
    args = dict()

    raise NotImplementedError('''set up args for draw-samples
predict name for samples file (target for weigh-samples) based on outdir, tag
''')

    args['outpath'] = samples_path(prefix='draw-samples', directory=directory, tag=tag)

    return args2str(**args)

#---

def sub_weigh_samples(path, **classads):
    """write the sub file for weigh-samples into path
    """
    classads = basic_classads(which('weigh-samples'), path2logdir(path), **classads)
    classads['arguments'] = '$(source) $(target) $(output) $(columns) $(logcolumns) $(weight_column) $(weight_column_is_log) $(invert_weight_column) $(column_range) $(reflect) $(prune) $(output_weight_column) $(do_not_log_output_weights) $(bandwidth) --verbose'
    write_sub(path, **classads)

def args_weigh_samples(
        source,
        target,
        columns,
        logcolumns=[],
        weight_columns=[],
        weight_column_is_log=[],
        column_range=dict(),
        reflect=False,
        prune=False,
        output_weight_column=utils.DEFAULT_WEIGHT_COLUMN,
        do_not_log_output_weights=False,
        bandwidth=utils.DEFAULT_BANDWIDTH,
        num_per_directory=DEFAULT_NUM_PER_DIRECTORY,
        directory=DEFAULT_OUTDIR,
        tag=DEFAULT_TAG,
    ):
    """format the ARGS string for weigh-samples nodes in a DAG
    """
    args = dict()

    ### I/O routing
    args['source'] = source

    args['target'] = samples_path(prefix='draw-samples', directory=directory, tag=tag)
    args['output'] = samples_path(prefix='weigh-samples', directory=directory, tag=tag)

    args['output_weight_column'] = output_weight_column
    args['do_not_log_output_weights'] = '--do-not-log-output-weights' if do_not_log_output_weights else ''

    ### KDE model
    args['bandwidth'] = '%.9e'%bandwidth

    # which columns to use in KDE model
    args['columns'] = ' '.join(columns)
    for col in logcolumns:
        assert col in columns, 'specifying --logcolumn for unknown column: '+col
    args['logcolumns'] = ' '.join('--logcolumn '+_ for _ in logcolumns)

    args['column_range'] = ' '.join('--column-range %s %.9e %.9e'%(col, m, M) for col, (m, M) in column_range.items())

    args['reflect'] = '--reflect' if reflect else ''
    args['prune'] = '--prune' if prune else ''

    # weights for KDE model
    args['weight_column'] = ' '.join('--weight-column '+_ for _ in weight_column)
    for col in weight_column_is_log:
        assert col in weight_columns, 'specifying --weight-column-is-log for unknown weight-column: '+col
    args['weight_column_is_log'] = ' '.join('--weight-column-is-log '+_ for _ in weight_column_is_log)
    for col in invert_weight_column:
        assert col in weight_columns, 'specifying --invert-weight-column for unknown weight-column: '+col
    args['invert_weight_column'] = ' '.join('--invert-weight-column '+_ for _ in invert_weight_column)

    return args2str(**args)

#---

def sub_marginalize_samples(path, **classads):
    """write the sub file for marginalize-samples into path
    """
    classads = basic_classads(which('marginalize-samples'), path2logdir(path), **classads)
    classads['arguments'] = '$(samples) $(columns) $(weight_column) $(weight_column_is_log) --output-path $(outpath) --verbose'
    write_sub(path, **classads)

def args_marginalize_samples(
        columns=[utils.DEFAULT_UID_COLUMN],
        weight_column=[utils.DEFAULT_WEIGHT_COLUMN],
        weight_column_is_log=[utils.DEFAULT_WEIGHT_COLUMN],
        directory=DEFAULT_OUTDIR,
        tag=DEFAULT_TAG,
    ):
    """format the ARGS string for marginalize-samples nodes in a DAG
    """
    args = dict()

    ### I?O routing
    args['samples'] = samples_path(prefix='weigh-samples', directory=directory, tag=tag)
    args['outpath'] = samples_path(prefix='marginalize-samples', directory=directory, tag=tag)

    args['columns'] = ' '.join(columns)
    args['weight_column'] = ' '.join('--weight-column '+_ for _ in weight_column)
    args['weight_column_is_log'] = ' '.join('--weight-column-is-log '+_ for _ in weight_column_is_log)

    return args2str(**args)

#---

def sub_collate_samples(path, **classads):
    """write the sub file for collate-samples into path
    """
    classads = basic_classads(which('collate-samples'), path2logdir(path), **classads)
    classads['arguments'] = '$(reference_column) $(outpath) $(samples) $(column_map) --Verbose'
    write_sub(path, **classads)

def args_collate_samples(
        samples,
        column_map=None,
        reference_column=utils.DEFAULT_UID_COLUMN,
        outdir=DEFAULT_OUTDIR,
        tag=DEFAULT_TAG,
        *args, **kwargs
    ):
    """format the ARGS string for collate-samples nodes in a DAG
    """
    args = dict()

    ### I/O routing
    args['outpath'] = samples_path(prefix='collate-samples', directory=outdir, tag=tag)

    ### collating arguments
    args['reference_column'] = reference_column
    args['samples'] = ' '.join('--samples %s %s'%(path, ' '.join(columns)) for path, columns in samples.items())
    args['column_map'] = ' '.join('--column-map %s %s %s'%(path, old, new) for path, pairs in column_map.items() for old, new in pairs)

    return args2str(**args)

#---

def sub_concatenate_samples(path, **classads):
    """write the sub file for collate-samples into path
    """
    classads = basic_classads(which('concatenate-samples'), path2logdir(path), **classads)
    classads['arguments'] = '$(inpaths) --outpath $(outpath) $(columns) --verbose'
    write_sub(path, **classads)

def args_concatenate_samples(inpaths, columns, outdir=DEFAULT_OUTDIR, tag=DEFAULT_TAG):
    """format the ARGS string for concatenate-samples nodes in a DAG
    """
    args = dict()

    args['inpaths'] = ' '.join(inpaths)
    args['outpath'] = samples_path(prefix='concatenate-samples', directory=outdir, tag=tag)
    args['columns'] = ' '.join(columns)

    return args2str(**args)

#-------------------------------------------------

### DAG writing logic

def dag_draw_and_integrate(*args, **kwargs):
    """generate realizations of synthetic EOS via draw-gpr and integrate-phi
    """
    raise NotImplementedError

def dag_weigh_eos(*args, **kwargs):
    """weigh reailizations of syntehtic EOS via draw-samples, weigh-samples, marginalize-samples, collate-samples, and concatenate-samples
    """
    raise NotImplementedError

def dag_draw_and_weigh_eos(*args, **kwargs):
    """generate realizations of synthetic EOS via draw-gpr and integrate-phi and then weigh reailizations of syntehtic EOS via draw-samples, weigh-samples, marginalize-samples, collate-samples, and concatenate-samples
    """
    raise NotImplementedError
