__doc__ = "a module housing logic to identify separate stable branches from sequences of solutions to the TOV equations"
__author__ = "reed.essick@ligo.org"

#-------------------------------------------------
# utility functions for processing process directory structures
#-------------------------------------------------

def data2samples(*args, **kwargs):
    raise NotImplementedError

def process2samples(
        data,
        tmp,
        mod,
        xcolumn,
        ycolumns,
        static_x_test=None,
        dynamic_x_test=None,
        verbose=False,
        nearest_neighbor=False,
    ):
    """manages I/O and extracts samples at the specified places
    returns an array that is ordered as follows
        for each ycolumn: for each static x_test
    """
    loadcolumns = [xcolumn] + ycolumns

    if static_x_test is not None:
        Nref = len(static_x_test)
    else:
        Nref = 0

    if dynamic_x_test is not None:
        assert len(dynamic_x_test)==len(data)
        Ndyn = np.shape(dynamic_x_test)[1]
    else:
        Ndyn = 0

    Ntot = Nref+Ndyn
    assert Ntot > 0, 'must provide at least one static_x_test or dynamic_x_test'

    ans = np.empty((len(data), (Nref+Ndyn)*len(ycolumns)), dtype=float)
    for i, eos in enumerate(data):
        path = tmp%{'moddraw':eos//mod, 'draw':eos}
        if verbose:
            print('    '+path)
        d, c = load(path, loadcolumns)

        if nearest_neighbor:
            if Nref > 0:
                truth = np.zeros(len(d), dtype=bool)
                for x in static_x_test:
                    truth[np.argmin(np.abs(d[:,0]-x))] = True

            if Ndyn > 0:
                dyn_truth = np.zeros(len(d), dtype=bool)
                for x in dynamic_x_test[i]:
                    dyn_truth[np.argmin(np.abs(d[:,0]-x))] = True

        if (Nref > 0) and (Ndyn > 0):
            for j, column in enumerate(c[1:]):
                if nearest_neighbor:
                    ans[i,j*Ntot:j*Ntot+Nref] = d[:,1+j][truth]
                    ans[i,j*Ntot+Nref:(j+1)*Ntot] = d[:,1+j][dyn_truth]
                else:
                    ans[i,j*Ntot:j*Ntot+Nref] = np.interp(static_x_test, d[:,0], d[:,1+j])
                    ans[i,j*Ntot+Nref:(j+1)*Ntot] = np.interp(dynamic_x_test[i], d[:,0], d[:,1+j])

        elif Nref > 0:
            for j, column in enumerate(c[1:]):
                if nearest_neighbor:
                    ans[i,j*Nref:(j+1)*Nref] = d[:,1+j][truth]
                else:
                    ans[i,j*Nref:(j+1)*Nref] = np.interp(static_x_test, d[:,0], d[:,1+j])

        else: ### Ndyn > 0
            for j, column in enumerate(c[1:]):
                if nearest_neighbor:
                    ans[i,j*Ndyn:(j+1)*Ndyn] = d[:,1+j][dyn_truth]
                else:
                    ans[i,j*Ndyn:(j+1)*Ndyn] = np.interp(dynamic_x_test[i], d[:,0], d[:,1+j])

    return ans

#-------------------------------------------------

def process2quantiles(
        data,
        tmp,
        mod,
        xcolumn,
        ycolumn,
        x_test,
        quantiles,
        quantile_type='sym',
        x_multiplier=1.,
        y_multiplier=1.,
        weights=None,
        verbose=False,
    ):
    """manages I/O and extracts quantiles at the specified places
    """
    y_test = [] ### keep this as a list because we don't know how many stable branches there are
    w_test = []
    num_points = len(x_test)

    truth = np.empty(num_points, dtype=bool) ### used to extract values

    columns = [xcolumn, ycolumn]
    if weights is None:
        weights = np.ones(len(data), dtype=float) / len(data)

    for eos, weight in zip(data, weights): ### iterate over samples and compute weighted moments
        for eos_path in glob.glob(tmp%{'moddraw':eos//mod, 'draw':eos}):
            if verbose:
                print('    '+eos_path)
            d, _ = load(eos_path, columns)

            d[:,0] *= x_multiplier
            d[:,1] *= y_multiplier

            _y = np.empty(num_points, dtype=float)
            _y[:] = np.nan ### signal that nothing was available at this x-value

            truth[:] = (np.min(d[:,0])<=x_test)*(x_test<=np.max(d[:,0])) ### figure out which x-test values are contained in the data
            _y[truth] = np.interp(x_test[truth], d[:,0], d[:,1]) ### fill those in with interpolated values

            y_test.append( _y ) ### add to the total list
            w_test.append( weight )

    if len(y_test)==0:
        raise RuntimeError('could not find any files matching "%s"'%tmp)

    y_test = np.array(y_test) ### cast to an array
    w_test = np.array(w_test)

    ### compute the quantiles
    Nquantiles = len(quantiles)
    if quantile_type=='hpd':
        Nquantiles *= 2 ### need twice as many indicies for this

    qs = np.empty((Nquantiles, num_points), dtype=float)
    med = np.empty(num_points, dtype=float)

    for i in xrange(num_points):

        _y = y_test[:,i]
        truth = _y==_y
        _y = _y[truth] ### only keep things that are not nan
        _w = w_test[truth]

        if quantile_type=="sym":
            qs[:,i] = stats.quantile(_y, quantiles, weights=_w)     ### compute quantiles

        elif quantile_type=="hpd":

            ### FIXME: the following returns bounds on a contiguous hpd interval, which is not necessarily what we want...

            bounds = stats.samples2crbounds(_y, quantiles, weights=_w) ### get the bounds
            qs[:,i] = np.array(bounds).flatten()

        else:
            raise ValueError('did not understand --quantile-type=%s'%quantile_type)

        med[i] = stats.quantile(_y, [0.5], weights=_w)[0] ### compute median

    return qs, med
