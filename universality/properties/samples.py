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
