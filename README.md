# universality

An investigation into the application of universal, EOS-independent relations in Gravitational Wave astronomy.

The basic idea is to use KDE sampling to approximate distributions from a finite set of samples. We provide tools to determine the optimal bandwidth for the KDE algorithm along with tools to both visualize and apply the resulting KDE to sets of samples, computing things like marginal likelihoods or effective priors.

### executables

These are presented in roughly the order they might be used in a typical workflow for KDE-based sampling/inference

  * simulate_samples
    * Generates CSV files with simulated samples. 
    * Samples may have an arbitrary number of columns, and each column is drawn independently from separate truncated Gaussian distributions. 
      * Users must specify each column's name, the mean of the Gaussian, the standard deviation of the Gaussian, and the minimum/maximum values allowed.

  * investigate_bandwidth
    * Estimates the cross-validation likelihood for Kernel Density Estimates (KDEs) derived from a set of samples using a specific bandwidth (standard deviation).
    * This allows users to determine which bandwidth is "best" when representing their samples with a KDE.
    * Users specify which columns they want to include from a CSV file along with a list of bandwidths. Note, the KDE is performed with "whitened" data, and the bandwidths specified correspond to scales in the whitened data, *not* physical scales associated with the values actually contained in the CSV file.

  * corner_samples
    * A simple wrapper around corner.py for visualization purposes.

  * overlay_samples
    * A simple wrapper around corner.py that allows users to overlay 2 distributions on top of one another for visualization purposes.

  * weigh_samples
    * Estimates a KDE based on a set of source samples and evaluates that KDE at a set of target samples. This can be used to compute the marginal likelihood and/or effective prior as needed.
    * Users must specify the bandwidth used in the KDE and the columns used within the KDE. Note that both source and target samples must share the columns used in the KDE but can have as many or few other parameters as they want.

  * estimate_evidence
    * Computes a rudimentary estimate of the evidence for a model from a set of samples from a prior that have been weighted with a marginal likelihood in some way (e.g.: via weigh_samples).

The following are presented in roughly the order in which they'd be used to generate mock EOS drawn from a process over phi=log(de/dp-1)

  * gpr_phi
    * Uses Gaussian Process Regression to infer a process over `phi=log(de/dp-1)`. 
    * `phi` in `[-inf, +inf]` is allowed for all causal equations of state, and therefore any curve we can possibly draw for `phi` will correspond to a causal EOS.
    * This is done by first using a Gaussian Process to "resample" tabulated EOS and generate processes for each over `phi`. A separate Gaussian Process is used to model the theoretical modeling uncertainty between different tabulated EOS.

  * draw_foo
    * a general purpose executable that can draw realizations of a process from a standard output file. This should be able to ingest the output pkl file from gpr_phi as is.
    * produces a CSV file.

  * integrate_phi
    * a simple numerical integration to convert the realization of the process over phi (stored in a CSV file) into a corresponding curve for energy_density(pressure), which is written to disk as a CSV file.
    * We note that no special tricks are taken here to improve numerical accuracy of the integration. As long as the tabulated spacing is small compared to the length-scale assumed in the Gaussian Process, a simple trapazoidal rule should be sufficiently accurate. Generating a finely spaced grid of evaluation points within the Gaussian Process Regression is also not computationally expensive; that calculation is dominated by a matrix inversion which scales with the number of observed data points.

If one wants to simply draw from a process directly on energy_density instead of phi, they can do the following

  * gpr_energy_density
    * perform a Gaussian Process Regression directly on log(energy_density) vs log(pressure). 
    * like gpr_phi, this is done by first resampling each tabulated EOS and then performing an inference over the collection of resampled EOSs.

  * draw_foo
    * see the description above. The caller will need to provide slightly different syntax, but a CSV file will be produced as desired.
