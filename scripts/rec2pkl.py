# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import logging
import pickle
import nems
import nems.initializers
import nems.priors
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.analysis.api
import nems.utils
import nems.uri
import nems.recording as recording
from nems.signal import RasterizedSignal
from nems.fitters.api import scipy_minimize
from nems.plots.recording_browser import browse_recording

# ----------------------------------------------------------------------------
# CONFIGURATION

logging.basicConfig(level=logging.INFO)

# figure out data and results paths:
nems_dir = os.path.abspath(os.path.dirname(recording.__file__) + '/..')
signals_dir = nems_dir + '/recordings'
modelspecs_dir = nems_dir + '/modelspecs'

# ----------------------------------------------------------------------------
# DATA LOADING
#
# GOAL: Get your data loaded into memory as a Recording object

logging.info('Loading data...')

# Method #1: Load the data from a local directory
# download demo data if necessary:
recording.get_demo_recordings(signals_dir, name="TAR010c-18-1.tgz")

# load into a recording object
rec = recording.load_recording(signals_dir + "/TAR010c-18-1.tgz")

# ----------------------------------------------------------------------------
# DATA PREPROCESSING
#
# GOAL: Split your data into estimation and validation sets so that you can
#       know when your model exhibits overfitting.

logging.info('Splitting into estimation and validation data sets...')

# Method #1: Find which stimuli have the most reps, use those for val
est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')

# Optional: Take nanmean of ALL occurrences of all signals
est = preproc.average_away_epoch_occurrences(est, epoch_regex='^STIM_').apply_mask()
val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_').apply_mask()

# aside - generate datasets from scratch
X_est = est['stim'].as_continuous()
Y_est = est['resp'].as_continuous()
X_val = val['stim'].as_continuous()
Y_val = val['resp'].as_continuous()
fs = est['resp'].fs
recname = 'demo'
cellid="TAR010c-18-1"

pkl_file=signals_dir + "/TAR010c-18-1.pkl"

with open(pkl_file, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([cellid, recname, fs, X_est, Y_est, X_val, Y_val], f)



# Getting back the objects:
with open(pkl_file, 'rb') as f:  # Python 3: open(..., 'rb')
    cellid, recname, fs, X_est, Y_est, X_val, Y_val = pickle.load(f)




epochs = est['resp'].epochs
stimchans = [str(x) for x in range(X_est.shape[0])]
# borrowed from recording.load_recording_from_arrays

# est recording
resp = RasterizedSignal(fs, Y_est, 'resp', recname, chans=[cellid])
stim = RasterizedSignal(fs, X_est, 'stim', recname, chans=stimchans)
signals = {'resp': resp, 'stim': stim}
est = recording.Recording(signals)

# val recording
resp = RasterizedSignal(fs, Y_val, 'resp', recname, chans=[cellid])
stim = RasterizedSignal(fs, X_val, 'stim', recname, chans=stimchans)
signals = {'resp': resp, 'stim': stim}
val = recording.Recording(signals)


browse_recording(est, signals=['stim', 'resp'], cellid=cellid)



# Method #1: Split based on time, where the first 80% is estimation data and
#            the last, last 20% is validation data.
# est, val = rec.split_at_time(0.8)

# Method #2: Split based on repetition number, rounded to the nearest rep.
# est, val = rec.split_at_rep(0.8)

# Method #3: Use the whole data set! (Usually for doing n-fold cross-val)
# est = rec
# val = rec


# ----------------------------------------------------------------------------
# INITIALIZE MODELSPEC
#
# GOAL: Define the model that you wish to test

logging.info('Initializing modelspec(s)...')

# Method #1: create from "shorthand" keyword string
# very simple linear model
modelspec_name='wc.18x2.g-fir.2x15-lvl.1'

# Method #1b: constrain spectral tuning to be gaussian, add static output NL
#modelspec_name='wc.18x2.g-fir.2x15-lvl.1-dexp.1'

modelspec = nems.initializers.from_keywords(modelspec_name)

# Method #2: Generate modelspec directly
# TODO: implement this

# record some meta data for display and saving
modelspec[0]['meta'] = {'cellid': 'TAR010c-18-1', 'batch': 271,
                        'modelname': modelspec_name}


# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

logging.info('Fitting modelspec(s)...')

# Option 1: Use gradient descent on whole data set(Fast)
# modelspecs = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

# Option 2: quick fit linear part first, then fit full nonlinear model
modelspec = nems.initializers.prefit_to_target(
        est, modelspec, nems.analysis.api.fit_basic,
        target_module='levelshift',
        fitter=scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}})
modelspecs = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

# ----------------------------------------------------------------------------
# GENERATE SUMMARY STATISTICS

logging.info('Generating summary statistics...')

# generate predictions
est, val = nems.analysis.api.generate_prediction(est, val, modelspecs)

# evaluate prediction accuracy
modelspecs = nems.analysis.api.standard_correlation(est, val, modelspecs, rec)

logging.info("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
        modelspecs[0][0]['meta']['r_fit'],
        modelspecs[0][0]['meta']['r_test']))

# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

# GOAL: Save your results to disk. (BEFORE you screw it up trying to plot!)

# logging.info('Saving Results...')
# ms.save_modelspecs(modelspecs_dir, modelspecs)


# ----------------------------------------------------------------------------
# GENERATE PLOTS
#
# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

logging.info('Generating summary plot...')

# Generate a summary plot
context = {'val': val, 'modelspecs': modelspecs, 'est': est, 'rec': rec}
fig = nplt.quickplot(context)

# Optional: Save your figure
# fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)


# ----------------------------------------------------------------------------
# SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.

# TODO
