# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import logging
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from copy import deepcopy
import matplotlib.pyplot as plt

import nems
#import nems.initializers
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
from nems.gui.recording_browser import browse_recording, browse_context
import nems.cnn as cnn

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CONFIGURATION


# figure out data and results paths:
signals_dir = nems.NEMS_PATH + '/recordings'
modelspecs_dir = nems.NEMS_PATH + '/modelspecs'
recording.get_demo_recordings(signals_dir)

datafile = signals_dir + "/TAR010c-18-1.pkl"

# ----------------------------------------------------------------------------
# LOAD AND FORMAT RECORDING DATA

with open(datafile, 'rb') as f:
        cellid, recname, fs, X, Y, epochs = pickle.load(f)

stimchans = [str(x) for x in range(X.shape[0])]
# borrowed from recording.load_recording_from_arrays

resp = RasterizedSignal(fs, Y, 'resp', recname, epochs=epochs, chans=[cellid])
stim = RasterizedSignal(fs, X, 'stim', recname, epochs=epochs, chans=stimchans)
signals = {'resp': resp, 'stim': stim}
rec = recording.Recording(signals)

#est, val = rec.split_at_time(0.2)
est, val = rec.split_using_epoch_occurrence_counts(epoch_regex="^STIM_")
est = preproc.average_away_epoch_occurrences(est, epoch_regex="^STIM_").apply_mask()
val = preproc.average_away_epoch_occurrences(val, epoch_regex="^STIM_").apply_mask()

sr_Hz=est['resp'].fs
time_win_sec = 0.1

n_feats = est['stim'].shape[0]
n_tps_per_stim = 550
n_stim = int(est['stim'].shape[1]/n_tps_per_stim)
n_resp = 1
feat_dims = [n_stim, n_tps_per_stim, n_feats]
data_dims = [n_stim, n_tps_per_stim, n_resp]
v_feat_dims = [3, n_tps_per_stim, n_feats]
v_data_dims = [3, n_tps_per_stim, n_resp]

#F = np.random.randn(feat_dims[0], feat_dims[1], feat_dims[2])
F = np.reshape(est['stim'].as_continuous().copy().T, feat_dims)
m_stim=np.mean(F, axis=(0, 1), keepdims=True)
s_stim=np.std(F, axis=(0, 1), keepdims=True)
F -= m_stim
F /= s_stim
Fv = np.reshape(val['stim'].as_continuous().copy().T, v_feat_dims)
Fv -= m_stim
Fv /= s_stim


# parameters
P = {}
P['rank'] = 2
P['act'] = 'relu'

layers = []

# convolutional layer
layer = {}
layer['type'] = 'conv'
layer['time_win_sec'] = time_win_sec
layer['act'] = P['act']
layer['n_kern'] = 1
layer['rank'] = P['rank']
layers.append(layer)

#net1_seed = 13
#tf.reset_default_graph()
#net1 = cnn.Net(data_dims, n_feats, sr_Hz, deepcopy(layers), seed=net1_seed, log_dir=modelspecs_dir)
#net1.build()
#D = net1.predict(F)
#
#net1_layer_vals = net1.layer_vals()

D = np.reshape(est['resp'].as_continuous().copy().T, data_dims)
Dv = np.reshape(val['resp'].as_continuous().copy().T, v_data_dims)


# create network
net1_seed = 50
tf.reset_default_graph()
net2 = cnn.Net(data_dims, n_feats, sr_Hz, deepcopy(layers), seed=net1_seed, log_dir=modelspecs_dir)
#net2.optimizer = 'GradientDescent'
#net2.optimizer = 'RMSProp'
net2.build()
net2_layer_init = net2.layer_vals()

train_val_test = np.zeros(data_dims[0])
train_val_test[80:] = 1
net2.train(F, D, max_iter=1000, train_val_test=train_val_test)
net2_layer_vals = net2.layer_vals()

# test model
D_pred = net2.predict(F)
Dv_pred = net2.predict(Fv)

# plot results (init vs. final STRF, pred comp)
plt.figure()

plt.subplot(2, 2, 1)
plt.imshow(np.fliplr(net2_layer_init[0]['W'][:,:,0].T), interpolation='none', origin='lower')
plt.ylabel('time'); plt.xlabel('feature')
plt.title('Init weights')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(np.fliplr(net2_layer_vals[0]['W'][:,:,0].T), interpolation='none', origin='lower')
plt.ylabel('time'); plt.xlabel('feature')
plt.title('Fit weights')
plt.colorbar()

plt.subplot(2, 1, 2)
respidx = 0
plt.plot(Dv[respidx, :, 0])
plt.plot(Dv_pred[respidx, :, 0])
cc=np.corrcoef(Dv.flatten(), Dv_pred.flatten())
plt.title('prediction corr: {:.3f}'.format(cc[0,1]))


"""
net1_seed = 7
tf.reset_default_graph()
net1 = cnn.Net(data_dims, n_feats, sr_Hz, deepcopy(layers), seed=net1_seed, log_dir=modelspecs_dir)
net1.build()
net1_layer_vals = net1.layer_vals()
#D = net1.predict(F)
D = np.reshape(est['resp'].as_continuous().T, data_dims)

net2_seed = 10
tf.reset_default_graph()
net2 = cnn.Net(data_dims, n_feats, sr_Hz, deepcopy(layers), seed=net2_seed, log_dir=modelspecs_dir)
net2.optimizer = 'GradientDescent'

net2.build()
net2_layer_init = net2.layer_vals()

train_val_test = np.zeros(data_dims[0])
train_val_test[80:] = 1
net2.train(F, D, learning_rate=0.5, max_iter=1000, eval_interval=50, batch_size=None,
              train_val_test=train_val_test, early_stopping_steps=5, print_iter=True)

net2_layer_vals = net2.layer_vals()

# display results
plt.figure()

plt.subplot(2, 2, 1)
plt.imshow(net1_layer_vals[0]['W'][:, :, 0], interpolation='none')
plt.ylabel('time'); plt.xlabel('feature')
plt.colorbar()
plt.title('Actual weights 1')

plt.subplot(2, 2, 3)
plt.imshow(net2_layer_init[0]['W'][:, :, 0], interpolation='none')
plt.ylabel('time'); plt.xlabel('feature')
plt.colorbar()
plt.title('Init weights 1')

plt.subplot(2, 2, 4)
plt.imshow(net2_layer_vals[0]['W'][:, :, 0], interpolation='none')
plt.ylabel('time'); plt.xlabel('feature')
plt.title('Fit weights 1')
plt.colorbar()
plt.show()

"""

"""

# ----------------------------------------------------------------------------
# INITIALIZE MODELSPEC
#
# GOAL: Define the model that you wish to test

log.info('Initializing modelspec(s)...')

# Method #1: create from "shorthand" keyword string
# very simple linear model
modelspec_name='wc.18x1.g-fir.1x15-lvl.1'
#modelspec_name='wc.18x2.g-fir.2x15-lvl.1'

# Method #1b: constrain spectral tuning to be gaussian, add static output NL
#modelspec_name='wc.18x2.g-fir.2x15-lvl.1-dexp.1'

# record some meta data for display and saving
meta = {'cellid': cellid, 'batch': 271,
        'modelname': modelspec_name, 'recording': cellid}
modelspec = nems.initializers.from_keywords(modelspec_name, meta=meta)

# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

log.info('Fitting modelspec(s)...')

# quick fit linear part first to avoid local minima
modelspec = nems.initializers.prefit_to_target(
        est, modelspec, nems.analysis.api.fit_basic,
        target_module='levelshift',
        fitter=scipy_minimize,
        fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}})


# then fit full nonlinear model
modelspec = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

# ----------------------------------------------------------------------------
# GENERATE SUMMARY STATISTICS

log.info('Generating summary statistics...')

# generate predictions
est, val = nems.analysis.api.generate_prediction(est, val, modelspec)

# evaluate prediction accuracy
modelspec = nems.analysis.api.standard_correlation(est, val, modelspec)

log.info("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
        modelspec.meta['r_fit'][0],
        modelspec.meta['r_test'][0]))

# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

# uncomment to save model to disk:

# logging.info('Saving Results...')
# ms.save_modelspecs(modelspecs_dir, modelspecs)


# ----------------------------------------------------------------------------
# GENERATE PLOTS
#
# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

log.info('Generating summary plot...')

# Generate a summary plot
context = {'val': val, 'modelspec': modelspec, 'est': est}
fig = nplt.quickplot(context)
fig.show()

# Optional: uncomment to save your figure
# fname = nplt.save_figure(fig, modelspecs=modelspecs, save_dir=modelspecs_dir)

# browse the validation data
#aw = browse_recording(val, signals=['stim', 'pred', 'resp'], cellid=cellid)



# ----------------------------------------------------------------------------
# SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.

# TODO
"""