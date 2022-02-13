from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import timeit
import pickle
import copy

import note_seq
import numpy as np
import tensorflow.compat.v1 as tf
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import configs
from note_seq.protobuf import music_pb2

from note_seq.protobuf.music_pb2 import NoteSequence
from music_vae.music_vae_generate import _slerp

NUMBER_OF_SAMPLES = 370000
METRICS = [
  'density'
]
logging = tf.logging
FLAGS = {
    'checkpoint_file': '../Test/cat-mel_2bar_big.ckpt',
    'config': 'cat-mel_2bar_big',
    'mode': 'sample', # sample or interpolate
    'num_outputs': NUMBER_OF_SAMPLES,
    'max_batch_size': 8192,
    'temperature': 0.5, # The randomness of the decoding process
    'log': 'INFO' # DEBUG, INFO, WARN, ERROR, or FATAL
}


def generate():
  print('Generating')
  z = np.random.randn(NUMBER_OF_SAMPLES, 512).astype(np.float32)
  results = model.decode(
    length=config.hparams.max_seq_len,
    z=z,
    temperature=FLAGS['temperature'])
  return z, results

def measure(z, samples):
  print('Measuring')
  attribute_vectors = {}

  num_samples = len(samples)
  for metric in METRICS:
    vals = np.empty(num_samples)
    for i, sample in enumerate(samples):
      vals[i] = measure_metric(sample, metric)

    # calculate quartiles
    quartiles = np.percentile(vals, [25, 50, 75])

    top_quartile_mask = vals >= quartiles[2]
    bottom_quartile_mask = vals <= quartiles[0]

    # masked select
    top_quartile_z = z[~top_quartile_mask,:]
    bottom_quartile_z = z[~bottom_quartile_mask,:]

    top_quartile_avg = np.average(top_quartile_z, axis=0)
    bottom_quartile_avg = np.average(bottom_quartile_z, axis=0)

    attribute_vec = top_quartile_avg - bottom_quartile_avg
    attribute_vectors[metric] = attribute_vec
  return attribute_vectors

def measure_metric(sequence, metric):
  if metric == 'density':
    return len(sequence.notes._values)


config = configs.CONFIG_MAP[FLAGS['config']]
config.data_converter.max_tensors_per_item = None
logging.info('Loading model...')
checkpoint_file = os.path.expanduser(FLAGS['checkpoint_file'])
model = TrainedModel(
  config, batch_size=min(FLAGS['max_batch_size'], FLAGS['num_outputs']),
  checkpoint_dir_or_path=checkpoint_file)

# z, results = generate()
# with open('samples.p', 'wb') as handle:
#     pickle.dump((z, results), handle)

print('Loading pickle, ~60 seconds')
with open('samples.p', 'rb') as handle:
  z, samples = pickle.load(handle)

attribute_vectors = measure(z, samples)

x = 1