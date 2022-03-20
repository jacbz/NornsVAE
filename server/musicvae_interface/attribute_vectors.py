from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import pickle

import numpy as np
import tensorflow.compat.v1 as tf
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import configs

from musicvae_interface.configs import ATTRIBUTES

NUMBER_OF_SAMPLES = 370000
logging = tf.logging
FLAGS = {
    'checkpoint_file': 'ckpt',
    'config': 'cat-drums_1bar_8class',
    'mode': 'sample', # sample or interpolate
    'num_outputs': NUMBER_OF_SAMPLES,
    'max_batch_size': 8192,
    'temperature': 0.5, # The randomness of the decoding process
    'log': 'WARN' # DEBUG, INFO, WARN, ERROR, or FATAL
}

def attribute_string(values):
  return ", ".join([attr + '{0:+}'.format(int(val)) for attr, val in zip(ATTRIBUTES, values)])

def generate():
  print('Generating')
  z = np.empty([0, config.hparams.z_size]).astype(np.float32)
  results = []
  batch_size = FLAGS['max_batch_size']
  batches = math.ceil(NUMBER_OF_SAMPLES / batch_size)

  for i in range(math.ceil(NUMBER_OF_SAMPLES / batch_size)):
    print(f'Batch {i+1} of {batches}')
    current_z = np.random.randn(batch_size, config.hparams.z_size).astype(np.float32)
    results += model.decode(
      length=config.hparams.max_seq_len,
      z=current_z,
      temperature=FLAGS['temperature'])
    z = np.append(z, current_z, 0)
  return z, results

def measure(z, samples):
  print('Measuring')
  attribute_vectors = {}

  num_samples = len(samples)
  for attr in ATTRIBUTES:
    vals = np.empty(num_samples)
    for i, sample in enumerate(samples):
      vals[i] = measure_metric(sample, attr)

    # calculate quartiles
    quartiles = np.percentile(vals, [25, 50, 75])

    top_quartile_mask = vals >= quartiles[2]
    bottom_quartile_mask = vals <= quartiles[0]

    # masked select
    top_quartile_z = z[~top_quartile_mask,:]
    bottom_quartile_z = z[~bottom_quartile_mask,:]

    top_quartile_avg = np.average(top_quartile_z, axis=0)
    bottom_quartile_avg = np.average(bottom_quartile_z, axis=0)

    attribute_vec = bottom_quartile_avg - top_quartile_avg
    attribute_vectors[attr] = attribute_vec
  return attribute_vectors

def measure_metric(sequence, metric):
  if metric == 'DSTY':
    return len(sequence.notes)
  if metric == 'BD':
    return sum(note.pitch == 36 for note in sequence.notes)
  if metric == 'SD':
    return sum(note.pitch == 38 for note in sequence.notes)
  if metric == 'HH':
    return sum(note.pitch == 42 or note.pitch == 46 for note in sequence.notes)
  if metric == 'TO':
    return sum(note.pitch == 45 or note.pitch == 48 or note.pitch == 50 for note in sequence.notes)
  if metric == 'CY':
    return sum(note.pitch == 49 or note.pitch == 51 for note in sequence.notes)
  if metric == 'AVG_ITVL':
    pitches = [note.pitch for note in sequence.notes]
    intervals = [abs(t - s) for s, t in zip(pitches, pitches[1:])]
    if len(intervals) == 0:
      return 0
    return np.mean(intervals)


if __name__ == '__main__':
  config = configs.CONFIG_MAP[FLAGS['config']]
  config.data_converter.max_tensors_per_item = None
  logging.info('Loading model...')
  checkpoint_file = os.path.expanduser(FLAGS['checkpoint_file'])
  model = TrainedModel(
    config, batch_size=FLAGS['max_batch_size'],
    checkpoint_dir_or_path=checkpoint_file)

  prefix = 'drums'

  z, samples = generate()
  with open(f'{prefix}_samples.p', 'wb') as handle:
      pickle.dump((z, samples), handle)

  # print('Loading pickle, ~60 seconds')
  # with open(f'{prefix}_samples.p', 'rb') as handle:
  #   z, samples = pickle.load(handle)

  attribute_vectors = measure(z, samples)
  with open(f'{prefix}_attribute_vectors.p', 'wb') as handle:
    pickle.dump(attribute_vectors, handle)

  # PCA
  # pca = PCA(n_components=2)
  # pca_model = pca.fit(z)
  # with open('pca_model.p', 'wb') as handle:
  #   pickle.dump(pca_model, handle)