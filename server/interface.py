from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import timeit
import pickle
import copy

import note_seq
import numpy as np
import tensorflow.compat.v1 as tf
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import configs
from magenta.models.music_vae.music_vae_generate import _slerp
from note_seq.protobuf import music_pb2
from note_seq.protobuf.music_pb2 import NoteSequence

import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from attribute_vectors import METRICS, attribute_string

logging = tf.logging
FLAGS = {
    'checkpoint_file': '../Test/cat-mel_2bar_big.ckpt',
    'config': 'cat-mel_2bar_big',
    'mode': 'sample', # sample or interpolate
    'batch_size': 128,
    'temperature': 0.5, # The randomness of the decoding process
    'log': 'INFO' # DEBUG, INFO, WARN, ERROR, or FATAL
}

INTERPOLATION_STEPS = 11
ATTR_STEPS = 9
ATTR_STEP_SIZE = 0.5
# [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]
ATTR_MULTIPLIERS = [ATTR_STEP_SIZE*x - (math.floor(ATTR_STEPS/2)*ATTR_STEP_SIZE) for x in range(ATTR_STEPS)]

class Interface():
  def __init__(self):
    self.config = configs.CONFIG_MAP[FLAGS['config']]
    self.config.data_converter.max_tensors_per_item = None
    logging.info('Loading model...')
    checkpoint_file = os.path.expanduser(FLAGS['checkpoint_file'])
    self.model = TrainedModel(
      self.config, batch_size=FLAGS['batch_size'],
      checkpoint_dir_or_path=checkpoint_file)

    with open('note_seq.p', 'rb') as handle:
      self.base_note_seq = pickle.load(handle)

    with open('attribute_vectors.p', 'rb') as handle:
      self.attribute_vectors = pickle.load(handle)

    self.z_memory = {}
    self.lookahead_memory = {}

    self.seq1 = None
    self.seq2 = None
    self.init()

  def init(self):
    initial_sequences = self.sample(2)
    self.seq1 = initial_sequences[0]['hash']
    self.seq2 = initial_sequences[1]['hash']

  def lookahead(self, attr_values, attribute):
    attribute_vector = self.attribute_vectors[attribute]
    attr_i = METRICS.index(attribute)

    z1 = self.z_memory[self.seq1] - attr_values[attr_i] * attribute_vector
    z2 = self.z_memory[self.seq2] - attr_values[attr_i] * attribute_vector

    z = []
    z_ids = []
    for attr_step, m in enumerate(ATTR_MULTIPLIERS):
      for interpolation_step, t in zip(range(INTERPOLATION_STEPS), np.linspace(0, 1, INTERPOLATION_STEPS)):
        z.append(_slerp(z1 + m*attribute_vector, z2 + m*attribute_vector, t))

        attr_v = attr_values.copy()
        attr_v[attr_i] = m
        z_ids.append((interpolation_step, attribute_string(attr_v, ATTR_STEP_SIZE)))

    self.visualize(z)
    results = self.decode(z)

    output_map = {}
    for i, (interpolation_step, attr_id) in enumerate(z_ids):
      if interpolation_step not in output_map:
        output_map[interpolation_step] = {}
      output_map[interpolation_step][attr_id] = results[i]

    return output_map


  def encode(self, seq):
    _, mu, _ = self.model.encode(seq)
    return mu

  def decode(self, z):
    results = self.model.decode(
      length=self.config.hparams.max_seq_len,
      z=z,
      temperature=FLAGS['temperature'])
    return self.quantize_and_convert(results, z)

  def dict_to_note_seq(self, dict):
    return self.z_memory[dict['hash']]
    # sequence = copy.deepcopy(self.base_note_seq)
    # sequence.ticks_per_quarter = dict['ticks_per_quarter']
    #
    # sequence.notes._values = []
    # for note in dict['notes']:
    #   seq_note = sequence.notes.add()
    #   seq_note.pitch = note['pitch']
    #   seq_note.start_time = note['start_time']
    #   seq_note.end_time = note['end_time']
    #
    # return sequence

  def note_seq_to_dict(self, sequence, z):
    sequence_hash = hash(str(sequence))
    dict = {
      'hash': sequence_hash,
      'ticks_per_quarter': sequence.ticks_per_quarter,
      'notes': {note.quantized_start_step: {
        'pitch': note.pitch,
        'duration': note.quantized_end_step - note.quantized_start_step
      } for note in sequence.notes._values}
    }
    self.z_memory[sequence_hash] = z
    # print(f"Generated note sequence with hash {sequence_hash}")
    return dict

  def sample(self, n):
    logging.info('Sampling...')
    start = timeit.default_timer()

    z = np.random.randn(n, 512).astype(np.float32)
    results = self.decode(z)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # self.save_as_midi(results, 'sample')

    return results

  def sample_and_interpolate(self, n):
    z = np.random.randn(2, 512).astype(np.float32)
    return self._interpolate(z[0], z[1], n)

  def quantize_and_convert(self, results, z):
    results = [note_seq.quantize_note_sequence(sequence, 4) for sequence in results]

    # convert to dict
    results = [self.note_seq_to_dict(sequence, z[i]) for i, sequence in enumerate(results)]
    return results

  def save_as_midi(self, results, mode):
    for i, ns in enumerate(results):
      note_seq.sequence_proto_to_midi_file(ns, f'{mode}_{i}.mid')

  def interpolate_existing(self, hash1, hash2, num_outputs):
    z1 = self.z_memory[hash1]
    z2 = self.z_memory[hash2]
    return self._interpolate(z1, z2, num_outputs)

  def interpolate(self, dict1, dict2, num_outputs):
    seq1 = self.dict_to_note_seq(dict1)
    seq2 = self.dict_to_note_seq(dict2)
    output = self.encode([seq1, seq2])
    return self._interpolate(output[0], output[1], num_outputs)

  def _interpolate(self, z1, z2, num_outputs):
    logging.info('Interpolating...')
    start = timeit.default_timer()

    z = np.array([_slerp(z1, z2, t) for t in np.linspace(0, 1, num_outputs)])
    results = self.decode(z)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # self.save_as_midi(results, 'interpolate')

    return results, z

  def attribute_arithmetics(self, attribute, hash1, num_outputs):
    start = timeit.default_timer()

    z = self.z_memory[hash1]
    half = math.floor(num_outputs/2)
    multipliers = [ATTR_STEP_SIZE * x - (half*ATTR_STEP_SIZE) for x in range(num_outputs)]
    attribute_vector = self.attribute_vectors[attribute]
    z = np.array(z + [m * attribute_vector for m in multipliers])
    self.visualize(z)
    results = self.decode(z)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    return results

  def visualize(self, z):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(z)
    pca_df = pd.DataFrame(columns=['pca1', 'pca2'])
    pca_df['pca1'] = pca_result[:, 0]
    pca_df['pca2'] = pca_result[:, 1]
    top_two_comp = pca_df[['pca1', 'pca2']]

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    x = top_two_comp.values
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    plt.show()