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

logging = tf.logging
FLAGS = {
    'checkpoint_file': '../Test/cat-mel_2bar_big.ckpt',
    'config': 'cat-mel_2bar_big',
    'mode': 'sample', # sample or interpolate
    'batch_size': 128,
    'temperature': 0.5, # The randomness of the decoding process
    'log': 'INFO' # DEBUG, INFO, WARN, ERROR, or FATAL
}



class Interface():
  def __init__(self):
    self.config = configs.CONFIG_MAP[FLAGS['config']]
    self.config.data_converter.max_tensors_per_item = None
    logging.info('Loading model...')
    checkpoint_file = os.path.expanduser(FLAGS['checkpoint_file'])
    self.model = TrainedModel(
      self.config, batch_size=FLAGS['batch_size'],
      checkpoint_dir_or_path=checkpoint_file)

    # TODO: only store z vector here
    self.memory = {}

    with open('note_seq.p', 'rb') as handle:
      self.base_note_seq = pickle.load(handle)

    with open('attribute_vectors.p', 'rb') as handle:
      self.attribute_vectors = pickle.load(handle)

  def encode(self, seq):
    _, mu, _ = self.model.encode(seq)
    return mu

  def decode(self, z):
    results = self.model.decode(
      length=self.config.hparams.max_seq_len,
      z=z,
      temperature=FLAGS['temperature'])
    return results

  def dict_to_note_seq(self, dict):
    return self.memory[dict['hash']]
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
      'notes': [{
        'pitch': note.pitch,
        'start_time': note.quantized_start_step,
        'end_time': note.quantized_end_step
      } for note in sequence.notes._values]
    }
    self.memory[sequence_hash] = z
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

    return self.quantize_and_convert(results, z)

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
    z1 = self.memory[hash1]
    z2 = self.memory[hash2]
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

    return self.quantize_and_convert(results, z)

  def attribute_arithmetics(self, attribute, hash1, num_outputs):
    start = timeit.default_timer()

    z = self.memory[hash1]
    step_size = 0.3
    half = math.floor(num_outputs/2)
    multipliers = [step_size * x - (half*step_size) for x in range(num_outputs)]
    attribute_vector = self.attribute_vectors[attribute]
    z = np.array(z + [m * attribute_vector for m in multipliers])
    self.visualize(z)
    results = self.decode(z)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    return self.quantize_and_convert(results, z)

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