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

DRUMS = True
logging = tf.logging
FLAGS = {
    'checkpoint_file': '../Test/cat-drums_2bar_small.lokl.ckpt',
    'config': 'cat-drums_2bar_small' if DRUMS else 'cat-mel_2bar_big',
    'mode': 'sample', # sample or interpolate
    'batch_size': 128,
    'temperature': 0.5, # The randomness of the decoding process
    'log': 'INFO' # DEBUG, INFO, WARN, ERROR, or FATAL
}

DRUM_MAP = {
  36: 8,
  38: 7,
  42: 6,
  46: 5,
  45: 4,
  48: 3,
  50: 2,
  49: 1,
  51: 1
}

INTERPOLATION_STEPS = 11
ATTR_STEPS = 9
ATTR_STEP_SIZE = 0.5
# [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]
ATTR_MULTIPLIERS = [ATTR_STEP_SIZE*x - (math.floor(ATTR_STEPS/2)*ATTR_STEP_SIZE) for x in range(ATTR_STEPS)]

SCREEN_WIDTH = 50
SCREEN_HEIGHT = 50
PCA_CLIP = 20

class Interface():
  def __init__(self):
    self.config = configs.CONFIG_MAP[FLAGS['config']]
    self.config.data_converter.max_tensors_per_item = None
    logging.info('Loading model...')
    checkpoint_file = os.path.expanduser(FLAGS['checkpoint_file'])
    self.model = TrainedModel(
      self.config, batch_size=FLAGS['batch_size'],
      checkpoint_dir_or_path=checkpoint_file)

    prefix = 'drums' if DRUMS else 'mel'
    with open(f'{prefix}_note_seq.p', 'rb') as handle:
      self.base_note_seq = pickle.load(handle)

    with open(f'{prefix}_attribute_vectors.p', 'rb') as handle:
      self.attribute_vectors = pickle.load(handle)

    with open(f'{prefix}_pca_model.p', 'rb') as handle:
      self.pca_model = pickle.load(handle)

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
    z1 = self.z_memory[self.seq1]
    z2 = self.z_memory[self.seq2]

    # apply attribute vectors, expect for the one we are calculating now
    for i, attr in enumerate(METRICS):
      if attribute != attr:
        z1 += self.attribute_vectors[attr] * attr_values[i] * ATTR_STEP_SIZE
        z2 += self.attribute_vectors[attr] * attr_values[i] * ATTR_STEP_SIZE

    z = []
    z_ids = []
    attr_i = METRICS.index(attribute)
    attribute_vector = self.attribute_vectors[attribute]
    for attr_step, m in enumerate(ATTR_MULTIPLIERS):
      # for interpolation_step, t in zip(range(INTERPOLATION_STEPS),
      #                                  np.linspace(z1 + m*attribute_vector, z2 + m*attribute_vector, INTERPOLATION_STEPS)):
      #   z.append(t)
      for interpolation_step, t in zip(range(INTERPOLATION_STEPS), np.linspace(0, 1, INTERPOLATION_STEPS)):
        z.append(_slerp(z1 + m*attribute_vector, z2 + m*attribute_vector, t))

        attr_v = attr_values.copy()
        attr_v[attr_i] = m / ATTR_STEP_SIZE
        z_ids.append((interpolation_step, attribute_string(attr_v)))

    results = self.decode(z)

    pca = self.pca_2d(z)

    output_map = {}
    for i, (interpolation_step, attr_id) in enumerate(z_ids):
      if interpolation_step not in output_map:
        output_map[interpolation_step] = {}

      results[i]['x'] = round(pca[i][0])
      results[i]['y'] = round(pca[i][1])
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
      'notes': {}
    }
    for note in sequence.notes._values:
      if note.quantized_start_step not in dict['notes']:
        dict['notes'][note.quantized_start_step] = []
      dict['notes'][note.quantized_start_step].append({
          'pitch': DRUM_MAP[note.pitch] if DRUMS else note.pitch,
          'duration': note.quantized_end_step - note.quantized_start_step
        })

    self.z_memory[sequence_hash] = z
    # print(f"Generated note sequence with hash {sequence_hash}")
    return dict

  def sample(self, n):
    logging.info('Sampling...')
    start = timeit.default_timer()

    z = np.random.randn(n, self.config.hparams.z_size).astype(np.float32)
    results = self.decode(z)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # self.save_as_midi(results, 'sample')

    return results

  def quantize_and_convert(self, results, z):
    results = [note_seq.quantize_note_sequence(sequence, 4) for sequence in results]

    # convert to dict
    results = [self.note_seq_to_dict(sequence, z[i]) for i, sequence in enumerate(results)]
    return results

  def save_as_midi(self, results, mode):
    for i, ns in enumerate(results):
      note_seq.sequence_proto_to_midi_file(ns, f'{mode}_{i}.mid')

  def pca_2d(self, z):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(z)
    # pca_result = self.pca_model.transform(z)

    x = pca_result[:, 0]
    pca_result[:, 0] = (x + PCA_CLIP) / (2*PCA_CLIP)
    pca_result[:, 0] = pca_result[:, 0] * 0.8 * SCREEN_WIDTH + 0.1 * SCREEN_WIDTH

    y =  pca_result[:, 1]
    pca_result[:, 1] = (y + PCA_CLIP) / (2*PCA_CLIP)
    pca_result[:, 1] = 0.9 * SCREEN_HEIGHT - (pca_result[:, 1] * 0.8 * SCREEN_HEIGHT)

    self.visualize(pca_result)
    return pca_result

  def visualize(self, pca_result):
    pca_df = pd.DataFrame(columns=['pca1', 'pca2'])
    pca_df['pca1'] = pca_result[:, 0]
    pca_df['pca2'] = pca_result[:, 1]
    x = pca_df[['pca1', 'pca2']].values

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.scatter(x[:,0], x[:,1], lw=0, s=10)
    ax.axis('off')
    ax.axis('tight')

    plt.show()