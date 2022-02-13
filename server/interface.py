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
from note_seq.protobuf import music_pb2

from note_seq.protobuf.music_pb2 import NoteSequence
from music_vae.music_vae_generate import _slerp


logging = tf.logging
FLAGS = {
    'checkpoint_file': '../Test/cat-mel_2bar_big.ckpt',
    'config': 'cat-mel_2bar_big',
    'mode': 'sample', # sample or interpolate
    'num_outputs': 32,
    'max_batch_size': 32,
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
      self.config, batch_size=min(FLAGS['max_batch_size'], FLAGS['num_outputs']),
      checkpoint_dir_or_path=checkpoint_file)

    # TODO: only store z vector here
    self.memory = {}

    with open('note_seq.p', 'rb') as handle:
      self.base_note_seq = pickle.load(handle)

    with open('attribute_vectors.p', 'rb') as handle:
      self.attribute_vectors = pickle.load(handle)

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
    results = self.model.decode(
      length=self.config.hparams.max_seq_len,
      z=z,
      temperature=FLAGS['temperature'])

    # results = self.model.sample(
    #   n=n if n > 0 else 1,
    #   length=self.config.hparams.max_seq_len,
    #   temperature=FLAGS['temperature'])
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
    _, mu, _ = self.model.encode([seq1, seq2])
    return self._interpolate(mu[0], mu[1], num_outputs)

  def _interpolate(self, z1, z2, num_outputs):
    logging.info('Interpolating...')
    start = timeit.default_timer()

    z = np.array([_slerp(z1, z2, t) for t in np.linspace(0, 1, num_outputs)])
    results = self.model.decode(
      length=self.config.hparams.max_seq_len,
      z=z,
      temperature=FLAGS['temperature'])

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # self.save_as_midi(results, 'interpolate')

    return self.quantize_and_convert(results, z)

  def attribute_arithmetics(self, attribute, num_outputs, hash=None):
    start = timeit.default_timer()

    z = self.memory[hash] if hash is not None else np.random.randn(1, 512).astype(np.float32)

    step_size = 0.3
    half = math.floor(num_outputs/2)
    multipliers = [step_size * x - (half*step_size) for x in range(num_outputs)]

    attribute_vector = self.attribute_vectors[attribute]
    z = np.array(z + [m * attribute_vector for m in multipliers])
    results = self.model.decode(
      length=self.config.hparams.max_seq_len,
      z=z,
      temperature=FLAGS['temperature'])

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    self.save_as_midi(results, 'attr')
    return self.quantize_and_convert(results, z)
