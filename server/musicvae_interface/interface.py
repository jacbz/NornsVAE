from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import pickle
import timeit
import numpy as np
import tensorflow.compat.v1 as tf
from note_seq import quantize_note_sequence
from sklearn.decomposition import PCA

from magenta.models.music_vae import TrainedModel
from .attribute_vectors import ATTRIBUTES, attribute_string
from .configs import MUSICVAE_CONFIG, MAX_SEQ_LENGTH, ATTR_STEP_SIZE, ATTR_MULTIPLIERS, INTERPOLATION_STEPS, \
    DRUM_MAP_INVERTED, DRUM_MAP, PCA_CLIP, SCREEN_HEIGHT, SCREEN_WIDTH

logging = tf.logging
FLAGS = {
    # 'checkpoint_file': '../Test/cat-drums_2bar_small.hikl.ckpt',
    'config': 'cat-drums_1bar_8class',
    'mode': 'sample',  # sample or interpolate
    'batch_size': 128,
    'temperature': 0.5,  # The randomness of the decoding process
    'log': 'ERROR'  # DEBUG, INFO, WARN, ERROR, or FATAL
}


class Interface():
    def __init__(self, assets_folder):
        self.config = MUSICVAE_CONFIG
        self.config.data_converter.max_tensors_per_item = None
        self.config.hparams.max_seq_len = MAX_SEQ_LENGTH
        logging.info('Loading model...')
        checkpoint_file = os.path.expanduser(f"{assets_folder}/checkpoint")
        self.model = TrainedModel(
            self.config, batch_size=FLAGS['batch_size'],
            checkpoint_dir_or_path=checkpoint_file)

        with open(f'{assets_folder}/drums_note_seq.p', 'rb') as handle:
            self.base_note_seq = pickle.load(handle)

        with open(f'{assets_folder}/drums_attribute_vectors.p', 'rb') as handle:
            self.attribute_vectors = pickle.load(handle)

        with open(f'{assets_folder}/drums_pca_model.p', 'rb') as handle:
            self.pca_model = pickle.load(handle)

        self.z_memory = {}
        self.lookahead_memory = {}

        self.seq1hash = None
        self.seq2hash = None
        self.current_attr_values = None
        self.current_attr = ATTRIBUTES[0]
        self.init()

    def init(self):
        initial_sequences = self.sample(2)
        self.seq1hash = initial_sequences[0]['hash']
        self.seq2hash = initial_sequences[1]['hash']

    def lookahead(self, attr_values=None, attribute=None):
        z1 = np.copy(self.z_memory[self.seq1hash])
        z2 = np.copy(self.z_memory[self.seq2hash])
        if attr_values is None:
            attr_values = [0] * len(ATTRIBUTES)
        if attribute is not None:
            self.current_attr = attribute
        else:
            attribute = self.current_attr

        # apply attribute vectors, except for the one we are calculating now
        for i, attr in enumerate(ATTRIBUTES):
            if attribute != attr:
                z1 += self.attribute_vectors[attr] * attr_values[i] * ATTR_STEP_SIZE
                z2 += self.attribute_vectors[attr] * attr_values[i] * ATTR_STEP_SIZE

        z = []
        z_ids = []
        attr_i = ATTRIBUTES.index(attribute)
        attribute_vector = self.attribute_vectors[attribute]
        for attr_step, m in enumerate(ATTR_MULTIPLIERS):
            # for interpolation_step, t in zip(range(INTERPOLATION_STEPS),
            #                                  np.linspace(z1 + m*attribute_vector, z2 + m*attribute_vector, INTERPOLATION_STEPS)):
            #   z.append(t)
            for interpolation_step, t in zip(range(INTERPOLATION_STEPS), np.linspace(0, 1, INTERPOLATION_STEPS)):
                z.append(self.slerp(z1 + m * attribute_vector, z2 + m * attribute_vector, t))

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

    def replace(self, dict1, dict2):
        note_seq1 = self.dict_to_note_seq(dict1)
        note_seq2 = self.dict_to_note_seq(dict2)
        z = self.encode([note_seq1, note_seq2])
        dict1 = self.note_seq_to_dict(note_seq1, z[0])
        dict2 = self.note_seq_to_dict(note_seq2, z[1])
        self.seq1hash = dict1['hash']
        self.seq2hash = dict2['hash']

        return self.lookahead()

    def encode(self, seq):
        _, z, _ = self.model.encode(seq)
        return z

    def decode(self, z):
        results = self.model.decode(
            length=MAX_SEQ_LENGTH,
            z=z,
            temperature=FLAGS['temperature'])
        return self.quantize_and_convert(results, z)

    def dict_to_note_seq(self, dict):
        sequence = copy.deepcopy(self.base_note_seq)
        sequence.ticks_per_quarter = dict['ticks_per_quarter']
        sequence.total_time = 2.0

        for step in sorted(dict['notes'].keys()):
            for note in dict['notes'][step]:
                seq_note = sequence.notes.add()
                seq_note.pitch = DRUM_MAP_INVERTED[note]

                step_num = float(step) / 8
                duration_num = 1 / 8
                seq_note.start_time = step_num
                seq_note.end_time = step_num + duration_num

                seq_note.velocity = 80
                seq_note.instrument = 9
                seq_note.is_drum = True

        return sequence

    def note_seq_to_dict(self, sequence, z):
        sequence_hash = hash(str(sequence))
        dict = {
            'hash': sequence_hash,
            'ticks_per_quarter': sequence.ticks_per_quarter,
            'notes': {}
        }
        for note in sequence.notes:
            # if note.quantized_start_step >= MAX_SEQ_LENGTH:
            #     continue
            if note.quantized_start_step not in dict['notes']:
                dict['notes'][note.quantized_start_step] = []
            entry = DRUM_MAP[note.pitch]
            dict['notes'][note.quantized_start_step].append(entry)

        self.z_memory[sequence_hash] = z
        # print(f"Generated note sequence with hash {sequence_hash}")
        return dict

    def slerp(self, p0, p1, t):
        """Spherical linear interpolation."""
        omega = np.arccos(
            np.dot(np.squeeze(p0 / np.linalg.norm(p0)),
                   np.squeeze(p1 / np.linalg.norm(p1))))
        so = np.sin(omega)
        return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

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
        results = [quantize_note_sequence(sequence, 4) for sequence in results]

        # convert to dict
        results = [self.note_seq_to_dict(sequence, z[i]) for i, sequence in enumerate(results)]
        return results

    def pca_2d(self, z):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(z)
        # pca_result = self.pca_model.transform(z)

        x = pca_result[:, 0]
        pca_result[:, 0] = (x + PCA_CLIP) / (2 * PCA_CLIP)
        pca_result[:, 0] = pca_result[:, 0] * 0.8 * SCREEN_WIDTH + 0.1 * SCREEN_WIDTH

        y = pca_result[:, 1]
        pca_result[:, 1] = (y + PCA_CLIP) / (2 * PCA_CLIP)
        pca_result[:, 1] = 0.9 * SCREEN_HEIGHT - (pca_result[:, 1] * 0.8 * SCREEN_HEIGHT)
        return pca_result
