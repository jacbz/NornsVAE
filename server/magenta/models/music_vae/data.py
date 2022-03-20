# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MusicVAE data library."""
import abc
import collections
import copy
import functools

from magenta.pipelines import drum_pipelines
import note_seq
from note_seq import drums_encoder_decoder
from note_seq import sequences_lib
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

PIANO_MIN_MIDI_PITCH = 21
PIANO_MAX_MIDI_PITCH = 108
MIN_MIDI_PITCH = 0
MAX_MIDI_PITCH = 127
MIDI_PITCHES = 128

MAX_INSTRUMENT_NUMBER = 127

MEL_PROGRAMS = range(0, 32)  # piano, chromatic percussion, organ, guitar
BASS_PROGRAMS = range(32, 40)
ELECTRIC_BASS_PROGRAM = 33

# 9 classes: kick, snare, closed_hh, open_hh, low_tom, mid_tom, hi_tom, crash,
# ride
REDUCED_DRUM_PITCH_CLASSES = drums_encoder_decoder.DEFAULT_DRUM_TYPE_PITCHES
# 61 classes: full General MIDI set
FULL_DRUM_PITCH_CLASSES = [
    [p] for p in  # pylint:disable=g-complex-comprehension
    [36, 35, 38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85, 42, 44,
     54, 68, 69, 70, 71, 73, 78, 80, 46, 67, 72, 74, 79, 81, 45, 29, 41, 61, 64,
     84, 48, 47, 60, 63, 77, 86, 87, 50, 30, 43, 62, 76, 83, 49, 55, 57, 58, 51,
     52, 53, 59, 82]
]
ROLAND_DRUM_PITCH_CLASSES = [
    # kick drum
    [36],
    # snare drum
    [38, 37, 40],
    # closed hi-hat
    [42, 22, 44],
    # open hi-hat
    [46, 26],
    # low tom
    [43, 58],
    # mid tom
    [47, 45],
    # high tom
    [50, 48],
    # crash cymbal
    [49, 52, 55, 57],
    # ride cymbal
    [51, 53, 59]
]

OUTPUT_VELOCITY = 80

CHORD_SYMBOL = note_seq.NoteSequence.TextAnnotation.CHORD_SYMBOL


def _maybe_pad_seqs(seqs, dtype, depth):
  """Pads sequences to match the longest and returns as a numpy array."""
  if not len(seqs):  # pylint:disable=g-explicit-length-test,len-as-condition
    return np.zeros((0, 0, depth), dtype)
  lengths = [len(s) for s in seqs]
  if len(set(lengths)) == 1:
    return np.array(seqs, dtype)
  else:
    length = max(lengths)
    return (np.array([np.pad(s, [(0, length - len(s)), (0, 0)], mode='constant')
                      for s in seqs], dtype))


def _extract_instrument(note_sequence, instrument):
  extracted_ns = copy.copy(note_sequence)
  del extracted_ns.notes[:]
  extracted_ns.notes.extend(
      n for n in note_sequence.notes if n.instrument == instrument)
  return extracted_ns


def maybe_sample_items(seq, sample_size, randomize):
  """Samples a seq if `sample_size` is provided and less than seq size."""
  if not sample_size or len(seq) <= sample_size:
    return seq
  if randomize:
    indices = set(np.random.choice(len(seq), size=sample_size, replace=False))
    return [seq[i] for i in indices]
  else:
    return seq[:sample_size]


def combine_converter_tensors(converter_tensors, max_num_tensors=None,
                              randomize_sample=True):
  """Combines multiple `ConverterTensors` into one and samples if required."""
  results = []
  for result in converter_tensors:
    results.extend(zip(*result))
  sampled_results = maybe_sample_items(results, max_num_tensors,
                                       randomize_sample)
  if sampled_results:
    return ConverterTensors(*zip(*sampled_results))
  else:
    return ConverterTensors()


def np_onehot(indices, depth, dtype=np.bool):
  """Converts 1D array of indices to a one-hot 2D array with given depth."""
  onehot_seq = np.zeros((len(indices), depth), dtype=dtype)
  onehot_seq[np.arange(len(indices)), indices] = 1.0
  return onehot_seq


class NoteSequenceAugmenter(object):
  """Class for augmenting NoteSequences.

  Attributes:
    transpose_range: A tuple containing the inclusive, integer range of
        transpose amounts to sample from. If None, no transposition is applied.
    stretch_range: A tuple containing the inclusive, float range of stretch
        amounts to sample from.
  Returns:
    The augmented NoteSequence.
  """

  def __init__(self, transpose_range=None, stretch_range=None):
    self._transpose_range = transpose_range
    self._stretch_range = stretch_range

  def augment(self, note_sequence):
    """Python implementation that augments the NoteSequence.

    Args:
      note_sequence: A NoteSequence proto to be augmented.

    Returns:
      The randomly augmented NoteSequence.
    """
    transpose_min, transpose_max = (
        self._transpose_range if self._transpose_range else (0, 0))
    stretch_min, stretch_max = (
        self._stretch_range if self._stretch_range else (1.0, 1.0))

    return sequences_lib.augment_note_sequence(
        note_sequence,
        stretch_min,
        stretch_max,
        transpose_min,
        transpose_max,
        delete_out_of_range_notes=True)

  def tf_augment(self, note_sequence_scalar):
    """TF op that augments the NoteSequence."""
    def _augment_str(note_sequence_str):
      note_sequence = note_seq.NoteSequence.FromString(
          note_sequence_str.numpy())
      augmented_ns = self.augment(note_sequence)
      return [augmented_ns.SerializeToString()]

    augmented_note_sequence_scalar = tf.py_function(
        _augment_str,
        inp=[note_sequence_scalar],
        Tout=tf.string,
        name='augment')
    augmented_note_sequence_scalar.set_shape(())
    return augmented_note_sequence_scalar


class ConverterTensors(collections.namedtuple(
    'ConverterTensors', ['inputs', 'outputs', 'controls', 'lengths'])):
  """Tuple of tensors output by `to_tensors` method in converters.

  Attributes:
    inputs: Input tensors to feed to the encoder.
    outputs: Output tensors to feed to the decoder.
    controls: (Optional) tensors to use as controls for both encoding and
        decoding.
    lengths: Length of each input/output/control sequence.
  """

  def __new__(cls, inputs=None, outputs=None, controls=None, lengths=None):
    if inputs is None:
      inputs = []
    if outputs is None:
      outputs = []
    if lengths is None:
      lengths = [len(i) for i in inputs]
    if not controls:
      controls = [np.zeros([l, 0]) for l in lengths]
    return super(ConverterTensors, cls).__new__(
        cls, inputs, outputs, controls, lengths)


class BaseNoteSequenceConverter(object):
  """Base class for data converters between items and tensors.

  Inheriting classes must implement the following abstract methods:
    -`to_tensors`
    -`from_tensors`
  """

  __metaclass__ = abc.ABCMeta

  def __init__(self,
               input_depth,
               input_dtype,
               output_depth,
               output_dtype,
               control_depth=0,
               control_dtype=np.bool,
               end_token=None,
               max_tensors_per_notesequence=None,
               length_shape=(),
               presplit_on_time_changes=True):
    """Initializes BaseNoteSequenceConverter.

    Args:
      input_depth: Depth of final dimension of input (encoder) tensors.
      input_dtype: DType of input (encoder) tensors.
      output_depth: Depth of final dimension of output (decoder) tensors.
      output_dtype: DType of output (decoder) tensors.
      control_depth: Depth of final dimension of control tensors, or zero if not
          conditioning on control tensors.
      control_dtype: DType of control tensors.
      end_token: Optional end token.
      max_tensors_per_notesequence: The maximum number of outputs to return for
          each input.
      length_shape: Shape of length returned by `to_tensor`.
      presplit_on_time_changes: Whether to split NoteSequence on time changes
        before converting.
    """
    self._input_depth = input_depth
    self._input_dtype = input_dtype
    self._output_depth = output_depth
    self._output_dtype = output_dtype
    self._control_depth = control_depth
    self._control_dtype = control_dtype
    self._end_token = end_token
    self._max_tensors_per_input = max_tensors_per_notesequence
    self._str_to_item_fn = note_seq.NoteSequence.FromString
    self._mode = None
    self._length_shape = length_shape
    self._presplit_on_time_changes = presplit_on_time_changes

  def set_mode(self, mode):
    if mode not in ['train', 'eval', 'infer']:
      raise ValueError('Invalid mode: %s' % mode)
    self._mode = mode

  @property
  def is_training(self):
    return self._mode == 'train'

  @property
  def is_inferring(self):
    return self._mode == 'infer'

  @property
  def str_to_item_fn(self):
    return self._str_to_item_fn

  @property
  def max_tensors_per_notesequence(self):
    return self._max_tensors_per_input

  @max_tensors_per_notesequence.setter
  def max_tensors_per_notesequence(self, value):
    self._max_tensors_per_input = value

  @property
  def end_token(self):
    """End token, or None."""
    return self._end_token

  @property
  def input_depth(self):
    """Dimension of inputs (to encoder) at each timestep of the sequence."""
    return self._input_depth

  @property
  def input_dtype(self):
    """DType of inputs (to encoder)."""
    return self._input_dtype

  @property
  def output_depth(self):
    """Dimension of outputs (from decoder) at each timestep of the sequence."""
    return self._output_depth

  @property
  def output_dtype(self):
    """DType of outputs (from decoder)."""
    return self._output_dtype

  @property
  def control_depth(self):
    """Dimension of control inputs at each timestep of the sequence."""
    return self._control_depth

  @property
  def control_dtype(self):
    """DType of control inputs."""
    return self._control_dtype

  @property
  def length_shape(self):
    """Shape of length returned by `to_tensor`."""
    return self._length_shape

  @abc.abstractmethod
  def to_tensors(self, item):
    """Python method that converts `item` into list of `ConverterTensors`."""
    pass

  @abc.abstractmethod
  def from_tensors(self, samples, controls=None):
    """Python method that decodes model samples into list of items."""
    pass


class DrumsConverter(BaseNoteSequenceConverter):
  """Converter for legacy drums with either pianoroll or one-hot tensors.

  Inputs/outputs are either a "pianoroll"-like encoding of all possible drum
  hits at a given step, or a one-hot encoding of the pianoroll.

  The "roll" input encoding includes a final NOR bit (after the optional end
  token).

  Attributes:
    max_bars: Optional maximum number of bars per extracted drums, before
      slicing.
    slice_bars: Optional size of window to slide over raw Melodies after
      extraction.
    gap_bars: If this many bars or more follow a non-empty drum event, the
      drum track is ended. Disabled when set to 0 or None.
    pitch_classes: A collection of collections, with each sub-collection
      containing the set of pitches representing a single class to group by. By
      default, groups valid drum pitches into 9 different classes.
    add_end_token: Whether or not to add an end token. Recommended to be False
      for fixed-length outputs.
    steps_per_quarter: The number of quantization steps per quarter note.
    quarters_per_bar: The number of quarter notes per bar.
    pad_to_total_time: Pads each input/output tensor to the total time of the
      NoteSequence.
    roll_input: Whether to use a pianoroll-like representation as the input
      instead of a one-hot encoding.
    roll_output: Whether to use a pianoroll-like representation as the output
      instead of a one-hot encoding.
    max_tensors_per_notesequence: The maximum number of outputs to return
      for each NoteSequence.
    presplit_on_time_changes: Whether to split NoteSequence on time changes
      before converting.
  """

  def __init__(self, max_bars=None, slice_bars=None, gap_bars=1.0,
               pitch_classes=None, add_end_token=False, steps_per_quarter=4,
               quarters_per_bar=4, pad_to_total_time=False, roll_input=False,
               roll_output=False, max_tensors_per_notesequence=5,
               presplit_on_time_changes=True):
    self._pitch_classes = pitch_classes or REDUCED_DRUM_PITCH_CLASSES
    self._pitch_class_map = {}
    for i, pitches in enumerate(self._pitch_classes):
      self._pitch_class_map.update({p: i for p in pitches})
    self._steps_per_quarter = steps_per_quarter
    self._steps_per_bar = steps_per_quarter * quarters_per_bar
    self._slice_steps = self._steps_per_bar * slice_bars if slice_bars else None
    self._pad_to_total_time = pad_to_total_time
    self._roll_input = roll_input
    self._roll_output = roll_output

    self._drums_extractor_fn = functools.partial(
        drum_pipelines.extract_drum_tracks,
        min_bars=1,
        gap_bars=gap_bars or float('inf'),
        max_steps_truncate=self._steps_per_bar * max_bars if max_bars else None,
        pad_end=True)

    num_classes = len(self._pitch_classes)

    self._pr_encoder_decoder = note_seq.PianorollEncoderDecoder(
        input_size=num_classes + add_end_token)
    # Use pitch classes as `drum_type_pitches` since we have already done the
    # mapping.
    self._oh_encoder_decoder = note_seq.MultiDrumOneHotEncoding(
        drum_type_pitches=[(i,) for i in range(num_classes)])

    if self._roll_output:
      output_depth = num_classes + add_end_token
    else:
      output_depth = self._oh_encoder_decoder.num_classes + add_end_token

    if self._roll_input:
      input_depth = num_classes + 1 + add_end_token
    else:
      input_depth = self._oh_encoder_decoder.num_classes + add_end_token

    super(DrumsConverter, self).__init__(
        input_depth=input_depth,
        input_dtype=np.bool,
        output_depth=output_depth,
        output_dtype=np.bool,
        end_token=output_depth - 1 if add_end_token else None,
        presplit_on_time_changes=presplit_on_time_changes,
        max_tensors_per_notesequence=max_tensors_per_notesequence)

  def _to_tensors_fn(self, note_sequence):
    """Converts NoteSequence to unique sequences."""
    try:
      quantized_sequence = note_seq.quantize_note_sequence(
          note_sequence, self._steps_per_quarter)
      if (note_seq.steps_per_bar_in_quantized_sequence(quantized_sequence) !=
          self._steps_per_bar):
        return ConverterTensors()
    except (note_seq.BadTimeSignatureError, note_seq.NonIntegerStepsPerBarError,
            note_seq.NegativeTimeError) as e:
      return ConverterTensors()

    new_notes = []
    for n in quantized_sequence.notes:
      if not n.is_drum:
        continue
      if n.pitch not in self._pitch_class_map:
        continue
      n.pitch = self._pitch_class_map[n.pitch]
      new_notes.append(n)
    del quantized_sequence.notes[:]
    quantized_sequence.notes.extend(new_notes)

    event_lists, unused_stats = self._drums_extractor_fn(quantized_sequence)

    if self._pad_to_total_time:
      for e in event_lists:
        e.set_length(len(e) + e.start_step, from_left=True)
        e.set_length(quantized_sequence.total_quantized_steps)
    if self._slice_steps:
      sliced_event_tuples = []
      for l in event_lists:
        for i in range(self._slice_steps, len(l) + 1, self._steps_per_bar):
          sliced_event_tuples.append(tuple(l[i - self._slice_steps: i]))
    else:
      sliced_event_tuples = [tuple(l) for l in event_lists]

    unique_event_tuples = list(set(sliced_event_tuples))
    unique_event_tuples = maybe_sample_items(unique_event_tuples,
                                             self.max_tensors_per_notesequence,
                                             self.is_training)

    rolls = []
    oh_vecs = []
    for t in unique_event_tuples:
      if self._roll_input or self._roll_output:
        if self.end_token is not None:
          t_roll = list(t) + [(self._pr_encoder_decoder.input_size - 1,)]
        else:
          t_roll = t
        rolls.append(np.vstack([
            self._pr_encoder_decoder.events_to_input(t_roll, i).astype(np.bool)
            for i in range(len(t_roll))]))
      if not (self._roll_input and self._roll_output):
        labels = [self._oh_encoder_decoder.encode_event(e) for e in t]
        if self.end_token is not None:
          labels += [self._oh_encoder_decoder.num_classes]
        oh_vecs.append(np_onehot(
            labels,
            self._oh_encoder_decoder.num_classes + (self.end_token is not None),
            np.bool))

    if self._roll_input:
      input_seqs = [
          np.append(roll, np.expand_dims(np.all(roll == 0, axis=1), axis=1),
                    axis=1) for roll in rolls]
    else:
      input_seqs = oh_vecs

    output_seqs = rolls if self._roll_output else oh_vecs

    return ConverterTensors(inputs=input_seqs, outputs=output_seqs)

  def to_tensors(self, item):
    note_sequence = item
    return split_process_and_combine(note_sequence,
                                     self._presplit_on_time_changes,
                                     self.max_tensors_per_notesequence,
                                     self.is_training, self._to_tensors_fn)

  def from_tensors(self, samples, unused_controls=None):
    output_sequences = []
    for s in samples:
      if self._roll_output:
        if self.end_token is not None:
          end_i = np.where(s[:, self.end_token])
          if len(end_i):  # pylint: disable=g-explicit-length-test,len-as-condition
            s = s[:end_i[0]]
        events_list = [frozenset(np.where(e)[0]) for e in s]
      else:
        s = np.argmax(s, axis=-1)
        if self.end_token is not None and self.end_token in s:
          s = s[:s.tolist().index(self.end_token)]
        events_list = [self._oh_encoder_decoder.decode_event(e) for e in s]
      # Map classes to exemplars.
      events_list = [
          frozenset(self._pitch_classes[c][0] for c in e) for e in events_list]
      track = note_seq.DrumTrack(
          events=events_list,
          steps_per_bar=self._steps_per_bar,
          steps_per_quarter=self._steps_per_quarter)
      output_sequences.append(track.to_sequence(velocity=OUTPUT_VELOCITY))
    return output_sequences


def count_examples(examples_path, tfds_name, data_converter,
                   file_reader=tf.python_io.tf_record_iterator):
  """Counts the number of examples produced by the converter from files."""
  def _file_generator():
    filenames = tf.gfile.Glob(examples_path)
    for f in filenames:
      tf.logging.info('Counting examples in %s.', f)
      reader = file_reader(f)
      for item_str in reader:
        yield data_converter.str_to_item_fn(item_str)

  def _tfds_generator():
    ds = tfds.as_numpy(
        tfds.load(tfds_name, split=tfds.Split.VALIDATION, try_gcs=True))
    # TODO(adarob): Generalize to other data types if needed.
    for ex in ds:
      yield note_seq.midi_to_note_sequence(ex['midi'])

  num_examples = 0

  generator = _tfds_generator if tfds_name else _file_generator
  for item in generator():
    tensors = data_converter.to_tensors(item)
    num_examples += len(tensors.inputs)
  tf.logging.info('Total examples: %d', num_examples)
  return num_examples


def split_process_and_combine(note_sequence, split, sample_size, randomize,
                              to_tensors_fn):
  """Splits a `NoteSequence`, processes and combines the `ConverterTensors`.

  Args:
    note_sequence: The `NoteSequence` to split, process and combine.
    split: If True, the given note_sequence is split into multiple based on time
      changes, and the tensor outputs are concatenated.
    sample_size: Outputs are sampled if size exceeds this value.
    randomize: If True, outputs are randomly sampled (this is generally done
      during training).
    to_tensors_fn: A fn that converts a `NoteSequence` to `ConverterTensors`.

  Returns:
    A `ConverterTensors` obj.
  """
  note_sequences = sequences_lib.split_note_sequence_on_time_changes(
      note_sequence) if split else [note_sequence]
  results = []
  for ns in note_sequences:
    tensors = to_tensors_fn(ns)
    sampled_results = maybe_sample_items(
        list(zip(*tensors)), sample_size, randomize)
    if sampled_results:
      results.append(ConverterTensors(*zip(*sampled_results)))
    else:
      results.append(ConverterTensors())
  return combine_converter_tensors(results, sample_size, randomize)


def convert_to_tensors_op(item_scalar, converter):
  """TensorFlow op that converts item into output tensors.

  Sequences will be padded to match the length of the longest.

  Args:
    item_scalar: A scalar of type tf.String containing the raw item to be
      converted to tensors.
    converter: The DataConverter to be used.

  Returns:
    inputs: A Tensor, shaped [num encoded seqs, max(lengths), input_depth],
        containing the padded input encodings.
    outputs: A Tensor, shaped [num encoded seqs, max(lengths), output_depth],
        containing the padded output encodings resulting from the input.
    controls: A Tensor, shaped
        [num encoded seqs, max(lengths), control_depth], containing the padded
        control encodings.
    lengths: A tf.int32 Tensor, shaped [num encoded seqs], containing the
      unpadded lengths of the tensor sequences resulting from the input.
  """

  def _convert_and_pad(item_str):
    item = converter.str_to_item_fn(item_str.numpy())  # pylint:disable=not-callable
    tensors = converter.to_tensors(item)
    inputs = _maybe_pad_seqs(tensors.inputs, converter.input_dtype,
                             converter.input_depth)
    outputs = _maybe_pad_seqs(tensors.outputs, converter.output_dtype,
                              converter.output_depth)
    controls = _maybe_pad_seqs(tensors.controls, converter.control_dtype,
                               converter.control_depth)
    return inputs, outputs, controls, np.array(tensors.lengths, np.int32)

  inputs, outputs, controls, lengths = tf.py_function(
      _convert_and_pad,
      inp=[item_scalar],
      Tout=[
          converter.input_dtype, converter.output_dtype,
          converter.control_dtype, tf.int32
      ],
      name='convert_and_pad')
  inputs.set_shape([None, None, converter.input_depth])
  outputs.set_shape([None, None, converter.output_depth])
  controls.set_shape([None, None, converter.control_depth])
  lengths.set_shape([None] + list(converter.length_shape))
  return inputs, outputs, controls, lengths


def get_dataset(
    config,
    tf_file_reader=tf.data.TFRecordDataset,
    is_training=False,
    cache_dataset=True):
  """Get input tensors from dataset for training or evaluation.

  Args:
    config: A Config object containing dataset information.
    tf_file_reader: The tf.data.Dataset class to use for reading files.
    is_training: Whether or not the dataset is used in training. Determines
      whether dataset is shuffled and repeated, etc.
    cache_dataset: Whether to cache the dataset in memory for improved
      performance.

  Returns:
    A tf.data.Dataset containing input, output, control, and length tensors.

  Raises:
    ValueError: If no files match examples path.
  """
  batch_size = config.hparams.batch_size
  examples_path = (
      config.train_examples_path if is_training else config.eval_examples_path)
  note_sequence_augmenter = (
      config.note_sequence_augmenter if is_training else None)
  data_converter = config.data_converter
  data_converter.set_mode('train' if is_training else 'eval')

  if examples_path:
    tf.logging.info('Reading examples from file: %s', examples_path)
    num_files = len(tf.gfile.Glob(examples_path))
    if not num_files:
      raise ValueError(
          'No files were found matching examples path: %s' %  examples_path)
    files = tf.data.Dataset.list_files(examples_path)
    dataset = files.interleave(
        tf_file_reader,
        cycle_length=tf.data.experimental.AUTOTUNE,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Use instead of `deterministic` kwarg for backward compatibility.
    options = tf.data.Options()
    options.experimental_deterministic = not is_training
    dataset = dataset.with_options(options)
  elif config.tfds_name:
    tf.logging.info('Reading examples from TFDS: %s', config.tfds_name)
    dataset = tfds.load(
        config.tfds_name,
        split=tfds.Split.TRAIN if is_training else tfds.Split.VALIDATION,
        shuffle_files=is_training,
        try_gcs=True)
    def _tf_midi_to_note_sequence(ex):
      return tf.py_function(
          lambda x:  # pylint:disable=g-long-lambda
          [note_seq.midi_to_note_sequence(x.numpy()).SerializeToString()],
          inp=[ex['midi']],
          Tout=tf.string,
          name='midi_to_note_sequence')
    dataset = dataset.map(
        _tf_midi_to_note_sequence,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    raise ValueError(
        'One of `config.examples_path` or `config.tfds_name` must be defined.')

  def _remove_pad_fn(padded_seq_1, padded_seq_2, padded_seq_3, length):
    if length.shape.ndims == 0:
      return (padded_seq_1[0:length], padded_seq_2[0:length],
              padded_seq_3[0:length], length)
    else:
      # Don't remove padding for hierarchical examples.
      return padded_seq_1, padded_seq_2, padded_seq_3, length

  if note_sequence_augmenter is not None:
    dataset = dataset.map(
        note_sequence_augmenter.tf_augment,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.map(
      tf.autograph.experimental.do_not_convert(
          functools.partial(convert_to_tensors_op, converter=data_converter)),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.unbatch()
  dataset = dataset.map(
      _remove_pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if cache_dataset:
    dataset = dataset.cache()
  if is_training:
    dataset = dataset.shuffle(buffer_size=10 * batch_size).repeat()

  dataset = dataset.padded_batch(
      batch_size,
      tf.data.get_output_shapes(dataset),
      drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

  return dataset
