# Copyright 2022 The Magenta Authors.
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

"""Testing support code."""

import os

from absl.testing import absltest

from note_seq import encoder_decoder
from note_seq.protobuf import compare
from note_seq.protobuf import music_pb2

from google.protobuf import descriptor_pool
from google.protobuf import text_format

# Shortcut to text annotation types.
BEAT = music_pb2.NoteSequence.TextAnnotation.BEAT
CHORD_SYMBOL = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL


def add_track_to_sequence(note_sequence,
                          instrument,
                          notes,
                          is_drum=False,
                          program=0):
  """Adds instrument track to NoteSequence."""
  for pitch, velocity, start_time, end_time in notes:
    note = note_sequence.notes.add()
    note.pitch = pitch
    note.velocity = velocity
    note.start_time = start_time
    note.end_time = end_time
    note.instrument = instrument
    note.is_drum = is_drum
    note.program = program
    if end_time > note_sequence.total_time:
      note_sequence.total_time = end_time


def add_chords_to_sequence(note_sequence, chords):
  for figure, time in chords:
    annotation = note_sequence.text_annotations.add()
    annotation.time = time
    annotation.text = figure
    annotation.annotation_type = CHORD_SYMBOL


def add_key_signatures_to_sequence(note_sequence, keys):
  for key, time in keys:
    ks = note_sequence.key_signatures.add()
    ks.time = time
    ks.key = key


def add_beats_to_sequence(note_sequence, beats, beats_per_bar=None):
  for i, time in enumerate(beats):
    annotation = note_sequence.text_annotations.add()
    annotation.time = time
    annotation.annotation_type = BEAT
    if beats_per_bar:
      annotation.text = str((i % beats_per_bar) + 1)


def add_control_changes_to_sequence(note_sequence, instrument, control_changes):
  for time, control_number, control_value in control_changes:
    control_change = note_sequence.control_changes.add()
    control_change.time = time
    control_change.control_number = control_number
    control_change.control_value = control_value
    control_change.instrument = instrument


def add_pitch_bends_to_sequence(
    note_sequence, instrument, program, pitch_bends):
  for time, bend in pitch_bends:
    pitch_bend = note_sequence.pitch_bends.add()
    pitch_bend.time = time
    pitch_bend.bend = bend
    pitch_bend.program = program
    pitch_bend.instrument = instrument
    pitch_bend.is_drum = False  # Assume false for this test method.


def add_quantized_steps_to_sequence(sequence, quantized_steps):
  assert len(sequence.notes) == len(quantized_steps)

  for note, quantized_step in zip(sequence.notes, quantized_steps):
    note.quantized_start_step = quantized_step[0]
    note.quantized_end_step = quantized_step[1]

    if quantized_step[1] > sequence.total_quantized_steps:
      sequence.total_quantized_steps = quantized_step[1]


def add_quantized_chord_steps_to_sequence(sequence, quantized_steps):
  chord_annotations = [a for a in sequence.text_annotations
                       if a.annotation_type == CHORD_SYMBOL]
  assert len(chord_annotations) == len(quantized_steps)
  for chord, quantized_step in zip(chord_annotations, quantized_steps):
    chord.quantized_step = quantized_step


def add_quantized_control_steps_to_sequence(sequence, quantized_steps):
  assert len(sequence.control_changes) == len(quantized_steps)

  for cc, quantized_step in zip(sequence.control_changes, quantized_steps):
    cc.quantized_step = quantized_step


class TrivialOneHotEncoding(encoder_decoder.OneHotEncoding):
  """One-hot encoding that uses the identity encoding."""

  def __init__(self, num_classes, num_steps=None):
    if num_steps is not None and len(num_steps) != num_classes:
      raise ValueError('num_steps must have length num_classes')
    self._num_classes = num_classes
    self._num_steps = num_steps

  @property
  def num_classes(self):
    return self._num_classes

  @property
  def default_event(self):
    return 0

  def encode_event(self, event):
    return event

  def decode_event(self, index):
    event = index
    return event

  def event_to_num_steps(self, event):
    if self._num_steps is not None:
      return self._num_steps[event]
    else:
      return 1


def parse_test_proto(proto_type, proto_string):
  instance = proto_type()
  text_format.Parse(proto_string, instance)
  return instance


def get_testdata_dir():
  dir_path = 'note_seq/testdata'
  return os.path.join(absltest.get_default_test_srcdir(), dir_path)


class ProtoTestCase(absltest.TestCase):
  """Adds assertProtoEquals from tf.test.TestCase."""

  def setUp(self):
    self.maxDiff = None  # pylint:disable=invalid-name
    self.steps_per_quarter = 4
    self.note_sequence = parse_test_proto(
        music_pb2.NoteSequence, """
        time_signatures: {
          numerator: 4
          denominator: 4
        }
        tempos: {
          qpm: 60
        }
        """)
    super().setUp()

  def _AssertProtoEquals(self, a, b, msg=None):  # pylint:disable=invalid-name
    """Asserts that a and b are the same proto.

    Uses ProtoEq() first, as it returns correct results
    for floating point attributes, and then use assertProtoEqual()
    in case of failure as it provides good error messages.

    Args:
      a: a proto.
      b: another proto.
      msg: Optional message to report on failure.
    """
    if not compare.ProtoEq(a, b):
      compare.assertProtoEqual(self, a, b, normalize_numbers=True, msg=msg)

  def assertProtoEquals(self, expected_message_maybe_ascii, message, msg=None):
    """Asserts that message is same as parsed expected_message_ascii.

    Creates another prototype of message, reads the ascii message into it and
    then compares them using self._AssertProtoEqual().

    Args:
      expected_message_maybe_ascii: proto message in original or ascii form.
      message: the message to validate.
      msg: Optional message to report on failure.
    """
    msg = msg if msg else ''
    if isinstance(expected_message_maybe_ascii, type(message)):
      expected_message = expected_message_maybe_ascii
      self._AssertProtoEquals(expected_message, message)
    elif isinstance(expected_message_maybe_ascii, str):
      expected_message = type(message)()
      text_format.Parse(
          expected_message_maybe_ascii,
          expected_message,
          descriptor_pool=descriptor_pool.Default())
      self._AssertProtoEquals(expected_message, message, msg=msg)
    else:
      assert False, ("Can't compare protos of type %s and %s. %s" %
                     (type(expected_message_maybe_ascii), type(message), msg))
