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

"""Utility functions for working with polyphonic performances."""

import abc
import collections
import math

from absl import logging
import attr
from note_seq import constants
from note_seq import events_lib
from note_seq import sequences_lib
from note_seq.protobuf import music_pb2

MAX_MIDI_PITCH = constants.MAX_MIDI_PITCH
MIN_MIDI_PITCH = constants.MIN_MIDI_PITCH

MAX_MIDI_VELOCITY = constants.MAX_MIDI_VELOCITY
MIN_MIDI_VELOCITY = constants.MIN_MIDI_VELOCITY
MAX_NUM_VELOCITY_BINS = MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1

STANDARD_PPQ = constants.STANDARD_PPQ

DEFAULT_MAX_SHIFT_STEPS = 100
DEFAULT_MAX_SHIFT_QUARTERS = 4

DEFAULT_PROGRAM = 0


@attr.s(frozen=True)
class PerformanceEvent(object):
  """Class for storing events in a performance."""
  event_type = attr.ib()
  event_value = attr.ib()

  # Start of a new note.
  NOTE_ON = 1
  # End of a note.
  NOTE_OFF = 2
  # Shift time forward.
  TIME_SHIFT = 3
  # Change current velocity.
  VELOCITY = 4
  # Duration of preceding NOTE_ON.
  # For Note-based encoding, used instead of NOTE_OFF events.
  DURATION = 5

  @event_type.validator
  def _check_event(self, attribute, value):
    """Validate event contents."""
    del attribute, value  # This checks the whole object.
    if self.event_type in (PerformanceEvent.NOTE_ON, PerformanceEvent.NOTE_OFF):
      if not MIN_MIDI_PITCH <= self.event_value <= MAX_MIDI_PITCH:
        raise ValueError('Invalid pitch value: %s' % self.event_value)
    elif self.event_type == PerformanceEvent.TIME_SHIFT:
      if not 0 <= self.event_value:
        raise ValueError('Invalid time shift value: %s' % self.event_value)
    elif self.event_type == PerformanceEvent.DURATION:
      if not 1 <= self.event_value:
        raise ValueError('Invalid duration value: %s' % self.event_value)
    elif self.event_type == PerformanceEvent.VELOCITY:
      if not 1 <= self.event_value <= MAX_NUM_VELOCITY_BINS:
        raise ValueError('Invalid velocity value: %s' % self.event_value)
    else:
      raise ValueError('Invalid event type: %s' % self.event_type)


def _velocity_bin_size(num_velocity_bins):
  return int(math.ceil(
      (MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1) / num_velocity_bins))


def velocity_to_bin(velocity, num_velocity_bins):
  return ((velocity - MIN_MIDI_VELOCITY) //
          _velocity_bin_size(num_velocity_bins) + 1)


def velocity_bin_to_velocity(velocity_bin, num_velocity_bins):
  return (
      MIN_MIDI_VELOCITY + (velocity_bin - 1) *
      _velocity_bin_size(num_velocity_bins))


def _program_and_is_drum_from_sequence(sequence, instrument=None):
  """Get MIDI program and is_drum from sequence and (optional) instrument.

  Args:
    sequence: The NoteSequence from which MIDI program and is_drum will be
        extracted.
    instrument: The instrument in `sequence` from which MIDI program and
        is_drum will be extracted, or None to consider all instruments.

  Returns:
    A tuple containing program and is_drum for the sequence and optional
    instrument. If multiple programs are found (or if is_drum is True),
    program will be None. If multiple values of is_drum are found, is_drum
    will be None.
  """
  notes = [note for note in sequence.notes
           if instrument is None or note.instrument == instrument]
  # Only set program for non-drum tracks.
  if all(note.is_drum for note in notes):
    is_drum = True
    program = None
  elif all(not note.is_drum for note in notes):
    is_drum = False
    programs = set(note.program for note in notes)
    program = programs.pop() if len(programs) == 1 else None
  else:
    is_drum = None
    program = None
  return program, is_drum


class BasePerformance(events_lib.EventSequence):
  """Stores a polyphonic sequence as a stream of performance events.

  Events are PerformanceEvent objects that encode event type and value.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, start_step, num_velocity_bins, max_shift_steps,
               program=None, is_drum=None):
    """Construct a BasePerformance.

    Args:
      start_step: The offset of this sequence relative to the beginning of the
          source sequence.
      num_velocity_bins: Number of velocity bins to use.
      max_shift_steps: Maximum number of steps for a single time-shift event.
      program: MIDI program used for this performance, or None if not specified.
      is_drum: Whether or not this performance consists of drums, or None if not
          specified.

    Raises:
      ValueError: If `num_velocity_bins` is larger than the number of MIDI
          velocity values.
    """
    if num_velocity_bins > MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1:
      raise ValueError(
          'Number of velocity bins is too large: %d' % num_velocity_bins)

    self._start_step = start_step
    self._num_velocity_bins = num_velocity_bins
    self._max_shift_steps = max_shift_steps
    self._program = program
    self._is_drum = is_drum

  @property
  def start_step(self):
    return self._start_step

  @property
  def max_shift_steps(self):
    return self._max_shift_steps

  @property
  def program(self):
    return self._program

  @property
  def is_drum(self):
    return self._is_drum

  def _append_steps(self, num_steps):
    """Adds steps to the end of the sequence."""
    if (self._events and
        self._events[-1].event_type == PerformanceEvent.TIME_SHIFT and
        self._events[-1].event_value < self._max_shift_steps):
      # Last event is already non-maximal time shift. Increase its duration.
      added_steps = min(num_steps,
                        self._max_shift_steps - self._events[-1].event_value)
      self._events[-1] = PerformanceEvent(
          PerformanceEvent.TIME_SHIFT,
          self._events[-1].event_value + added_steps)
      num_steps -= added_steps

    while num_steps >= self._max_shift_steps:
      self._events.append(
          PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT,
                           event_value=self._max_shift_steps))
      num_steps -= self._max_shift_steps

    if num_steps > 0:
      self._events.append(
          PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT,
                           event_value=num_steps))

  def _trim_steps(self, num_steps):
    """Trims a given number of steps from the end of the sequence."""
    steps_trimmed = 0
    while self._events and steps_trimmed < num_steps:
      if self._events[-1].event_type == PerformanceEvent.TIME_SHIFT:
        if steps_trimmed + self._events[-1].event_value > num_steps:
          self._events[-1] = PerformanceEvent(
              event_type=PerformanceEvent.TIME_SHIFT,
              event_value=(self._events[-1].event_value -
                           num_steps + steps_trimmed))
          steps_trimmed = num_steps
        else:
          steps_trimmed += self._events[-1].event_value
          self._events.pop()
      else:
        self._events.pop()

  def set_length(self, steps, from_left=False):
    """Sets the length of the sequence to the specified number of steps.

    If the event sequence is not long enough, pads with time shifts to make the
    sequence the specified length. If it is too long, it will be truncated to
    the requested length.

    Args:
      steps: How many quantized steps long the event sequence should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    if from_left:
      raise NotImplementedError('from_left is not supported')

    if self.num_steps < steps:
      self._append_steps(steps - self.num_steps)
    elif self.num_steps > steps:
      self._trim_steps(self.num_steps - steps)

    assert self.num_steps == steps

  def append(self, event):
    """Appends the event to the end of the sequence.

    Args:
      event: The performance event to append to the end.

    Raises:
      ValueError: If `event` is not a valid performance event.
    """
    if not isinstance(event, PerformanceEvent):
      raise ValueError('Invalid performance event: %s' % event)
    self._events.append(event)

  def truncate(self, num_events):
    """Truncates this Performance to the specified number of events.

    Args:
      num_events: The number of events to which this performance will be
          truncated.
    """
    self._events = self._events[:num_events]

  def __len__(self):
    """How many events are in this sequence.

    Returns:
      Number of events as an integer.
    """
    return len(self._events)

  def __getitem__(self, i):
    """Returns the event at the given index."""
    return self._events[i]

  def __iter__(self):
    """Return an iterator over the events in this sequence."""
    return iter(self._events)

  def __str__(self):
    strs = []
    for event in self:
      if event.event_type == PerformanceEvent.NOTE_ON:
        strs.append('(%s, ON)' % event.event_value)
      elif event.event_type == PerformanceEvent.NOTE_OFF:
        strs.append('(%s, OFF)' % event.event_value)
      elif event.event_type == PerformanceEvent.TIME_SHIFT:
        strs.append('(%s, SHIFT)' % event.event_value)
      elif event.event_type == PerformanceEvent.VELOCITY:
        strs.append('(%s, VELOCITY)' % event.event_value)
      else:
        raise ValueError('Unknown event type: %s' % event.event_type)
    return '\n'.join(strs)

  @property
  def end_step(self):
    return self.start_step + self.num_steps

  @property
  def num_steps(self):
    """Returns how many steps long this sequence is.

    Returns:
      Length of the sequence in quantized steps.
    """
    steps = 0
    for event in self:
      if event.event_type == PerformanceEvent.TIME_SHIFT:
        steps += event.event_value
    return steps

  @property
  def steps(self):
    """Return a Python list of the time step at each event in this sequence."""
    step = self.start_step
    result = []
    for event in self:
      result.append(step)
      if event.event_type == PerformanceEvent.TIME_SHIFT:
        step += event.event_value
    return result

  @staticmethod
  def _from_quantized_sequence(quantized_sequence, start_step,
                               num_velocity_bins, max_shift_steps,
                               instrument=None):
    """Extract a list of events from the given quantized NoteSequence object.

    Within a step, new pitches are started with NOTE_ON and existing pitches are
    ended with NOTE_OFF. TIME_SHIFT shifts the current step forward in time.
    VELOCITY changes the current velocity value that will be applied to all
    NOTE_ON events.

    Args:
      quantized_sequence: A quantized NoteSequence instance.
      start_step: Start converting the sequence at this time step.
      num_velocity_bins: Number of velocity bins to use. If 0, velocity events
          will not be included at all.
      max_shift_steps: Maximum number of steps for a single time-shift event.
      instrument: If not None, extract only the specified instrument. Otherwise,
          extract all instruments into a single event list.

    Returns:
      A list of events.
    """
    notes = [note for note in quantized_sequence.notes
             if note.quantized_start_step >= start_step
             and (instrument is None or note.instrument == instrument)]
    sorted_notes = sorted(notes, key=lambda note: (note.start_time, note.pitch))

    # Sort all note start and end events.
    onsets = [(note.quantized_start_step, idx, False)
              for idx, note in enumerate(sorted_notes)]
    offsets = [(note.quantized_end_step, idx, True)
               for idx, note in enumerate(sorted_notes)]
    note_events = sorted(onsets + offsets)

    current_step = start_step
    current_velocity_bin = 0
    performance_events = []

    for step, idx, is_offset in note_events:
      if step > current_step:
        # Shift time forward from the current step to this event.
        while step > current_step + max_shift_steps:
          # We need to move further than the maximum shift size.
          performance_events.append(
              PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT,
                               event_value=max_shift_steps))
          current_step += max_shift_steps
        performance_events.append(
            PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT,
                             event_value=int(step - current_step)))
        current_step = step

      # If we're using velocity and this note's velocity is different from the
      # current velocity, change the current velocity.
      if num_velocity_bins:
        velocity_bin = velocity_to_bin(
            sorted_notes[idx].velocity, num_velocity_bins)
        if not is_offset and velocity_bin != current_velocity_bin:
          current_velocity_bin = velocity_bin
          performance_events.append(
              PerformanceEvent(event_type=PerformanceEvent.VELOCITY,
                               event_value=current_velocity_bin))

      # Add a performance event for this note on/off.
      event_type = (
          PerformanceEvent.NOTE_OFF if is_offset else PerformanceEvent.NOTE_ON)
      performance_events.append(
          PerformanceEvent(event_type=event_type,
                           event_value=sorted_notes[idx].pitch))

    return performance_events

  @abc.abstractmethod
  def to_sequence(self, velocity, instrument, program, max_note_duration=None):
    """Converts the Performance to NoteSequence proto.

    Args:
      velocity: MIDI velocity to give each note. Between 1 and 127 (inclusive).
          If the performance contains velocity events, those will be used
          instead.
      instrument: MIDI instrument to give each note.
      program: MIDI program to give each note, or None to use the program
          associated with the Performance (or the default program if none
          exists).
      max_note_duration: Maximum note duration in seconds to allow. Notes longer
          than this will be truncated. If None, notes can be any length.

    Returns:
      A NoteSequence proto.
    """
    pass

  def _to_sequence(self, seconds_per_step, velocity, instrument, program,
                   max_note_duration=None):
    sequence_start_time = self.start_step * seconds_per_step

    sequence = music_pb2.NoteSequence()
    sequence.ticks_per_quarter = STANDARD_PPQ

    step = 0

    if program is None:
      # Use program associated with the performance (or default program).
      program = self.program if self.program is not None else DEFAULT_PROGRAM
    is_drum = self.is_drum if self.is_drum is not None else False

    # Map pitch to list because one pitch may be active multiple times.
    pitch_start_steps_and_velocities = collections.defaultdict(list)
    for i, event in enumerate(self):
      if event.event_type == PerformanceEvent.NOTE_ON:
        pitch_start_steps_and_velocities[event.event_value].append(
            (step, velocity))
      elif event.event_type == PerformanceEvent.NOTE_OFF:
        if not pitch_start_steps_and_velocities[event.event_value]:
          logging.debug(
              'Ignoring NOTE_OFF at position %d with no previous NOTE_ON', i)
        else:
          # Create a note for the pitch that is now ending.
          pitch_start_step, pitch_velocity = pitch_start_steps_and_velocities[
              event.event_value][0]
          pitch_start_steps_and_velocities[event.event_value] = (
              pitch_start_steps_and_velocities[event.event_value][1:])
          if step == pitch_start_step:
            logging.debug(
                'Ignoring note with zero duration at step %d', step)
            continue
          note = sequence.notes.add()
          note.start_time = (pitch_start_step * seconds_per_step +
                             sequence_start_time)
          note.end_time = step * seconds_per_step + sequence_start_time
          if (max_note_duration and
              note.end_time - note.start_time > max_note_duration):
            note.end_time = note.start_time + max_note_duration
          note.pitch = event.event_value
          note.velocity = pitch_velocity
          note.instrument = instrument
          note.program = program
          note.is_drum = is_drum
          if note.end_time > sequence.total_time:
            sequence.total_time = note.end_time
      elif event.event_type == PerformanceEvent.TIME_SHIFT:
        step += event.event_value
      elif event.event_type == PerformanceEvent.VELOCITY:
        assert self._num_velocity_bins
        velocity = velocity_bin_to_velocity(
            event.event_value, self._num_velocity_bins)
      else:
        raise ValueError('Unknown event type: %s' % event.event_type)

    # There could be remaining pitches that were never ended. End them now
    # and create notes.
    for pitch in pitch_start_steps_and_velocities:
      for pitch_start_step, pitch_velocity in pitch_start_steps_and_velocities[
          pitch]:
        if step == pitch_start_step:
          logging.debug('Ignoring note with zero duration at step %d', step)
          continue
        note = sequence.notes.add()
        note.start_time = (pitch_start_step * seconds_per_step +
                           sequence_start_time)
        note.end_time = step * seconds_per_step + sequence_start_time
        if (max_note_duration and
            note.end_time - note.start_time > max_note_duration):
          note.end_time = note.start_time + max_note_duration
        note.pitch = pitch
        note.velocity = pitch_velocity
        note.instrument = instrument
        note.program = program
        note.is_drum = is_drum
        if note.end_time > sequence.total_time:
          sequence.total_time = note.end_time

    return sequence


class Performance(BasePerformance):
  """Performance with absolute timing and unknown meter."""

  def __init__(self, quantized_sequence=None, steps_per_second=None,
               start_step=0, num_velocity_bins=0,
               max_shift_steps=DEFAULT_MAX_SHIFT_STEPS, instrument=None,
               program=None, is_drum=None):
    """Construct a Performance.

    Either quantized_sequence or steps_per_second should be supplied.

    Args:
      quantized_sequence: A quantized NoteSequence proto.
      steps_per_second: Number of quantized time steps per second, if using
          absolute quantization.
      start_step: The offset of this sequence relative to the
          beginning of the source sequence. If a quantized sequence is used as
          input, only notes starting after this step will be considered.
      num_velocity_bins: Number of velocity bins to use. If 0, velocity events
          will not be included at all.
      max_shift_steps: Maximum number of steps for a single time-shift event.
      instrument: If not None, extract only the specified instrument from
          `quantized_sequence`. Otherwise, extract all instruments.
      program: MIDI program used for this performance, or None if not specified.
          Ignored if `quantized_sequence` is provided.
      is_drum: Whether or not this performance consists of drums, or None if not
          specified. Ignored if `quantized_sequence` is provided.

    Raises:
      ValueError: If both or neither of `quantized_sequence` or
          `steps_per_second` is specified.
    """
    if (quantized_sequence, steps_per_second).count(None) != 1:
      raise ValueError(
          'Must specify exactly one of quantized_sequence or steps_per_second')

    if quantized_sequence:
      sequences_lib.assert_is_absolute_quantized_sequence(quantized_sequence)
      self._steps_per_second = (
          quantized_sequence.quantization_info.steps_per_second)
      self._events = self._from_quantized_sequence(
          quantized_sequence, start_step, num_velocity_bins,
          max_shift_steps=max_shift_steps, instrument=instrument)
      program, is_drum = _program_and_is_drum_from_sequence(
          quantized_sequence, instrument)

    else:
      self._steps_per_second = steps_per_second
      self._events = []

    super(Performance, self).__init__(
        start_step=start_step,
        num_velocity_bins=num_velocity_bins,
        max_shift_steps=max_shift_steps,
        program=program,
        is_drum=is_drum)

  @property
  def steps_per_second(self):
    return self._steps_per_second

  def to_sequence(self,
                  velocity=100,
                  instrument=0,
                  program=None,
                  max_note_duration=None):
    """Converts the Performance to NoteSequence proto.

    Args:
      velocity: MIDI velocity to give each note. Between 1 and 127 (inclusive).
          If the performance contains velocity events, those will be used
          instead.
      instrument: MIDI instrument to give each note.
      program: MIDI program to give each note, or None to use the program
          associated with the Performance (or the default program if none
          exists).
      max_note_duration: Maximum note duration in seconds to allow. Notes longer
          than this will be truncated. If None, notes can be any length.

    Returns:
      A NoteSequence proto.
    """
    seconds_per_step = 1.0 / self.steps_per_second
    return self._to_sequence(
        seconds_per_step=seconds_per_step,
        velocity=velocity,
        instrument=instrument,
        program=program,
        max_note_duration=max_note_duration)


class MetricPerformance(BasePerformance):
  """Performance with quarter-note relative timing."""

  def __init__(self, quantized_sequence=None, steps_per_quarter=None,
               start_step=0, num_velocity_bins=0,
               max_shift_quarters=DEFAULT_MAX_SHIFT_QUARTERS, instrument=None,
               program=None, is_drum=None):
    """Construct a MetricPerformance.

    Either quantized_sequence or steps_per_quarter should be supplied.

    Args:
      quantized_sequence: A quantized NoteSequence proto.
      steps_per_quarter: Number of quantized time steps per quarter note, if
          using metric quantization.
      start_step: The offset of this sequence relative to the
          beginning of the source sequence. If a quantized sequence is used as
          input, only notes starting after this step will be considered.
      num_velocity_bins: Number of velocity bins to use. If 0, velocity events
          will not be included at all.
      max_shift_quarters: Maximum number of quarter notes for a single time-
          shift event.
      instrument: If not None, extract only the specified instrument from
          `quantized_sequence`. Otherwise, extract all instruments.
      program: MIDI program used for this performance, or None if not specified.
          Ignored if `quantized_sequence` is provided.
      is_drum: Whether or not this performance consists of drums, or None if not
          specified. Ignored if `quantized_sequence` is provided.

    Raises:
      ValueError: If both or neither of `quantized_sequence` or
          `steps_per_quarter` is specified.
    """
    if (quantized_sequence, steps_per_quarter).count(None) != 1:
      raise ValueError(
          'Must specify exactly one of quantized_sequence or steps_per_quarter')

    if quantized_sequence:
      sequences_lib.assert_is_relative_quantized_sequence(quantized_sequence)
      self._steps_per_quarter = (
          quantized_sequence.quantization_info.steps_per_quarter)
      self._events = self._from_quantized_sequence(
          quantized_sequence, start_step, num_velocity_bins,
          max_shift_steps=self._steps_per_quarter * max_shift_quarters,
          instrument=instrument)
      program, is_drum = _program_and_is_drum_from_sequence(
          quantized_sequence, instrument)

    else:
      self._steps_per_quarter = steps_per_quarter
      self._events = []

    super(MetricPerformance, self).__init__(
        start_step=start_step,
        num_velocity_bins=num_velocity_bins,
        max_shift_steps=self._steps_per_quarter * max_shift_quarters,
        program=program,
        is_drum=is_drum)

  @property
  def steps_per_quarter(self):
    return self._steps_per_quarter

  def to_sequence(self,
                  velocity=100,
                  instrument=0,
                  program=None,
                  max_note_duration=None,
                  qpm=120.0):
    """Converts the Performance to NoteSequence proto.

    Args:
      velocity: MIDI velocity to give each note. Between 1 and 127 (inclusive).
          If the performance contains velocity events, those will be used
          instead.
      instrument: MIDI instrument to give each note.
      program: MIDI program to give each note, or None to use the program
          associated with the Performance (or the default program if none
          exists).
      max_note_duration: Maximum note duration in seconds to allow. Notes longer
          than this will be truncated. If None, notes can be any length.
      qpm: The tempo to use, in quarter notes per minute.

    Returns:
      A NoteSequence proto.
    """
    seconds_per_step = 60.0 / (self.steps_per_quarter * qpm)
    sequence = self._to_sequence(
        seconds_per_step=seconds_per_step,
        velocity=velocity,
        instrument=instrument,
        program=program,
        max_note_duration=max_note_duration)
    sequence.tempos.add(qpm=qpm)
    return sequence


class NotePerformanceError(Exception):
  pass


class TooManyTimeShiftStepsError(NotePerformanceError):
  pass


class TooManyDurationStepsError(NotePerformanceError):
  pass


class NotePerformance(BasePerformance):
  """Stores a polyphonic sequence as a stream of performance events.

  Events are PerformanceEvent objects that encode event type and value.
  In this version, the performance is encoded in 4-event tuples:
  TIME_SHIFT, NOTE_ON, VELOCITY, DURATION.
  """

  def __init__(self, quantized_sequence, num_velocity_bins, instrument=0,
               start_step=0, max_shift_steps=1000, max_duration_steps=1000):
    """Construct a NotePerformance.

    Args:
      quantized_sequence: A quantized NoteSequence proto.
      num_velocity_bins: Number of velocity bins to use.
      instrument: If not None, extract only the specified instrument from
          `quantized_sequence`. Otherwise, extract all instruments.
      start_step: The offset of this sequence relative to the beginning of the
          source sequence.
      max_shift_steps: Maximum number of steps for a time-shift event.
      max_duration_steps: Maximum number of steps for a duration event.

    Raises:
      ValueError: If `num_velocity_bins` is larger than the number of MIDI
          velocity values.
    """
    program, is_drum = _program_and_is_drum_from_sequence(
        quantized_sequence, instrument)

    super(NotePerformance, self).__init__(
        start_step=start_step,
        num_velocity_bins=num_velocity_bins,
        max_shift_steps=max_shift_steps,
        program=program,
        is_drum=is_drum)

    self._max_duration_steps = max_duration_steps

    sequences_lib.assert_is_absolute_quantized_sequence(quantized_sequence)
    self._steps_per_second = (
        quantized_sequence.quantization_info.steps_per_second)
    self._events = self._from_quantized_sequence(
        quantized_sequence, instrument)

  @property
  def steps_per_second(self):
    return self._steps_per_second

  def set_length(self, steps, from_left=False):
    # This is not actually implemented, but to avoid raising exceptions during
    # generation just return instead of raising NotImplementedError.
    # TODO(fjord): Implement this.
    return

  def append(self, event):
    """Appends the event to the end of the sequence.

    Args:
      event: The performance event tuple to append to the end.

    Raises:
      ValueError: If `event` is not a valid performance event tuple.
    """
    if not isinstance(event, tuple):
      raise ValueError('Invalid performance event tuple: %s' % event)
    self._events.append(event)

  def __str__(self):
    strs = []
    for event in self:
      strs.append('TIME_SHIFT<%s>, NOTE_ON<%s>, VELOCITY<%s>, DURATION<%s>' % (
          event[0].event_value, event[1].event_value, event[2].event_value,
          event[3].event_value))
    return '\n'.join(strs)

  @property
  def num_steps(self):
    """Returns how many steps long this sequence is.

    Returns:
      Length of the sequence in quantized steps.
    """
    steps = 0
    for event in self._events:
      steps += event[0].event_value
    if self._events:
      steps += self._events[-1][3].event_value
    return steps

  @property
  def steps(self):
    """Return a Python list of the time step at each event in this sequence."""
    step = self.start_step
    result = []
    for event in self:
      step += event[0].event_value
      result.append(step)
    return result

  def _from_quantized_sequence(self, quantized_sequence, instrument):
    """Extract a list of events from the given quantized NoteSequence object.

    Within a step, new pitches are started with NOTE_ON and existing pitches are
    ended with NOTE_OFF. TIME_SHIFT shifts the current step forward in time.
    VELOCITY changes the current velocity value that will be applied to all
    NOTE_ON events.

    Args:
      quantized_sequence: A quantized NoteSequence instance.
      instrument: If not None, extract only the specified instrument. Otherwise,
          extract all instruments into a single event list.

    Returns:
      A list of events.

    Raises:
      TooManyTimeShiftStepsError: If the maximum number of time
        shift steps is exceeded.
      TooManyDurationStepsError: If the maximum number of duration
        shift steps is exceeded.
    """
    notes = [note for note in quantized_sequence.notes
             if note.quantized_start_step >= self.start_step
             and (instrument is None or note.instrument == instrument)]
    sorted_notes = sorted(notes, key=lambda note: (note.start_time, note.pitch))

    current_step = self.start_step
    performance_events = []

    for note in sorted_notes:
      sub_events = []

      # TIME_SHIFT
      time_shift_steps = note.quantized_start_step - current_step
      if time_shift_steps > self._max_shift_steps:
        raise TooManyTimeShiftStepsError(
            'Too many steps for timeshift: %d' % time_shift_steps)
      else:
        sub_events.append(
            PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT,
                             event_value=time_shift_steps))
      current_step = note.quantized_start_step

      # NOTE_ON
      sub_events.append(
          PerformanceEvent(event_type=PerformanceEvent.NOTE_ON,
                           event_value=note.pitch))

      # VELOCITY
      velocity_bin = velocity_to_bin(note.velocity, self._num_velocity_bins)
      sub_events.append(
          PerformanceEvent(event_type=PerformanceEvent.VELOCITY,
                           event_value=velocity_bin))

      # DURATION
      duration_steps = note.quantized_end_step - note.quantized_start_step
      if duration_steps > self._max_duration_steps:
        raise TooManyDurationStepsError(
            'Too many steps for duration: %s' % note)
      sub_events.append(
          PerformanceEvent(event_type=PerformanceEvent.DURATION,
                           event_value=duration_steps))

      performance_events.append(tuple(sub_events))

    return performance_events

  def to_sequence(self, instrument=0, program=None, max_note_duration=None):
    """Converts the Performance to NoteSequence proto.

    Args:
      instrument: MIDI instrument to give each note.
      program: MIDI program to give each note, or None to use the program
          associated with the Performance (or the default program if none
          exists).
      max_note_duration: Not used in this implementation.

    Returns:
      A NoteSequence proto.
    """
    seconds_per_step = 1.0 / self.steps_per_second
    sequence_start_time = self.start_step * seconds_per_step

    sequence = music_pb2.NoteSequence()
    sequence.ticks_per_quarter = STANDARD_PPQ

    step = 0

    if program is None:
      # Use program associated with the performance (or default program).
      program = self.program if self.program is not None else DEFAULT_PROGRAM
    is_drum = self.is_drum if self.is_drum is not None else False

    for event in self:
      step += event[0].event_value

      note = sequence.notes.add()
      note.start_time = step * seconds_per_step + sequence_start_time
      note.end_time = ((step + event[3].event_value) * seconds_per_step +
                       sequence_start_time)
      note.pitch = event[1].event_value
      note.velocity = velocity_bin_to_velocity(
          event[2].event_value, self._num_velocity_bins)
      note.instrument = instrument
      note.program = program
      note.is_drum = is_drum

      if note.end_time > sequence.total_time:
        sequence.total_time = note.end_time

    return sequence
