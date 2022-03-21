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

"""Imports classes and utils into the top-level namespace."""

from note_seq.constants import *  # pylint: disable=wildcard-import

from note_seq.drums_encoder_decoder import MultiDrumOneHotEncoding

from note_seq.drums_lib import DrumTrack

from note_seq.encoder_decoder import ConditionalEventSequenceEncoderDecoder
from note_seq.encoder_decoder import EventSequenceEncoderDecoder
from note_seq.encoder_decoder import LookbackEventSequenceEncoderDecoder
from note_seq.encoder_decoder import MultipleEventSequenceEncoder
from note_seq.encoder_decoder import OneHotEncoding
from note_seq.encoder_decoder import OneHotEventSequenceEncoderDecoder
from note_seq.encoder_decoder import OneHotIndexEventSequenceEncoderDecoder
from note_seq.encoder_decoder import OptionalEventSequenceEncoder

from note_seq.events_lib import NonIntegerStepsPerBarError

from note_seq.pianoroll_encoder_decoder import PianorollEncoderDecoder
from note_seq.pianoroll_lib import PianorollSequence

from note_seq.protobuf import music_pb2
from note_seq.protobuf.music_pb2 import NoteSequence  # pylint:disable=g-importing-member

from note_seq.sequences_lib import apply_sustain_control_changes
from note_seq.sequences_lib import BadTimeSignatureError
from note_seq.sequences_lib import concatenate_sequences
from note_seq.sequences_lib import extract_subsequence
from note_seq.sequences_lib import MultipleTempoError
from note_seq.sequences_lib import MultipleTimeSignatureError
from note_seq.sequences_lib import NegativeTimeError
from note_seq.sequences_lib import quantize_note_sequence
from note_seq.sequences_lib import quantize_note_sequence_absolute
from note_seq.sequences_lib import quantize_to_step
from note_seq.sequences_lib import sequence_to_pianoroll
from note_seq.sequences_lib import split_note_sequence
from note_seq.sequences_lib import steps_per_bar_in_quantized_sequence
from note_seq.sequences_lib import steps_per_quarter_to_steps_per_second
from note_seq.sequences_lib import trim_note_sequence

import note_seq.version
from note_seq.version import __version__
