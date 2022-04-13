import math

from tensor2tensor.utils.hparam import HParams
from magenta.common import merge_hparams
from magenta.models.music_vae import Config, MusicVAE, lstm_models, data

DRUM_TYPE_PITCHES = [
    # kick drum
    [36, 35],

    # snare drum
    [38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85],

    # closed hi-hat
    [42, 44, 54, 68, 69, 70, 71, 73, 78, 80, 22],

    # open hi-hat
    [46, 67, 72, 74, 79, 81, 26],

    # low tom
    [45, 29, 41, 43, 61, 64, 84],

    # high tom + mid tom combined
    [50, 30, 62, 76, 83, 48, 47, 60, 63, 77, 86, 87],

    # crash cymbal
    [49, 52, 55, 57, 58],

    # ride cymbal
    [51, 53, 59, 82]
]
MUSICVAE_CONFIG_NAME = 'cat-drums_1bar_8class'
MUSICVAE_CONFIG = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.CategoricalLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=16,  # 1 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            free_bits=64,
            max_beta=0.2,
            sampling_schedule='inverse_sigmoid',
            sampling_rate=1000,
        )),
    note_sequence_augmenter=None,
    data_converter=data.DrumsConverter(
        max_bars=100,  # Truncate long drum sequences before slicing.
        slice_bars=1,
        steps_per_quarter=4,
        roll_input=True,
        pitch_classes=DRUM_TYPE_PITCHES),
    train_examples_path=None,
    eval_examples_path=None,
)

DRUM_MAP = {
    36: 8,  # bass drum
    38: 7,  # snare drum
    42: 6,  # closed hi-hat
    46: 5,  # open hi-hat
    45: 4,  # low tom
    # 48: 3,  # mid tom (map to high tom)
    50: 3,  # high tom
    49: 2,  # crash cymbal
    51: 1   # ride cymbal
}
DRUM_MAP_INVERTED = {v: k for k, v in DRUM_MAP.items()}

INTERPOLATION_STEPS = 11
ATTR_STEPS = 9
ATTR_STEP_SIZE = 0.5
# [-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]
ATTR_MULTIPLIERS = [ATTR_STEP_SIZE * x - (math.floor(ATTR_STEPS / 2) * ATTR_STEP_SIZE) for x in range(ATTR_STEPS)]
MAX_SEQ_LENGTH = 16


ATTRIBUTES = [
  'DS',
  'BD',
  'SD',
  'HH',
  'TO',
  'CY'
]

SCREEN_WIDTH = 50
SCREEN_HEIGHT = 50
PCA_CLIP = 20