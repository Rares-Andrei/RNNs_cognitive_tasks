from rnn_neuro.models.base_rnn import BaseRNN
from rnn_neuro.models.ctrnn import CTRNN

MODEL_REGISTRY = {
    "base_rnn": BaseRNN,
    "ctrnn": CTRNN
}
