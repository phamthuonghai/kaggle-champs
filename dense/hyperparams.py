import sys


class Default:
    num_layers = 6
    d_model = 128
    dropout_rate = 0.1
    classifier = (1024, 512, 512, 512)
    loss = 'log_loss'
    lr = 'karpathy'


def default():
    hparams = Default()
    return hparams


def get_hparams(hparams):
    return getattr(sys.modules[__name__], hparams)()
