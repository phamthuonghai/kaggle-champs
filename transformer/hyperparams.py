import sys


class Default:
    num_layers = 5
    d_model = 128
    dff = 128
    num_heads = 128
    dropout_rate = 0.1
    rel_att = (0, 1, 2, 3, 4)
    classifier = (128, 128, 64)
    shared_rel_att = False
    warmup_steps = 1000
    loss = 'MAE'


def default():
    return Default()


def long_cls():
    hparams = Default()
    hparams.classifier = (128, 128, 128, 64)
    return hparams


def broad():
    hparams = Default()
    hparams.d_model = 512
    hparams.classifier = (512, 256)
    return hparams


def broad2():
    hparams = Default()
    hparams.d_model = 512
    hparams.dff = 512
    hparams.classifier = (512, 256)
    return hparams


def broad3():
    hparams = Default()
    hparams.d_model = 512
    hparams.dff = 512
    hparams.classifier = (1024, 512)
    return hparams


def broad_mse():
    hparams = broad()
    hparams.loss = 'mse'
    return hparams


def broad_huber1():
    hparams = broad()
    hparams.loss = 'huber1'
    return hparams


def get_hparams(hparams):
    return getattr(sys.modules[__name__], hparams)()
