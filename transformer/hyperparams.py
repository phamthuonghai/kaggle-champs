import sys


class Legacy:
    num_layers = 5
    d_model = 128
    dff = 128
    num_heads = 128
    dropout_rate = 0.1
    rel_att = (0, 1, 2, 3, 4)
    classifier = (128, 128, 64)
    shared_rel_att = False
    warmup_steps = 1000
    loss = 'loss'  # MAE
    lr = 'custom'


class Default:
    num_layers = 6
    d_model = 256
    dff = 256
    num_heads = 64
    dropout_rate = 0.1
    rel_att = (0, 1, 2, 3, 4, 5)
    classifier = (512, 512)
    shared_rel_att = False
    warmup_steps = 1000
    loss = 'loss'
    lr = 'custom'


def legacy():
    return Legacy()


def long_cls():
    hparams = Legacy()
    hparams.classifier = (128, 128, 128, 64)
    return hparams


def broad():
    hparams = Legacy()
    hparams.d_model = 512
    hparams.classifier = (512, 256)
    return hparams


def broad2():
    hparams = Legacy()
    hparams.d_model = 512
    hparams.dff = 512
    hparams.classifier = (512, 256)
    return hparams


def broad3():
    hparams = Legacy()
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
    hparams.num_heads = 8  # mistake, might wanna check with num_heads=128
    hparams.loss = 'huber1'
    return hparams


def broad_huber128():
    hparams = broad()
    hparams.loss = 'huber1'
    return hparams


def broad_logloss():
    hparams = broad()
    hparams.loss = 'log_loss'
    return hparams


def broader():
    hparams = Legacy()
    hparams.num_layers = 4
    hparams.rel_att = (0, 1, 2, 3)
    hparams.d_model = 1024
    hparams.dff = 1024
    hparams.classifier = (1024, 1024)
    return hparams


def broad8():
    hparams = broad2()
    hparams.num_layers = 8
    hparams.rel_att = (0, 1, 2, 3, 4, 5, 6, 7)
    hparams.loss = 'log_loss'
    return hparams


def broada512():
    hparams = broad_logloss()
    hparams.dff = 512


def default():
    hparams = Default()
    return hparams


def karpathy():
    hparams = Default()
    hparams.loss = 'loss'
    hparams.lr = 'karpathy'
    return hparams


def karpathy_logloss():
    hparams = Default()
    hparams.loss = 'log_loss'
    hparams.lr = 'karpathy'
    return hparams


def get_hparams(hparams):
    return getattr(sys.modules[__name__], hparams)()
