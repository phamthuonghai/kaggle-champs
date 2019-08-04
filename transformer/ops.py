import sys

import tensorflow as tf

# LOSSES = ['log_loss', 'loss', 'mse', 'huber1']
LOSSES = ['log_loss', 'loss', 'mse']


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, hparams):
        super(CustomSchedule, self).__init__()

        self.d_model = hparams.d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = hparams.warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        return lr


class KarpathySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, hparams):
        super(KarpathySchedule, self).__init__()

    def __call__(self, step):
        return 3e-4


def get_schedule(name):
    return getattr(sys.modules[__name__], name.title() + 'Schedule')


def loss_function(real, pred, mask):
    losses = {
        'loss': tf.reduce_mean(tf.boolean_mask(tf.abs(pred - real), mask)),
        'mse': tf.reduce_mean(tf.boolean_mask(tf.square(pred - real), mask)),
        # 'huber1': tf.losses.Huber(delta=1.)(real, pred, mask),
    }
    losses['log_loss'] = tf.math.log(losses['loss'])
    return losses


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


@tf.function
def train_step(feature, transformer, optimizer, train_losses, optimize_loss):
    enc_padding_mask = create_padding_mask(feature['atom'])

    with tf.GradientTape() as tape:
        predictions, _ = transformer(feature, True, enc_padding_mask)
        losses = loss_function(feature['target'], predictions, feature['target_mask'])

    gradients = tape.gradient(losses[optimize_loss], transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    for loss in LOSSES:
        train_losses[loss](losses[loss])


def eval_step(feature, transformer):
    enc_padding_mask = create_padding_mask(feature['atom'])

    predictions, _ = transformer(feature, False, enc_padding_mask)
    losses = loss_function(feature['target'], predictions, feature['target_mask'])

    return losses