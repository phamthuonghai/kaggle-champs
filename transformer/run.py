import os
import time

import fire
import tensorflow as tf

import hyperparams
from models import Transformer
from utils import get_dataset


LOSSES = ['log_loss', 'loss', 'mse', 'huber1']


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        return lr


def loss_function(real, pred, mask):
    losses = {
        'loss': tf.reduce_mean(tf.boolean_mask(tf.abs(pred - real), mask)),
        'mse': tf.reduce_mean(tf.boolean_mask(tf.square(pred - real), mask)),
        'huber1': tf.losses.Huber(delta=1.)(real, pred, mask),
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


def train(n_epochs=1000, train_batch_size=128, hparam_set='default', data_path='./data',
          log_freq=500, val_size=1000, bond_target='all', load_ckpt=False):
    hparams = hyperparams.get_hparams(hparam_set)
    checkpoint_path = f'./checkpoints/{bond_target}/{hparam_set}/'

    summary_writer = tf.summary.create_file_writer(f'./summaries/{bond_target}/{hparam_set}/')

    # Load data
    val_dataset, train_dataset = get_dataset(
        os.path.join(data_path, 'train.pkl'), train_batch_size, val_size, bond_target)

    with summary_writer.as_default():
        # Model def
        learning_rate = CustomSchedule(hparams.d_model, hparams.warmup_steps)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        train_losses = {}
        for loss in LOSSES:
            train_losses[loss] = tf.keras.metrics.Mean(name=f'train_{loss}')
        transformer = Transformer(hparams)

        # Checkpoint init
        ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            if load_ckpt:
                ckpt.restore(ckpt_manager.latest_checkpoint)
                print('Latest checkpoint restored!')
            else:
                print(f'{checkpoint_path} is not empty')
                exit(0)

        # Training
        for epoch in range(n_epochs):
            start = time.time()
            for loss in LOSSES:
                train_losses[loss].reset_states()

            for (batch, feature) in enumerate(train_dataset):
                train_step(feature, transformer, optimizer, train_losses, hparams.loss)

                if tf.equal(optimizer.iterations % log_freq, 0):
                    for loss in LOSSES:
                        tf.summary.scalar(f'train/{loss}', train_losses[loss].result(), step=optimizer.iterations)

                if batch % log_freq == 0:  # not similar to step-wise log_freq above
                    print('Epoch {} batch {} loss {:.4f}'.format(epoch + 1, batch, train_losses[hparams.loss].result()))

            eval_losses = [eval_step(feature, transformer) for feature in val_dataset]
            for loss in LOSSES:
                sep_loss = [l[loss] for l in eval_losses]
                sep_loss = sum(sep_loss) / len(sep_loss)
                tf.summary.scalar(f'val/{loss}', sep_loss, step=optimizer.iterations)
                if loss == hparams.loss:
                    print('Epoch {}, train_loss {:.4f}, eval_loss {:.4f}, in {:.2f}s'.format(
                        epoch + 1, train_losses[hparams.loss].result(), sep_loss, time.time() - start))

            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        # 'evaluate': evaluate,
        # 'predict': predict,
    })
