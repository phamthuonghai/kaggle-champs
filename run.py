import os
import time

import fire
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from ops import get_schedule, train_step, eval_step, predict_step, LOSSES
from utils import get_dataset, get_model, get_hparams


def train(n_epochs=1000, train_batch_size=256, model='transformer', hparam_set='default', data_path='./data',
          log_freq=500, val_size=1000, bond_target='all', load_ckpt=False):
    hparams = get_hparams(model, hparam_set)
    checkpoint_path = f'./checkpoints/{bond_target}/{model}/{hparam_set}/'

    summary_writer = tf.summary.create_file_writer(f'./summaries/{bond_target}/{model}/{hparam_set}/')

    # Load data
    val_dataset, train_dataset = get_dataset(
        os.path.join(data_path, 'train.pkl'), train_batch_size, val_size, bond_target)

    with summary_writer.as_default():
        # Model def
        learning_rate = get_schedule(hparams.lr)(hparams)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        train_losses = {}
        for loss in LOSSES:
            train_losses[loss] = tf.keras.metrics.Mean(name=f'train_{loss}')
        transformer = get_model(model)(hparams)

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


def predict(model='transformer', hparam_set='default', data_path='./data', bond_target='all', test_batch_size=2048,
            raw_test='../input/test.csv', checkpoint_path=''):
    hparams = get_hparams(model, hparam_set)
    if checkpoint_path == '':
        checkpoint_path = f'./checkpoints/{bond_target}/{hparam_set}/'
    # Load data
    test_dataset = get_dataset(
        os.path.join(data_path, 'test.pkl'), test_batch_size, bond_target=bond_target)
    transformer = get_model(model)(hparams)

    ckpt = tf.train.Checkpoint(transformer=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    raw_test_df = pd.read_csv(raw_test)
    if bond_target != 'all':
        raw_test_df = raw_test_df[raw_test_df['type'] == bond_target]
    raw_test_df['scalar_coupling_constant'] = None
    mol_test_df = raw_test_df.groupby('molecule_name').aggregate(list)
    raw_test_df.set_index('id', inplace=True)

    for (batch, feature) in tqdm(enumerate(test_dataset)):
        batch_predictions = predict_step(feature, transformer)
        for mol_name, pred in zip(feature['name'].numpy(), batch_predictions):
            mol_name = mol_name.decode()
            pred = pred.numpy()
            for _id, (test_id, id_0, id_1) in enumerate(zip(mol_test_df.at[mol_name, 'id'],
                                                            mol_test_df.at[mol_name, 'atom_index_0'],
                                                            mol_test_df.at[mol_name, 'atom_index_1'])):
                raw_test_df.at[test_id, 'scalar_coupling_constant'] = pred[id_0, id_1]

    output_file = f'output/{bond_target}_{hparam_set}.csv'
    raw_test_df.to_csv(output_file, columns=['scalar_coupling_constant'])


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'predict': predict,
    })
