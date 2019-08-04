import json
import logging
import pickle

import csv
import fire
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

BOND_TYPES = ['<PAD>', '1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']
BOND_IDS = dict([(v, k) for k, v in enumerate(BOND_TYPES)])
BOND_NUM = len(BOND_TYPES)
ATOM_TYPES = ['<PAD>', 'C', 'F', 'H', 'N', 'O']
ATOM_IDS = dict([(v, k) for k, v in enumerate(ATOM_TYPES)])
ATOM_NUM = len(ATOM_TYPES)

MAX_NUM_ATOMS = 30

MG_ATOM_FEATURES = ['hybridization', 'chirality', 'formal_charge', 'partial_charge', 'acceptor', 'donor', 'aromatic']
MG_BOND_FEATURES = ['graph_distance', 'same_ring', 'spatial_distance']

NUM_ATOM_FEATURES = {
    'hybridization': 7, 'chirality': 3, 'acceptor': 2, 'donor': 2, 'aromatic': 2
}
NUM_BOND_FEATURES = {'graph_distance': 11, 'same_ring': 2}


def check_atom_index(indices):
    for rid, vid in enumerate(indices):
        if rid != vid:
            return False
    return True


def extract_features(data, mg, is_test=False):
    n_lines = len(data)
    features = {
            'name': [],
            'atom': np.zeros((n_lines, MAX_NUM_ATOMS), np.int),
            'pos': np.zeros((n_lines, MAX_NUM_ATOMS, 3), np.float),

            'hybridization': np.zeros((n_lines, MAX_NUM_ATOMS), np.int),
            'chirality': np.zeros((n_lines, MAX_NUM_ATOMS), np.int),
            'formal_charge': np.zeros((n_lines, MAX_NUM_ATOMS), np.float),
            'partial_charge': np.zeros((n_lines, MAX_NUM_ATOMS), np.float),
            'acceptor': np.zeros((n_lines, MAX_NUM_ATOMS), np.bool),
            'donor': np.zeros((n_lines, MAX_NUM_ATOMS), np.bool),
            'aromatic': np.zeros((n_lines, MAX_NUM_ATOMS), np.bool),

            'target': np.zeros((n_lines, MAX_NUM_ATOMS, MAX_NUM_ATOMS), np.float),
            'target_mask': np.zeros((n_lines, MAX_NUM_ATOMS, MAX_NUM_ATOMS), np.bool),

            'bond_type': np.zeros((n_lines, MAX_NUM_ATOMS, MAX_NUM_ATOMS), np.int),
            'dist': np.zeros((n_lines, MAX_NUM_ATOMS, MAX_NUM_ATOMS, 3), np.float),

            'same_ring': np.zeros((n_lines, MAX_NUM_ATOMS, MAX_NUM_ATOMS), np.bool),
            'spatial_distance': np.zeros((n_lines, MAX_NUM_ATOMS, MAX_NUM_ATOMS), np.float),
            'graph_distance': np.zeros((n_lines, MAX_NUM_ATOMS, MAX_NUM_ATOMS), np.int),
    }
    cnt_id = 0
    for _id, row in tqdm(data.iterrows()):
        assert check_atom_index(row.atom_index),  f"Atom indices are not in order in {_id}"
        cur_mg = mg[_id] if _id in mg else None
        num_atoms = len(row.atom_index)

        if cur_mg is not None:
            assert all(ma['element'] == sa for ma, sa in zip(cur_mg['atoms'], row.atom)), \
                f"MG.pkl and structures.csv not matched in {_id}"
        # Atom features
        features['atom'][cnt_id, :num_atoms] = np.array([ATOM_IDS[a] for a in row.atom])

        if cur_mg is not None:
            for mg_atom_feature in MG_ATOM_FEATURES:
                features[mg_atom_feature][cnt_id, :num_atoms] = np.array([a[mg_atom_feature] for a in cur_mg['atoms']])

        features['pos'][cnt_id, :num_atoms, :] = np.array(list(zip(row.x, row.y, row.z)))

        # Bond features from structures.csv
        for ai0, ai1, bond, target in zip(row.atom_index_0, row.atom_index_1, row.type,
                                          np.zeros(len(row.atom_index_0)) if is_test else row.scalar_coupling_constant):
            features['target'][cnt_id, ai0, ai1] = features['target'][cnt_id, ai1, ai0] = target
            features['target_mask'][cnt_id, ai0, ai1] = features['target_mask'][cnt_id, ai1, ai0] = True
            features['bond_type'][cnt_id, ai0, ai1] = features['bond_type'][cnt_id, ai1, ai0] = BOND_IDS[bond]
            features['dist'][cnt_id, ai0, ai1, :] = features['pos'][cnt_id, ai0] - features['pos'][cnt_id, ai1]
            features['dist'][cnt_id, ai1, ai0, :] = features['pos'][cnt_id, ai1] - features['pos'][cnt_id, ai0]

        # Bond features from MG.pkl
        if cur_mg is not None:
            for pairs in cur_mg['atom_pairs']:
                ai0 = pairs['a_idx']
                ai1 = pairs['b_idx']
                for mg_bond_feature in MG_BOND_FEATURES:
                    features[mg_bond_feature][cnt_id, ai0, ai1] = features[mg_bond_feature][cnt_id, ai1, ai0] = \
                        pairs[mg_bond_feature]

        features['name'].append(_id)
        cnt_id += 1

    features['name'] = np.array(features['name'])

    print(f'Processed {cnt_id}/{n_lines}')
    print(f'Max '
          f'hybridization {np.max(features["hybridization"])}, '
          f'chirality {np.max(features["chirality"])}, '
          f'graph_distance {np.max(features["graph_distance"])}')

    for k, v in features.items():
        features[k] = v[:cnt_id]

    return features


def preprocess(input_folder='../input', data_folder='./data'):
    os.makedirs(data_folder, exist_ok=True)

    src_structures = os.path.join(input_folder, 'structures.csv')
    src_mg = os.path.join(input_folder, 'MG.pkl')
    src_train = os.path.join(input_folder, 'train.csv')
    src_test = os.path.join(input_folder, 'test.csv')
    dst_train = os.path.join(data_folder, 'train.pkl')
    dst_test = os.path.join(data_folder, 'test.pkl')

    print('Read data from %s' % src_structures)
    structures = pd.read_csv(src_structures).groupby('molecule_name').aggregate(list)
    print('Read data from %s' % src_mg)
    with open(src_mg, 'rb') as f:
        mg = pickle.load(f)
    print('Read data from %s' % src_train)
    train_data = pd.read_csv(src_train).groupby('molecule_name').aggregate(list)
    print('Read data from %s' % src_test)
    test_data = pd.read_csv(src_test).groupby('molecule_name').aggregate(list)
    train_data = train_data.join(structures, how='inner')
    test_data = test_data.join(structures, how='inner')

    print('Extracting train features')
    train_features = extract_features(train_data, mg)
    print('Writing train features to %s' % dst_train)
    with open(dst_train, 'wb') as f:
        pickle.dump(train_features, f)

    print('Extracting test features')
    test_features = extract_features(test_data, mg, is_test=True)
    print('Writing test features to %s' % dst_test)
    with open(dst_test, 'wb') as f:
        pickle.dump(test_features, f)


def get_dataset(filepath, batch_size, val=None, bond_target='all'):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        d = {}

        id_mask = None
        if bond_target != 'all':
            target_bond_mask = data['bond_type'] == BOND_IDS[bond_target]
            data['target_mask'] = np.logical_and(target_bond_mask, data['target_mask'])
            id_mask = np.logical_or.reduce(data['target_mask'], axis=(1, 2))

        for k, v in data.items():
            if id_mask is not None:
                v = v[id_mask]

            if v.dtype.type is np.bool_:
                d[k] = tf.constant(v, shape=v.shape, dtype=tf.bool)
            elif v.dtype.type is np.int64:
                d[k] = tf.constant(v, shape=v.shape, dtype=tf.int64)
            elif v.dtype.type is np.float64:
                d[k] = tf.constant(v, shape=v.shape, dtype=tf.float32)
            elif v.dtype.type is np.str_:
                d[k] = tf.constant(v, shape=v.shape, dtype=tf.string)

        d = tf.data.Dataset.from_tensor_slices(d)

        if val is not None:
            return d.take(val).batch(batch_size=batch_size, drop_remainder=False),\
                   d.skip(val).batch(batch_size=batch_size, drop_remainder=False)
        else:
            return d.batch(batch_size=batch_size, drop_remainder=False)


def convert_mg(input_path='../input/MG.json', output_path='../input/MG.pkl'):
    res = {}
    with open(input_path, 'r') as in_file:
        for line in tqdm(csv.reader(in_file, delimiter=',')):
            idx = 'dsgdb9nsd_' + line[0][4:]
            data = json.loads(line[1])
            res[idx] = data

    print(f'Saving {len(res)} molecules to {output_path}')
    with open(output_path, 'wb') as out_file:
        pickle.dump(res, out_file)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', level=logging.INFO)
    fire.Fire({
        'preprocess': preprocess,
        'convert_mg': convert_mg,
    })
