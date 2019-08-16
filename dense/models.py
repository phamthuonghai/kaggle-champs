import tensorflow as tf

from utils import NUM_ATOM_FEATURES, NUM_BOND_FEATURES, ATOM_NUM, BOND_NUM


class Model(tf.keras.Model):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.d_model = hparams.d_model

        # Atom features' embeddings
        self.atom_embeddings = {'atom': tf.keras.layers.Embedding(ATOM_NUM, self.d_model)}
        for atom_ft, ft_size in NUM_ATOM_FEATURES.items():
            self.atom_embeddings[atom_ft] = tf.keras.layers.Embedding(ft_size, self.d_model)

        # Bond features' embeddings
        self.bond_embeddings = {'bond_type': tf.keras.layers.Embedding(BOND_NUM, self.d_model)}
        for bond_ft, ft_size in NUM_BOND_FEATURES.items():
            self.bond_embeddings[bond_ft] = tf.keras.layers.Embedding(ft_size, self.d_model)

        self.pos_encoding = tf.keras.layers.Dense(self.d_model, activation='relu')
        self.charge_encoding = tf.keras.layers.Dense(self.d_model, activation='relu')

        self.dropout = tf.keras.layers.Dropout(hparams.dropout_rate)

        self.final_layers = [tf.keras.layers.Dense(size, activation='relu') for size in hparams.classifier]
        self.final_layers += [tf.keras.layers.Dense(1)]

    def call(self, feature, training=False, mask=None):
        # Atom embeddings
        x = self.atom_embeddings['atom'](feature['atom'])  # (batch_size, input_seq_len, d_model)
        for atom_ft in NUM_ATOM_FEATURES.keys():
            x += self.atom_embeddings[atom_ft](feature[atom_ft])
        x = tf.concat([x, self.charge_encoding(tf.stack(
            [feature['formal_charge'], feature['partial_charge']], -1))])
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = tf.concat([x, self.pos_encoding(feature['pos'])], -1)

        # Bond embeddings
        x = tf.concat([feature['dist'], tf.expand_dims(feature['spatial_distance'], axis=-1)], -1)
        x += self.bond_transforms['bond_type'](self.bond_embeddings['bond_type'](feature['bond_type']))
        for bond_ft in NUM_BOND_FEATURES.keys():
            x += self.bond_transforms[bond_ft](self.bond_embeddings[bond_ft](feature[bond_ft]))

        final_output = self.dropout(x, training=training)

        for layer in self.final_layers:
            final_output = layer(final_output)
        final_output = tf.squeeze(final_output, [3])  # (batch_size, input_seq_len, input_seq_len)

        return final_output, None
