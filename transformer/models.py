import tensorflow as tf

from utils import ATOM_NUM, BOND_NUM, NUM_ATOM_FEATURES, NUM_BOND_FEATURES, MAX_NUM_ATOMS


def scaled_dot_product_attention(q, k, v, mask, relative_position=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (batch_size, num_heads, seq_len_q, depth)
      k: key shape == (batch_size, num_heads, seq_len_k, depth)
      v: value shape == (batch_size, num_heads, seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
      relative_position: relative positional embeddings shape == (batch_size, seq_len_q, seq_len_k, depth)

    Returns:
      output, attention_weights
    """
    qk_matmul = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)

    if relative_position is None:
        logits = qk_matmul
    else:
        qt = tf.transpose(q, [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        # qtr_matmul (batch_size, seq_len_q, num_heads, seq_len_k)
        qtr_matmul = tf.matmul(qt, relative_position, transpose_b=True)
        qtr_matmul_t = tf.transpose(qtr_matmul, [0, 2, 1, 3])
        logits = qk_matmul + qtr_matmul_t

    # scale logits
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = logits / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, hparams):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = hparams.num_heads
        self.d_model = hparams.d_model

        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)

        self.dense = tf.keras.layers.Dense(self.d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, relative_position=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, relative_position)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, hparams):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(hparams)
        self.ffn = point_wise_feed_forward_network(hparams.d_model, hparams.dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(hparams.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(hparams.dropout_rate)

    def call(self, x, training, mask, relative_position):
        # (batch_size, input_seq_len, d_model)
        attn_output, attention_weights = self.mha(x, x, x, mask, relative_position)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attention_weights


class RelativeAttentionEmbedding(tf.keras.layers.Layer):
    def __init__(self, hparams, bond_embeddings=None):
        super(RelativeAttentionEmbedding, self).__init__()
        self.bond_embeddings = bond_embeddings
        self.bond_transforms = {
            'bond_type': tf.keras.layers.Dense(hparams.d_model // hparams.num_heads, activation='relu')
        }
        for bond_ft in NUM_BOND_FEATURES.keys():
            self.bond_transforms[bond_ft] = tf.keras.layers.Dense(
                hparams.d_model // hparams.num_heads, activation='relu')
        self.dist_transform = tf.keras.layers.Dense(hparams.d_model // hparams.num_heads, activation='relu')

    def call(self, feature):
        x = self.dist_transform(tf.concat([feature['dist'], tf.expand_dims(feature['spatial_distance'], axis=-1)], -1))
        x += self.bond_transforms['bond_type'](self.bond_embeddings['bond_type'](feature['bond_type']))
        for bond_ft in NUM_BOND_FEATURES.keys():
            x += self.bond_transforms[bond_ft](self.bond_embeddings[bond_ft](feature[bond_ft]))
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        self.num_layers = hparams.num_layers
        self.num_heads = hparams.num_heads
        self.d_model = hparams.d_model
        assert hparams.d_model % hparams.num_heads == 0
        self.depth = self.d_model // self.num_heads

        # Atom features' embeddings
        self.atom_embeddings = {'atom': tf.keras.layers.Embedding(ATOM_NUM, self.d_model)}
        for atom_ft, ft_size in NUM_ATOM_FEATURES.items():
            self.atom_embeddings[atom_ft] = tf.keras.layers.Embedding(ft_size, self.d_model)

        # Bond features' embeddings
        self.bond_embeddings = {'bond_type': tf.keras.layers.Embedding(BOND_NUM, self.depth)}
        for bond_ft, ft_size in NUM_BOND_FEATURES.items():
            self.bond_embeddings[bond_ft] = tf.keras.layers.Embedding(ft_size, self.depth)

        self.pos_encoding = tf.keras.layers.Dense(self.d_model, activation='relu')
        self.charge_encoding = tf.keras.layers.Dense(self.d_model, activation='relu')

        self.rel_att = [None] * self.num_layers
        if hparams.rel_att is not None:
            the_first = hparams.rel_att[0]
            self.rel_att[the_first] = RelativeAttentionEmbedding(hparams, self.bond_embeddings)
            for l in hparams.rel_att[1:]:
                self.rel_att[l] = self.rel_att[the_first] if hparams.shared_rel_att else \
                    RelativeAttentionEmbedding(hparams, self.bond_embeddings)

        self.enc_layers = [EncoderLayer(hparams) for _ in range(hparams.num_layers)]

        self.dropout = tf.keras.layers.Dropout(hparams.dropout_rate)

    def call(self, feature, training, mask):
        # adding embedding and position encoding.
        x = self.atom_embeddings['atom'](feature['atom'])  # (batch_size, input_seq_len, d_model)
        for atom_ft in NUM_ATOM_FEATURES.keys():
            x += self.atom_embeddings[atom_ft](feature[atom_ft])
        x += self.charge_encoding(tf.stack(
            [feature['formal_charge'], feature['partial_charge']], -1))
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding(feature['pos'])

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            if self.rel_att[i] is not None:
                rel_att = self.rel_att[i](feature)
                x, att_weights = self.enc_layers[i](x, training, mask, rel_att)
            else:
                x, att_weights = self.enc_layers[i](x, training, mask)

        # (batch_size, input_seq_len, d_model), (batch_size, num_heads, input_seq_len, input_seq_len)
        return x, att_weights


def cartesian(a, b):
    tile_a = tf.tile(tf.expand_dims(a, 2), [1, 1, MAX_NUM_ATOMS, 1])
    tile_b = tf.tile(tf.expand_dims(b, 1), [1, MAX_NUM_ATOMS, 1, 1])
    return tf.concat([tile_a, tile_b], axis=-1)


class Model(tf.keras.Model):
    def __init__(self, hparams):
        super(Model, self).__init__()

        self.encoder = Encoder(hparams)

        self.final_layers = [tf.keras.layers.Dense(size, activation='relu') for size in hparams.classifier]
        self.final_layers += [tf.keras.layers.Dense(1)]

    def call(self, feature, training=False, mask=None):
        # (batch_size, input_seq_len, d_model), (batch_size, num_heads, input_seq_len, input_seq_len)
        enc_output, att_weights = self.encoder(feature, training, mask)

        att_weights = tf.transpose(att_weights, [0, 2, 3, 1])

        final_output = tf.concat([cartesian(enc_output, enc_output), att_weights], -1)

        for layer in self.final_layers:
            final_output = layer(final_output)

        final_output = tf.squeeze(final_output, [3])  # (batch_size, input_seq_len, input_seq_len)

        return final_output, att_weights
