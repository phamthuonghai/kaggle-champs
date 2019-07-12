class Default:
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    rel_att = (0, 1, 2, 3)
    shared_rel_att = False
    warmup_steps = 1000


def default():
    return Default()
