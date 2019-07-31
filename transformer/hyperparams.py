class Default:
    num_layers = 5
    d_model = 128
    dff = 128
    num_heads = 128
    dropout_rate = 0.1
    rel_att = (0, 1, 2, 3, 4)
    classifier = (128, 128)
    shared_rel_att = False
    warmup_steps = 1000


def default():
    return Default()
