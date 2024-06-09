import logging


class Args(object):
    """
    This class is an interface, there is nothing should be changed
    """
    def __init__(self):
        self.dataset = None
        self.model = None
        self.is_recon = None
        self.run_time = None  # No need to change

        self.dataset_path = None  # No
        self.config_path = None  # No
        self.exp_path = None  # No
        self.exp_id = None

        self.win_size = None
        self.lr = None
        self.batch_size = None
        self.epochs = None

        self.top_k = None

        self.g_dim = None
        self.g_out_layer_num = None
        self.g_out_layer_inter_dim = None
        self.g_top_k = None


