import subprocess

from tqdm import tqdm


class RandomTrain(object):

    def __init__(self):
        self.dataset = "wadi_less"
        self.model = "GumbelGraphormer"
        self.lr = 0.0001
        self.win_size = 90
        self.g_dim = 64
        self.g_out_layer_num = 1
        self.g_out_layer_inter_dim = 256
        self.g_top_k = 40
        self.temp = 1
        self.temp_drop_frac = 0.99
        self.batch_size = 32
        self.epochs = 5
        self.recon = None
        self.train_ratio = 1
        self.train_times = 1
        recon = "recon" if self.recon is not None else "pred"
        self.save_pattern = f"{self.dataset}_win{self.win_size}_{self.model}_{recon}_{self.train_ratio}"

    def train(self):
        cmd = f"python train.py \
        --save_pattern {self.save_pattern} \
        --dataset {self.dataset} \
        --model {self.model} --lr {self.lr} \
        --win_size {self.win_size} \
        --g_dim {self.g_dim} \
        --g_out_layer_num {self.g_out_layer_num} \
        --g_out_layer_inter_dim {self.g_out_layer_inter_dim} \
        --g_top_k {self.g_top_k} \
        --temp {self.temp} \
        --temp_drop_frac {self.temp_drop_frac} \
        --batch_size {self.batch_size} \
        --epochs {self.epochs} \
        --train_ratio {self.train_ratio}"

        if self.recon is None:
            cmd = cmd
            print("CMD:", cmd)
        else:
            cmd = cmd + f" --recon"
            print("CMD:", cmd)

        pbar = tqdm(range(self.train_times), desc="Training progress")
        for i in pbar:
            pbar.set_description(desc=f"Training iteration {i}")
            subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main = RandomTrain()
    main.train()
