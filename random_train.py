import subprocess

from tqdm import tqdm


class RandomTrain(object):

    def __init__(self):
        self.dataset = "swat"
        self.model = "GTranVTV"
        self.lr = 0.0001
        self.win_size = 20
        self.g_dim = 64
        self.g_out_layer_num = 1
        self.g_out_layer_inter_dim = 256
        self.g_top_k = 40
        self.batch_size = 32
        self.epochs = 10
        self.recon = None
        self.train_ratio = 1
        self.train_times = 1
        self.save_pattern = f"{self.dataset}_win{self.win_size}_{self.model}_{self.recon}_{self.train_ratio}"

    def train(self):
        if self.recon is not None:
            cmd = f"python train.py \
            --recon \
            --save_pattern {self.save_pattern} \
            --dataset {self.dataset} \
            --model {self.model} --lr {self.lr} \
            --win_size {self.win_size} \
            --g_dim {self.g_dim} \
            --g_out_layer_num {self.g_out_layer_num} \
            --g_out_layer_inter_dim {self.g_out_layer_inter_dim} \
            --g_top_k {self.g_top_k} \
            --batch_size {self.batch_size} \
            --epochs {self.epochs} \
            --train_ratio {self.train_ratio}"
            print("CMD:", cmd)
        else:
            cmd = f"python train.py \
            --save_pattern {self.save_pattern} \
            --dataset {self.dataset} \
            --model {self.model} --lr {self.lr} \
            --win_size {self.win_size} \
            --g_dim {self.g_dim} \
            --g_out_layer_num {self.g_out_layer_num} \
            --g_out_layer_inter_dim {self.g_out_layer_inter_dim} \
            --g_top_k {self.g_top_k} \
            --batch_size {self.batch_size} \
            --epochs {self.epochs} \
            --train_ratio {self.train_ratio}"
            print("CMD:", cmd)

        pbar = tqdm(range(self.train_times), desc="Training progress")
        for i in pbar:
            pbar.set_description(desc=f"Training iteration {i}")
            subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main = RandomTrain()
    main.train()
