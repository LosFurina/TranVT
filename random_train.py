import subprocess

from tqdm import tqdm


class RandomTrain(object):

    def __init__(self):
        self.dataset = "wadi"
        self.model = "TranVTV"
        self.lr = 0.0001
        self.win_size = 50
        self.batch_size = 32
        self.epochs = 10
        self.recon = "recon"
        self.train_ratio = 0.8
        self.train_times = 2
        self.save_pattern = f"{self.dataset}_win{self.win_size}_{self.model}_{self.recon}_{self.train_ratio}"

    def train(self):
        if self.recon is not None:
            cmd = f"python train.py --recon --save_pattern {self.save_pattern} --dataset {self.dataset} --model {self.model} --lr {self.lr} --win_size {self.win_size} --batch_size {self.batch_size} --epochs {self.epochs} --train_ratio {self.train_ratio}"
            print("CMD:", cmd)
        else:
            cmd = f"python train.py --save_pattern {self.save_pattern} --dataset {self.dataset} --model {self.model} --lr {self.lr} --win_size {self.win_size} --batch_size {self.batch_size} --epochs {self.epochs} --train_ratio {self.train_ratio}"
            print("CMD:", cmd)

        pbar = tqdm(range(self.train_times), desc="Training progress")
        for i in pbar:
            pbar.set_description(desc=f"Training iteration {i}")
            subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main = RandomTrain()
    main.train()
