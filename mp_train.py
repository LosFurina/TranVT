import multiprocessing
import subprocess


def run_command_with_args(args):
    subprocess.run(args)


if __name__ == "__main__":

    args_list = []
    for win_size in range(2, 3):
        args = ["python", "train.py",
                "--save_pattern", "wadi_win_TranVTV",
                "--dataset", "wadi",
                "--model", "TranVTV",
                "--lr", "0.0001",
                f"--win_size={win_size}",
                "--batch_size", "128",
                "--epochs", "20"]
        args_list.append(args)

    with multiprocessing.Pool(processes=2) as pool:
        pool.map(run_command_with_args, args_list)
