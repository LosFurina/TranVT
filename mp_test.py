import multiprocessing
import subprocess
from pathlib import Path


def run_command_with_args(args):
    subprocess.run(args)


if __name__ == "__main__":
    save_pattern = "swat_win_TranVTV"
    path = Path(__file__).parent.joinpath("exp", save_pattern)
    print(path)

    args_list = []
    for item in path.glob("*"):
        if item.is_dir():
            print(item.name)

            args = ["python", "test.py",
                    "--test",
                    "--save_pattern", "swat_win_TranVTV",
                    "--exp_id", f"{item.name}",
                    "--top_k", "1",
                    ]
            args_list.append(args)

    with multiprocessing.Pool(processes=5) as pool:
        pool.map(run_command_with_args, args_list)
