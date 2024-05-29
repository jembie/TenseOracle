import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to create a Job File from a template"
    )
    parser.add_argument("--time_quick", type=str, default="24:00:00")
    parser.add_argument("--time_slow", type=str, default="24:00:00")
    parser.add_argument("--strategy", type=str, default="PredictionEntropy")
    parser.add_argument("--filter_strategy", type=str, default="RandomFilter")
    parser.add_argument("--branch", type=str, default="main")

    parser.add_argument(
        "--template_address",
        type=str,
        required=True,
        help="Where is the template file?",
    )
    parser.add_argument(
        "--job_directory",
        type=str,
        required=True,
        help="Directory to save the final script in?",
    )
    return parser.parse_args()


def create_file():
    with open(args.template_address, mode="r", encoding="utf-8") as f:
        text = f.read()
    text = text.replace("[strategy]", args.strategy)
    text = text.replace("[filter_strategy]", args.filter_strategy)
    text = text.replace("[branch]", args.branch)
    text = text.replace("[job_directory]", args.job_directory)
    text_quick = text.replace("[time]", args.time_quick)
    text_slow = text.replace("[time]", args.time_slow)
    job_dir = Path(args.job_directory)
    job_dir.mkdir(parents=True, exist_ok=True)
    with open(job_dir.joinpath("job_quick.sh"), mode="w+", encoding="utf-8") as f:
        f.write(text_quick)

    with open(job_dir.joinpath("job_slow.sh"), mode="w+", encoding="utf-8") as f:
        f.write(text_slow)


if __name__ == "__main__":
    args = parse_args()
    create_file()
