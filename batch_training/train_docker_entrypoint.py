import os
import sys
import subprocess
import json


os.environ['PERSONAL_ACCESS_TOKEN'] = json.loads(os.environ["github_personal_access_token"])['PERSONAL_ACCESS_TOKEN']
os.environ["WANDB_API_KEY"] = json.loads(os.environ["lynxight-wandb-api-key"])['WANDB_API_KEY']


def set_parser():
    parser = default_argument_parser()
    parser.add_argument("--train-name", type=str,
    help="name for training, taken from wandb if empty", default="")
    parser.add_argument("--branch", type=str, default="develop")
    parser.add_argument("--commit", type=str, default="develop")
    return parser


def main():
    args = set_parser().parse_args()

    # -------------CLONE GIT REPO--------------#
    clone_cmd = ["git",
    "clone",
    "--depth",
    "1",
    "-b",
    f"{args.branch}",
    f"https://{os.environ['PERSONAL_ACCESS_TOKEN']}@github.com/Lynxight/LxTrainFramework.git",
    ]
    checkout_cmd = "git checkout <commit_sha>"

    print(f"Cloned repo on branch {args.branch}", flush=True)
    subprocess.run(clone_cmd)

    # ------------RUN TRAINING-----------------#
    os.chdir("/MAFAT-SATTELLITE-VISION-CHALLENGE")
    train_cmd = ["dvc exp run"]
    try:
        subprocess.run(train_cmd, check=True, env={**os.environ, "PYTHONPATH": os.path.abspath("../..")})
    except subprocess.CalledProcessError as ex:
        print(f"process did not completed successfully, exited with code:{ex.returncode}")
    sys.exit(ex.returncode)


if __name__ == "__main__":
    main()
