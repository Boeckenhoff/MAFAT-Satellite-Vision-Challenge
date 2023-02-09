import os
import sys
import subprocess
import json


os.environ['PERSONAL_ACCESS_TOKEN'] = json.loads(os.environ["github_personal_access_token"])['PERSONAL_ACCESS_TOKEN']
os.environ["WANDB_API_KEY"] = json.loads(os.environ["lynxight-wandb-api-key"])['WANDB_API_KEY']


def set_parser():
    parser = default_argument_parser()
    parser.add_argument("--train-dataset-path", type=str,
    required=True, help="path to train json file")
    parser.add_argument("--val-dataset-path", type=str, default='empty', help="path to val json file, "
    "if no val dataset is given, "
    "set to 'empty'")
    parser.add_argument("--output-path", type=str, required=True,
    default='/opt/lynxight/train_framework_output')
    parser.add_argument("--override-config", type=str,
    help="Optional py/YAML config file for overriding base config file fields.")
    parser.add_argument("--train-name", type=str,
    help="name for training, taken from wandb if empty", default="")
    # general args
    parser.add_argument("--branch", type=str, default="develop")
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

    # -------------VALIDATE EFS MOUNT--------------#
    efs_cmf = ["df", "-h"]
    subprocess.run(efs_cmf)

    # ------------RUN TRAINING-----------#
    os.chdir("/LxTrainFramework")
    train_cmd = ["python3.8",
    "lx_train_v2.py",
    "--num-gpus",
    f"{args.num_gpus}",
    "--config-file",
    f"{args.config_file}",
    "--train-dataset-path",
    f"{args.train_dataset_path}",
    "--val-dataset-path",
    f"{args.val_dataset_path}",
    "--output-path",
    f"{args.output_path}",
    "--train-name",
    f"{args.train_name}"]

    if args.resume:
    train_cmd.extend(["--resume"])
    print("Resuming training", flush=True)

    try:
    subprocess.run(train_cmd, check=True, env={**os.environ, "PYTHONPATH": os.path.abspath("../..")})
    except subprocess.CalledProcessError as ex:
    print(f"process did not completed successfully, exited with code:{ex.returncode}")
    sys.exit(ex.returncode)


if __name__ == "__main__":
    main()
