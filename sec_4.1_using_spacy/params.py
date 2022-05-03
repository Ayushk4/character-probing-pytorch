import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=4214)
parser.add_argument("--test_mode", type=str, default="True")
# parser.add_argument("--dataset_path", type=str, default="data")

parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--n_epochs", type=int, default=5)

parser.add_argument("--dummy_run", dest="dummy_run", action="store_true", help="To make the model run on only one training sample for debugging")
parser.add_argument("--control", dest="control", action="store_true", help="Run Control")
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")

parser.add_argument("--pos", dest="pos", action="store_true", default=False)
parser.add_argument("--ner", dest="ner", action="store_true", default=False)
parser.add_argument("--tag", dest="tag", action="store_true", default=False)

parser.add_argument("--wandb", dest="wandb", action="store_true", default=False)

params = parser.parse_args()
assert params.pos or params.tag or params.ner
