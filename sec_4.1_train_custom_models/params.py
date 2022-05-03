import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=4214)

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--n_epochs", type=int, default=5)

parser.add_argument("--dummy_run", dest="dummy_run", action="store_true", help="To make the model run on only one training sample for debugging")
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")

parser.add_argument("--bert_type", type=str, required=True, help="Bert model to be used for training")

parser.add_argument("--task", type=str, required="true",
                        choices=["pos", "ner"],
                        help="Model type to train.")

parser.add_argument("--task_level", type=str, required="true",
                        choices=["sentence", "token"],
                        help="Model type to train.")

parser.add_argument("--wandb", dest="wandb", action="store_true", default=False)

params = parser.parse_args()

