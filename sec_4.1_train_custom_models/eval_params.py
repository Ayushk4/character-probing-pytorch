import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=16)

parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")

parser.add_argument("--bert_type", type=str, required=True, help="Bert model to be used for training")

parser.add_argument("--task", type=str, required="true",
                        choices=["pos", "ner"],
                        help="Model type to train.")

parser.add_argument("--task_level", type=str, required="true",
                        choices=["sentence", "token"],
                        help="Model type to train.")

parser.add_argument("--weight_path", type=str, required=True)
parser.add_argument("--target_names", type=str, required=True)

params = parser.parse_args()

