import string
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

parser.add_argument("--model_card", type=str, default="facebook/mbart-large-cc25", help="Model Card")
parser.add_argument('--language', choices=['hindi', 'hiragana', 'arabic', 'russian', 'english', 'exclusivegerman', 'german'], required=True)

parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")

parser.add_argument("--case_sensitive", dest="case_sensitive", action="store_true", help="Consider lowercase and uppercase separately.")

parser.add_argument("--wandb", dest="wandb", action="store_true", default=False)

params = parser.parse_args()

assert params.model_card == 'facebook/mbart-large-cc25', "Apart from mbart, no other model supported yet"

HINDI_START_END = (2304, 2400)
HIRAGANA_START_END = (12353, 12447)
ARABIC_START_END = (1536, 1791)
RUSSIAN_START_END = (1040, 1103)
ENGLISH_START_END = list(string.ascii_lowercase)
EXCLUSIVE_GERMAN_START_END = ['ä', 'ö', 'ü', 'ß']
GERMAN_START_END = ENGLISH_START_END + EXCLUSIVE_GERMAN_START_END

PERMISSIBLE_CHAR_START_END = {'hindi': HINDI_START_END,
                            'hiragana': HIRAGANA_START_END,
                            'russian': RUSSIAN_START_END,
                            'arabic': ARABIC_START_END,
                            'english': ENGLISH_START_END,
                            'exclusivegerman': EXCLUSIVE_GERMAN_START_END,
                            'german': GERMAN_START_END
                        }[params.language]
