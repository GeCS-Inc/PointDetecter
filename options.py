from argparse import ArgumentParser, Namespace

parser = ArgumentParser()
parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--no_logger", action='store_true')

def get_options():
    return parser.parse_args()