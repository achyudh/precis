import os

from lib import Config


class Code2SeqConfig(Config):
    def __init__(self):
        parser = self.get_args()
        parser.add_argument('--top-k', type=int, default=1)
        parser.add_argument('--max-norm', type=float, default=5)
        parser.add_argument('--max-contexts', type=int, default=200)
        parser.add_argument('--max-subtokens', type=int, default=5)
        parser.add_argument('--max-path-nodes', type=int, default=9)
        parser.add_argument('--max-target-length', type=int, default=6)
        parser.add_argument('--encoder-hidden-dim', type=int, default=256)
        parser.add_argument('--decoder-hidden-dim', type=int, default=512)
        parser.add_argument('--node-embedding-dim', type=int, default=128)
        parser.add_argument('--target-embedding-dim', type=int, default=128)
        parser.add_argument('--subtoken-embedding-dim', type=int, default=128)
        parser.add_argument('--max-subtoken-vocab-size', type=int, default=190000)
        parser.add_argument('--max-node-vocab-size', type=int, default=27000)
        parser.add_argument('--max-target-vocab-size', type=int, default=27000)

        args = parser.parse_args()
        args.train_data_path = os.path.join(args.data_dir, args.dataset, 'train_seq.csv')
        args.dev_data_path = os.path.join(args.data_dir, args.dataset, 'dev_seq.csv')
        args.test_data_path = os.path.join(args.data_dir, args.dataset, 'test_seq.csv')
        args.word_freq_dict_path = os.path.join(args.data_dir, args.dataset, 'dict_seq.bin')

        self.__dict__.update(args.__dict__)
