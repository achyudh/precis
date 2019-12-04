import os

from lib import Config


class Code2VecConfig(Config):
    def __init__(self):
        parser = self.get_args()
        parser.add_argument('--top-k', type=int, default=1)
        parser.add_argument('--max-contexts', type=int, default=200)
        parser.add_argument('--path-embedding-dim', type=int, default=128)
        parser.add_argument('--token-embedding-dim', type=int, default=128)
        parser.add_argument('--max-token-vocab-size', type=int, default=1301136)
        parser.add_argument('--max-path-vocab-size', type=int, default=911417)
        parser.add_argument('--max-target-vocab-size', type=int, default=261245)

        args = parser.parse_args()
        args.context_vector_size = args.path_embedding_dim + 2 * args.token_embedding_dim
        args.train_data_path = os.path.join(args.data_dir, args.dataset, 'train.csv')
        args.dev_data_path = os.path.join(args.data_dir, args.dataset, 'dev.csv')
        args.test_data_path = os.path.join(args.data_dir, args.dataset, 'test.csv')
        args.word_freq_dict_path = os.path.join(args.data_dir, args.dataset, 'dict.bin')

        self.__dict__.update(args.__dict__)
