#coding=utf-8
import argparse
import sys


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt


def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--dataroot', default='./data', help='root of data')
    arg_parser.add_argument('--word2vec_path', default='./word2vec-768.txt', help='path of word2vector file path')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=0, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arg_parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')
    #### Dataset Configuration ####
    arg_parser.add_argument('--tag_bi', type=bool, default=True, help='whether to explicitly use BI tags, if False, use BI will be added in postprocessing')
    arg_parser.add_argument('--no_tag_bi', dest='tag_bi', action='store_false')
    arg_parser.add_argument('--dialogue', action='store_true', help='whether to use dialogue dataset')
    #### Common Encoder Hyperparams ####
    arg_parser.add_argument('--encoder_cell', default='LSTM', choices=['LSTM', 'GRU', 'RNN', 'BERT'], help='root of data')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--embed_size', default=1024, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
    arg_parser.add_argument('--num_layer', default=2, type=int, help='number of layer')
    arg_parser.add_argument('--checkpoint', default=None, help='path of checkpoint')
    #### Pretrained Backbone Hyperparams ####
    arg_parser.add_argument('--pretrained_model', default='bert', help='pretrained model name, e.g., bert, electra, etc.')
    arg_parser.add_argument('--finetune_pretrained', action='store_true', help='whether to finetune the pretrained backbone')
    arg_parser.add_argument('--finetune_lr', type=float, default=None, help='learning rate for finetuning')
    ## Custom Checkpoint for Pretrained Backbone ##
    arg_parser.add_argument('--load_pretrained', action='store_true', help='whether to load pretrained model')
    arg_parser.add_argument(
        '--pretrained_checkpoint_dir',
        default=None,
        type=str,
        help='path to pretrained checkpoint directory',
    )
    arg_parser.add_argument(
        "--pretrained_framework",
        default="pytorch",
        type=str,
        help="framework of pretrained model, e.g., pytorch, tensorflow, etc.",
    )
    arg_parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str,
        help="Path to the checkpoint."
    )
    arg_parser.add_argument(
        "--pretrained_config_path",
        default=None,
        type=str,
        help="The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.",
    )

    return arg_parser