import argparse

parser = argparse.ArgumentParser(description='PyTorch T-FedAvg')

# device
parser.add_argument('--device',     type=str,    default='cuda',     help= 'GPU or not')
parser.add_argument('--gpu_id',     type=int,    default=0,          help= 'number of gpu to use')

# federated learning
parser.add_argument('--rounds',     type=int,    default=100,        help= 'global epoch')
parser.add_argument('--local_e',    type=int,    default=5,          help= 'local epoch')
parser.add_argument('--batch_size', type=int,    default=64,         help= 'batch size')

# optimizer
parser.add_argument('--optimizer',  type=str,    default='Adam',     help= 'use Adam or SGD')
parser.add_argument('--lr',         type=float,  default=0.008,      help= 'learning rate')
parser.add_argument('--wd',         type=float,  default=0,          help= 'global weight decay for all trainable params')
parser.add_argument('--momentum',   type=float,  default=0.9,        help= 'value for momentum of NAG')
parser.add_argument('--nag_off',    dest='nag',  action='store_false',  default=True, help= 'if not to use nesterov acceleration')

# ternary quantization
parser.add_argument('--T_thresh',   type=float,  default=0.05,       help= 'fixed threshold')
parser.add_argument('--T_a',        type=float,  default=0.14,       help= 'vaule of adaptive quantization threshold at client')
parser.add_argument('--T_a_server', type=float,  default=0.05,       help= 'vaule of adaptive quantization threshold at server')
parser.add_argument('--fedmdl',     type=str,    default='s3',       help= 'quantization strategy of fed model')
parser.add_argument('--ada_thresh_off',  dest='ada_thresh',  action='store_false', default=True,  help= 'not to use adaptive threshold')

# other
parser.add_argument('--partial',         dest='partial',     action='store_true',  default=False, help= 'only quantize Conv4x')
parser.add_argument('--save_record_off', dest='save_record', action='store_false', default=True,  help= 'not to save the training records to csv file')
parser.add_argument('--train_conv1_off', dest='train_conv1', action='store_false', default=True,  help= 'not to train conv1 layer')

# the following arguments are no longer needed to alter, just leave them as their default forms
parser.add_argument('--dataset',    type=str,    default='cifar10',  help= 'dataset')
parser.add_argument('--model',      type=str,    default='ResNet',   help= 'model')
parser.add_argument('--frac',       type=float,  default=1,          help= 'participation ratio')
parser.add_argument('--num_C',      type=int,    default=3,          help= 'client number')
parser.add_argument('--Nc',         type=int,    default=10,         help= 'class number on clients')
parser.add_argument('--iid',        type=bool,   default=True,       help= 'iid or not')
parser.add_argument('--seed',       type=int,    default=1234,       help= 'seed')


Args = parser.parse_args()
