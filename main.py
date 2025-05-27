import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train
import os
import numpy as np

# Set CUDA device order for NVLink
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# # 临时设置只使用单GPU来避免DataParallel问题
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用GPU 0

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis with BC-PMJRS')
parser.add_argument('-f', default='', type=str)

# Model selection
parser.add_argument('--model', type=str, default='BCPMJRS',
                    help='name of the model to use (MulT, BCPMJRS)')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true',
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='/root/temp/MAC-AHRI/data/',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=4,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# BC-PMJRS specific parameters
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='hidden dimension for BC-PMJRS (default: 256)')
parser.add_argument('--lambda_mi_min', type=float, default=0.1,
                    help='weight for MI minimization loss (default: 0.1)')
parser.add_argument('--lambda_mi_max', type=float, default=0.1,
                    help='weight for MI maximization loss (default: 0.1)')
parser.add_argument('--use_adaptive_opt', action='store_true', default=True,
                    help='use adaptive optimizer with LTP/LTD (default: True)')
parser.add_argument('--ltp_threshold', type=float, default=0.7,
                    help='threshold for long-term potentiation (default: 0.7)')
parser.add_argument('--ltd_threshold', type=float, default=0.3,
                    help='threshold for long-term depression (default: 0.3)')

# Tuning
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-4)')
parser.add_argument('--optim', type=str, default='AdamW',
                    help='optimizer to use (default: AdamW)')
parser.add_argument('--num_epochs', type=int, default=60,
                    help='number of epochs (default: 60)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='bcpmjrs_enhanced',
                    help='name of the trial (default: "bcpmjrs_enhanced")')

args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# Enable cuDNN autotuner for better performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}

criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True
        print(f"CUDA available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

# Load the dataset
print("Start loading the data....")
train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')

# Create generator for DataLoader based on CUDA availability
if use_cuda:
    # Use GPU 0 as the primary device for data loading
    g = torch.Generator(device='cuda:0')
    # For multi-GPU training, we still use cuda:0 for the generator
    # DataParallel will handle distribution to other GPUs
else:
    g = torch.Generator(device='cpu')

# Create DataLoaders with proper generator
# Remove num_workers temporarily to avoid multiprocessing issues with CUDA
if use_cuda:
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                             generator=g, pin_memory=False)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True,
                             generator=g, pin_memory=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True,
                            generator=g, pin_memory=False)
else:
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                             generator=g, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True,
                             generator=g, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True,
                            generator=g, num_workers=4)

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

# Set hyperparameters
hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')

# Print configuration
print("\n" + "="*50)
print("Configuration:")
print(f"  Model: {hyp_params.model}")
print(f"  Dataset: {dataset}")
print(f"  Batch size: {args.batch_size}")
print(f"  Learning rate: {args.lr}")
print(f"  Hidden dimension: {args.hidden_dim}")
print(f"  Number of epochs: {args.num_epochs}")
print(f"  Using adaptive optimizer: {args.use_adaptive_opt}")
print(f"  Lambda MI-min: {args.lambda_mi_min}")
print(f"  Lambda MI-max: {args.lambda_mi_max}")
print(f"  Modalities: L={args.lonly}, A={args.aonly}, V={args.vonly}")
print("="*50 + "\n")

if __name__ == '__main__':
    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)