import argparse
import os

import dataset_import
import torch
from DatasetManager.dataset_manager import DatasetManager
from Transformer.transformer import Transformer
from osc_server import OrchestraServer

# Debug mode
__DEBUG__ = False

parser = argparse.ArgumentParser()
parser.add_argument('--in_port', type=int, default=5000)
parser.add_argument('--out_port', type=int, default=5001)
parser.add_argument('--ip', type=str, default="127.0.0.1")
# Model arguments
parser.add_argument('--hierarchical', type=bool, default=False)
parser.add_argument('--nade', type=bool, default=False)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--per_head_dim', type=int, default=64)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--local_position_embedding_dim', type=int, default=8)
parser.add_argument('--position_ff_dim', type=int, default=2048)
parser.add_argument('--enc_dec_conditioning', type=str, default='split')
parser.add_argument('--dataset_type', type=str, default='arrangement_voice')
parser.add_argument('--conditioning', type=bool, default=True)
parser.add_argument('--double_conditioning', type=str, default=None)
parser.add_argument('--subdivision', type=int, default=16)
parser.add_argument('--sequence_size', type=int, default=7)
parser.add_argument('--velocity_quantization', type=int, default=2)
parser.add_argument('--max_transposition', type=int, default=12)
parser.add_argument('--instrument_presence_in_encoder', type=bool, default=False)
parser.add_argument('--block_attention', type=bool, default=False)
parser.add_argument('--suffix', type=str, default='REFERENCE')
args = parser.parse_args()


"""
###################
Load model
###################
"""
dropout = 0.
input_dropout = 0.
input_dropout_token = 0.
mixup = False
scheduled_training = 0.
group_instrument_per_section = False
reduction_flag = False
lr = 1.
cpc_config_name = None
subdivision = args.subdivision

# Use all gpus available
gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]
print(f'Using GPUs {gpu_ids}')
if len(gpu_ids) == 0:
    device = 'cpu'
else:
    device = 'cuda'

# Get dataset
dataset_manager = DatasetManager()
dataset, processor_decoder, processor_encoder, processor_encodencoder = \
    dataset_import.get_dataset(dataset_manager, args.dataset_type, args.subdivision, args.sequence_size,
                               args.velocity_quantization, args.max_transposition,
                               args.num_heads, args.per_head_dim, args.local_position_embedding_dim,
                               args.block_attention,
                               group_instrument_per_section, args.nade, cpc_config_name, args.double_conditioning,
                               args.instrument_presence_in_encoder)

# Load model
model = Transformer(dataset=dataset,
                    data_processor_encodencoder=processor_encodencoder,
                    data_processor_encoder=processor_encoder,
                    data_processor_decoder=processor_decoder,
                    num_heads=args.num_heads,
                    per_head_dim=args.per_head_dim,
                    position_ff_dim=args.position_ff_dim,
                    enc_dec_conditioning=args.enc_dec_conditioning,
                    hierarchical_encoding=args.hierarchical,
                    block_attention=args.block_attention,
                    nade=args.nade,
                    conditioning=args.conditioning,
                    double_conditioning=args.double_conditioning,
                    num_layers=args.num_layers,
                    dropout=dropout,
                    input_dropout=input_dropout,
                    input_dropout_token=input_dropout_token,
                    lr=lr, reduction_flag=reduction_flag,
                    gpu_ids=gpu_ids,
                    suffix=args.suffix,
                    mixup=mixup,
                    scheduled_training=scheduled_training
                    )

model.load_overfit(device=device)
model.to(device)
model = model.eval()

# Dir for writing generated files
writing_dir = f'{os.getcwd()}/generation'
if not os.path.isdir(writing_dir):
    os.makedirs(writing_dir)

# Create server
server = OrchestraServer(args.in_port,
                         args.out_port,
                         args.ip,
                         model=model,
                         subdivision=subdivision,
                         writing_dir=writing_dir)
# server.load_piano_score('/Users/leo/Recherche/Arte_orchestration/Orchestration/Databases/arrangement/source_for_generation/b_3_4_small.mid')
# server.orchestrate()

if (__DEBUG__):
    # Test pitch analysis
    print('[Debug mode : Testing server on given functions]')
else:
    print('[Running server on ports in : %d - out : %d]' % (args.in_port, args.out_port))
    server.run()
