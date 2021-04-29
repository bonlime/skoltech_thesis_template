"""script to benchmark speed of different 1x1 convolutions"""
import torch
import argparse
import torch.utils.benchmark as benchmark

def get_params_str(module):
    num = sum([p.numel() for p in module.parameters()]) / 1e3
    return f"{num:.1f}k"

def adjust_for_bs(measurement):
    measurement.raw_times = [t / hparams.bs for t in measurement.raw_times]
    return measurement

# some simple args for the script
parser = argparse.ArgumentParser()
parser.add_argument("--half", action="store_true", help="flag to use fp16")
parser.add_argument("--bs", type=int, default=64, help="batch size")

hparams = parser.parse_args()

# Input for benchmarking
"""
want to benchmark: 
* input conv1x1
* deeper conv1x1
"""

in_chs, out_chs = 16, 128
in_chs2 = 1024
# represent tensor at the beggining of ResNet-like model
inp = torch.randn(hparams.bs, in_chs, 224, 224).cuda()
# represent tensor in the middle of ResNet-like model
inp2 = torch.randn(hparams.bs, in_chs2, 32, 32).cuda()

num_threads = torch.get_num_threads()

conv_pw = torch.nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False).cuda()
conv_pw2 = torch.nn.Conv2d(in_chs2, in_chs2, 1, 1, 0, bias=False).cuda()

if hparams.half:
    inp = inp.half()
    inp2 = inp2.half()
    conv_pw = conv_pw.half()
    conv_pw2 = conv_pw2.half()
    
label1 = f"Stem conv. Shape: {inp.shape}"
t0 = benchmark.Timer(
    stmt='conv(inp)',
    globals={'inp': inp, 'conv': conv_pw},
    num_threads=num_threads,
    label="PW Stem convs",
#     sub_label=f'Reg Conv. Params: {get_params_str(conv)}',
    description='description',
).blocked_autorange(min_run_time=1)

t1 = benchmark.Timer(
    stmt='conv(inp)',
    globals={'inp': inp2, 'conv': conv_pw2},
    num_threads=num_threads,
    label="PW deeper convs",
#     sub_label=f'Conv DW. Params: {get_params_str(conv_dw)}',
    description='description',
).blocked_autorange(min_run_time=1)

## divide speed by batch size
t0 = adjust_for_bs(t0)
t1 = adjust_for_bs(t1)

compare = benchmark.Compare([t0, t1])
compare.print()