"""script to benchmark speed of different convolutions"""
import torch
import argparse
import torch.utils.benchmark as benchmark

def get_params_str(module):
    num = sum([p.numel() for p in module.parameters()]) / 1e3
    return f"{num:.1f}k"

# def get_flop_str(module):
    

def adjust_for_bs(measurement):
    measurement.raw_times = [t / hparams.bs for t in measurement.raw_times]
    return measurement

# some simple args for the script
parser = argparse.ArgumentParser()
parser.add_argument("--half", action="store_true", help="flag to use fp16")
parser.add_argument("--bs", type=int, default=64, help="batch size")

hparams = parser.parse_args()

# Input for benchmarking
in_chs, out_chs = 128, 128
in_chs2 = 1024
# represent tensor at the beggining of ResNet-like model
inp = torch.randn(hparams.bs, in_chs, 224, 224).cuda()
# represent tensor in the middle of ResNet-like model
inp2 = torch.randn(hparams.bs, in_chs2, 32, 32).cuda()

num_threads = torch.get_num_threads()

conv = torch.nn.Conv2d(in_chs, out_chs, 3, 1, 1, bias=False).cuda()
conv_dw = torch.nn.Conv2d(in_chs, out_chs, 3, 1, 1, bias=False, groups=in_chs).cuda()
conv_sep = torch.nn.Sequential(
    torch.nn.Conv2d(in_chs, out_chs, 3, 1, 1, bias=False, groups=in_chs),
    torch.nn.Conv2d(in_chs, out_chs, 1, 1, 0, bias=False)
).cuda()

conv2 = torch.nn.Conv2d(in_chs2, in_chs2, 3, 1, 1, bias=False).cuda()
conv2_dw = torch.nn.Conv2d(in_chs2, in_chs2, 3, 1, 1, bias=False, groups=in_chs2).cuda()
conv2_sep = torch.nn.Sequential(
    torch.nn.Conv2d(in_chs2, in_chs2, 3, 1, 1, bias=False, groups=in_chs2),
    torch.nn.Conv2d(in_chs2, in_chs2, 1, 1, 0, bias=False)
).cuda()

if hparams.half:
    inp = inp.half()
    inp2 = inp2.half()
    conv = conv.half()
    conv_dw = conv_dw.half()
    conv_sep = conv_sep.half()
    conv2 = conv2.half()
    conv2_dw = conv2_dw.half()
    conv2_sep = conv2_sep.half()
    
label1 = f"Stem conv. Shape: {inp.shape}"
t0 = benchmark.Timer(
    stmt='conv(inp)',
    globals={'inp': inp, 'conv': conv},
    num_threads=num_threads,
    label=label1,
    sub_label=f'Reg Conv. Params: {get_params_str(conv)}',
    description='description',
).blocked_autorange(min_run_time=1)

t1 = benchmark.Timer(
    stmt='conv_dw(inp)',
    globals={'inp': inp, 'conv_dw': conv_dw},
    num_threads=num_threads,
    label=label1,
    sub_label=f'Conv DW. Params: {get_params_str(conv_dw)}',
    description='description',
).blocked_autorange(min_run_time=1)

t2 = benchmark.Timer(
    stmt='conv_sep(inp)',
    globals={'inp': inp, 'conv_sep': conv_sep},
    num_threads=num_threads,
    label=label1,
    sub_label=f'Conv Sep. Params: {get_params_str(conv_sep)}',
    description='description',
).blocked_autorange(min_run_time=1)

# same but for different input
label2 = f"Middle conv. Shape: {inp2.shape}"
t20 = benchmark.Timer(
    stmt='conv(inp)',
    globals={'inp': inp2, 'conv': conv2},
    num_threads=num_threads,
    label=label2,
    sub_label=f'Reg Conv. Params: {get_params_str(conv2)}',
    description='description',
).blocked_autorange(min_run_time=1)

t21 = benchmark.Timer(
    stmt='conv_dw(inp)',
    globals={'inp': inp2, 'conv_dw': conv2_dw},
    num_threads=num_threads,
    label=label2,
    sub_label=f'Conv DW. Params: {get_params_str(conv2_dw)}',
    description='description',
).blocked_autorange(min_run_time=1)

params = get_params_str(conv2_sep)
t22 = benchmark.Timer(
    stmt='conv_sep(inp)',
    globals={'inp': inp2, 'conv_sep': conv2_sep},
    num_threads=num_threads,
    label=label2,
    sub_label=f'Conv Sep. Params: {get_params_str(conv2_sep)}',
    description='description',
).blocked_autorange(min_run_time=1)

## divide speed by batch size
t0 = adjust_for_bs(t0)
t1 = adjust_for_bs(t1)
t2 = adjust_for_bs(t2)

t20 = adjust_for_bs(t20)
t21 = adjust_for_bs(t21)
t22 = adjust_for_bs(t22)

compare = benchmark.Compare([t0, t1, t2, t20, t21, t22])
compare.print()

# print(t0)
# print(dir(t0))
# print(t0.median)
# print(t1)
# print(t2)