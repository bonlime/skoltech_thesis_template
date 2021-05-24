"""
script to benchmark speed of stems SE vs ECA vs GAP
my belief is that ECA is not really faster
"""
import torch
import argparse
import torch.utils.benchmark as benchmark
from pytorch_tools.modules import FastGlobalAvgPool2d
from pytorch_tools.modules.residual import SEModule
from pytorch_tools.modules.residual import ECAModule

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
* input stem
* input deep

* GAP
* SE
* ECA
"""

# represent tensor in the middle of ResNet-like model
inp1 = torch.randn(hparams.bs, 32, 224, 224).cuda()
# represent tensor in the end of ResNet-like model
inp2 = torch.randn(hparams.bs, 256, 14, 14).cuda()

num_threads = torch.get_num_threads()

gap = FastGlobalAvgPool2d().cuda()
se = SEModule(32, 16).cuda()
se2 = SEModule(256, 64).cuda()
eca = ECAModule().cuda()

if hparams.half:
    inp1 = inp1.half()
    inp2 = inp2.half()
    gap = gap.half()
    se = se.half()
    se2 = se2.half()
    eca = eca.half()

all_res = []

label1 = f"Middle stem. Shape: {inp1.shape}"
all_res.append(
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp1, 'conv': gap},
        num_threads=num_threads,
        label=label1,
        sub_label="GAP",
        description='description',
    ).blocked_autorange(min_run_time=1)
)
all_res.append(
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp1, 'conv': se},
        num_threads=num_threads,
        label=label1,
        sub_label="SE(0.5)",
        description='description',
    ).blocked_autorange(min_run_time=1)
)
all_res.append(
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp1, 'conv': eca},
        num_threads=num_threads,
        label=label1,
        sub_label="ECA",
        description='description',
    ).blocked_autorange(min_run_time=1)
)

label2 = f"Deeper stem. Shape: {inp2.shape}"
all_res.append(
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp2, 'conv': gap},
        num_threads=num_threads,
        label=label2,
        sub_label="GAP",
        description='description',
    ).blocked_autorange(min_run_time=1)
)
all_res.append(
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp2, 'conv': se2},
        num_threads=num_threads,
        label=label2,
        sub_label="SE(0.5)",
        description='description',
    ).blocked_autorange(min_run_time=1)
)
all_res.append(
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp2, 'conv': eca},
        num_threads=num_threads,
        label=label2,
        sub_label="ECA",
        description='description',
    ).blocked_autorange(min_run_time=1)
)


## divide speed by batch size
all_res = [adjust_for_bs(i) for i in all_res]

compare = benchmark.Compare(all_res)
compare.print()