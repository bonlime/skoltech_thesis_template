"""script to benchmark speed of stems space2depth vs conv2x2(s) vs default conv7x7"""
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
* input
* input (32 channels)


* conv7x7 input
* space2depth
* conv2x2 (s)
"""

# represent tensor at the beggining of ResNet-like model
inp1 = torch.randn(hparams.bs, 3, 224, 224).cuda()
# represent tensor in the middle of ResNet-like model
inp2 = torch.randn(hparams.bs, 32, 224, 224).cuda()

num_threads = torch.get_num_threads()

# @torch.jit.script
# class SpaceToDepthJIT:
#     def __init__(self, block_size: int = 2):
#         self.block_size = block_size 
        
#     def call (self, x: torch.Tensor):
#         BS = self.block_size
#         N, C, H, W = x.size()
#         x = x.view(N, C, H // BS, BS, W // BS, BS)
#         x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
#         x = x.view(N,C * BS * BS,H//BS,W//BS)
#         return x

class SpaceToDepth(torch.nn.Module):
    def __init__(self, block_size: int = 2):
        super().__init__()
        self.block_size = block_size 
        
    def forward(self, x: torch.Tensor):
        BS = self.block_size
        N, C, H, W = x.size()
        x = x.view(N, C, H // BS, BS, W // BS, BS)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N,C * BS * BS,H//BS,W//BS)
        return x


conv7x7 = torch.nn.Conv2d(3, 32, 7, 2, 3, bias=False).cuda()
conv7x7_2 = torch.nn.Conv2d(32, 32, 7, 2, 3, bias=False).cuda()
conv7x7_maxpool = torch.nn.Sequential( # output is 12 to match s2d
    torch.nn.Conv2d(3, 12, 7, 2, 3, bias=False),
    torch.nn.MaxPool2d(2, 2),
).cuda()

conv2x2 = torch.nn.Conv2d(3, 32, 2, 2, 0, bias=False).cuda()
conv2x2_2 = torch.nn.Conv2d(32, 32, 2, 2, 0, bias=False).cuda()

conv3x3 = torch.nn.Conv2d(3, 32, 3, 2, 1, bias=False).cuda()

s2d_jit = torch.jit.script(SpaceToDepth())

space2depth = torch.nn.Sequential(
    SpaceToDepth(),
    torch.nn.Conv2d(3 * 4, 32, 3, 1, 1, bias=False).cuda()
)
space2depth_2 = torch.nn.Sequential(
    SpaceToDepth(),
    torch.nn.Conv2d(32 * 4, 32, 3, 1, 1, bias=False).cuda()
)

space2depth_jit = torch.nn.Sequential(
    s2d_jit,
    torch.nn.Conv2d(3 * 4, 32, 3, 1, 1, bias=False).cuda()
)

space2depth_jit_2 = torch.nn.Sequential(
    s2d_jit,
    torch.nn.Conv2d(32 * 4, 32, 3, 1, 1, bias=False).cuda()
)

space2depth4x4 = SpaceToDepth(4)
space2depth2x2 = SpaceToDepth()


if hparams.half:
    pp = [
#         inp1,
#         inp2,
        conv7x7,
        conv7x7_2,
        conv2x2,
        conv2x2_2,
        space2depth,
        space2depth_2,
        space2depth_jit,
        space2depth_jit_2,
    ]
    inp1 = inp1.half()
    inp2 = inp2.half()
    conv7x7 = conv7x7.half()
    conv7x7_2 = conv7x7_2.half()
    conv7x7_maxpool = conv7x7_maxpool.half()
    conv2x2 = conv2x2.half()
    conv2x2_2 = conv2x2_2.half()
    conv3x3 = conv3x3.half()
    space2depth = space2depth.half()
    space2depth_2 = space2depth_2.half()
    space2depth_jit = space2depth_jit.half()
    space2depth_jit_2 = space2depth_jit_2.half()


    
o = [conv7x7(inp1).shape, conv2x2(inp1).shape, conv3x3(inp1).shape, space2depth(inp1).shape, space2depth_jit(inp1).shape]
assert o[0] == o[1] and o[0] == o[2] and o[0] == o[3] and o[0] == o[4] # and o[0] == o[5] and o[0] == o[4]
label1 = f"RGB stem. Shape: {inp1.shape}"
all_res = []
all_res.append(
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp1, 'conv': conv7x7},
        num_threads=num_threads,
        label=label1,
        sub_label="conv7x7",
        description='description',
    ).blocked_autorange(min_run_time=1)
)

all_res.append(
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp1, 'conv': conv2x2},
        num_threads=num_threads,
        label=label1,
        sub_label="conv2x2",
        description='description',
    ).blocked_autorange(min_run_time=1)
)

all_res.append(
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp1, 'conv': conv3x3},
        num_threads=num_threads,
        label=label1,
        sub_label="conv3x3",
        description='description',
    ).blocked_autorange(min_run_time=1)
)

all_res.append(
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp1, 'conv': space2depth},
        num_threads=num_threads,
        label=label1,
        sub_label="space2depth",
        description='description',
    ).blocked_autorange(min_run_time=1)
)

all_res.append(
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp1, 'conv': space2depth_jit},
        num_threads=num_threads,
        label=label1,
        sub_label="space2depth_jit",
        description='description',
    ).blocked_autorange(min_run_time=1)
)

all_res.append( # default R50 stem
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp1, 'conv': conv7x7_maxpool},
        num_threads=num_threads,
        label=label1,
        sub_label="conv7x7 maxpool (OS=4)",
        description='description',
    ).blocked_autorange(min_run_time=1)
)

all_res.append( # default TResNet50 stem
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp1, 'conv': space2depth4x4},
        num_threads=num_threads,
        label=label1,
        sub_label="space2depth 4x4",
        description='description',
    ).blocked_autorange(min_run_time=1)
)

all_res.append(
    benchmark.Timer(
        stmt='conv(inp)',
        globals={'inp': inp1, 'conv': space2depth2x2},
        num_threads=num_threads,
        label=label1,
        sub_label="space2depth 2x2",
        description='description',
    ).blocked_autorange(min_run_time=1)
)

label2 = f"Deeper stem. Shape: {inp2.shape}"



## divide speed by batch size
all_res = [adjust_for_bs(i) for i in all_res]

compare = benchmark.Compare(all_res)
compare.print()