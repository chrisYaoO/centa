import argparse, os, time, torch, torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--size_mb", type=int, default=100)
parser.add_argument("--iters",   type=int, default=200)
args = parser.parse_args()

################ ① 初始化分布式 ################
dist.init_process_group(
    backend="nccl",
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    world_size=int(os.environ["WORLD_SIZE"]),
    rank=int(os.environ["RANK"])
)
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(0)

################ ② 构造“模型参数”tensor ################
num_elems = args.size_mb * 1024 * 1024 // 4   # float32=4 bytes
param = torch.empty(num_elems, dtype=torch.float32,
                    device="cuda").normal_()

################ ③ benchmark：all_reduce ################
# warm-up 10 次
for _ in range(10):
    dist.all_reduce(param, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()

# 正式计时
t0 = time.time()
for _ in range(args.iters):
    dist.all_reduce(param, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
elapsed = (time.time() - t0) / args.iters   # 单次平均

################ ④ 输出 ################
bytes_sent = args.size_mb * 1e6 * 2          # all_reduce：每节点收+发各一遍
bw = bytes_sent / elapsed / 1e9              # GiB/s

if rank == 0:
    print(f"[2-node] payload {args.size_mb} MB  "
          f"latency {elapsed*1e3:.2f} ms  "
          f"bw {bw:.2f} GB/s")

dist.destroy_process_group()
