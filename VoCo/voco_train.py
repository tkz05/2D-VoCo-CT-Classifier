
import argparse
import os
from time import time
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from models.voco_head import VoCoHeadV2,VoCoHeadBatched
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import *
from utils.ops import *
from utils.utils import AverageMeter, distributed_all_gather
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))

from tqdm import tqdm
import json
import random
import numpy as np

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 可選，會影響速度
    torch.backends.cudnn.benchmark = False     # 可選，會影響速度

def main():

    set_seed(42)
    
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)
       
    def train_v2(args, global_step, train_loader, val_best, scaler, writer):
        model.train()
        loss_train = []
        training_logs = []
        run_loss = AverageMeter()
        intra_avg, inter_avg, reg_avg = AverageMeter(), AverageMeter(), AverageMeter()

        train_loader_tqdm = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc="Training Progress",
            ncols=150,
            dynamic_ncols=True,
        )

        for step, (img, labels, crops) in train_loader_tqdm:
            t1 = time()
            img, labels, crops = img.squeeze(0), labels.squeeze(0), crops.squeeze(0)
            img, crops, labels = img.cuda(), crops.cuda(), labels.cuda()

            with autocast(enabled=args.amp):
                try:
                    loss_intra, loss_inter, loss_reg = model(img, crops, labels)

                    alpha = 1.0
                    beta = 1.0
                    gamma = 1.0
                    loss = alpha * loss_intra + beta * loss_inter + gamma * loss_reg

                    if torch.isnan(loss):
                        print("❌ NaN loss detected. Skipping this batch.")
                        print(f"  intra_loss = {loss_intra.item()}")
                        print(f"  inter_loss = {loss_inter.item()}")
                        print(f"  reg_loss = {loss_reg.item()}")
                        continue

                    loss_train.append(loss.item())

                except Exception as e:
                    print(f"❌ Error during forward pass: {e}")
                    continue

            try:
                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
            except Exception as e:
                print(f"❌ Error during backward/step: {e}")
                optimizer.zero_grad()
                continue

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()

            run_loss.update(loss.item(), n=args.batch_size)
            intra_avg.update(loss_intra.item(), n=args.batch_size)
            inter_avg.update(loss_inter.item(), n=args.batch_size)
            reg_avg.update(loss_reg.item(), n=args.batch_size)

            lr = optimizer.param_groups[0]["lr"]

            # TensorBoard log
            writer.add_scalar("Loss/Total", run_loss.avg, global_step)
            writer.add_scalar("Loss/Intra", intra_avg.avg, global_step)
            writer.add_scalar("Loss/Inter", inter_avg.avg, global_step)
            writer.add_scalar("Loss/Reg", reg_avg.avg, global_step)
            writer.add_scalar("Learning Rate", lr, global_step)

            # tqdm status
            train_loader_tqdm.set_postfix(
                Loss=f"{run_loss.avg:.4f}",
                Intra=f"{intra_avg.avg:.4f}",
                Inter=f"{inter_avg.avg:.4f}",
                Reg=f"{reg_avg.avg:.4f}",
                LR=f"{lr:.2e}"
            )

            global_step += 1

            # Checkpoint saving
            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            if val_cond:
                checkpoint = {
                    "global_step": global_step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer,
                }
                save_ckp(checkpoint, logdir + "/model_current_epoch.pt")

            if global_step % 5000 == 0:
                torch.save(model.state_dict(), logdir + f"/model_pth/model_step{global_step}.pth")
                save_ckp({
                    "global_step": global_step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer,
                }, logdir + f"/model_step{global_step}.pt")

            # JSON log
            if global_step % 100 == 0 or global_step == 1:
                log_entry = {
                    "Step": int(global_step),
                    "Loss": float(run_loss.avg),
                    "Intra": float(intra_avg.avg),
                    "Inter": float(inter_avg.avg),
                    "Reg": float(reg_avg.avg),
                    "LR": float(lr),
                }
                training_logs.append(log_entry)
                with open(os.path.join(args.logdir, "training_logs.json"), "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

        return global_step, loss, val_best
    roi = 64
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="logs", type=str, help="directory to save logs")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=250000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=5000, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", default=True, help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str,
                        help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", default=True, help="do NOT use amp for training")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")

    args = parser.parse_args()
    logdir = args.logdir

    torch.cuda.set_device(0)

    args.amp = True
    torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    model = VoCoHeadV2(args) # V2
    # model = VoCoHeadBatched(args) # V1

    model.cuda()

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, amsgrad=True)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    global_step = 0
    if args.resume:
        print('resume from previous checkpoints')
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        model.load_state_dict(model_dict, strict=False)
        global_step = model_dict["global_step"]
        # optimizer = model_dict["optimizer"]["state_dict"]

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank])

    train_loader = get_loader(args)
    # for i, batch in enumerate(train_loader):
    #     img, labels, crops = batch
    #     print(f"Batch {i}: img shape: {img.shape}, labels shape: {labels.shape}, crops shape: {crops.shape}")
    #     break
    writer = SummaryWriter(log_dir=args.logdir)  # TensorBoard writer
    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    while global_step < args.num_steps:
        global_step, loss, best_val = train_v2(args, global_step, train_loader, best_val, scaler, writer)
    checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    writer.close()
    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), logdir + "final_model.pth")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), logdir + "final_model.pth")
    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    main()
