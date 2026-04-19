from prettytable import PrettyTable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs

from NPVA.NCNV import ncnv7
from NPVA.get_PL import get_PL
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TranTextReID Text")
    # sub = './run_logs/CUHK-PEDES/20241206_112502_RDE_TAL+sr0.3_tau0.025_margin0.1_n0.5'
    sub = 'run_logs_bat_identitty/ICFG-PEDES/20250324_051931_RDE_TAL+sr0.3_tau0.025_margin0.1_n0.5_bs256'
    parser.add_argument("--config_file", default=f'{sub}/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.training = False
    logger = setup_logger('RDE', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    args.output_dir =sub
    test_img_loader, test_txt_loader, refer_txt_loader, num_classes = build_dataloader(args)
    args.training = True
    train_loader, val_img_loader, val_txt_loader, refer_txt_loader, num_classes, train_loader_select = build_dataloader(args)
    print(train_loader)
    asss = ['best.pth', 'last.pth']
    for i in range(len(asss)):
        if os.path.exists(op.join(args.output_dir, asss[i])):
            model = build_model(args,num_classes)
            checkpointer = Checkpointer(model)
            checkpointer.load(f=op.join(args.output_dir, asss[i]))
            model = model.cuda()
            do_inference(model, test_img_loader, test_txt_loader, refer_txt_loader, args)
    # pred_C = ncnv7(model, "cuda", args, train_loader, 0.35, epoch = 0)
    # logger.info("ncnv")
    # logger.info(str(pred_C.tolist().count(1)))
    # logger.info(str(pred_C.tolist().count(1) / 68126))
    # PL = get_PL(model, "cuda", args, train_loader, pred_C, logger)



