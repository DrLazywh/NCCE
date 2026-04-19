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
from utils.my_metrics import Evaluator
import argparse
from utils.iotools import load_train_configs
from NPVA.NCNV import ncnv7


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TranTextReID Text")
    sub = './run_logs/CUHK-PEDES/20241122_161020_RDE_TAL+sr0.3_tau0.025_margin0.1_n0.5'
    parser.add_argument("--config_file", default=f'{sub}/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.training = True
    logger = setup_logger('RDE', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    args.output_dir =sub
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    
    
    asss = ['best.pth','last.pth']
    for i in range(len(asss)):
        if os.path.exists(op.join(args.output_dir, asss[i])):
            model = build_model(args,num_classes)
            checkpointer = Checkpointer(model)
            checkpointer.load(f=op.join(args.output_dir, asss[i]))
            model = model.cuda()
            do_inference(model, val_img_loader, val_txt_loader, train_loader)
            
    real_labels = train_loader.dataset.real_correspondences
    
    
    
    
    
    pred_C = ncnv7(model, "cuda", args, train_loader, 0.57, num_neighbor=3)
    res1 = torch.logical_and(torch.from_numpy(real_labels), pred_C.to(torch.int64))
    res2 = torch.logical_xor(torch.from_numpy(real_labels), pred_C.to(torch.int64))
#         print(type(pred_A))
#         print(pred_A.shape)
    # consensus_division = pred_A + pred_B # 0,1,2 
    # consensus_division[consensus_division==1] += torch.randint(0, 2, size=(((consensus_division==1)+0).sum(),))
    # label_hat = consensus_division.clone()
    # label_hat[consensus_division>1] = 1
    # label_hat[consensus_division<=1] = 0 
    # print(label_hat[0:30])
#         print("RDE")
#         print(label_hat.tolist().count(1))
#         print(label_hat.tolist().count(1) / 68126)

    logger.info("ncnv")
    logger.info(str(pred_C.tolist().count(1)))
    logger.info(str(pred_C.tolist().count(1) / 68126))

    logger.info("and")
    logger.info(str(res1.tolist().count(1)))
    logger.info(str(res1.tolist().count(1) / pred_C.tolist().count(1)))

    logger.info("XOR")
    logger.info(str(res2.tolist().count(1)))
    logger.info(str(res2.tolist().count(1) / 68126))
    
    

