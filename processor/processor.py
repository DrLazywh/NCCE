import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
import numpy as np
from matplotlib import pyplot as plt
from pylab import xticks,yticks,np
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from NPVA.NCNV import ncnv7
from NPVA.NCNV import ncnv7_for_more_pairs
from NPVA.NCNV import ncnv7_pro
from NPVA.get_PL import get_PL
from utils.checkpoint import Checkpointer
import os.path as op
################### CODE FOR THE BETA MODEL  ########################

import scipy.stats as stats
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)


def split_prob(prob, threshld):
    if prob.min() > threshld:
        """From https://github.com/XLearning-SCU/2021-NeurIPS-NCR"""
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)

def get_loss(model, data_loader):
    logger = logging.getLogger("RDE.train")
    model.eval()
    device = "cuda"
    data_size = data_loader.dataset.__len__()
    real_labels = data_loader.dataset.real_correspondences
    lossA, lossB, simsA,simsB = torch.zeros(data_size), torch.zeros(data_size), torch.zeros(data_size),torch.zeros(data_size)
    for i, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        index = batch['index']
        with torch.no_grad(): 
            la, lb, sa, sb = model.compute_per_loss(batch)
            for b in range(la.size(0)):
                lossA[index[b]]= la[b]
                lossB[index[b]]= lb[b]
                simsA[index[b]]= sa[b]
                simsB[index[b]]= sb[b]
            if i % 100 == 0:
                logger.info(f'compute loss batch {i}')

    losses_A = (lossA-lossA.min())/(lossA.max()-lossA.min())    
    losses_B = (lossB-lossB.min())/(lossB.max()-lossB.min())
    
    input_loss_A = losses_A.reshape(-1,1) 
    input_loss_B = losses_B.reshape(-1,1)
 
    logger.info('\nFitting GMM ...') 
 
    if model.args.noisy_rate > 0.4 or model.args.dataset_name=='RSTPReid':
        # should have a better fit 
        gmm_A = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
        gmm_B = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    else:
        gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]

    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]
    print(type(prob_B))
 
    pred_A = split_prob(prob_A, 0.5)
    pred_B = split_prob(prob_B, 0.5)
  
    return torch.Tensor(pred_A), torch.Tensor(pred_B), torch.Tensor(prob_A), torch.Tensor(prob_B)




def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, train_loader_select):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("RDE.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "bge_loss": AverageMeter(),
        "tse_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "crossid_loss": AverageMeter(),
        "bge_img_acc": AverageMeter(),
        "tse_img_acc": AverageMeter(),
        "bge_txt_acc": AverageMeter(),
        "tse_txt_acc": AverageMeter(),
        "txt_id_loss": AverageMeter(),
        "relabel_num": AverageMeter(),
        "relabel_acc": AverageMeter()
    }
    tb_writer = SummaryWriter(log_dir=args.output_dir)
    
    best_top1 = 0.0
    # evaluator.eval(model.eval())
    # train
    sims = []
    for epoch in range(start_epoch, num_epoch + 1):
        logger.info("GCE(q = 0.7) + delta=0.42 + new ncnv7 + bge+tse sims + no reg + tau0.025")
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        
        
        # sub = './run_logs/CUHK-PEDES/20250305_065026_RDE_TAL+sr0.3_tau0.025_margin0.1_n0.5' 
        # checkpointer = Checkpointer(model)
        # checkpointer.load(f=op.join(sub, "last.pth"))
        # model = model.cuda()
        
        logger.info("")
        model.train()
        model.epoch = epoch
        # data_size = train_loader.dataset.__len__()
        # pred_A, pred_B  =  torch.ones(data_size), torch.ones(data_size)
        real_labels = train_loader.dataset.real_correspondences
        print(real_labels)
        # pred_A, pred_B, prob_A, prob_B = get_loss(model, train_loader)
        # pred_C = ncnv7(model, "cuda", args, train_loader_select, 0.35, epoch = epoch)
        # if epoch > 5:
        #     args.tau = 0.025
        #     h = 0.35
        # else:
        #     args.tau = 0.04
        #     h = 0.37
        
        # if epoch < 19:
        #     h = 0.35
        # else:
        #     h = 0.42
        h = 0.42
        print(h)
        # epoch = 60
        # pred_C = torch.randint(0, 2, (len(real_labels),))
        pred_C = ncnv7(model, "cuda", args, train_loader, h, epoch = epoch, num_neighbor=args.num_neighbor)
        if torch.all(pred_C == 1) or epoch < 210:
            PL_labels_selected, PL_labels_selected_index = None, None
        else:
            print("getting PL")
            PL_labels_selected, PL_labels_selected_index = get_PL(model, "cuda", args, train_loader, pred_C, logger)
        # pred_C = ncnv7_pro(model, "cuda", args, train_loader, 0.35, epoch = epoch, prob_A = prob_A, prob_B = prob_B)
        # ncnv7_for_more_pairs
        
        # epoch = 40
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
        logger.info(str(pred_C.tolist().count(1) / 37010))

        logger.info("and")
        logger.info(str(res1.tolist().count(1)))
        logger.info(str(res1.tolist().count(1) / pred_C.tolist().count(1)))

        logger.info("XOR")
        logger.info(str(res2.tolist().count(1)))
        logger.info(str(res2.tolist().count(1) / 37010))
        
        # ------------------------------------------------------------------------------------------------
        # 设定 noisy_rate，假设你是通过 argparse 或其他方式传入的
        if args.noisy_rate == 0.0:
            label_hat = torch.ones_like(pred_C)  # 设置 label_hat 为与 pred_C 形状相同的全1张量
        else:
            label_hat = pred_C
        # ------------------------------------------------------------------------------------------------
        # model.train()
        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            index = batch['index']
            # print(index[0])
            batch['label_hat'] = label_hat[index.cpu()]
            if (PL_labels_selected != None) and (PL_labels_selected_index != None) and (PL_labels_selected.numel() != 0) and (PL_labels_selected_index.numel() != 0):
                matches = torch.isin(index.cpu(), PL_labels_selected_index.cpu())
                if torch.sum(matches) > 0:
                    batch["is_relabel"] = matches # bool
                    print(batch["is_relabel"])
                    print(matches.shape)

                    # matches2 = torch.isin(PL_labels_selected_index.cpu(), index.cpu())
                    # matched2_indices = torch.nonzero(matches2).squeeze()
                    # batch["PL"] = PL_labels_selected[matched2_indices]

                    # indices = torch.tensor([torch.where(PL_labels_selected_index.cpu() == idx)[0].item() for idx in index.cpu()])
                    indices = []
                    for idx in index.cpu():
                        match = torch.where(PL_labels_selected_index.cpu() == idx)[0]
                        if match.numel() > 0:
                            indices.append(match.item())
                    indices = torch.Tensor(indices)
                    batch["PL"] = PL_labels_selected[indices.long()]
                    print(indices)
                    print(index)
                else:
                    batch["is_relabel"] = None
                    batch["PL"] = None
            else:
                batch["is_relabel"] = None
                batch["PL"] = None
            
            ret = model(batch, epoch)
            
            
            
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['bge_loss'].update(ret.get('bge_loss', 0), batch_size)
            meters['tse_loss'].update(ret.get('tse_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['crossid_loss'].update(ret.get('crossid_loss', 0), batch_size)
            meters['txt_id_loss'].update(ret.get('txt_id_loss', 0), batch_size)
            meters['bge_img_acc'].update(ret.get('bge_img_acc', 0), batch_size)   
            meters['tse_img_acc'].update(ret.get('tse_img_acc', 0), batch_size)
            meters['bge_txt_acc'].update(ret.get('bge_txt_acc', 0), batch_size)
            meters['tse_txt_acc'].update(ret.get('tse_txt_acc', 0), batch_size)
            if epoch >= 13:
                # print("!")
                # print(ret["relabel_num"])
                meters['relabel_num'].update(ret.get('relabel_num', 0), batch_size)
                meters['relabel_acc'].update(ret.get('relabel_acc', 0), batch_size)
            # if epoch >= 8 and epoch <= 11:
            #     for param_group in optimizer.param_groups:
            #         param_group['weight_decay'] = 0.02
            # else:
            #     for param_group in optimizer.param_groups:
            #         param_group['weight_decay'] = 4e-5
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg >= 0:
                        info_str += f", {k}: {v.avg:.5f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
 
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
 
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

    arguments["epoch"] = epoch
    checkpointer.save("last", **arguments)
                    
def do_inference(model, test_img_loader, test_txt_loader, refer_txt_loader, args):

    logger = logging.getLogger("RDE.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, refer_txt_loader, args)
    top1 = evaluator.eval(model.eval())
