from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
 
from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
from nnn import NNNRetriever, NNNRanker

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    print("*"*100)

    pred_labels = g_pids[indices.cpu()]  # q * k
    print(pred_labels.shape)
    print(q_pids.view(-1, 1).shape)
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def custom_rank(indices, q_pids, g_pids, max_rank = 10):
         # acclerate sort with topk

    pred_labels = g_pids[indices.cpu()]  # q * k
    print(pred_labels.shape)
    print(q_pids.view(-1, 1).shape)
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]


    return all_cmc, indices


def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    if retur_indices:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]], indices
    else:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]]


class Evaluator():
    def __init__(self, img_loader, txt_loader, train_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("RDE.eval")
        self.train_loader = train_loader

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats, reference_embeddings = [], [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption).cpu()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img).cpu()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        
        # reference_embeddings
        for batch in self.train_loader:
            caption = batch["caption_ids"]
            caption = caption.to(device)
            with torch.no_grad():
                reference_embedding = model.encode_text(caption).cpu()
            reference_embeddings.append(reference_embedding)
        reference_embeddings = torch.cat(reference_embeddings, 0)
        
        
        
        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu(), reference_embeddings.cpu()
    
    def _compute_embedding_tse(self, model):
        model = model.eval() 
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats, reference_embeddings_tse = [], [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text_tse(caption).cpu()
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image_tse(img).cpu()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0) 
        
        # reference_embeddings
        for batch in self.train_loader:
            caption = batch["caption_ids"]
            caption = caption.to(device)
            with torch.no_grad():
                reference_embedding_tse = model.encode_text_tse(caption).cpu()
            reference_embeddings_tse.append(reference_embedding_tse)
        reference_embeddings_tse = torch.cat(reference_embeddings_tse, 0)
        
        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu(), reference_embeddings_tse.cpu()
    
    def eval(self, model, i2t_metric=True):
        qfeats, gfeats, qids, gids, reference_embeddings = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features
        sims_bse = qfeats @ gfeats.t()
  
        vq_feats, vg_feats, _, _, reference_embeddings_tse = self._compute_embedding_tse(model)
        vq_feats = F.normalize(vq_feats, p=2, dim=1) # text features
        vg_feats = F.normalize(vg_feats, p=2, dim=1) # image features
        sims_tse = vq_feats@vg_feats.t()
        
        
        gfeats = gfeats.numpy()
        reference_embeddings = reference_embeddings.numpy()
        qfeats = qfeats.numpy()
        nnn_retriever = NNNRetriever(gfeats.shape[1], use_gpu=True, gpu_id=0) 
        nnn_ranker = NNNRanker(nnn_retriever, gfeats, reference_embeddings, alternate_ks=128, alternate_weight=0.1, batch_size=1024, use_gpu=True, gpu_id=0)
        scores, indices = nnn_ranker.search(qfeats, top_k=3074)
        indices = torch.tensor(indices)
        all_cmc, indices = custom_rank(indices, q_pids=qids, g_pids=gids)
        print(all_cmc)
        print("*"*100)
        print(scores.shape)
        print(type(scores))
        
        vg_feats = vg_feats.numpy()
        reference_embeddings_tse = reference_embeddings_tse.numpy()
        vq_feats = vq_feats.numpy()
        nnn_retriever = NNNRetriever(vg_feats.shape[1], use_gpu=True, gpu_id=0) 
        nnn_ranker = NNNRanker(nnn_retriever, vg_feats, reference_embeddings_tse, alternate_ks=128, alternate_weight=0.1, batch_size=1024, use_gpu=True, gpu_id=0)
        scores, indices = nnn_ranker.search(vq_feats, top_k=3074)
        indices = torch.tensor(indices)
        all_cmc, indices = custom_rank(indices, q_pids=qids, g_pids=gids)
        print(all_cmc)
        print("*"*100)
        print(scores.shape)
        print(type(scores))
        
        
        
        sims_dict = {
            'BGE': sims_bse,
            'TSE': sims_tse,
            'BGE+TSE': (sims_bse+sims_tse)/2
        }

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP","rSum"])
        
        for key in sims_dict.keys():
            sims = sims_dict[key]
            rs = get_metrics(sims, qids, gids, f'{key}-t2i',False)
            table.add_row(rs)
            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=sims.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
                i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
                print(i2t_cmc)
                # table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        
        table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
        table.custom_format["RSum"] = lambda f, v: f"{v:.2f}"
        self.logger.info('\n' + str(table))
        
        return rs[1]
