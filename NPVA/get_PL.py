import logging
import os
import os.path as op
import torch
import numpy as np
import random
import time
import math
from collections import defaultdict, Counter

def get_PL(model, device, args, train_loader, label_hat, logger):
    model.eval()
    """
    This function generates partial labels for a given configuration, probabilities, and output logits.

    Args:
        config: The configuration parameters.
        probs (Tensor): The probabilities.
        output_logits (Tensor): The output logits.
        info (optional): Additional information.

    Returns:
        Tensor, Tensor: The partial labels and a mask indicating non-zero rows.
    """
    
    # 获得噪声样本（需要relabel的文本）分类概率    
    probs, noisy_feats_index, cap_r_pid = get_probs(model, device, args, train_loader, label_hat)
    # torch.set_printoptions(profile="full")
    # for i in probs:
    #     logger.info(i)
    # p = probs.cpu().numpy()
    # np.savetxt('probs1.txt', p, fmt='%.4e')
    print("!!"*100)
    print(probs.shape)
    print(noisy_feats_index.shape)
    print("get probs")
    print(probs.shape)
    bge_txt_pred = torch.argmax(probs, dim=1)
    bge_txt_precision = (bge_txt_pred == cap_r_pid).float().mean()
    print(bge_txt_precision)
    
    # torch.set_printoptions(profile="full")
    max_probs, _ = probs.max(dim=1)
    print(max_probs)
    print(max_probs.shape)
    output_logits = None
    
    conf_thr_final_1, PL_labels_1 = get_partialY_byAutoThr(
        output_logits, probs, 
        'quantile', 
        'intra_inst',
        target_quantile=99
    )
    
    # conf_thr_final_2, PL_labels_2 = get_partialY_byAutoThr(
    #     output_logits, probs, 
    #     'quantile', 
    #     'inter_inst',
    #     target_quantile= 90,
    # )
    
    
    PL_labels_2 = get_partialY_byThr(
        logits=output_logits, 
        probs_list=probs, 
        threshold= 0.99, 
        candidate_method='inter_inst',
        thr1=0.999
    )
    
#     PL_labels_2_1 = get_partialY_byThr(
#         logits=output_logits, 
#         probs_list=probs, 
#         threshold=0.99, 
#         candidate_method='inter_inst'
#     )

#     PL_labels_2_2 = get_partialY_byThr(
#         logits=output_logits, 
#         probs_list=probs, 
#         threshold=0.1, 
#         candidate_method='inter_inst'
#     )
#     print("?"*100)
#     print((PL_labels_2_1 > 1e-7).sum(dim=1))
#     print((PL_labels_2_2 > 1e-7).sum(dim=1))
    # Combine the two sets of partial labels through intersection
    # torch.set_printoptions(profile="full")
    # PL_labels_1 = torch.zeros_like(PL_labels_1)
    # candidate_mask = ((PL_labels_1 > 0) & (PL_labels_2 > 0)).float()
    # print(PL_labels_1)
    candidate_mask = (PL_labels_2 > 1e-7)
    print("$"*100)
    PL2 = (PL_labels_2 > 0)
    print(PL2.sum(dim=1))
    # If the sum of candidates in a row is 0, tag this row as a zero row
    zero_row_mask = (candidate_mask.sum(dim=1) == 0)
    # print(zero_row_mask)
    # print((PL_labels_1 > 1e-7)[0])
    print(candidate_mask.sum(dim=1))
    # Normalize the PL_label according to the PLL requirements
    PL_label = candidate_mask * probs
    base_value = PL_label.sum(dim=1).unsqueeze(1).repeat(1, PL_label.shape[1])
    PL_labels = PL_label / base_value
    
    print("100 + 0.97")
    
    
    # conf_thr_final = (conf_thr_final_1, config.REGULAR_THRESHOLD) 
    # log.info(f"So, selected CONF_THRESHOLD for method {config.CANDIDATE_METHOD}, is <{conf_thr_final}>")
    # print(PL_labels)
    print((~zero_row_mask).shape)
    true_count = torch.sum((~zero_row_mask))
    print("@"*100)
    print(PL_labels.shape)
    print(true_count) 
    # print(~zero_row_mask)
    
    mask_idxs = ~zero_row_mask
    info_1 = check_partialY_acc(
        PL_labels[mask_idxs], 
        cap_r_pid, 
        None, 
        0.0,)
    
    print(info_1)
    
    Selector = InstanceSelector(None, None)
    batch_size = 64
    iter_num = math.ceil(len(train_loader.dataset) / batch_size)
    print(iter_num)
    noisy_num = label_hat.numel() - torch.sum(label_hat)
    # selected_idxs, info_2 = Selector.select_topk_for_eachcls(
    #         PL_labels=(PL_labels > 1e-7).float()[mask_idxs],
    #         output_all=probs[mask_idxs],
    #         indexs_all=torch.arange(noisy_num).cuda()[mask_idxs],
    #         K_max=4,
    #         candidate_method="CPL",
    #         N_iter=iter_num,
    #     )
    # print(selected_idxs.shape)
    # print(selected_idxs)
    PL_labels_selected = (PL_labels[selected_idxs.long(), :] > 1e-7).float()
#     print(PL_labels_selected.shape)
    
    # PL_labels_nonzero=(PL_labels > 1e-7).float()[mask_idxs]
    # PL_labels_nonzero_index = noisy_feats_index[mask_idxs]
    
    PL_labels_selected_index = noisy_feats_index[selected_idxs.long()]
    
    
    return PL_labels_selected, PL_labels_selected_index

def check_partialY_acc(PL_labels, filepaths, target_partialR, init_partialR):
    # check the accuracy of pseudolabels
    # gt_labels = self._get_gt_label(impath=filepaths, dtype=torch.long)
    gt_labels = filepaths
    # initialize a list to store the results
    results = []
    distribution = []
    # iterate over each row of PL_labels and the corresponding gt_labels
    for i in range(PL_labels.shape[0]):
        # get the indices where the values are 1.0 in the current row
        indices = torch.nonzero(PL_labels[i], as_tuple=True)

        # test if the corresponding gt_label is in these indices
        is_in = gt_labels[i] in indices[0]
        distribution.extend(indices[0].tolist())

        # append the result to the list
        results.append(is_in)

    results = torch.tensor(results)
    coverage_acc = results.sum() / results.shape[0]
    ct = Counter(distribution)
    ct = sorted(ct.items(), key=lambda x: x[0])
    partial_avgnum = (PL_labels > 1e-7).sum(dim=1).float()

    print(f"\t label_estimate_acc: {coverage_acc}")
    # log.info(f"coverage distribution: {ct}")
    partialR = partial_avgnum.mean().item()/PL_labels.shape[1]

    return {"label_estimate_acc": coverage_acc.item(), 
            "partial_ratio": partialR, 
            }
    
def get_probs(model, device, args, train_loader, label_hat):
    batch_size = 64
    num_classes = model.num_classes
    bge_txt_logits = torch.rand(len(train_loader.dataset), num_classes).t().cuda() #(11003, 68126)
    tse_txt_logits = torch.rand(len(train_loader.dataset), num_classes).t().cuda() #(11003, 68126)

    bge_classifier = model.bge_classifier
    tse_classifier = model.tse_classifier
    
    feats_index_total = []
    label_hat_all = []
    cap_r_pid_list = []
    for batch_idx, batch in enumerate(train_loader):
        images = batch['images'].to(device)
        caption_ids = batch['caption_ids'].to(device)
        index = batch['index']
        cap_r_pid = batch['cap_r_pid']
        label_hat_all.append(label_hat[index])
        feats_index_total.append(index)
        cap_r_pid_list.append(cap_r_pid)
        current_batch_size = len(caption_ids)
        bge_image_batch_feats, atten_i, bge_text_batch_feats, atten_t = model.base_model(images, caption_ids)

        tse_text_batch_feats = model.texual_emb_layer(bge_text_batch_feats, caption_ids, atten_t)

        bge_text_batch_feats = bge_text_batch_feats[torch.arange(bge_text_batch_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        # bge_text_batch_feats = bge_text_batch_feats / bge_text_batch_feats.norm(dim=-1, keepdim=True)
        # tse_text_batch_feats = tse_text_batch_feats / tse_text_batch_feats.norm(dim=-1, keepdim=True)
        
        
        bge_txt_batch_logits = bge_classifier(bge_text_batch_feats.half())
        tse_txt_batch_logits = tse_classifier(tse_text_batch_feats.half())
               
        bge_txt_logits[:, batch_idx * batch_size:batch_idx * batch_size + current_batch_size] = bge_txt_batch_logits.data.t()
        tse_txt_logits[:, batch_idx * batch_size:batch_idx * batch_size + current_batch_size] = tse_txt_batch_logits.data.t()
        
    label_hat_all = torch.cat(label_hat_all, dim=0).cuda()
    cap_r_pid_tensor = torch.cat(cap_r_pid_list, dim=0).cuda()
    feats_index_total = torch.cat(feats_index_total, dim=0).cuda()
    print(bge_txt_logits.shape)
    bge_txt_probs = torch.softmax(bge_txt_logits, dim=0)
    print(bge_txt_probs)
    tse_txt_probs = torch.softmax(tse_txt_logits, dim=0)
    print(tse_txt_probs)
    txt_probs = (bge_txt_probs + tse_txt_probs) / 2
    
    # txt_probs = bge_txt_probs
    txt_probs = txt_probs.t() #(num_samples, num_classes)
    
    noisy_txt_probs = txt_probs[label_hat_all == 0]
    cap_r_pid_tensor = cap_r_pid_tensor[label_hat_all == 0]
    noisy_feats_index = feats_index_total[label_hat_all == 0]
    # noisy_txt_probs = txt_probs
    # noisy_txt_probs = torch.cat(noisy_txt_probs, dim=0).cuda()
    return noisy_txt_probs, noisy_feats_index, cap_r_pid_tensor

# 采用自适应阈值获得伪标签
def get_partialY_byAutoThr(output_logits, probs, conf_thr_method, select_method, 
                           target_quantile=None):
    """
    Adjusts the confidence threshold according to the given method and gets the partial labels.

    Args:
        output_logits (Tensor): The output logits.
        probs (Tensor): The probabilities.
        conf_thr_method (str): The method to adjust the confidence threshold.
        select_method (str): The method to select the partial labels.
        target_quantile (float, optional): The target quantile to adjust the confidence threshold. Default is None.

    Returns:
        float, Tensor: The adjusted confidence threshold and the partial labels.
    """
    # 获得经过自适应运算后阈值
    if conf_thr_method == 'quantile':
        if target_quantile == 0:
            conf_thr = probs.max(dim=1).values.mean().cpu().item()
        else:
            if select_method == "intra_inst":
                print("1")
            # set the confidence threshold to the quantile of the maximum probabilities
                conf_thr = torch.quantile(probs.max(dim=1).values, target_quantile/100).cpu().item()
            print(conf_thr)
        # 用自适应阈值获得伪标签 
        PL_labels_ = get_partialY_byThr(
            threshold=conf_thr,
            candidate_method=select_method,
            probs_list=probs,
            logits=output_logits, thr1=0)

    else:
        raise ValueError("conf_thr_method should be 'quantile'")

    return conf_thr, PL_labels_



# 采用硬标签获得伪标签
def get_partialY_byThr(threshold, candidate_method, probs_list, logits, thr1):
    """
    Generates partial labels by a given threshold and candidate method.

    Args:
        threshold (float): The threshold to detect candidates.
        candidate_method (str): The method to detect candidates. Should be either 'intra_inst' or 'inter_inst'.
        probs_list (Tensor): The probabilities.
        logits (Tensor): The output logits.

    Returns:
        Tensor: The partial labels.
    """
    if candidate_method == 'intra_inst':
        # method 1: use cumulated prob to detect candidate:
        data = probs_list
        candidate_mask = detect_candidate_bycum(data=data, thr=threshold)     #prob is shape of (batch_size, num_classes)
    elif candidate_method == 'inter_inst':
        # method 2: use cdf as class-wise inter_inst percentage to detect candidate:
        data = probs_list
        candidate_mask = detect_candidate_bycls_thr(data=probs_list, thr=threshold, thr1=thr1)
    else:
        raise ValueError("candidate_method should be 'intra_inst' or 'inter_inst'")
    # pred_id = labels_range[candidate_mask]
    PL_label = candidate_mask.float() * data

    # normalize the PL_label according to the PLL requirements
    base_value = PL_label.sum(dim=1).unsqueeze(1).repeat(1, PL_label.shape[1])
    PL_label_ = PL_label / base_value
    return PL_label_

# intra_inst
def detect_candidate_bycum(data, thr):
    """
    Generates a candidate mask by cumulatively summing the data to a threshold and getting the summed items of each row.

    Args:
        data (Tensor or any): The data to be summed. If not a Tensor, it will be converted to a Tensor.
        thr (float): The threshold for the cumulative sum.

    Returns:
        Tensor: The candidate mask.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)

    sorted_data, indices = torch.sort(data, dim=-1, descending=True)
    cum_data = torch.cumsum(sorted_data, dim=-1)
    assert isinstance(thr, float), "thr should be a float number"

    # Create a mask for the first element of each row
    first_elem_mask = torch.zeros_like(cum_data, dtype=torch.bool)
    first_elem_mask[:, 0] = True

    # Create last_elem_mask
    exceeds_thr_mask = cum_data > thr
    # print(exceeds_thr_mask.shape)
    # torch.set_printoptions(profile="full")
    # print(cum_data[0])
    # print((cum_data[0]).shape)
    # print((exceeds_thr_mask[0]).shape)
    shifted_exceeds_thr_mask = torch.cat([torch.zeros_like(exceeds_thr_mask[:, :1]), 
                                          exceeds_thr_mask[:, :-1]], 
                                        dim=1)
    last_elem_mask = (~shifted_exceeds_thr_mask) & exceeds_thr_mask

    # Combine masks
    candidate_mask = cum_data <= thr
    candidate_mask |= last_elem_mask
    candidate_mask |= (exceeds_thr_mask & first_elem_mask)

    # Create a new tensor and use indices and row indices to get elements from candidate_mask
    rows = torch.arange(candidate_mask.size(0)).unsqueeze(-1)
    candidate_mask = candidate_mask[rows, indices.sort(dim=-1).indices]   # sort idxs in ascending order

    return candidate_mask

def cdf_at_value(data, value, mode='count', batch_size=512):
    """
    Calculate the cumulative distribution function (CDF) at a given value for a histogram.

    Parameters:
    data (torch.Tensor or np.ndarray): The input data defining the custom PDF.
    value (float or np.ndarray): The value(s) at which to calculate the CDF.

    Returns:
    float or np.ndarray: The CDF at the given value(s).
    """
    # Expand dimensions for broadcasting
    if mode == 'count':
        scores = torch.zeros_like(value)
        for i in range(0, len(value), batch_size):
            data_ = data.unsqueeze(0)  # shape becomes (1, n)
            value_ = value[i:i+batch_size].unsqueeze(1)  # shape becomes (m, 1)
            # Count the number of elements in 'data' that are smaller than each element in 'value'
            # The result is a tensor of shape (m, n), where each row corresponds to an element in 'value'
            counts = (data_ <= value_).sum(dim=1)
            scores[i:i+batch_size] = counts / data.shape[0]

    return scores

# inter_inst
def detect_candidate_bycls_thr(data, thr, thr1):
    """
    Generates a candidate mask by class-wise inter-instance percentage.

    Args:
        data (Tensor or any): The data to be processed. If not a Tensor, it will be converted to a Tensor.
        thr (float): The threshold for the CDF scores.

    Returns:
        Tensor: The candidate mask.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    candidate_mask = torch.zeros_like(data, dtype=torch.bool)
    range_list = torch.arange(data.shape[0])

    for cls in range(data.shape[1]):
        cls_output = data[range_list, cls]

        thr_val = torch.quantile(cls_output, thr)
        thr_val1 = torch.quantile(cls_output, thr1)
        cls_mask = (cls_output >= thr_val)
        # cls_mask = ((cls_output >= thr_val) & (cls_output <= thr_val1))

        # assert (cls_mask_ == cls_mask).all(), "The two masks should be the same"
        candidate_mask[range_list, cls] = cls_mask

    return candidate_mask

class InstanceSelector(object):
    # opt params (assigned during code running)

    def __init__(self, 
                 label_to_idx=None, 
                 train_labels_true=None,
                 eps=1e-7, cfg=None,
                 convert_pred_method=None,):
        super(InstanceSelector, self).__init__()
        self.cfg = cfg
        self.label_to_idx = label_to_idx
        self.eps=eps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # class_ids = list(range(0, len(label_to_idx)))
        class_ids = list(range(0, 11003))
        self.Pools = PoolsAggregation(class_ids=class_ids, K=4)
        if convert_pred_method is not None:
            self.convert_pred_idxs_to_real = convert_pred_method
            for cls_idx, pool in self.Pools.cls_pools_dict.items():
                pool.convert_pred_idxs_to_real = convert_pred_method

        if train_labels_true is not None:
            for cls_idx, pool in self.Pools.cls_pools_dict.items():
                pool.labels_true = train_labels_true


    def cal_pred_conf_uncs(self, logits, PL_labels, method):
        """
        Calculates the prediction confidence and uncertainties for each data point.
        
        And the uncs will be used in the following steps to select the top-k samples (in ascending order).
        This munipulation is equivalent to just use conf to select top-k samples (in descending order).
        """
        # method 1. cal conf by prob:
        # conf = F.softmax(logits, dim=1)
        conf = logits
        uncs = 1 - conf
    
        # assign attrs:
        conf = conf * PL_labels
        uncs = uncs * PL_labels

        return conf, uncs


    def _prepare_items_attrs(self, PL_labels, outputs, method, max_num="all"):   
        """
        Prepares labels and uncertainties for all items based on model outputs and a specified method.
        
        Args:
            PL_labels (Tensor): The pseudolabels for the data points.
            outputs (Tensor): The output from the model for all data points.
            method (str): The method used to calculate uncertainties.
            max_num (str or int, optional): The maximum number of labels to process.
        
        Returns:
            tuple: A tuple containing tensors of labels and uncertainties for the data points.
        """
        conf_all, uncs_all = self.cal_pred_conf_uncs(outputs, PL_labels, method=method)

        if max_num == "all":
            # Find the max count of non-zero labels across all items
            max_num = (PL_labels > self.eps).sum(dim=1).max().item()

        labels_ = []; uncs_ = []
        for i in range(max_num):
            max_val, max_idx = torch.max(conf_all, dim=1)   # get max val in each row
            mask = (max_val < self.eps)                     
            uncs = uncs_all[torch.arange(0, uncs_all.shape[0]), max_idx] #uncs is shape of (batch_size, max_num)
            uncs[mask] = torch.inf
            conf_all[torch.arange(0, conf_all.shape[0]), max_idx] = -torch.inf

            labels_.append(max_idx.unsqueeze(1))
            uncs_.append(uncs.unsqueeze(1))
        labels_ = torch.cat(labels_, dim=1)
        uncs_ = torch.cat(uncs_, dim=1)

        return labels_, uncs_

    def convert_pred_idxs_to_real(self, pred_idxs):
        """convert pred_idxs to real idxs"""
        # default do nothing and would be redefined (in __init__) only in TRZSL setting
        return pred_idxs

    def select_topk_for_eachcls(self, PL_labels, indexs_all, output_all, 
                                K_max, candidate_method, 
                                N_iter=1, increase_percentage=0):
        """
        Selects the top K instances for each class based on the provided criteria.
        
        Args:
            PL_labels (Tensor): The pseudolabels for the data points.
            indexs_all (Tensor): Indices of all data points.
            output_all (Tensor): The output from the model for all data points.
            K_max (int): The maximum number of instances to select for each class.
            candidate_method (str): The method used to select candidates.
            N_iter (int, optional): The current iteration number. Defaults to 1.
            increase_percentage (float, optional): The percentage to increase the pool size by.
        
        Returns:
            tuple: A tuple containing the indices of the selected instances and info.
        """
        #1. prepare necessary attrs for all items:
        labels, uncs = self._prepare_items_attrs(PL_labels, output_all, candidate_method)
        max_iters = PL_labels.sum(dim=1).long().cpu()
        #check Top-1 pred_label ACC:
        labels_top1 = self.convert_pred_idxs_to_real(labels[:, 0])
        # acc = labels_top1.cpu() == self.Pools.cls_pools_dict[0].labels_true[indexs_all.cpu()]
        # ACC = acc.sum()/labels.shape[0]
        # log.info(f"Top-1 pred_label ACC: {ACC}")
        
        #2. revise the pool caps:
        if increase_percentage == 0:
            past_caps = self.Pools.get_pool_caps()
            increment_nums = K_max - max(past_caps)
            assert increment_nums >= 0
            pool_caps = [min(past_caps[i] + increment_nums, K_max) 
                            for i in range(len(self.Pools))]
        else:
            raise ValueError("increase_percentage should be == 0")
        
        self.Pools.reset_all()
        self.Pools.scale_all_pools(scale_nums=pool_caps)  

        #3. fill pools with samples according to the uncs for each class:
        not_inpool_feat_idxs, notinpool_uncs = self._fill_pools(
            labels, uncs, indexs_all, 
            max_iter_num=max_iters, 
            record_popped=True,)
        
        selected_idxs = self.Pools.get_all_feat_idxs()
        # print info:
        self.Pools.print()
        self.Pools.cal_pool_ACC()

        # return selected_idxs, {"top1_acc": ACC.item()}
        return selected_idxs, 0


    def _fill_pools(self, labels, uncs, feat_idxs, max_iter_num, 
                    record_popped=True, pool_idxs=None):
        """
        Fills pools with top samples for each class based on uncertainties.
        
        Args:
            labels (Tensor): The labels for the data points.
            uncs (Tensor): The uncertainties associated with each data point.
            feat_idxs (Tensor): The feature indices of the data points.
            max_iter_num (Tensor): The maximum number of iterations for each data point.
            record_popped (bool, optional): Whether to record popped elements. Defaults to True.
            pool_idxs (Tensor, optional): The pool indices. If None, they are generated.
        """
        # Initialize pool indices if not provided
        if pool_idxs is None:
            # pool_idxs = torch.arange(0, len(self.label_to_idx)).to(self.device)
            pool_idxs = torch.arange(0, 11003).to(self.device)
        if labels.shape[1] == 0:
            return 
        assert max_iter_num.all() >= 1

        # Initialize tensors for tracking pool status
        not_in_pool_init = torch.ones(labels.shape[0], dtype=torch.bool)
        del_elems_init = torch.zeros(labels.shape[0], dtype=torch.bool)
        all_idxs = torch.arange(0, labels.shape[0])
        quary_num = torch.zeros(labels.shape[0], dtype=torch.long)      
        
        def recursion(top_uncs, top_labels, not_in_pool, del_elems):
            """
            Recursively fills pools with samples, updating their status.
            
            Args:
                top_uncs (Tensor): The top uncertainties for each data point.
                top_labels (Tensor): The top labels for each data point.
                not_in_pool (Tensor): A boolean tensor indicating if a data point is not in the pool.
                del_elems (Tensor): A boolean tensor indicating if a data point should be deleted.
            """
            in_itering = (not_in_pool & ~del_elems)
            if (in_itering).sum() == 0 or (not_in_pool==False).all():
                return 
            else:
                # Select indices and corresponding uncertainties and labels for this iter
                this_loop_idxs = all_idxs[in_itering]  
                this_loop_uncs = top_uncs[this_loop_idxs, quary_num[this_loop_idxs]]
                this_loop_labels = top_labels[this_loop_idxs, quary_num[this_loop_idxs]]
                not_in_pool[:] = True
                quary_num[this_loop_idxs] += 1
                assert labels.shape[0] == (this_loop_idxs.shape[0] + 
                                           self.Pools.cal_pool_sum_num() + 
                                           del_elems.sum()), \
                        "All_samples = not_in + in_pool + not_in_but_enough_iter"
                # Fill the pool with the selected samples
                self.Pools.batch_fill_assigned_pool(
                    feat_idxs[this_loop_idxs], 
                    this_loop_uncs, 
                    this_loop_labels
                )
                # Update pool and deletion status
                inpool_idxs = self.Pools.get_all_feat_idxs()  
                # self.Pools.cal_pool_ACC()
                elem_idxs = find_elem_idx_BinA(A=feat_idxs, B=inpool_idxs)  

                not_in_pool[elem_idxs] = False
                del_elems = (quary_num[:] == max_iter_num[:]) & (not_in_pool)
                recursion(top_uncs, top_labels, not_in_pool, del_elems)
        
        # call recursion:
        recursion(uncs, labels, not_in_pool_init, del_elems_init)
        return feat_idxs[not_in_pool_init], uncs[not_in_pool_init, 0]    #get top 1 uncertainty for each sample
        

class PoolsAggregation:
    """
    Administer the pool of each class.
    """

    def __init__(self, class_ids, K, max_capacity_per_class=None):
        """
        Initialize the PoolsAggregation.
        Args:
            cfg (Config): The configuration object.
            class_ids (list): a list of class ids.
            K (int): Number of top samples to select per class.
            max_capacity_per_class (dict): Maximum capacity per class. 
        """
        self.min_cap = K
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize cls_pools_dict
        self.cls_pools_dict = {}
        if max_capacity_per_class is None:
            max_capacity_per_class = {cls: K for cls in class_ids}

        # Convert max_capacity_per_class to a tensor for efficient computation
        max_capacity_per_class = [max_capacity_per_class[cls] for cls in class_ids]

        # Loop through each unique class id
        for i, cls in enumerate(class_ids):                                   
            self.cls_pools_dict[cls] = ClassPool(max_capacity=max_capacity_per_class[i], 
                                                 cls_id=cls)

    def __len__(self):
        return len(self.cls_pools_dict)

    def scale_all_pools(self, scale_nums):
        """Manipulate the scale of each pool in its government"""
        for cls_idx, pool in self.cls_pools_dict.items():
            next_capacity = scale_nums[cls_idx]
            next_capacity = next_capacity
            pool.scale_pool(next_capacity=next_capacity)


    def reset_all(self):
        """Reset all pools in its government"""
        for pool in self.cls_pools_dict.values():
            pool.reset()


    def batch_fill_assigned_pool(self, feat_idxs: torch.LongTensor, feat_uncs: torch.Tensor, pool_ids):
        """
        Fill the assigned pool with new values in batch.
        Args:
            feat_idxs (torch.Tensor): A tensor of feature indices, better to be ascending order.
            feat_uncs (torch.Tensor): A tensor of feature uncertainties.
            pool_ids: the assigned_pool to fill all the items.
        """
        # in_pool = torch.zeros_like(feat_idxs, dtype=torch.bool)
        for pool_id in pool_ids.unique():
            mask = pool_ids == pool_id
            cur_pool = self.cls_pools_dict[pool_id.item()]
            cur_pool.batch_update(feat_idxs[mask], feat_uncs[mask]) # in_pool[mask] = 


    def get_all_feat_idxs(self):
        """
        Get all feature indices for all pool in pool_dict.
        Returns:
            torch.Tensor: A tensor of feature indices.
        """
        feat_idxs = torch.LongTensor([])
        for pool in self.cls_pools_dict.values():
            feat_idxs = torch.cat((feat_idxs, pool.pool_idx), dim=0)
        return feat_idxs
    

    def cal_pool_sum_num(self):
        sum_num = 0
        for i, pool in enumerate(self.cls_pools_dict.values()):
            sum_num += pool.pool_capacity
            # print(f'pool_id: {i}, pool_capacity: {pool.pool_capacity}')
        return sum_num
    
    def get_pool_caps(self):
        cap_list = []
        for i, pool in enumerate(self.cls_pools_dict.values()):
            cap_list.append(pool.pool_capacity)
            # print(f'pool_id: {i}, pool_capacity: {pool.pool_capacity}')
        return cap_list

    def cal_pool_ACC(self):
        correct_num = 0
        all_num = 0
        for pool in self.cls_pools_dict.values():
            pred_labels = pool.convert_pred_idxs_to_real(torch.LongTensor([pool.cls_id]))
            # correct = (pool.labels_true[pool.pool_idx] == pred_labels).sum()
            correct = 0
            correct_num += correct
            all_num += pool.pool_capacity
        # log.info(f'====> overall pools ACC: {correct_num}/{all_num} = {correct_num/all_num}')
        # print(f'====> overall pools ACC: {correct_num}/{all_num} = {correct_num/all_num}')


    def print(self):
        for pool_id, cur_pool in self.cls_pools_dict.items():
            # log.info(cur_pool)
            print(cur_pool)

class ClassPool:
    """
    Store the average and current values for uncertainty of each class samples and the max capacity of the pool.
    """

    def __init__(self, max_capacity: int, cls_id):
        """
        Initialize the ClassPool.
        Args:
            max_capacity (int): The maximum capacity of the pool.
            items_idx (torch.LongTensor): A tensor of item indices.
            items_unc (torch.Tensor): A tensor of item uncertainties.
        """
        self.pool_max_capacity = max_capacity
        self.is_freeze = False
        self.cls_id = cls_id
        self.device = 'cuda'
        self.unc_dtype = torch.float32
        self.baseline_capacity = max_capacity
        self.reset()
        
    def _update_pool_attr(self):
        """
        Update the pool attributes.
        """
        # self.unc_avg = torch.mean(self.pool_unc)
        self.unc_max, self.unc_max_idx = torch.max(self.pool_unc, dim=0)
        assert self.pool_unc.shape == self.pool_unc.shape
        assert self.pool_unc.shape[0] <= self.pool_max_capacity
    
    def reset(self):
        """
        Reset the pool.
        """
        self.pool_idx = torch.LongTensor([])
        self.pool_unc = torch.Tensor([]).type(self.unc_dtype).to(self.device)
        self.popped_idx = torch.LongTensor([])
        self.popped_unc = torch.Tensor([]).type(self.unc_dtype).to(self.device)
        #attribute:
        self.pool_capacity = 0
        self.unc_max = 1e-10

        assert self.is_freeze == False
        self.pool_unc_past = None
        self.pool_idx_past = None
        self.replace_num = 0
        self.not_in_num = 0

    def scale_pool(self, next_capacity: int):
        """
        Scale the pool to the next iter capacity.
        """
        self.pool_max_capacity = next_capacity
        return


    def batch_update(self, feat_idxs: torch.LongTensor, feat_uncs: torch.Tensor, record_popped=False):
        """
        Update the pool with new values in batch.
        Args:
            feat_idxs (torch.Tensor): A tensor of feature indices, better to be ascending order.
            feat_uncs (torch.Tensor): A tensor of feature uncertainties.
        """
        in_pool = torch.zeros_like(feat_idxs, dtype=torch.bool)
        for i, (feat_idx, feat_unc) in enumerate(zip(feat_idxs, feat_uncs)):
            in_pool[i] = self.update(feat_idx, feat_unc, record_popped)
        return in_pool


    def update(self, feat_idx: torch.LongTensor, feat_unc: torch.Tensor, record_popped=False):
        """
        Update the pool with new values.
        Args:
            feat_idxs (torch.Tensor): A tensor of feature indices, better to be ascending order.
            feat_unc (torch.Tensor): A tensor of feature uncertainties.
        """
        if self.pool_capacity < self.pool_max_capacity:
            if feat_unc < 1e4:
                self.pool_idx = torch.cat((self.pool_idx.cpu(), feat_idx.unsqueeze(0).cpu()))  
                self.pool_unc = torch.cat((self.pool_unc.cpu(), feat_unc.unsqueeze(0).cpu()))  
                # self.saved_logits = torch.cat((self.saved_logits, feat_logit.unsqueeze(0)))  
                self.pool_capacity += 1
                in_pool = True
            else:
                in_pool = False
        else:
            assert self.pool_max_capacity >= self.pool_capacity, \
                f"pool_max_capacity: {self.pool_max_capacity}, pool_capacity: {self.pool_capacity}"
            if self.unc_max <= feat_unc:
                if record_popped:
                    self.popped_idx = torch.cat((self.popped_idx, feat_idx.unsqueeze(0)))  
                    self.popped_unc = torch.cat((self.popped_unc, feat_unc.unsqueeze(0)))  
                in_pool = False
            else:
                if record_popped:
                    self.popped_idx = torch.cat((self.popped_idx, 
                                                 self.pool_idx[self.unc_max_idx].unsqueeze(0)))
                    self.popped_unc = torch.cat((self.popped_unc, 
                                                 self.pool_unc[self.unc_max_idx].unsqueeze(0)))
                    # self.popped_img_feats.append(info_dict['image_feat'])      
                    # self.poped_logits.append(info_dict['logit'])
                
                self.pool_idx[self.unc_max_idx] = feat_idx
                self.pool_unc[self.unc_max_idx] = feat_unc
                # self.saved_logits[self.unc_max_idx] = feat_logit
                in_pool = True
                
        if in_pool:
            self._update_pool_attr()

        return in_pool


    def __str__(self):
        str_ = f'pool_id: {self.cls_id}, '
        if hasattr(self, 'unc_avg'):
            str_ += f"unc_avg: {self.unc_avg:.6f}, "
        if self.unc_max != None:
            str_ += f"unc_max: {self.unc_max:.6f}, "
        else:
            str_ += f"unc_max: None, "
        if hasattr(self, 'labels_true'):
            pred_labels = self.convert_pred_idxs_to_real(torch.LongTensor([self.cls_id]))
            corrcet_num = (self.labels_true[self.pool_idx] == pred_labels).sum()
            str_ += f"pool ACC: {corrcet_num}/{self.pool_capacity}, "
        return str_ + f"pool_capacity: {self.pool_capacity}/{self.pool_max_capacity}"
    
    
    def convert_pred_idxs_to_real(self, pred_idxs):
        """convert pred_idxs to real idxs"""
        #default do nothing and would be redefined in TRZSL setting
        return pred_idxs
    

def find_elem_idx_BinA(A, B):
    """
    This function finds the indices of the elements of tensor b in tensor a.
    
    Parameters:
    a (torch.Tensor): The tensor in which to find the indices.
    b (torch.Tensor): The tensor whose elements' indices are to be found.
    
    Returns:
    torch.Tensor: A tensor containing the indices of the elements of b in a.
    """
    # Create a dictionary with elements of a as keys and their indices as values
    a_dict = {item.item(): i for i, item in enumerate(A)}
    
    # Map the elements of b to their corresponding indices in a using the dictionary
    indices = torch.tensor([a_dict[item.item()] for item in B], dtype=torch.long)
    
    return indices
