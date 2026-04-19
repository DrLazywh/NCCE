import torch
import math
from torch.nn.functional import normalize

import psutil

import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from nnn import NNNRetriever, NNNRanker, BaseRetriever, BaseRanker


def ncnv1(model, device, args, train_loader, bge_threshold, tse_threshold, num_neighbor=20):
    model.eval()
    batch_size = args.batch_size

    image_bge_features, text_bge_features, bge_feats_index_total = get_bge_features(model.eval(), train_loader, batch_size, device)

    neighbors_bge_feats = find_kn_neighbors(image_bge_features, num_neighbor, batch_size)
    pred_neighbors_labels = pred_neighbor_label(neighbors_bge_feats, text_bge_features)
    sver_total = calculate_sver(pred_neighbors_labels, text_bge_features)
    pred_binary_label = selecting(sver_total, bge_threshold)
    bge_pred = reorder_binary_label(pred_binary_label, train_loader, bge_feats_index_total, batch_size)
    
    print("bge_pred")
    print(bge_pred.tolist().count(1))
    print(bge_pred.tolist().count(1) / 68126)
    
    image_tse_features, text_tse_features, tse_feats_index_total = get_tse_features(model.eval(), train_loader, batch_size, device)

    neighbors_tse_feats = find_kn_neighbors(image_tse_features, num_neighbor, batch_size)
    pred_neighbors_labels = pred_neighbor_label(neighbors_tse_feats, text_tse_features)
    sver_total = calculate_sver(pred_neighbors_labels, text_tse_features)
    pred_binary_label = selecting(sver_total, tse_threshold)
    tse_pred = reorder_binary_label(pred_binary_label, train_loader, tse_feats_index_total, batch_size)
    
    print("tse_pred")
    print(tse_pred.tolist().count(1))
    print(tse_pred.tolist().count(1) / 68126)    

    # ==2就是两次于预测都是clean; ==1就是其中有一次预测是clean,一次是noisy; ==0就是两次预测都是noisy
    consensus_division = bge_pred + tse_pred  # 0,1,2

    # 对于等于1的，让它们随机 + 1或者 + 0，就是随机决定它们是clean还是noisy
    consensus_division[consensus_division == 1] += torch.randint(0, 2, size=(((consensus_division == 1) + 0).sum(),))
    pred = consensus_division.clone()
    pred[consensus_division>1] = 1
    pred[consensus_division<=1] = 0 
    
    return pred

def ncnv2(model, device, args, train_loader, threshold, num_neighbor=20):
    model.eval()
    batch_size = args.batch_size
    image_bge_features, text_bge_features = get_ordered_bge_features(model.eval(), train_loader, batch_size, device)
    neighbors_bge_feats = find_kn_neighbors(image_bge_features, num_neighbor, batch_size)
    pred_neighbors_labels = pred_neighbor_label(neighbors_bge_feats, text_bge_features)
    bge_sver_total = calculate_sver(pred_neighbors_labels, text_bge_features)

    
    image_tse_features, text_tse_features = get_ordered_tse_features(model.eval(), train_loader, batch_size, device)
    neighbors_tse_feats = find_kn_neighbors(image_tse_features, num_neighbor, batch_size)
    pred_neighbors_labels = pred_neighbor_label(neighbors_tse_feats, text_tse_features)
    tse_sver_total = calculate_sver(pred_neighbors_labels, text_tse_features)

    sver_total = torch.stack(bge_sver_total) + torch.stack(tse_sver_total)
    
    pred_binary_label = selecting(sver_total, threshold)
    
    return torch.tensor(pred_binary_label).to(torch.float32)

def ncnv3(model, device, args, train_loader, bge_threshold, num_neighbor=20):
    model.eval()
    batch_size = args.batch_size

    image_bge_features, text_bge_features, bge_feats_index_total = get_bge_features(model.eval(), train_loader, batch_size, device)

    neighbors_bge_feats = find_kn_neighbors(image_bge_features, num_neighbor, batch_size)
    pred_neighbors_labels = pred_neighbor_label(neighbors_bge_feats, text_bge_features)
    sver_total = calculate_sver(pred_neighbors_labels, text_bge_features)
    pred_binary_label = selecting(sver_total, bge_threshold)
    bge_pred = reorder_binary_label(pred_binary_label, train_loader, bge_feats_index_total, batch_size)
    pred = bge_pred
    
    return pred

def ncnv4(model, device, args, train_loader, threshold, num_neighbor=20):
    model.eval()
    batch_size = args.batch_size
    image_bge_features, text_bge_features, image_tse_features, text_tse_features, feats_index_total = get_bge_and_tse_featuers(model.eval(), train_loader, batch_size, device)
    neighbors_bge_feats, neighbors_tse_feats = find_joint_kn_neighbors(image_bge_features, image_tse_features, num_neighbor, batch_size)
    pred_neighbors_bge_labels, pred_neighbors_tse_labels = pred_joint_neighbor_label(neighbors_bge_feats, neighbors_tse_feats, text_bge_features, text_tse_features)
    bge_sver_total = calculate_sver(pred_neighbors_bge_labels, text_bge_features)
    tse_sver_total = calculate_sver(pred_neighbors_tse_labels, text_tse_features)
    pred_binary_label = joint_selecting(bge_sver_total, tse_sver_total, threshold)
    pred = reorder_binary_label(pred_binary_label, train_loader, feats_index_total, batch_size)
    
    return pred

def ncnv5(model, device, args, epoch, real_label, train_loader, threshold, num_neighbor=20):
    model.eval()
    batch_size = args.batch_size
    image_bge_features, text_bge_features, image_tse_features, text_tse_features, feats_index_total = get_bge_and_tse_featuers(model.eval(), train_loader, batch_size, device)
    neighbors_bge_feats, neighbors_tse_feats = find_joint_kn_neighbors(image_bge_features, image_tse_features, num_neighbor, batch_size)
    pred_neighbors_bge_labels, pred_neighbors_tse_labels = pred_joint_neighbor_label(neighbors_bge_feats, neighbors_tse_feats, text_bge_features, text_tse_features)
    bge_sver_total = calculate_sver(pred_neighbors_bge_labels, text_bge_features)
    tse_sver_total = calculate_sver(pred_neighbors_tse_labels, text_tse_features)
    if epoch % 10 == 0:
        print("drawing...")
        statistic_and_draw(args, train_loader, epoch, bge_sver_total, tse_sver_total, feats_index_total, real_label)
    
    pred_binary_label = joint_selecting(bge_sver_total, tse_sver_total, threshold)
    pred = reorder_binary_label(pred_binary_label, train_loader, feats_index_total, batch_size)
    
    return pred



def ncnv6(model, device, args, epoch, real_label, train_loader, threshold, num_neighbor=20):
    model.eval()
    batch_size = args.batch_size
    image_bge_features, text_bge_features, image_tse_features, text_tse_features, feats_index_total = get_bge_and_tse_featuers(model.eval(), train_loader, batch_size, device)
    neighbors_bge_feats, neighbors_tse_feats = find_joint_kn_neighbors(image_bge_features, image_tse_features, num_neighbor, batch_size)
    pred_neighbors_bge_labels, pred_neighbors_tse_labels = pred_joint_neighbor_label(neighbors_bge_feats, neighbors_tse_feats, text_bge_features, text_tse_features)
    bge_sver_total = calculate_sver(pred_neighbors_bge_labels, text_bge_features)
    tse_sver_total = calculate_sver(pred_neighbors_tse_labels, text_tse_features)
    if epoch % 10 == 0 or epoch == 12:
        print("drawing...")
        statistic_and_draw(args, train_loader, epoch, bge_sver_total, tse_sver_total, feats_index_total, real_label)      
    pred_binary_label = gmm_selecting(bge_sver_total, tse_sver_total)
    pred = reorder_binary_label(pred_binary_label, train_loader, feats_index_total, batch_size)

    return pred

def ncnv7(model, device, args, train_loader, threshold, epoch, num_neighbor=20):
    model.eval()
    print("new ncnv7 1")
    batch_size = args.batch_size
    print(0)
    image_bge_features, text_bge_features, image_tse_features, text_tse_features, feats_index_total = get_bge_and_tse_featuers(model.eval(), train_loader, batch_size, device)
    print(1)
    neighbors_bge_feats, neighbors_tse_feats, neighbors_sims = find_joint_kn_neighbors_with_sims(image_bge_features, image_tse_features, num_neighbor, batch_size, epoch)
    print(2)
    # print(neighbors_bge_feats[0].shape)
    # print(neighbors_tse_feats[0].shape)
    # print(neighbors_sims[0].shape)
    pred_neighbors_bge_labels, pred_neighbors_tse_labels = pred_joint_neighbor_label(neighbors_bge_feats, neighbors_tse_feats, text_bge_features, text_tse_features)
    print(3)
    bge_sver_total = calculate_weighted_sver(pred_neighbors_bge_labels, text_bge_features, neighbors_sims)
    print(4)
    tse_sver_total = calculate_weighted_sver(pred_neighbors_tse_labels, text_tse_features, neighbors_sims)
    print(5)
    pred_binary_label = joint_selecting(bge_sver_total, tse_sver_total, threshold)
    print(6)
    print(len(pred_binary_label))
    pred = reorder_binary_label(pred_binary_label, train_loader, feats_index_total, batch_size)
    print("done")
    
    return pred

def ncnv7_pro(model, device, args, train_loader, threshold, epoch, prob_A, prob_B, num_neighbor=20):
    model.eval()
    print("new ncnv7 1")
    batch_size = args.batch_size
    print(0)
    image_bge_features, text_bge_features, image_tse_features, text_tse_features, feats_index_total = get_bge_and_tse_featuers(model.eval(), train_loader, batch_size, device)
    print(1)
    neighbors_bge_feats, neighbors_tse_feats, neighbors_sims = find_joint_kn_neighbors_with_sims(image_bge_features, image_tse_features, num_neighbor, batch_size, epoch)
    print(2)
    # print(neighbors_bge_feats[0].shape)
    # print(neighbors_tse_feats[0].shape)
    # print(neighbors_sims[0].shape)
    pred_neighbors_bge_labels, pred_neighbors_tse_labels = pred_joint_neighbor_label(neighbors_bge_feats, neighbors_tse_feats, text_bge_features, text_tse_features)
    print(3)
    bge_sver_total = calculate_weighted_sver(pred_neighbors_bge_labels, text_bge_features, neighbors_sims)
    print(4)
    tse_sver_total = calculate_weighted_sver(pred_neighbors_tse_labels, text_tse_features, neighbors_sims)
    
    
    reorder_bge_sver_total = reorder_binary_label(bge_sver_total, train_loader, feats_index_total, batch_size)
    reorder_tse_sver_total = reorder_binary_label(tse_sver_total, train_loader, feats_index_total, batch_size)
    reorder_bge_sver_total = torch.Tensor(reorder_bge_sver_total)
    reorder_tse_sver_total = torch.Tensor(reorder_tse_sver_total)
    
    print(5)
    pred_binary_label = joint_selecting(bge_sver_total, tse_sver_total, threshold)
    pred = torch.Tensor(pred_binary_label)
    
    
    
    
    
    # print(6)
    # print(len(pred_binary_label))
    # pred = reorder_binary_label(pred_binary_label, train_loader, feats_index_total, batch_size)
    # print("done")
    
    return pred

def ncnv7_for_more_pairs(model, device, args, train_loader, threshold, epoch, num_neighbor=20):
    model.eval()
    pred_binary_label_all = []
    print("new ncnv7 1")
    batch_size = args.batch_size
    print(0)
    image_bge_features, text_bge_features, image_tse_features, text_tse_features, feats_index_total = get_bge_and_tse_featuers(model.eval(), train_loader, batch_size, device)
    print(1)
    neighbors_bge_feats, neighbors_tse_feats, neighbors_sims = find_joint_kn_neighbors_with_sims(image_bge_features, image_tse_features, num_neighbor, batch_size, epoch)
    print(2)
    for i in range(len(neighbors_bge_feats)):
        # print(i)
        batch_neighbors_bge_feats = [neighbors_bge_feats[i]]
        batch_neighbors_tse_feats = [neighbors_tse_feats[i]]
        batch_neighbors_sims = [neighbors_sims[i]]
        pred_neighbors_bge_labels, pred_neighbors_tse_labels = pred_joint_neighbor_label(batch_neighbors_bge_feats, batch_neighbors_tse_feats, text_bge_features, text_tse_features)
        # print(3)
        bge_sver_total = calculate_weighted_sver(pred_neighbors_bge_labels, text_bge_features, batch_neighbors_sims)
        # print(4)
        tse_sver_total = calculate_weighted_sver(pred_neighbors_tse_labels, text_tse_features, batch_neighbors_sims)
        # print(5)
        pred_binary_label = joint_selecting(bge_sver_total, tse_sver_total, threshold)
        print((i + 1) * len(pred_binary_label))
        # print(6)
        pred_binary_label_all.append(pred_binary_label)
    pred_binary_label_all = [item for sublist in pred_binary_label_all for item in sublist]
    print(len(pred_binary_label_all))
    batch_size = 512
    pred = reorder_binary_label(pred_binary_label_all, train_loader, feats_index_total, batch_size)
    print("done")
    
    return pred

def ncnv8(model, device, args, train_loader, threshold, num_neighbor=20):
    model.eval()
    batch_size = args.batch_size
    image_bge_features, text_bge_features, image_tse_features, text_tse_features, feats_index_total = get_bge_and_tse_featuers(model.eval(), train_loader, batch_size, device)
    neighbors_bge_ifeats, neighbors_tse_ifeats, neighbors_isims = find_joint_kn_neighbors_with_sims(image_bge_features,
                                                                                                 image_tse_features,
                                                                                                 num_neighbor,
                                                                                                 batch_size)
    pred_neighbors_bge_tlabels, pred_neighbors_tse_tlabels = pred_joint_neighbor_label(neighbors_bge_ifeats,
                                                                                     neighbors_tse_ifeats,
                                                                                     text_bge_features,
                                                                                     text_tse_features)
    i_based_bge_sver_total = calculate_weighted_sver(pred_neighbors_bge_tlabels, text_bge_features, neighbors_isims)
    i_based_tse_sver_total = calculate_weighted_sver(pred_neighbors_tse_tlabels, text_tse_features, neighbors_isims)
    i_based_sver_total = torch.cat([t.unsqueeze(0) for t in i_based_bge_sver_total]) + torch.cat([t.unsqueeze(0) for t in i_based_tse_sver_total])
    #
    neighbors_bge_tfeats, neighbors_tse_tfeats, neighbors_tsims = find_joint_kn_neighbors_with_sims(text_bge_features,
                                                                                                    text_tse_features,
                                                                                                    num_neighbor,
                                                                                                    batch_size)
    pred_neighbors_bge_ilabels, pred_neighbors_tse_ilabels = pred_joint_neighbor_label(neighbors_bge_tfeats,
                                                                                       neighbors_tse_tfeats,
                                                                                       image_bge_features,
                                                                                       image_tse_features)
    t_based_bge_sver_total = calculate_weighted_sver(pred_neighbors_bge_ilabels, image_bge_features, neighbors_tsims)
    t_based_tse_sver_total = calculate_weighted_sver(pred_neighbors_tse_ilabels, image_tse_features, neighbors_tsims)
    t_based_sver_total = torch.cat([t.unsqueeze(0) for t in t_based_bge_sver_total]) + torch.cat([t.unsqueeze(0) for t in t_based_tse_sver_total])

    mask = (i_based_sver_total > 0.4) & (i_based_sver_total < threshold)
    i_based_sver_total[mask] = (i_based_sver_total[mask] + t_based_sver_total[mask]) / 2
    sver_total = i_based_sver_total
    pred_binary_label = selecting(sver_total, threshold)
    pred = reorder_binary_label(pred_binary_label, train_loader, feats_index_total, batch_size)
    
    return pred

def ncnv9_1(model, device, args, train_loader, threshold, num_neighbor=20):
    model.eval()
    batch_size = args.batch_size
    image_bge_features, text_bge_features, image_tse_features, text_tse_features, feats_index_total = get_bge_and_tse_featuers(model.eval(), train_loader, batch_size, device)
    neighbors_bge_ifeats, neighbors_tse_ifeats, neighbors_isims = find_joint_kn_neighbors_with_sims(image_bge_features,
                                                                                                 image_tse_features,
                                                                                                 num_neighbor,
                                                                                                 batch_size)
    pred_neighbors_bge_tlabels, pred_neighbors_tse_tlabels = pred_joint_neighbor_label(neighbors_bge_ifeats,
                                                                                     neighbors_tse_ifeats,
                                                                                     text_bge_features,
                                                                                     text_tse_features)
    i_based_bge_sver_total = calculate_weighted_sver(pred_neighbors_bge_tlabels, text_bge_features, neighbors_isims)
    i_based_tse_sver_total = calculate_weighted_sver(pred_neighbors_tse_tlabels, text_tse_features, neighbors_isims)
    i_based_sver_total = (torch.cat([t.unsqueeze(0) for t in i_based_bge_sver_total]) + torch.cat([t.unsqueeze(0) for t in i_based_tse_sver_total])) / 2
    
    return i_based_sver_total, image_bge_features, text_bge_features, image_tse_features, text_tse_features, feats_index_total

def ncnv9_2(model, device, args, train_loader, threshold, i_based_sver_total, image_bge_features, text_bge_features, image_tse_features, text_tse_features, feats_index_total,num_neighbor=20):
    model.eval()
    batch_size = args.batch_size
    neighbors_bge_tfeats, neighbors_tse_tfeats, neighbors_tsims = find_joint_kn_neighbors_with_sims(text_bge_features,
                                                                                                    text_tse_features,
                                                                                                    num_neighbor,
                                                                                                    batch_size)
    pred_neighbors_bge_ilabels, pred_neighbors_tse_ilabels = pred_joint_neighbor_label(neighbors_bge_tfeats,
                                                                                       neighbors_tse_tfeats,
                                                                                       image_bge_features,
                                                                                       image_tse_features)
    t_based_bge_sver_total = calculate_weighted_sver(pred_neighbors_bge_ilabels, image_bge_features, neighbors_tsims)
    t_based_tse_sver_total = calculate_weighted_sver(pred_neighbors_tse_ilabels, image_tse_features, neighbors_tsims)
    t_based_sver_total = (torch.cat([t.unsqueeze(0) for t in t_based_bge_sver_total]) + torch.cat([t.unsqueeze(0) for t in t_based_tse_sver_total])) / 2
    
    # sver_total = (i_based_sver_total +  t_based_sver_total) / 2
    
    mask_left = (i_based_sver_total > 0.45) & (i_based_sver_total < threshold)
    i_based_sver_total[mask_left] = (i_based_sver_total[mask_left] + t_based_sver_total[mask_left]) / 2 - 0.05
    
    mask_right = (i_based_sver_total >= threshold) & (i_based_sver_total < 0.6)
    i_based_sver_total[mask_right] = (i_based_sver_total[mask_right] + t_based_sver_total[mask_right]) / 2
    
    
    sver_total = i_based_sver_total
    pred_binary_label = selecting(sver_total, threshold)
    pred = reorder_binary_label(pred_binary_label, train_loader, feats_index_total, batch_size) 
    
    return pred

def calculate_weighted_sver(pred_neighbors_labels, correspond_label, neighbors_sims):
    neighbor_sims = torch.cat(neighbors_sims, dim=0)
    # print(neighbor_sims.shape)
    weight = torch.softmax(neighbor_sims, dim=1)
    weighted_sver_total = []
    
    index = 0
    for per_sample_neighbors_labels in pred_neighbors_labels:
        scores = correspond_label[index] @ per_sample_neighbors_labels.t()
        weighted_sver = sum(weight[index] * scores)
        weighted_sver_total.append(weighted_sver)
        index += 1

    return weighted_sver_total

def gmm_selecting(bge_sver_total, tse_sver_total):
    bge_sver_total = np.array(torch.tensor(bge_sver_total).cpu())
    tse_sver_total = np.array(torch.tensor(tse_sver_total).cpu())
    sver_total = (tse_sver_total + bge_sver_total) / 2
    data = sver_total.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
    gmm.fit(data)
    prob = gmm.predict_proba(data)
    prob = prob[:, gmm.means_.argmax()]
    pred = split_prob(prob, 0.5)

    return torch.from_numpy(pred)

def split_prob(prob, threshld):
    if prob.min() > threshld:
        """From https://github.com/XLearning-SCU/2021-NeurIPS-NCR"""
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print('No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled.')
        threshld = np.sort(prob)[len(prob)//100]
    pred = (prob > threshld)
    return (pred+0)

def statistic_and_draw(args, train_loader, epoch, bge_sver_total, tse_sver_total, feats_index_total, real_label):
    batch_size = args.batch_size
    reorded_bge_sver_total = reorder_binary_label(bge_sver_total, train_loader, feats_index_total, batch_size)
    reorded_tse_sver_total = reorder_binary_label(tse_sver_total, train_loader, feats_index_total, batch_size)
    tse_sver_total = np.array(reorded_tse_sver_total.cpu())
    bge_sver_total = np.array(reorded_bge_sver_total.cpu())
    
    sver_total = (tse_sver_total + bge_sver_total) / 2
    noisy = sver_total[np.array(real_label) == 0]
    clean = sver_total[np.array(real_label) == 1]
    plt.style.use('_mpl-gallery')
    data = sver_total
    data = data.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm.fit(data)
    z = np.linspace(data.min() - 1, data.max() + 1, 1000).reshape(-1, 1)
    logprob = gmm.score_samples(z)
    pdf = np.exp(logprob)
    # 绘制每个组件的曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['red', 'green']
    for i in range(gmm.n_components):
        mean = gmm.means_[i]
        cov = gmm.covariances_[i]
        weight = gmm.weights_[i]  # 获取组件的权重
        component_pdf = weight * norm.pdf(z, loc=mean[0], scale=np.sqrt(cov[0][0]))
        ax.plot(z, component_pdf, color=colors[i], linestyle='--', label=f'Component {i + 1}', linewidth=1.5)
    # 绘制总拟合曲线
    ax.plot(z, pdf, '-k', label='GMM PDF', linewidth=1.2, alpha=0.5)
    
  
    # 绘制直方图
    
    ax.hist(noisy, bins=400, linewidth=0, edgecolor="white", color='red', alpha=0.3, density=True, label="Noisy Pairs")
    ax.hist(clean, bins=400, linewidth=0, edgecolor="white", color="green", alpha=0.3, density=True, label="Clean Pairs")
    ax.set(xlim=(0, 20), xticks=np.arange(1, 20),
           ylim=(0, 1), yticks=np.arange(0, 1, 0.2))
    plt.xlabel('Score')
    plt.ylabel('Density')

    plt.tight_layout()
    ax.legend()
    print("finish drawing")
    plt.savefig(str(epoch) + 'example_plot.png', dpi=500)



def joint_selecting(bge_sver_total, tse_sver_total, threshold):
    pred_binary_label = []

    for i in range(len(bge_sver_total)):
        bge_sver = bge_sver_total[i]
        tse_sver = tse_sver_total[i]
        sver = (bge_sver.item() + tse_sver.item()) / 2
        
        if sver > threshold:
            pred_binary_label.append(1)
        else:
            pred_binary_label.append(0)

    return pred_binary_label
    
    
def pred_joint_neighbor_label(neighbors_bge_feats, neighbors_tse_feats, bge_label, tse_label):
    # print(len(neighbors_bge_feats))
    # print(neighbors_bge_feats[0].shape)
#     neighbors_bge_feats = [tensor.cpu() for tensor in neighbors_bge_feats]
#     neighbors_tse_feats = [tensor.cpu() for tensor in neighbors_tse_feats]
#     neighbors_bge_feats = torch.cat(neighbors_bge_feats, dim = 0).cpu()
#     neighbors_tse_feats = torch.cat(neighbors_tse_feats, dim = 0).cpu()
#     print(neighbors_bge_feats.shape)

#     with torch.no_grad():
#         for i in range(len(neighbors_bge_feats)):
#             print(i)
#             per_sample_neighbors_bge_feats = neighbors_bge_feats[i].cuda()
#             per_sample_neighbors_tse_feats = neighbors_tse_feats[i].cuda()
#             bge_sims = torch.mm(per_sample_neighbors_bge_feats, bge_label.t())
#             tse_sims = torch.mm(per_sample_neighbors_tse_feats, tse_label.t())
#             sims = (bge_sims + tse_sims) / 2
#             _, pred_label_index = sims.topk(1, dim=1, largest=True, sorted=True)
#             pred_label_index_flattened = pred_label_index.view(-1)
#             pred_bge_label = bge_label[pred_label_index_flattened]
#             pred_tse_label = tse_label[pred_label_index_flattened]
#             pred_bge_label_total.append(pred_bge_label)
#             pred_tse_label_total.append(pred_tse_label)
            
#             del per_sample_neighbors_bge_feats, per_sample_neighbors_tse_feats, bge_sims, tse_sims, pred_label_index, pred_label_index_flattened, pred_bge_label, pred_tse_label
#             torch.cuda.empty_cache()
    pred_bge_label_total = []
    pred_tse_label_total = []    
    
    with torch.no_grad():
        for i in range(len(neighbors_bge_feats)):
            neighbors_batch_bge_feats = neighbors_bge_feats[i]
            neighbors_batch_tse_feats = neighbors_tse_feats[i]

            for j in range(len(neighbors_batch_bge_feats)):
                per_sample_neighbors_bge_feats = neighbors_batch_bge_feats[j]
                per_sample_neighbors_tse_feats = neighbors_batch_tse_feats[j]

                # print(per_sample_neighbors_feats.shape)
                # print(label.t().shape)
                # print(per_sample_neighbors_bge_feats.shape)
                bge_sims = torch.mm(per_sample_neighbors_bge_feats, bge_label.t())
                tse_sims = torch.mm(per_sample_neighbors_tse_feats, tse_label.t())
                sims = (bge_sims + tse_sims) / 2
                _, pred_label_index = sims.topk(1, dim=1, largest=True, sorted=True)
                pred_label_index_flattened = pred_label_index.view(-1)
                pred_bge_label = bge_label[pred_label_index_flattened]
                pred_tse_label = tse_label[pred_label_index_flattened]
                pred_bge_label_total.append(pred_bge_label)
                pred_tse_label_total.append(pred_tse_label)

    # print(len(pred_label_total))
    # print(pred_label_total[0].shape)
    return pred_bge_label_total, pred_tse_label_total
    
def pred_joint_neighbor_label1(neighbors_bge_feats, neighbors_tse_feats, bge_label, tse_label):
    pred_bge_label_total = []
    pred_tse_label_total = []
    
    print("nnn")
    
    alpha = 0.4
    k = 128
    reference_embeddings = bge_label.cpu().numpy()
    nnn_retriever_bge = NNNRetriever(bge_label.shape[1], use_gpu=True, gpu_id = 0)
    nnn_ranker_bge = NNNRanker(nnn_retriever_bge, bge_label.cpu().numpy(), reference_embeddings, alternate_ks=k,
                                       alternate_weight=alpha, batch_size=256, use_gpu=True, gpu_id = 0)
    
    reference_embeddings = tse_label.cpu().numpy()
    nnn_retriever_tse = NNNRetriever(tse_label.shape[1], use_gpu=True, gpu_id = 0)
    nnn_ranker_tse = NNNRanker(nnn_retriever_tse, tse_label.cpu().numpy(), reference_embeddings, alternate_ks=k,
                                       alternate_weight=alpha, batch_size=256, use_gpu=True, gpu_id = 0)

    for i in range(len(neighbors_bge_feats)):
        neighbors_batch_bge_feats = neighbors_bge_feats[i]
        neighbors_batch_tse_feats = neighbors_tse_feats[i]

        for j in range(len(neighbors_batch_bge_feats)):
            per_sample_neighbors_bge_feats = neighbors_batch_bge_feats[j]
            per_sample_neighbors_tse_feats = neighbors_batch_tse_feats[j]

            # print(per_sample_neighbors_feats.shape)
            # print(label.t().shape)
            bge_sims = torch.mm(per_sample_neighbors_bge_feats, bge_label.t())
            tse_sims = torch.mm(per_sample_neighbors_tse_feats, tse_label.t())
            
            
            nnn_scores_bge, nnn_indices1 = nnn_ranker_bge.search(per_sample_neighbors_bge_feats.cpu().numpy(), k)
            nnn_scores_tse, nnn_indices2 = nnn_ranker_tse.search(per_sample_neighbors_tse_feats.cpu().numpy(), k)
            
            sims = torch.from_numpy((nnn_scores_bge + nnn_scores_tse) / 2)

            
            
            
            
            # sims = bge_sims
            
            
            
            _, pred_label_index = sims.topk(1, dim=1, largest=True, sorted=True)
            pred_label_index_flattened = pred_label_index.view(-1)
            pred_bge_label = bge_label[pred_label_index_flattened]
            pred_tse_label = tse_label[pred_label_index_flattened]
            pred_bge_label_total.append(pred_bge_label)
            pred_tse_label_total.append(pred_tse_label)

    # print(len(pred_label_total))
    # print(pred_label_total[0].shape)
    return pred_bge_label_total, pred_tse_label_total
    

def get_ordered_tse_features(model, train_loader, batch_size, device):
    feat_dim = 2048
    text_features = torch.rand(len(train_loader.dataset), feat_dim).t().cuda() # (512, 68126)
    image_features = torch.rand(len(train_loader.dataset), feat_dim).t().cuda() # (512, 68126)
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['images'].to(device)
        caption_ids = batch['caption_ids'].to(device)
        current_batch_size = len(caption_ids)
        image_batch_feats, atten_i, text_batch_feats, atten_t = model.base_model(images, caption_ids)
        index = batch['index']
        image_batch_feats = model.visul_emb_layer(image_batch_feats, atten_i)
        text_batch_feats = model.texual_emb_layer(text_batch_feats, caption_ids, atten_t)
        with torch.no_grad(): 
            for b in range(current_batch_size):
                image_features.t()[index[b]] = image_batch_feats[b]
                text_features.t()[index[b]] = text_batch_feats[b]
                
    image_features = normalize(image_features.t())
    text_features = normalize(text_features.t())
    
    return image_features, text_features

def get_ordered_bge_features(model, train_loader, batch_size, device):
    feat_dim = 512
    text_features = torch.rand(len(train_loader.dataset), feat_dim).t().cuda() # (512, 68126)
    image_features = torch.rand(len(train_loader.dataset), feat_dim).t().cuda() # (512, 68126)
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['images'].to(device)
        caption_ids = batch['caption_ids'].to(device)
        current_batch_size = len(caption_ids)
        image_batch_feats, atten_i, text_batch_feats, atten_t = model.base_model(images, caption_ids)
        index = batch['index']
        image_batch_feats = image_batch_feats[:, 0, :].float()
        text_batch_feats = text_batch_feats[torch.arange(text_batch_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        with torch.no_grad(): 
            for b in range(current_batch_size):
                image_features.t()[index[b]] = image_batch_feats[b]
                text_features.t()[index[b]] = text_batch_feats[b]
                
    image_features = normalize(image_features.t())
    text_features = normalize(text_features.t())
    
    return image_features, text_features



def get_tse_features(model, train_loader, batch_size, device):
    feat_dim = 1024
    text_features = torch.rand(len(train_loader.dataset), feat_dim).t().cuda()  # (1024, 68126)
    image_features = torch.rand(len(train_loader.dataset), feat_dim).t().cuda()  # (1024, 68126)

    feats_index_total = []
    for batch_idx, batch in enumerate(train_loader):
        images = batch['images'].to(device)
        caption_ids = batch['caption_ids'].to(device)
        index = batch['index']
        feats_index_total.append(index)
        current_batch_size = len(caption_ids)
        image_batch_feats, atten_i, text_batch_feats, atten_t = model.base_model(images, caption_ids)
        image_batch_feats = model.visul_emb_layer(image_batch_feats, atten_i)
        text_batch_feats = model.texual_emb_layer(text_batch_feats, caption_ids, atten_t)

        image_features[:, batch_idx * batch_size:batch_idx * batch_size + current_batch_size] = image_batch_feats.data.t()

        text_features[:, batch_idx * batch_size:batch_idx * batch_size + current_batch_size] = text_batch_feats.data.t()

    # 特征归一化
    image_features = normalize(image_features.t())
    text_features = normalize(text_features.t())
    
    return image_features, text_features, feats_index_total

def get_bge_features(model, train_loader, batch_size, device):
    feat_dim = 512
    # 初始化
    text_features = torch.rand(len(train_loader.dataset), feat_dim).t().cuda()  # (512, 68126)
    image_features = torch.rand(len(train_loader.dataset), feat_dim).t().cuda()  # (512, 68126)

    feats_index_total = []
    for batch_idx, batch in enumerate(train_loader):
        images = batch['images'].to(device)
        caption_ids = batch['caption_ids'].to(device)
        index = batch['index']
        feats_index_total.append(index)
        current_batch_size = len(caption_ids)
        image_batch_feats, atten_i, text_batch_feats, atten_t = model.base_model(images, caption_ids)

        image_batch_feats = image_batch_feats[:, 0, :].float()
        image_features[:, batch_idx * batch_size:batch_idx * batch_size + current_batch_size] = image_batch_feats.data.t()
        text_batch_feats = text_batch_feats[torch.arange(text_batch_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        text_features[:, batch_idx * batch_size:batch_idx * batch_size + current_batch_size] = text_batch_feats.data.t()

    # 特征归一化
    image_features = normalize(image_features.t())
    text_features = normalize(text_features.t())
    return image_features, text_features, feats_index_total

def get_bge_and_tse_featuers(model, train_loader, batch_size, device):
    bge_feat_dim = 512
    tse_feat_dim = 1024
    # batch_size = 512
    with torch.no_grad():
        bge_text_features = torch.rand(len(train_loader.dataset), bge_feat_dim).t().cuda()  # (512, 68126)
        bge_image_features = torch.rand(len(train_loader.dataset), bge_feat_dim).t().cuda()  # (512, 68126)

        tse_text_features = torch.rand(len(train_loader.dataset), tse_feat_dim).t().cuda() # (1024, 68126)
        tse_image_features = torch.rand(len(train_loader.dataset), tse_feat_dim).t().cuda()  # (1024, 68126)

        feats_index_total = []
        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            caption_ids = batch['caption_ids'].to(device)
            index = batch['index']
            feats_index_total.append(index)
            current_batch_size = len(caption_ids)
            bge_image_batch_feats, atten_i, bge_text_batch_feats, atten_t = model.base_model(images, caption_ids)

            tse_image_batch_feats = model.visul_emb_layer(bge_image_batch_feats, atten_i)
            tse_text_batch_feats = model.texual_emb_layer(bge_text_batch_feats, caption_ids, atten_t)

            bge_image_batch_feats = bge_image_batch_feats[:, 0, :].float()
            bge_image_features[:, batch_idx * batch_size:batch_idx * batch_size + current_batch_size] = bge_image_batch_feats.data.t()
            
            bge_text_batch_feats = bge_text_batch_feats[torch.arange(bge_text_batch_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            bge_text_features[:, batch_idx * batch_size:batch_idx * batch_size + current_batch_size] = bge_text_batch_feats.data.t()
            
            tse_image_features[:, batch_idx * batch_size:batch_idx * batch_size + current_batch_size] = tse_image_batch_feats.data.t()
            tse_text_features[:, batch_idx * batch_size:batch_idx * batch_size + current_batch_size] = tse_text_batch_feats.data.t()
            
            
        # 特征归一化
        bge_image_features = normalize(bge_image_features.t())
        bge_text_features = normalize(bge_text_features.t())
        tse_image_features = normalize(tse_image_features.t())
        tse_text_features = normalize(tse_text_features.t())
    
    return bge_image_features, bge_text_features, tse_image_features, tse_text_features, feats_index_total

def find_kn_neighbors(sample_features, num_neighbor, batch_size):
    neighbors_index = []
    neighbors_feats= []
    num_batch = math.ceil(sample_features.shape[0] / batch_size)

    for batch_idx in range(num_batch):
        sample_batch_feats = sample_features[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        dist = torch.mm(sample_batch_feats, sample_features.t())
        # dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1
        _, batch_neighbors_index = dist.topk(num_neighbor, dim=1, largest=True, sorted=True)  # find contrastive neighbors
        neighbors_index.append(batch_neighbors_index)

    for index in neighbors_index:
        batch_feats = sample_features[index] 
        neighbors_feats.append(batch_feats)

    return neighbors_feats

def find_joint_kn_neighbors(sample_bge_features, sample_tse_features, num_neighbor, batch_size):
    neighbors_index = []
    neighbors_bge_feats= []
    neighbors_tse_feats = []

    num_batch = math.ceil(sample_bge_features.shape[0] / batch_size)

    for batch_idx in range(num_batch):
        bge_sample_batch_feats = sample_bge_features[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        tse_sample_batch_feats = sample_tse_features[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        bge_dist = torch.mm(bge_sample_batch_feats, sample_bge_features.t())
        tse_dist = torch.mm(tse_sample_batch_feats, sample_tse_features.t())
        joint_dist = (bge_dist + tse_dist) / 2
       
        # dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1
        _, batch_neighbors_index = joint_dist.topk(num_neighbor, dim=1, largest=True, sorted=True)  # find contrastive neighbors
        neighbors_index.append(batch_neighbors_index)
        
    for index in neighbors_index:
        batch_bge_feats = sample_bge_features[index]
        batch_tse_feats = sample_tse_features[index]
        neighbors_tse_feats.append(batch_tse_feats)
        neighbors_bge_feats.append(batch_bge_feats)

    return neighbors_bge_feats, neighbors_tse_feats

def find_joint_kn_neighbors_with_sims(sample_bge_features, sample_tse_features, num_neighbor, batch_size, epoch):
    neighbors_index = []
    neighbors_bge_feats= []
    neighbors_tse_feats = []
    neighbors_sims = []
    batch_size = 2048
    num_batch = math.ceil(sample_bge_features.shape[0] / batch_size)

    for batch_idx in range(num_batch):
        # 取出一个batch的特征
        bge_sample_batch_feats = sample_bge_features[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        tse_sample_batch_feats = sample_tse_features[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        
        bge_dist = torch.mm(bge_sample_batch_feats, sample_bge_features.t())
        tse_dist = torch.mm(tse_sample_batch_feats, sample_tse_features.t())
        # if epoch > 29:
        #     joint_dist = (bge_dist + 4 * tse_dist) / 5
        # else:
        joint_dist = (bge_dist + tse_dist) / 2
        # joint_dist = (bge_dist + tse_dist) / 2
        # dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1
        batch_neighbors_sims, batch_neighbors_index = joint_dist.topk(num_neighbor, dim=1, largest=True, sorted=True)  # find contrastive neighbors
        neighbors_index.append(batch_neighbors_index)
        neighbors_sims.append(batch_neighbors_sims)
        
    for index in neighbors_index:
        batch_bge_feats = sample_bge_features[index]
        batch_tse_feats = sample_tse_features[index]
        neighbors_tse_feats.append(batch_tse_feats)
        neighbors_bge_feats.append(batch_bge_feats)

    return neighbors_bge_feats, neighbors_tse_feats, neighbors_sims


def find_k_reciprocal_neighbors(sample_features):
    reranked_dist_total = []
    batch_size = 4
    num_batch = math.ceil(sample_features.shape[0] / batch_size)
    for batch_idx in range(num_batch):
        print(batch_idx)
        mem1 = psutil.virtual_memory()
        print(mem1.used)
        q_feats = sample_features[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        print(q_feats.shape)
        g_feats = sample_features
        q_g_dist = torch.mm(q_feats, g_feats.t()).cpu().numpy()
        q_q_dist = torch.mm(q_feats, q_feats.t()).cpu().numpy()
        g_g_dist = torch.mm(g_feats, g_feats.t()).cpu().numpy()
        mem2 = psutil.virtual_memory()
        print(mem2.used)
        reranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        reranked_dist_total.append(reranked_dist)
    
def pred_neighbor_label(neighbors_feats, label):
    pred_label_total = []

    for neighbors_batch_feats in neighbors_feats:
        for per_sample_neighbors_feats in neighbors_batch_feats:
            # print(per_sample_neighbors_feats.shape)
            # print(label.t().shape)
            sims = torch.mm(per_sample_neighbors_feats, label.t())
            _, pred_label_index = sims.topk(1, dim=1, largest=True, sorted=True)
            pred_label_index_flattened = pred_label_index.view(-1)
            pred_label = label[pred_label_index_flattened]
            pred_label_total.append(pred_label)
            
    # print(len(pred_label_total))
    # print(pred_label_total[0].shape)
    return pred_label_total

def calculate_sver(pred_neighbors_labels, correspond_label):
    sver_total = []
    index = 0

    for per_sample_neighbors_labels in pred_neighbors_labels:
        scores = correspond_label[index] @ per_sample_neighbors_labels.t()
        sver = sum(scores)
        sver_total.append(sver)
        index += 1

    return sver_total

def selecting(sver_total, threshold):
    pred_binary_label = []

    for sver in sver_total:
        if sver.item() > threshold:
            pred_binary_label.append(1)
        else:
            pred_binary_label.append(0)

    return pred_binary_label

def reorder_binary_label(pred_binary_label, train_loader, index_total, batch_size):
    pred = torch.zeros(len(train_loader.dataset))
    batch_idx = 0

    for index in index_total:
        # print(pred[index].shape)
        # print(torch.tensor(pred_binary_index)[batch_idx * batch_size:batch_idx * batch_size + len(index)].dtype)
        # print(pred[index].dtype)
        pred[index] = torch.tensor(pred_binary_label)[batch_idx * batch_size:batch_idx * batch_size + len(index)].to(torch.float32)
        batch_idx += 1
    return pred

#########################################################################################################################################################################

# import torch
# import math
# from torch.nn.functional import normalize

def ncnv(model, device, args, train_loader, threshold, feat_dim=512, num_neighbor=20):
    model.eval()
    batch_size = args.batch_size
    # loading given samples
    # 随机初始化两个tensor分别用来装所有的文本特征（trainFeatures）和所有的图片特征（imageFeatures）
    trainFeatures = torch.rand(len(train_loader.dataset), feat_dim).t().cuda() # (512, 68126)
    imageFeatures = torch.rand(len(train_loader.dataset), feat_dim).t().cuda() # (512, 68126)
    
#     for batch_idx, batch in enumerate(train_loader):
#         images = batch['images'].to(device)
#         caption_ids = batch['caption_ids'].to(device)
#         current_batch_size = len(caption_ids)
#         image_feats, atten_i, text_feats, atten_t = model.base_model(images, caption_ids)

#         # 获取所有的图片特征，以供后面"CLIP式"的推理使用
#         # 获取所有的文本特征，使用文本特征作为candidate sample，对应的图片特征作为标签
#         index = batch['index']
#         image_feats = image_feats[:, 0, :].float()
#         text_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
#         with torch.no_grad(): 
#             for b in range(current_batch_size):
#                 imageFeatures.t()[index[b]] = image_feats[b]
#                 trainFeatures.t()[index[b]] = text_feats[b]
    index_total = []
    for batch_idx, batch in enumerate(train_loader):
        images = batch['images'].to(device)
        caption_ids = batch['caption_ids'].to(device)
        index = batch['index']
        index_total.append(index)
        current_batch_size = len(caption_ids)
        image_feats, atten_i, text_feats, atten_t = model.base_model(images, caption_ids)

        image_feats = image_feats[:, 0, :].float()
        imageFeatures[:, batch_idx * batch_size:batch_idx * batch_size + current_batch_size] = image_feats.data.t()
        text_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        trainFeatures[:, batch_idx * batch_size:batch_idx * batch_size + current_batch_size] = text_feats.data.t()
                
                
                
    # 特征归一化
    imageFeatures = normalize(imageFeatures.t())
    trainFeatures = normalize(trainFeatures.t())
    print(imageFeatures.shape)

    # caculating neighborhood-based label inconsistency score
    num_batch = math.ceil(trainFeatures.shape[0] / batch_size)
    sver_collection = []
    pred_image_feats_total = []
    pred_binary_index = []

#     for batch_idx in range(num_batch):
#         text_feats = trainFeatures[batch_idx * batch_size:batch_idx * batch_size + batch_size] #(64, 512)
#         dist = torch.mm(text_feats, trainFeatures.t())
#         # dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1
#         _, neighbors = dist.topk(num_neighbor, dim=1, largest=True, sorted=True)  # find contrastive neighbors
#         # print(neighbors.shape) (64, 10)

#         # neighbors中的每一个item为一个样本的所有邻居的index
#         for item in neighbors: 
#             # print("item:" + str(item.shape)) (10)
#             # 通过索引获取一个样本的所有邻居的特征（文本特征），shape应为(num_neighbors, feat_dim)
#             neighbors_feats = trainFeatures[item]
#             # print("neighbors_feats:" + str(neighbors_feats.shape)) (10, 512)

#             # 使用现有模型对邻居的特征进行"CLIP式"的推理
#             # 计算邻居特征（文本特征）与所有图片特征的相似度
#             sims = torch.mm(neighbors_feats, imageFeatures.t())

#             # 对于每一个文本特征选取与之相似度最高的图片特征作为其标签
#             # pred_image_feats_index为这10个邻居的标签（对应的图片）的索引
#             _, pred_image_feats_index = sims.topk(1, dim=1, largest=True, sorted=True)
#             # print("pred_image_feats_index:" + str(pred_image_feats_index.shape)) (10, 1)

#             # 将索引展平
#             pred_image_feats_index_flattened = pred_image_feats_index.view(-1)
#             # 通过索引获得对应的标签（图片特征）
#             pred_image_feats = imageFeatures[pred_image_feats_index_flattened]
#             # print("pred_image_feats:" + str(pred_image_feats.shape)) (10, 512)

#             pred_image_feats_total.append(pred_image_feats)

#     index = 0
#     for neighbors_pred_image_feat in pred_image_feats_total:
#         scores = imageFeatures[index] @ neighbors_pred_image_feat.t()
#         sver = sum(scores)
#         sver_collection.append(sver)
#         index += 1
    # print(sver_collection)
    # print(len(sver_collection))
    
    for batch_idx in range(num_batch):
        # text_feats = trainFeatures[batch_idx * batch_size:batch_idx * batch_size + batch_size] #(64, 512)
        image_feats = imageFeatures[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        dist = torch.mm(image_feats, imageFeatures.t())
        # dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1
        _, neighbors = dist.topk(num_neighbor, dim=1, largest=True, sorted=True)  # find contrastive neighbors
        # print(neighbors.shape) (64, 10)

        # neighbors中的每一个item为一个样本的所有邻居的index
        for item in neighbors: 
            # print("item:" + str(item.shape)) (10)
            # 通过索引获取一个样本的所有邻居的特征（文本特征），shape应为(num_neighbors, feat_dim)
            neighbors_feats = imageFeatures[item]
            # print("neighbors_feats:" + str(neighbors_feats.shape)) (10, 512)

            # 使用现有模型对邻居的特征进行"CLIP式"的推理
            # 计算邻居特征（文本特征）与所有图片特征的相似度
            sims = torch.mm(neighbors_feats, trainFeatures.t())

            # 对于每一个文本特征选取与之相似度最高的图片特征作为其标签
            # pred_image_feats_index为这10个邻居的标签（对应的图片）的索引
            _, pred_image_feats_index = sims.topk(1, dim=1, largest=True, sorted=True)
            # print("pred_image_feats_index:" + str(pred_image_feats_index.shape)) (10, 1)

            # 将索引展平
            pred_image_feats_index_flattened = pred_image_feats_index.view(-1)
            # 通过索引获得对应的标签（图片特征）
            pred_image_feats = trainFeatures[pred_image_feats_index_flattened]
            # print("pred_image_feats:" + str(pred_image_feats.shape)) (10, 512)

            pred_image_feats_total.append(pred_image_feats)

    index = 0
    for neighbors_pred_image_feat in pred_image_feats_total:
        scores = trainFeatures[index] @ neighbors_pred_image_feat.t()
        sver = sum(scores)
        sver_collection.append(sver)
        index += 1

    for sver in sver_collection:
        if sver.item() > threshold:
            pred_binary_index.append(1)
        else:
            pred_binary_index.append(0)
    
    # pred_part_of_binary_index = torch.tensor(pred_binary_index)[0:30]
    # print(pred_part_of_binary_index)
    pred = torch.zeros(len(train_loader.dataset))
    batch_idx = 0
    for index in index_total:
        # print(pred[index].shape)
        # print(torch.tensor(pred_binary_index)[batch_idx * batch_size:batch_idx * batch_size + len(index)].dtype)
        # print(pred[index].dtype)
        pred[index] = torch.tensor(pred_binary_index)[batch_idx * batch_size:batch_idx * batch_size + len(index)].to(torch.float32)
        batch_idx += 1
    return pred
    # print(len(pred_binary_index))
    # print(pred_binary_index.count(1) / 68126)