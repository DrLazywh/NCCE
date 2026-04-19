from model import objectives

from model.CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from model.clip_model import build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train, do_inference
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

from torch.nn.functional import normalize

from time import sleep

def get_feat(model, batch, device):
    images = batch['images'].to(device)
    caption_ids = batch['caption_ids'].to(device)
    image_feats, atten_i, text_feats, atten_t = model.base_model(images, caption_ids)
    i_feats = image_feats[:, 0, :].float()
    # i_feats = image_feats.float() # for CLIP ResNet visual model
    t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

    i_tse_f = model.visul_emb_layer(image_feats, atten_i)
    t_tse_f = model.texual_emb_layer(text_feats, caption_ids, atten_t)
    print("*" * 100)
    print(type(batch))
    print(len(batch))
    print("i_bge_f.shape:" + str(i_feats.shape))
    print("t_bge_f.shape:" + str(t_feats.shape))
    print("i_tse_f.shape:" + str(i_tse_f.shape))
    print("t_tse_f.shape:" + str(t_tse_f.shape))
    print("*" * 100)


def KNN(net, eval_loader, batch_size, feat_dim=512, num_neighbor=20):
    net.eval()

    # loading given samples
    # 初始化两个tensor分别用来装所有的文本特征（trainFeatures）和所有的图片特征（imageFeatures）    
    trainFeatures = torch.rand(len(eval_loader.dataset), feat_dim).t().cuda()
    imageFeatures = torch.rand(len(eval_loader.dataset), feat_dim).t().cuda()
    for batch_idx, batch in enumerate(eval_loader):
        images = batch['images'].to(device)
        caption_ids = batch['caption_ids'].to(device)
        image_ids = batch['image_ids'].to(device)
        batchSize = len(caption_ids)
        image_feats, atten_i, text_feats, atten_t = net.base_model(images, caption_ids)

        # 获取所有的图片特征，以供后面"CLIP式"的推理使用
        image_feats = image_feats[:, 0, :].float()
        imageFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = image_feats.data.t()

        # 获取所有的文本特征，使用文本特征作为candidate sample，对应的图片特征作为标签        
        text_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = text_feats.data.t()

    # 特征归一化    
    imageFeatures = normalize(imageFeatures.t())
    trainFeatures = normalize(trainFeatures.t())

    # caculating neighborhood-based label inconsistency score
    # num_batch = 579
    num_batch = 1065
    sver_collection = []
    pred_total = []
    pred_binary_index = []
    for batch_idx in range(num_batch):
        text_feats = trainFeatures[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        dist = torch.mm(text_feats, trainFeatures.t())
        # dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1  
        _, neighbors = dist.topk(num_neighbor, dim=1, largest=True, sorted=True)  # find contrastive neighbors
        # print(neighbors.shape) (64, 20)

        # neighbors中的每一个item为一个样本的所有邻居的index
        for item in neighbors:
            # print("item:" + str(item.shape)) (20)
            # 通过索引获取一个样本的所有邻居的特征（文本特征），shape应为(num_neighbors, feat_dim)
            neighbors_feats = trainFeatures[item]
            # print("neighbors_feats:" + str(neighbors_feats.shape)) (20, 512)

            # 使用现有模型对邻居的特征进行"CLIP式"的推理
            # 计算邻居特征（文本特征）与所有图片特征的相似度
            sims = torch.mm(neighbors_feats, imageFeatures.t())

            # 对于每一个文本特征选取与之相似度最高的图片特征作为其标签
            # pred_index为这20个邻居的标签的索引
            _, pred_index = sims.topk(1, dim=1, largest=True, sorted=True)
            # print("pred_index:" + str(pred_index.shape)) (20, 1)

            # 将索引展平
            pred_index_flattened = pred_index.view(-1)
            # 通过索引获得对应的标签（图片特征）
            pred = imageFeatures[pred_index_flattened]
            # print("pred:" + str(pred.shape)) (20, 512)

            pred_total.append(pred)

    index = 0
    for neighbors_pred_image_feat in pred_total:
        scores = imageFeatures[index] @ neighbors_pred_image_feat.t()
        sver = sum(scores)
        sver_collection.append(sver)
        index += 1
    # print(sver_collection)
    # print(len(sver_collection))

    for sver in sver_collection:
        if sver.item() > 14:
            pred_binary_index.append(1)
        else:
            pred_binary_index.append(0)

    # print(len(pred_binary_index))        
    print(pred_binary_index.count(1) / 68126)

    
    
def get_feature(net, eval_loader, batch_size, feat_dim=512, num_neighbor=20):
    net.eval()
    # 初始化两个tensor分别用来装所有的文本特征（trainFeatures）和所有的图片特征（imageFeatures）    
    trainFeatures = torch.rand(len(eval_loader.dataset), feat_dim).t().cuda()
    imageFeatures = torch.rand(len(eval_loader.dataset), feat_dim).t().cuda()
    
    for batch_idx, batch in enumerate(eval_loader):
        images = batch['images'].to(device)
        caption_ids = batch['caption_ids'].to(device)
        image_ids = batch['image_ids'].to(device)
        batchSize = len(caption_ids)
        image_feats, atten_i, text_feats, atten_t = net.base_model(images, caption_ids)
        
        # 获取所有的图片特征，以供后面"CLIP式"的推理使用
        image_feats = image_feats[:, 0, :].float()  
        
        # 获取所有的文本特征，使用文本特征作为candidate sample，对应的图片特征作为标签        
        text_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        if batchSize == batch_size:
            imageFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = image_feats.data.t()     
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = text_feats.data.t()
        else:
            imageFeatures[:, batch_idx * batch_size:batch_idx * batch_size + batchSize] = image_feats.data.t()
            trainFeatures[:, batch_idx * batch_size:batch_idx * batch_size + batchSize] = text_feats.data.t()
            
    imageFeatures = normalize(imageFeatures.t())
    trainFeatures = normalize(trainFeatures.t())
    return imageFeatures, trainFeatures
    

def test(net, eval_loader, batch_size, feat_dim=512, num_neighbor=20):
    net.eval()
    # loading given samples
    # 初始化两个tensor分别用来装所有的文本特征（trainFeatures）和所有的图片特征（imageFeatures）    
    trainFeatures1 = torch.rand(len(eval_loader.dataset), feat_dim).t().cuda()
    imageFeatures1 = torch.rand(len(eval_loader.dataset), feat_dim).t().cuda()

    # trainFeatures2 = torch.rand(len(eval_loader.dataset), feat_dim).t().cuda()
    # imageFeatures2 = torch.rand(len(eval_loader.dataset), feat_dim).t().cuda()
    
    for batch_idx, batch in enumerate(eval_loader):
        # print(batch_idx)
        images = batch['images'].to(device)
        caption_ids = batch['caption_ids'].to(device)
        image_ids = batch['image_ids'].to(device)
        batchSize = len(caption_ids)
        image_feats1, atten_i, text_feats1, atten_t = net.base_model(images, caption_ids)
        # image_feats2, atten_i, text_feats2, atten_t = net.base_model(images, caption_ids)
        
        # 获取所有的图片特征，以供后面"CLIP式"的推理使用
        image_feats1 = image_feats1[:, 0, :].float()
        # image_feats2 = image_feats2[:, 0, :].float()
        
        
        # 获取所有的文本特征，使用文本特征作为candidate sample，对应的图片特征作为标签        
        text_feats1 = text_feats1[torch.arange(text_feats1.shape[0]), caption_ids.argmax(dim=-1)].float()
        # text_feats2 = text_feats2[torch.arange(text_feats2.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        if batchSize == batch_size:
            imageFeatures1[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = image_feats1.data.t()
            # imageFeatures2[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = image_feats2.data.t()
            
            trainFeatures1[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = text_feats1.data.t()
            # trainFeatures2[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = text_feats2.data.t()
        else:
            imageFeatures1[:, batch_idx * batch_size:batch_idx * batch_size + batchSize] = image_feats1.data.t()
            # imageFeatures2[:, batch_idx * batch_size:batch_idx * batch_size + batchSize] = image_feats2.data.t()
            trainFeatures1[:, batch_idx * batch_size:batch_idx * batch_size + batchSize] = text_feats1.data.t()
            # trainFeatures2[:, batch_idx * batch_size:batch_idx * batch_size + batchSize] = text_feats2.data.t()

    # print(trainFeatures1 == trainFeatures2)
    # print((trainFeatures1 == trainFeatures2).sum())
    # print(imageFeatures1 == imageFeatures2)
    # print((imageFeatures1 == imageFeatures2).sum())
    # neighbors_list = count_sver_scores(imageFeatures1, trainFeatures1, batch_size, num_neighbor)
    # count_sver_scores(imageFeatures2, trainFeatures2, batch_size, num_neighbor)
    return imageFeatures1, trainFeatures1
    
        
        
        # print(text_feats1 == text_feats2)
        # print((text_feats1 == text_feats2).sum())
        # if (text_feats1 == text_feats2).sum() != batchSize * feat_dim:
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     break
#         print(image_feats1 == image_feats2)
#         print((image_feats1 == image_feats2).sum())
        
#     print(trainFeatures1 == trainFeatures2)
#     print((trainFeatures1 == trainFeatures2).sum())
#     print(imageFeatures1 == imageFeatures2)
#     print((imageFeatures1 == imageFeatures2).sum())
        
def count_sver_scores(imageFeatures, trainFeatures, batch_size, num_neighbor):
    # 特征归一化    
    imageFeatures = normalize(imageFeatures.t())
    trainFeatures = normalize(trainFeatures.t())

    # caculating neighborhood-based label inconsistency score
    # num_batch = 579
    num_batch = 1065
    sver_collection = []
    pred_total = []
    pred_binary_index = []
    neighbors_list = []
    for batch_idx in range(num_batch):
        text_feats = trainFeatures[batch_idx * batch_size:batch_idx * batch_size + batch_size]
        dist = torch.mm(text_feats, trainFeatures.t())
        
        # dist[torch.arange(dist.size()[0]), torch.arange(dist.size()[0])] = -1  
        _, neighbors = dist.topk(num_neighbor, dim=1, largest=True, sorted=True)  # find contrastive neighbors
        # print(neighbors.shape) (64, 20)
        neighbors_list.append(neighbors)
        # neighbors中的每一个item为一个样本的所有邻居的index
        for item in neighbors:
            # print("item:" + str(item.shape)) (20)
            # 通过索引获取一个样本的所有邻居的特征（文本特征），shape应为(num_neighbors, feat_dim)
            neighbors_feats = trainFeatures[item]
            # print("neighbors_feats:" + str(neighbors_feats.shape)) (20, 512)

            # 使用现有模型对邻居的特征进行"CLIP式"的推理
            # 计算邻居特征（文本特征）与所有图片特征的相似度
            sims = torch.mm(neighbors_feats, imageFeatures.t())

            # 对于每一个文本特征选取与之相似度最高的图片特征作为其标签
            # pred_index为这20个邻居的标签的索引
            _, pred_index = sims.topk(1, dim=1, largest=True, sorted=True)
            # print("pred_index:" + str(pred_index.shape)) (20, 1)

            # 将索引展平
            pred_index_flattened = pred_index.view(-1)
            # 通过索引获得对应的标签（图片特征）
            pred = imageFeatures[pred_index_flattened]
            # print("pred:" + str(pred.shape)) (20, 512)

            pred_total.append(pred)

#     index = 0
#     for neighbors_pred_image_feat in pred_total:
#         scores = imageFeatures[index] @ neighbors_pred_image_feat.t()
#         sver = sum(scores)
#         sver_collection.append(sver)
#         index += 1
#     # print(sver_collection)
#     # print(len(sver_collection))

#     for sver in sver_collection:
#         if sver.item() > 14:
#             pred_binary_index.append(1)
#         else:
#             pred_binary_index.append(0)

#     # print(len(pred_binary_index))        
#     print(pred_binary_index.count(1) / 68126)
    return neighbors_list


if __name__ == '__main__':
    args = get_args()
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader1, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    # train_loader2, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)    
    model.to(device)
    imageFeatures1, trainFeatures1 = get_feature(model, train_loader1, 64)
    imageFeatures2, trainFeatures2 = get_feature(model, train_loader1, 64)
    print("imageFeatures1:")
    print(imageFeatures1)
    print("imageFeatures2:")
    print(imageFeatures2)
    
    
    
    
    
    print(imageFeatures1 == imageFeatures2)
    print(trainFeatures1 == trainFeatures2)
    print((imageFeatures1 == imageFeatures2).sum())
    print((trainFeatures1 == trainFeatures2).sum())
   



    
    
    
#     imf1, trf1 = test(model.eval(), train_loader1, 64)
#     model.eval()
#     imf2, trf2 = test(model.eval(), train_loader1, 64)
#     print(imf1.shape)
#     print(imf2.shape)
#     print(trf1.shape)
#     print(trf2.shape)
#     print(imf1 == imf2)
#     print(trf1 == trf2)
#     print((imf1 == imf2).sum())
#     print((trf1 == trf2).sum())
    # for i, batch in enumerate(train_loader):
    #     print(i)
    #     get_feat(model, batch, device)
    # KNN(model, train_loader, 64)
    # if len(pred_total1) != len(pred_total2):
    #     print("!!!!!!#######")
    #     sleep(20)
    # for index in range(len(pred_total1)):
    #     print(pred_total1[index] == pred_total2[index])
    #     print((pred_total1[index] == pred_total2[index]).sum())
    #     if (pred_total1[index] == pred_total2[index]).sum() != 20 * 512:
    #         print("!!!!!!!!!!!!")
    #         break