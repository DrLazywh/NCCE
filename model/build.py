from model import objectives

from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights, Transformer, LayerNorm
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.cuda.amp import autocast
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

# _tokenizer = _Tokenizer()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=11003):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class GeneralizedCrossEntropy(nn.Module):
    """Computes the generalized cross-entropy loss, from `
    "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
    <https://arxiv.org/abs/1805.07836>`_
    Args:
        q: Box-Cox transformation parameter, :math:`\in (0,1]`
    Shape:
        - Input: the raw, unnormalized score for each class.
                tensor of size :math:`(minibatch, C)`, with C the number of classes
        - Target: the labels, tensor of size :math:`(minibatch)`, where each value
                is :math:`0 \leq targets[i] \leq C-1`
        - Output: scalar
    """

    def __init__(self, q: float = 0.7) -> None:
        super().__init__()
        self.q = q
        self.epsilon = 1e-6
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = self.softmax(input)
        p = p[torch.arange(p.shape[0]), target]
        # Avoid undefined gradient for p == 0 by adding epsilon
        p += self.epsilon
        loss = (1 - p ** self.q) / self.q
        return torch.mean(loss)

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs) 
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1) 
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

def forward_cc(x, y, index=None):
        sm_outputs = F.softmax(x, dim=1)
        final_outputs = sm_outputs * y
        loss = - torch.log(final_outputs.sum(dim=1))
        return loss

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def mixup(inputs, targets, alpha):
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)
    idx = torch.randperm(inputs.size(0))
    input_a, input_b = inputs, inputs[idx]
    target_a, target_b = targets, targets[idx]
    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b
    return mixed_input, mixed_target

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
 
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)
        
        self.bge_classifier = nn.Linear(512, self.num_classes)
        nn.init.normal_(self.bge_classifier.weight.data, std=0.001)
        nn.init.constant_(self.bge_classifier.bias.data, val=0.0)
        
        self.tse_classifier = nn.Linear(1024, self.num_classes)
        nn.init.normal_(self.tse_classifier.weight.data, std=0.001)
        nn.init.constant_(self.tse_classifier.bias.data, val=0.0)
        
 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
        gceq = args.gceq
        self.GCE_loss = GeneralizedCrossEntropy(q = gceq)
        self.SCE_loss = SCELoss(2, 0.01)

        if self.args.cross_id:
            self.classifier_proj = nn.Linear(self.embed_dim, self.num_classes, bias=False)
            self.classifier_proj.apply(weights_init_classifier)
            self.bottleneck_proj = nn.BatchNorm1d(self.embed_dim)
            self.bottleneck_proj.bias.requires_grad_(False)
            self.bottleneck_proj.apply(weights_init_kaiming)

            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                    layers=args.cmt_depth,
                                                    heads=self.embed_dim //
                                                            64)
            scale = self.cross_modal_transformer.width ** -0.5
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)
            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)
        
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x[0].unsqueeze(0)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def compute_per_loss(self, batch):
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        lossA, simsA = objectives.compute_per_loss(i_feats, t_feats, batch['pids'], \
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        lossB, simsB = objectives.compute_per_loss(i_tse_f, t_tse_f, batch['pids'],\
                                                    tau=self.args.tau, \
                                                    margin=self.args.margin, \
                                                    loss_type=self.loss_type, \
                                                    logit_scale=self.logit_scale)
        
        return lossA.detach().cpu(), lossB.detach().cpu(), simsA, simsB
    
    
    def forward(self, batch, epoch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        pid = batch['pids']
        # print(images.shape)
        
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        i_tse_f = self.visul_emb_layer(image_feats, atten_i)
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            
        label_hat = batch['label_hat'].to(i_feats.device) 
        filtered_image = images[label_hat == 1]
        
        loss1, loss2 = objectives.compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, batch['pids'], \
                                              label_hat=label_hat, margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        ret.update({'bge_loss':loss1})
        ret.update({'tse_loss':loss2})
        
        
        criterion = nn.CrossEntropyLoss(reduction="mean")
        # criterion = CrossEntropyLabelSmooth(num_classes=11003)
        bge_image_logits = self.bge_classifier(i_feats.half()).float()
        bge_id_loss = criterion(bge_image_logits, pid) 
        
        tse_image_logits = self.tse_classifier(i_tse_f.half()).float()
        tse_id_loss = criterion(tse_image_logits, pid) 
        bge_image_pred = torch.argmax(bge_image_logits, dim=1)
        bge_image_precision = (bge_image_pred == batch['pids']).float().mean()
        tse_image_pred = torch.argmax(tse_image_logits, dim=1)
        tse_image_precision = (tse_image_pred == batch['pids']).float().mean()

        ret.update({'bge_img_acc': bge_image_precision})
        ret.update({'tse_img_acc': tse_image_precision})
        id_loss = (bge_id_loss + tse_id_loss) / 2
        
        ret.update({'id_loss': id_loss})
        bge_txt_logits = self.bge_classifier(t_feats.half()).float()
        tse_txt_logits = self.tse_classifier(t_tse_f.half()).float()
        
        if (epoch < 21) or (batch["is_relabel"] == None) or (batch["PL"] == None): 
            # bge_txt_logits = self.bge_classifier(t_feats.half()).float()
            # tse_txt_logits = self.tse_classifier(t_tse_f.half()).float()
            bge_txt_id_loss = self.GCE_loss(bge_txt_logits, pid)
            tse_txt_id_loss = self.GCE_loss(tse_txt_logits, pid)
        else:
            re_bge_txt_logits = bge_txt_logits[batch["is_relabel"] == True]
            re_tse_txt_logits = tse_txt_logits[batch["is_relabel"] == True]
            PL = batch["PL"]
            re_bge_loss = forward_cc(x=re_bge_txt_logits, y=PL).mean()
            re_tse_loss = forward_cc(x=re_tse_txt_logits, y=PL).mean()
            # re_loss = (re_bge_loss + re_tse_loss) / 2
            
            la_bge_txt_logits = bge_txt_logits[batch["is_relabel"] == False]
            la_tse_txt_logits = tse_txt_logits[batch["is_relabel"] == False]
            
            txt_pid = pid[batch["is_relabel"] == False]
            
            la_bge_loss = self.GCE_loss(la_bge_txt_logits, txt_pid)
            la_tse_loss = self.GCE_loss(la_tse_txt_logits, txt_pid)
            
            bge_txt_id_loss = re_bge_loss + la_bge_loss
            tse_txt_id_loss = re_tse_loss + la_tse_loss

        txt_id_loss = (bge_txt_id_loss + tse_txt_id_loss) / 2
        # print(txt_id_loss)
        bge_txt_pred = torch.argmax(bge_txt_logits, dim=1)
        bge_txt_precision = (bge_txt_pred == batch['cap_r_pid']).float().mean()
        tse_txt_pred = torch.argmax(tse_txt_logits, dim=1)
        tse_txt_precision = (tse_txt_pred == batch['cap_r_pid']).float().mean()
        ret.update({'bge_txt_acc': bge_txt_precision})
        ret.update({'tse_txt_acc': tse_txt_precision})
        ret.update({'txt_id_loss': txt_id_loss})


        if self.args.cross_id:
            with autocast():
                cross_x = self.cross_former(t_feats.unsqueeze(1), image_feats, image_feats)
                cross_x_bn = self.bottleneck_proj(cross_x.squeeze(1))
                cls_score = self.classifier_proj(cross_x_bn.half()).float()
                cross_id_loss = objectives.compute_id(cls_score, batch['pids']) 
                ret.update({'crossid_loss': cross_id_loss * self.args.weight_id})
        return ret


def build_model(args, num_classes=3701):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
