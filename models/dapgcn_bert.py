'''
Description: 
version: 
Author: chenhao
Date: 2021-06-09 14:17:37
'''
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import csv
import os

global_tokenizer = None


def _unpack_bert_outputs(outputs):
    if isinstance(outputs, tuple):
        sequence_output = outputs[0]
        pooled_output = outputs[1] if len(outputs) > 1 else None
    elif hasattr(outputs, "last_hidden_state"):
        sequence_output = outputs.last_hidden_state
        pooled_output = getattr(outputs, "pooler_output", None)
    elif isinstance(outputs, dict):
        sequence_output = outputs["last_hidden_state"]
        pooled_output = outputs.get("pooler_output")
    else:
        raise TypeError("Unsupported BERT output type: {}".format(type(outputs)))
    return sequence_output, pooled_output

def hotPicture_all_tokens(inputs, adj_ag, adj, feature_dir, target_sentence_id=None):
    """
    处理所有token的语义/句法矩阵，支持按sentence_id筛选
    
    :param target_sentence_id: 指定需要保存的句子ID（None表示保存所有）
    """
    text_indices = inputs[0] 
    attention_mask = inputs[2]
    sentence_ids = inputs[-1]
    batch_size = adj_ag.size(0)
    
    for i in range(batch_size):
        curr_sid = sentence_ids[i].item()
        
        if target_sentence_id is not None and curr_sid != target_sentence_id:
            continue

        text_np = text_indices[i].cpu().numpy()
        mask_np = attention_mask[i].cpu().numpy()
        ag_np = adj_ag[i].cpu().detach().numpy()
        dep_np = adj[i].cpu().detach().numpy()


        valid_indices = np.where(mask_np != 0)[0]
        valid_length = len(valid_indices)


        semantic_full = ag_np[:valid_length, :valid_length] 
        syntactic_full = dep_np[:valid_length, :valid_length] 
        

        full_tokens = global_tokenizer.convert_ids_to_tokens(text_np[valid_indices])
        full_tokens = full_tokens[:valid_length]  # 确保长度对齐
        

        np.savez_compressed(
            os.path.join(feature_dir, f"all_tokens_{curr_sid}.npz"),
            semantic=semantic_full,
            syntactic=syntactic_full,
            tokens=full_tokens 
        )

def hotPicture_aspect(inputs, adj_ag, adj, feature_dir):
    text_indices = inputs[0] 
    attention_mask = inputs[2] 
    sentence_ids = inputs[-1]
    asp_start = inputs[3]
    asp_end = inputs[4]
    batch_size = adj_ag.size(0)
    
    for i in range(batch_size):
        curr_sid = sentence_ids[i].item()  # 获取唯一句子ID
        
        # 转换numpy数据
        text_np = text_indices[i].cpu().numpy()
        mask_np = attention_mask[i].cpu().numpy()
        ag_np = adj_ag[i].cpu().detach().numpy()
        dep_np = adj[i].cpu().detach().numpy()
        
        
        valid_indices = np.where(mask_np != 0)[0]  
        adjusted_start = asp_start[i].item() + 1  
        adjusted_end = asp_end[i].item() + 1     

        
        assert adjusted_end <= mask_np.sum(), f"Aspect超出有效范围: sid={curr_sid}"

        
        aspect_rows = slice(adjusted_start, adjusted_end)
        
        
        ag_matrix = adj_ag[i].cpu().numpy()
        semantic_sub = ag_matrix[aspect_rows, :len(valid_indices)]  
        
       
        dep_matrix = adj[i].cpu().numpy()
        syntactic_sub = dep_matrix[aspect_rows, :len(valid_indices)]
        
        
        full_tokens = global_tokenizer.convert_ids_to_tokens(text_np[valid_indices])
        full_tokens = full_tokens[:len(valid_indices)]
        aspect_tokens = full_tokens[adjusted_start:adjusted_end]
        
        
        np.savez_compressed(
            os.path.join(feature_dir, f"aspect_vertical_{curr_sid}.npz"),
            semantic=semantic_sub,
            syntactic=syntactic_sub,
            aspect_tokens=aspect_tokens,  # 纵轴标签
            full_tokens=full_tokens       # 横轴标签
        )




class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6): 
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)) 
        self.b_2 = nn.Parameter(torch.zeros(features)) 
        self.eps = eps 

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) 
        std = x.std(-1, keepdim=True) 
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 # 对输入张量进行标准化,将每个元素减去该维度的均值，然后除以标准差.self.eps 是一个防止除零的常数.
        

class DAPGCNBertClassifier(nn.Module):
    def __init__(self, bert, opt, tokenizer):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(opt.bert_dim*2, opt.polarities_dim)
        global global_tokenizer
        global_tokenizer = tokenizer  # 保存tokenizer
        

    def forward(self, inputs):
        outputs1, outputs2, adj_ag, adj_dep, pooled_output = self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, outputs2, pooled_output), dim=-1)
        logits = self.classifier(final_outputs)

        adj_ag_T = adj_ag.transpose(1, 2)
        identity = torch.eye(adj_ag.size(1)).cuda()
        identity = identity.unsqueeze(0).expand(adj_ag.size(0), adj_ag.size(1), adj_ag.size(1))
        ortho = adj_ag@adj_ag_T

        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))
            ortho[i] += torch.eye(ortho[i].size(0)).cuda()

        penal = None
        if self.opt.losstype == 'doubleloss':
            penal1 = (torch.norm(ortho - identity) / adj_ag.size(0)).cuda()
            penal2 = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).cuda()
            penal = self.opt.alpha * penal1 + self.opt.beta * penal2
        
        elif self.opt.losstype == 'orthogonalloss':
            penal = (torch.norm(ortho - identity) / adj_ag.size(0)).cuda()
            penal = self.opt.alpha * penal

        elif self.opt.losstype == 'differentiatedloss':
            penal = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).cuda()
            penal = self.opt.beta * penal
   
        
        return logits, penal




# 新增门控模块
class AspectGate(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, mask1, mask2):
        # mask1/mask2 shape: (batch_size, seq_len)
        combined = torch.stack([mask1.float(), mask2.float()], dim=-1)  # (batch_size, seq_len, 2)
        gate_weights = self.gate(combined)  # (batch_size, seq_len, 1)
        return gate_weights.squeeze(-1) * mask1 + (1 - gate_weights.squeeze(-1)) * mask2







class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)
        
        
        
        
        self.fusion_type = 'attention'
        print("dep_fusion:",self.fusion_type)
        # 添加融合策略相关参数
        if self.fusion_type == 'weighted_sum':
            self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习权重
            self.beta = nn.Parameter(torch.tensor(0.5))
        elif self.fusion_type == 'gate':
            self.gate_fc = nn.Linear(2, 1)  # 门控层
        elif self.fusion_type == 'conv':
            self.conv1d = nn.Conv2d(2, 1, kernel_size=1)  # 1x1卷积融合
        elif self.fusion_type == 'attention':
            self.attn_fc = nn.Sequential(
                nn.Linear(2, 4),
                nn.ReLU(),
                nn.Linear(4, 1)
            )
    
        
        
        
        # 添加门控层
        self.aspect_gate = AspectGate(input_dim=2)  # 输入两个mask的特征
        
        
        
    def _fuse_adjacency(self, adj_dep, adj_asp):
        if self.fusion_type == 'attention':
            # 注意力融合
            stacked = torch.stack([adj_dep, adj_asp], dim=-1)  # [B,L,L,2]
            attn_scores = self.attn_fc(stacked).squeeze(-1)    # [B,L,L]
            weights = F.softmax(attn_scores, dim=-1)
            return F.softmax(weights * adj_dep + (1 - weights) * adj_asp, dim=-1)
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")   
        
        
        
        

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, D_min, M_pos, src_mask, aspect_mask , aspect_mask2,com_aspect_mask, mask_position,_= inputs # 
        # 之前的所有bert数据都是adj_dep还未加上D_min就送入gcn了，所以修改为放在D_min后面
        # h1, h2, adj_ag, pooled_output = self.gcn(adj_dep, inputs)
        


        # 使用门控机制融合两个aspect mask
        gate_output = self.aspect_gate(aspect_mask, aspect_mask2)
        com_aspect_mask = gate_output  # 替换原有的com_aspect_mask




        #A_asp
        # adj_asp = 1/(D_min + 1)
        # adj_asp[adj_asp == 1] = 0
        # adj_dep = adj_asp
        # adj_dep = F.softmax(adj_dep + adj_asp) 
        # 融合依赖矩阵
        adj_asp = torch.exp(-1 * D_min)
        adj_asp[D_min == 0] = 0
        

        
        adj_dep = self._fuse_adjacency(adj_dep, adj_asp)  # 使用选定策略


        
        
        
        # 消融临时用的，记得删
        # com_aspect_mask = aspect_mask
        
        
        h1, h2, adj_ag, pooled_output = self.gcn(adj_dep, inputs, com_aspect_mask, adj_asp)



        # # avg pooling asp feature
        # asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        # aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2) 
        # outputs1 = (h1*aspect_mask).sum(dim=1) / asp_wn
        # outputs2 = (h2*aspect_mask).sum(dim=1) / asp_wn
        # return outputs1, outputs2, adj_ag, adj_dep, pooled_output
        
        
        # # avg pooling asp feature
        # # 修改池化计算
        # mask_wn = aspect_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        # mask_mask_expanded = aspect_mask.unsqueeze(-1).expand_as(h1)
        
        # outputs1 = (h1 * mask_mask_expanded).sum(dim=1) / mask_wn
        # outputs2 = (h2 * mask_mask_expanded).sum(dim=1) / mask_wn
        # return outputs1, outputs2, adj_ag, adj_dep, pooled_output
        
        
        
        # avg pooling asp feature
        # 修改池化计算
        mask_wn = com_aspect_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        mask_mask_expanded = com_aspect_mask.unsqueeze(-1).expand_as(h1)
        
        outputs1 = (h1 * mask_mask_expanded).sum(dim=1) / mask_wn
        outputs2 = (h2 * mask_mask_expanded).sum(dim=1) / mask_wn
        return outputs1, outputs2, adj_ag, adj_dep, pooled_output


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        # opt.bert_dim 表示 BERT 模型的隐藏状态维度（或者说是嵌入维度）。
        # 在标准的 BERT 模型中，bert_dim 通常为 768（对于 bert-base-uncased）或 1024（对于 bert-large-uncased）。
        # 这表示每个 token 在经过 BERT 模型处理后的特征向量的大小。
        # 将 记忆维度 (mem_dim) 设置为 BERT 模型维度的一半,可能用于内存网络或其他需要通过缩小特征维度来提高效率和减少计算负载的场景。
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)
        
        # self.feature_dir = "posDmin_attention_features"
        # self.feature_dir = "selfDep_attention_features"
        # self.feature_dir = "s_fusion_attention_features_noPos"
        # self.feature_dir = "posDmin_attention_features_allToken"
        self.feature_dir = "selfDep_attention_features_allToken"
        self.secondary_dir = os.path.join(self.feature_dir, self.opt.dataset)
        os.makedirs(self.secondary_dir, exist_ok=True)
        

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)
        
        
        
        
        # 更新日期2025、1、9
        # 初始化方面词的类
        # self.a = AspectAttention(self.bert_dim, self.bert_dim)





        # 初始化pos多头注意力机制
        self.pos_attn = PosMultiHeadAttention(opt.attention_heads, self.bert_dim) # 返回的是注意力结果   
        # 初始化 α 为一个可学习的参数
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可以根据需要初始化为其他值




        # 新增融合相关参数
        self.fusion_type = 'gate' # 目前最好的是pos:gate, dep:attention 
        print("pos_fusion:",self.fusion_type)
        if self.fusion_type == 'attention':
            self.att_fc = nn.Linear(opt.bert_dim * 2, 1)  # 注意力权重生成
        elif self.fusion_type == 'gate':
            self.gate_weight = nn.Parameter(torch.Tensor(1))
            self.gate_bias = nn.Parameter(torch.Tensor(1))
        elif self.fusion_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(2, 4),
                nn.ReLU(),
                nn.Linear(4, 1)
            )
        elif self.fusion_type == 'similarity':
            self.sim_scaler = nn.Parameter(torch.Tensor([1.0]))




        
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        
        
    def _fuse_adjacency(self, adj_ag, adj_pos, features=None):
        if self.fusion_type == 'gate':
            # 门控机制 [B, L, L]
            gate = torch.sigmoid(self.gate_weight * adj_ag + self.gate_bias)
            return gate * adj_ag + (1 - gate) * adj_pos
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        
 
    def forward(self, adj, inputs, com_aspect_mask, adj_asp):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, D_min, MM_pos, src_mask, aspect_mask,aspect_mask2,com_aspect_mask_pre, mask_position,_ = inputs # 
        src_mask = src_mask.unsqueeze(-2)
        
        bert_outputs = self.bert(
            text_bert_indices,
            attention_mask=attention_mask,
            token_type_ids=bert_segments_ids,
        )
        sequence_output, pooled_output = _unpack_bert_outputs(bert_outputs)
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        denom_dep = adj.sum(2).unsqueeze(2) + 1
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        multi_head_list = []
        outputs_dep = None
        adj_ag = None
        
        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
        adj_ag = src_mask.transpose(1, 2) * adj_ag

        adj_ag_pre = adj_ag








        K = gcn_inputs
        # aspect_mask = aspect_mask.unsqueeze(-1)
        # aspect_mask2 = aspect_mask2.unsqueeze(-1).expand_as(gcn_inputs)
        com_aspect_mask = com_aspect_mask.unsqueeze(-1).expand_as(gcn_inputs)
        Q_asp = gcn_inputs * com_aspect_mask # 仅保留方面词的特征，其余设为 0
        pos_maxlen = com_aspect_mask.shape[1]
        
        # pos的掩码矩阵
        M_pos = (torch.zeros_like(MM_pos)!=MM_pos).float().unsqueeze(-1)[:, :pos_maxlen]
        
        # 语义邻接矩阵的实现
        pos_attn_tensor = self.pos_attn(Q_asp, K, M_pos) # 用多头注意力机制从 gcn_inputs 生成注意力张量,张量 attn_tensor，形状为 (batch_size, num_heads, seq_len, seq_len)，这是一个典型的自注意力（Self-Attention）权重张量。# self.attn 是一个注意力机制函数，将 gcn_inputs 作为查询（query）、键（key）、和值（value）输入，同时通过 src_mask 指示有效的位置。
        pos_attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(pos_attn_tensor, 1, dim=1)] # 将生成的 attn_tensor 按维度 1 切割，得到多个注意力矩阵，存储在 attn_adj_list 中。# squeeze(1) 将多余的维度去掉，以便后续处理。这样可以得到多个独立的头部注意力矩阵。
        adj_pos = None # 初始化语义邻接矩阵，用于后续从多个注意力头中生成最终的邻接矩阵

        # * Average Multi-head Attention matrixes
        # 将多个注意力头的矩阵求平均，生成语义邻接矩阵 adj_ag
        for i in range(self.attention_heads):
            if adj_pos is None:
                adj_pos = pos_attn_adj_list[i]
            else:
                adj_pos += pos_attn_adj_list[i]
        adj_pos =adj_pos / self.attention_heads
        # adj_pos = pos_attn_adj_list
        # 调整pos邻接矩阵
        for j in range(adj_pos.size(0)):
            adj_pos[j] -= torch.diag(torch.diag(adj_pos[j])) # adj_ag[j] -= torch.diag(torch.diag(adj_ag[j])) 将矩阵的对角线元素设置为 0，避免自环（self-loop）
            adj_pos[j] += torch.eye(adj_pos[j].size(0)).cuda() # adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda() 添加单位矩阵，确保每个节点至少与自身连接
        adj_pos = M_pos * adj_pos # 确保语义邻接矩阵只在有效的词语位置进行操作
        adj_ag = self._fuse_adjacency(adj_ag, adj_pos, features=gcn_inputs)


        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs_ag = gcn_inputs
        outputs_dep = gcn_inputs

        for l in range(self.layers):
            # ************SynGCN*************
            Ax_dep = adj.bmm(outputs_dep)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)

            # ************SemGCN*************
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.weight_list[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            # * mutual Biaffine module
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)
            gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A2, gAxW_dep)
            outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag

        return outputs_ag, outputs_dep, adj_ag, pooled_output


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn
    



def pos_attention(query, key, M_pos, bias, dropout=None): 
    d_k = query.size(-1) 
    
    M_pos = M_pos.unsqueeze(1)
    M_pos_expanded = M_pos.expand(-1, -1, -1, key.size(-1)) 
    key = torch.matmul(key, M_pos_expanded.transpose(-1,-2))

    
    linear_layer = nn.Linear(key.size(-1),query.size(-1)).cuda()
    batch_size, _, seq_len, in_features = key.shape
    key = key.view(batch_size * seq_len, in_features).cuda()
    key = linear_layer(key)
    key = key.view(batch_size, 1, seq_len, -1)
    
    
    scores = torch.matmul(query, key.transpose(-1,-2)) + bias.unsqueeze(0)
    scores = torch.tanh(scores)

    
    if M_pos is not None:
        scores = scores.masked_fill(M_pos == 0, -1e9) 


    p_attn = F.softmax(scores, dim=-1) 
    if dropout is not None: 
        p_attn = dropout(p_attn)

    return p_attn # 返回计算得到的注意力权重



class PosMultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1): 
        super(PosMultiHeadAttention, self).__init__()
        assert d_model % h == 0 
        self.d_k = d_model // h 
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2) 
        self.dropout = nn.Dropout(p=dropout) 
        self.bias = nn.Parameter(torch.Tensor(h, 1, 1))

    def forward(self, query, key, M_pos):

        nbatches = query.size(0) # 获取批次大小
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]     

        attn = pos_attention(query, key, M_pos, self.bias, dropout=self.dropout) 
        return attn # 返回计算得到的注意力结果。






# 更新日期2025、1、9
# 关注方面词的类
class AspectAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AspectAttention, self).__init__()
        # 初始化可训练参数
        self.W_a = nn.Parameter(torch.randn(input_dim, attention_dim))
        self.W_k = nn.Parameter(torch.randn(input_dim, attention_dim))
        self.b = nn.Parameter(torch.zeros(100))

    def forward(self, H_a, K):
        """
        Args:
            H_a: Aspect representation, shape (batch_size, seq_len, input_dim)
            K: Context representation from BERT, shape (batch_size, seq_len, input_dim)
        Returns:
            A_a: Attention weights, shape (batch_size, seq_len, attention_dim)
        """
        
        # 计算 Ha * W_a
        H_a = H_a.float()
        Ha_Wa = torch.matmul(H_a, self.W_a)  # shape: (batch_size, seq_len, attention_dim)
        
        # 计算 K * W_k^T
        K_WkT = torch.matmul(K, self.W_k)  # shape: (batch_size, seq_len, attention_dim)
        
        # 计算 tanh(Ha_Wa + K_WkT + b)
        # print("Ha_Wa shape:",Ha_Wa.shape)
        # print("K_WkT shape:",K_WkT.shape)
        attention_scores = torch.matmul(Ha_Wa, K_WkT.transpose(1, 2)) + self.b
        A_a = torch.tanh(attention_scores)  # shape: (batch_size, seq_len, attention_dim)
        
        return A_a

