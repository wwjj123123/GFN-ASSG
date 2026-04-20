import torch
import torch.nn.functional as F

class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算 logits 的 log-softmax
        log_probs = F.log_softmax(inputs, dim=-1)  # Shape: (batch_size, num_classes)

        # 获取每个样本的类别标签
        loss_per_class = -log_probs.gather(1, targets.unsqueeze(1))  # Shape: (batch_size, 1)

        # 根据类别权重进行加权，如果 weight 为 None，则没有加权
        if self.weight is not None:
            weight = self.weight[targets]  # 选择对应类别的权重
            loss_per_class = loss_per_class.squeeze(1) * weight  # 加权损失

        # 计算最终的损失
        if self.reduction == 'mean':
            return loss_per_class.mean()  # 求平均损失
        elif self.reduction == 'sum':
            return loss_per_class.sum()  # 求总损失
        else:
            return loss_per_class  # 不进行缩减，返回每个样本的损失
