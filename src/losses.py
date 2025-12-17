import torch, torch.nn as nn, torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# Multi-Similarity Loss 
class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, base=0.5):
        super().__init__()
        self.alpha = alpha  # Hard negative'ler icin itme gucu
        self.beta = beta    # Hard positive'ler icin cekme gucu
        self.base = base    # Sınır 

    def forward(self, z, y):
        # z: [B, D] embedding'ler (normalize edilmiş)
        # y: [B] etiketler
        
        sim_mat = F.linear(z, z) # Cosine benzerliği 
        B = y.size(0)
        
        # Pozitif ve Negatif maskeleri oluştur
        pos_mask = y.unsqueeze(1) == y.unsqueeze(0)
        pos_mask.fill_diagonal_(False)
        
        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(False)

        #Hard ornekler
    
        neg_loss_mat = torch.exp(self.beta * (sim_mat - self.base)) * neg_mask
        neg_loss = (1.0 / self.beta) * torch.log(1 + neg_loss_mat.sum(dim=1))

        # Pozitif ciftler
        pos_loss_mat = torch.exp(-self.alpha * (sim_mat - self.base)) * pos_mask
        pos_loss = (1.0 / self.alpha) * torch.log(1 + pos_loss_mat.sum(dim=1))
        
        loss = F.softplus(pos_loss + neg_loss) # numerically stable
        
        # Etiketi olmayan (veya tek olan) örnekleri ignore et
        valid_idx = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
        if valid_idx.sum() == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)
            
        return loss[valid_idx].mean()
