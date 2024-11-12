import torch
import torch.nn as nn
from einops import repeat, rearrange

def saliency_guided_masking(x, base_mask_ratio, mask_ratio_var, delta):
    # Input shape: (num_patches, batch_size, embed_dim)
    N, B, C = x.shape 
    
    x = rearrange(x, 'n b c -> b n c')
    aff = torch.matmul(x, x.permute(0, 2, 1))
    aff = torch.einsum('nip,njp->nij', x, x)
    aff = nn.functional.softmax(aff, dim=2)
    
    aff_sum = torch.sum(aff, dim=1)

    # Normalize the affinity sums for saliency
    aff_sum_normalized = (aff_sum - aff_sum.min(dim=1, keepdim=True)[0]) / \
                            (aff_sum.max(dim=1, keepdim=True)[0] - aff_sum.min(dim=1, keepdim=True)[0])

    # Compute dynamic mask ratios and keep length
    y = (aff_sum_normalized > delta).sum(dim=1)
    y_max = aff_sum_normalized.size(1)
    y_normalized = y.float().mean() / y_max
    dynamic_mask_ratios = base_mask_ratio - mask_ratio_var + 2 * mask_ratio_var * y_normalized
    len_keep = int(N * (1 - dynamic_mask_ratios))

    print(dynamic_mask_ratios)
    # Add random noise for stochasticity
    noise = torch.rand(B, N , device=x.device)
    saliency_guided_noise = aff_sum_normalized + noise

    # Sorting indices by saliency
    ids_shuffle = torch.argsort(saliency_guided_noise, dim=1, descending=True)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    
    # Mask the patches and rearrange indices
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
    ids_restore = rearrange(ids_restore, 'b t -> t b')
    x_masked = rearrange(x_masked, 't b e -> b t e')
    
    return x_masked,  ids_restore