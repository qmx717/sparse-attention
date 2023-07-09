import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from mycode_res_add_nlsa.sga_attention.common import *

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SparseAxialAttenion(nn.Module):
    def __init__(self, in_planes, out_planes, n_hashes=4,chunk_size=25,groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(SparseAxialAttenion, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.n_hashes = n_hashes
        self.groups = groups
        self.chunk_size = chunk_size
        self.group_planes = out_planes // groups
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.conv_match = qkv_transform(in_planes, out_planes * 3//2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_match = nn.BatchNorm1d(out_planes * 3//2)
        # Priority on encoding
        ## Initial values
        self.f_gw = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_gv1 = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_gv2 = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(kernel_size,self.group_planes * 3//2), requires_grad=True)
        relative_index = torch.arange(kernel_size).unsqueeze(0)
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()
        # self.print_para()
    def LSH(self, hash_buckets, x):
        # x: [N,H*W,C]
        N = x.shape[0]
        device = x.device

        # generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)  # [1,C,n_hashes,hash_buckets//2]
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1,
                                                                                             -1)  # [N, C, n_hashes, hash_buckets//2]

        # locality sensitive hashing
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations)  # [N, n_hashes, H*W, hash_buckets//2]
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)  # [N, n_hashes, H*W, hash_buckets]

        # get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1)  # [N,n_hashes,H*W]

        # add offsets to avoid hash codes overlapping between hash rounds
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,))  # [N,n_hashes*H*W]

        return hash_codes  # 哈希编码

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, :, :,-1:, ...], x[:, :,:, :-1, ...]], dim=3)
        x_extra_forward = torch.cat([x[:, :, :,1:, ...], x[:, :,:, :1, ...]], dim=3)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=4)

    def add_adjacent_buckets_embed(self, x):
        x_extra_back = torch.cat([x[-1:, ...], x[ :-1, ...]], dim=0)
        x_extra_forward = torch.cat([x[ 1:, ...], x[:1, ...]], dim=0)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=1)

    def reset_parameters(self):
        self.conv_match.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)
        # number of hash buckets/hash bits
        hash_buckets = min(H // self.chunk_size + (H// self.chunk_size) % 2, 10)
        # Transform
        qkv = self.bn_match(self.conv_match(x))
        w_match,v_match= torch.split(qkv.reshape(N * W,H,self.out_planes * 3//2),#此处有修改，reshape变成peemute
                              [self.out_planes // 2, self.out_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 0, self.flatten_index).view(self.kernel_size,
                                                                                       self.group_planes * 3//2)
        w_embedding, v_embedding = torch.split(all_embeddings,
                                [self.group_planes // 2, self.group_planes],dim=1)

        # get assigned hash codes/bucket number
        hash_codes = self.LSH(hash_buckets, w_match)  # [N*W,n_hashes*H]
        hash_codes = hash_codes.detach()

        # group elements with same hash code by sorting
        _, indices = hash_codes.sort(dim=-1)  # [N*H,n_hashes*W]
        _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
        mod_indices = (indices % H)  # now range from (0->H)
        w_embed_sorted = batched_index_select(w_match, mod_indices)  # [N*H,n_hashes*W,C]
        v_embed_sorted = batched_index_select(v_match, mod_indices)  # [N*H,n_hashes*W,C]

        # pad the embedding if it cannot be divided by chunk_size
        padding = self.chunk_size - H % self.chunk_size if H % self.chunk_size != 0 else 0
        w_att_buckets = torch.reshape(w_embed_sorted, (N*W, self.n_hashes, -1, self.out_planes // 2))  # [N*W, n_hashes,H,C]
        v_att_buckets = torch.reshape(v_embed_sorted, (N*W, self.n_hashes, -1, self.out_planes))
        if padding:
            pad_w = w_att_buckets[:, :, -padding:, :].clone()
            pad_v = v_att_buckets[:, :, -padding:, :].clone()
            w_att_buckets = torch.cat([w_att_buckets, pad_w], dim=2)
            v_att_buckets = torch.cat([v_att_buckets, pad_v], dim=2)
            #padding the embedding
            pad_w_emb = w_embedding[-padding:, :].clone()
            pad_v_emb = v_embedding[-padding:, :].clone()
            w_embedding = torch.cat([w_embedding, pad_w_emb], dim=0)
            v_embedding = torch.cat([v_embedding, pad_v_emb], dim=0)
        # get the buckets
        w_att_buckets = torch.reshape(w_att_buckets, (#此处reshape不可如此用
            N*W, self.groups,self.n_hashes, -1, self.chunk_size, self.group_planes // 2))  # [N*w,g,n_hashes, num_chunks, chunk_size, C]
        v_att_buckets = torch.reshape(v_att_buckets, (N*W, self.groups,self.n_hashes, -1, self.chunk_size, self.group_planes))
        w_embedding = torch.reshape(w_embedding,(-1,self.chunk_size,self.group_planes // 2))
        v_embedding = torch.reshape(v_embedding, (-1, self.chunk_size, self.group_planes))
        w_att_match = F.normalize(w_att_buckets, p=2, dim=-1, eps=5e-5)

        # allow attend to adjacent buckets
        w_att_match = self.add_adjacent_buckets(w_att_match)
        v_att_buckets = self.add_adjacent_buckets(v_att_buckets)# [N*w,g,n_hashes, num_chunks, chunk_size*3, C]
        w_embedding = self.add_adjacent_buckets_embed(w_embedding) #[num_chunks, chunk_size*3, C]
        v_embedding = self.add_adjacent_buckets_embed(v_embedding) #[num_chunks, chunk_size*3, C]

        # unormalized attention score
        raw_score = torch.einsum('bghkie,bghkje->bghkij', w_att_buckets,
                                 w_att_match)  # [N*w,g,n_hashes, num_chunks, chunk_size, chunk_size*3]
        w_add_w_embedd = torch.einsum('bghkie,kje->bghkij', w_att_buckets,
                                 w_embedding)  # [N*w,g,n_hashes, num_chunks, chunk_size, chunk_size*3]
        w_add_w_embedd = torch.mul(w_add_w_embedd,self.f_gw)
        score_similarity = torch.cat([raw_score, w_add_w_embedd], dim=1)
        score_similarity = F.normalize(score_similarity,p=2, dim=1, eps=5e-5).view\
            (N * W, 2, self.groups,self.n_hashes, -1, self.chunk_size, self.chunk_size*3).sum(dim=1)

        # softmax
        bucket_score = torch.logsumexp(score_similarity, dim=-1, keepdim=True)
        score = torch.exp(score_similarity - bucket_score)  # (after softmax)
        bucket_score = torch.reshape(bucket_score, [N * W,self.groups,self.n_hashes,-1])

        # attention
        ret_out = torch.einsum('bgukij,bgukje->bgukie', score, v_att_buckets)  # [N, g,n_hashes, num_chunks, chunk_size, C]
        ret_embedd = torch.einsum('bgukij,kje->bgukie', score, v_embedding)
        ret_embedd =  torch.mul(ret_embedd,self.f_gv1)
        ret_out =  torch.mul(ret_out,self.f_gv2)
        # add embedd
        ret = torch.cat([ret_out, ret_embedd], dim=-1).view\
            (N * W,self.n_hashes, -1, self.chunk_size, self.out_planes*2)
        # weighted sum multi-groups
        ret = torch.reshape(ret,(N * W, self.out_planes * 2, self.n_hashes, -1, self.chunk_size))
        ret = F.normalize(ret,p=2, dim=1, eps=5e-5).view\
            (N * W,self.n_hashes, -1, self.chunk_size,self.out_planes,2).sum(dim=-1)
        bucket_score = F.normalize(bucket_score,p=2, dim=1, eps=5e-5).sum(dim=1)# [N*W, n_hashes,H]

        # weighted sum multi-round
        ret = torch.reshape(ret, (N * W, self.n_hashes, -1,self.out_planes))  # [N*W, n_hashes*H,C]
            # if padded, then remove extra elements
        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        # recover the original order
        ret = torch.reshape(ret, (N*W,-1,self.out_planes))  # [N*W,n_hashes*H,C]
        bucket_score = torch.reshape(bucket_score, (N*W, -1,))  # [N*W,n_hashes*H]
        ret = batched_index_select(ret, undo_sort)  # [N*W, n_hashes*H*,C]
        bucket_score = bucket_score.gather(1, undo_sort)  # [N*W,n_hashes*H]
        #sum multi-round
        ret = torch.reshape(ret, (N*W, self.n_hashes, H, self.out_planes))  # [N*W, n_hashes*H,C]
        bucket_score = torch.reshape(bucket_score, (N*W, self.n_hashes, H, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)# [N*W, H,C]
        ret = ret.view(N, W, H,self.out_planes)
        if self.width:
            ret = ret.permute(0, 3, 1, 2)
        else:
            ret = ret.permute(0, 3, 2, 1)
        if self.stride > 1:
            ret = self.pooling(ret)
        return ret

class SparseAxialAttenion_Block(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56,
                 chunk_size = 25):
        super(SparseAxialAttenion_Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = SparseAxialAttenion(width, width, groups=groups, kernel_size=kernel_size,chunk_size=chunk_size)
        self.width_block = SparseAxialAttenion(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                                  chunk_size=chunk_size,width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
if __name__ == "__main__":
    #x = torch.randn(4, 64, 128, 128)
    import cv2
    import numpy as np
    x = cv2.imread ('C:/Users/HP/Desktop/img_seg/Brain_seg/validation/img/0004.png',1)
    x = torch.from_numpy(np.transpose(x, (2, 0, 1)))
    x = torch.unsqueeze(x,dim=0)
    model1 = SparseAxialAttenion(in_planes=64, out_planes=64, kernel_size=128)
    model2 = SparseAxialAttenion(in_planes=64, out_planes=64, kernel_size=128, width=True)
    conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    #flops, params = profile(model,inputs=(4,3,256,256))
#    from torchstat import stat
#    stat(model, (64, 128, 128))
    x1 = conv1(x)
    fulldir_1 = 'C:/Users/HP/Desktop/4_1.png'
    fulldir_2 = 'C:/Users/HP/Desktop/4_2.png'
    fulldir_3 = 'C:/Users/HP/Desktop/4_3.png'
    cv2.imwrite(fulldir_1,x1[0, 0, :, :])
    x2 = model1(x1)
    x3 = model2(x2)
    cv2.imwrite(fulldir_2, x2[0, 0, :, :])
    cv2.imwrite(fulldir_3, x3[0, 0, :, :])
    print(y.shape)






