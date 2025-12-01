import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class OrthoPQ(nn.Module):

    def __init__(self, in_features, out_features, num_books, num_words, code_books, sc=30.0, m=0.50):
        super(OrthoPQ, self).__init__()
        self.in_features = in_features
        self.out_features = out_features    # num. of classes
        self.num_books = num_books
        self.num_words = num_words
        self.len_word = int(self.in_features / self.num_books)
        self.s = sc
        self.m = m
        self.weight = Parameter(torch.FloatTensor(self.num_books, self.out_features, self.len_word))
        self.mlp = Parameter(torch.FloatTensor(self.num_books, self.len_word, self.num_words))
        self.codebooks = Parameter(code_books, requires_grad=False)
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.mlp)

    def forward(self, input, label):

        # ------------------------- cos(theta) & phi(theta) ---------------------------------
        x_m = torch.split(input, self.len_word, dim=1)
        cosine_xw = []
        # x = []
        xc_prod_softs = []
        weights_norm = F.normalize(self.weight, dim=2)
        for i in range(self.num_books):
            x_norm = F.normalize(x_m[i])
            cosine_xw.append(F.linear(x_norm, weights_norm[i]))
            xc_prod_softmax = F.softmax(x_m[i] @ self.mlp[i], dim=1)
            xc_prod_softs.append(xc_prod_softmax)
        xc_softmax = torch.stack(xc_prod_softs, dim=0) 
        cosine_xww = torch.stack(cosine_xw, dim=0)

        # ------------------------- cos(sw)---------------------------------------------------
        s_m = torch.matmul(self.codebooks, torch.transpose(xc_softmax, 1, 2))   # construct s_m: (num_books * len_word * bs)

        cosine_sw = []
        for i in range(self.num_books):
            s_norm = F.normalize(s_m[i].t())
            cosine_sw.append(F.linear(s_norm, weights_norm[i]))
        cosine_sww = torch.stack(cosine_sw, dim=0) # shape: (M * bs * out_features)

        # ------------------------- OPQN-Loss.------------------------------------------------
        cosine_xww = cosine_xww.clamp(-1, 1)
        cosine_sww = cosine_sww.clamp(-1, 1)   # for numerical stability
        phi1 = cosine_xww - self.m
        phi2 = cosine_sww - self.m
        one_hot = torch.zeros(cosine_sww.shape[1:], device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        one_hot_all = one_hot.repeat(self.num_books, 1, 1)

        output1 = (one_hot_all * phi1) + ((1.0 - one_hot_all) * cosine_xww)  # similarity between x_i and w_i
        output2 = (one_hot_all * phi2) + ((1.0 - one_hot_all) * cosine_sww)  # similarity between s_i and w_i

        output1 *= self.s
        output2 *= self.s

        return torch.transpose(output1, 0, 1), torch.transpose(output2, 0, 1), torch.transpose(xc_softmax, 0, 1)

class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine).to(input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
    


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Các tham số phụ (thường không đổi)
        self.easy_margin = easy_margin   # True → dùng easy margin (ít dùng)
        self.ls_eps = ls_eps             # label smoothing (thường = 0.0)

        # Ngưỡng cos để tính arccos ổn định
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-6)  # tránh sqrt âm do numerical error
        
        phi = cosine * self.cos_m - sine * self.sin_m   # cos(theta + m)
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- label smoothing (tùy chọn) ---------------------------
        if self.ls_eps > 0:
            one_hot = one_hot * (1 - self.ls_eps) + (1 - one_hot) * self.ls_eps / (self.out_features - 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output