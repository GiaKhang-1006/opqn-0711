# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# import torch
# from ptflops import get_model_complexity_info
# from backbone import resnet20_pq, SphereNet20_pq, ResNet_q, resnet20_hashing, EdgeFaceBackbone # import các class cần tính

# import torch.nn as nn
# import torch.nn.functional as F

# # === Copy nguyên các class cần thiết từ file của em ===

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# class Block(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.prelu1 = nn.PReLU(channels)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(channels)
#         self.prelu2 = nn.PReLU(channels)

#     def forward(self, x):
#         short_cut = x
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.prelu1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.prelu2(x)

#         return x + short_cut

# # Hàm tiện để in đẹp
# def print_flops_params(model, input_size):
#     macs, params = get_model_complexity_info(
#         model,
#         (3, input_size, input_size),  # kênh 3, H=W
#         as_strings=True,
#         print_per_layer_stat=False,   # True nếu muốn xem chi tiết từng layer
#         verbose=False,
#         backend='aten'                # khuyến nghị dùng 'aten' cho chính xác cao hơn
#     )
#     print(f"Input size: {input_size}x{input_size}")
#     print(f"   Params : {params}")
#     print(f"   MFLOPs (≈ 2 x {macs})")  # tool báo GMACs, nhân 2 ≈ GFLOPs
#     print("")

# # Ví dụ tính các model chính của em
# print("=== resnet20_pq (input 112x112) ===")
# model = resnet20_pq(num_layers=20, feature_dim=512, channel_max=512, size=7)
# print_flops_params(model, 112)


# print("=== resnet20_pq nhỏ hơn (input 32x32) ===")
# model_small = resnet20_pq(num_layers=20, size=4)  # thường cho 32x32, filter nhỏ hơn
# print_flops_params(model_small, 32)


# from backbones import get_model  # từ file __init__.py của em


# model = get_model('edgeface_xs_gamma_06')
# # Nếu cần wrap như class EdgeFaceBackbone (feature_dim=512)
# model = EdgeFaceBackbone(model_name='edgeface_xs_gamma_06', feature_dim=512)

# print("=== edgeface_xs_gamma_06 (input 112x112) ===")
# print_flops_params(model, 112)

# print("=== edgeface_xs_gamma_06 (input 32x32) ===")
# print_flops_params(model, 32)


# model = get_model('edgeface_xxs')
# # Nếu cần wrap như class EdgeFaceBackbone (feature_dim=512)
# model = EdgeFaceBackbone(model_name='edgeface_xxs', feature_dim=512)

# print("=== edgeface_xxs (input 112x112) ===")
# print_flops_params(model, 112)

# print("=== edgeface_xxs (input 32x32) ===")
# print_flops_params(model, 32)



import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
# import ptflops
from ptflops import get_model_complexity_info

# Chỉ import những gì bạn thực sự dùng để chạy
from backbone import resnet20_pq, EdgeFaceBackbone 
# Nếu class EdgeFaceBackbone đã tự gọi get_model bên trong thì không cần import get_model ở đây
# from backbones import get_model 

# Hàm tiện để in đẹp (Logic của bạn ở đây RẤT ĐÚNG: MACs * 2 = FLOPs)
def print_flops_params(model, input_size):
    # Chuyển model về CPU để đảm bảo ptflops chạy ổn định nhất (tránh lỗi device mismatch)
    model = model.cpu() 
    
    macs, params = get_model_complexity_info(
        model,
        (3, input_size, input_size), 
        as_strings=True,
        print_per_layer_stat=False, 
        verbose=False,
        backend='aten' 
    )
    print(f"Input size: {input_size}x{input_size}")
    print(f"   Params : {params}")
    print(f"   MFLOPs (≈ 2 x {macs})") 
    print("-" * 30)

# 1. Tính ResNet20 (Baseline)
print("=== resnet20_pq (input 112x112) ===")
# Đảm bảo các tham số num_layers, feature_dim khớp với config bạn train
model_res = resnet20_pq(num_layers=20, feature_dim=512, channel_max=512, size=7)
print_flops_params(model_res, 112)

print("=== resnet20_pq nhỏ hơn (input 32x32) ===")
model_res_small = resnet20_pq(num_layers=20, size=4) 
print_flops_params(model_res_small, 32)

# 2. Tính EdgeFace XS (Proposed)
print("=== edgeface_xs_gamma_06 (input 112x112) ===")
# Chỉ khởi tạo 1 lần duy nhất
model_xs = EdgeFaceBackbone(model_name='edgeface_xs_gamma_06', feature_dim=512)
print_flops_params(model_xs, 112)

print("=== edgeface_xs_gamma_06 (input 32x32) ===")
print_flops_params(model_xs, 32)

# 3. Tính EdgeFace XXS (Reference)
print("=== edgeface_xxs (input 112x112) ===")
model_xxs = EdgeFaceBackbone(model_name='edgeface_xxs', feature_dim=512)
print_flops_params(model_xxs, 112)