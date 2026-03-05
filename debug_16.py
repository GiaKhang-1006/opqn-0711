import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import loader từ file data_loader.py của bạn
# Đảm bảo file data_loader.py nằm cùng thư mục với file này
try:
    from data_loader import get_datasets_transform
except ImportError:
    print("❌ LỖI: Không tìm thấy file 'data_loader.py'. Hãy kiểm tra lại.")
    sys.exit(1)

# ==========================================
# 1. ĐỊNH NGHĨA MODEL (ResNet20 & Block)
# ==========================================
class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu1 = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.prelu2 = nn.PReLU(channels)

    def forward(self, x):
        short_cut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        return x + short_cut

class resnet20_pq(nn.Module):
    def __init__(self, num_layers=20, feature_dim=512, channel_max=512, size=7):    
        super().__init__()
        assert num_layers in [20, 64], 'spherenet num_layers should be 20 or 64'
        if num_layers == 20:
            layers = [1, 2, 4, 1]
        elif num_layers == 64:
            layers = [3, 8, 16, 3]
        else:
            raise ValueError('sphere' + str(num_layers) + "is not supported!")
        
        if channel_max == 512:
            filter_list = [3, 64, 128, 256, 512]
            stride_list = [2, 2, 2, 2] if size == 7 else [1, 2, 2, 2]
        else:
            filter_list = [3, 16, 32, 64, 128]
            stride_list = [1, 2, 2, 2]

        block = Block
        self.feature_dim = feature_dim
        
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=stride_list[0])
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=stride_list[1])
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=stride_list[2])
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=stride_list[3])
        
        self.bn = nn.BatchNorm1d(channel_max*size*size)
        self.fc = nn.Linear(channel_max*size*size, self.feature_dim)
        self.last_bn = nn.BatchNorm1d(self.feature_dim)
        self.drop = nn.Dropout()

    def _make_layer(self, block, inplanes, planes, num_units, stride):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.PReLU(planes))
        for i in range(num_units):
            layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.drop(x)
        x = self.fc(x)
        out = self.last_bn(x) 
        return out

# ==========================================
# 2. HÀM CHẨN ĐOÁN (DIAGNOSTIC TOOL)
# ==========================================
def diagnose_16bit_model(model, dataloader, device):
    print(f"\n--- BẮT ĐẦU CHẨN ĐOÁN TRÊN: {device} ---")
    model.eval()
    
    try:
        images, labels = next(iter(dataloader))
        print(f"Input batch shape: {images.shape}")
        # Kiểm tra nhanh range của ảnh
        print(f"Image Range -> Min: {images.min():.2f}, Max: {images.max():.2f} (Kỳ vọng: ~ -1 đến 1)")
    except Exception as e:
        print(f"❌ Lỗi Data Loader: {e}")
        return

    images = images.to(device)
    
    with torch.no_grad():
        embeddings = model(images) 
        vals = embeddings.cpu().numpy()

    # --- VẼ BIỂU ĐỒ ---
    plt.figure(figsize=(14, 6))
    
    # # Histogram
    # plt.subplot(1, 2, 1)
    # sns.histplot(vals.flatten(), bins=50, kde=True, color='blue')
    # plt.title("Phân bố giá trị Output (Histogram)")
    # plt.xlabel("Giá trị Feature")
    
    # # Bit Variance
    # bit_std = np.std(vals, axis=0) 
    # plt.subplot(1, 2, 2)
    # plt.bar(range(len(bit_std)), bit_std, color='red')
    # plt.title("Độ biến thiên (Std Dev) của từng Bit")
    # plt.xlabel("Bit Index (0-15)")
    # plt.axhline(y=0.01, color='black', linestyle='--', label='Ngưỡng chết')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(vals.flatten(), bins=50, kde=True, color='blue')
    plt.title("Output Value Distribution (Histogram)")
    plt.xlabel("Feature Value")

    # Bit Variance
    bit_std = np.std(vals, axis=0) 
    plt.subplot(1, 2, 2)
    plt.bar(range(len(bit_std)), bit_std, color='red')
    plt.title("Variance (Std Dev) of Each Bit")
    plt.xlabel("Bit Index (0-15)")
    plt.axhline(y=0.01, color='black', linestyle='--', label='Dead Threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # --- IN KẾT QUẢ ---
    print("\n--- PHÂN TÍCH ---")
    print(f"Trung bình: {np.mean(vals):.4f} | Max: {np.max(vals):.4f} | Min: {np.min(vals):.4f}")
    
    dead_bits = np.where(bit_std < 0.01)[0]
    if len(dead_bits) > 0:
        print(f"❌ CẢNH BÁO: {len(dead_bits)} bit bị 'CHẾT' (Variance gần 0): {dead_bits}")
    else:
        print("✅ Các bit đều hoạt động (Variance > 0.01).")

    from scipy.spatial.distance import pdist
    avg_dist = np.mean(pdist(vals, metric='euclidean'))
    print(f"Khoảng cách trung bình giữa các ảnh: {avg_dist:.4f}")
    
    if avg_dist < 0.1:
        print("💀 LỖI NGHIÊM TRỌNG: Mode Collapse (Ảnh khác nhau ra vector giống hệt nhau).")

# ==========================================
# 3. CLASS WRAPPER (Để nối Transform vào Dataset)
# ==========================================
class DebugDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, index):
        img, label = self.dataset[index] # img là Tensor 0-1
        # Sử dụng self.transform để xử lý ảnh
        return self.transform(img), label 
    
    def __len__(self):
        return len(self.dataset)

# ==========================================
# 4. CHẠY CHƯƠNG TRÌNH (MAIN)
# ==========================================
if __name__ == '__main__':
    # Cấu hình thiết bị
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # ---------------------------------------------------------
    # [CẦN SỬA] ĐƯỜNG DẪN DỮ LIỆU & CHECKPOINT
    # ---------------------------------------------------------
    # Trỏ đến thư mục CHA chứa folder 'facescrub' (không trỏ sâu vào facescrub/test)
    # Ví dụ folder cấu trúc là: /data/facescrub/test/... thì data_root là /data
    data_root = '/Users/giakhangha/Desktop/xu_ly_anh/facescrub_112'  
    # Đường dẫn file checkpoint 16-bit
    checkpoint_path = '/Users/giakhangha/Downloads/baseline_16.tar'         
    # ---------------------------------------------------------

    if os.path.exists(data_root):
        try:
            print("--> Đang khởi tạo DataLoader...")
            
            # Gọi hàm loader từ file code của bạn
            # input_size=112 ĐỂ ĐẢM BẢO TRANSFORM ĐÚNG CHO RESNET/EDGEFACE
            data_config = get_datasets_transform(
                dataset="facescrub", 
                data_dir=data_root, 
                cross_eval=False, 
                input_size=112 
            )
            
            raw_testset = data_config['dataset'][1]       # Dataset thô
            transform_test_func = data_config['transform'][1] # Transform chuẩn

            # Dùng Wrapper để kết hợp chúng lại
            dataset = DebugDatasetWrapper(raw_testset, transform_test_func)
            test_loader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            print(f"--> Đã load {len(dataset)} ảnh (Mode: 112x112).")

            # Load Model & Chạy Chẩn Đoán
            if os.path.exists(checkpoint_path):
                print(f"--> Đang load model từ: {checkpoint_path}")
                
                # --- [SỬA ĐỔI QUAN TRỌNG] ---
                # Thay feature_dim=16 thành feature_dim=512
                # Vì backbone luôn output 512 chiều, việc hash xảy ra sau đó.
                model = resnet20_pq(num_layers=20, feature_dim=512, channel_max=512, size=7)
                
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # 1. Trích xuất key 'backbone'
                if 'backbone' in checkpoint:
                    state_dict = checkpoint['backbone']
                    print("   -> Đã tìm thấy key 'backbone', đang trích xuất...")
                else:
                    state_dict = checkpoint
                
                # 2. Xử lý prefix 'module.'
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace("module.", "") 
                    new_state_dict[name] = v
                
                # Load vào model
                try:
                    model.load_state_dict(new_state_dict)
                    print("✅ Load weights thành công!")
                except RuntimeError as e:
                    print(f"❌ Vẫn lệch key: {e}")
                    sys.exit(1)

                model.to(device)
                
                # Chạy chẩn đoán
                diagnose_16bit_model(model, test_loader, device)
                
            else:
                print(f"❌ Không tìm thấy file checkpoint tại: {checkpoint_path}")
        except Exception as e:
            # In đầy đủ lỗi để dễ debug
            import traceback
            traceback.print_exc()
            print(f"❌ Lỗi ngoại lệ: {e}")
    else:
        print(f"❌ Không tìm thấy thư mục data tại: {data_root}")










# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# from data_loader import get_datasets_transform  # Import file loader của bạn

# # ==========================================
# # 1. ĐỊNH NGHĨA MODEL (Theo code bạn cung cấp)
# # ==========================================

# # Class Block của bạn (Dùng PReLU)
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

# # Class ResNet20 chính của bạn
# class resnet20_pq(nn.Module):
#     def __init__(self, num_layers=20, feature_dim=512, channel_max=512, size=7):    
#         super().__init__()
#         assert num_layers in [20, 64], 'spherenet num_layers should be 20 or 64'
#         if num_layers == 20:
#             layers = [1, 2, 4, 1]
#         elif num_layers == 64:
#             layers = [3, 8, 16, 3]
#         else:
#             raise ValueError('sphere' + str(num_layers) + "is not supported!")
        
#         if channel_max == 512:
#             filter_list = [3, 64, 128, 256, 512]
#             if size == 7:
#                 stride_list = [2, 2, 2, 2] # 112 -> 56 -> 28 -> 14 -> 7
#             else:
#                 stride_list = [1, 2, 2, 2]
#         else:
#             filter_list = [3, 16, 32, 64, 128]
#             stride_list = [1, 2, 2, 2]

#         block = Block # Sử dụng class Block ở trên
#         self.feature_dim = feature_dim
        
#         # Tạo các layer
#         self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=stride_list[0])
#         self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=stride_list[1])
#         self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=stride_list[2])
#         self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=stride_list[3])
        
#         self.bn = nn.BatchNorm1d(channel_max*size*size)
#         self.fc = nn.Linear(channel_max*size*size, self.feature_dim)
#         self.last_bn = nn.BatchNorm1d(self.feature_dim)
#         self.drop = nn.Dropout()

#     def _make_layer(self, block, inplanes, planes, num_units, stride):
#         layers = []
#         layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
#         layers.append(nn.BatchNorm2d(planes))
#         layers.append(nn.PReLU(planes))
#         for i in range(num_units):
#             layers.append(block(planes))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = x.view(x.size(0), -1)
#         x = self.bn(x)
#         x = self.drop(x)
#         x = self.fc(x)
#         out = self.last_bn(x) # Output cuối cùng
#         return out

# # ==========================================
# # 2. HÀM CHẨN ĐOÁN LỖI (DIAGNOSTIC TOOL)
# # ==========================================
# def diagnose_16bit_model(model, dataloader, device):
#     print(f"\n--- BẮT ĐẦU CHẨN ĐOÁN TRÊN THIẾT BỊ: {device} ---")
#     model.eval()
    
#     # Lấy 1 batch
#     try:
#         images, labels = next(iter(dataloader))
#         print(f"Input batch shape: {images.shape}")
#     except Exception as e:
#         print(f"Lỗi Data Loader: {e}")
#         return

#     images = images.to(device)
    
#     with torch.no_grad():
#         # Lấy output feature
#         embeddings = model(images) 
#         vals = embeddings.cpu().numpy()

#     # --- VẼ BIỂU ĐỒ ---
#     plt.figure(figsize=(14, 6))
    
#     # 1. Histogram: Xem phân bố giá trị
#     plt.subplot(1, 2, 1)
#     sns.histplot(vals.flatten(), bins=50, kde=True, color='blue')
#     plt.title("Phân bố giá trị Output (Histogram)")
#     plt.xlabel("Giá trị Feature")
#     plt.ylabel("Số lượng")
    
#     # 2. Bit Variance: Xem bit nào bị chết
#     bit_std = np.std(vals, axis=0) # Độ lệch chuẩn từng cột
#     plt.subplot(1, 2, 2)
#     bars = plt.bar(range(len(bit_std)), bit_std, color='red')
#     plt.title("Độ biến thiên (Std Dev) của từng Bit")
#     plt.xlabel("Bit Index (0-15)")
#     plt.ylabel("Standard Deviation")
#     plt.axhline(y=0.01, color='black', linestyle='--', label='Ngưỡng chết')
#     plt.xticks(range(16)) # Hiển thị đủ 16 số
#     plt.legend()
    
#     plt.tight_layout()
#     plt.show()

#     # --- IN KẾT QUẢ SỐ LIỆU ---
#     print("\n--- PHÂN TÍCH CHI TIẾT ---")
#     print(f"Trung bình (Mean): {np.mean(vals):.4f}")
#     print(f"Max: {np.max(vals):.4f} | Min: {np.min(vals):.4f}")
    
#     # Đếm bit chết
#     dead_bits = np.where(bit_std < 0.01)[0]
#     if len(dead_bits) > 0:
#         print(f"❌ CẢNH BÁO: Có {len(dead_bits)} bit bị 'CHẾT' (không đổi giá trị):")
#         print(f"   Danh sách bit chết: {dead_bits}")
#     else:
#         print("✅ Tất cả các bit đều hoạt động (Variance > 0.01).")

#     # Kiểm tra Mode Collapse (tất cả ảnh ra vector giống nhau)
#     from scipy.spatial.distance import pdist
#     dists = pdist(vals, metric='euclidean')
#     avg_dist = np.mean(dists)
#     print(f"Khoảng cách trung bình giữa các ảnh (Euclidean): {avg_dist:.4f}")
    
#     if avg_dist < 0.1:
#         print("💀 LỖI NGHIÊM TRỌNG: Mode Collapse (Các ảnh khác nhau ra vector giống hệt nhau).")
#     elif avg_dist < 5.0:
#         print("⚠ CẢNH BÁO: Khoảng cách giữa các ảnh rất nhỏ, model phân tách kém.")
#     else:
#         print("✅ Khoảng cách giữa các ảnh tốt.")

# # ==========================================
# # 3. CHẠY CHƯƠNG TRÌNH (MAIN)
# # ==========================================
# if __name__ == '__main__':
#     # A. Cấu hình thiết bị
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
#     # B. Cấu hình Đường dẫn [BẠN CẦN SỬA Ở ĐÂY]
#     # ---------------------------------------------------------
#     data_path = '/Users/giakhangha/Desktop/xu_ly_anh/facescrub_112'  # <-- Sửa đường dẫn ảnh
#     checkpoint_path = '/Users/giakhangha/Downloads/baseline_16.tar'         # <-- Sửa đường dẫn file .pth
#     # ---------------------------------------------------------

#     # ==========================================
#     # C. Tạo DataLoader (Đã sửa để dùng đúng data_loader.py của bạn)
#     # ==========================================

#     if os.path.exists(data_path):
#         try:
#             # 1. Gọi hàm loader gốc
#             # LƯU Ý: Phải truyền input_size=112 để lấy đúng transform cho ResNet/EdgeFace
#             # data_path ở đây là thư mục cha chứa 'facescrub', 'vggface2'... 
#             # Nếu data_path của bạn đã trỏ thẳng vào facescrub, bạn cần chỉnh lại tham số data_dir cho phù hợp
#             # Ví dụ: nếu data_path = '/.../facescrub/test', thì data_dir nên là '/.../'
#             # Để an toàn nhất, tôi sẽ giả định data_path bạn set ở trên là root dir.
            
#             # Gọi hàm lấy config
#             # Bạn cần trỏ data_dir về thư mục gốc chứa dataset
#             # Ví dụ: data_path bên trên bạn set là '/Users/username/data'
#             data_config = get_datasets_transform(
#                 dataset="facescrub", 
#                 data_dir='/Users/giakhangha/Desktop/xu_ly_anh/facescrub_112', # Thử lấy thư mục cha nếu path quá sâu
#                 cross_eval=False, 
#                 input_size=112 # QUAN TRỌNG: Ép dùng size 112
#             )
            
#             # 2. Lấy thành phần rời rạc
#             raw_testset = data_config['dataset'][1]       # Dataset chỉ có ToTensor (0-1)
#             transform_test_func = data_config['transform'][1] # Transform (Resize, Norm -1..1)

#             # 3. Tạo Wrapper để "dính" Transform vào Dataset
#             # Vì code gốc của bạn tách riêng, ta cần class này để Debug chạy được
#             class DebugDatasetWrapper(torch.utils.data.Dataset):
#                 def __init__(self, dataset, transform):
#                     self.dataset = dataset
#                     self.transform = transform
                
#                 def __getitem__(self, index):
#                     img, label = self.dataset[index] # img lúc này là Tensor 0-1
#                     return self.transform(img), label # img sau khi qua transform là Tensor -1..1 chuẩn
                
#                 def __len__(self):
#                     return len(self.dataset)

#             # 4. Tạo DataLoader từ Wrapper
#             # Wrapper sẽ tự động Resize và Normalize mỗi khi bạn lấy ảnh
#             dataset = DebugDatasetWrapper(raw_testset, transform_test_func)
#             test_loader = DataLoader(dataset, batch_size=64, shuffle=True)
            
#             print(f"--> Đã load {len(dataset)} ảnh (Mode: 112x112 qua Wrapper).")

#         except Exception as e:
#             print(f"❌ Lỗi tạo DataLoader: {e}")
#             print("Mẹo: Kiểm tra lại 'data_path' xem có đúng cấu trúc thư mục mà data_loader.py yêu cầu không.")
#     else:
#         print(f"❌ Không tìm thấy thư mục ảnh tại: {data_path}")

#     if os.path.exists(data_path):
#         try:
#             # Dùng ImageFolder giả định cấu trúc thư mục: data_path/class_name/image.jpg
#             dataset = datasets.ImageFolder(root=data_path, transform=transform)
#             test_loader = DataLoader(dataset, batch_size=64, shuffle=True)
#             print(f"Đã load {len(dataset)} ảnh.")

#             # D. Load Model & Chạy Chẩn Đoán
#             if os.path.exists(checkpoint_path):
#                 print("Đang load model...")
#                 # Khởi tạo model: 16 bit, size=7 (cho ảnh 112x112)
#                 model = resnet20_pq(num_layers=20, feature_dim=16, channel_max=512, size=7)
                
#                 # Load weights
#                 state_dict = torch.load(checkpoint_path, map_location=device)
                
#                 # Xử lý nếu state_dict có prefix 'module.'
#                 if list(state_dict.keys())[0].startswith('module.'):
#                     from collections import OrderedDict
#                     new_state_dict = OrderedDict()
#                     for k, v in state_dict.items():
#                         new_state_dict[k.replace("module.", "")] = v
#                     state_dict = new_state_dict
                
#                 model.load_state_dict(state_dict)
#                 model.to(device)
#                 print("Load model thành công! Đang chạy chẩn đoán...")
                
#                 # ---> GỌI HÀM CHẨN ĐOÁN <---
#                 diagnose_16bit_model(model, test_loader, device)
                
#             else:
#                 print(f"❌ Không tìm thấy file checkpoint tại: {checkpoint_path}")
#         except Exception as e:
#             print(f"❌ Lỗi trong quá trình chạy: {e}")
#     else:
#         print(f"❌ Không tìm thấy thư mục ảnh tại: {data_path}")