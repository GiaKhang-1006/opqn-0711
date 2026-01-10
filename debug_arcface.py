import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys
import math
from backbone import EdgeFaceBackbone 
from margin_metric import ArcFace
from data_loader import get_datasets_transform
from utils import AverageMeter

# --- CẤU HÌNH DEBUG (GIỐNG MAIN.PY) ---
ARGS_DATASET = 'facescrub'
ARGS_DATA_DIR = '/Users/giakhangha/Desktop/xu_ly_anh/facescrub_32' # Sửa lại đường dẫn của bạn nếu chạy local
ARGS_BS = 64  # Batch size nhỏ để debug nhanh
ARGS_IMAGE_SIZE = 32 # Đang test trên 32x32
ARGS_LR = 0.0001 # LR Backbone
ARGS_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ArcFace params (giống main)
S_ARCFACE = 64.0 
M_ARCFACE = 0.5

def debug_arcface():
    print(f"--- BẮT ĐẦU DEBUG ARCFACE TRÊN {ARGS_DEVICE} ---")
    
    # 1. LOAD DATA (Logic giống hệt main.py)
    print("--> Đang load data...")
    try:
        data_config = get_datasets_transform(ARGS_DATASET, ARGS_DATA_DIR, cross_eval=False, input_size=ARGS_IMAGE_SIZE)
        trainset = data_config['dataset'][0]
        transform_train = data_config['transform'][0] # Transform rời
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=ARGS_BS, shuffle=True, num_workers=0)
        num_classes = len(trainset.classes)
        print(f"   + Số class: {num_classes}")
        print(f"   + Số ảnh train: {len(trainset)}")
    except Exception as e:
        print(f"❌ Lỗi Load Data: {e}")
        return

    # 2. KHỞI TẠO MODEL
    print("--> Đang khởi tạo EdgeFace + ArcFace...")
    try:
        # Feature dim = 512 mặc định
        net = EdgeFaceBackbone(feature_dim=512)
        metric = ArcFace(in_features=512, out_features=num_classes, s=S_ARCFACE, m=M_ARCFACE)
        
        net = net.to(ARGS_DEVICE)
        metric = metric.to(ARGS_DEVICE)
    except Exception as e:
        print(f"❌ Lỗi khởi tạo Model: {e}")
        return

    # 3. OPTIMIZER (Giống main)
    optimizer = optim.AdamW([
        {'params': net.parameters(), 'lr': ARGS_LR},
        {'params': metric.parameters(), 'lr': ARGS_LR * 10}
    ], weight_decay=5e-4)
    
    criterion = nn.CrossEntropyLoss()

    # 4. VÒNG LẶP DEBUG (Chạy thử 50 batch)
    print("\n--- BẮT ĐẦU TRAINING THỬ (50 Steps) ---")
    net.train()
    metric.train()
    
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= 50: break # Chỉ chạy 50 batch rồi dừng
        
        inputs, targets = inputs.to(ARGS_DEVICE), targets.to(ARGS_DEVICE)
        
        # --- QUAN TRỌNG: CHECK KÍCH THƯỚC ẢNH VÀO ---
        if batch_idx == 0:
            print(f"   [Step 0] Raw Input Shape: {inputs.shape} (Nên là 32x32)")
            
        # Áp dụng Transform (Giống main.py)
        transformed_images = transform_train(inputs)
        
        if batch_idx == 0:
            print(f"   [Step 0] Transformed Input Shape: {transformed_images.shape}")
            # Nếu ở đây vẫn là 32x32 mà EdgeFace không có F.interpolate -> SẼ FAIL
            
            # Check nhanh giá trị min/max
            print(f"   [Step 0] Pixel Range: Min={transformed_images.min():.2f}, Max={transformed_images.max():.2f}")

        # Forward
        try:
            features = net(transformed_images)
            
            # Check kích thước feature output
            if batch_idx == 0:
                print(f"   [Step 0] Backbone Output Feature Shape: {features.shape} (Nên là [B, 512])")
                if features.shape[1] != 512:
                    print("   ❌ Feature dim sai!")
            
            outputs = metric(features, targets)
            loss = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Check Gradient Norm
            grad_norm_b = torch.norm(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])).item()
            grad_norm_m = torch.norm(torch.cat([p.grad.flatten() for p in metric.parameters() if p.grad is not None])).item()
            
            # Clip grad
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(metric.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            # Tính Accuracy batch này
            _, predicted = outputs.max(1)
            correct = predicted.eq(targets).sum().item()
            acc = 100. * correct / targets.size(0)

            print(f"Step {batch_idx}: Loss={loss.item():.4f} | Acc={acc:.2f}% | Grad_Backbone={grad_norm_b:.4f} | Grad_Metric={grad_norm_m:.4f}")

            if math.isnan(loss.item()):
                print("❌ LỖI: Loss bị NaN! (Gradient Explosion)")
                break

        except RuntimeError as e:
            print(f"❌ Lỗi Runtime (thường do sai kích thước matrix): {e}")
            break

    print("\n--- KẾT THÚC DEBUG ---")
    if loss.item() > 10.0 and batch_idx > 10:
         print("⚠️ KẾT LUẬN: Loss không giảm. ArcFace Margin quá khó hoặc Model không học được.")
    elif acc < 1.0 and batch_idx > 10:
         print("⚠️ KẾT LUẬN: Accuracy quá thấp. Model đang đoán mò.")
    else:
         print("✅ KẾT LUẬN SƠ BỘ: Model có vẻ đang học (Loss giảm, Acc tăng).")

if __name__ == "__main__":
    debug_arcface()