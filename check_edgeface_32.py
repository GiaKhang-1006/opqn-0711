import torch
import torch.nn as nn
from backbone import EdgeFaceBackbone # Import class của bạn

def check_architecture():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- KIỂM TRA EDGEFACE TRÊN {device} ---")
    
    # 1. Giả lập ảnh đầu vào 32x32 (Batch size = 2)
    # Lưu ý: Facescrub 32x32
    dummy_input_32 = torch.randn(2, 3, 32, 32).to(device)
    
    # 2. Khởi tạo model (Feature dim = 512)
    try:
        model = EdgeFaceBackbone(feature_dim=512)
        model.to(device)
        model.eval()
        print("✅ Khởi tạo EdgeFace thành công.")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo model: {e}")
        return

    # 3. Hook để xem kích thước sau từng block
    print("\n--- SOI KÍCH THƯỚC QUA TỪNG LỚP ---")
    
    # Hàm in shape
    def print_shape(name):
        def hook(module, input, output):
            # Output có thể là tuple, lấy cái đầu tiên
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            print(f"{name}: {out.shape}")
        return hook

    # Đăng ký hook vào các lớp con (tùy vào cách bạn đặt tên trong EdgeFaceBackbone)
    # Thường EdgeFace có các stage hoặc layers. Tôi sẽ thử đoán tên phổ biến.
    # Bạn có thể mở file backbone.py để xem tên chính xác.
    for name, layer in model.named_children():
        layer.register_forward_hook(print_shape(name))

    # 4. Forward Pass
    try:
        print(f"Input Shape: {dummy_input_32.shape}")
        output = model(dummy_input_32)
        print(f"Final Output Shape: {output.shape}")
    except Exception as e:
        print(f"\n❌ LỖI CRASH KHI FORWARD: {e}")
        print("Gợi ý: Có thể do kích thước Feature Map bị giảm về 0 hoặc mismatch kích thước FC.")

if __name__ == "__main__":
    check_architecture()