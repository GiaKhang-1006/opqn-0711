# data_loader.py - Phiên bản cập nhật hỗ trợ FaceScrub-EdgeFace 112x112 cho EdgeFace testing
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import os

def get_datasets_transform(dataset, data_dir="/kaggle/input/facescrub-32x32-opqn", cross_eval=False, backbone='resnet'):
    """
    ĐÚNG THEO CODE CHÍNH CHỦ OPQN + hỗ trợ EdgeFace 112x112
    - FaceScrub gốc: 32x32 + mean/std cụ thể
    - EdgeFace: 112x112 + [-1,1] + augments thêm (Blur, Grayscale)
    - Tự động detect qua data_dir và backbone
    """
    to_tensor = transforms.ToTensor()

    # === ĐƯỜNG DẪN ===
    if dataset != "vggface2":
        train_path = os.path.join(data_dir, dataset, "train")
        test_path  = os.path.join(data_dir, dataset, "test")
    else:
        if cross_eval:
            train_path = os.path.join(data_dir, "vggface2", "cross_train")
            test_path  = os.path.join(data_dir, "vggface2", "cross_test")
        else:
            train_path = os.path.join(data_dir, "vggface2", "train")
            test_path  = os.path.join(data_dir, "vggface2", "test")

    print(f"[OPQN Loader] Dataset: {dataset} | Backbone: {backbone} | DataDir: {data_dir}")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")

    trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
    testset  = datasets.ImageFolder(root=test_path,  transform=to_tensor)

    # === TỰ ĐỘNG PHÁT HIỆN KÍCH THƯỚC ===
    is_32x32_facescrub = dataset != "vggface2" and any(x in data_dir.lower() for x in ['32x32', '32', '0210', '0210-3'])
    is_112x112_facescrub = dataset != "vggface2" and any(x in data_dir.lower() for x in ['0710','0710-1'])

    # === EDGEFACE AUGMENTS (Blur + Grayscale) - CHỈ KHI backbone=edgeface ===
    # edgeface_augs = [
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    # ] if backbone == 'edgeface' else []

    # === FACESCRUB 32x32 (OPQN GỐC) ===
    if is_32x32_facescrub:
        norm_mean = [0.639, 0.479, 0.404]
        norm_std  = [0.216, 0.183, 0.171]
        resize_size = 35
        crop_size = 32

        print("→ FaceScrub 32x32 (OPQN Official): mean/std cụ thể + Resize(35) → Crop(32)")

        base_train = [
            transforms.Resize(resize_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
        ] #+ edgeface_augs  # Thêm nếu backbone=edgeface, nhưng thường không dùng cho 32x32

        base_test = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
        ]

    # === EDGEFACE 112x112 (MỚI THÊM CHO TEST) ===
    elif is_112x112_facescrub:
        norm_mean = (0.5, 0.5, 0.5)
        norm_std  = (0.5, 0.5, 0.5)
        resize_size = 120
        crop_size = 112

        print("→ FaceScrub-EdgeFace 112x112: [-1,1] normalize + Resize(120) → Crop(112) + EdgeFace augments")

        base_train = [
            transforms.Resize(resize_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
        ] #+ edgeface_augs  # Luôn thêm cho EdgeFace

        base_test = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
        ]

    # === VGGFACE2 HOẶC MẶC ĐỊNH (112x112) ===
    else:
        norm_mean = (0.5, 0.5, 0.5)
        norm_std  = (0.5, 0.5, 0.5)
        resize_size = 120
        crop_size = 112

        print("→ VGGFace2 / Default 112x112: [-1,1] normalize + Resize(120) → Crop(112)")

        base_train = [
            transforms.Resize(resize_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
        ] #+ edgeface_augs

        base_test = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
        ]

    # === FINAL TRANSFORMS ===
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    if cross_eval:
        transform_train = transforms.Compose(base_test + [
            transforms.ConvertImageDtype(torch.float),
            normalize,
        ])
        transform_test = transform_train
    else:
        transform_train = transforms.Compose(base_train + [
            transforms.ConvertImageDtype(torch.float),
            normalize,
        ])
        transform_test = transforms.Compose(base_test + [
            transforms.ConvertImageDtype(torch.float),
            normalize,
        ])

    #print(f"  → Size: {crop_size}x{crop_size} | Norm: {norm_mean[:1]}... | Augs: {'+Blur/Grayscale' if edgeface_augs else ''}")
    print(f"  → Size: {crop_size}x{crop_size} | Norm: {norm_mean[:1]}... ")

    return {
        "dataset": [trainset, testset],
        "transform": [transform_train, transform_test]
    }