import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.io
import os
import cv2
import dlib
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as TF

detector = dlib.get_frontal_face_detector()  # Global dlib detector for align

def get_datasets_transform(dataset, data_dir="/kaggle/input/facescrub-edgeface-0710-1", cross_eval=False, backbone='resnet'):
    to_tensor = transforms.ToTensor()

    # Auto detect Kaggle and use specific paths
    if 'kaggle' in os.environ.get('PWD', ''):
        if dataset == 'facescrub':
            base_path = os.path.join(data_dir, 'facescrub')  # e.g., /kaggle/input/facescrub-0210-3/facescrub
        else:
            base_path = data_dir  # Fallback for other datasets
    else:
        base_path = data_dir  # Local environment

    # Define paths with folder existence check
    if dataset == 'facescrub':
        train_path = os.path.join(base_path, 'train', 'actors')
        test_path = os.path.join(base_path, 'test', 'actors')
        if not os.path.exists(train_path):
            train_path = os.path.join(base_path, 'train')  # Fallback
        if not os.path.exists(test_path):
            test_path = os.path.join(base_path, 'test')  # Fallback
    elif dataset == 'vggface2':
        if cross_eval:
            train_path = os.path.join(base_path, 'cross_train') if os.path.exists(os.path.join(base_path, 'cross_train')) else os.path.join(base_path, 'train')
            test_path = os.path.join(base_path, 'cross_test') if os.path.exists(os.path.join(base_path, 'cross_test')) else os.path.join(base_path, 'test')
        else:
            train_path = os.path.join(base_path, 'train')
            test_path = os.path.join(base_path, 'test')
    else:
        train_path = os.path.join(base_path, 'train')
        test_path = os.path.join(base_path, 'test')
        if not os.path.exists(train_path):
            train_path = base_path  # Fallback
        if not os.path.exists(test_path):
            test_path = base_path  # Fallback

    # Debug print
    print(f"Dataset: {dataset}, Cross-eval: {cross_eval}, Backbone: {backbone}")
    print(f"Train path: {train_path}, Test path: {test_path}")

    trainset = datasets.ImageFolder(root=train_path, transform=to_tensor)
    testset = datasets.ImageFolder(root=test_path, transform=to_tensor)

    # sample_image_path = "/kaggle/input/facescrub-edgeface-0710-1/facescrub/train/actors/Aaron_Eckhart/Aaron_Eckhart_1_1.jpeg"
    # sample_image = torchvision.io.read_image(sample_image_path)
    # transformed = transform_train(sample_image)
    # print("Sample transformed image shape:", transformed.shape, "Mean:", transformed.mean(), "Std:", transformed.std())

    # Hàm align dùng dlib (chỉ dùng nếu backbone=='edgeface' và dataset chưa pre-aligned)
    def align_face(img):  # img là PIL Image
        try:
            img_cv = np.array(img)[:,:,::-1]  # PIL RGB -> OpenCV BGR
            faces = detector(img_cv, 1)
            if not faces:
                return TF.to_tensor(img_cv[:,:,::-1])  # Fallback if no face
            main_face = max(faces, key=lambda f: f.width() * f.height())
            x, y, w, h = main_face.left(), main_face.top(), main_face.width(), main_face.height()
            margin = int(w * 0.35)
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(img_cv.shape[1], x + w + margin), min(img_cv.shape[0], y + h + margin)
            crop = img_cv[y1:y2, x1:x2]
            return TF.to_tensor(crop[:,:,::-1])  # BGR -> RGB tensor
        except Exception as e:
            print(f"Align error: {e}")
            return TF.to_tensor(img)  # Fallback

    # Align only for EdgeFace if dataset not pre-aligned (FaceScrub is pre-aligned 112x112)
    align_transform = nn.Identity()  # Skip align since dataset is pre-aligned
    # Uncomment below if you want to enable alignment for EdgeFace
    # align_transform = transforms.Lambda(align_face) if backbone == 'edgeface' else nn.Identity()

    # Normalize and resize conditional
    if backbone == 'edgeface':
        # norm_mean = [0.618, 0.465, 0.393]
        # norm_std = [0.238, 0.202, 0.190]

        norm_mean = (0.5, 0.5, 0.5) #Norm [-1 1] thay vì. [0 1]
        norm_std = (0.5, 0.5, 0.5)
        resize_crop_size = 120
        crop_size = 112
        
        
    else:  # resnet (gốc OPQN)
        if dataset == "vggface2" or cross_eval:
            norm_mean = (0.5, 0.5, 0.5)
            norm_std = (0.5, 0.5, 0.5)
            resize_crop_size = 120
            crop_size = 112
        else:  # facescrub
            norm_mean = [0.639, 0.479, 0.404]
            norm_std = [0.216, 0.183, 0.171]
            resize_crop_size = 35
            crop_size = 32

    # Transforms
    if cross_eval:
        transform_train = nn.Sequential(
            align_transform,
            transforms.Resize(resize_crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(norm_mean, norm_std),
        )
        transform_test = transform_train
    else:
        if dataset == "vggface2":
            transform_train = nn.Sequential(
                align_transform,
                transforms.Resize(resize_crop_size),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomGrayscale(p=0.2),  # Thêm giống EdgeFace
                # transforms.GaussianBlur(kernel_size=3),  # Thêm giống EdgeFace
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(norm_mean, norm_std),
            )
            transform_test = nn.Sequential(
                align_transform,
                transforms.Resize(resize_crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(norm_mean, norm_std),
            )
        else:
            transform_train = nn.Sequential(
                align_transform,
                transforms.Resize(resize_crop_size),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),  # Thêm giống EdgeFace
                transforms.GaussianBlur(kernel_size=3),  # Thêm giống EdgeFace
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(norm_mean, norm_std),
            )
            transform_test = nn.Sequential(
                align_transform,
                transforms.Resize(resize_crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(norm_mean, norm_std),
            )

    return {"dataset": [trainset, testset], "transform": [transform_train, transform_test]}


