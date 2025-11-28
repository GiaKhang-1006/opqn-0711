import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from datetime import datetime
import torch.distributions as Distributions
import math
import argparse
import sys
import time
import os
from utils import Logger, AverageMeter, compute_quant, compute_quant_indexing, PqDistRet_Ortho
from backbone import resnet20_pq, SphereNet20_pq, EdgeFaceBackbone
from margin_metric import OrthoPQ
from data_loader import get_datasets_transform
import random
import numpy as np  # Thêm import nếu chưa có

parser = argparse.ArgumentParser(description='PyTorch Implementation of Orthonormal Product Quantization (OPQN-v2.1)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate mode')
parser.add_argument('-c', '--cross-dataset', action='store_true', help='generalize on unseen identities')
parser.add_argument('--bs', type=int, default=256, help='batch size')
parser.add_argument('--save', nargs='+', help='path to saving models')
parser.add_argument('--load', nargs='+', help='path to loading models')
parser.add_argument('--len', nargs='+', type=int, help='code length in bits')
parser.add_argument('--dataset', type=str, default='facescrub', help='dataset: facescrub, youtube, cfw, vggface2')
parser.add_argument('--num', nargs='+', type=int, help='num. of codebooks (4, 8, ...)')
parser.add_argument('--words', nargs='+', type=int, default=[256, 256, 256, 256], help='num of words per book (2**n)')
parser.add_argument('--margin', default=0.4, type=float, help='cosine margin')
parser.add_argument('--miu', default=0.1, type=float, help='entropy loss weight')
parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'edgeface'])
parser.add_argument('--data_dir', type=str, default='/kaggle/input/facescrub-0210-3', help='dataset root')
parser.add_argument('--sc', default=40, type=float, help='scale s for metric init (paper: 40)')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--scheduler_type', default='step', choices=['step', 'plateau'], help='lr scheduler')
parser.add_argument('--freeze', action='store_true', help='freeze backbone (for finetuning)')
parser.add_argument('--image_size', default=32, type=float, help='Input image size (32 or 112)')

try:
    args = parser.parse_args()
except Exception as e:
    print(f"Parser error: {e}")
    sys.exit(1)

# trainset, testset = get_datasets_transform(args.dataset, cross_eval=args.cross_dataset, data_dir=args.data_dir)['dataset']
# transform_train, transform_test = get_datasets_transform(args.dataset, cross_eval=args.cross_dataset, data_dir=args.data_dir)['transform']


trainset, testset = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, input_size=args.image_size)['dataset']
transform_train, transform_test = get_datasets_transform(args.dataset, args.data_dir, cross_eval=args.cross_dataset, input_size=args.image_size)['transform']

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, pin_memory=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, pin_memory=True, num_workers=4)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#torch.cuda.manual_seed_all(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   # <--- quan trọng nhất

class adjust_lr:
    def __init__(self, step, decay):
        self.step = step
        self.decay = decay

    def adjust(self, optimizer, epoch):
        lr = args.lr * (self.decay ** (epoch // self.step))
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
        return lr

def train(save_path, length, num, words, feature_dim):
    best_acc = 0  # Khôi phục biến này (dù không dùng)
    best_mAP = 0
    best_epoch = 1
    print('==> Building model..')
    num_classes = len(trainset.classes)
    print("number of identities: ", num_classes)
    print("number of training images: ", len(trainset))
    print("number of test images: ", len(testset))
    print("number of training batches per epoch:", len(train_loader))
    print("number of testing batches per epoch:", len(test_loader))

    d = int(feature_dim / num)
    matrix = torch.randn(d, d)
    for k in range(d):
        for j in range(d):
            matrix[j, k] = math.cos((j+0.5)*k*math.pi/d)
    matrix[:, 0] /= math.sqrt(2)    # divided by sqrt(2)
    matrix /= math.sqrt(d/2)    # divided by sqrt(N/2) to got orthonormal
    code_books = torch.Tensor(num, d, words)
    code_books[0] = matrix[:, :words]
    for i in range(1, num):
        code_books[i] = matrix @ code_books[i-1]

    if args.backbone == 'resnet':
        if args.cross_dataset or args.dataset == "vggface2":
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
        else:
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
    elif args.backbone == 'edgeface':
        net = SphereNet20_pq(feature_dim=feature_dim)  # Giả sử SphereNet20_pq cho edgeface; chỉnh nếu cần

    metric = OrthoPQ(in_features=feature_dim, out_features=num_classes, num_books=num, code_books=code_books, num_words=words, sc=args.sc, m=args.margin)

    num_books = metric.num_books
    len_word = metric.len_word
    num_words = metric.num_words
    len_bit = int(num_books * math.log(num_words, 2))
    assert length == len_bit, "something went wrong with code length"
    criterion = nn.CrossEntropyLoss()
    print("num. of codebooks: ", num_books)
    print("num. of words per book: ", num_words)
    print("dim. of word: ", len_word)
    print("code length: %d-bit \t learning rate: %.3f \t scale length: %d \t penalty margin: %.2f \t balance_weight: %.3f" % 
          (len_bit, args.lr, metric.s, metric.m, args.miu))
    net = nn.DataParallel(net).to(device)
    metric = nn.DataParallel(metric).to(device)
    #cudnn.benchmark = True

    if args.freeze:
        for param in net.parameters():
            param.requires_grad = False

    optimizer = optim.SGD([{'params': net.parameters()}, {'params': metric.parameters()}], lr=args.lr, weight_decay=args.wd, momentum=0.9)

    if args.scheduler_type == 'step':
        if args.dataset in ["facescrub", "cfw", "youtube"]:
            scheduler = adjust_lr(35, 0.5)
            EPOCHS = 200
        else:
            scheduler = adjust_lr(20, 0.5)
            EPOCHS = 160
    elif args.scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        if args.dataset in ["facescrub", "cfw", "youtube"]:
            EPOCHS = 200
        else:
            EPOCHS = 160

    since = time.time()
    best_loss = 1e3

    for epoch in range(EPOCHS):
        print('==> Epoch: %d' % (epoch+1))
        net.train()
        losses = AverageMeter()
        if args.scheduler_type == 'step':
            scheduler.adjust(optimizer, epoch)
        start = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            transformed_images = transform_train(inputs)
            features = net(transformed_images)
            output1, output2, xc_probs = metric(features, targets)
            # Subspacewise joint clf. loss
            loss_clf1 = [criterion(output1[:, i, :], targets) for i in range(num_books)]  # logits from original features
            loss_clf2 = [criterion(output2[:, i, :], targets) for i in range(num_books)]  # logits from soft quantized features
            loss_clf = 0.5 * (sum(loss_clf1) / len(loss_clf1) + sum(loss_clf2) / len(loss_clf2))

            # Entropy minimization
            xc_entropy = [Distributions.categorical.Categorical(probs=xc_probs[:, i, :]).entropy().sum() for i in range(num_books)]  # -p * logP
            loss_entropy = sum(xc_entropy) / (num_books * len(inputs))
            loss = loss_clf + args.miu * loss_entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), len(inputs))

        epoch_elapsed = time.time() - start
        print('Epoch %d | Loss: %.4f' % (epoch+1, losses.avg))
        print("Epoch Completed in {:.0f}min {:.0f}s".format(epoch_elapsed // 60, epoch_elapsed % 60))
        if args.scheduler_type == 'plateau':
            scheduler.step(losses.avg)

        if (epoch+1) % 5 == 0:
            net.eval()
            with torch.no_grad():
                mlp_weight = metric.module.mlp
                index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
                queries, test_labels = compute_quant(transform_test, test_loader, net, device)
                start = time.time()
                mAP, top_k = PqDistRet_Ortho(queries, test_labels, train_labels, index, mlp_weight, len_word, num_books, device, top=50)
                time_elapsed = time.time() - start
                print("Code generated in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
                print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

            if losses.avg < best_loss:
                best_loss = losses.avg
                # best_mAP = mAP  # Khôi phục dòng này (commented trong gốc)
                print('Saving..')
                checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({'backbone': net.state_dict(), 'mlp': metric.module.mlp}, os.path.join(checkpoint_dir, save_path))
                best_epoch = epoch + 1
    time_elapsed = time.time() - since
    print("Training Completed in {:.0f}min {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best mAP {:.4f} at epoch {}".format(best_mAP, best_epoch))
    print("Model saved as %s" % save_path)

def test(load_path, length, num, words, feature_dim=512):
    len_bit = int(num * math.log(words, 2))
    assert length == len_bit, "something went wrong with code length"

    print(f"=============== Evaluation on model {load_path} ===============")
    print(f"Train classes: {len(trainset.classes)}   | Test classes: {len(testset.classes)}")
    print(f"Train images: {len(trainset)}            | Test images: {len(testset)}")

    # ----------------------------- LOAD BACKBONE ----------------------------- #
    if args.backbone == 'edgeface':
        net = EdgeFaceBackbone(feature_dim=feature_dim)
    else:
        if args.dataset == 'facescrub':
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
        else:
            net = resnet20_pq(num_layers=20, feature_dim=feature_dim)

    net = nn.DataParallel(net).to(device)

    # ----------------------------- CHECKPOINT PATH ---------------------------- #
    checkpoint_dir = (
        '/kaggle/working/opqn-0210/checkpoint/'
        if 'kaggle' in os.environ.get('PWD', '')
        else 'checkpoint'
    )
    checkpoint_path = load_path if os.path.isabs(load_path) else os.path.join(checkpoint_dir, load_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['backbone'])
    mlp_weight = checkpoint.get('mlp', None)

    net.eval()
    len_word = feature_dim // num

    # ----------------------------- FEATURE EXTRACTION ----------------------------- #
    start_total = time.perf_counter()

    with torch.no_grad():
       # 1. PHASE OFFLINE (Không tính giờ vào metric truy vấn)
        print("Indexing Training Set (Offline)...")
        index, train_labels = compute_quant_indexing(
            transform_test, train_loader, net, len_word, mlp_weight, device
        )
        
        # -------------------------------------------------------------------
        # 2. ĐO THỜI GIAN FEATURE EXTRACTION (Của tập Test/Query)
        # -------------------------------------------------------------------
        # Warm-up GPU (chạy nháp 1 chút để GPU nóng máy, tránh lần đầu bị chậm)
        dummy_input = torch.randn(1, 3, 112, 112).to(device)
        _ = net(dummy_input)
        torch.cuda.synchronize() 
        
        start_extract = time.perf_counter()
        
        # Trích xuất đặc trưng query
        query_features, test_labels = compute_quant(
            transform_test, test_loader, net, device
        )
        
        torch.cuda.synchronize() # Chờ GPU trích xuất xong hết mới dừng giờ
        extract_time_total = (time.perf_counter() - start_extract)
        
        # -------------------------------------------------------------------
        # 3. ĐO THỜI GIAN SEARCH (TÍNH KHOẢNG CÁCH + SẮP XẾP)
        # -------------------------------------------------------------------
        # Chỉ đo 1 lần tìm kiếm duy nhất (ví dụ lấy top lớn nhất cần dùng, ở đây là len(trainset))
        
        torch.cuda.synchronize()
        start_search = time.perf_counter()
        
        # Gọi hàm search 1 lần duy nhất
        # Lưu ý: Hàm này nên trả về khoảng cách/kết quả thô, 
        # việc tính mAP hay Accuracy là việc của đánh giá, không phải thời gian search thực tế.
        # Tuy nhiên nếu hàm PqDistRet_Ortho gộp chung thì ta đành đo chung.
        # Để công bằng, chỉ đo lần chạy nặng nhất (top=len(trainset)) hoặc top-100
        
        mAP, _ = PqDistRet_Ortho(
            query_features, test_labels,
            train_labels, index,
            mlp_weight, len_word, num,
            device,
            top=len(trainset) 
        )
        
        torch.cuda.synchronize() # Chờ search xong
        search_time_total = (time.perf_counter() - start_search)

        # -------------------------------------------------------------------
        # 4. TÍNH TOÁN KẾT QUẢ
        # -------------------------------------------------------------------
        num_queries = len(testset)
        
        # Thời gian trung bình trích xuất 1 ảnh
        avg_extract_ms = (extract_time_total * 1000) / num_queries
        
        # Thời gian trung bình tìm kiếm cho 1 ảnh
        avg_search_ms = (search_time_total * 1000) / num_queries
        
        # Tổng thời gian latency cho 1 query
        avg_time_per_query = avg_extract_ms + avg_search_ms

        print(f"=============== Latency Report ===============")
        print(f"Total Queries: {num_queries}")
        print(f"Feature Extraction Time: {avg_extract_ms:.4f} ms/img")
        print(f"Search Time (Retrieval): {avg_search_ms:.4f} ms/query")
        print(f"--> Average Time per Query: {avg_time_per_query:.4f} ms")

        # 2) Tính top-k từ 10 → 100
        topk_dict = {}
        for k in range(10, 101, 10):
            _, acc_k = PqDistRet_Ortho(
                query_features, test_labels,
                train_labels, index,
                mlp_weight, len_word, num,
                device,
                top=k
            )
            topk_dict[k] = acc_k


    total_time_ms = (time.perf_counter() - start_total) * 1000
    avg_query_time = total_time_ms / len(testset)

    # ----------------------------- OUTPUT ----------------------------- #
    print(f"[Evaluate] mAP: {100 * mAP:.2f}%")
   
    for k, acc in topk_dict.items():
        print(f"[Evaluate @ top-{k}] accuracy: {100 * acc:.2f}%")

# def test(load_path, length, num, words, feature_dim=512):
#     len_bit = int(num * math.log(words, 2))
#     assert length == len_bit, "something went wrong with code length"

#     print(f"=============== Evaluation on model {load_path} ===============")
#     print(f"Train classes: {len(trainset.classes)}   | Test classes: {len(testset.classes)}")
#     print(f"Train images: {len(trainset)}            | Test images: {len(testset)}")

#     # ----------------------------- LOAD BACKBONE ----------------------------- #
#     if args.backbone == 'edgeface':
#         net = EdgeFaceBackbone(feature_dim=feature_dim)
#     else:
#         if args.dataset == 'facescrub':
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
#         else:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)

#     net = nn.DataParallel(net).to(device)

#     # ----------------------------- CHECKPOINT PATH ---------------------------- #
#     checkpoint_dir = (
#         '/kaggle/working/opqn-0210/checkpoint/'
#         if 'kaggle' in os.environ.get('PWD', '')
#         else 'checkpoint'
#     )
#     checkpoint_path = load_path if os.path.isabs(load_path) else os.path.join(checkpoint_dir, load_path)

#     if not os.path.exists(checkpoint_path):
#         raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint['backbone'])
#     mlp_weight = checkpoint.get('mlp', None)

#     net.eval()
#     len_word = feature_dim // num

#     # ----------------------------- FEATURE EXTRACTION ----------------------------- #
#     start_total = time.perf_counter()

#     with torch.no_grad():
#         # index + label của TRAIN
#         index, train_labels = compute_quant_indexing(
#             transform_test, train_loader, net, len_word, mlp_weight, device
#         )

#         # feature của TEST
#         query_features, test_labels = compute_quant(
#             transform_test, test_loader, net, device
#         )

#         # ----------------------------- COMPUTE mAP + TOP-K ----------------------------- #
#         start_map = time.perf_counter()

#         # 1) Tính mAP (top = full database)
#         mAP, _ = PqDistRet_Ortho(
#             query_features, test_labels,
#             train_labels, index,
#             mlp_weight, len_word, num,
#             device,
#             top=len(trainset)     # mAP cần so với toàn bộ tập train
#         )

#         map_time_ms = (time.perf_counter() - start_map) * 1000

#         # 2) Tính top-k từ 10 → 100
#         topk_dict = {}
#         for k in range(10, 101, 10):
#             _, acc_k = PqDistRet_Ortho(
#                 query_features, test_labels,
#                 train_labels, index,
#                 mlp_weight, len_word, num,
#                 device,
#                 top=k
#             )
#             topk_dict[k] = acc_k


#     total_time_ms = (time.perf_counter() - start_total) * 1000
#     avg_query_time = total_time_ms / len(testset)

#     # ----------------------------- OUTPUT ----------------------------- #
#     print(f"[Evaluate] mAP: {100 * mAP:.2f}%")
#     print(f"mAP compute time: {map_time_ms:.2f} ms "
#           f"({map_time_ms / len(testset):.4f} ms/image)")

#     for k, acc in topk_dict.items():
#         print(f"[Evaluate @ top-{k}] accuracy: {100 * acc:.2f}%")

#     print(f"Total time (feature + mAP + top-k): {total_time_ms:.2f} ms")
#     print(f"Average time per query: {avg_query_time:.4f} ms/query")

    
# def test(load_path, length, num, words, feature_dim):
#     len_bit = int(num * math.log(words, 2))
#     assert length == len_bit, "something went wrong with code length"
#     # top_list = torch.linspace(20, 300, 15).int().tolist()  # Khôi phục dòng này (commented trong gốc)

#     d = int(feature_dim / num)
#     matrix = torch.randn(d, d)
#     for k in range(d):
#         for j in range(d):
#             matrix[j, k] = math.cos((j+0.5)*k*math.pi/d)
#     matrix[:, 0] /= math.sqrt(2)    # divided by sqrt(2)
#     matrix /= math.sqrt(d/2)    # divided by sqrt(N/2)
#     code_books = torch.Tensor(num, d, words)
#     code_books[0] = matrix[:, :words]
#     for i in range(1, num):
#         code_books[i] = matrix @ code_books[i-1]

#     print("===============evaluation on model %s===============" % load_path)

#     if args.backbone == 'resnet':
#         if args.cross_dataset:
#             net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#         else:
#             if args.dataset in ["facescrub", "cfw", "youtube"]:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim, channel_max=512, size=4)
#             else:
#                 net = resnet20_pq(num_layers=20, feature_dim=feature_dim)
#     elif args.backbone == 'edgeface':
#         net = SphereNet20_pq(feature_dim=feature_dim)  # Giả sử SphereNet20_pq cho edgeface; chỉnh nếu cần

#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=False, num_workers=4)
#     test_loader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)
#     num_classes = len(trainset.classes)
#     num_classes_test = len(testset.classes)
#     print("number of train identities: ", num_classes)
#     print("number of test identities: ", num_classes_test)
#     print("number of training images: ", len(trainset))
#     print("number of test images: ", len(testset))
#     print("number of training batches per epoch:", len(train_loader))
#     print("number of testing batches per epoch:", len(test_loader))

#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     net = nn.DataParallel(net).to(device)

#     checkpoint_dir = '/kaggle/working/opqn-0210/checkpoint/' if 'kaggle' in os.environ.get('PWD', '') else 'checkpoint'
#     checkpoint = torch.load(os.path.join(checkpoint_dir, load_path))
#     net.load_state_dict(checkpoint['backbone'])
#     mlp_weight = checkpoint['mlp']
#     len_word = int(feature_dim / num)
#     net.eval()
#     with torch.no_grad():
#         index, train_labels = compute_quant_indexing(transform_test, train_loader, net, len_word, mlp_weight, device)
#         start = datetime.now()
#         query_features, test_labels = compute_quant(transform_test, test_loader, net, device)
#         if args.dataset != "vggface2":
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=5)
#         else:
#             mAP, top_k = PqDistRet_Ortho(query_features, test_labels, train_labels, index, mlp_weight, len_word, num, device, top=10)

#         time_elapsed = datetime.now() - start
#         print("Query completed in %d ms" % int(time_elapsed.total_seconds() * 1000))
#         print('[Evaluate Phase] MAP: %.2f%% top_k: %.2f%%' % (100. * float(mAP), 100. * float(top_k)))

if __name__ == "__main__":
    save_dir = 'log'
    if args.evaluate:
        if len(args.load) != len(args.num) or len(args.load) != len(args.len) or len(args.load) != len(args.words):
            print("Warning: Args lengths don't match. Adjusting to shortest length.")
            min_len = min(len(args.load), len(args.num), len(args.len), len(args.words))
            args.load = args.load[:min_len]
            args.num = args.num[:min_len]
            args.len = args.len[:min_len]
            args.words = args.words[:min_len]
        for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
            if args.cross_dataset:
                feature_dim = num_s * words_s
            else:
                if args.dataset != "vggface2":
                    if args.len[i] != 36:
                        feature_dim = 512
                    else:
                        feature_dim = 516
                else:
                    feature_dim = num_s * words_s
            test(args.load[i], args.len[i], num_s, words_s, feature_dim=feature_dim)
    else:
        if len(args.save) != len(args.num) or len(args.save) != len(args.len) or len(args.save) != len(args.words):
            print("Warning: Args lengths don't match. Adjusting to shortest length.")
            min_len = min(len(args.save), len(args.num), len(args.len), len(args.words))
            args.save = args.save[:min_len]
            args.num = args.num[:min_len]
            args.len = args.len[:min_len]
            args.words = args.words[:min_len]
        for i, (num_s, words_s) in enumerate(zip(args.num, args.words)):
            sys.stdout = Logger(os.path.join(save_dir,
                str(args.len[i]) + 'bits' + '_' + args.dataset + '_' + datetime.now().strftime('%m%d%H%M') + '.txt'))
            print("[Configuration] Training on dataset: %s\n  Len_bits: %d\n Batch_size: %d\n learning rate: %.3f\n num_books: %d\n num_words: %d" %
                  (args.dataset, args.len[i], args.bs, args.lr, num_s, words_s))
            print("HyperParams:\nmargin: %.3f\t miu: %.4f" % (args.margin, args.miu))
            if args.dataset != "vggface2":
                if args.len[i] != 36:
                    feature_dim = 512
                else:
                    feature_dim = 516
            else:
                feature_dim = num_s * words_s
            train(args.save[i], args.len[i], num_s, words_s, feature_dim=feature_dim)