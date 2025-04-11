"""
Author: brooklyn

train with weak datasets like ICDAR2013
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os
import re
from net.craft import CRAFT
import sys
import glob
from eval import copyStateDict, eval_net_finetune, eval_net
from utils.cal_loss import cal_fakeData_loss, cal_synthText_loss
from dataset.synthDataset import SynthDataset
from dataset.icdar2013_dataset import Icdar2013Dataset
from dataset.icdar2017_dataset import Icdar2017Dataset
from dataset.textdetect_dataset import TextDetectDataset
import argparse
import logging
from predict import inference
from evaluation2 import eval_text_detection


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")
parser = argparse.ArgumentParser(description='CRAFT Train Fine-Tuning')
parser.add_argument('--gt_path', default='H:/Dataset/SynthText/SynthText/gt.mat', type=str, help='SynthText gt.mat')
parser.add_argument('--synth_dir', default='H:/Dataset/SynthText/SynthText', type=str, help='SynthText image dir')
parser.add_argument('--ic13_root', default='/home/brooklyn/ICDAR/icdar2013', type=str, help='icdar2013 data dir')
parser.add_argument('--ic17_root', default='data/ICDAR2017', type=str, help='icdar2017 data dir')
parser.add_argument('--td_root', default='data/char_lvl', type=str, help='Text detect data dir')
parser.add_argument('--test_folder', default='data/char_lvl', type=str, help='Test data dir')
parser.add_argument('--eval_iou', default=False, type=str2bool, help='Use iou in valid set')
parser.add_argument('--data_type', default='td', type=str, help='data type (td, ic17)')
parser.add_argument('--label_size', default=384, type=int, help='target label size')
parser.add_argument('--batch_size', default=16, type=int, help='training data batch size')
parser.add_argument('--test_batch_size', default=16, type=int, help='training data batch size')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda to train model')
parser.add_argument('--pretrained_model', default='model/craft_mlt_25k.pth', type=str, help='pretrained model path')
parser.add_argument('--vgg_path', default='model/vgg16_bn-6c64b313.pth', type=str, help='pretrained vgg model path')
parser.add_argument('--from_scratch', default=False, type=str2bool, help='Train from scratch')
parser.add_argument('--lr', default=3e-5, type=float, help='initial learning rate')
parser.add_argument('--gamma', default=0.8, type=float, help='gamma for learning rate step scheduler')
parser.add_argument('--step_size', default=10000, type=int, help='decay step size for learning rate step scheduler')
parser.add_argument('--epochs', default=20, type=int, help='training epochs')
parser.add_argument('--test_interval', default=40, type=int, help='test interval')
parser.add_argument('--log_file', default='training.log', type=str, help='Output training log')
parser.add_argument('--output_model_dir', default='finetune/', type=str, help='Output model directory')
parser.add_argument('--num_workers', default=16, type=int, help='number of data loading cpu workers')
args = parser.parse_args()



# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(args.log_file)  # 输出到文件
    ]
)

image_transform = transforms.Compose([
    transforms.Resize((args.label_size*2,args.label_size*2)),
    transforms.ToTensor()
])
label_transform = transforms.Compose([
    transforms.Resize((args.label_size,args.label_size)),
    transforms.ToTensor()
])

# Saving function - Save model weights and additional params
def save_model(epoch, iter_num, model_save_path, optimizer_state_dict, scheduler_state_dict):
    output_model_dir = os.path.dirname(model_save_path)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    model_state = {
        'epoch': epoch,
        'iteration': iter_num,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict
    }
    torch.save(model_state, model_save_path)
    logging.info(f'Model saved at {model_save_path}')

# Load function - to load model with saved hyperparameters
"""
def load_model(model_path, net, optimizer, scheduler, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Load scheduler state
    epoch = checkpoint['epoch']
    iteration = checkpoint['iteration']
    logging.info(f"Loaded model from {model_path}, epoch {epoch}, iteration {iteration}")
    return net, optimizer, scheduler, epoch, iteration
"""

def train(net, epochs, batch_size, test_batch_size, lr, test_interval, test_model_path, output_model_dir, save_weight=True, device="cpu",type="td", optimizer=None, scheduler=None, start_epoch=0, start_iter=0):
    logging.info("cuda: {}".format(args.cuda))
    logging.info("device: {}".format(device))
    logging.info(f"Number of available GPUs: {torch.cuda.device_count()}")
    logging.info("Image resize shape: ({} x {})".format(args.label_size*2, args.label_size*2))
    logging.info(f"Using StepLR: step_size={scheduler.step_size}, gamma={scheduler.gamma}")
    logging.info(f"Initial learning rate: {scheduler.get_last_lr()[0]}")
    logging.info('Batch size: train: {}, valid: {}'.format(batch_size, test_batch_size))
    logging.info("Test interval: {}".format(test_interval))
    logging.info("Total training epochs: {}".format(epochs))
    logging.info("Start epoch: {}, start iter: {}".format(start_epoch, start_iter))
    
    #print ("cuda:", args.cuda)
    #print ("device:", device)
    
    if type == "synth":
        synth_data = SynthDataset(image_transform=image_transform,
                              label_transform=label_transform,
                              file_path=args.gt_path,
                              image_dir=args.synth_dir)
        train_loader = torch.utils.data.DataLoader(synth_data, batch_size, shuffle=True)
        td_val_data = TextDetectDataset(image_transform=image_transform,
                                    label_transform=label_transform,
                                    images_dir=os.path.join(args.td_root, 'valid_images'),
                                    labels_dir=os.path.join(args.td_root, 'valid_labels'))
        val_loader = torch.utils.data.DataLoader(td_val_data, batch_size=test_batch_size, shuffle=False)
        logging.info('##### Data Type: SynthText, Data Number: train: {}, valid (Text Detection): {}'.format(len(synth_data), len(td_val_data)))
        iters_per_epoch = len(synth_data) // batch_size
        logging.info('Number of iters per epoch: {}'.format(iters_per_epoch))
        logging.info('Total iters: {}'.format(iters_per_epoch * epochs))
    elif type == "ic13":
        ic13_data = Icdar2013Dataset(cuda=args.cuda,
                                    image_transform=image_transform,
                                    label_transform=label_transform,
                                    model_path=test_model_path,
                                    images_dir=os.path.join(args.ic13_root, 'train_images'),
                                    labels_dir=os.path.join(args.ic13_root, 'train_labels'))
        ic13_length = len(ic13_data)
        train_loader = torch.utils.data.DataLoader(ic13_data, batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(ic13_data, batch_size=test_batch_size, shuffle=False)
        print('len train data:', len(ic13_data))
    elif type == "ic17":
        ic17_train_data = Icdar2017Dataset(cuda=args.cuda,
                                    image_transform=image_transform,
                                    label_transform=label_transform,
                                    model_path=test_model_path,
                                    images_dir=os.path.join(args.ic17_root, 'train_images'),
                                    labels_dir=os.path.join(args.ic17_root, 'train_labels'))
        ic17_val_data = Icdar2017Dataset(cuda=args.cuda,
                                    image_transform=image_transform,
                                    label_transform=label_transform,
                                    model_path=test_model_path,
                                    images_dir=os.path.join(args.ic17_root, 'valid_images'),
                                    labels_dir=os.path.join(args.ic17_root, 'valid_labels'))
        train_loader = torch.utils.data.DataLoader(ic17_train_data, batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(ic17_val_data, batch_size=test_batch_size, shuffle=False)
        logging.info('##### Data Type: ICDAR17, Data Number: train: {}, valid: {}'.format(len(ic17_train_data), len(ic17_val_data)))
        iters_per_epoch = len(ic17_train_data) // batch_size
        logging.info('Number of iters per epoch: {}'.format(iters_per_epoch))
        logging.info('Total iters: {}'.format(iters_per_epoch * epochs))
    elif type == "td":
        td_train_data = TextDetectDataset(image_transform=image_transform,
                                    label_transform=label_transform,
                                    images_dir=os.path.join(args.td_root, 'train_images'),
                                    labels_dir=os.path.join(args.td_root, 'train_labels'))
        td_val_data = TextDetectDataset(image_transform=image_transform,
                                    label_transform=label_transform,
                                    images_dir=os.path.join(args.td_root, 'valid_images'),
                                    labels_dir=os.path.join(args.td_root, 'valid_labels'))
        train_loader = torch.utils.data.DataLoader(td_train_data, batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.cuda)
        val_loader = torch.utils.data.DataLoader(td_val_data, batch_size=test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.cuda)
        logging.info('##### Data Type: Text Detection, Data Number: train: {}, valid: {}'.format(len(td_train_data), len(td_val_data)))
        iters_per_epoch = len(td_train_data) // batch_size
        logging.info('Number of iters per epoch: {}'.format(iters_per_epoch))
        logging.info('Total iters: {}'.format(iters_per_epoch * epochs))

    """
    synth_data = SynthDataset(image_transform=image_transform,
                              label_transform=label_transform,
                              file_path=args.gt_path,
                              image_dir=args.synth_dir)
    #弱数据集与强数据集比例1：5
    synth_data = torch.utils.data.Subset(synth_data, range(5*ic13_length))

    #合并弱数据集和强数据集
    fine_tune_data = torch.utils.data.ConcatDataset([synth_data, ic13_data])
    train_data, val_data = torch.utils.data.random_split(fine_tune_data, [5*ic13_length, ic13_length])
    """

    
    criterion = nn.MSELoss(reduction='none')
    #if optimizer is None:
    #    optimizer = optim.Adam(net.parameters(), lr)

    for epoch in range(start_epoch, epochs):
        print('epoch = ', epoch)
        for i, (images, labels_region, labels_affinity, sc_map) in enumerate(train_loader, start=start_iter):

            images = images.to(device)
            labels_region = labels_region.to(device)
            labels_affinity = labels_affinity.to(device)
            sc_map = sc_map.to(device)
            labels_region = torch.squeeze(labels_region, 1)
            labels_affinity = torch.squeeze(labels_affinity, 1)

            #前向传播
            y, _ = net(images)
            score_text = y[:, :, :, 0]
            score_link = y[:, :, :, 1]
            sc_map = torch.squeeze(sc_map, 1)
            #强弱数据集分别计算损失
            #if sc_map.size() == labels_region.size():
            if type in ["ic13", "ic17"]:
                loss = cal_fakeData_loss(criterion, score_text, score_link, labels_region, labels_affinity, sc_map,
                                         device)
                #print ("fake loss")
            else:
                loss = cal_synthText_loss(criterion, score_text, score_link, labels_region, labels_affinity, device)
                #print ("synth loss")

            #back propagation
            optimizer.zero_grad()  #梯度清零
            loss.backward()  #计算梯度
            optimizer.step() #更新权重
            scheduler.step()
            if i % 10 == 0:
                #print('i = ', i,': loss = ', loss.item())
                logging.info(f'i = {i}: loss = {loss.item()}')
            
            if i % 1000 == 0:  # 每 1000 次迭代打印一次学习率
                logging.info(f'Epoch {epoch}, Iteration {i}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}')

            if (epoch == 0 and i == 0) or (i != 0 and i % test_interval) == 0:
                #test_loss = eval_net_finetune(net, val_loader, criterion, device)
                test_loss = eval_net(net, val_loader, criterion, device)
                model_save_path = os.path.join(output_model_dir, 'finetuned_epoch_' + str(epoch) + '_iter' + str(i) + '.pth')
                logging.info(f'Evaluating Valid Set: i = {i}, test_loss = {test_loss}, lr = {scheduler.get_last_lr()[0]}')
                if args.eval_iou:
                    net.eval()
                    gold_bbox_list, pred_bbox_list = inference(net, args.test_folder, cuda=args.cuda)
                    results = eval_text_detection(gold_bbox_list, pred_bbox_list, iou_threshold=0.5)
                    net.train()
                    logging.info(f'Evaluating Valid Set (IoU): i = {i}, IoU = {results["iou"]}, acc = {results["acc"]}')
                if save_weight:
                    #torch.save(net.state_dict(), model_save_path)
                    save_model(epoch, i, model_save_path, optimizer.state_dict(), scheduler.state_dict())
        # set start iter back to 0 after 1 epoch
        start_iter = 0

def load_model(model_path, net, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)
    # 去掉 "module." 前缀
    new_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # 去掉 "module."
        else:
            new_state_dict[k] = v

    # 加载去掉 "module." 的 state_dict
    net.load_state_dict(new_state_dict)
    return net


def load_latest_model(output_model_dir, net, optimizer, scheduler, device):
    # 获取output_model_dir下所有的模型文件（.pth）
    #model_files = glob.glob(os.path.join(output_model_dir, "finetuned_epoch_*_iter_*.pth"))
    model_files = glob.glob(os.path.join(output_model_dir, "*.pth"))
    print ("model_files:", model_files)
    if not model_files:
        return net, optimizer, scheduler, 0, 0  # 如果没有模型文件，返回初始状态
    
    # 提取epoch和iter信息，按epoch和iter排序
    def extract_epoch_iter(model_path):
        match = re.search(r'finetuned_epoch_(\d+)_iter(\d+)', model_path)
        if match:
            epoch = int(match.group(1))
            iter_num = int(match.group(2))
            return epoch, iter_num
        return 0, 0  # 如果无法提取epoch和iter，默认返回0
    
    # 按照(epoch, iter)元组排序，选择最新的模型
    latest_model = max(model_files, key=lambda x: extract_epoch_iter(x))

    logging.info(f"Loading the latest model from: {latest_model}")
    
    # 加载模型和optimizer状态
    checkpoint = torch.load(latest_model, map_location=device)
    net = load_model(latest_model, net, device=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    iter_num = checkpoint['iteration'] + 1 # should start from the next iter
    
    return net, optimizer, scheduler, epoch, iter_num



if __name__ == "__main__":

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs  # 遍历数据集次数
    lr = args.lr  # 学习率
    test_interval = args.test_interval #测试间隔
    pretrained_model = args.pretrained_model #预训练模型
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    if args.from_scratch:
        net = CRAFT(pretrained=True, vgg_path=args.vgg_path)  # craft模型
    else:
        net = CRAFT(pretrained=False)

    optimizer = optim.Adam(net.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    epoch, iter_num = 0, 0
    if args.from_scratch:
        # Check if there are saved models in the output directory
        if os.path.exists(args.output_model_dir):
            net, optimizer, scheduler, epoch, iter_num = load_latest_model(args.output_model_dir, net, optimizer, scheduler, device)
            logging.info(f'Resuming from epoch {epoch}, iteration {iter_num}')
        else:
            logging.info(f'Training from scratch')
            #epoch, iter_num = 0, 0  # Start from the beginning
    else:
        if os.path.exists(args.output_model_dir):
            net, optimizer, scheduler, epoch, iter_num = load_latest_model(args.output_model_dir, net, optimizer, scheduler, device)
            logging.info(f'Resuming from epoch {epoch}, iteration {iter_num}')
        else:
            logging.info(f'Loading pretrained params from: {pretrained_model}')
            checkpoint = torch.load(pretrained_model)
            if 'model_state_dict' in checkpoint:
                if args.cuda:
                    net = load_model(pretrained_model, net, device=device)
                else:
                    net = load_model(pretrained_model, net, device=device)
            else:
                if args.cuda:
                    net.load_state_dict(copyStateDict(torch.load(pretrained_model)))
                else:
                    net.load_state_dict(copyStateDict(torch.load(pretrained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net = net.to(device)
    net.train()
    #model_save_prefix = 'finetune/craft_finetune_'
    #model_save_prefix = os.path.join(args.output_model_dir, 'craft_finetune_')
    #try:
    train(net=net,
              epochs=epochs,
              batch_size=batch_size,
              test_batch_size=test_batch_size,
              lr=lr,test_interval=test_interval,
              test_model_path=pretrained_model,
              output_model_dir=args.output_model_dir,
              device=device,
              type=args.data_type,
              optimizer=optimizer,
              scheduler=scheduler,
              start_epoch=epoch,  # Start from the loaded epoch
              start_iter=iter_num)  # Start from the loaded iteration
    #except KeyboardInterrupt:
    #    torch.save(net.state_dict(), 'INTERRUPTED.pth')
    #    print('Saved interrupt')
    #    try:
    #        sys.exit(0)
    #    except SystemExit:
    #        os._exit(0)
