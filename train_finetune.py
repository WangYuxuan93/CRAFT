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
from net.craft import CRAFT
import sys
from eval import copyStateDict, eval_net_finetune, eval_net
from utils.cal_loss import cal_fakeData_loss, cal_synthText_loss
from dataset.synthDataset import SynthDataset
from dataset.icdar2013_dataset import Icdar2013Dataset
from dataset.icdar2017_dataset import Icdar2017Dataset
from dataset.textdetect_dataset import TextDetectDataset
import argparse
import logging


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")
parser = argparse.ArgumentParser(description='CRAFT Train Fine-Tuning')
parser.add_argument('--gt_path', default='H:/Dataset/SynthText/SynthText/gt.mat', type=str, help='SynthText gt.mat')
parser.add_argument('--synth_dir', default='H:/Dataset/SynthText/SynthText', type=str, help='SynthText image dir')
parser.add_argument('--ic13_root', default='/home/brooklyn/ICDAR/icdar2013', type=str, help='icdar2013 data dir')
parser.add_argument('--ic17_root', default='data/ICDAR2017', type=str, help='icdar2017 data dir')
parser.add_argument('--td_root', default='data/char_lvl', type=str, help='Text detect data dir')
parser.add_argument('--data_type', default='td', type=str, help='data type (td, ic17)')
parser.add_argument('--label_size', default=96, type=int, help='target label size')
parser.add_argument('--batch_size', default=16, type=int, help='training data batch size')
parser.add_argument('--test_batch_size', default=16, type=int, help='training data batch size')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda to train model')
parser.add_argument('--pretrained_model', default='model/craft_mlt_25k.pth', type=str, help='pretrained model path')
parser.add_argument('--lr', default=3e-5, type=float, help='initial learning rate')
parser.add_argument('--epochs', default=20, type=int, help='training epochs')
parser.add_argument('--test_interval', default=40, type=int, help='test interval')
parser.add_argument('--log_file', default='training.log', type=str, help='Output training log')
parser.add_argument('--output_model_dir', default='finetune/', type=str, help='Output model directory')
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

def train(net, epochs, batch_size, test_batch_size, lr, test_interval, test_model_path, output_model_dir, save_weight=True, device="cpu",type="td"):
    logging.info("cuda: {}".format(args.cuda))
    logging.info("device: {}".format(device))
    logging.info(f"Number of available GPUs: {torch.cuda.device_count()}")
    #print ("cuda:", args.cuda)
    #print ("device:", device)
    
    if type == "synth":
        synth_data = SynthDataset(image_transform=image_transform,
                              label_transform=label_transform,
                              file_path=args.gt_path,
                              image_dir=args.synth_dir)
        train_loader = torch.utils.data.DataLoader(synth_data, batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(synth_data, batch_size=test_batch_size, shuffle=False)
        print('len train data:', len(synth_data))
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
    elif type == "td":
        td_train_data = TextDetectDataset(image_transform=image_transform,
                                    label_transform=label_transform,
                                    images_dir=os.path.join(args.td_root, 'train_images'),
                                    labels_dir=os.path.join(args.td_root, 'train_labels'))
        td_val_data = TextDetectDataset(image_transform=image_transform,
                                    label_transform=label_transform,
                                    images_dir=os.path.join(args.td_root, 'valid_images'),
                                    labels_dir=os.path.join(args.td_root, 'valid_labels'))
        train_loader = torch.utils.data.DataLoader(td_train_data, batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(td_val_data, batch_size=test_batch_size, shuffle=False)
        logging.info('##### Data Type: Text Detection, Data Number: train: {}, valid: {}'.format(len(td_train_data), len(td_val_data)))

    steps_per_epoch = 100

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
    optimizer = optim.Adam(net.parameters(), lr)

    for epoch in range(epochs):
        print('epoch = ', epoch)
        for i, (images, labels_region, labels_affinity, sc_map) in enumerate(train_loader):

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
            if i % 10 == 0:
                #print('i = ', i,': loss = ', loss.item())
                logging.info(f'i = {i}: loss = {loss.item()}')

            if i != 0 and i % test_interval == 0:
                #test_loss = eval_net_finetune(net, val_loader, criterion, device)
                test_loss = eval_net(net, val_loader, criterion, device)
                model_save_path = os.path.join(output_model_dir, 'finetuned_epoch_' + str(epoch) + '_iter' + str(i) + '.pth')
                logging.info(f'Evaluating Valid Set: i = {i}, test_loss = {test_loss}, lr = {lr}, Saving model to {model_save_path}')
                if save_weight:
                    torch.save(net.state_dict(), model_save_path)

if __name__ == "__main__":

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    epochs = args.epochs  # 遍历数据集次数
    lr = args.lr  # 学习率
    test_interval = args.test_interval #测试间隔
    pretrained_model = args.pretrained_model #预训练模型
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    net = CRAFT(pretrained=False)  # craft模型

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
    if not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)
    try:
        train(net=net,
              epochs=epochs,
              batch_size=batch_size,
              test_batch_size=test_batch_size,
              lr=lr,test_interval=test_interval,
              test_model_path=pretrained_model,
              output_model_dir=args.output_model_dir,
              device=device,
              type=args.data_type)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
