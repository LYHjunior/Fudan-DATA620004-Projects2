from model.fcos import FCOSDetector
import torch
from dataset.VOC_dataset import VOCDataset
from utils.eval import *
import math, time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0', help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
transform = Transforms()
train_dataset = VOCDataset(root_dir='.dataset/VOCdevkit/VOC2007',resize_size=[800,1333],
                           split='trainval',use_difficult=False,is_train=True,augment=transform)

model = FCOSDetector(mode="training").cuda()
model = torch.nn.DataParallel(model)
# model.load_state_dict(torch.load('/mnt/cephfs_new_wj/vc/zhangzhenghao/FCOS.Pytorch/output1/model_6.pth'))

BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs
#WARMPUP_STEPS_RATIO = 0.12
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_dataset.collate_fn,
                                           num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))
print("total_images : {}".format(len(train_dataset)))
steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMPUP_STEPS = 501

GLOBAL_STEPS = 1
# LR_INIT = 2e-3
# LR_END = 2e-5
LR_INIT = 1e-3
optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=1e-5)

if not os.path.exists('./checkpoint'):
    os.mkdir('./checkpoint')


writer_dict = {
        'writer': SummaryWriter(log_dir='./'),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

# def lr_func():
#      if GLOBAL_STEPS < WARMPUP_STEPS:
#          lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
#      else:
#          lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
#              (1 + math.cos((GLOBAL_STEPS - WARMPUP_STEPS) / (TOTAL_STEPS - WARMPUP_STEPS) * math.pi))
#          )
#      return float(lr)


def eval(epoch, writer=None):
    eval_dataset = VOCDataset(root_dir='.dataset/VOCdevkit/VOC2007', resize_size=[800, 1333],
                               split='test', use_difficult=False, is_train=False, augment=None)
    print("INFO===>eval dataset has %d imgs"%len(eval_dataset))
    eval_loader=torch.utils.data.DataLoader(eval_dataset,batch_size=1,shuffle=False,collate_fn=eval_dataset.collate_fn)

    model=FCOSDetector(mode="inference")
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./checkpoint/model_" + str(epoch+1) + ".pth",map_location=torch.device('cpu')))
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")
    model=model.cuda().eval()
    print("===>success loading model")

    gt_boxes=[]
    gt_classes=[]
    pred_boxes=[]
    pred_classes=[]
    pred_scores=[]
    num=0
    for img,boxes,classes in eval_loader:
        with torch.no_grad():
            out=model([img.cuda(), boxes.cuda(), classes.cuda()])
            pred_boxes.append(out[2][0].cpu().numpy())
            pred_classes.append(out[1][0].cpu().numpy())
            pred_scores.append(out[0][0].cpu().numpy())
            losses = out[-1]
        gt_boxes.append(boxes[0].numpy())
        gt_classes.append(classes[0].numpy())
        num+=1
        print(num,end='\r')

    pred_boxes,pred_classes,pred_scores=sort_by_score(pred_boxes,pred_classes,pred_scores)
    all_AP, all_acc, all_mIoU=eval_ap_2d(gt_boxes,gt_classes,pred_boxes,pred_classes,pred_scores,0.5,len(eval_dataset.CLASSES_NAME))
    print("all classes AP=====>\n")
    for key,value in all_AP.items():
        print('ap for {} is {}'.format(eval_dataset.id2name[int(key)],value))
    mAP=0.
    acc=0.
    mIoU=0.
    for class_id in all_AP.keys():
        class_mAP = all_AP[class_id]
        mAP+=float(class_mAP)
        class_acc = all_acc[class_id]
        # print(class_acc)
        acc+=float(class_acc)
        class_mIoU = all_mIoU[class_id]
        # print(class_mIoU)
        mIoU+=float(class_mIoU)
    mAP/=(len(eval_dataset.CLASSES_NAME)-1)
    acc/=(len(eval_dataset.CLASSES_NAME)-1)
    mIoU/=(len(eval_dataset.CLASSES_NAME)-1)
    print("mAP=====>%.3f\n"%mAP)
    if writer:
        writer.add_scalar('loss/test_cls_loss', losses[0].mean(), epoch)
        writer.add_scalar('loss/test_cnt_loss', losses[1].mean(), epoch)
        writer.add_scalar('loss/test_reg_loss', losses[2].mean(), epoch)
        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('Acc', acc, epoch)
        writer.add_scalar('mIoU', mIoU, epoch)


for epoch in range(EPOCHS):
    model.train()
    for epoch_step, data in enumerate(train_loader):

        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()

        #lr = lr_func()
        if GLOBAL_STEPS < WARMPUP_STEPS:
           lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
           for param in optimizer.param_groups:
               param['lr'] = lr
        if GLOBAL_STEPS == 40001:
           lr = LR_INIT * 0.1
           for param in optimizer.param_groups:
               param['lr'] = lr
        if GLOBAL_STEPS == 60001:
           lr = LR_INIT * 0.1
           for param in optimizer.param_groups:
              param['lr'] = lr
        start_time = time.time()

        optimizer.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        loss.mean().backward()
        optimizer.step()

        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)
        print(
            "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
            (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
             losses[2].mean(), cost_time, lr, loss.mean()))

        GLOBAL_STEPS += 1
    writer = writer_dict['writer']
    writer.add_scalar('loss/train_cls_loss', losses[0].mean(), epoch)
    writer.add_scalar('loss/train_cnt_loss', losses[1].mean(), epoch)
    writer.add_scalar('loss/train_reg_loss', losses[2].mean(), epoch)

    torch.save(model.state_dict(),
               "./checkpoint/model.pth")

    eval(epoch, writer)











