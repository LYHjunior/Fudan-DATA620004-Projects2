# packge import 
import torch
from configs.config_task1 import get_cfg_defaults
from data.argument_type import Mixup,Cutmix
from data.dataset import load_cifar_dataset
from task1_train import prepare_config
from configs.config_task1 import get_cfg_defaults
from model.resnet import ResNet18
from torchmetrics.functional import accuracy, precision,recall,f1_score
from tqdm import tqdm

cfg = get_cfg_defaults()
cfg.TRAIN.batch_size = 128

# load data

train_dataset,test_dataset,num_classes = load_cifar_dataset(cfg)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=cfg.TRAIN.batch_size,
                                            shuffle=False,
                                            pin_memory=False,
                                            num_workers=8)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=cfg.TRAIN.batch_size,
                                            shuffle=False,
                                            pin_memory=False,
                                            num_workers=8)


model = ResNet18(num_classes=num_classes)

# baseline
model.load_state_dict(torch.load('./trained_model/resnet18_(False, False, False, False)/best_model_new.pt'))
model = model.to('cuda:0')

# test
preds = []
targets = []
for images, labels in tqdm(test_loader):
    images = images.to('cuda:0')
    labels = labels.to('cuda:0')

    with torch.no_grad():
        pred = model(images)

    pred = torch.max(pred.data, 1)[1]
    preds.append(pred)
    targets.append(labels)
preds = torch.cat(preds)
targets = torch.cat(targets)
acc = accuracy(preds,targets,task="multiclass", num_classes=num_classes)
pre = precision(preds, targets, average='macro',task="multiclass", num_classes=num_classes)
recall = recall(preds, targets, average='macro',task="multiclass", num_classes=num_classes)
f1 = f1_score(preds,targets,task="multiclass",num_classes = num_classes)
print(acc.item(),pre.item(),recall.item(),f1.item())
