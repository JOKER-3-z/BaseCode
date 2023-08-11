import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torchvision import transforms,datasets

import os
import argparse
from tqdm import tqdm

from model.resnet18 import Resnet18
from utils.proce import AverageMeter,accuracy

parser=argparse.ArgumentParser()
parser.add_argument("--datapath",default='./dataset',type=str,help='epochs')
parser.add_argument("--resume",default=None,type=str,help='loading pretrained model')
parser.add_argument("--eval",default=False,type=bool,help='use the eval first')
parser.add_argument("--model_name",default="resnet",type=str,help='model_name to save')
parser.add_argument("--num_class",default=7,type=int,help='how many classes need to output')
parser.add_argument("--lr",default=2e-4,type=float,help='learning rate')
parser.add_argument("--bs",default=256,type=int,help='batch_size')
parser.add_argument("--epochs",default=50,type=int,help='epochs')

args=parser.parse_args()
DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model,train_loader,losses,optimizer):
    model.train()

    loss_recorder=AverageMeter()
    acc_recorder=AverageMeter()
    for i,(image,target) in enumerate(train_loader):
        optimizer.zero_grad()
        input=image.to(DEVICE)
        target=target.to(DEVICE)
        out=model(input)
        loss=losses(out,target)

        loss_recorder.update(loss,n=input.shape[0])
        acc = accuracy(out, target)
        #print(acc)
        acc_recorder.update(acc[0],n=input.shape[0])

        loss.backward()
        optimizer.step()
    loss=loss_recorder.avg
    acc=acc_recorder.avg

    return loss,acc

def trainer(model,train_loader,val_loader,losses,optimizer,scheduler,epochs):
    best_acc=-1
    model_name=args.model_name

    if not os.path.exists("./save_point"):
        os.mkdir("./save_point")
    for epoch in range(epochs+1):
        train_loss, train_acc = train_one_epoch(model,train_loader,losses,optimizer)
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, train_acc, train_loss,optimizer.param_groups[0]['lr']))
        val_loss,val_acc=eval(model,val_loader,losses)
        tqdm.write('[Epoch %d] eval accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, val_acc, val_loss,optimizer.param_groups[0]['lr']))
        if val_acc > best_acc:
            best_acc=val_acc
            pthname='epochs:{}_{:.3f}_{}.pth'.format(epoch+1,best_acc.item(),model_name)
            torch.save(model.state_dict(),os.path.join('./save_point',pthname))
        tqdm.write('Best acc : %.3f'%(best_acc))
        scheduler.step()

def eval(model,val_loader,losses):
    model.eval()
    loss_recoder=AverageMeter()
    acc_recoder=AverageMeter()
    with torch.no_grad():
        for i,(image,target) in enumerate(val_loader):
            input=image.to(DEVICE)
            target=target.to(DEVICE)
            out=model(input)
            loss_recoder.update(losses(out,target).item(),n=input.shape[0])
            acc_recoder.update(accuracy(out,target)[0],n=input.shape[0])
        loss=loss_recoder.avg
        acc=acc_recoder.avg
    return loss,acc

    

def main(forzen_method=None):
    #超参数定义
    batch_size=args.bs
    num_class=args.num_class
    lr=args.lr
    epochs=args.epochs
    datapath=args.datapath
    pretrained_model=args.resume
    Iseval=args.eval
    #dataloader
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32)
            ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25)),
        ])
    
    train_set=datasets.ImageFolder(root=os.path.join(datapath,'train'),transform=data_transforms)
    val_set=datasets.ImageFolder(root=os.path.join(datapath,'val'),transform=data_transforms)

    train_loader=data.DataLoader(train_set,batch_size=batch_size,num_workers=4,shuffle=True, pin_memory=True)
    val_loader=data.DataLoader(val_set,batch_size=batch_size,num_workers=4,shuffle=True, pin_memory=True)

    #model define
    model=Resnet18(num_class=num_class)
    if pretrained_model is not None:
        model.load_state_dict(torch.load(pretrained_model),strict=False)
        print("load pretrained weights : "+pretrained_model+" finished")
    if forzen_method is not None:
        model=forzen_method(model)
    #optimizer,loss
    criterion_cls = nn.CrossEntropyLoss()

    params=list(model.parameters())
    optimizer=optim.SGD(
        params,
        lr,
        weight_decay = 1e-4,
        momentum=0.9
    )
    scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(DEVICE)

    if Iseval:
        val_loss,val_acc=eval(model,val_loader,criterion_cls)
        tqdm.write('[initial weights] eval accuracy: %.4f. Loss: %.3f. LR %.6f' % (val_acc, val_loss,optimizer.param_groups[0]['lr']))
    trainer(model,train_loader,val_loader,criterion_cls,optimizer,scheduler,epochs)


if __name__ == "__main__":
    main()