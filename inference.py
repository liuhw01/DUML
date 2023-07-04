
import time
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import datetime
from DUML_inference import MSFER
from util import *
import warnings
import argparse
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")
best_acc = 0
print('Training time: ' + now.strftime("%m-%d %H:%M"))




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--pin_memory', type=str, default=False)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--checkpoint_load', type=str, default="./checkpoint/CK_P.pth")
    parser.add_argument('--source_root', type=str, default='./data')
    parser.add_argument('--source_label',  default=[
                          './data/label_multi/new_Aff_data_label_train.txt',
                          './data/label_multi/new_CK_data_label.txt',
                          './data/label_multi/new_RAF_data_label.txt',
                          './data/label_multi/new_JAFFE_label.txt',
                          #'./data/label_multi/new_NI_Oulu_label.txt',
                        #'./data/label_multi/new_FER2013_label_train.txt',
          ] )
    parser.add_argument('--target_root', type=str, default='./data')
    parser.add_argument('--target_label', type=str, default='./data/label_multi/new_CK_data_label.txt')
    parser.add_argument('--target_name', type=str, default='CK')
    parser.add_argument('--class_num', type=int, default=7)
    parser.add_argument('--domain_num', type=int, default=4)
    parser.add_argument('--nfeat', type=int, default=512)
    parser.add_argument('--lr', type=int, default=0.01)
    parser.add_argument('--momentum', type=int, default=0.9)
    parser.add_argument('--weight_decay', type=int, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=99)
    parser.add_argument('--iteration', type=int, default=60)

    opt = parser.parse_known_args()[0]
    checkpoint = torch.load(opt.checkpoint_load)
    model= MSFER(opt.class_num, opt.domain_num, opt.nfeat)
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    model.cuda()
    # source label

    return opt, model

if __name__ == '__main__':
    opt, model = parse_opt()
    optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    recorder = RecorderMeter(opt.iteration)

    # 数据集变换
    mytransform_train = transforms.Compose([transforms.Resize([224,224]),
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                      transforms.RandomGrayscale(p=0.2),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomAffine((-15, 15), translate=(0.05, 0.05), scale=(0.9, 1.05),
                                                              fillcolor=0),
                                      transforms.ToTensor()])  # transform [0,255] to [0,1]
    mytransform_test=transforms.Compose([transforms.Resize([224,224]), transforms.ToTensor()])  # transform [0,255] to [0,1]

    # 处理source label


    # 读取source数据集
    #source_loader=source_load(opt.source_root, opt.source_label, mytransform_train,kwargs)
    # 读取target数据集

    target_test_loader =target_load(opt.target_root, opt.target_label, mytransform_train,mytransform_test,opt)

    def vail (model,target_test_loader,opt):
        top1 = AverageMeter('Accuracy', ':6.3f')
        top1_1 = AverageMeter('Accuracy', ':6.3f')
        top1_2 = AverageMeter('Accuracy', ':6.3f')
        top2_1 = AverageMeter('Accuracy', ':6.3f')
        top2_2 = AverageMeter('Accuracy', ':6.3f')
        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            for i, (images, target, fn) in enumerate(target_test_loader):
                images = images.cuda()
                target = target.cuda()
                output,resul,poslud_wei= model(images)
                # measure accuracy and record loss
                acc1, _ = accuracy(output, target, topk=(1, 5))
                acc2, _ = accuracy(resul[0], target, topk=(1, 5))
                acc3, _ = accuracy(resul[1], target, topk=(1, 5))
                acc4, _ = accuracy(resul[2], target, topk=(1, 5))
                acc5, _ = accuracy(poslud_wei, target, topk=(1, 5))

                top1.update(acc1[0], target.size(0))
                top1_1.update(acc2[0], target.size(0))
                top1_2.update(acc3[0], target.size(0))
                top2_1.update(acc4[0], target.size(0))
                top2_2.update(acc5[0], target.size(0))

            print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1)+'* Accuracy1 {top1_1.avg:.3f}*'.format(top1_1=top1_1) + ' *Accuracy2 {top1_2.avg:.3f}*'.format(top1_2=top1_2)
                  +  ' * Accuracy3 {top2_1.avg:.3f}'.format(top2_1=top2_1)+'*Pseudo_label_weight  {top2_2.avg:.3f}'.format(top2_2=top2_2)

    vail(model,target_test_loader,opt)








