from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
from PIL import Image
import torch






def random_choose_raf(label_path):
    file = open(label_path)
    lines = file.readlines()
    train_label=[]
    test_label=[]
    for i in range(len(lines)):
        if lines[i][0:3]=='tra':
            num=int(lines[i][-2])
            s1=list(lines[i])
            s1[-2]=str(num)
            s=''.join(s1)
            train_label.append(s)
        if lines[i][0:3]=='tes':
            num = int(lines[i][-2])
            s1 = list(lines[i])
            s1[-2] = str(num)
            s = ''.join(s1)
            test_label.append(s)

    return train_label, test_label  # output the list and delvery it into ImageFolder

def choose_source_data(label_path):
    random.seed(1)
    file = open(label_path)
    lines = file.readlines()
    slice_initial = random.sample(lines, len(lines))  # if don't change this ,it will be all the same

    return slice_initial  # output the list and delvery it into ImageFolder


def choose_multi_source_data(label_path):
    random.seed(1)
    file = open(label_path)
    lines = file.readlines()
    num_1=[]
    for i in range(len(lines)):
        num_1.append(int(lines[i].split()[1]))
    max_num=np.array(num_1).max()
    max_num=7 if max_num>6 else max_num+1
    num=[[] for _ in range(max_num)]
    cls_all=[]
    for i in range(len(lines)):
        cls=lines[i].split()
        if int(cls[1])<max_num:
            num[int(cls[1])].append(i)
            cls_all.append(cls)
    num_train=[[] for _ in range(max_num)]
    for i in range(max_num):
        random.shuffle(num[i])
        num_train[i] = num[i][0:5000] if len(num[i])>5000 else num[i]
    choose_list=[lines[num_train[i][j]] for i in range(max_num) for j in range(len(num_train[i]))]

    return choose_list  # output the list and delvery it into ImageFolder

#
def target_load(target_root, target_label, mytransform_train,mytransform_test,opt):
    if opt.target_name=='RAF':
        train_label_load,target_label_load=random_choose_raf(target_label)
    else:

        target_label_load = choose_source_data(target_label)


    target_test=myImageFloder(root=target_root, label=target_label_load, transform=mytransform_test)

    target_test_loader = torch.utils.data.DataLoader(target_test, batch_size=opt.batch_size, shuffle=True,
                                         num_workers=opt.num_workers, pin_memory=opt.pin_memory)


    return  target_test_loader




def source_load (source_root, source_label, mytransform_train,kwargs):

    source_loader=list()
    for i in range(len(source_label)):
        print(i)
        label_load=choose_multi_source_data(source_label[i])
        source_data = myImageFloder(root=source_root, label=label_load, transform=mytransform_train)
        source = torch.utils.data.DataLoader(source_data, batch_size=kwargs['batch_size'], shuffle=True,
                                                   num_workers=kwargs['num_workers'], pin_memory=kwargs['pin_memory'])


        source_loader.append(source)

    return source_loader



def default_loader(path):
    return Image.open(path).convert('RGB')  # operation object is the PIL image object


class myImageFloder(datasets.ImageFolder):  # Class inheritance，
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
         # fh = open(label)
        c = 0
        imgs = []
        class_names = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        #class_names = ['anger','disgust','fear','happy','neural','sad','surprise']
        for line in label:  # label is a list
            cls = line.split()  # cls is a lis
            fn = cls.pop(0)
            la=int(cls[0])
            if os.path.isfile(os.path.join(fn)):
                #imgs.append((fn, tuple([float(v) for v in cls[:len(cls)]])))
                imgs.append((fn, la))
                # access the last label
                # images is the list,and the content is the tuple, every image corresponds to a label
                # despite the label's dimension
                # we can use the append way to append the element for list
            c = c + 1
        print('the total image is',c)
        print(class_names)
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        fn, label = self.imgs[index]  # even though the imgs is just a list, it can return the elements of it
        # in a proper way
        img = self.loader(os.path.join(fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label), fn


    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes







class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)





def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res





class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        #self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        #y_axis[:] = self.epoch_losses[:, 1]
        #plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        #plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)

def save_checkpoint(state, is_best, kwargs):
    torch.save(state, kwargs['checkpoint_path'])
    if is_best:
        shutil.copyfile(kwargs['checkpoint_path'], kwargs['best_checkpoint_path'])


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch,time_str):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

