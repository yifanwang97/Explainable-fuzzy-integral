import argparse
import os, sys
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from data_loader import data_loader
from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate
import resnet
from aux_bn import MixBatchNorm2d
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.path.abspath('..')))
from project_dir import project_dir

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')


# parser.add_argument('--data',default='data/texture_biased_dataset/feature_images/texture', help='path to dataset')
parser.add_argument('--data',default='data/all_datasets/feature_images/shape', help='path to dataset')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='data/iLab/model/ori_resnet18/')
parser.add_argument('-a', '--arch', metavar='ARCH', default='data/all_datasets/model/shape_resnet18/')
parser.add_argument('-p', '--pretrained', type=bool, default=True)
parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size (default: 32)')
parser.add_argument('--resume', type=str, help='path to latest checkpoitn, (default: None)',
                    default='')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful to restarts)')

parser.add_argument('--epochs', default=50, type=int, help='numer of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=0, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true', help='use pin memory')
parser.add_argument('--print-freq', '-f', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate',dest='evaluate', action='store_true', help='evaluate model on validation set')

best_prec1 = 0.0

def load_resnet18():
    model = models.resnet18(pretrained=args.pretrained)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, class_num)
    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    return model.cuda()


def main():
    global args, best_prec1, class_num
    args = parser.parse_args(sys.argv[1:])
    argv = parser.parse_args(sys.argv[1:])
    for k, v in vars(argv).items():
        try:
            if '/' in v:
                if not os.path.exists(v):
                    exec('args.' + k + ' = os.path.join(project_dir, v)')
        except:
            pass
        print(k, eval('args.' + k))

    # Data loading
    train_loader, val_loader, test_loader = data_loader(args.data, args.batch_size, args.workers, args.pin_memory)
    class_num = len(train_loader.dataset.classes)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    os.makedirs(args.arch, exist_ok=True)

    model = load_resnet18()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if os.path.isfile(args.resume):
        optimizer.load_state_dict(torch.load(args.resume)['optimizer'])
    '''
    if args.evaluate:
        validate(train_loader2,val_loader, model, criterion,59)
        return
    '''
    best_epoch = 0
    best_valid_accuracy = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        if not args.evaluate:
            train(train_loader, model, criterion, optimizer, epoch, args.print_freq)
            temp_train_acc, temp_epoch = validate(train_loader, model, epoch, 'train')
            temp_valid_acc, temp_epoch = validate(val_loader, model, epoch, 'val')
            temp_test_acc, temp_epoch = validate(test_loader, model, epoch, 'test')
            if temp_valid_acc > best_valid_accuracy:
                best_valid_accuracy = temp_valid_acc
                test_accuracy_best_val = temp_test_acc
                best_epoch = temp_epoch

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, args.arch + str(epoch) + '.pth')
        else:
            validate(val_loader, model, epoch, 'val')
            break

    

    log = open(os.path.join(args.arch, 'log.txt'), 'a')
    log.write('\n')
    log.write('\n')
    log.write("best_epoch is "+str(best_epoch)+'.')
    log.write('\n')
    log.write("best_valid_accuracy is "+str(best_valid_accuracy)+'.')
    log.write('\n')
    log.write("test_accuracy is "+str(test_accuracy_best_val)+'.')
    log.write('\n')

        

def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time

        target = target.cuda()
        input = input.cuda()
        output = model(input)
        loss = criterion(output, target)

        prec1, _ = accuracy(output.data, target, topk=(1,1))
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), loss=losses, top1=top1))


def validate(val_loader, model, epoch, flag):
    model.eval()
    correct = [0] * (class_num + 1)
    total = [0] * (class_num + 1)
    acc = [0] * (class_num + 1)
    for (img,label) in val_loader:
        img = img.cuda()
        label = label.cuda()
        out = model(img)
        _, pre = torch.max(out.data, 1)
        total[0] += label.size(0)
        pre = pre.squeeze()
        correct[0] += (pre == label).sum().item()
        for i in range(class_num):
            tmp = (torch.ones(label.size())) * i
            tmp = tmp.cuda()
            tmp = tmp.long()
            total[i+1] += (tmp == label).sum().item()
            correct[i+1] += ((tmp == label)*(pre == label)).sum().item()

    for i in range(class_num + 1):
        try:
            acc[i] = correct[i]/total[i]
        except:
            acc[i] = 0
    print('{} accuracy: {}'.format(flag, correct[0] / total[0]))
    print(str(total))
    print(str(correct))
    

    log = open(os.path.join(args.arch, 'log.txt'), 'a')
    log.write("epoch "+str(epoch)+" in %s:\n"%flag)
    log.write(str(acc))
    log.write('\n')
    if flag == 'valid':
        log.write('\n')
    if flag == 'test':
        log.write('\n')
    log.close()
    return acc[0], epoch


if __name__ == '__main__':
    main()