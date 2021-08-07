# -*- coding: utf-8 -*-
# @File    : main.py

import ast
import os
import sys

cwd = os.getcwd()
root = os.path.split(os.path.split(cwd)[0])[0]
sys.path.append(root)

from tqdm import tqdm
from torch.utils import data
import torchvision.transforms as transform
from encoding import utils as utils
from encoding.models import get_segmentation_model
from encoding.dataset import get_segmentation_dataset
from encoding.models.criterion import *
import argparse
import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def logger(file='/log.txt', str=None):
    if not os.path.exists(os.path.split(file)[0]):
        os.makedirs(os.path.split(file)[0])
    with open(file, mode='a', encoding='utf-8') as f:
        f.write(str + '\n')


class Trainer():
    def __init__(self, args):
        self.args = args
        self.log_file = args.resume_dir + '/' + args.log_file
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'img_size': args.img_size}
        trainset = get_segmentation_dataset(args.dataset, mode='train', augment=True, **data_kwargs)
        valset = get_segmentation_dataset(args.dataset, mode='vis', augment=False, **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, mode='test', augment=False, **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(valset, batch_size=args.batch_size, drop_last=False, shuffle=False, **kwargs)
        self.testloader = data.DataLoader(testset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                          **kwargs)
        self.nclass = trainset.NUM_CLASS
        # model
        model = get_segmentation_model(args.model, dataset=args.dataset, backbone=args.backbone,
                                       pretrained=args.pretrained, batchnorm=torch.nn.BatchNorm2d,
                                       img_size=args.img_size, dilated=args.dilated, deep_base=args.deep_base,
                                       multi_grid=args.multi_grid, multi_dilation=args.multi_dilation,
                                       output_stride=args.output_stride, high_rates=args.high_rates,
                                       trans_rates=args.trans_rates, is_train=args.is_train, num_layers=args.num_layers,
                                       pretrained_file=args.pretrained_file, trans_out_dim=args.trans_out_dim,
                                       reduce_dim=args.reduce_dim, pooling=args.pooling)


        # optimizer using different LR
        params_list = [{'params': model.pretrain_model.parameters(), 'lr': args.lr}]
        params_list.extend([{'params': model.af_1.parameters(), 'lr': args.lr},
                            {'params': model.af_2.parameters(), 'lr': args.lr},
                            {'params': model.af_3.parameters(), 'lr': args.lr},
                            {'params': model.af_4.parameters(), 'lr': args.lr},
                            {'params': model.head.parameters(), 'lr': args.lr * args.head_lr_factor}])
            # params_list.extend([{'params': model.reduce_conv1.parameters(), 'lr': args.lr * args.head_lr_factor},
            #                     {'params': model.reduce_conv2.parameters(), 'lr': args.lr * args.head_lr_factor},
            #                     {'params': model.reduce_conv3.parameters(), 'lr': args.lr * args.head_lr_factor},
            #                     {'params': model.reduce_conv4.parameters(), 'lr': args.lr * args.head_lr_factor},
            #                     {'params': model.head.parameters(), 'lr': args.lr * args.head_lr_factor}])

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params_list, momentum=args.momentum, weight_decay=args.weight_decay,
                                        nesterov=args.nesterov)
        elif args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params_list)

        # criterion
        self.criterion = SegmentationLoss(args.ce_weight, args.dice_weight)
        self.model, self.optimizer = model, optimizer
        self.model = nn.DataParallel(self.model).cuda()
        self.criterion = self.criterion.cuda()

        # lr_scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_mode, args.lr, args.epochs, len(self.trainloader),
                                            freezn=args.freezn, decode_lr_factor=args.head_lr_factor,
                                            trans=args.trans_rates)

        self.best_pred = 0.0
        # resuming checkpoint
        if args.resume_dir is not None:
            if not os.path.isfile(args.resume_dir + '/checkpoint.pth.tar'):
                print('=> no chechpoint found at {}'.format(args.resume_dir))
                logger(self.log_file, '=> no chechpoint found at {}'.format(args.resume_dir))
                args.start_epoch = 0
            else:
                if args.freezn:
                    shutil.copyfile(args.resume_dir + '/checkpoint.pth.tar',
                                    args.resume_dir + '/checkpoint_origin.pth.tar')
                    shutil.copyfile(args.resume_dir + '/model_best.pth.tar',
                                    args.resume_dir + '/model_best_origin.pth.tar')
                if not args.ft:
                    checkpoint = torch.load(args.resume_dir + '/checkpoint.pth.tar')
                else:
                    if not os.path.isfile(args.resume_dir + '/fineturned_checkpoint.pth.tar') and not os.path.isfile(
                            args.resume_dir + '/checkpoint.pth.tar'):
                        print('=> no chechpoint found at {}'.format(args.resume_dir))
                        logger(self.log_file, '=> no chechpoint found at {}'.format(args.resume_dir))
                    elif os.path.isfile(args.resume_dir + '/fineturned_checkpoint.pth.tar'):
                        checkpoint = torch.load(args.resume_dir + '/fineturned_checkpoint.pth.tar')
                    else:
                        checkpoint = torch.load(args.resume_dir + '/checkpoint.pth.tar')
                args.start_epoch = checkpoint['epoch']
                self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
                if not args.ft and not args.freezn:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint['best_pred']
                print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume_dir, checkpoint['epoch']))
                logger(self.log_file,
                       '=> loaded checkpoint {0} (epoch {1})'.format(args.resume_dir, checkpoint['epoch']))

                # clear start epoch if fine-turning
                if args.ft:
                    args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        ce_loss = 0.0
        dice_loss = 0.0
        structure_loss = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader, desc='\r')
        for i, (image, target, _) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            image = torch.autograd.Variable(image)
            target = torch.autograd.Variable(target)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            logists = self.model(image)
            loss = self.criterion(logists, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Epoch-{0}: Train loss: {1:.3f}'.format(epoch, train_loss / (i + 1)))
        logger(self.log_file,'Epoch-{0}: Train loss: {1:.3f}'.format(epoch, train_loss / (tbar.__len__() + 1)))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred
            }, self.args, is_best)

    def validation_and_test(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target, origin_img):
            target = target.cuda()
            logists = model(image)
            pred = logists.softmax(1)
            batch_size, batch_acc, batch_dice, batch_jacc, batch_sensitivity, batch_specificity = utils.batch_sores(
                pred.detach(), target, origin_img)

            return batch_size, batch_acc, batch_dice, batch_jacc, batch_sensitivity, batch_specificity

        is_best = False
        self.model.eval()

        val_num_img, test_num_img = 0, 0
        val_sum_acc, test_sum_acc = 0, 0
        val_sum_dice, test_sum_dice = 0, 0
        val_sum_jacc, test_sum_jacc = 0, 0
        val_sum_sensitivity, test_sum_se = 0, 0
        val_sum_specificity, test_sum_sp = 0, 0

        test_tbar = tqdm(self.testloader, desc='\r')
        for i, (image, target, origin_img) in enumerate(test_tbar):
            with torch.no_grad():
                test_batch_size, test_batch_acc, test_batch_dice, test_batch_jacc, test_batch_sensitivity, test_batch_specificity = eval_batch(
                    self.model, image, target, origin_img)

            test_num_img += test_batch_size
            test_sum_acc += test_batch_acc
            test_sum_dice += test_batch_dice
            test_sum_jacc += test_batch_jacc
            test_sum_se += test_batch_sensitivity
            test_sum_sp += test_batch_specificity

            test_avg_acc = test_sum_acc / test_num_img
            test_avg_dice = test_sum_dice / test_num_img
            test_avg_jacc = test_sum_jacc / test_num_img
            test_avg_sensitivity = test_sum_se / test_num_img
            test_avg_specificity = test_sum_sp / test_num_img
            test_tbar.set_description(
                'Test      : JA: {0:.4f}, DI: {1:.4f}, SE: {2:.4f}, SP: {3:.4f}, AC: {4:.4f}, img_num: {5}'.format(
                    test_avg_jacc, test_avg_dice, test_avg_sensitivity, test_avg_specificity, test_avg_acc,
                    test_num_img))
        logger(self.log_file,
               'Test      : JA: {0:.4f}, DI: {1:.4f}, SE: {2:.4f}, SP: {3:.4f}, AC: {4:.4f}, img_num: {5}'.format(
                   test_avg_jacc, test_avg_dice, test_avg_sensitivity, test_avg_specificity, test_avg_acc,
                   test_num_img))

        self.model.eval()

        new_pred = test_avg_jacc
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            checkpoint_name = 'fineturned_checkpoint.pth.tar' if self.args.ft else 'checkpoint.pth.tar'
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, False, checkpoint_name)
            print('best checkpoint saved !!!\n')
            logger(self.log_file, 'best checkpoint saved !!!\n')


def parse_args():
    # Traning setting options
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--dataset', type=str, default='covid_19_seg')
    parser.add_argument('--model', type=str, default='ffrnet')
    parser.add_argument('--is-train', type=ast.literal_eval, default=True)
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--checkname', default='exp-0804_ffrnet')
    parser.add_argument('--dilated', type=ast.literal_eval, default=True)
    parser.add_argument('--deep-base', type=ast.literal_eval, default=False)
    parser.add_argument('--img-size', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--multi-grid', type=ast.literal_eval, default=True)
    parser.add_argument('--multi-dilation', type=int, nargs='+', default=[4, 4, 4])
    parser.add_argument('--output-stride', type=int, default=8)
    parser.add_argument('--high-rates', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--trans-rates', type=int, nargs='+', default=[6, 12, 18])
    parser.add_argument('--trans-out-dim', type=int, default=128)
    parser.add_argument('--pooling', type=str, default='adptiveavg', choices=['max', 'adptivemax', 'adptiveavg'])
    parser.add_argument('--reduce-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=1)
    #parser.add_argument('--lstm-layers', type=int, default=1)
    parser.add_argument('--ce-weight', type=float, default=1.0)
    parser.add_argument('--weight', type=float, nargs='+', default=None)
    parser.add_argument('--dice-weight', type=float, default=1.0)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='which optimizer to use. (default: adam)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--head-lr-factor', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument('--momentum', type=float, default='0.9', metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=1e-4, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', type=ast.literal_eval, default=False)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pretrained', type=ast.literal_eval, default=True,
                        help='whether to use pretrained base model')
    parser.add_argument('--freezn', type=ast.literal_eval, default=False)
    parser.add_argument('--pretrained-file', type=str, default=None, help='resnet101-2a57e44d.pth')
    parser.add_argument('--no-val', type=bool, default=False,
                        help='whether not using validation (default: False)')
    parser.add_argument('--ft', type=ast.literal_eval, default=False,
                        help='whether to fine turning (default: True for training)')
    parser.add_argument('--log-file', type=str, default='log.txt')
    parser.add_argument('--resume-dir', type=str,
                        default=parser.parse_args().dataset + '/' + parser.parse_args().model + '_model/' + parser.parse_args().checkname,
                        metavar='PATH')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    logger(args.resume_dir + '/' + args.log_file, ' '.join(sys.argv))
    print(args)
    logger(args.resume_dir + '/' + args.log_file, str(args))

    return args


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch: {}'.format(args.start_epoch))
    logger(args.resume_dir + '/' + args.log_file, 'Starting Epoch: {}'.format(args.start_epoch))
    print('Total Epochs: {}'.format(args.epochs))
    logger(args.resume_dir + '/' + args.log_file, 'Total Epochs: {}'.format(args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if not args.no_val:
            trainer.validation_and_test(epoch)
        torch.cuda.empty_cache()
