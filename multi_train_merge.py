import torch
import torch.nn as nn
import argparse
from utils.saver import Saver
from utils.multi_util import *
from models.denselab import *
from models.resnetlab import *
from models.triplet_transformer import *
from utils.metrics import *
from utils.multi_bar import *
from utils.lr_scheduler import LR_Scheduler
from torch.autograd import Variable


# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1, 2]))
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [6]))
# os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus


def threshold_tensor_batch(pd, base=0.5):
    '''make sure at least one label for batch'''
    p_max = torch.max(pd, dim=1)[0]
    pivot = torch.cuda.FloatTensor([base]).expand_as(p_max)
    threshold = torch.min(p_max, pivot)
    pd_threshold = torch.ge(pd, threshold.unsqueeze(dim=1))
    return pd_threshold


class Trainer(object):
    def __init__(self, args, **kwargs):
        self.args = args

        # define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        print("Experiment id is", self.saver.run_id)

        # Define Dataloader
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_merge_data_loader(args, **kwargs)

        # Define network
        # model_resample = resnetlab(n_classes=self.nclass)
        # model_patch = resnetlab(n_classes=self.nclass)
        model_resample = ResNetTripletFormer(num_classes=self.nclass)
        model_patch = ResNetTripletFormer(num_classes=self.nclass)

        model = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, self.nclass)
        )

        # Define Optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
        optimizer = torch.optim.Adam([
            {'params': model_resample.parameters()},
            {'params': model_patch.parameters()},
            {'params': model.parameters()},
        ], lr=0.0001, weight_decay=0.001)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001,
        #                              amsgrad=False)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
        self.optimizer = optimizer

        # Define Criterion
        self.bce_loss = nn.BCEWithLogitsLoss().to(args.device)
        self.tri_loss = nn.TripletMarginLoss(margin=1, p=2.0, eps=1e-6, swap=False, reduction='mean').to(args.device)
        self.alpha = 0.0

        # self.model = model.cuda()
        self.model_resample = nn.DataParallel(model_resample, device_ids=[0])
        self.model_patch = nn.DataParallel(model_patch, device_ids=[0])
        self.model = nn.DataParallel(model, device_ids=[0])
        self.model_resample.to(args.device)
        self.model_patch.to(args.device)
        self.model.to(args.device)

        # define scheuler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        # self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # Resuming checkpoint
        self.subacc = 0.0
        self.hl = 0.0
        # self.precision = 0.0
        # self.f1 = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.subacc = checkpoint['best_pred(sub_acc)']
            self.hl = checkpoint['best_hammingloss']
            # self.precision = checkpoint['best_precision']
            # self.f1 = checkpoint['best_f1']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, train_progressor):
        train_loss = 0.0
        self.model_resample.train()
        self.model_patch.train()
        self.model.train()
        pre, tar = [], []
        # sample, positive, negative (1,2,3)
        for iter, (img_resample, img_patch, label) in enumerate(self.train_loader):
            img_resample = Variable(img_resample).to(self.args.device)
            img_patch = Variable(img_patch).to(self.args.device)
            label1 = Variable(torch.from_numpy(np.array(label,dtype='float32'))).to(self.args.device)

            pred_resample, feats_resample = self.model_resample(img_resample)
            pred_patch, feats_patch = self.model_patch(img_patch)
            feats = torch.cat([feats_resample, feats_patch], dim=-1)
            pd = self.model(feats)

            #loss
            bce = self.bce_loss(pd, label1)
            loss = bce

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_progressor.current_loss = train_loss / (iter + 1)

            pred = F.sigmoid(pd)
            p = threshold_tensor_batch(pred)

            pre.extend(p.cpu().numpy().astype(np.int32))
            tar.extend(label1.cpu().numpy())
            # cal current batch Accuracy
            e = Multi_eva(pre, tar)
            train_progressor.current_hl = e.getHloss()
            train_progressor.current_subacc = e.getSubAcc()

            print('[Train: Epoch: %d, Batch: %d, Loss_iter: %.3f, Acc: %f, LR: %e]' % (
            train_progressor.epoch, iter, loss.item(), train_progressor.current_subacc, self.optimizer.state_dict()['param_groups'][0]['lr']))

        self.scheduler.step()
        train_progressor.done()

    def testing(self, test_progressor):
        self.model_resample.eval()
        self.model_patch.eval()
        self.model.eval()
        test_loss = 0.0
        pre, tar = [], []
        gene_pred_dict, gene_cnt_dict = {}, {}
        gene_label_dict = {}
        for iter, (genes, ImgResample, ImgPatch, Label) in enumerate(self.test_loader):
            l = len(ImgResample)
            b = Label.shape[0]
            # Label = Variable(torch.from_numpy(np.array(Label, dtype='float32'))).to(self.args.device)
            vote = 0

            for i in range(l):
                input_resample = Variable(ImgResample[i]).to(self.args.device)
                input_patch = Variable(ImgPatch[i]).to(self.args.device)
                with torch.no_grad():
                    pred_resample, feats_resample = self.model_resample(input_resample, training=False)
                    pred_patch, feats_patch = self.model_patch(input_patch, training=False)
                    feats = torch.cat([feats_resample, feats_patch], dim=-1)
                    # output = pred_patch
                    output = (pred_resample + pred_patch) / 2
                    # output = self.model(feats)

                pred = F.sigmoid(output)
                # p = getTopKout(pred)
                vote += pred
            vote /= l

            for i, gene in enumerate(genes):
                if gene not in gene_pred_dict:
                    gene_pred_dict[gene] = vote[i, :]
                    gene_cnt_dict[gene] = 1
                    gene_label_dict[gene] = Label[i, :]
                else:
                    gene_pred_dict[gene] += vote[i, :]
                    gene_cnt_dict[gene] += 1

            predL = threshold_tensor_batch(vote).cpu().numpy().astype(np.int32)

            test_progressor.current_loss = test_loss / (iter + 1)
            test_progressor.current_loss = 0

            pre.extend(predL)
            tar.extend(np.array(Label,dtype="float32"))
            e = Multi_eva(pre, tar)
            test_progressor.current_hl = e.getHloss()
            test_progressor.current_subacc = e.getSubAcc()

            print('[Test: Epoch: %d, Batch: %d, subAcc: %f]' % (
            test_progressor.epoch, iter, test_progressor.current_subacc))

        correct_cnt = 0
        pre = []
        tar = []
        for gene in gene_pred_dict.keys():
            gene_pred_dict[gene] = gene_pred_dict[gene]/gene_cnt_dict[gene]
            predL = threshold_tensor_batch(gene_pred_dict[gene].unsqueeze(0)).cpu().numpy().astype(np.int32)
            pre.extend(predL)
            tar.extend(gene_label_dict[gene].unsqueeze(0).detach().cpu().numpy())
        subset_acc = torch_metrics(np.array(tar), np.array(pre))

        test_progressor.done()

        # save checkpoint every epoch
        # curpred = test_progressor.current_subacc
        curpred = subset_acc

        if curpred > self.subacc:
            self.subacc = curpred
            self.hl = test_progressor.current_hl

            is_best = False
            self.saver.save_checkpoint({
                'epoch': test_progressor.epoch + 1,
                'state_dict_resample': self.model_resample.state_dict(),
                'state_dict_patch': self.model_patch.state_dict(),
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred(subacc)': self.subacc,
                'best_hl': self.hl,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch CNN_self Training")
    parser.add_argument('--dataset', type=str, default='Multi',
                        choices=['IHC', 'Alta', 'Multi', 'Imploc'],
                        help='dataset name (default: IHC)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--workers', type=int, default=config.workers,
                        metavar='N', help='dataloader threads')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=config.epochs, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=config.batch_size,
                        metavar='N', help='input batch size for \
                                        training (default: auto)')
    parser.add_argument('--lr', type=float, default=config.lr, metavar='LR',
                        help='learning rate (default: auto)')

    # cuda, seed and logging
    # parser.add_argument('--gpu-ids', type=str, default=None)
    parser.add_argument('--seed', type=int, default=config.seed, metavar='S',
                        help='random seed (default: 1)')

    # optimizer params
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--weight-decay', type=float, default=config.weight_decay,
                        metavar='M', help='w-decay (default: 5e-4)')

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=config.model_name,
                        help='set the checkpoint name')

    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    parser.add_argument('--conti', type=bool, default='False')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    # if args.gpu_ids == None:
    #     args.gpu_ids = config.gpus

    # args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]

    print(args)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(args)
    if args.conti:
        # '/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/Imploc/21_Imploc_resample_attn_pretrain/'
        # ''
        # checkpoint_resample = torch.load("/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/Multi/HPA_18_Resample/checkpoint.pth.tar")
        # checkpoint_patch = torch.load("/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/Multi/imploc_ce/checkpoint.pth.tar")
        checkpoint_resample = torch.load("/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/Imploc/21_Imploc_resample_attn_pretrain/checkpoint.pth.tar")
        checkpoint_patch = torch.load("/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/Imploc/16Imploc_patch_attn_pretrain/checkpoint.pth.tar")
        trainer.model_resample.load_state_dict(checkpoint_resample['state_dict'])
        trainer.model_patch.load_state_dict(checkpoint_patch['state_dict'])
        # checkpoint = torch.load("/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/Multi/experiment_274/checkpoint.pth.tar")
        # trainer.model.load_state_dict(checkpoint['state_dict'])
        # # trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        # trainer.args.start_epoch = checkpoint['epoch']
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        train_progressor = ProgressBar(mode="Train", epoch=epoch, total_epoch=trainer.args.epochs,
                                       model_name=trainer.args.checkname, total=len(trainer.train_loader))
        trainer.training(train_progressor)

        #test
        if epoch % 5 == 0:
            val_progressor = ProgressBar(mode="Val", epoch=epoch, total_epoch=trainer.args.epochs,
                                           model_name=trainer.saver.run_id, total=len(trainer.val_loader))
            trainer.testing(val_progressor)
            # test_progressor = ProgressBar(mode="Test", epoch=epoch, total_epoch=trainer.args.epochs,
            #                                model_name=trainer.args.checkname, total=len(trainer.test_loader))
            # trainer.testing(test_progressor)


if __name__ =="__main__":
    main()



