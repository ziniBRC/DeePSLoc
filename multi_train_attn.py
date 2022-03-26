import torch
import torch.nn as nn
import argparse
from utils.saver import Saver
from utils.multi_util import *
from models.denselab import *
from models.resnetlab import *
from utils.metrics import *
from utils.multi_bar import *
from models.triplet_transformer import ResNetTripletFormer
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
    def __init__(self, args):
        self.args = args

        # define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        print("Experiment id is", self.saver.run_id)

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_patch_data_loader(args, **kwargs)

        # Define network
        model = ResNetTripletFormer(num_classes=self.nclass, device=args.device)
        # model = resnetlab(n_classes=self.nclass)
        # model = denselab(n_classes=self.nclass)

        # Define Optimizer
        special_layers = torch.nn.ModuleList([model.attn_layer_ap, model.attn_layer_an])
        special_layers_params = list(map(id, special_layers.parameters()))
        base_params = filter(lambda p: id(p) not in special_layers_params, model.parameters())
        optimizer = torch.optim.Adam([{'params': base_params, 'initial_lr': args.lr}],
                                     lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
        optimizer_attn = torch.optim.Adam([{'params': special_layers.parameters(), 'initial_lr': args.lr}],
                                          lr=args.lr, betas=(0.9, 0.999), eps= 1e-08, weight_decay=0, amsgrad=False)

        self.optimizer = optimizer
        self.optimizer_attn = optimizer_attn


        # Define Criterion
        self.attn_margins = [-1, -0.6, -0.2, 0.2, 0.6, 1]
        self.bce_loss = nn.BCEWithLogitsLoss().to(args.device)
        self.tri_loss = nn.TripletMarginLoss(margin=1, p=2.0, eps=1e-6, swap=False, reduction='mean').to(args.device)
        self.attn_tri_loss_funcs, self.attn_tri_loss_funcs_reverse = [], []
        for attn_margin in self.attn_margins:
            self.attn_tri_loss_funcs.append(nn.TripletMarginLoss(margin=attn_margin, p=2, eps=1e-6, swap=False, reduction='mean').to(args.device))
            self.attn_tri_loss_funcs_reverse.append(nn.TripletMarginLoss(margin=-attn_margin, p=2, eps=1e-6, swap=False, reduction='mean').to(args.device))
        self.alpha = 0.25

        # self.model = model.cuda()
        self.model = nn.DataParallel(model, device_ids=[0])
        self.model.to(args.device)

        # define scheuler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.scheduler_attn = torch.optim.lr_scheduler.StepLR(self.optimizer_attn, step_size=100, gamma=0.1)
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
        # self.model.train()
        pre, tar = [], []
        # sample, positive, negative (1,2,3)
        for iter, (imgs, labels) in enumerate(self.train_loader):
            imgs = torch.cat([imgs[:, 0, :], imgs[:, 1, :], imgs[:, 2, :]], dim=0)
            labels = torch.cat([labels[:, 0], labels[:, 1], labels[:, 2]], dim=0)
            imgs = Variable(imgs).to(self.args.device)
            labels = Variable(torch.from_numpy(np.array(labels,dtype='float32'))).to(self.args.device)
            N = imgs.shape[0]

            pos_mask = cal_same_label_mask(labels)

            # attn loss backward
            self.model.train()
            output_a, feature_a, hard_feature_p, hard_feature_n = self.model(imgs, train_attn=True, same_mask=pos_mask)
            loss = 0
            for i in range(len(self.attn_tri_loss_funcs)):
                attn_tri_loss = self.attn_tri_loss_funcs[i]
                attn_tri_loss_reverse = self.attn_tri_loss_funcs_reverse[i]
                loss = loss + attn_tri_loss(feature_a.squeeze(0), hard_feature_n[i].squeeze(0), hard_feature_p[i].squeeze(0)) \
                       + attn_tri_loss_reverse(feature_a.squeeze(0), hard_feature_p[i].squeeze(0), hard_feature_n[i].squeeze(0))
            loss = loss / len(self.attn_tri_loss_funcs)
            self.optimizer_attn.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer_attn.step()

            # ce loss backward
            self.model.module.attn_layer_ap.eval()
            self.model.module.attn_layer_an.eval()
            output_a, feature_a, hard_feature_p, hard_feature_n = self.model(imgs, train_attn=False, same_mask=pos_mask)
            ce = self.bce_loss(output_a, labels)
            tri, dis_an = 0, 0
            for i in range(len(self.attn_tri_loss_funcs)):
                tri = tri + self.tri_loss(feature_a.squeeze(0), hard_feature_p[i].squeeze(0), hard_feature_n[i].squeeze(0))
                dis_an += torch.mean(torch.norm(feature_a.squeeze(0) - hard_feature_n[i].squeeze(0), dim=-1, p=2, keepdim=True))
            tri = tri / len(self.attn_tri_loss_funcs)
            dis_an = dis_an / len(self.attn_tri_loss_funcs)
            loss = (1 - self.alpha) * ce + self.alpha * tri
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_progressor.current_loss = train_loss / (iter + 1)

            if self.args.dataset == 'Imploc':
                pred = F.sigmoid(output_a)
                p = threshold_tensor_batch(pred)
                pre.extend(p.cpu().numpy().astype(np.int32))
                tar.extend(labels.cpu().numpy())
            else:
                pred = F.sigmoid(output_a).data.cpu().numpy()
                p = getKTout(pred)
                pre.extend(p.astype(np.int32))
                tar.extend(labels.cpu().numpy())

            # cal current batch Accuracy
            e = Multi_eva(pre, tar)
            train_progressor.current_hl = e.getHloss()
            train_progressor.current_subacc = e.getSubAcc()

            print('[Train: Epoch: %d, Batch: %d, Loss_iter: %.3f, Tri: %.3f, Acc: %f, LR: %e, Dis_an: %.3f]' % (
            train_progressor.epoch, iter, loss.item(), tri.item(), train_progressor.current_subacc, self.optimizer.state_dict()['param_groups'][0]['lr'], dis_an.item()))

        self.scheduler.step()
        train_progressor.done()

    def testing(self, test_progressor):
        self.model.eval()
        test_loss = 0.0
        pre, tar = [], []
        gene_pred_dict, gene_cnt_dict = {}, {}
        gene_label_dict = {}
        for iter, tuples in enumerate(self.test_loader):
            if self.args.dataset == 'Imploc':
                (genes, Img, Label) = tuples
            else:
                (Img, Label) = tuples
            l = len(Img)
            b = Label.shape[0]
            # Label = Variable(torch.from_numpy(np.array(Label, dtype='float32'))).to(self.args.device)
            vote = 0

            # ploss = np.zeros(l)
            # target = Variable(torch.from_numpy(np.array(Label)).long()).to(self.args.device)
            for i in range(l):
                input = Variable(Img[i]).to(self.args.device)
                with torch.no_grad():
                    output, feats = self.model(input, training=False)
                # ploss[i] = self.ce_loss(output, target)
                if self.args.dataset == 'Imploc':
                    pred = F.sigmoid(output)
                else:
                    pred = F.sigmoid(output).data.cpu().numpy()
                vote += pred
            # vote /= l
            if self.args.dataset == 'Imploc':
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
            else:
                vote /= l
                predL = getKTout(vote, T=0.2)

            # loss = self.ce_loss(pr, target)
            # test_loss += loss.item()
            test_progressor.current_loss = test_loss / (iter + 1)
            test_progressor.current_loss = 0

            pre.extend(predL)
            tar.extend(np.array(Label,dtype="float32"))
            # cal current batch Accuracy
            e = Multi_eva(pre, tar)
            test_progressor.current_hl = e.getHloss()
            test_progressor.current_subacc = e.getSubAcc()

            print('[Test: Epoch: %d, Batch: %d, subAcc: %f]' % (
            test_progressor.epoch, iter, test_progressor.current_subacc))

        if self.args.dataset == 'Imploc':
            correct_cnt = 0
            pre = []
            tar = []
            for gene in gene_pred_dict.keys():
                gene_pred_dict[gene] = gene_pred_dict[gene]/gene_cnt_dict[gene]
                predL = threshold_tensor_batch(gene_pred_dict[gene].unsqueeze(0)).cpu().numpy().astype(np.int32)
                pre.extend(predL)
                tar.extend(gene_label_dict[gene].unsqueeze(0).detach().cpu().numpy())
            subset_acc = torch_metrics(np.array(tar), np.array(pre))
            curpred = subset_acc
        else:
            subset_acc = torch_metrics(np.array(tar), np.array(pre))
            curpred = test_progressor.current_subacc
        test_progressor.done()

        # save checkpoint every epoch
        if curpred > self.subacc:
            self.subacc = curpred
            self.hl = test_progressor.current_hl

            is_best = False
            self.saver.save_checkpoint({
                'epoch': test_progressor.epoch + 1,
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
    parser.add_argument('--mode', type=str, default='hard',
                        help='set the model training mode')
    parser.add_argument('--patch', type=bool, default='False')

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
    # if args.conti:
    #     # checkpoint = torch.load("/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/Multi/imploc_ce/checkpoint.pth.tar")
    #     # checkpoint = torch.load("/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/Multi/experiment_419/checkpoint.pth.tar")
    #     # checkpoint = torch.load("/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/Imploc/experiment_21/checkpoint.pth.tar")
    #     checkpoint = torch.load("/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/Multi/466Multi_patch_attn/checkpoint.pth.tar")
    #     trainer.model.load_state_dict(checkpoint['state_dict'])
    #     trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    #     trainer.args.start_epoch = checkpoint['epoch']
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



