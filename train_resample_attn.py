import torch
import torch.nn as nn
import argparse
from utils.saver import Saver
from utils.util import *
from models.denselab import *
from models.triplet_transformer import *
from models.resnetlab import *
from utils.metrics import *
from utils.progress_bar import *
from utils.lr_scheduler import LR_Scheduler
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [7]))
# os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus


def sortPatch(act_map):
    act_map = act_map.reshape(act_map.shape[0], act_map.shape[1] * act_map.shape[2])
    ind_sort = np.argsort(act_map, axis=1)
    return ind_sort


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        self.writer = SummaryWriter('logs/')
        print("Experiment id is", self.saver.run_id)

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        # self.train_loader,  self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_resample_data_loader(args, **kwargs)

        # Define network
        # model = resnetlab(n_classes=self.nclass)
        model = ResNetTripletFormer(num_classes=self.nclass, device=args.device)
        # model = denselab(n_classes=self.nclass)

        # Define Optimizer-
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
        special_layers = torch.nn.ModuleList([model.attn_layer_ap, model.attn_layer_an])
        special_layers_params = list(map(id, special_layers.parameters()))
        base_params = filter(lambda p: id(p) not in special_layers_params, model.parameters())
        optimizer = torch.optim.Adam([{'params': base_params, 'initial_lr': args.lr}],
                                     lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
        optimizer_attn = torch.optim.Adam([{'params': special_layers.parameters(), 'initial_lr': args.lr}],
                                          lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        self.optimizer = optimizer
        self.optimizer_attn = optimizer_attn

        # Define Criterion
        self.attn_margins = [0, 0.2, 0.4, 0.6, 0.8, 1]
        self.ce_loss = nn.CrossEntropyLoss().to(args.device)
        self.tri_loss = nn.TripletMarginLoss(margin=1, p=2.0, eps=1e-6, swap=False, reduction='mean').to(args.device)
        self.attn_tri_loss_funcs = []
        for attn_margin in self.attn_margins:
            self.attn_tri_loss_funcs.append(nn.TripletMarginLoss(margin=attn_margin, p=2, eps=1e-6, swap=False, reduction='mean').to(args.device))
        self.alpha = 0.25
        # self.model = model.cuda()
        self.model = nn.DataParallel(model, device_ids=[0,])
        self.model.to(args.device)

        # define scheuler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.scheduler_attn = torch.optim.lr_scheduler.StepLR(self.optimizer_attn, step_size=100, gamma=0.1)
        # self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

        # Resuming checkpoint
        self.best_pred = 0.0
        self.presicion = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred(acc)']
            self.presicion = checkpoint['best_precision']
            self.recall = checkpoint['best_recall']
            self.f1 = checkpoint['best_f1']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def my_train(self, train_progressor):
        train_loss = 0.0
        self.scheduler.step()
        self.model.train()
        pre, tar = [], []
        tri_sum = 0
        # sample, positive, negative (1,2,3)
        for iter, (img, label) in enumerate(self.train_loader):
            img = torch.cat([img[:, 0, :], img[:, 1, :], img[:, 2, :]], dim=0)
            label = torch.cat([label[:, 0], label[:, 1], label[:, 2]], dim=0)
            img = Variable(img).to(self.args.device)
            label = Variable(torch.from_numpy(np.array(label)).long()).to(self.args.device)
            pos_mask = cal_same_label_mask(label)

            # attn loss backward
            output_a, feature_a, hard_feature_p, hard_feature_n = self.model(img, train_attn=True, pos_mask=pos_mask)
            loss = 0
            for i in range(len(self.attn_tri_loss_funcs)):
                attn_tri_loss = self.attn_tri_loss_funcs[i]
                loss = loss + attn_tri_loss(feature_a.squeeze(0), hard_feature_n[i].squeeze(0), hard_feature_p[i].squeeze(0))
            loss = loss / len(self.attn_tri_loss_funcs)
            self.optimizer_attn.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer_attn.step()

            # ce loss backward
            self.model.module.attn_layer_ap.eval()
            self.model.module.attn_layer_an.eval()
            output_a, feature_a, hard_feature_p, hard_feature_n = self.model(img, train_attn=False, pos_mask=pos_mask)
            ce = self.ce_loss(output_a, label)
            tri, dis_an = 0, 0
            for i in range(len(self.attn_tri_loss_funcs)):
                tri = tri + self.tri_loss(feature_a.squeeze(0), hard_feature_p[i].squeeze(0), hard_feature_n[i].squeeze(0))
                dis_an += torch.mean(torch.norm(feature_a.squeeze(0) - hard_feature_n[0].squeeze(0), dim=-1, p=2, keepdim=True))
            tri = tri / len(self.attn_tri_loss_funcs)
            dis_an = dis_an / len(self.attn_tri_loss_funcs)
            loss = (1 - self.alpha) * ce + self.alpha * tri
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_progressor.current_loss = train_loss / (iter + 1)

            pred = F.softmax(output_a, dim=1).data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            pre.extend(pred)
            tar.extend(label.cpu().numpy())
            # cal current batch Accuracy
            e = Evaluator(pre, tar)
            train_progressor.current_acc = e.getAcc()
            train_progressor.current_pre = e.getPrecision()
            train_progressor.current_recall = e.getRecall()
            train_progressor.current_F1 = e.getF1()

            print('[Train: Epoch: %d, Batch: %d, Loss_iter: %.3f, Triplet: %.3f, Acc: %f, LR: %e, Dis_an: %.3f]' % (
            train_progressor.epoch, iter, loss.item(), tri.item(), train_progressor.current_acc, self.optimizer.state_dict()['param_groups'][0]['lr'], dis_an.item()))
            # print('[Train: Epoch: %d, Batch: %d, Loss_iter: %.3f, Tri: %.3f, Acc: %f, LR: %e, Dis_an: %.3f]' % (
            # train_progressor.epoch, iter, loss.item(), tri.item(), train_progressor.current_subacc, self.optimizer.state_dict()['param_groups'][0]['lr'], dis_an.item()))
        train_progressor.done()

    def testing(self, test_progressor):
        self.model.eval()
        test_loss, pred_prob = 0.0, []
        pre, tar = [], []
        resnet_feats, file_list = [], []
        for iter, (Img, Label, filename) in enumerate(self.test_loader):
            Img = Variable(Img).to(self.args.device)
            Label = Variable(torch.from_numpy(np.array(Label)).long()).to(self.args.device)
            file_list.extend(filename)

            with torch.no_grad():
                output, feats = self.model(Img, training=False)

            dis_array = cal_distance_array(feats)
            pos_mask = cal_same_label_mask(Label)
            neg_mask = ~pos_mask
            pos_ind = torch.max(dis_array.masked_fill(neg_mask, float(0)), dim=1)[1]
            neg_ind = torch.min(dis_array.masked_fill(pos_mask, float('inf')), dim=1)[1]
            feature_p = feats[pos_ind]
            feature_n = feats[neg_ind]
            tri = self.tri_loss(feats, feature_p, feature_n)

            resnet_feats.extend(feats.data.cpu().numpy())

            prob = F.softmax(output, dim=1).data.cpu().numpy()
            pred_prob.extend(prob)
            pred = np.argmax(prob, axis=1)
            pre.extend(pred)
            tar.extend(Label.cpu().numpy())

            test_progressor.current_loss = test_loss / (iter + 1)
            test_progressor.current_loss = 0

            # cal current batch Accuracy
            e = Evaluator(pre, tar)
            test_progressor.current_acc = e.getAcc()
            test_progressor.current_pre = e.getPrecision()
            test_progressor.current_recall = e.getRecall()
            test_progressor.current_F1 = e.getF1()

            print('[Test: Epoch: %d, Batch: %d, Triplet: %.3f, Acc: %f]' % (
            test_progressor.epoch, iter, tri.item(), test_progressor.current_acc))

        test_progressor.done()

        curpred= test_progressor.current_acc
        if curpred > self.best_pred:
        # if True:
            self.best_pred = curpred
            self.precision = test_progressor.current_pre
            self.recall = test_progressor.current_recall
            self.f1 = test_progressor.current_F1

            is_best = False
            self.saver.save_checkpoint({
                'epoch': test_progressor.epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_attn': self.optimizer_attn.state_dict(),
                'best_pred(acc)': self.best_pred,
                'best_precision': self.precision,
                'best_recall': self.recall,
                'best_f1': self.f1,

            }, is_best)

        # np.save('/data/users/liuziyi/PyProgram/deep_PSL/dataset/train/Alta/resample_pred.npy', np.array(pred_prob))
        # resnet_feats = np.mean(np.array(resnet_feats), axis=1)
        # np.save('/data/users/liuziyi/PyProgram/deep_PSL/dataset/train/Alta/resnet_feats.npy', resnet_feats)
        # sort_ind = sortPatch(resnet_feats)
        # np.save('/data/users/liuziyi/PyProgram/deep_PSL/dataset/train/Alta/sort_index.npy', sort_ind)
        # size = 256
        # for index, filename in enumerate(file_list):
        #     img = Image.open(filename)
        #     num = sort_ind[index, -1]
        #     x = (2200 // 16) * (num % 16)
        #     y = (2200 // 16) * (num // 16)
        #     cropImg = img.crop((x - size // 2, y - size // 2, x + size // 2, y + size // 2))
        #     plt.imshow(np.array(cropImg))
        #     plt.show()
        # pass


def main():
    parser = argparse.ArgumentParser(description="PyTorch CNN_self Training")
    parser.add_argument('--dataset', type=str, default='Alta',
                        choices=['IHC', 'Alta'],
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

    # finetuning pre-trained modelsoptimizer
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    parser.add_argument('--conti', type=bool, default='False')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    print("gpu available number = ", torch.cuda.device_count())
    # if args.gpu_ids == None:
    #     args.gpu_ids = config.gpus

    # args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]

    print(args)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(args)
    # if args.conti:
    #     # checkpoint = torch.load("/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/Alta/resnet_resample/checkpoint.pth.tar")
    #     checkpoint = torch.load("/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/IHC/experiment_67/checkpoint.pth.tar")
    #     # checkpoint = torch.load("/data/users/liuziyi/PyProgram/deep_PSL/checkpoints/Alta/HardResample/checkpoint.pth.tar")
    #     trainer.model.load_state_dict(checkpoint['state_dict'])
    #     trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    #     trainer.optimizer_attn.load_state_dict(checkpoint['optimizer_attn'])
    #     trainer.args.start_epoch = checkpoint['epoch']
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        train_progressor = ProgressBar(mode="Train", epoch=epoch, total_epoch=trainer.args.epochs,
                                       model_name=trainer.saver.run_id, total=len(trainer.train_loader))
        trainer.my_train(train_progressor)

        #test
        if epoch % 5 == 0:
            val_progressor = ProgressBar(mode="Val", epoch=epoch, total_epoch=trainer.args.epochs,
                                           model_name=trainer.saver.run_id, total=len(trainer.val_loader))
            trainer.testing(val_progressor)
            # test_progressor = ProgressBar(mode="Test", epoch=epoch, total_epoch=trainer.args.epochs,
            #                                model_name=trainer.saver.run_id, total=len(trainer.test_loader))
            # trainer.testing(test_progressor)


if __name__ =="__main__":
    main()



