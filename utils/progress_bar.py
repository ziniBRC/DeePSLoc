import sys


class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"

    def __init__(self, mode, epoch=None, total_epoch=None, current_loss=None, current_pre=None, current_recall=None,
                 current_acc=None, current_F1=None, total=None, current=None, model_name=None, width=50, symbol=">",output=sys.stderr):
        assert len(symbol) == 1
        self.mode = mode
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.current_loss = current_loss
        self.current_acc = current_acc
        self.current_pre = current_pre
        self.current_recall = current_recall
        self.current_F1 = current_F1
        self.total = total
        self.current = current
        self.width = width
        self.model_name = model_name
        self.output = output
        self.symbol = symbol

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "mode": self.mode,
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "current_loss": self.current_loss,
            "current_acc": self.current_acc,
            "current_pre": self.current_pre,
            "current_recall": self.current_recall,
            "current_F1": self.current_F1,
            "epoch": self.epoch + 1,
            "epochs": self.total_epoch
        }
        message = "\033[1;32;40m%(mode)s Epoch:  %(epoch)d/%(epochs)d %(bar)s\033[0m  [Current: Loss: %(current_loss)f " \
                  "Acc: %(current_acc)f  ] %(current)d/%(total)d \033[1;32;40m[ %(percent)3d%% ]\033[0m" % args
        self.write_message = "%(mode)s Epoch:  %(epoch)d/%(epochs)d %(bar)s  [Current: Loss %(current_loss)f " \
                             "Acc: %(current_acc)f Precision: %(current_pre)f Recall: %(current_recall)f F1: %(current_F1)f ]  " \
                             "%(current)d/%(total)d [ %(percent)3d%% ]" % args
        print("\r" + message, file=self.output, end="")


    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)
        with open("./logs/%s.txt" % self.model_name, "a") as f:
            print(self.write_message, file=f)