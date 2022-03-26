import sys


class ProgressBar(object):
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"

    def __init__(self, mode, epoch=None, total_epoch=None, current_loss=None, current_hl=None, current_subacc=None,
                  total=None, current=None, model_name=None, width=50, symbol=">",output=sys.stderr):
        assert len(symbol) == 1
        self.mode = mode
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.current_loss = current_loss
        self.current_subacc = current_subacc
        self.current_hl = current_hl
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
            "current_subacc": self.current_subacc,
            "current_hl": self.current_hl,
            "epoch": self.epoch + 1,
            "epochs": self.total_epoch
        }
        message = "\033[1;32;40m%(mode)s Epoch:  %(epoch)d/%(epochs)d %(bar)s\033[0m  [Current: Loss: %(current_loss)f " \
                  "SubAcc: %(current_subacc)f  ] %(current)d/%(total)d \033[1;32;40m[ %(percent)3d%% ]\033[0m" % args
        self.write_message = "%(mode)s Epoch:  %(epoch)d/%(epochs)d %(bar)s  [Current: Loss %(current_loss)f " \
                             "SubAcc: %(current_subacc)f HammingLoss: %(current_hl)f]  " \
                             "%(current)d/%(total)d [ %(percent)3d%% ]" % args
        print("\r" + message, file=self.output, end="")

    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)

        with open('./logs/multi_%s.txt' % self.model_name, "a") as f:
            print(self.write_message, file=f)