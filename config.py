import os

class DefaultConfigs(object):
    #1 string parameters
    root = os.getcwd()

    train_data_IHC = os.path.join(root, 'data/train/cropIHC/')
    test_data_IHC = os.path.join(root, 'data/test/cropIHC/')

    train_data_Multi = os.path.join(root, 'data/train/cropMulti/')
    test_data_Multi = os.path.join(root, 'data/test/cropMulti/')

    train_data_alta = os.path.join(root, 'data/train/cropAlta/')
    test_data_alta = os.path.join(root, 'data/test/cropAlta/')

    model_name = "MyDeeplab"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/densenet"
    model_name = ""
    gpus = "1"


    #2 numeric parameters
    epochs = 300
    batch_size = 64
    workers = 16
    img_height = 256
    img_height = 256
    img_weight = 256
    ihc_classes = 7
    alta_classes = 8
    multi_classes = 7
    seed = 400
    lr = 1e-4
    # lr_decay = 1e-4
    weight_decay = 0


config = DefaultConfigs()