from mvit import get_mvit
class FakeCfg():
    def __init__(self):
        self.mode = 'finetune'
        self.weight = 'pretrain'
        self.dir_weights = '/home/models/'
        self.num_classes = [100]
    
cfg = FakeCfg()

def get_mvit_model():
    mvit = get_mvit(cfg)
    print(mvit)
    return mvit