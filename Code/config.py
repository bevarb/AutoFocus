import warnings
class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model = 'CNN'  # 使用的模型， 名字必须与models/__init__.py中的名字一致
    load_model_path = 'checkpoints/model.pth'  # 加载预训练模型的路径，为None代表不加载
    batch_size = 128
    use_gpu = True
    num_workers = 4  # how many workers for loading data
    print_freq = 20

    debug_file = 'tmp/debug'  # if os.path.exits(debug_file):enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4

    def parse(self, kwargs):
        '''
        根据字典kwargs更新config参数
        '''
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))