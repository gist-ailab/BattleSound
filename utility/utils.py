import json
import os
import torch

class config():
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.result = {}

    def get_config_data(self):
        config_data = self.load_json(os.path.join(self.config_dir, 'data.json'))
        self.result['data'] = config_data

    def get_config_network(self):
        config_network = self.load_json(os.path.join(self.config_dir, 'network.json'))
        self.result['network'] = config_network

    def get_config_train(self):
        config_train = self.load_json(os.path.join(self.config_dir, 'train.json'))
        self.result['train'] = config_train

    def get_config_meta(self):
        config_meta = self.load_json(os.path.join(self.config_dir, 'meta.json'))
        self.result['meta'] = config_meta

    def get_config_tune(self):
        config_tune = self.load_json(os.path.join(self.config_dir, 'tune.json'))
        self.result['tune'] = config_tune

    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            out = json.load(f)
        return out

    def import_config(self, config_path):
        config_total = self.load_json(config_path)
        self.result = config_total

    def export_config(self, config_save_path):
        with open(config_save_path, 'w') as f:
            json.dump(self.result, f)


class train_module():
    def __init__(self, total_epoch, criterion_list, multi_gpu=False):
        self.total_epoch = total_epoch
        self.save_dict = {'criterion': criterion_list,
                          'model': [],
                          'addon': [],
                          'optimizer': [],
                          'scheduler': [],
                          'save_epoch': -1}

        self.init_epoch = 0
        self.multi_gpu = multi_gpu


    def import_module(self, module_path, map_location=None):
        if map_location is not None:
            result = torch.load(module_path, map_location=map_location)
        else:
            result = torch.load(module_path)

        self.save_dict = result
        self.init_epoch = int(result['save_epoch']) + 1


    def export_module(self, save_path):
        torch.save(self.save_dict, save_path)