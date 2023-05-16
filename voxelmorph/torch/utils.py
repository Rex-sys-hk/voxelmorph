from torch.utils.data import Dataset
import numpy as np
import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

class RegData(Dataset):
    def __init__(self, generator, length = 100):
        print('RegData init')
        self.generator = generator
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs, y_true = next(self.generator)
        # print(inputs[0].shape, inputs[1].shape, y_true[0].shape, y_true[1].shape)
        return inputs[0].squeeze(-1), inputs[1].squeeze(-1), y_true[0].squeeze(-1), y_true[1].squeeze(0).transpose(2, 0, 1)
    
class CompData(Dataset):
    def __init__(self, file_list):
        print('CompData init')
        # self.generator = generator
        self.length = len(file_list)
        self.file_list = file_list
        self.load_file = vxm.py.utils.load_volfile 

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        source = self.load_file(self.file_list[index][0], add_batch_axis=True, add_feat_axis=False)
        source_label = self.load_file(self.file_list[index][1], add_batch_axis=True, add_feat_axis=False)
        target_id = np.random.randint(0, self.length, [1], dtype=np.int32)[0]
        target = self.load_file(self.file_list[target_id][0], add_batch_axis=True, add_feat_axis=False)
        target_label = self.load_file(self.file_list[target_id][1], add_batch_axis=True, add_feat_axis=False)
        return source, target, source_label, target_label
