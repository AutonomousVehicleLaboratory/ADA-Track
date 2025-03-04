from mmdet3d.models import build_detector
import mmcv
import os
from mmcv import Config, DictAction
from torch.utils.data import Dataset, DataLoader
import torch
from skimage import io, transform
import numpy as np
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.apis import single_gpu_test
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.datasets.pipelines import Compose

# Specify paths
config_file = './plugin/configs/ada_track_detr3d.py'
checkpoint_file = './ckpt/ada_track_detr3d_epoch24.pth'
img_folder = './data/rosbag/2024-02-16-21-42-46_0/avt_cameras_camera1_image_rect_color_compressed/'
output_folder = './data/rosbag/2024-02-16-21-42-46_0/ada-results'

cfg = Config.fromfile(config_file)
if hasattr(cfg, 'plugin') & cfg.plugin:
    import importlib
    if hasattr(cfg, 'plugin_dir'):
        plugin_dir = cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
        _module_dir = _module_dir.split('/')
        _module_path = _module_dir[0]

        for m in _module_dir[1:]:
            _module_path = _module_path + '.' + m
        print(_module_path)
        plg_lib = importlib.import_module(_module_path)
    else:
        # import dir is the dirpath for the config file
        _module_dir = os.path.dirname(config_file)
        _module_dir = _module_dir.split('/')
        _module_path = _module_dir[0]
        for m in _module_dir[1:]:
            _module_path = _module_path + '.' + m
        print(_module_path)
        plg_lib = importlib.import_module(_module_path)

# Initialize the detector
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

class UCSDDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, pipeline_single=None, pipeline_post=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        if pipeline_single is not None:
            self.pipeline_single = Compose(pipeline_single)

        if pipeline_post is not None:
            self.pipeline_post = Compose(pipeline_post)
        self.file_names = [f for f in os.listdir(self.root_dir)]

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,self.file_names[idx])
        result = {}
        result['img_filename'] = img_name
        # image = io.imread(img_name)

        # if self.transform:
        #     sample = self.transform(sample)
        # time_stamp = self.file_names[idx].split['j'][0]
        # img_metas = {}

        return result
print('dataset')
dataset = UCSDDataset(root_dir=img_folder)
print('dataloader')
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
# outputs = single_gpu_test(model, data_loader, False, './results')
model.eval()
results = []
print('inference')
for i, data in enumerate(data_loader):
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    results.extend(result)

    # batch_size = len(result)
    # for _ in range(batch/
print(results)

