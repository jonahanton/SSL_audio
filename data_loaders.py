import multiprocessing as mp

from data_manager.audioset import AudioSetLoader
from utils import utils 


# load training params from .ymal config file
cfg = utils.load_yaml_config('./configs/pretrain/config.yaml')

cpu_count = mp.cpu_count()
print(f'cpu count: {cpu_count}')

print(f'Loading AudioSet dataloader with num_workers={cpu_count}')
data_loader = AudioSetLoader(
    cfg,
    pretrain=True,
    balanced_only=cfg.data.audioset.balanced_only,
    num_workers=cpu_count,
).get_loader() 