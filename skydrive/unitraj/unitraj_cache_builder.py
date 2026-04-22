""" 
Build UniTraj training and validation cache for the Songdo dataset.

This is for cache-building only, after that, train with UniTraj and set the data path as in the bottom of this script.
The cache is model-specific because many model configs include data related settings, such as `max_num_agents`.
Training on the cache of a different config may cause errors.
It's recommmended to keep separate cache folders for different models if you want to compare them, e.g., `--cache-path ./cache/autobot` and `-cache-path ./cache/wayformer`.

Unitraj locates the cache files using relative path from `UNITRAJ_ROOT/unitraj`, but SkyDrive starts from the project root.
The cache stored to `SKYDRIVE_ROOT/cache` should be mapped to `UNITRAJ_ROOT/unitraj/cache` when training with UniTraj (e.g., via symbolic link)

Example usage:
python skydrive/unitraj/songdo_cache_builder.py \
    --method autobot \
    --train-data-path ./datasets/songdo_drive/train \
    --val-data-path ./datasets/songdo_drive/test \
    --unitraj-config-dir ../UniTraj/unitraj/configs \
    --cache-path ./cache/autobot \
    --overwrite-cache \
"""
import argparse
import os
import pickle
import shutil
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

from skydrive.common.songdo_scene_loader import SongdoSceneLoader
from skydrive.preprocess.common import SESSION_FRAMES_DIRNAME
from skydrive.unitraj.unitraj_converter import UniTrajConverter
from unitraj.datasets.base_dataset import BaseDataset
from unitraj.datasets.common_utils import is_ddp

# DEBUG = True

class SongdoCacheBuilder(BaseDataset):

    def __init__(self, config=None, method_name=None, is_validation=False, num_workers=16):
        self.num_workers = num_workers
        self.dataset_name = "songdo_drive"
        self.method_name = method_name
        super().__init__(config, is_validation)

    def get_songdo_session_files(self, data_path: Path) -> tuple[Path, str, list[str]]:
        """Resolve one Songdo split and return the session files to cache."""

        session_dir = data_path / SESSION_FRAMES_DIRNAME
        session_files = [path.name for path in sorted(session_dir.glob('*.pkl'))]

        return session_files

    def load_data(self):
        """ Similar to the load_data function in BaseDataset, but read data list differently """
        self.data_loaded = {}
        if self.is_validation:
            print('Loading validation data...')
        else:
            print('Loading training data...')

        for cnt, data_path in enumerate(self.data_path):
            # example data_path ./datasets/songdo_drive/train
            data_path = Path(data_path)
            session_files = self.get_songdo_session_files(data_path)
            split_name = data_path.parts[-1]  # 'train' or 'test'
            # to keep the cache path consistent with unitraj basedataset
            self.cache_path = os.path.join(self.config['cache_path'], data_path.parts[-1], data_path.parts[-2])

            data_usage_this_dataset = self.config['max_data_num'][cnt]
            self.starting_frame = self.config['starting_frame'][cnt]
            if self.config['use_cache'] or is_ddp():
                file_list = self.get_data_list(data_usage_this_dataset)
            else:
                if os.path.exists(self.cache_path) and self.config.get('overwrite_cache', False) is False:
                    print('Warning: cache path {} already exists, skip '.format(self.cache_path))
                    file_list = self.get_data_list(data_usage_this_dataset)
                else:
                    if os.path.exists(self.cache_path):
                        shutil.rmtree(self.cache_path)
                    os.makedirs(self.cache_path, exist_ok=True)
                    process_num = min(self.num_workers, os.cpu_count() // 2)
                    print('Using {} processes to load data...'.format(process_num))

                    # if DEBUG:
                    #     session_files = session_files[:2*process_num] # for debug
                    data_splits = np.array_split(session_files, process_num)
                    data_splits = [(split_name, list(data_splits[i]))for i in range(process_num)]
                    os.makedirs('tmp/{}'.format(self.method_name), exist_ok=True)
                    for i in range(process_num):
                        with open(os.path.join('tmp/{}'.format(self.method_name), '{}.pkl'.format(i)), 'wb') as f:
                            pickle.dump(data_splits[i], f)

                    # results = self.process_data_chunk(0) # for debug
                    with Pool(processes=process_num) as pool:
                        results = pool.map(self.process_data_chunk, list(range(process_num)))

                    file_list = {}
                    for result in results:
                        file_list.update(result)

                    with open(os.path.join(self.cache_path, 'file_list.pkl'), 'wb') as f:
                        pickle.dump(file_list, f)
                    
                    # delete the temporary files one the file list is saved
                    shutil.rmtree('tmp/{}'.format(self.method_name))

                    data_list = list(file_list.items())
                    np.random.shuffle(data_list)
                    if not self.is_validation:
                        file_list = dict(data_list[:data_usage_this_dataset])

            print('Loaded {} samples from {}'.format(len(file_list), data_path))
            self.data_loaded.update(file_list)

        self.data_loaded_keys = list(self.data_loaded.keys())
        print('Data loaded')


    def process_data_chunk(self, worker_index):
        """Cache one chunk of Songdo monitoring sessions into the standard HDF5 format."""
        with open(os.path.join('tmp/{}'.format(self.method_name), '{}.pkl'.format(worker_index)), 'rb') as f:
            data_chunk = pickle.load(f)
        file_list = {}
        split_name, session_files = data_chunk
        scene_loader = SongdoSceneLoader()
        converter = UniTrajConverter()
        hdf5_path = os.path.join(self.cache_path, f'{worker_index}.h5')

        with h5py.File(hdf5_path, 'w') as f:
            for cnt, session_filename in enumerate(session_files):
                if worker_index == 0 and cnt % max(int(len(session_files) / 10), 1) == 0:
                    print(f'{cnt}/{len(session_files)} sessions processed', flush=True)
                session_frames, veh_time_pairs, metadata = scene_loader.load_session(split_name, session_filename)

                for scene_index, ego_info in enumerate(veh_time_pairs):
                    try:
                        # replaces the original preprocess
                        scene = scene_loader.build_scene_from_session(session_frames, ego_info, metadata)
                        internal_format = converter.convert_scene(scene)

                        output = self.process(internal_format)

                        output = self.postprocess(output)

                    except Exception as e:
                        print('Warning: {} in {} scene {}'.format(e, session_filename, scene_index))
                        output = None

                    if output is None:
                        continue

                    kalman_difficulty = np.stack([x['kalman_difficulty'] for x in output])
                    for output_index, record in enumerate(output):
                        # in each session, we can generate multiple scenes by selecting a time window and an ego vehicle
                        # in each scene, unitraj generates multiple outputs by centering on different vehicles
                        grp_name = f'{self.dataset_name}-{split_name}-{worker_index}-{cnt}-{scene_index}-{output_index}'
                        grp = f.create_group(grp_name)
                        for key, value in record.items():
                            if isinstance(value, str):
                                value = np.bytes_(value)
                            grp.create_dataset(key, data=value)
                        file_list[grp_name] = {
                            'kalman_difficulty': kalman_difficulty,
                            'h5_path': hdf5_path,
                        }
                    del output
                del session_frames
                del veh_time_pairs

        return file_list


if __name__ == '__main__':
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description='Build UniTraj cache files for the Songdo dataset.')
    parser.add_argument(
        '--unitraj-config-dir',
        type=Path,
        default=Path('../UniTraj/unitraj/configs'),
        help='Path to the Unitraj configs directory.',
    )
    parser.add_argument(
        '--train-data-path',
        nargs='+',
        default=['./datasets/songdo_drive/train'],
        help='One or more Songdo training split roots.',
    )
    parser.add_argument(
        '--val-data-path',
        nargs='+',
        default=['./datasets/songdo_drive/test'],
        help='One or more Songdo validation split roots.',
    )
    parser.add_argument(
        '--cache-path',
        type=str,
        default='./cache',
        help='Path to save the generated cache files.',
    )
    parser.add_argument(
        '--overwrite-cache',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Whether to overwrite any existing cache directory.',
    )
    parser.add_argument(
        '--method',
        default='autobot',
        type=str,
        help='Unitraj method config name under configs/method, for example autobot or wayformer.',
        choices=['autobot', 'wayformer', 'MTR']
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=16,
        help='Number of worker processes to use.',
    )
    args = parser.parse_args()
    print('Arguments parsed:')
    print(args)

    unitraj_config_dir = Path(args.unitraj_config_dir)
    method_name = args.method

    config = OmegaConf.load(unitraj_config_dir / 'config.yaml')
    OmegaConf.set_struct(config, False)
    
    config.method = OmegaConf.load(unitraj_config_dir / 'method' / f'{method_name}.yaml')
    config = OmegaConf.merge(config, config.method)

    config.overwrite_cache = args.overwrite_cache
    config.train_data_path = args.train_data_path
    config.val_data_path = args.val_data_path
    config.cache_path = args.cache_path
    config.only_train_on_ego = True

    print('Creating training cache for method {}...'.format(method_name))
    builder = SongdoCacheBuilder(config=config, method_name=method_name, is_validation=False, num_workers=args.num_workers)
    print('Creating validation cache for method {}...'.format(method_name))
    builder = SongdoCacheBuilder(config=config, method_name=method_name, is_validation=True, num_workers=args.num_workers)
