""" Animate a scene in Songdo Drive dataset, and visualize converted scene with UniTraj's functions.

	Example usage: python -m skydrive.unitraj.viz 

	Files will be saved to `figures/animations`
"""

import copy
import math
from pathlib import Path
from typing import Union

import numpy as np

import matplotlib.pyplot as plt
from skydrive.common.scene_viz import animate_scene


def load_unitraj_visualization_config(
	unitraj_config_dir: Union[str, Path],
	method_name: str,
	past_len: int,
	future_len: int,
):
	"""Load a UniTraj config for building centered scene-check inputs."""
	from omegaconf import OmegaConf

	unitraj_config_dir = Path(unitraj_config_dir)
	config = OmegaConf.load(unitraj_config_dir / 'config.yaml')
	OmegaConf.set_struct(config, False)
	config.method = OmegaConf.load(unitraj_config_dir / 'method' / f'{method_name}.yaml')
	config = OmegaConf.merge(config, config.method)
	config.past_len = past_len
	config.future_len = future_len
	return config


def _build_unitraj_scene_check_records(converted_scene: dict, unitraj_config) -> list[dict]:
	"""Run one converted Songdo scene through UniTraj's centered-scene processing."""
	from unitraj.datasets.base_dataset import BaseDataset

	class _VisualizationDataset(BaseDataset):
		def __init__(self, config) -> None:
			self.config = config
			self.is_validation = True
			self.data_path = []
			self.data_loaded_memory = []
			self.file_cache = {}

	processed = _VisualizationDataset(unitraj_config).process(copy.deepcopy(converted_scene))
	if not processed:
		print('Warning: No tracks to predict in the converted scene, skipping')
		return None
	return processed


def save_unitraj_scene_checks(
	converted_scene: dict,
	save_path: Union[str, Path],
	unitraj_config,
	max_samples: int = 4,
) -> None:
	"""Save UniTraj's built-in centered-scene sanity-check plots for one scenario."""
	from unitraj.utils.visualization import check_loaded_data

	if max_samples <= 0:
		raise ValueError('max_samples must be positive.')

	processed_samples = _build_unitraj_scene_check_records(converted_scene, unitraj_config)
	if processed_samples is None:
		return
	
	scene_ego_id = str(converted_scene['track_infos']['object_id'][converted_scene['sdc_track_index']])
	processed_samples.sort(key=lambda sample: str(sample['center_objects_id']) != scene_ego_id)
	processed_samples = processed_samples[:min(max_samples, len(processed_samples))]

	num_samples = len(processed_samples)
	num_cols = min(2, num_samples)
	num_rows = math.ceil(num_samples / num_cols)
	fig, axes = plt.subplots(
		num_rows,
		num_cols,
		figsize=(6.5 * num_cols, 6.5 * num_rows),
		dpi=120,
		squeeze=False,
	)
	axes_flat = axes.ravel()
	for axis_index, ax in enumerate(axes_flat):
		if axis_index >= num_samples:
			ax.axis('off')
			continue

		sample = processed_samples[axis_index]
		center_object_id = str(sample['center_objects_id'])
		plt.sca(ax)
		check_loaded_data(plt, sample)
		title = f'center={center_object_id}'
		if center_object_id == scene_ego_id:
			title += ' (scene ego)'
		ax.set_title(title, fontsize=10)

	fig.suptitle(converted_scene['scenario_id'], fontsize=11)
	fig.tight_layout(rect=(0, 0, 1, 0.97))
	fig.savefig(save_path, bbox_inches='tight')
	plt.close(fig)


if __name__ == '__main__':
	import argparse

	from skydrive.common.songdo_scene_loader import SongdoSceneLoader
	from skydrive.preprocess.common import DATASET_FRAME_RATE, PROCESSED_FOLDER, FIGURE_FOLDER
	from skydrive.unitraj.unitraj_converter import UniTrajConverter

	parser = argparse.ArgumentParser(description='Visualize a Songdo scene through the full pipeline.')
	parser.add_argument('--split', type=str, default='train', help='Dataset split name (train or test).')
	parser.add_argument('--session', type=str, default='2022-10-04_A_AM1.pkl', help='Session filename.')
	parser.add_argument('--processed-folder', type=str, default=str(PROCESSED_FOLDER), help='Root of preprocessed data.')
	parser.add_argument('--output-dir', type=str, default=FIGURE_FOLDER/'animations', help='Directory to save outputs.')
	parser.add_argument('--target-fps', type=int, default=10, help='Target frame rate for the converted scene.')
	parser.add_argument(
		'--unitraj-config-dir',
		type=Path,
		default=Path('../UniTraj/unitraj/configs'),
		help='Path to the UniTraj configs directory.',
	)
	parser.add_argument(
		'--unitraj-method',
		type=str,
		default='autobot',
		choices=['autobot', 'wayformer', 'MTR'],
		help='UniTraj method config used to build centered scene-check inputs.',
	)
	parser.add_argument(
		'--unitraj-check-count',
		type=int,
		default=4,
		help='Number of UniTraj scene-check panels to save for each converted scene.',
	)
	parser.add_argument('--box-size', type=float, default=120.0, help='Bounding box size in meters for the animation.')
	parser.add_argument('--num-scenes', type=int, default=20, help='Number of random scene indices to visualize from the session.')
	parser.add_argument('--random-seed', type=int, default=0, help='Random seed used to sample scene indices.')
	args = parser.parse_args()

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# 1. Load session
	loader = SongdoSceneLoader(processed_folder=args.processed_folder)
	session_frames, veh_time_pairs, metadata = loader.load_session(args.split, args.session)
	print(f'Loaded session {args.session}: {len(session_frames)} frames, {len(veh_time_pairs)} ego segments')

	rng = np.random.default_rng(args.random_seed)
	scene_indices = sorted(
		rng.choice(len(veh_time_pairs), size=min(args.num_scenes, len(veh_time_pairs)), replace=False).tolist()
	)
	print(f'Selected scene indices: {scene_indices}')

	# 2. Convert sampled scenes to the UniTraj internal format and visualize them
	converter = UniTrajConverter(
		source_frame_rate=DATASET_FRAME_RATE,
		target_frame_rate=args.target_fps,
	)
	unitraj_visualization_config = load_unitraj_visualization_config(
		unitraj_config_dir=args.unitraj_config_dir,
		method_name=args.unitraj_method,
		past_len=converter.history_frames + 1,
		future_len=converter.future_frames,
	)
	session_stem = Path(args.session).stem
	for scene_index in scene_indices:
		ego_info = veh_time_pairs[scene_index]
		scene = loader.build_scene_from_session(session_frames, ego_info, metadata)
		print(f'Built scene {scene_index}: ego vehicle {scene["ego_info"]["Vehicle_ID"]}, {len(scene["frames"])} frames')

		converted = converter.convert_scene(scene)
		print(f'Converted scene {scene_index}: {converted["scenario_id"]}, '
			f'{converted["track_infos"]["trajs"].shape[0]} tracks, '
			f'{len(converted["tracks_to_predict"]["track_index"])} tracks to predict')

		unitraj_checks_path = output_dir / f'{session_stem}_scene{scene_index}_unitraj_checks.png'
		save_unitraj_scene_checks(
			converted,
			save_path=unitraj_checks_path,
			unitraj_config=unitraj_visualization_config,
			max_samples=args.unitraj_check_count,
		)
		print(f'Saved UniTraj scene checks to {unitraj_checks_path}')

		gif_path = output_dir / f'{session_stem}_scene{scene_index}.gif'
		animate_scene(scene, save_path=gif_path, box_size_m=args.box_size, fps=DATASET_FRAME_RATE)
		print(f'Saved animation to {gif_path}')
