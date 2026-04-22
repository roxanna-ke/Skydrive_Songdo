import pickle
from pathlib import Path

import numpy as np
from skydrive.preprocess.common import (
	EGO_VEHICLE_INFO_DIRNAME,
	INTERSECTION_METADATA_PATH,
	PROCESSED_FOLDER,
	SESSION_FRAMES_DIRNAME,
)


class _CompatUnpickler(pickle.Unpickler):
	"""Load NumPy pickles saved from either NumPy 1.x or 2.x."""

	def find_class(self, module, name):
		if module.startswith('numpy._core'):
			module = module.replace('numpy._core', 'numpy.core', 1)
		return super().find_class(module, name)


class SongdoSceneLoader:
	"""Read saved Songdo session artifacts and slice them into scene dictionaries."""
	def __init__(self, processed_folder=PROCESSED_FOLDER):
		"""Store the processed dataset root written by ``process_songdo_traffic.py``."""
		self.processed_folder = Path(processed_folder)
	
	def _load_pickle(self, path):
		"""Read one preprocessing artifact from disk with NumPy-version-compatible pickle loading."""
		with path.open('rb') as fp:
			return _CompatUnpickler(fp).load()

	def load_session(self, split_name, session_filename):
		"""Read one monitoring session together with its saved ego segments and map data."""
		session_stem = Path(session_filename).stem
		session_frames = self._load_pickle(
			self.processed_folder / split_name / SESSION_FRAMES_DIRNAME / f'{session_stem}.pkl'
		)
		veh_time_pairs = self._load_pickle(
			self.processed_folder / split_name / EGO_VEHICLE_INFO_DIRNAME / f'{session_stem}.pkl'
		)
		date, intersection, session = session_stem.split('_')
		metadata = self._load_pickle(self.processed_folder / INTERSECTION_METADATA_PATH)[intersection]
		metadata['date'] = date
		metadata['intersection'] = intersection
		metadata['session'] = session
		return (
			session_frames,
			veh_time_pairs,
			metadata,
		)

	@staticmethod
	def build_scene_from_session(session_frames, ego_info, metadata):
		"""Slice one ego segment from saved session frames using precomputed frame timestamps."""
		ego_drone_id = int(ego_info['Drone_ID'])
		session_frames = [ frame for frame in session_frames if int(frame['Drone_ID']) == ego_drone_id ]

		# this is not most efficient way, but leave it here for readability
		frame_time_ms = np.asarray([frame['Local_Time'] for frame in session_frames]).astype(np.int64)
		start_time = ego_info['Start_Time'].to_timedelta64().astype('timedelta64[ms]')
		end_time = ego_info['End_Time'].to_timedelta64().astype('timedelta64[ms]')
		start_time_ms = int(start_time.astype(np.int64))
		end_time_ms = int(end_time.astype(np.int64))
		start_idx = int(np.searchsorted(frame_time_ms, start_time_ms, side='left'))
		end_idx = int(np.searchsorted(frame_time_ms, end_time_ms, side='left'))

		return {
			'ego_info': {
				'Vehicle_ID': int(ego_info['Vehicle_ID']),
				'Drone_ID': ego_drone_id,
				'Start_Time': start_time,
				'End_Time': end_time,
			},
			'frames': session_frames[start_idx:end_idx],
			'metadata': metadata,
		}


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Load one Songdo session and build one ego scene.')
	parser.add_argument('--split', type=str, default='train', help='Dataset split name (train or test).')
	parser.add_argument('--session', type=str, default='2022-10-04_A_AM1.pkl', help='Session filename.')
	parser.add_argument(
		'--processed-folder',
		type=Path,
		default=PROCESSED_FOLDER,
		help='Root of preprocessed data.',
	)
	parser.add_argument('--scene-index', type=int, default=0, help='Index into the ego scene list of the session.')
	args = parser.parse_args()

	loader = SongdoSceneLoader(processed_folder=args.processed_folder)
	session_frames, veh_time_pairs, metadata = loader.load_session(args.split, args.session)
	print(f'Loaded session {args.session}: {len(session_frames)} frames, {len(veh_time_pairs)} ego segments')

	ego_info = veh_time_pairs[args.scene_index]
	scene = loader.build_scene_from_session(session_frames, ego_info, metadata)
	print(f'Built scene {args.scene_index}: ego vehicle {scene["ego_info"]["Vehicle_ID"]}, {len(scene["frames"])} frames')
	print(
		f'Scene window: drone {scene["ego_info"]["Drone_ID"]}, '
		f'{scene["ego_info"]["Start_Time"]} -> {scene["ego_info"]["End_Time"]}'
	)
