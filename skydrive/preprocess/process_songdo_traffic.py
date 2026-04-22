import argparse
import json
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from common import (
	DATASET_COLUMN_DTYPES,
	DATASET_FRAME_RATE,
	EGO_VEHICLE_INFO_DIRNAME,
	PROCESSED_FOLDER,
	RAW_DATASET_FOLDER,
	SESSION_FRAMES_DIRNAME,
	TRAIN_TEST_SPLIT_PATH,
	get_csv_path
)

# looks like we need xy smoothing but less smoothing on heading
SMOOTH_XY = True
SMOOTH_HEADING = False
SUMMARY_SPLITS = ('train', 'test')


def _new_summary_info() -> dict[str, object]:
	return {
		'session_count': 0,
		'segment_count': 0,
		'total_time_original_minutes': 0.0,
		'total_time_segments_minutes': 0.0,
		'valid_vehicle_count': 0,
		'bad_traj_counts': Counter(),
	}


def _merge_summary_info(summary: dict[str, object], info: dict[str, object]) -> None:
	summary['session_count'] += int(info['session_count'])
	summary['segment_count'] += int(info['segment_count'])
	summary['total_time_original_minutes'] += float(info['total_time_original_minutes'])
	summary['total_time_segments_minutes'] += float(info['total_time_segments_minutes'])
	summary['valid_vehicle_count'] += int(info['valid_vehicle_count'])
	summary['bad_traj_counts'].update(info['bad_traj_counts'])


def _print_summary(label: str, info: dict[str, object]) -> None:
	print(
		f"{label}: "
		f"{info['session_count']} sessions, "
		f"{info['segment_count']} ego segments, "
		f"{info['valid_vehicle_count']} valid vehicles, "
		f"{info['total_time_original_minutes']:.2f} minutes original, "
		f"{info['total_time_segments_minutes']:.2f} minutes valid segments, "
		f"filtered {dict(info['bad_traj_counts'])}"
	)

def load_split_tasks(split_path: Path) -> list[tuple[str, str]]:
	"""Read the train/test split file and return ordered `(split_name, session_filename)` tasks.
		for example, an element can be ('train', '2022-10-04_A_AM1.csv'), which means we load the csv 
		file `2022-10-04_A_AM1.csv` and save its session artifacts into the 'train' split folders.
	"""
	with split_path.open('r', encoding='utf-8') as fp:
		split_payload = json.load(fp)

	tasks: list[tuple[str, str]] = []
	for split_name in ('train', 'test'):
		tasks.extend((split_name, session_filename) for session_filename in split_payload[split_name])
	return tasks

def read_songdo_csv(
	file_path: Path,
	add_heading_speed: bool = True,
	required_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
	"""Read one Songdo CSV with shared dtypes and optionally append heading and velocity features."""
	if required_columns is None:
		required_columns = list(DATASET_COLUMN_DTYPES.keys())
		data_types = DATASET_COLUMN_DTYPES
	else:
		data_types = {column: DATASET_COLUMN_DTYPES[column] for column in required_columns}

	df = pd.read_csv(file_path, dtype=data_types, usecols=required_columns)
	df = df.sort_values(by='Local_Time').reset_index(drop=True)
	df['Local_Time'] = pd.to_timedelta(df['Local_Time']).astype('timedelta64[ms]')
	df = keep_dominant_drone_per_vehicle(df)
	# after resolving overlapping drone views, keep the first repeated row on the 
	# same vehicle/timestamp from the chosen drone stream.
	df = df.drop_duplicates(subset=['Vehicle_ID', 'Local_Time'], keep='first')

	if add_heading_speed:
		df = calculate_heading_speed(df)

	return df


def keep_dominant_drone_per_vehicle(df: pd.DataFrame) -> pd.DataFrame:
	"""Keep the drone stream with more valid detections for each vehicle.

	Some vehicles appear in overlapping drone views with subtly different frame timestamps. 
	Downstream scene slicing assumes one consistent frame timeline per vehicle, so we keep 
	the drone that contributes more valid detections for that vehicle and break ties by 
	smaller ``Drone_ID`` for deterministic output.
	"""
	valid_rows = df[df['Local_X'].notna() & df['Local_Y'].notna()]

	dominant_drone = (
		valid_rows.groupby(['Vehicle_ID', 'Drone_ID'], sort=False)
		.size() # count valid detections for each (Vehicle_ID, Drone_ID) pair
		.rename('valid_detection_count')
		.reset_index()
		.sort_values(['Vehicle_ID', 'valid_detection_count', 'Drone_ID'], ascending=[True, False, True])
		.drop_duplicates(subset=['Vehicle_ID'], keep='first') # keep the first row for each Vehicle_ID, which has the max valid_detection_count
		.set_index('Vehicle_ID')['Drone_ID']
	)
	keep_mask = df['Drone_ID'].eq(df['Vehicle_ID'].map(dominant_drone))
	return df[keep_mask].reset_index(drop=True)


def calculate_heading_speed(
	df: pd.DataFrame,
	smooth_window: int = 5,
	speed_threshold: float = 3.0,
) -> pd.DataFrame:
	"""Estimate heading and velocity by smoothing within each vehicle timeline and respecting gaps."""
	session_times = pd.TimedeltaIndex(df['Local_Time'].sort_values().drop_duplicates())

	def _heading_speed_func(group: pd.DataFrame) -> pd.DataFrame:
		group = group.sort_values('Local_Time')
		# Smooth on the original timeline so missing detections remain gaps inside each rolling window.
		out = pd.DataFrame({'Heading': np.nan, 'Speed': np.nan, 'Vx': np.nan, 'Vy': np.nan}, index=group.index)
		start_idx = session_times.searchsorted(group['Local_Time'].iloc[0], side='left')
		end_idx = session_times.searchsorted(group['Local_Time'].iloc[-1], side='right')
		vehicle_times = session_times[start_idx:end_idx]
		group = group.assign(_orig_index=group.index).set_index('Local_Time').reindex(vehicle_times)
		group.index.name = 'Local_Time'
		valid_mask = group['Local_X'].notna() & group['Local_Y'].notna()
		if valid_mask.sum() < 2:
			return out

		x = group['Local_X']
		y = group['Local_Y']
		if SMOOTH_XY:
			x = x.rolling(window=smooth_window, center=True, min_periods=1).mean()
			y = y.rolling(window=smooth_window, center=True, min_periods=1).mean()
		valid_index = group.loc[valid_mask, '_orig_index'].astype(int)

		x = x[valid_mask]
		y = y[valid_mask]
		t = group.index[valid_mask].total_seconds().to_numpy(dtype=np.float64)
		vx = np.gradient(x.to_numpy(dtype=np.float64), t)  # meters per second
		vy = np.gradient(y.to_numpy(dtype=np.float64), t)  # meters per second
		heading = np.arctan2(vy, vx)

		# if SMOOTH_HEADING:
		# 	heading = np.unwrap(heading)
		# 	heading = pd.Series(heading, index=valid_index).ewm(span=smooth_window, adjust=False).mean().to_numpy()
		# 	heading = np.arctan2(np.sin(heading), np.cos(heading))

		speed = np.sqrt(vx**2 + vy**2)
		heading[speed < speed_threshold] = np.nan
		heading = pd.Series(heading, index=valid_index).ffill().bfill().to_numpy()

		out.loc[valid_index, 'Heading'] = heading
		out.loc[valid_index, 'Speed'] = speed
		out.loc[valid_index, 'Vx'] = vx
		out.loc[valid_index, 'Vy'] = vy
		return out

	out = df.groupby('Vehicle_ID', sort=False).apply(_heading_speed_func, include_groups=False).reset_index(level=0, drop=True)
	df[['Heading', 'Speed', 'Vx', 'Vy']] = out[['Heading', 'Speed', 'Vx', 'Vy']]
	return df


def _sample_relative_displacement(
	positions: np.ndarray,
	num_points: int = 9,
) -> np.ndarray:
	"""Sample one ego path and remove translation so nearby leader/follower tracks can be compared."""
	sample_indices = np.round(np.linspace(0, len(positions) - 1, num=num_points)).astype(np.int64)
	sampled_positions = positions[sample_indices]
	return sampled_positions - sampled_positions[:1]


def _time_overlap_ratio(
	start_a: pd.Timedelta,
	end_a: pd.Timedelta,
	start_b: pd.Timedelta,
	end_b: pd.Timedelta,
) -> float:
	"""Return temporal overlap normalized by the shorter fixed-duration segment."""
	overlap = min(end_a, end_b) - max(start_a, start_b)
	if overlap <= pd.Timedelta(0):
		return 0.0
	return float(overlap / min(end_a - start_a, end_b - start_b))


def _filter_duplicate_ego_segments(
	veh_time_pairs: list[dict],
	overlap_ratio_threshold: float = 0.75,
	distance_threshold_m: float = 15.0,
	shape_threshold_m: float = 3.0,
) -> tuple[list[dict], int]:
	""" Filter out the segments if they meet the following conditions at the same time:
		1. belong to two different vehicles
		2. time stamp overlaps more than `overlap_ratio_threshold` (0.75 by default)
		3. mid-point distance less than `distance_threshold_m` (15m by default)
		4. maximum distance between evenly sampled 9 points is less than `shape_threshold_m` (3m by default)

		Typical cases are when two vehicles follow each other, or they are in parallel and close to each other.
	"""
	sorted_pairs = sorted(veh_time_pairs,key=lambda pair: (pair['Start_Time'], pair['Vehicle_ID']))
	filtered_pairs: list[dict] = []
	active_pairs: list[dict] = []
	removed_count = 0

	for pair in sorted_pairs:
		is_duplicate = False
		for active_pair in active_pairs:
			if active_pair['Vehicle_ID'] == pair['Vehicle_ID']:
				continue
			time_overlap = _time_overlap_ratio(pair['Start_Time'], pair['End_Time'], active_pair['Start_Time'], active_pair['End_Time'])
			if time_overlap < overlap_ratio_threshold:
				continue
			if np.linalg.norm(pair['_mid_point'] - active_pair['_mid_point']) > distance_threshold_m:
				continue

			traj_diff = np.linalg.norm(pair['_rel_disp'] - active_pair['_rel_disp'], axis=1).max()
			if traj_diff <= shape_threshold_m:
				is_duplicate = True
				break

		if is_duplicate:
			removed_count += 1
			continue

		filtered_pairs.append(pair)
		active_pairs.append(pair)

	return filtered_pairs, removed_count


def select_ego_vehicle_traj(
	session_df: pd.DataFrame,
	segment: float = 8.0,
	pad: float = 1.0,
	step_size: float = 4.0,
	speed_threshold_kmh: float = 5.0,
	break_tolerance: float = 0.04,
) -> tuple[list[dict], dict[str, object]]:
	"""
	From one monitoring session, select car trajectories and split them into fixed-duration segments like below.

	```
	# |------------------ trajectory duration ------------------------|
	# |<-pad->|<-segment 1->|<-pad->|---------------------------------|
	# |<-pad->|<-step_size x n ->|<-segment n+1->|<-pad->|------------|
	```

	We filter out trajectories that are:
		1. too short to fit one segment
		2. vehicle locations are missing for more than `break_tolerance` seconds within any segment
		3. another vehicle already yields a near-identical overlapping segment on the same lane

	Args:
		session_df: All vehicle trajectories in one monitoring session.
		segment: Segment length in seconds.
		pad: Padding time in seconds kept at both trajectory ends.
		step_size: Time step in seconds between neighboring segment starts. Default to 4.0s, which means the input windows (4s) are not overlapping.
		speed_threshold_kmh: The vehicle's max speed should be higher than this, otherwise it will be considered invalid (not moving)
		break_tolerance: Allowed temporal gap in seconds when checking trajectory continuity. Default 0.04s. By default, do not allow any missing frame in ego trajectories, video is 30 FPS so normal frame time difference = 0.033s, but inconsistency exists.
	Returns:
		Tuple ``(veh_time_pairs, info)`` where ``veh_time_pairs`` is the list of valid ego segments and ``info``
		contains aggregate filtering statistics for the session.

		The segment should contain `segment * DATASET_FRAME_RATE + 1` frames if there is no missing frame in the original video. 
		This allows us to split the segment into a historical window, a prediction window and 1 frame to be considered as "current_time"
	"""

	segment_frame_count = int(round(segment * DATASET_FRAME_RATE)) + 1
	segment = pd.Timedelta(seconds=segment).as_unit('ms')
	pad = pd.Timedelta(seconds=pad).as_unit('ms')
	step = pd.Timedelta(seconds=step_size).as_unit('ms')
	break_tolerance = pd.Timedelta(seconds=break_tolerance).as_unit('ms')
	veh_time_pairs: list[dict] = []

	# keep cars only.
	session_df = session_df[session_df['Vehicle_Class'] == 0]
	vehicle_id_groups = session_df.groupby('Vehicle_ID', sort=False)
	typical_speed = vehicle_id_groups['Vehicle_Speed'].quantile(0.5).fillna(0.0)
	valid_vehicle_ids = typical_speed[typical_speed > speed_threshold_kmh].index

	bad_traj = defaultdict(lambda: 0)

	for vehicle_id in valid_vehicle_ids:
		vehicle_df = vehicle_id_groups.get_group(vehicle_id).sort_values('Local_Time').reset_index(drop=True)
		vehicle_drone_id = int(vehicle_df['Drone_ID'].iat[0])

		traj_time_index = pd.TimedeltaIndex(vehicle_df['Local_Time'])
		traj_start = traj_time_index[0]
		traj_end = traj_time_index[-1]
		traj_duration = traj_end - traj_start
	
		num_segments = (traj_duration - (2* pad + segment)) // step + 1
		if num_segments <= 0:
			bad_traj["short_traj"] += 1
			continue
		
		seg_start_times = [traj_start + pad + step * n for n in range(num_segments)]

		missing_frame = vehicle_df['Local_Time'].diff() > break_tolerance
		for seg_start in seg_start_times:
			start_idx = int(traj_time_index.searchsorted(seg_start, side='left'))
			end_idx = start_idx + segment_frame_count
			segment_vehicle_df = vehicle_df.iloc[start_idx:end_idx]

			# there are missing locations in the trajectory
			# Don't confuse with `.loc[]`, which is label-based and includes the end index
			if missing_frame.iloc[start_idx:end_idx].any():
				bad_traj["missing_frame"] += 1
				continue
			if segment_vehicle_df[['Local_X', 'Local_Y']].isna().any().any():
				bad_traj["invalid_location"] += 1
				continue
			if segment_vehicle_df[['Vehicle_Length', 'Vehicle_Width']].isna().any().any():
				bad_traj["invalid_size"] += 1
				continue
			if segment_vehicle_df[['Heading', 'Speed', "Vx", "Vy"]].isna().any().any():
				bad_traj["invalid_velocity"] += 1
				continue
			# check the number of frames match required
			if len(segment_vehicle_df) != segment_frame_count:
				bad_traj["wrong_frame_count"] += 1
				continue

			# the two indexes above works for the good cases, but we need to filter out the bad cases
			seg_start = traj_time_index[start_idx]
			seg_end = traj_time_index[end_idx]

			ego_positions = segment_vehicle_df[['Local_X', 'Local_Y']].to_numpy(dtype=np.float64)
			veh_time_pairs.append({
				'Vehicle_ID': vehicle_id,
				'Drone_ID': vehicle_drone_id,
				'Start_Time': seg_start,
				'End_Time': seg_end,
				'_mid_point': ego_positions[len(ego_positions) // 2],
				'_rel_disp': _sample_relative_displacement(ego_positions),
			})

	veh_time_pairs, removed_duplicate_segments = _filter_duplicate_ego_segments(veh_time_pairs)
	bad_traj["duplicate_overlap"] += removed_duplicate_segments
	veh_time_pairs = [
		{
			'Vehicle_ID': pair['Vehicle_ID'],
			'Drone_ID': pair['Drone_ID'],
			'Start_Time': pair['Start_Time'],
			'End_Time': pair['End_Time'],
		}
		for pair in veh_time_pairs
	]

	info = {
		'total_time_original_minutes': float(
			(session_df['Local_Time'].diff()[session_df['Local_Time'].diff() < break_tolerance]).sum()
			/ pd.Timedelta(minutes=1)
		),
		'total_time_segments_minutes': float(len(veh_time_pairs) * segment / pd.Timedelta(minutes=1)),
		'valid_vehicle_count': len(valid_vehicle_ids),
		'segment_count': len(veh_time_pairs),
		'bad_traj_counts': dict(bad_traj),
	}

	return veh_time_pairs, info


def build_session_artifacts(
	session_filename: str,
	dataset_folder: Path,
	segment_seconds: float,
) -> tuple[list[dict], list[dict], dict[str, object]]:
	"""Load one monitoring-session CSV and build the session-level artifacts saved for later conversion."""
	date, intersection, session = Path(session_filename).stem.split('_')
	session_csv_path = get_csv_path(
		date,
		intersection,
		session,
		raw_dataset_folder=dataset_folder,
	)
	session_df = read_songdo_csv(session_csv_path, add_heading_speed=True)
	
	veh_time_pairs, session_info = select_ego_vehicle_traj(
		session_df,
		segment=segment_seconds,
	)
	
	# Pack one monitoring session into per-frame records for later scene slicing in the converter.
	session_frames: list[dict] = []
	for (local_time, frame_drone_id), frame_df in session_df.groupby(['Local_Time', 'Drone_ID'], sort=False):
		session_frames.append(
			{
				'Local_Time': pd.to_timedelta(local_time).as_unit('ms').to_numpy(),
				'Drone_ID': np.int64(frame_drone_id),
				'Vehicle_Position': frame_df[['Local_X', 'Local_Y']].to_numpy(dtype=np.float64),
				'Vehicle_Size': frame_df[['Vehicle_Length', 'Vehicle_Width']].to_numpy(dtype=np.float64),
				'Vehicle_Class': frame_df['Vehicle_Class'].fillna(-1).astype(int).to_numpy(dtype=np.int64),
				'Vehicle_ID': frame_df['Vehicle_ID'].to_numpy(dtype=np.int64),
				'Heading': frame_df['Heading'].to_numpy(dtype=np.float64),
				'Vx': frame_df['Vx'].to_numpy(dtype=np.float64),
				'Vy': frame_df['Vy'].to_numpy(dtype=np.float64),
			}
		)
	
	return session_frames, veh_time_pairs, session_info


def export_session(
	task: tuple[str, str],
	dataset_folder: Path,
	processed_folder: Path,
	segment_seconds: float,
) -> tuple[str, str, dict[str, object]]:
	"""Save one session's frame cache and ego-vehicle segments and return any failure to ``main``."""
	split_name, session_filename = task
	try:
		session_frames, veh_time_pairs, session_info = build_session_artifacts(
			session_filename=session_filename,
			dataset_folder=dataset_folder,
			segment_seconds=segment_seconds,
		)
		session_stem = Path(session_filename).stem
		session_frames_dir = processed_folder / split_name / SESSION_FRAMES_DIRNAME
		ego_vehicle_info_dir = processed_folder / split_name / EGO_VEHICLE_INFO_DIRNAME
		session_frames_dir.mkdir(parents=True, exist_ok=True)
		ego_vehicle_info_dir.mkdir(parents=True, exist_ok=True)

		with (session_frames_dir / f'{session_stem}.pkl').open('wb') as fp:
			pickle.dump(session_frames, fp, protocol=pickle.HIGHEST_PROTOCOL)
		with (ego_vehicle_info_dir / f'{session_stem}.pkl').open('wb') as fp:
			pickle.dump(veh_time_pairs, fp, protocol=pickle.HIGHEST_PROTOCOL)
		session_info['session_count'] = 1
		session_info['segment_count'] = len(veh_time_pairs)
		session_info['succeeded'] = True
		return split_name, session_filename, session_info

	except Exception as _:
		session_info = _new_summary_info()
		session_info['succeeded'] = False
		return split_name, session_filename, session_info


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset-folder', type=str, default=RAW_DATASET_FOLDER, help="Directory of dataset")
	parser.add_argument('--processed-folder', type=str, default=PROCESSED_FOLDER, help="Directory of processed data")
	parser.add_argument('--split-path', type=str, default=TRAIN_TEST_SPLIT_PATH, help='Path to the train/test split JSON file.')
	parser.add_argument('--segment-seconds', type=float, default=8.0, help='Segment length in seconds.')
	parser.add_argument('--num-workers', type=int, help='Number of worker processes. Defaults to min(16, cpu_count, task_count).')
	args = parser.parse_args()

	dataset_folder = Path(args.dataset_folder)
	processed_folder = Path(args.processed_folder)
	split_path = Path(args.split_path)
	tasks = load_split_tasks(split_path)
	max_workers = args.num_workers
	if max_workers is None:
		max_workers = min(16, os.cpu_count() or 1, len(tasks))

	print(f'Loaded {len(tasks)} session tasks from {split_path}')
	print(f'Using {max_workers} worker processes')

	info_totals = {split_name: _new_summary_info() for split_name in SUMMARY_SPLITS}
	failed_tasks: list[tuple[str, str]] = []
	with ProcessPoolExecutor(max_workers=max_workers) as executor:
		results = executor.map(
			export_session,
			tasks,
			[dataset_folder] * len(tasks),
			[processed_folder] * len(tasks),
			[args.segment_seconds] * len(tasks),
		)
		for split_name, session_filename, session_info in tqdm(
			results,
			total=len(tasks),
			desc='Saving sessions',
			unit='session',
		):
			if bool(session_info['succeeded']):
				_merge_summary_info(info_totals[split_name], session_info)
			else:
				failed_tasks.append((split_name, session_filename))
	
	if failed_tasks:
		print(f"\n{len(failed_tasks)} tasks failed during export:")
		for split_name, session_filename in failed_tasks:
			print(f"  - {split_name}/{session_filename}")

	overall_info = _new_summary_info()
	for split_name in SUMMARY_SPLITS:
		_print_summary(f"{split_name.capitalize()} summary", info_totals[split_name])
		_merge_summary_info(overall_info, info_totals[split_name])
	_print_summary("Overall filtering summary", overall_info)


def debug_session(csv_file_name:str):
	"""Build one session's saved artifacts and print the returned filtering statistics."""
	session_filename = csv_file_name
	split_name, session_filename, session_info = export_session(
		task=('debug', session_filename),
		dataset_folder=RAW_DATASET_FOLDER,
		processed_folder=PROCESSED_FOLDER,
		segment_seconds=8.0,
	)
	_print_summary(f"Debug session {session_filename} summary", session_info)


if __name__ == '__main__':
	main()
	# debug_session('2022-10-04_A_AM1.csv')
