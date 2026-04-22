from typing import TypedDict

import numpy as np
import numpy.typing as npt
from unitraj.datasets.common_utils import get_polyline_dir, interpolate_polyline


class TracksToPredict(TypedDict):
	track_index: list[int]
	object_type: list[int]


class TrackInfos(TypedDict):
	object_id: list[str] # shape (num_objects,)
	object_type: list[int] # shape (num_objects,)
	# shape (num_objects, length, 10)
	# length is usually 2s history (20 frames at 10Hz) + 1 current time frame + 6s future = 81 frames
	# the 10 channels are: x, y, z, l, w, h, heading, vx, vy, valid
	trajs: npt.NDArray[np.float64] 


class MapInfos(TypedDict):
	lane: list[object]
	road_line: list[object]
	road_edge: list[float]
	stop_sign: list[float]
	crosswalk: list[object]
	speed_bump: list[float]
	# shape (num_points, 7), where the 7 channels are: x, y, z, dx, dy, dz, type.
	# dx, dy, dz are calculated with `get_polyline_dir` in unitraj
	# type is given by MetaDriveType
	all_polylines: npt.NDArray[np.float32]


class DatasetSchema(TypedDict):
	dataset: str
	scenario_id: str
	timestamps_seconds: npt.NDArray[np.float64]
	current_time_index: int
	sdc_track_index: int
	map_center: npt.NDArray[np.float64] # shape (1, 3)
	tracks_to_predict: TracksToPredict
	track_infos: TrackInfos
	map_infos: MapInfos


def _songdo_class_to_unitraj_type(vehicle_class: int) -> tuple[str, int]:
	# In songdo traffic dataset: car/van (0), bus (1), truck (2), motorcycle (3)
	# In unitraj.dataset.types: unset (0), vehicle (1), pedestrian (2), cyclist (3), other (4)
	if vehicle_class == 3:
		return 'CYCLIST', 3
	if vehicle_class in (0, 1, 2):
		return 'VEHICLE', 1
	return 'OTHER', 4


def _max_continuous_valid_length(valid_mask: npt.NDArray[np.bool_]) -> int:
	max_length = 0
	current_length = 0
	for is_valid in valid_mask:
		if is_valid:
			current_length += 1
			max_length = max(max_length, current_length)
		else:
			current_length = 0
	return max_length


def trajectory_filter(data):
	"""Copy-paste from `unitraj.datasets.base_dataset.BaseDataset.trajectory_filter`.
	"""

	trajs = data['track_infos']['trajs']
	current_idx = data['current_time_index']
	obj_summary = data['object_summary']

	tracks_to_predict = {}
	for idx,(k,v) in enumerate(obj_summary.items()):
		type = v['type']
		positions = trajs[idx, :, 0:2]
		validity = trajs[idx, :, -1]
		if type not in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']: 
			continue
		valid_ratio = v['valid_length']/v['track_length']
		if valid_ratio < 0.5: 
			continue
		moving_distance = v['moving_distance']
		if moving_distance < 2.0 and type=='VEHICLE': 
			continue
		is_valid_at_m = validity[current_idx]>0
		if not is_valid_at_m: 
			continue

		# past_traj = positions[:current_idx+1, :]  # Time X (x,y)
		# gt_future = positions[current_idx+1:, :]
		# valid_past = count_valid_steps_past(validity[:current_idx+1])


		future_mask =validity[current_idx+1:]
		future_mask[-1]=0
		idx_of_first_zero = np.where(future_mask == 0)[0]
		idx_of_first_zero = len(future_mask) if len(idx_of_first_zero) == 0 else idx_of_first_zero[0]

		#past_trajectory_valid = past_traj[-valid_past:, :]  # Time(valid) X (x,y)

		# try:
		#     kalman_traj = estimate_kalman_filter(past_trajectory_valid, idx_of_first_zero)  # (x,y)
		#     kalman_diff = calculate_epe(kalman_traj, gt_future[idx_of_first_zero-1])
		# except:
		#     continue
		# if kalman_diff < 20: continue

		tracks_to_predict[k] = {'track_index': idx, 'track_id': k, 'difficulty': 0, 'object_type': type}

	return tracks_to_predict


class UniTrajConverter:
	"""Convert preprocessed Songdo scenes into the fixed-window UniTraj scenario schema."""
	def __init__(
		self,
		source_frame_rate: int = 30,
		target_frame_rate: int = 10,
		history_seconds: float = 2.0,
		future_seconds: float = 6.0,
		dataset_name: str = 'songdo_drive',
	) -> None:
		"""Store the fixed UniTraj export configuration for Songdo scenes."""
		self.source_frame_rate = source_frame_rate
		self.target_frame_rate = target_frame_rate
		self.history_seconds = history_seconds
		self.future_seconds = future_seconds
		self.dataset_name = dataset_name
		self.history_frames = int(self.target_frame_rate * self.history_seconds)
		self.future_frames = int(self.target_frame_rate * self.future_seconds)
		self.frame_stride = self.source_frame_rate // self.target_frame_rate
		self.current_frame_index = int(self.source_frame_rate * self.history_seconds)


	def convert_scene(self, scene: dict) -> DatasetSchema:
		""" The functionality is similar to the `preprocess` method of `BaseDataset` in `unitraj.datasets.base_dataset`
		"""

		frame_indices = self.current_frame_index + np.arange(
			-self.history_frames,
			self.future_frames + 1,
			dtype=np.int64,
		) * self.frame_stride
		frames = [scene['frames'][int(frame_index)] for frame_index in frame_indices]
		frame_times_ms = np.array(
			[frame['Local_Time'].astype('timedelta64[ms]').astype(np.int64) for frame in frames],
			dtype=np.int64,
		)
		timestamps_seconds = (frame_times_ms - frame_times_ms[0]).astype(np.float64) / 1000.0
		current_time_index = self.history_frames

		track_ids = list(
			dict.fromkeys(
				int(vehicle_id)
				for frame in frames
				for vehicle_id in frame['Vehicle_ID']
			)
		)
		track_id_to_index = {track_id: idx for idx, track_id in enumerate(track_ids)}

		num_tracks = len(track_ids)
		num_frames = len(frames)
		trajs = np.zeros((num_tracks, num_frames, 10), dtype=np.float64)
		vehicle_classes = np.full(num_tracks, -1, dtype=np.int64)
		for frame_index, frame in enumerate(frames):
			for vehicle_index, vehicle_id in enumerate(frame['Vehicle_ID']):
				track_index = track_id_to_index[int(vehicle_id)]
				position = frame['Vehicle_Position'][vehicle_index]
				size = frame['Vehicle_Size'][vehicle_index]
				heading = frame['Heading'][vehicle_index]
				vx = frame['Vx'][vehicle_index]
				vy = frame['Vy'][vehicle_index]
				trajs[track_index, frame_index, 0:2] = position
				trajs[track_index, frame_index, 3:5] = size
				trajs[track_index, frame_index, 6] = heading
				trajs[track_index, frame_index, 7] = vx
				trajs[track_index, frame_index, 8] = vy
				# UniTraj treats valid timesteps as numerically usable state, not just agent presence.
				trajs[track_index, frame_index, 9] = 1.0 if (
					np.isfinite(position).all()
					and np.isfinite(size).all()
					and np.isfinite(heading)
					and np.isfinite(vx)
					and np.isfinite(vy)
				) else 0.0
				if vehicle_classes[track_index] < 0:
					vehicle_classes[track_index] = int(frame['Vehicle_Class'][vehicle_index])

		object_type_ids: list[int] = []
		object_summary: dict[str, dict] = {}
		track_id_strings = [str(track_id) for track_id in track_ids]
		for track_index, track_id in enumerate(track_id_strings):
			object_type_name, object_type_id = _songdo_class_to_unitraj_type(int(vehicle_classes[track_index]))
			object_type_ids.append(object_type_id)

			valid_mask = trajs[track_index, :, 9].astype(bool)
			valid_indices = np.flatnonzero(valid_mask)
			valid_positions = trajs[track_index, valid_indices, 0:2]
			moving_distance = np.linalg.norm(np.diff(valid_positions, axis=0), axis=1).sum() if valid_indices.size > 1 else 0.0
			object_summary[track_id] = {
				'type': object_type_name,
				'object_id': track_id,
				'track_length': num_frames,
				'moving_distance': moving_distance,
				'valid_length': int(valid_mask.sum()),
				'continuous_valid_length': _max_continuous_valid_length(valid_mask),
			}

		track_infos: TrackInfos = {
			'object_id': track_id_strings,
			'object_type': object_type_ids,
			'trajs': trajs,
		}

		target_track_id = int(scene['ego_info']['Vehicle_ID'])
		sdc_track_index = track_id_to_index[target_track_id]
		tracks_to_predict: TracksToPredict = {
			'track_index': [sdc_track_index],
			'object_type': [track_infos['object_type'][sdc_track_index]],
		}
		map_infos: MapInfos = {
			'lane': [],
			'road_line': [],
			'road_edge': [],
			'stop_sign': [],
			'crosswalk': [],
			'speed_bump': [],
			'all_polylines': np.zeros((0, 7), dtype=np.float32),
		}
		lane_centerlines = scene['metadata']['lane_centerlines'][scene['metadata']['direction_valid']]
		all_polylines = []
		point_count = 0
		for lane_index, lane_centerline in enumerate(lane_centerlines):
			lane_polyline = interpolate_polyline(lane_centerline)
			lane_directions = get_polyline_dir(lane_polyline)
			lane_type = np.full((lane_polyline.shape[0], 1), 2, dtype=np.float64)
			polyline = np.concatenate((lane_polyline, lane_directions, lane_type), axis=-1).astype(np.float32)
			map_infos['lane'].append(
				{
					'id': str(lane_index),
					'type': 2,
					'polyline_index': (point_count, point_count + len(polyline)),
				}
			)
			all_polylines.append(polyline)
			point_count += len(polyline)

		if all_polylines:
			map_infos['all_polylines'] = np.concatenate(all_polylines, axis=0)

		start_time_ms = np.asarray(scene['ego_info']['Start_Time']).astype('timedelta64[ms]').astype(np.int64)
		end_time_ms = np.asarray(scene['ego_info']['End_Time']).astype('timedelta64[ms]').astype(np.int64)
		date = scene['metadata']['date']
		intersection = scene['metadata']['intersection']
		session = scene['metadata']['session']
		track_infos['object_id'] = [f'{date}_{intersection}_{session}_{track_id}' for track_id in track_infos['object_id']]

		return {
			'dataset': self.dataset_name,
			'scenario_id': f'songdo_drive_{date}_{intersection}_{session}_{target_track_id}_{start_time_ms}_{end_time_ms}',
			'timestamps_seconds': timestamps_seconds,
			'current_time_index': current_time_index,
			'sdc_track_index': sdc_track_index,
			'map_center': np.zeros((1, 3), dtype=np.float64),
			'tracks_to_predict': tracks_to_predict,
			'track_infos': track_infos,
			'map_infos': map_infos,
		}
