"""
Export Songdo Drive dataset scenes into ScenarioNet format.

Example usage:
conda activate unitraj
python skydrive/scenarionet/songdo_exporter.py \
  --train-data-path ./datasets/songdo_drive/train \
  --val-data-path ./datasets/songdo_drive/test \
  --output-path ./datasets/songdo_scenarionet \
  --dataset-version v1 \
  --verify
"""

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np
from metadrive.constants import DATA_VERSION
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.type import MetaDriveType
from scenarionet.common_utils import save_summary_and_mapping
from scenarionet.verifier.utils import verify_database

from skydrive.common.songdo_scene_loader import SongdoSceneLoader
from skydrive.preprocess.common import SESSION_FRAMES_DIRNAME


DATASET_NAME = "songdo_drive"
DEFAULT_DATASET_VERSION = "v1"
SOURCE_FRAME_RATE = 30
TARGET_FRAME_RATE = 10
HISTORY_SECONDS = 2.0
FUTURE_SECONDS = 6.0
FRAME_STRIDE = SOURCE_FRAME_RATE // TARGET_FRAME_RATE
HISTORY_FRAMES = int(TARGET_FRAME_RATE * HISTORY_SECONDS)
FUTURE_FRAMES = int(TARGET_FRAME_RATE * FUTURE_SECONDS)
CURRENT_FRAME_INDEX = int(SOURCE_FRAME_RATE * HISTORY_SECONDS)
TRACK_LENGTH = HISTORY_FRAMES + FUTURE_FRAMES + 1


def _songdo_class_to_metadrive_type(vehicle_class: int) -> str:
	if vehicle_class == 3:
		return MetaDriveType.CYCLIST
	if vehicle_class in (0, 1, 2):
		return MetaDriveType.VEHICLE
	return MetaDriveType.OTHER


def _to_xyz(points_xy: np.ndarray) -> np.ndarray:
	points_xy = np.asarray(points_xy, dtype=np.float32)
	return np.concatenate((points_xy, np.zeros((len(points_xy), 1), dtype=np.float32)), axis=-1)


def _qualified_track_id(date: str, intersection: str, session: str, track_id: int) -> str:
	return f"{date}_{intersection}_{session}_{track_id}"


def _scenario_id(scene: dict) -> str:
	date = scene["metadata"]["date"]
	intersection = scene["metadata"]["intersection"]
	session = scene["metadata"]["session"]
	vehicle_id = int(scene["ego_info"]["Vehicle_ID"])
	start_time_ms = int(np.asarray(scene["ego_info"]["Start_Time"]).astype("timedelta64[ms]").astype(np.int64))
	end_time_ms = int(np.asarray(scene["ego_info"]["End_Time"]).astype("timedelta64[ms]").astype(np.int64))
	return f"{date}_{intersection}_{session}_{vehicle_id}_{start_time_ms}_{end_time_ms}"


def _sample_scene_frames(scene: dict) -> tuple[list[dict], np.ndarray]:
	frame_indices = CURRENT_FRAME_INDEX + np.arange(
		-HISTORY_FRAMES,
		FUTURE_FRAMES + 1,
		dtype=np.int64,
	) * FRAME_STRIDE
	frames = [scene["frames"][int(frame_index)] for frame_index in frame_indices]
	frame_times_ms = np.array(
		[frame["Local_Time"].astype("timedelta64[ms]").astype(np.int64) for frame in frames],
		dtype=np.int64,
	)
	timestamps_seconds = (frame_times_ms - frame_times_ms[0]).astype(np.float32) / 1000.0
	return frames, timestamps_seconds


def build_songdo_scenario(
	scene: dict,
	dataset_name: str = DATASET_NAME,
) -> dict:
	frames, timestamps_seconds = _sample_scene_frames(scene)
	metadata = scene["metadata"]
	date = metadata["date"]
	intersection = metadata["intersection"]
	session = metadata["session"]

	track_ids = list(
		dict.fromkeys(
			int(vehicle_id)
			for frame in frames
			for vehicle_id in frame["Vehicle_ID"]
		)
	)
	track_id_to_index = {track_id: idx for idx, track_id in enumerate(track_ids)}
	tracks = {}
	for track_id in track_ids:
		object_id = _qualified_track_id(date, intersection, session, track_id)
		tracks[object_id] = {
			"type": MetaDriveType.UNSET,
			"state": {
				"position": np.zeros((TRACK_LENGTH, 3), dtype=np.float32),
				"length": np.zeros(TRACK_LENGTH, dtype=np.float32),
				"width": np.zeros(TRACK_LENGTH, dtype=np.float32),
				"height": np.zeros(TRACK_LENGTH, dtype=np.float32),
				"heading": np.zeros(TRACK_LENGTH, dtype=np.float32),
				"velocity": np.zeros((TRACK_LENGTH, 2), dtype=np.float32),
				"valid": np.zeros(TRACK_LENGTH, dtype=bool),
			},
			"metadata": {
				"track_length": TRACK_LENGTH,
				"type": MetaDriveType.UNSET,
				"object_id": object_id,
				"original_id": str(track_id),
				"dataset": dataset_name,
			},
		}

	for frame_index, frame in enumerate(frames):
		for vehicle_index, vehicle_id in enumerate(frame["Vehicle_ID"]):
			track_id = int(vehicle_id)
			object_id = _qualified_track_id(date, intersection, session, track_id)
			track = tracks[object_id]
			position = np.asarray(frame["Vehicle_Position"][vehicle_index], dtype=np.float32)
			size = np.asarray(frame["Vehicle_Size"][vehicle_index], dtype=np.float32)
			heading = np.float32(frame["Heading"][vehicle_index])
			vx = np.float32(frame["Vx"][vehicle_index])
			vy = np.float32(frame["Vy"][vehicle_index])
			vehicle_class = int(frame["Vehicle_Class"][vehicle_index])
			is_valid = bool(
				np.isfinite(position).all()
				and np.isfinite(size).all()
				and np.isfinite(heading)
				and np.isfinite(vx)
				and np.isfinite(vy)
			)

			track["type"] = _songdo_class_to_metadrive_type(vehicle_class)
			track["metadata"]["type"] = track["type"]
			if is_valid:
				track["state"]["position"][frame_index, :2] = position
				track["state"]["length"][frame_index] = size[0]
				track["state"]["width"][frame_index] = size[1]
				track["state"]["heading"][frame_index] = heading
				track["state"]["velocity"][frame_index] = np.array([vx, vy], dtype=np.float32)
			track["state"]["valid"][frame_index] = is_valid

	sdc_track_id = int(scene["ego_info"]["Vehicle_ID"])
	sdc_object_id = _qualified_track_id(date, intersection, session, sdc_track_id)
	scenario_id = _scenario_id(scene)

	map_features = {}
	lane_centerlines = metadata["lane_centerlines"][metadata["direction_valid"]]
	for lane_index, lane_centerline in enumerate(lane_centerlines):
		map_features[str(lane_index)] = {
			"type": MetaDriveType.LANE_SURFACE_STREET,
			"polyline": _to_xyz(lane_centerline),
		}

	sdc_track_index = track_id_to_index[sdc_track_id]
	return {
		ScenarioDescription.ID: scenario_id,
		ScenarioDescription.VERSION: DATA_VERSION,
		ScenarioDescription.LENGTH: TRACK_LENGTH,
		ScenarioDescription.TRACKS: tracks,
		ScenarioDescription.DYNAMIC_MAP_STATES: {},
		ScenarioDescription.MAP_FEATURES: map_features,
		ScenarioDescription.METADATA: {
			ScenarioDescription.ID: scenario_id,
			"dataset": dataset_name,
			"scenario_id": scenario_id,
			ScenarioDescription.METADRIVE_PROCESSED: False,
			ScenarioDescription.COORDINATE: MetaDriveType.COORDINATE_METADRIVE,
			ScenarioDescription.TIMESTEP: timestamps_seconds,
			ScenarioDescription.SDC_ID: sdc_object_id,
			"current_time_index": HISTORY_FRAMES,
			"track_length": TRACK_LENGTH,
			"source_file": metadata["source_file"],
			"tracks_to_predict": {
				sdc_object_id: {
					"track_index": sdc_track_index,
					"track_id": sdc_object_id,
					"difficulty": 0,
					"object_type": tracks[sdc_object_id]["type"],
				}
			},
		},
	}


def export_songdo_split(
	data_paths: list[Path],
	output_path: Path,
	split_name: str,
	overwrite: bool,
	dataset_version: str = DEFAULT_DATASET_VERSION,
) -> None:
	split_output_path = output_path / split_name
	if split_output_path.exists():
		if not overwrite:
			raise FileExistsError(f"Output path already exists: {split_output_path}")
		shutil.rmtree(split_output_path)
	split_output_path.mkdir(parents=True, exist_ok=True)

	summary = {}
	mapping = {}
	for data_path in data_paths:
		data_path = Path(data_path)
		loader = SongdoSceneLoader(processed_folder=data_path.parent)
		session_dir = data_path / SESSION_FRAMES_DIRNAME
		session_files = [path.name for path in sorted(session_dir.glob("*.pkl"))]
		for session_filename in session_files:
			session_stem = Path(session_filename).stem
			session_output_path = split_output_path / session_stem
			session_output_path.mkdir(parents=True, exist_ok=True)
			session_frames, veh_time_pairs, metadata = loader.load_session(split_name, session_filename)
			metadata["source_file"] = session_filename
			for ego_info in veh_time_pairs:
				scene = loader.build_scene_from_session(session_frames, ego_info, metadata)
				scenario = build_songdo_scenario(scene, dataset_name=DATASET_NAME)
				scenario = ScenarioDescription.update_summaries(ScenarioDescription(scenario)).to_dict()
				number_summary = scenario[ScenarioDescription.METADATA][ScenarioDescription.SUMMARY.NUMBER_SUMMARY]
				if not np.isfinite(number_summary[ScenarioDescription.SUMMARY.MAP_HEIGHT_DIFF]):
					# Songdo export intentionally omits road-line features, so map height should stay flat.
					number_summary[ScenarioDescription.SUMMARY.MAP_HEIGHT_DIFF] = 0.0
				ScenarioDescription.sanity_check(scenario, check_self_type=True)

				export_file_name = ScenarioDescription.get_export_file_name(
					DATASET_NAME,
					dataset_version,
					scenario[ScenarioDescription.ID],
				)
				if export_file_name in summary:
					raise ValueError(f"Duplicate scenario export name: {export_file_name}")
				with (session_output_path / export_file_name).open("wb") as fp:
					pickle.dump(scenario, fp, protocol=pickle.HIGHEST_PROTOCOL)
				summary[export_file_name] = scenario[ScenarioDescription.METADATA].copy()
				mapping[export_file_name] = session_stem

	save_summary_and_mapping(
		str(split_output_path / ScenarioDescription.DATASET.SUMMARY_FILE),
		str(split_output_path / ScenarioDescription.DATASET.MAPPING_FILE),
		summary,
		mapping,
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Export Songdo scenes into ScenarioNet datasets.")
	parser.add_argument(
		"--train-data-path",
		nargs="+",
		default=["./datasets/songdo_drive/train"],
		help="One or more processed Songdo training split roots.",
	)
	parser.add_argument(
		"--val-data-path",
		nargs="+",
		default=["./datasets/songdo_drive/test"],
		help="One or more processed Songdo validation split roots.",
	)
	parser.add_argument(
		"--output-path",
		type=Path,
		default=Path("./datasets/songdo_scenarionet"),
		help="Output root for exported ScenarioNet train/test datasets.",
	)
	parser.add_argument(
		"--overwrite",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Whether to overwrite existing exported split directories.",
	)
	parser.add_argument(
		"--verify",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Run ScenarioNet structural verification after export.",
	)
	parser.add_argument(
		"--dataset-version",
		type=str,
		default=DEFAULT_DATASET_VERSION,
		help="Dataset version suffix used in exported scenario filenames.",
	)
	args = parser.parse_args()

	split_data_paths = (
		("train", [Path(path) for path in args.train_data_path]),
		("test", [Path(path) for path in args.val_data_path]),
	)
	for split_name, data_paths in split_data_paths:
		export_songdo_split(data_paths, args.output_path, split_name, args.overwrite, args.dataset_version)

	if args.verify:
		for split_name in ("train", "test"):
			split_output_path = args.output_path / split_name
			error_file_path = Path("scratch") / "songdo_scenarionet_verify" / split_name
			error_file_path.mkdir(parents=True, exist_ok=True)
			success, errors = verify_database(
				dataset_path=str(split_output_path),
				error_file_path=str(error_file_path),
				overwrite=True,
				steps_to_run=0,
			)
			if not success:
				raise RuntimeError(
					f"ScenarioNet structural verification failed for {split_output_path}: {len(errors)} errors"
				)


if __name__ == "__main__":
	main()
