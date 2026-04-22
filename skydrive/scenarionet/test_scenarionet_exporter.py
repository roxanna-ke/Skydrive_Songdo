"""Build, verify, and visualize one exported Songdo ScenarioNet scenario.

Example:

conda activate unitraj
python skydrive/scenarionet/songdo_exporter_smoke_test.py \
	--split train \
	--session 2022-10-04_A_AM1.pkl \
	--scene-index 0
"""

import argparse
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.scenario_description import ScenarioDescription
from scenarionet.common_utils import read_dataset_summary, read_scenario, save_summary_and_mapping
from scenarionet.verifier.utils import verify_database

from skydrive.common.songdo_scene_loader import SongdoSceneLoader
from skydrive.preprocess.common import FIGURE_FOLDER, PROCESSED_FOLDER
from skydrive.scenarionet.scenarionet_exporter import DATASET_NAME, DEFAULT_DATASET_VERSION, build_songdo_scenario


def _render_exported_scenario(dataset_dir: Path, scenario: dict, figure_path: Path) -> None:
	figure_path.parent.mkdir(parents=True, exist_ok=True)
	current_time_index = scenario[ScenarioDescription.METADATA]['current_time_index']
	env = ScenarioEnv(
		{
			'use_render': False,
			'agent_policy': ReplayEgoCarPolicy,
			'manual_control': False,
			'render_pipeline': False,
			'show_interface': False,
			'show_logo': False,
			'show_fps': False,
			'no_traffic': True,
			'num_scenarios': 1,
			'horizon': int(scenario[ScenarioDescription.LENGTH]),
			'data_directory': str(dataset_dir),
		}
	)
	try:
		env.reset(seed=0)
		for _ in range(current_time_index):
			env.step([0.0, 0.0])
		image = env.render(
			mode='top_down',
			window=False,
			semantic_map=True,
			draw_center_line=True,
			target_agent_heading_up=False,
			screen_size=(1200, 1200),
			film_size=(3000, 3000),
			text={
				'scenario_id': scenario[ScenarioDescription.ID],
				'frame': env.episode_step,
			},
		)
	finally:
		env.close()

	if image is None:
		raise RuntimeError('Top-down rendering returned no image')
	if not cv2.imwrite(str(figure_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
		raise RuntimeError(f'Failed to write visualization to {figure_path}')


def main() -> None:
	parser = argparse.ArgumentParser(description='Smoke-test one Songdo ScenarioNet export.')
	parser.add_argument('--split', type=str, default='train', help='Dataset split name.')
	parser.add_argument('--session', type=str, default='2022-10-04_A_AM1.pkl', help='Session filename.')
	parser.add_argument('--scene-index', type=int, default=0, help='Index into the saved ego segments.')
	parser.add_argument(
		'--processed-folder',
		type=Path,
		default=PROCESSED_FOLDER,
		help='Root of the preprocessed Songdo dataset.',
	)
	parser.add_argument(
		'--output-dir',
		type=Path,
		default=Path('scratch') / 'songdo_scenarionet_single_check',
		help='Directory to write the one-scenario ScenarioNet dataset.',
	)
	parser.add_argument(
		'--figure-path',
		type=Path,
		default=FIGURE_FOLDER / 'scenarionet' / 'songdo_single_scenario_check.png',
		help='Path to the saved top-down visualization.',
	)
	parser.add_argument('--dataset-version', type=str, default=DEFAULT_DATASET_VERSION, help='ScenarioNet version tag.')
	parser.add_argument(
		'--overwrite',
		action=argparse.BooleanOptionalAction,
		default=True,
		help='Overwrite existing outputs.',
	)
	args = parser.parse_args()

	loader = SongdoSceneLoader(processed_folder=args.processed_folder)
	session_frames, veh_time_pairs, metadata = loader.load_session(args.split, args.session)
	metadata['source_file'] = args.session
	scene = loader.build_scene_from_session(session_frames, veh_time_pairs[args.scene_index], metadata)

	scenario = build_songdo_scenario(scene, dataset_name=DATASET_NAME)
	scenario = ScenarioDescription.update_summaries(ScenarioDescription(scenario)).to_dict()
	number_summary = scenario[ScenarioDescription.METADATA][ScenarioDescription.SUMMARY.NUMBER_SUMMARY]
	if not np.isfinite(number_summary[ScenarioDescription.SUMMARY.MAP_HEIGHT_DIFF]):
		# Lane-centerline-only exports have no road-line height variation.
		number_summary[ScenarioDescription.SUMMARY.MAP_HEIGHT_DIFF] = 0.0
	ScenarioDescription.sanity_check(scenario, check_self_type=True)

	dataset_dir = args.output_dir / f'{args.split}_{Path(args.session).stem}_scene{args.scene_index}'
	if dataset_dir.exists():
		if not args.overwrite:
			raise FileExistsError(f'Output path already exists: {dataset_dir}')
		shutil.rmtree(dataset_dir)
	dataset_dir.mkdir(parents=True, exist_ok=True)

	export_file_name = ScenarioDescription.get_export_file_name(
		DATASET_NAME,
		args.dataset_version,
		scenario[ScenarioDescription.ID],
	)
	with (dataset_dir / export_file_name).open('wb') as fp:
		pickle.dump(scenario, fp, protocol=pickle.HIGHEST_PROTOCOL)

	save_summary_and_mapping(
		str(dataset_dir / ScenarioDescription.DATASET.SUMMARY_FILE),
		str(dataset_dir / ScenarioDescription.DATASET.MAPPING_FILE),
		{export_file_name: scenario[ScenarioDescription.METADATA].copy()},
		{export_file_name: ''},
	)

	summary, scenario_ids, mapping = read_dataset_summary(str(dataset_dir))
	assert scenario_ids == [export_file_name]
	assert export_file_name in summary
	loaded_scenario = read_scenario(str(dataset_dir), mapping, export_file_name).to_dict()
	assert loaded_scenario[ScenarioDescription.LENGTH] == 81
	assert loaded_scenario[ScenarioDescription.METADATA]['current_time_index'] == 20
	sdc_id = loaded_scenario[ScenarioDescription.METADATA][ScenarioDescription.SDC_ID]
	assert sdc_id in loaded_scenario[ScenarioDescription.TRACKS]
	assert list(loaded_scenario[ScenarioDescription.METADATA]['tracks_to_predict']) == [sdc_id]
	ScenarioDescription.sanity_check(loaded_scenario, check_self_type=True)

	verify_dir = Path('scratch') / 'songdo_scenarionet_verify' / f'{dataset_dir.name}_single'
	verify_dir.mkdir(parents=True, exist_ok=True)
	success, errors = verify_database(
		dataset_path=str(dataset_dir),
		error_file_path=str(verify_dir),
		overwrite=True,
		steps_to_run=0,
	)
	if not success:
		raise RuntimeError(f'ScenarioNet structural verification failed: {len(errors)} errors')

	_render_exported_scenario(dataset_dir, loaded_scenario, args.figure_path)
	print(f'Exported scenario: {dataset_dir / export_file_name}')
	print(f'Verification directory: {verify_dir}')
	print(f'Visualization: {args.figure_path}')


if __name__ == '__main__':
	main()
