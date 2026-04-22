import argparse
import hashlib
import json

from common import (
	EXP_DATES,
	INTERSECTIONS,
	SESSIONS,
	TRAIN_TEST_SPLIT_PATH,
)

def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--test-split-percent',
		type=int,
		default=15,
		choices=range(0, 101, 5),
		help='Test split percentage for deterministic monitoring-session splitting.',
	)
	args = parser.parse_args()

	session_filenames = [
		f'{date}_{intersection}_{session}.csv'
		for date in EXP_DATES
		for intersection in INTERSECTIONS
		for session in SESSIONS
	]
	ranked_session_filenames = sorted(
		session_filenames,
		key=lambda session_filename: (
			int(hashlib.sha1(session_filename.encode('utf-8')).hexdigest(), 16),
			session_filename,
		),
	)
	num_test_sessions = len(ranked_session_filenames) * args.test_split_percent // 100
	train_test_split = {
		'train': sorted(ranked_session_filenames[num_test_sessions:]),
		'test': sorted(ranked_session_filenames[:num_test_sessions]),
	}

	TRAIN_TEST_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
	with TRAIN_TEST_SPLIT_PATH.open('w', encoding='utf-8') as fp:
		json.dump(train_test_split, fp, indent=2)

	print(f'Saved train/test split to {TRAIN_TEST_SPLIT_PATH}')
	print(f"Train session count: {len(train_test_split['train'])}")
	print(f"Test session count: {len(train_test_split['test'])}")


if __name__ == '__main__':
	main()
