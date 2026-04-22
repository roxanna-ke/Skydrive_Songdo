from pathlib import Path

RAW_DATASET_FOLDER = Path('datasets/songdo_traffic/')
PROCESSED_FOLDER = Path('datasets/songdo_drive/')
FIGURE_FOLDER = Path('figures/')
DATASET_FRAME_RATE = 30  # drone videos are 30 FPS
SESSION_FRAMES_DIRNAME = 'session_frames'
EGO_VEHICLE_INFO_DIRNAME = 'ego_vehicle_info'
INTERSECTION_METADATA_PATH = Path('metadata/intersection_metadata.pkl')
TRAIN_TEST_SPLIT_PATH = PROCESSED_FOLDER / 'train_test_split.json'
VELOCITY_CLUSTER_FOLDER = PROCESSED_FOLDER / 'metadata/velocity_clusters'

# d.p. = decimal places, the types are written like this to allow null value in pandas
DATASET_COLUMN_DTYPES = {
	'Vehicle_ID': 'Int64',  # 1, 2, ... ; Unique vehicle identifier within each CSV file.
	'Local_Time': 'string',  # hh:mm:ss.sss ; Local Korean time (GMT+9) in ISO 8601 format.
	'Drone_ID': 'Int64',  # 1, 2, ..., 10 ; Unique identifier for the drone capturing the data.
	'Ortho_X': 'Float64',  # px (1 d.p.) ; Vehicle center X in orthophoto cut-out image.
	'Ortho_Y': 'Float64',  # px (1 d.p.) ; Vehicle center Y in orthophoto cut-out image.
	'Local_X': 'Float64',  # m (2 d.p.) ; KGD2002 / Central Belt 2010 planar X (EPSG:5186).
	'Local_Y': 'Float64',  # m (2 d.p.) ; KGD2002 / Central Belt 2010 planar Y (EPSG:5186).
	'Latitude': 'Float64',  # degree DD (7 d.p.) ; WGS84 latitude in decimal degrees (EPSG:4326).
	'Longitude': 'Float64',  # degree DD (7 d.p.) ; WGS84 longitude in decimal degrees (EPSG:4326).
	'Vehicle_Length': 'Float64',  # m (2 d.p.) ; Estimated physical vehicle length.
	'Vehicle_Width': 'Float64',  # m (2 d.p.) ; Estimated physical vehicle width.
	'Vehicle_Class': 'Int64',  # categorical 0-3 ; 0 car/van, 1 bus, 2 truck, 3 motorcycle.
	'Vehicle_Speed': 'Float64',  # km/h (1 d.p.) ; Estimated speed from trajectory smoothing.
	'Vehicle_Acceleration': 'Float64',  # m/s^2 (2 d.p.) ; Estimated acceleration from speed.
	'Road_Section': 'string',  # N_G ; Road section identifier (N=node, G=lane group).
	'Lane_Number': 'Int64',  # 1, 2, ... ; Lane position (1 = leftmost in direction of travel).
	'Visibility': 'boolean',  # 0/1 ; True fully visible, False partially visible.
}
FRAME_RECORD_COLUMNS = [
	'Vehicle_ID',
	'Vehicle_Class',
	'Centered_Local_X',
	'Centered_Local_Y',
	'is_target_vehicle',
]
VEHICLE_TYPE_IDX = {
	'car': 0,
	'bus': 1,
	'truck': 2,
	'motorcycle': 3,
}
LANE_LABEL_COLUMNS = ['Section', 'Lane']
LANE_COORD_COLUMNS = ['tlx', 'tly', 'blx', 'bly', 'brx', 'bry', 'trx', 'try']
EXP_DATES = ['2022-10-04', '2022-10-05', '2022-10-06', '2022-10-07']
INTERSECTIONS = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']
SESSIONS = ['AM1', 'AM2', 'AM3', 'AM4', 'AM5', 'PM1', 'PM2', 'PM3', 'PM4', 'PM5']

def get_csv_path(
	date: str,
	intersection: str,
	session: str,
	raw_dataset_folder: Path = RAW_DATASET_FOLDER,
) -> Path:
	"""Build the monitoring-session CSV path from the session key parts."""
	return raw_dataset_folder / f"{date}_{intersection}/{date}_{intersection}_{session}.csv"
