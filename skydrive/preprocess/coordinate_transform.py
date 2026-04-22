import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer as ProjTransformer

from common import RAW_DATASET_FOLDER

ORTHOPHOTO_FOLDER = RAW_DATASET_FOLDER / 'orthophotos'
CUTOUT_WIDTH_PX = 15000
GPS_TO_LOCAL = ProjTransformer.from_crs('EPSG:4326', 'EPSG:5186', always_xy=True)


class OrthoCoordToGPSTransformer:

	def __init__(self, base_latitude: float, base_longitude: float, lon_per_px: float, lat_per_px: float):
		self.base_latitude = base_latitude
		self.base_longitude = base_longitude
		self.lon_per_px = lon_per_px
		self.lat_per_px = lat_per_px

	def __call__(self, ortho_x: np.ndarray, ortho_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		longitude = self.base_longitude + ortho_x * self.lon_per_px
		latitude = self.base_latitude + ortho_y * self.lat_per_px
		return longitude, latitude


def get_ortho_to_gps_transformer(intersection: str) -> OrthoCoordToGPSTransformer:
	lng_O, lat_O, lon_per_px, lat_per_px = np.loadtxt(ORTHOPHOTO_FOLDER / 'ortho_parameters.txt')
	center_x, center_y = np.loadtxt(ORTHOPHOTO_FOLDER / f'{intersection}_center.txt')
	ortho_photo = plt.imread(ORTHOPHOTO_FOLDER / f'{intersection}.png')
	scale = CUTOUT_WIDTH_PX / ortho_photo.shape[1]

	lng_B = lng_O + (center_x - CUTOUT_WIDTH_PX // 2) * lon_per_px
	lat_B = lat_O + (center_y - CUTOUT_WIDTH_PX // 2) * lat_per_px
	lon_per_px_resized = lon_per_px * scale
	lat_per_px_resized = lat_per_px * scale

	return OrthoCoordToGPSTransformer(
		base_latitude=lat_B,
		base_longitude=lng_B,
		lon_per_px=lon_per_px_resized,
		lat_per_px=lat_per_px_resized,
	)


def ortho_to_local(
	ortho_x: np.ndarray,
	ortho_y: np.ndarray,
	ortho2gps: OrthoCoordToGPSTransformer,
) -> tuple[np.ndarray, np.ndarray]:
	longitude, latitude = ortho2gps(ortho_x, ortho_y)
	return GPS_TO_LOCAL.transform(longitude, latitude)
