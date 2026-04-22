"""Shared scene-level visualization helpers."""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath


def _compound_polygon_path(polygons: list[np.ndarray]) -> MplPath:
	vertices = []
	codes = []
	for polygon in polygons:
		vertices.extend(polygon.tolist())
		vertices.append(polygon[0].tolist())
		codes.extend([MplPath.MOVETO])
		codes.extend([MplPath.LINETO] * (len(polygon) - 1))
		codes.extend([MplPath.CLOSEPOLY])
	return MplPath(np.asarray(vertices, dtype=np.float64), codes)


def draw_lane_background(
	ax,
	lane_polygons: np.ndarray,
	lane_centerlines: np.ndarray,
	direction_valid: np.ndarray,
	drivable_polygons: Optional[list[np.ndarray]] = None,
	stop_line_segments: Optional[np.ndarray] = None,
	base_zorder: int = 1,
) -> list[object]:
	artists: list[object] = []

	if drivable_polygons:
		drivable_patch = PathPatch(
			_compound_polygon_path(drivable_polygons),
			facecolor='#9ca3af',
			edgecolor='none',
			alpha=0.28,
			fill=True,
			zorder=base_zorder - 1,
		)
		ax.add_patch(drivable_patch)
		artists.append(drivable_patch)

	polygon_collection = PolyCollection(
		lane_polygons,
		facecolors='none',
		edgecolors='#cbd5e1',
		linewidths=0.8,
		alpha=0.8,
		zorder=base_zorder,
	)
	ax.add_collection(polygon_collection)
	artists.append(polygon_collection)

	if direction_valid.any():
		valid_centerline_collection = LineCollection(
			lane_centerlines[direction_valid],
			colors='#0f766e',
			linewidths=1.4,
			alpha=0.95,
			zorder=base_zorder + 1,
		)
		ax.add_collection(valid_centerline_collection)
		artists.append(valid_centerline_collection)

		arrow_centers = lane_centerlines[direction_valid].mean(axis=1)
		arrow_vectors = 0.65 * (lane_centerlines[direction_valid, 1] - lane_centerlines[direction_valid, 0])
		direction_quiver = ax.quiver(
			arrow_centers[:, 0],
			arrow_centers[:, 1],
			arrow_vectors[:, 0],
			arrow_vectors[:, 1],
			angles='xy',
			scale_units='xy',
			scale=1.0,
			pivot='mid',
			color='#ea580c',
			width=0.003,
			headwidth=4.5,
			headlength=6.0,
			headaxislength=5.0,
			alpha=0.95,
			zorder=base_zorder + 2,
		)
		artists.append(direction_quiver)

	if (~direction_valid).any():
		invalid_centerline_collection = LineCollection(
			lane_centerlines[~direction_valid],
			colors='#64748b',
			linewidths=1.0,
			alpha=0.75,
			zorder=base_zorder + 1,
		)
		ax.add_collection(invalid_centerline_collection)
		artists.append(invalid_centerline_collection)

	if stop_line_segments is not None and len(stop_line_segments):
		stop_line_outline = LineCollection(
			stop_line_segments,
			colors='#7c2d12',
			linewidths=4.2,
			alpha=0.92,
			zorder=base_zorder + 3,
		)
		ax.add_collection(stop_line_outline)
		artists.append(stop_line_outline)

		stop_line_fill = LineCollection(
			stop_line_segments,
			colors='#f8fafc',
			linewidths=2.8,
			alpha=0.98,
			zorder=base_zorder + 4,
		)
		ax.add_collection(stop_line_fill)
		artists.append(stop_line_fill)

	return artists


def animate_scene(
	scene: dict,
	save_path: Union[str, Path],
	box_size_m: float = 120.0,
	fps: int = 30,
) -> None:
	"""Render a scene as an animated GIF centered on the ego vehicle."""
	frames = scene.get('frames', [])
	if not frames:
		raise ValueError('Cannot render GIF: scene has no frames.')
	target_vehicle_id = int(scene['ego_info']['Vehicle_ID'])

	half_box = box_size_m / 2.0
	fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
	ax.set_aspect('equal', adjustable='box')
	ax.grid(True, linewidth=0.4, alpha=0.4)
	ax.set_xlabel('Local_X (m)')
	ax.set_ylabel('Local_Y (m)')

	metadata = scene.get('metadata', {})
	lane_polygons = metadata['lane_polygons']
	lane_centerlines = metadata['lane_centerlines']
	direction_valid = metadata['direction_valid']
	lane_background_artists = draw_lane_background(
		ax,
		lane_polygons,
		lane_centerlines,
		direction_valid,
		drivable_polygons=metadata.get('drivable_polygons'),
		stop_line_segments=np.asarray([stop_line['segment'] for stop_line in metadata['stop_lines']], dtype=np.float64),
		base_zorder=1,
	)
	other_scatter = ax.scatter([], [], s=14, c=[], cmap='tab10', vmin=0, vmax=4, marker='o', linewidths=0, zorder=2)
	target_scatter = ax.scatter([], [], s=80, marker='*', zorder=3)
	heading_lines = LineCollection([], colors='#111827', linewidths=1.2, zorder=4)
	ax.add_collection(heading_lines)
	time_text = ax.text(
		0.01,
		0.99,
		'',
		transform=ax.transAxes,
		va='top',
		ha='left',
		fontsize=9,
	)

	first_frame = frames[0]
	first_target_position = first_frame['Vehicle_Position'][first_frame['Vehicle_ID'] == target_vehicle_id][0]
	ax.set_xlim(first_target_position[0] - half_box, first_target_position[0] + half_box)
	ax.set_ylim(first_target_position[1] - half_box, first_target_position[1] + half_box)

	def _update(frame_idx: int):
		frame = frames[frame_idx]
		positions = frame['Vehicle_Position']
		vehicle_classes = frame['Vehicle_Class']
		vehicle_ids = frame['Vehicle_ID']
		headings = frame['Heading']
		target_position = positions[vehicle_ids == target_vehicle_id][0]

		def _as_offsets(points: np.ndarray) -> np.ndarray:
			return points if len(points) else np.empty((0, 2), dtype=np.float64)

		ax.set_xlim(target_position[0] - half_box, target_position[0] + half_box)
		ax.set_ylim(target_position[1] - half_box, target_position[1] + half_box)

		in_box_mask = (
			(np.abs(positions[:, 0] - target_position[0]) <= half_box)
			& (np.abs(positions[:, 1] - target_position[1]) <= half_box)
		)
		box_positions = positions[in_box_mask]
		box_classes = vehicle_classes[in_box_mask]
		box_ids = vehicle_ids[in_box_mask]
		box_headings = headings[in_box_mask]

		target_mask = box_ids == target_vehicle_id
		target_points = box_positions[target_mask]
		other_points = box_positions[~target_mask]
		other_classes = box_classes[~target_mask].astype(np.float64)
		other_classes[other_classes < 0] = 4.0

		other_scatter.set_offsets(_as_offsets(other_points))
		other_scatter.set_array(other_classes)
		target_scatter.set_offsets(_as_offsets(target_points))
		valid_heading_mask = ~np.isnan(box_headings)
		heading_points = box_positions[valid_heading_mask]
		heading_values = box_headings[valid_heading_mask]
		heading_len = 4.0
		heading_dx = np.cos(heading_values) * heading_len
		heading_dy = np.sin(heading_values) * heading_len
		heading_vectors = np.column_stack((heading_dx, heading_dy))
		heading_segments = np.stack((heading_points, heading_points + heading_vectors), axis=1) if len(heading_points) else []
		heading_lines.set_segments(heading_segments)
		time_text.set_text(str(frame['Local_Time']))

		return [*lane_background_artists, other_scatter, target_scatter, heading_lines, time_text]

	animation = FuncAnimation(
		fig,
		_update,
		frames=len(frames),
		interval=max(1, int(round(1000 / max(1, fps)))),
		blit=False,
	)
	animation.save(str(save_path), writer=PillowWriter(fps=max(1, fps)))
	plt.close(fig)
