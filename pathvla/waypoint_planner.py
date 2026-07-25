from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from pathvla.errors import PlanningError
from pathvla.schemas import PlannerDiagnosticModel, SceneSnapshotModel, SubgoalModel, WaypointPlanModel


@dataclass(frozen=True)
class GridPoint:
    ix: int
    iy: int


class AStarWaypointPlanner:
    def __init__(self, grid_resolution_m: float = 0.25):
        if grid_resolution_m <= 0.0:
            raise ValueError("grid_resolution_m must be positive")
        self.grid_resolution_m = grid_resolution_m

    def plan(self, scene: SceneSnapshotModel, subgoal: SubgoalModel) -> WaypointPlanModel:
        bounds = scene.bounds
        x_bounds = bounds["x"]
        y_bounds = bounds["y"]
        start = scene.robot.pose[:2]
        target_pose = self._resolve_target_pose(scene, subgoal)
        blocked, avoided_objects = self._build_blocked_set(scene, subgoal, x_bounds, y_bounds)

        start_gp = self._world_to_grid(start[0], start[1], x_bounds, y_bounds)
        goal_gp = self._world_to_grid(target_pose[0], target_pose[1], x_bounds, y_bounds)
        path, expanded_nodes = self._astar(start_gp, goal_gp, blocked, x_bounds, y_bounds)
        if not path:
            raise PlanningError(f"No path found from {scene.robot.pose} to target {subgoal.target}")
        world_path = [self._grid_to_world(node, x_bounds, y_bounds) for node in path]
        path_length = self._path_length(world_path)
        return WaypointPlanModel(
            target=subgoal.target,
            subgoal_type=subgoal.type,
            waypoints=[[x, y, float(scene.robot.pose[2])] for x, y in world_path],
            path_length_m=path_length,
            avoided_objects=avoided_objects,
            diagnostics=PlannerDiagnosticModel(
                grid_resolution_m=self.grid_resolution_m,
                expanded_nodes=expanded_nodes,
                blocked_cells=len(blocked),
                target_name=subgoal.target,
            ),
        )

    def _resolve_target_pose(self, scene: SceneSnapshotModel, subgoal: SubgoalModel) -> tuple[float, float]:
        for obj in scene.objects:
            if obj.name == subgoal.target:
                offset = 0.75 if subgoal.type.value in {"inspect", "pickup", "drop"} else 0.0
                return obj.pose[0] - offset, obj.pose[1]
        raise PlanningError(f"Target object '{subgoal.target}' not found in scene")

    def _build_blocked_set(
        self,
        scene: SceneSnapshotModel,
        subgoal: SubgoalModel,
        x_bounds: list[float],
        y_bounds: list[float],
    ) -> tuple[set[GridPoint], list[str]]:
        blocked: set[GridPoint] = set()
        avoid_names = set(subgoal.constraints.avoid)
        avoided_objects: list[str] = []
        for obj in scene.objects:
            if obj.name == subgoal.target:
                continue
            radius = obj.avoidance_radius
            if obj.name in avoid_names or obj.type in {"obstacle", "container", "inspectable"}:
                radius = max(radius, subgoal.constraints.safe_distance_m)
                avoided_objects.append(obj.name)
                min_x = obj.pose[0] - radius
                max_x = obj.pose[0] + radius
                min_y = obj.pose[1] - radius
                max_y = obj.pose[1] + radius
                gp0 = self._world_to_grid(min_x, min_y, x_bounds, y_bounds)
                gp1 = self._world_to_grid(max_x, max_y, x_bounds, y_bounds)
                for ix in range(min(gp0.ix, gp1.ix), max(gp0.ix, gp1.ix) + 1):
                    for iy in range(min(gp0.iy, gp1.iy), max(gp0.iy, gp1.iy) + 1):
                        if math.dist((obj.pose[0], obj.pose[1]), self._grid_to_world(GridPoint(ix, iy), x_bounds, y_bounds)) <= radius:
                            blocked.add(GridPoint(ix, iy))
        return blocked, sorted(set(avoided_objects))

    def _astar(
        self,
        start: GridPoint,
        goal: GridPoint,
        blocked: set[GridPoint],
        x_bounds: list[float],
        y_bounds: list[float],
    ) -> tuple[list[GridPoint], int]:
        frontier: list[tuple[float, int, GridPoint]] = []
        push_index = 0
        heapq.heappush(frontier, (0.0, push_index, start))
        came_from: dict[GridPoint, GridPoint | None] = {start: None}
        cost_so_far: dict[GridPoint, float] = {start: 0.0}
        expanded_nodes = 0

        while frontier:
            _, _, current = heapq.heappop(frontier)
            expanded_nodes += 1
            if current == goal:
                return self._reconstruct_path(came_from, goal), expanded_nodes
            for neighbor in self._neighbors(current, x_bounds, y_bounds):
                if neighbor in blocked:
                    continue
                new_cost = cost_so_far[current] + math.dist((current.ix, current.iy), (neighbor.ix, neighbor.iy))
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + math.dist((neighbor.ix, neighbor.iy), (goal.ix, goal.iy))
                    push_index += 1
                    heapq.heappush(frontier, (priority, push_index, neighbor))
                    came_from[neighbor] = current
        return [], expanded_nodes

    def _neighbors(self, node: GridPoint, x_bounds: list[float], y_bounds: list[float]) -> list[GridPoint]:
        max_ix = int(round((x_bounds[1] - x_bounds[0]) / self.grid_resolution_m))
        max_iy = int(round((y_bounds[1] - y_bounds[0]) / self.grid_resolution_m))
        result = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nx = node.ix + dx
            ny = node.iy + dy
            if 0 <= nx <= max_ix and 0 <= ny <= max_iy:
                result.append(GridPoint(nx, ny))
        return result

    def _reconstruct_path(self, came_from: dict[GridPoint, GridPoint | None], goal: GridPoint) -> list[GridPoint]:
        node: GridPoint | None = goal
        path: list[GridPoint] = []
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path

    def _world_to_grid(self, x: float, y: float, x_bounds: list[float], y_bounds: list[float]) -> GridPoint:
        ix = int(round((x - x_bounds[0]) / self.grid_resolution_m))
        iy = int(round((y - y_bounds[0]) / self.grid_resolution_m))
        return GridPoint(ix, iy)

    def _grid_to_world(self, node: GridPoint, x_bounds: list[float], y_bounds: list[float]) -> tuple[float, float]:
        x = x_bounds[0] + node.ix * self.grid_resolution_m
        y = y_bounds[0] + node.iy * self.grid_resolution_m
        return x, y

    @staticmethod
    def _path_length(path: list[tuple[float, float]]) -> float:
        total = 0.0
        for idx in range(1, len(path)):
            total += math.dist(path[idx - 1], path[idx])
        return total
