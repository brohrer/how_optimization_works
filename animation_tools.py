from typing import Iterable

import numpy as np


def make_trajectory(
    duration: float = 1,
    frame_rate: float = 1/30,
    loop: bool = False,
    max_submovement_duration: float = .5,
    reverse: bool = False,
    submovement_overlap: float = .4,
    waypoints: Iterable[float] = [0, 1],
) -> np.ndarray:
    """
    Create the entire trajectory of the value, smoothly varying between
    waypoints, blending discrete submovements, after the manner of human
    movement, in order to give it a more nautral and less mechanical look
    and feel.

    loop: Terminate the trajectory at the point where it started.
    reverse: Add a forward an reversed copy of the trajectory back to back.
    waypoints: Points to visit, in order. There needs to be at least
        two of them, a start and a stop.
    """
    waypoints_arr = np.array(waypoints)
    if loop:
        waypoints_arr = np.array(list(waypoints_arr) + [waypoints_arr[0]])
    if reverse:
        waypoints_arr = np.concatenate((
            waypoints_arr, waypoints_arr[::-1][1:]))
    n_waypoints = waypoints_arr.size
    n_trajectory_steps = int(np.ceil(duration / frame_rate))

    # Find how many submovements per waypoint.
    # We're looking for the smallest number that does not exceed
    # the max_submovement_duration provided.
    # The final waypoint just gets one submovement for termination.
    submovements_per_waypoint = 1
    while True:
        n_submovements = (n_waypoints - 1) * submovements_per_waypoint + 1
        n_submovements_effective = (
            n_submovements * submovement_overlap + (1 - submovement_overlap))
        # Account for the fact that submovements overlap
        submovement_duration = duration / n_submovements_effective
        if submovement_duration < max_submovement_duration:
            break
        submovements_per_waypoint += 1

    submovement_waypoints = []
    for i_waypoint in range(waypoints_arr.size - 1):
        start = waypoints_arr[i_waypoint]
        stop = waypoints_arr[i_waypoint + 1]
        submovement_waypoints += list(np.linspace(
            start, stop, num=submovements_per_waypoint, endpoint=False))
    submovement_waypoints.append(waypoints_arr[-1])

    deltas = np.diff(submovement_waypoints)
    submovement_duration_n_frames = int(submovement_duration / frame_rate)
    i_starts = np.floor(submovement_duration * submovement_overlap * (
        np.cumsum(np.ones(n_submovements)) - 1) / frame_rate).astype(np.int)

    submovement_steps = create_submovement(
        n_steps=submovement_duration_n_frames)

    trajectory = np.ones(n_trajectory_steps) * waypoints_arr[0]
    for i_delta, delta in enumerate(deltas):
        i_start = i_starts[i_delta]
        trajectory[i_start:i_start + submovement_duration_n_frames] += (
            delta * submovement_steps)
        trajectory[i_start + submovement_duration_n_frames:] += delta

    # if reverse:
    #     trajectory = np.concatenate((trajectory, trajectory[::-1]))
    return trajectory


def create_submovement_steps(
    n_steps: int = 10,
    start: float = 0,
    stop: float = 1,
):
    """
    Find the appropriate submovements, then find the relevant step size
    at each timestep.
    """
    submovement = create_submovement(
        n_steps=n_steps,
        start=start,
        stop=stop,
    )
    submovement_steps = np.zeros(n_steps)
    submovement_steps[1:] += np.diff(submovement)
    return submovement_steps


def create_submovement(
    n_steps: int = 10,
    start: float = 0,
    stop: float = 1,
):
    """
    Create a smooth submovement, spread out over discrete time steps.

    For now, the minimum jerk trajectory is a fine candidate.
    It has zero velocity and acceleration at its initial and final
    points. As presented by Hogan(1984b)

        x(t) = start + (stop - start) * (10(t/d)^3 - 15(t/d)^4  + 6(t/d)^5)

    to move from start to stop in d seconds.

    Hogan N (1984b) An organizing principle for a class of voluntary movements.
        J Neurosci. 4:2745-2754
    """

    position = start * np.ones(n_steps)
    amplitude = stop - start
    i_step = np.cumsum(np.ones(n_steps)) - 1
    position += amplitude * (
        10 * (i_step / n_steps) ** 3
        - 15 * (i_step / n_steps) ** 4
        + 6 * (i_step / n_steps) ** 5
    )
    return position
