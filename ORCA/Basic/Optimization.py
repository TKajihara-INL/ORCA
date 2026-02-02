import time
from typing import Callable, Optional

import numpy as np

class Optimization(object):
    """
    MPC dispatch optimization.

    This is the basic class. The only required method is the return_next_dispatch method.

    The return_next_dispatch method for this class returns the initial values for states,
    zeros for controls, and nothing for measurements.

    Parameters
    ----------
    t_window : float
        look ahead time horizon for MPC (in minutes)
    dt : float
        constant time step (in minutes)
    states : dict
        dictionary of information about state variables
    control : dict
        dictionary of information about control variables
    measurements : dict or None, optional
        dictionary of information about measurement variables

    Attributes
    ----------
    t_window : float
        look ahead time horizon for MPC (in minutes)
    dt : float
        constant time step (in minutes)
    n : int
        number of steps to take in time horizon
    states : dict
        dictionary of information about state variables
    control : dict
        dictionary of information about control variables
    measurements : dict or None, optional
        dictionary of information about measurement variables

    Methods
    -------
    check_states_control_measurements_dicts(name, test_dict)
        checks that states, control, and measurement dictionaries are properly inputted
    return_next_dispatch(rewards, x_init)
        returns state, control, and measurement information at optimal dispatch

    """

    def __init__(
        self,
        t_window=60.0 * 12.0,
        dt=5.0,
        states={},
        control={},
        measurements=None,
        objective={},
        **specs,
    ):
        # get time window, time step, and number of steps to take
        assert isinstance(t_window, float), "t_window must be float."
        assert isinstance(dt, float), "dt must be float."
        self.t_window = t_window
        self.dt = dt
        self.n = int(self.t_window / self.dt)

        # ensure states input dictionary has everything needed
        assert isinstance(states, dict), "states must be dictionary."
        self.check_states_control_measurements_dicts("states", states)
        self.states = states

        # ensure control input dictionary has everything needed
        assert isinstance(control, dict), "control must be dictionary."
        self.check_states_control_measurements_dicts("control", control)
        self.control = control

        # ensure optional measurements dictionary has everything needed
        if measurements is not None:
            assert isinstance(
                measurements, dict
            ), "measurements must be dictionary or None."
            self.check_states_control_measurements_dicts("measurements", measurements)
        self.measurements = measurements

        # ensure objective dictionary has everything needed
        assert isinstance(objective, dict), "objective must be dictionary."
        assert "sense" in objective, "objective must contain 'sense' key"
        assert objective["sense"] in [
            "maximize",
            "minimize",
        ], "'sense' must be either 'maximize' or 'minimize'"
        objective_keys = list(objective.keys())
        objective_keys.remove("sense")
        for key in objective_keys:
            # take care of state information
            assert (
                "state_multiplier" in objective[key]
            ), f"'state_multiplier' list must be in {key} for objective dictionary."
            assert isinstance(
                objective[key]["state_multiplier"], list
            ), f"'state_multiplier' in {key} for objective dictionary must be list."
            assert len(objective[key]["state_multiplier"]) == len(
                self.states["order"]
            ), f"number of states in {key} for objective dictionary must be same as in states dictionary."
            # take care of control information
            assert (
                "control_multiplier" in objective[key]
            ), f"'control_multiplier' list must be in {key} for objective dictionary."
            assert isinstance(
                objective[key]["control_multiplier"], list
            ), f"'control_multiplier' in {key} for objective dictionary must be list."
            assert len(objective[key]["control_multiplier"]) == len(
                self.control["order"]
            ), f"number of control variables in {key} for objective dictionary must be same as in control dictionary."
            # take care of measurement information (optional)
            if "measurement_multiplier" in objective[key]:
                assert isinstance(
                    objective[key]["measurement_multiplier"], list
                ), f"'measurement_multiplier' in {key} for objective dictionary must be list."
                assert isinstance(
                    self.measurements, dict
                ), f"to use 'measurement_multiplier' in {key} for objective dictionary, measurement dictionary must be defined."
                assert len(objective[key]["measurement_multiplier"]) == len(
                    self.measurements["order"]
                ), f"number of measurement variables in {key} for objective dictionary must be same as in measurement dictionary."
        self.objective = objective

    def check_states_control_measurements_dicts(self, name, test_dict):
        """
        Checks that all required keys are in dictionary, all values are lists, and
        all lists have the same length

        Parameters
        ----------
        name : str
            name of the dictionary
        test_dict : dict
            dictionary to test

        """
        # required keys
        req_keys = ["order", "lb", "ub"]
        lens = []
        for key in req_keys:
            assert key in test_dict, f"{key} missing from {name} dictionary."
            assert isinstance(test_dict[key], list), f"{key} in {name} must be list."
            lens.append(len(test_dict[key]))
        assert all(lens), f"all lists in {name} must have same length."

    def return_next_dispatch(self, rewards, x_init):
        """
        Solves the Pyomo ConcreteModel and returns state, control, and measurement values of next step

        Parameters
        ----------
        rewards : dict
            dictionary keys are names of reward/price, values are numpy.ndarray or list of n reward/price samples
        x_init : numpy.ndarray or list
            initial state values in order given by states['order']

        Returns
        -------
        result : dict
            dictionary with states, control, and measurements values in lists

        """

        # return values of states, control, measurements
        result = {"states": [], "control": [], "measurements": []}
        # states
        for i in range(len(x_init)):
            result["states"].append(x_init[i])
        # control
        for i in range(len(self.control["order"])):
            result["control"].append(0.0)
        # measurements
        if self.measurements is not None:
            for i in range(len(self.measurements["order"])):
                result["measurements"].append(0.0)

        return result


def clip_and_integrate_action(
    current_values,
    raw_action,
    bounds=None,
    delta_limits=None,
):
    """
    Generic helper to clip action deltas and integrate them into bounded setpoints.

    Parameters
    ----------
    current_values : sequence of float
        Current setpoint values (length = n).
    raw_action : sequence of float
        Policy output (deltas) with the same length as `current_values`.
    bounds : sequence of (low, high) or single (low, high), optional
        Per-dimension bounds. A single tuple applies to all dimensions.
        Defaults to unbounded if not provided.
    delta_limits : float or sequence of float, optional
        Maximum absolute delta per dimension. A scalar applies to all dims.
        Defaults to np.inf (no per-step limit) if not provided.
    """
    curr = np.asarray(current_values, dtype=float).reshape(-1)
    action = np.asarray(raw_action, dtype=float).reshape(-1)
    if action.size != curr.size:
        raise ValueError("raw_action must have the same length as current_values")

    n = curr.size
    if bounds is None:
        bounds = [(float("-inf"), float("inf"))] * n
    elif isinstance(bounds, tuple) and len(bounds) == 2 and not isinstance(
        bounds[0], (list, tuple)
    ):
        bounds = [bounds] * n
    if len(bounds) != n:
        raise ValueError("bounds must match the length of current_values")

    lows = np.array([float(b[0]) for b in bounds], dtype=float)
    highs = np.array([float(b[1]) for b in bounds], dtype=float)

    limits = np.asarray(delta_limits if delta_limits is not None else np.inf, dtype=float).reshape(-1)
    if limits.size == 1:
        limits = np.full(n, limits.item(), dtype=float)
    if limits.size != n:
        raise ValueError("delta_limits must be scalar or match the length of current_values")

    deltas = np.clip(action, -limits, limits)
    new_values = np.clip(curr + deltas, lows, highs)
    return deltas, new_values


def bounded_action_update(
    voltage: float,
    temperature: float,
    raw_action,
    voltage_bounds=(0.0, float("inf")),
    temp_bounds=(0.0, float("inf")),
    dv_limit: float = 0.5,
    dt_limit: float = 5.0,
):
    """
    Backward-compatible 2D helper for voltage/temperature control.
    Prefer `clip_and_integrate_action` for new use cases.
    """
    deltas, new_values = clip_and_integrate_action(
        current_values=[voltage, temperature],
        raw_action=raw_action,
        bounds=[voltage_bounds, temp_bounds],
        delta_limits=[dv_limit, dt_limit],
    )
    return float(deltas[0]), float(deltas[1]), float(new_values[0]), float(new_values[1])


class RLSetpointController:
    """
    Generic policy wrapper to apply bounded setpoint deltas.

    - Works with any number of setpoints.
    - Keeps backward compatibility with the voltage/temperature workflow.
    """

    def __init__(
        self,
        policy_fn: Callable,
        setpoint_names=None,
        init_setpoints=None,
        bounds=None,
        delta_limits=None,
        obs_builder: Optional[Callable] = None,
        init_voltage: float = 0.0,
        init_temperature: float = 0.0,
        dv_limit: float = 0.5,
        dt_limit: float = 5.0,
        voltage_bounds=(0.0, float("inf")),
        temp_bounds=(0.0, float("inf")),
    ):
        self.policy_fn = policy_fn
        self.obs_builder = obs_builder

        base_setpoints = (
            [init_voltage, init_temperature] if init_setpoints is None else init_setpoints
        )
        self.setpoints = np.asarray(base_setpoints, dtype=float).reshape(-1)
        if self.setpoints.size == 0:
            raise ValueError("init_setpoints must contain at least one value")

        n = self.setpoints.size
        if setpoint_names is None:
            setpoint_names = ["voltage", "temperature"] if n == 2 else [f"setpoint_{i}" for i in range(n)]
        if len(setpoint_names) != n:
            raise ValueError("setpoint_names must match the number of setpoints")
        self.setpoint_names = list(setpoint_names)

        if bounds is None:
            if n == 2:
                bounds = [voltage_bounds, temp_bounds]
            else:
                bounds = [(float("-inf"), float("inf"))] * n
        if len(bounds) != n:
            raise ValueError("bounds must match the number of setpoints")
        self.bounds = [(float(b[0]), float(b[1])) for b in bounds]
        self._bounds_low = np.array([b[0] for b in self.bounds], dtype=float)
        self._bounds_high = np.array([b[1] for b in self.bounds], dtype=float)

        if delta_limits is None:
            if n == 2:
                delta_limits = [dv_limit, dt_limit]
            else:
                delta_limits = [np.inf] * n
        delta_arr = np.asarray(delta_limits, dtype=float).reshape(-1)
        if delta_arr.size == 1:
            delta_arr = np.full(n, delta_arr.item(), dtype=float)
        if delta_arr.size != n:
            raise ValueError("delta_limits must be scalar or match the number of setpoints")
        self.delta_limits = delta_arr

        self.history = []
        self.reset(*self.setpoints.tolist())

    def reset(self, *setpoint_values) -> None:
        """
        Reset internal setpoints.
        - Pass a sequence or individual values matching the controller dimension.
        - If called with no arguments, uses the current setpoints.
        """
        if len(setpoint_values) == 0:
            values = self.setpoints
        elif len(setpoint_values) == 1 and not isinstance(setpoint_values[0], (int, float)):
            values = np.asarray(setpoint_values[0], dtype=float).reshape(-1)
        else:
            values = np.asarray(setpoint_values, dtype=float).reshape(-1)

        if values.size != self.setpoints.size:
            raise ValueError("reset values must match the number of setpoints")
        self.setpoints = np.clip(values, self._bounds_low, self._bounds_high)
        self.history = []

    def _build_observation(
        self,
        measured_current: Optional[float],
        target_current: Optional[float],
        observation,
    ) -> np.ndarray:
        if observation is not None:
            return np.asarray(observation, dtype=np.float32)
        if self.obs_builder is not None:
            return np.asarray(self.obs_builder(measured_current, target_current), dtype=np.float32)
        if measured_current is None or target_current is None:
            raise ValueError("Provide observation or (measured_current, target_current)")
        return np.array([measured_current, target_current], dtype=np.float32)

    def step(
        self,
        measured_current: Optional[float] = None,
        target_current: Optional[float] = None,
        timestamp: Optional[float] = None,
        observation=None,
    ):
        deltas, new_setpoints, raw_action, entry = self.step_vector(
            measured_current=measured_current,
            target_current=target_current,
            timestamp=timestamp,
            observation=observation,
        )
        if new_setpoints.size >= 2:
            return deltas, float(new_setpoints[0]), float(new_setpoints[1]), raw_action, entry
        if new_setpoints.size == 1:
            return deltas, float(new_setpoints[0]), None, raw_action, entry
        return deltas, None, None, raw_action, entry

    def step_vector(
        self,
        measured_current: Optional[float] = None,
        target_current: Optional[float] = None,
        timestamp: Optional[float] = None,
        observation=None,
    ):
        obs = self._build_observation(measured_current, target_current, observation)
        raw_action = np.asarray(self.policy_fn(obs), dtype=np.float32)
        deltas, new_setpoints = clip_and_integrate_action(
            current_values=self.setpoints,
            raw_action=raw_action,
            bounds=self.bounds,
            delta_limits=self.delta_limits,
        )

        self.setpoints = new_setpoints

        entry = {
            "timestamp": time.time() if timestamp is None else timestamp,
            "observation": obs.tolist(),
            "raw_action": raw_action.tolist(),
            "deltas": deltas.tolist(),
            "setpoints": new_setpoints.tolist(),
            "measured_current": None if measured_current is None else float(measured_current),
            "target_current": None if target_current is None else float(target_current),
        }
        for name, delta, sp in zip(self.setpoint_names, deltas, new_setpoints):
            entry[f"{name}_delta"] = float(delta)
            entry[f"{name}_command"] = float(sp)

        self.history.append(entry)
        return deltas, new_setpoints, raw_action, entry
