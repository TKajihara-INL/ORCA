import os
import pickle

import numpy as np
import pyomo.environ as pyo
import pyomo.dae as pde
import xml.etree.ElementTree as ET

from ORCA.Basic.Optimization import Optimization


class DAE_LTIStateSpaceMPCPyomoOptimization(Optimization):
    """
    Dispatch optimization using MPC in Pyomo with LTI state-space representation.

    The model is given by:

    x_k = Ax_{k-1} + Bu_{k-1}
    y_k = Cx_k

    Parameters
    ----------
    solver : str
        name of solver for Pyomo to use
    mode : str
        type of optimization, reward (LMP) or setpoint following (setpoint)
    matrices : str
        path to file containing A, B, C matrices
    t_window : float
        look ahead time horizon for MPC (in minutes) (Basic.Optimization initializes)
    dt : float
        constant time step (in minutes) (Basic.Optimization initializes)
    states : dict
        dictionary of information about state variables (Basic.Optimization initializes)
    control : dict
        dictionary of information about control variables (Basic.Optimization initializes)
    measurements : dict or None, optional
        dictionary of information about measurement variables (Basic.Optimization initializes)
    objective : dict
        dictionary of information about the objective function (Basic.Optimization initializes)

    Attributes
    ----------
    solver : pyomo.environ.SolverFactory
        solver for Pyomo problem
    A : numpy.ndarray
        state transition matrix
    B : numpy.ndarray
        control matrix
    C : numpy.ndarray or None, optional
        measurement matrix
    model : pyomo.environ.ConcreteModel
        Pyomo optimization model
    t_window : float
        look ahead time horizon for MPC (in minutes) (Basic.Optimization initializes)
    dt : float
        constant time step (in minutes) (Basic.Optimization initializes)
    n : int
        number of steps to take in time horizon (Basic.Optimization initializes)
    states : dict
        dictionary of information about state variables (Basic.Optimization initializes)
    control : dict
        dictionary of information about control variables (Basic.Optimization initializes)
    measurements : dict or None, optional
        dictionary of information about measurement variables (Basic.Optimization initializes)
    objective : dict
        dictionary of information about the objective function (Basic.Optimization initializes)

    Methods
    -------
    return_next_dispatch(rewards, x_init)
        returns state, control, and measurement information at optimal dispatch
    load_state_space_matrices(matrices)
        loads A, B, C matrices from specified file
    x_bounds(model, i, t)
        returns bounds for state variables
    u_bounds(model, i, t)
        returns bounds for control variables
    y_bounds(model, i, t)
        returns bounds for measurement variables
    initialize_reward(model, i, t)
        initializes reward/price arrays
    state_equation(model, i, t)
        state equation for use as constraint
    measurement equation(model, i, t)
        measurement variable equation for use as constraint
    objective_kernel(i, t)
        helper function to define objective function
    objective_kernel_for_setpoint_following(i, t)
        helper function to define objective function with setpoint following
    solve_model(rewards, x_init)
        solves the Pyomo ConcreteModel

    """

    def __init__(self, solver="ipopt", matrices=None, mode=0, **specs):
        super().__init__(**specs)

        try:
            self.t_sim = specs["t_sim"]
        except:
            self.t_sim = np.arange(0, self.n + 1, dtype=int)
        self.q1_sim = specs["q1_sim"]
        self.q2_sim = specs["q2_sim"]
        self.t1_setpoint = specs["t1_setpoint"]
        self.t = specs["t"]
        self.solver = solver
        if self.objective["sense"] == "maximize":
            sense = pyo.maximize
        else:
            sense = pyo.minimize

        self.x_init = [0, 0, 0, 0]

        self.model = pyo.ConcreteModel()

        # read in A, B, C matrices from file
        assert os.path.isfile(
            matrices
        ), f"{matrices} file for A, B, C matrices not found."
        assert matrices.lower()[-3:] in [
            "pkl",
            "xml",
        ], f"A, B, C matrices file must be pkl or xml."
        self.load_state_space_matrices(matrices)

        # check that inputs are sufficient to build the Pyomo model
        assert isinstance(
            self.A, np.ndarray
        ), f"A loaded from {matrices} must be numpy.ndarray."
        assert isinstance(
            self.B, np.ndarray
        ), f"B loaded from {matrices} must be numpy.ndarray."
        assert (
            self.A.shape[0] == self.B.shape[0]
        ), f"A and B loaded from {matrices} must have same number of rows."
        if self.C is not None:
            assert isinstance(
                self.C, np.ndarray
            ), f"C loaded from {matrices} must be numpy.ndarray."
            assert (
                self.C.shape[1] == self.A.shape[0]
            ), f"C loaded from {matrices} must have same number of columns as x entries."

        # set up index for variables
        self.model.t = pde.ContinuousSet(initialize=self.t_sim)  # time
        # self.model.t = pde.ContinuousSet(initialize=np.arange(0, self.n+1, dtype=int)) # time
        self.model.xi = pyo.Set(
            initialize=np.arange(0, len(self.states["order"]), dtype=int)
        )  # states
        self.model.ui = pyo.Set(
            initialize=np.arange(0, len(self.control["order"]), dtype=int)
        )  # control

        # initial state (updated when run successively)
        self.model.x_init = pyo.Param(
            self.model.xi, initialize=[0.0] * len(self.states["order"]), mutable=True
        )

        # state variables
        self.model.x = pyo.Var(
            self.model.xi,
            self.model.t,
            domain=pyo.NonNegativeReals,
            bounds=self.x_bounds,
        )

        # control variables
        self.model.u = pyo.Var(
            self.model.ui,
            self.model.t,
            domain=pyo.NonNegativeReals,
            bounds=self.u_bounds,
        )

        # measurement variables
        if self.measurements is not None:
            self.model.yi = pyo.Set(
                initialize=np.arange(0, len(self.measurements["order"]), dtype=int)
            )
            self.model.y = pyo.Var(
                self.model.yi,
                self.model.t,
                domain=pyo.NonNegativeReals,
                bounds=self.y_bounds,
            )
        else:
            self.model.yi = None
            self.model.y = None

        self.model.xdot = pde.DerivativeVar(self.model.x, wrt=self.model.t)

        self.model.x_ode = pyo.Constraint(
            self.model.xi, self.model.t, rule=self.diffx_equation
        )

        self.model.measurement = pyo.Constraint(
            self.model.yi, self.model.t, rule=self.y_equation
        )

        # input specifications
        self.model.q1sim = pyo.Constraint(
            range(0, len(self.t_sim)),
            rule=lambda m, k: m.u[0, self.t_sim[k]] == self.q1_sim[k],
        )
        self.model.q2sim = pyo.Constraint(
            range(0, len(self.t_sim)),
            rule=lambda m, k: m.u[1, self.t_sim[k]] == self.q2_sim[k],
        )
        if mode == 0:
            # initial conditions
            self.model.initialization = pyo.Constraint(
                self.model.xi, rule=self.initcond
            )
        elif mode == 1:

            for k in self.model.xi:
                self.model.x[k, self.t].fix(0)

            self.model.obj = pyo.Objective(
                expr=sum(
                    [
                        (self.t1_setpoint(t) - (self.model.x[2, t] + self.x_init[2]))
                        ** 2
                        for t in self.model.t
                    ]
                ),
                sense=sense,
            )

        pyo.TransformationFactory("dae.finite_difference").apply_to(
            self.model, nfe=len(self.model.t), scheme="BACKWARD"
        )
        pyo.SolverFactory(self.solver).solve(self.model, tee=False)

    # equation funcitons
    def diffx_equation(self, model, i, t):
        sol = sum(
            self.A[i, j] * model.x[j, t] for j in model.xi
        )  ### check ABC matric is properly loaded
        sol += sum(self.B[i, k] * model.u[k, t] for k in model.ui)
        return self.model.xdot[i, t] == sol

    def y_equation(self, model, i, t):
        sol = sum(self.C[i, j] * model.x[j, t] for j in model.xi)
        return model.y[i, t] == sol

    def initcond(self, model, i):
        x_init = np.zeros(len(self.states["order"]))
        return model.x[i, 0] == self.x_init[i]

    def load_state_space_matrices(self, matrices_path):
        """
        Loads the state space matrices (A, B, C) into the specs from the
        file given in the specs. This file can be a pickled dictionary
        or a RAVEN DMDc metadata XML file.

        Parameters
        ----------
        matrices_path : str
            path to file containing A, B, C matrices

        """

        if matrices_path.lower().endswith("xml"):
            tree = ET.parse(matrices_path)
            root = tree.getroot()
            self.A = self.load_matrix_from_xml(root, "A")
            self.B = self.load_matrix_from_xml(root, "B")
            self.C = self.load_matrix_from_xml(root, "C")
        else:
            # should be a pickled dictionary
            try:
                with open(matrices_path, "rb") as f:
                    matrices = pickle.load(f)
            except Exception as e:
                # lots of different errors may be thrown, that is why this is general
                raise ValueError(
                    f"matrices pickle file {matrices_path} had errors: ",
                    e,
                )

            # store matrices
            self.A = matrices["A"]
            self.B = matrices["B"]
            self.C = matrices["C"]

    def load_matrix_from_xml(self, root, letter):
        """
        Loads the specified matrix from RAVEN DMDc XML

        Parameters
        ----------
        root : xml.etree.ElementTree
            root of XML
        letter : str
            'A', 'B', or 'C'

        Returns
        -------
        matrix : numpy.ndarray
            A, B, or C matrix

        """

        tilde = root.find(f".//{letter}tilde")
        # shape of matrix
        shape = tilde.find(".//matrixShape").text.split(",")
        shape = [int(tmp) for tmp in shape]
        # values of matrix
        vals = tilde.find(".//real").text.split(" ")
        vals = [float(tmp) for tmp in vals]
        # cast to numpy array
        matrix = np.array(vals)
        # reshape (uses Fortran ordering, first index fastest then second)
        matrix = matrix.reshape(shape, order="F")

        return matrix

    def x_bounds(self, model, i, t):
        """
        Function to return lower and upper bounds for state variables

        Parameters
        ----------
        model : pyomo.environ.ConcreteModel
            Pyomo ConcreteModel
        i : int
            row index for states
        t : int
            time index for states

        Returns
        -------
        bounds : tuple
            tuple of (lower, upper) bounds for state variables

        """

        return (self.states["lb"][i], self.states["ub"][i])

    def u_bounds(self, model, i, t):
        """
        Function to return lower and upper bounds for control variables

        Parameters
        ----------
        model : pyomo.environ.ConcreteModel
            Pyomo ConcreteModel
        i : int
            row index for control vector
        t : int
            time index for controls

        Returns
        -------
        bounds : tuple
            tuple of (lower, upper) bounds for control variables

        """

        return (self.control["lb"][i], self.control["ub"][i])

    def y_bounds(self, model, i, t):
        """
        Function to return lower and upper bounds for control variables

        Parameters
        ----------
        model : pyomo.environ.ConcreteModel
            Pyomo ConcreteModel
        i : int
            row index for measurement variables
        t : int
            time index for measurements

        Returns
        -------
        bounds : tuple
            tuple of (lower, upper) bounds for measurement variables

        """

        return (self.measurements["lb"][i], self.measurements["ub"][i])

    def initialize_reward(self, model, i, t):
        """
        Function to initialize reward/price data by setting to zero

        Parameters
        ----------
        model : pyomo.environ.ConcreteModel
            Pyomo ConcreteModel
        i : int
            row index for reward/price variable
        t : int
            time index for reward/price variable

        Returns
        -------
        value : float
            zero

        """

        return 0.0

    def state_equation(self, model, i, t):
        """
        Returns a Pyomo expression for the state equation

        x_k = A*x_{k-1} + B*u_{k-1}

        Parameters
        ----------
        model : pyomo.environ.ConcreteModel
            Pyomo ConcreteModel
        i : int
            state row index
        t : int
            time index

        """

        if t == self.model.t.first():
            # return initial state
            sol = self.model.x_init[i]
        else:
            # state update
            sol = (
                sum(self.A[i, j] * self.model.x[j, t - 1] for j in self.model.xi)
                * self.dt
                + self.model.x[i, t - 1]
            )
            sol += (
                sum(self.B[i, k] * self.model.u[k, t - 1] for k in self.model.ui)
                * self.dt
            )

        return self.model.x[i, t] == sol

    def measurement_equation(self, model, i, t):
        """
        Returns a Pyomo expression for the measurement equation

        y_k = C*x_k

        Parameters
        ----------
        model : pyomo.environ.ConcreteModel
            Pyomo ConcreteModel
        i : int
            measurement row index
        t : int
            time index

        """

        sol = sum(self.C[i, j] * self.model.x[j, t] for j in self.model.xi)

        return self.model.y[i, t] == sol

    def objective_kernel(self, i, t):
        """
        Returns a Pyomo expression for the ith row and tth time for the objective function

        Parameters
        ----------
        i : int
            reward/price index
        t : int
            time index

        Returns
        -------
        f : pyomo
            Pyomo EqualityExpression

        """

        f = self.model.P[i, t] * (
            sum(
                self.objective[i]["state_multiplier"][j] * self.model.x[j, t]
                for j in self.model.xi
            )
            + sum(
                self.objective[i]["control_multiplier"][k] * self.model.u[k, t]
                for k in self.model.ui
            )
        )
        if self.measurements is not None:
            f += self.model.P[i, t] * (
                sum(
                    self.objective[i]["measurement_multiplier"][l] * self.model.y[l, t]
                    for l in self.model.yi
                )
            )
        return f

    def solve_model(self, rewards, x_init):
        """
        Solves the Pyomo ConcreteModel

        Parameters
        ----------
        rewards : dict
            dictionary keys are names of reward/price values are numpy.ndarray or list of n reward/price samples
        x_init : numpy.ndarray or list
            initial state values in order given by states['order']

        Returns
        -------
        results : pyomo

        """
        # update the reward/price in the Pyomo ConcreteModel
        for key in rewards:
            for x in self.model.t:
                self.model.P[key, x] = rewards[key][x]
        #         print(self.model.P[('LMP',0)])
        # update initial state values
        for i in self.model.xi:
            self.model.x_init[i] = x_init[i]
        # solve model
        results = self.solver.solve(self.model, tee=True)  # verbose solve
        #         results = self.solver.solve(self.model)
        #         self.full_debug(self.model)
        #         print(self.A)
        #         print(self.B)
        #         self.model.pprint()  #really verbose solve
        #         results.write()
        return results

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
            dictionary with states, control, and measurements values

        """

        # solve the model
        _ = self.solve_model(rewards, x_init)
        # return values of states, control, measurements
        result = {"states": [], "control": [], "measurements": []}
        # states
        for i in self.model.xi:
            result["states"].append(pyo.value(self.model.x[i, 1]))
        # control
        for i in self.model.ui:
            result["control"].append(pyo.value(self.model.u[i, 1]))
        # measurements
        if self.measurements is not None:
            for i in self.model.yi:
                result["measurements"].append(pyo.value(self.model.y[i, 1]))
        return result
