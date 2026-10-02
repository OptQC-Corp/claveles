# Copyright 2026 OptQC Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this file are derived from MQC3,
# originally licensed under the MIT License.

"""Dependency DAG constructor.

Dependency DAG is a Directed Acyclic Graph that represents the dependency of operations.
This module provides a class to construct the dependency graph from a quantum circuit.
"""

import math
from collections import defaultdict
from math import pi

import networkx as nx

import claveles.circuit.ops.intrinsic as cops
import claveles.graph.ops as gops
from claveles.circuit.program import CircOpParam, CircuitRepr
from claveles.circuit.state import HardwareConstrainedSqueezedState
from claveles.feedforward import FeedForward, ff_to_add_constant
from claveles.graph.constant import BLANK_MODE
from claveles.graph.embed._utility import _convert_cg_param, _is_displacement, convert_op
from claveles.graph.ops import GraphOpParam, Measurement, ModeMeasuredVariable
from claveles.graph.program import GraphRepr
from claveles.graph.program import Operation as GraphOp
from claveles.pb.claveles_cloud.program.v1.graph_pb2 import GraphOperation as PbOperation
from claveles.support import check_feature_representation, state_support

DEFAULT_COORDINATE = (-1, -1)


def _add_c_disp_to_g_disp(disp1: CircOpParam, disp2: GraphOpParam) -> GraphOpParam:
    """Construct a new displacement for graph by adding two displacements.

    Args:
        disp1 (CircOpParam): Displacement 1.
        disp2 (GraphOpParam): Displacement 2.

    Returns:
        GraphOpParam: New displacement.

    Raises:
        TypeError: Applying multiple displacement with feedforward is not allowed.
    """
    if isinstance(disp1, FeedForward):
        if isinstance(disp2, FeedForward):
            msg = "Applying multiple displacements with feedforward is not allowed."
            raise TypeError(msg)

        x1 = _convert_cg_param(disp1)
        x2 = disp2
    else:
        x1 = disp2
        x2 = disp1

    ff_func = ff_to_add_constant(x2)

    return ff_func(x1)


def _add_g_disps(disp1: GraphOpParam, disp2: GraphOpParam) -> GraphOpParam:
    """Construct a new displacement for graph by adding two displacements.

    Args:
        disp1 (GraphOpParam): Displacement 1.
        disp2 (GraphOpParam): Displacement 2.

    Returns:
        GraphOpParam: New displacement.

    Raises:
        TypeError: Applying multiple displacement with feedforward is not allowed.
    """
    if isinstance(disp1, FeedForward):
        if isinstance(disp2, FeedForward):
            msg = "Applying multiple displacements with feedforward is not allowed."
            raise TypeError(msg)
        x1, x2 = disp1, disp2
    else:
        x1, x2 = disp2, disp1

    return ff_to_add_constant(x2)(x1)


class DependencyDAG:
    """Directed Acyclic Graph which represents dependency relationship between operations."""

    def __init__(self, circuit_or_graph: CircuitRepr | GraphRepr) -> None:
        """Initializes the DependencyDAG by constructing its internal dependency graph.

        Args:
            circuit_or_graph (CircuitRepr | GraphRepr):
                The input quantum circuit representation or graph representation
                from which to build the DependencyDAG.
        """
        self.dag: nx.DiGraph

        builder = _DependencyBuilder()
        if isinstance(circuit_or_graph, CircuitRepr):
            self.dag = builder.from_circuit(circuit_or_graph)
        else:
            self.dag = builder.from_graph(circuit_or_graph)
        self.n_modes = len(builder.last_op_of_modes)


class _DependencyBuilder:
    next_node_id: int
    dep_graph: nx.DiGraph
    last_disp_of_modes: dict[int, tuple[GraphOpParam, GraphOpParam] | None]
    last_op_of_modes: dict[int, int]
    measurement_to_mode: dict[int, int]

    def __init__(self) -> None:
        pass

    def _clear(self) -> None:
        """Clear the dependency builder."""
        self.next_node_id = 0
        self.dep_graph = nx.DiGraph()
        self.last_disp_of_modes = defaultdict(lambda: None)
        self.last_op_of_modes = {}
        self.measurement_to_mode = {}

    def _apply_displacement(self, target: int) -> None:
        modes = self.dep_graph.nodes[target]["modes"]
        for mode in modes:
            prev_disp = self.last_disp_of_modes[mode]
            if prev_disp is not None:
                self.dep_graph.nodes[target]["displacements"].append((mode, prev_disp))
                self.last_disp_of_modes[mode] = None

    def _apply_dependency(self, target: int) -> None:
        modes = self.dep_graph.nodes[target]["modes"]
        for mode in modes:
            if mode in self.last_op_of_modes:
                self.dep_graph.add_edge(self.last_op_of_modes[mode], target)

            self.last_op_of_modes[mode] = target

    def _apply_feedforward(self, target: int) -> None:
        op: GraphOp = self.dep_graph.nodes[target]["op"]

        params = list(op.parameters)
        params.extend(self.dep_graph.nodes[target]["displacements"])
        for p in params:
            if not isinstance(p, FeedForward):
                continue
            if not isinstance(p.variable, ModeMeasuredVariable):
                msg = "The type of parameters of operation does not match."
                raise TypeError(msg)
            if p.variable.mode not in self.measurement_to_mode:
                msg = "The mode of ModeMeasuredVariable does not match."
                raise ValueError(msg)
            measurement_ind = self.measurement_to_mode[p.variable.mode]

            for n in self.dep_graph.predecessors(target):
                # avoid deadlock
                if nx.has_path(self.dep_graph, measurement_ind, n):
                    continue
                self.dep_graph.add_edge(n, measurement_ind)

            self.dep_graph.add_edge(measurement_ind, target)
            self.dep_graph.nodes[target]["has_nlff"] = True

    def _add_op_node(self, op: GraphOp, modes: list[int]) -> None:
        self.dep_graph.add_node(self.next_node_id)
        self.dep_graph.nodes[self.next_node_id]["op"] = op
        self.dep_graph.nodes[self.next_node_id]["modes"] = modes
        self.dep_graph.nodes[self.next_node_id]["displacements"] = []

        if isinstance(op, Measurement):
            self.measurement_to_mode[modes[0]] = self.next_node_id

        self._apply_displacement(self.next_node_id)
        self._apply_dependency(self.next_node_id)
        self._apply_feedforward(self.next_node_id)

        self.next_node_id += 1

    def _add_initialization(self, mode: int, theta: float = math.pi / 2) -> None:
        init = gops.Initialization((DEFAULT_COORDINATE), theta=theta, initialized_modes=(BLANK_MODE, mode))
        self._add_op_node(init, [mode])

    def _add_displacement(self, disp: cops.Displacement, mode: int) -> None:
        prev_disp = self.last_disp_of_modes[mode]
        if prev_disp is None:
            self.last_disp_of_modes[mode] = _convert_cg_param(disp.x), _convert_cg_param(disp.p)
        else:
            x, p = prev_disp
            self.last_disp_of_modes[mode] = (
                _add_c_disp_to_g_disp(disp.x, x),
                _add_c_disp_to_g_disp(disp.p, p),
            )

    def _add_displacement_from_graph(self, mode: int, params: tuple[GraphOpParam, GraphOpParam]) -> None:
        prev_disp = self.last_disp_of_modes[mode]
        if prev_disp is None:
            self.last_disp_of_modes[mode] = params
        else:
            self.last_disp_of_modes[mode] = (
                _add_g_disps(params[0], prev_disp[0]),
                _add_g_disps(params[1], prev_disp[1]),
            )

    def _reset_op(self, op: GraphOp) -> GraphOp:
        """Return the operation stripped of where it sat, ready to be placed somewhere else.

        A copy, not the operation itself: `from_graph` reads these off the caller's `GraphRepr`, and
        clearing them in place left that graph holding operations that reported `(-1, -1)` while still
        being filed under their real coordinates. Embedding a graph is a read of it.

        Args:
            op: The operation as the source graph holds it.

        Returns:
            GraphOp: A copy with its coordinate, swap and displacements cleared.
        """
        return op.replace(
            macronode=DEFAULT_COORDINATE,
            swap=False,
            displacement_k_minus_1=(0.0, 0.0),
            displacement_k_minus_n=(0.0, 0.0),
        )

    def _refuse_post_selection(self, circuit: CircuitRepr) -> None:
        """Refuse a circuit that forces a measurement outcome.

        Post-selection is something only the simulator can do, and a graph representation has no field
        to carry it, so a conversion that went ahead would produce a program that samples the
        measurement instead. The operations are read directly rather than through a serialized circuit,
        so that a circuit with open inputs is answered rather than refused: whether anything
        post-selects has nothing to do with which modes carry a state.

        Args:
            circuit: The circuit being converted.

        Raises:
            ValueError: If any measurement in the circuit post-selects.
        """  # noqa: DOC502  # raised by check_feature_representation
        if any(op.post_select is not None for op in circuit):
            check_feature_representation(
                "post_select", "graph", "Remove post_select before converting the circuit to a graph."
            )

    def _refuse_states_a_graph_cannot_carry(self, circuit: CircuitRepr) -> None:
        """Refuse a circuit whose modes carry a state the machine cannot prepare.

        The machine supplies its own squeezed state on every mode, so a placed state reaches a graph only
        when it is the one the machine already provides; which states those are is recorded in
        :mod:`claveles.support`, next to the rest of what each state can be used for. Anything else would
        become a default initialization, and a component spanning several modes would come apart into one
        initialization per mode, leaving the correlation between them behind.

        Refused rather than done, because nothing in the result would show that the program no longer
        starts where the circuit says it does. Two placements are open to a caller writing for the
        machine, and the message names both: leaving the mode open, which asks for the state the machine
        supplies, and placing a
        :class:`~claveles.circuit.state.HardwareConstrainedSqueezedState`, which is how the squeezing angle
        of an initialized mode is chosen. Only the placements that cannot be used are named for removal,
        so following the message does not cost a caller an angle they had set.

        Args:
            circuit: The circuit being converted.

        Raises:
            ValueError: If any mode carries a state a graph representation cannot express.
        """
        refused = [
            (component_modes, component)
            for component_modes, component in circuit.placed_components
            if not state_support(component).hardware
        ]
        if not refused:
            return

        modes = sorted({mode for component_modes, _ in refused for mode in component_modes})
        message = (
            f"Mode(s) {modes} carry a state that is not a HardwareConstrainedSqueezedState, which a graph "
            "representation cannot express: the machine supplies its own squeezed state on every mode and "
            "can prepare no other."
        )
        correlated = [component_modes for component_modes, _ in refused if len(component_modes) > 1]
        if correlated:
            message += (
                f" The state on mode(s) {correlated} spans several modes, which the machine would have to "
                "initialize one at a time, losing the correlation between them as well."
            )
        # Positions rather than modes, because that is what `remove` takes, and only of the placements
        # named above: a HardwareConstrainedSqueezedState the circuit places converts as it is, and
        # dropping it would throw away the squeezing angle it was placed for.
        refused_modes = set(modes)
        positions = [
            position
            for position, placement in enumerate(circuit.placements)
            if placement.op.is_state_preparation() and refused_modes.intersection(placement.modes)
        ]
        message += (
            " Leave those modes open, which asks for the state the machine does supply, or place a "
            "HardwareConstrainedSqueezedState to choose the squeezing angle of one. "
            f"`circuit.remove({positions})` drops the placements that cannot be used."
        )
        raise ValueError(message)

    def from_circuit(self, circuit: CircuitRepr) -> nx.DiGraph:
        """Build the dependency DAG of a circuit.

        Args:
            circuit: The circuit to read.

        Returns:
            nx.DiGraph: The dependency graph the embedders place.

        Raises:
            ValueError: If the circuit asks for something a graph representation cannot express.
        """  # noqa: DOC502  # raised by the checks below
        self._clear()

        # Both checks are made here rather than by a client, so that every route from a circuit into a
        # graph -- either embedder, and a `DependencyDAG` a caller builds itself -- answers the same.
        self._refuse_post_selection(circuit)
        self._refuse_states_a_graph_cannot_carry(circuit)

        for mode, initial_state in circuit.placed_states.items():
            # Every mode is initialized the way the hardware does it. A mode left open asks for exactly
            # that, and the check above leaves nothing else here: a placed state is either the hardware's
            # own, whose squeezing angle is carried over, or the conversion was refused.
            if isinstance(initial_state, HardwareConstrainedSqueezedState):
                # The `theta` argument is the measurement angle.
                # `initial_state.phi` is the squeezing angle of the initialized mode.
                self._add_initialization(mode, theta=initial_state.phi - pi / 2)
            else:
                self._add_initialization(mode, theta=-pi / 2)

        for placement in circuit.placements:
            op_in_circ = placement.op
            if op_in_circ.is_state_preparation():
                # The initializations above already account for the placed states.
                continue
            # A list, matching what `from_graph` puts on a node: the DAG's `modes` attribute is one
            # thing whichever side built it.
            modes = list(placement.modes)
            if _is_displacement(op_in_circ):
                # std.Displacement carries a complex amplitude, so lower it to the intrinsic
                # (x, p) form that the displacement bookkeeping below works with.
                for intrinsic_op, op_modes in op_in_circ.to_intrinsic_ops(modes):
                    assert isinstance(intrinsic_op, cops.Displacement)  # noqa: S101
                    self._add_displacement(intrinsic_op, op_modes[0])
                continue

            for op_in_graph, op_modes in convert_op(op_in_circ, (DEFAULT_COORDINATE), modes):
                self._add_op_node(op_in_graph, op_modes)

        return self.dep_graph

    def from_graph(self, graph: GraphRepr) -> nx.DiGraph:
        self._clear()
        modes = [BLANK_MODE] * (graph.n_local_macronodes + 1)
        for w in range(graph.n_steps):
            for h in range(graph.n_local_macronodes):
                left, up = modes[h], modes[-1]
                down, right = up, left
                op = graph.get_operation(h, w)
                if left != BLANK_MODE:
                    self._add_displacement_from_graph(left, op.displacement_k_minus_n)
                if up != BLANK_MODE:
                    self._add_displacement_from_graph(up, op.displacement_k_minus_1)
                if op.type() == PbOperation.OPERATION_TYPE_MEASUREMENT:
                    down, right = BLANK_MODE, BLANK_MODE
                elif op.type() == PbOperation.OPERATION_TYPE_INITIALIZATION:
                    down, right = op.initialized_modes
                elif graph.is_swap_macronode(h, w):
                    down, right = left, up
                if op.type() != PbOperation.OPERATION_TYPE_WIRING:
                    op = self._reset_op(op)
                    self._add_op_node(op, list({left, right, up, down} - {BLANK_MODE}))

                modes[h] = right
                modes[-1] = down

        return self.dep_graph
