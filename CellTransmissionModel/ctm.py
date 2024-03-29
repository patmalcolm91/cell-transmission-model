"""
Implementation of the CTM as described here:
  - https://connected-corridors.berkeley.edu/sites/default/files/atm_on_road_networks.pdf
"""

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
from CellTransmissionModel._Util import LineDataUnits, CircleDataUnits, EventManager
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnchoredText
import warnings


EPS = 1E-6  # epsilon (threshold for small values)


class FundamentalDiagram:
    def __init__(self, flow_capacity=1800, critical_density=18.6, congestion_wave_speed=20):
        """
        Initialize a fundamental diagram.

        :param flow_capacity: flow capacity (veh/h)
        :param critical_density: critical density (veh/km)
        :param congestion_wave_speed: congestion wave speed (km/h)
        """
        self._flow_capacity = flow_capacity
        self._critical_density = critical_density
        self._congestion_wave_speed = congestion_wave_speed

    @property
    def flow_capacity(self):
        return self._flow_capacity

    @property
    def critical_density(self):
        return self._critical_density

    @property
    def free_flow_speed(self):
        return self.flow_capacity / self.critical_density

    @property
    def congestion_wave_speed(self):
        return self._congestion_wave_speed

    @property
    def jam_density(self):
        return self.flow_capacity / self.congestion_wave_speed + self.critical_density

    def flow_at_density(self, density):
        if density < 0 or density > self.jam_density:
            raise ValueError("Provided density value is invalid. " + str(density))
        if density < self.critical_density:
            return self.free_flow_speed * density
        else:
            return self.flow_capacity - self.congestion_wave_speed * (density - self.critical_density)

    def speed_at_density(self, density):
        if density < 0 or density > self.jam_density:
            raise ValueError("Provided density value is invalid.")
        return self.free_flow_speed if density < self.critical_density else self.congestion_wave_speed


class Node:
    def __init__(self, pos, *, id=None, radius=0):
        self.id = id
        self.pos = np.array(pos)
        self.radius = radius
        self.incoming_links = []
        self.outgoing_links = []
        self._split_ratio_matrix = None

    def _generate_even_split_ratio_matrix(self):
        m, n = len(self.incoming_links), len(self.outgoing_links)
        split = 1/n if n > 0 else np.nan
        self._split_ratio_matrix = np.full((m, n), fill_value=split)

    @property
    def split_ratio_matrix(self):
        if self._split_ratio_matrix is None:
            warnings.warn("No split ratio matrix defined for node. Assuming even split.")
            self._generate_even_split_ratio_matrix()
        return self._split_ratio_matrix

    @split_ratio_matrix.setter
    def split_ratio_matrix(self, matrix):
        if matrix.shape != (len(self.incoming_links), len(self.outgoing_links)):
            warnings.warn("Specified split ratio matrix does not match number of incoming and outgoing links.")
        self._split_ratio_matrix = matrix

    def set_link_outgoing_split_ratios(self, link, ratios):
        """
        Set the outgoing split ratios for the specified incoming link. Ratios are defined from left to right relative to
        the direction of the specified link.
        """
        if type(ratios) == list:
            ratios = np.array(ratios)
        ratios = ratios / sum(ratios)  # normalize ratios
        if self._split_ratio_matrix is None:
            self._generate_even_split_ratio_matrix()
        link_index = self.incoming_links.index(link) if isinstance(link, Link) else [l.id for l in self.incoming_links].index(link)
        # get left-to-right order of outgoing links with respect to given incoming link
        _outgoing_link_headings = np.array([link.heading for link in self.outgoing_links])
        _outgoing_link_order = np.argsort(_outgoing_link_headings)[::-1]
        _incoming_ordered_pos = np.searchsorted(_outgoing_link_headings[_outgoing_link_order], link.heading)
        n = len(self.outgoing_links)
        _relative_ordered_indices = _outgoing_link_order[(np.arange(n) + _incoming_ordered_pos) % n]
        # set the ratios
        self._split_ratio_matrix[link_index, _relative_ordered_indices] = ratios

    def compute_flows(self):
        m, n = len(self.incoming_links), len(self.outgoing_links)
        # step 1
        supplies = [link.input_supply for link in self.outgoing_links]
        # step 2
        q = 0
        # step 3
        d_i = np.zeros((n+1, m))  # indices: q, i
        for i, link in enumerate(self.incoming_links):
            d_i[q, i] = link.free_flow_speed*link.density*(min(1, link.flow_capacity / (link.free_flow_speed*link.density)) if link.density > 0 else 1)
        # step 4
        d_j = np.zeros((n+1, n))  # indices: q, j
        for j in range(n):
            d_j[q, j] = sum([self.split_ratio_matrix[i, j]*d_i[q, i] for i in range(m)])
        # step 5
        for q in range(1, n+1):
            for i in range(m):
                if self.split_ratio_matrix[i, q-1] == 0:
                    d_i[q, i] = d_i[q-1, i]
                else:
                    d_i[q, i] = d_i[q-1, i] * min(1, supplies[q-1]/d_j[q-1, q-1])
            for j in range(n):
                d_j[q, j] = sum([self.split_ratio_matrix[i, j]*d_i[q, i] for i in range(m)])
        # step 6
        for i, link in enumerate(self.incoming_links):
            link.downstream_flow = d_i[-1, i]
        # step 7
        for j, link in enumerate(self.outgoing_links):
            link.upstream_flow = sum([self.split_ratio_matrix[i, j] * d_i[-1, i] for i in range(m)])

        # TODO: REMOVE THE FOLLOWING DEBUG LINES ONCE THIS FUNCTION HAS BEEN TESTED
        inflows = sum([link.downstream_flow for link in self.incoming_links])
        outflows = sum([link.upstream_flow for link in self.outgoing_links])
        if abs(inflows - outflows) > EPS:
            raise SystemError("Singularity detected in node! (net flow: " + str(inflows-outflows) + ")")

    def plot(self, ax=None, exaggeration=1, **kwargs):
        if ax is None:
            ax = plt.gca()  # type: plt.Axes
        artists = []
        if self.radius > 0:
            artists.append(CircleDataUnits(self.pos, radius=exaggeration*self.radius, color="black", **kwargs))
            ax.add_patch(artists[-1])
        return artists


class SourceNode(Node):
    def __init__(self, pos, inflow, *, id=None, radius=1):
        super().__init__(pos, id=id, radius=radius)
        self.inflow = inflow

    def compute_flows(self):
        if len(self.incoming_links) > 0:
            raise UserWarning("Source nodes are not allowed to have incoming links.")
        if len(self.outgoing_links) != 1:
            raise UserWarning("Source nodes must have exactly one outgoing link.")
        self.outgoing_links[0].upstream_flow = self.inflow

    def plot(self, ax=None, exaggeration=1, **kwargs):
        if ax is None:
            ax = plt.gca()  # type: plt.Axes
        artists = list()
        x, y, r = *self.pos, self.radius*exaggeration
        s = 0.5
        artists.append(CircleDataUnits(self.pos, radius=r, fc=(0, 0, 0, 0), ec="black", lw=0.2*r, **kwargs))
        ax.add_patch(artists[-1])
        artists.append(LineDataUnits([x-s*r, x+s*r], [y-s*r, y+s*r], solid_capstyle="butt", linewidth=0.2*r, color="black"))
        ax.add_line(artists[-1])
        artists.append(LineDataUnits([x-s*r, x+s*r], [y+s*r, y-s*r], solid_capstyle="butt", linewidth=0.2*r, color="black"))
        ax.add_line(artists[-1])
        return artists


class SinkNode(Node):
    def __init__(self, pos, *, id=None, radius=1):
        super().__init__(pos, id=id, radius=radius)

    def compute_flows(self):
        if len(self.outgoing_links) > 0:
            raise UserWarning("Sink nodes are not allowed to have outgoing links.")
        if len(self.incoming_links) != 1:
            raise UserWarning("Sink nodes must have exactly one incoming link.")
        link = self.incoming_links[0]
        link.downstream_flow = link.free_flow_speed*link.density*(min(1, link.flow_capacity/(link.free_flow_speed*link.density)) if link.density > 0 else 1)

    def plot(self, ax=None, exaggeration=1, **kwargs):
        if ax is None:
            ax = plt.gca()  # type: plt.Axes
        r = self.radius * exaggeration
        artists = list()
        artists.append(CircleDataUnits(self.pos, radius=r, fc=(0, 0, 0, 0), ec="black", lw=0.2*r, **kwargs))
        ax.add_patch(artists[-1])
        artists.append(CircleDataUnits(self.pos, radius=r*0.3, fc="black", lw=0, **kwargs))
        ax.add_patch(artists[-1])
        return artists


class Link:
    def __init__(self, from_node, to_node, fundamental_diagram, density=0, *, id=None):
        self.fundamental_diagram = fundamental_diagram  # type: FundamentalDiagram
        self.density = density
        self.from_node = from_node
        self.from_node.outgoing_links.append(self)
        self.to_node = to_node
        self.to_node.incoming_links.append(self)
        self.id = id if id is not None else str(self.from_node.id) + "->" + str(self.to_node.id)
        self._vec = self.to_node.pos - self.from_node.pos
        self.heading = np.arctan2(self._vec[1], self._vec[0])
        self.length_m = np.linalg.norm(self._vec)  # in meters
        self._unit_vector = self._vec / self.length_m
        self._upstream_flow = None
        self._downstream_flow = None

    @property
    def length_km(self):
        return self.length_m / 1000

    @property
    def direction(self):
        return self._unit_vector

    @property
    def flow_capacity(self):
        return self.fundamental_diagram.flow_capacity

    @property
    def critical_density(self):
        return self.fundamental_diagram.critical_density

    @property
    def free_flow_speed(self):
        return self.fundamental_diagram.free_flow_speed

    @property
    def congestion_wave_speed(self):
        return self.fundamental_diagram.congestion_wave_speed

    @property
    def jam_density(self):
        return self.fundamental_diagram.jam_density

    @property
    def upstream_flow(self):
        return self._upstream_flow

    @upstream_flow.setter
    def upstream_flow(self, flow):
        self._upstream_flow = flow

    @property
    def downstream_flow(self):
        return self._downstream_flow

    @downstream_flow.setter
    def downstream_flow(self, flow):
        self._downstream_flow = flow

    @property
    def flow(self):
        return self.fundamental_diagram.flow_at_density(self.density)

    @property
    def speed(self):
        return self.fundamental_diagram.speed_at_density(self.density)

    @property
    def input_supply(self):
        return min(self.flow_capacity, self.congestion_wave_speed * (self.jam_density - self.density))

    def set_outgoing_split_ratios(self, ratios):
        """
        Set relevant entries in to_node's split ratio matrix given split ratios leaving this edge.
        Ratios should be defined from left to right w.r.t. this link's direction. Ratios are auto-normalized.
        """
        self.to_node.set_link_outgoing_split_ratios(self, ratios)

    def update_state(self, dt):
        """
        Update the state of the link.

        :param dt: time step size (hours)
        :type dt: float
        :return: None
        """
        self.density = self.density + (dt / self.length_km) * (self.upstream_flow - self.downstream_flow)

    def plot(self, ax=None, exaggeration=1, **kwargs):
        if ax is None:
            ax = plt.gca()  # type: plt.gca()
        start = self.from_node.pos + exaggeration*self.from_node.radius*self._unit_vector
        delta = (self.length_m - exaggeration * (self.from_node.radius + self.to_node.radius)) * self._unit_vector
        artist = Arrow(*start, *delta, **{"width": exaggeration*self.flow_capacity*3.2/1800, **kwargs})
        ax.add_patch(artist)
        return [artist]


class BottleneckLink(Link):
    """A link which has a user-defined input supply."""
    def __init__(self, input_supply, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_supply = input_supply

    @property
    def input_supply(self):
        return self._input_supply

    @input_supply.setter
    def input_supply(self, val):
        self._input_supply = val


class Network:
    def __init__(self, nodes=None, links=None):
        self._nodes = [] if nodes is None else nodes
        self._links = [] if links is None else links
        self._mappable = None  # type: plt.cm.ScalarMappable
        self._max_dt = min(l.length_km / l.free_flow_speed for l in links)  # max value that the time step may take

    @classmethod
    def from_yaml(cls, file):
        with open(file) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        # load any named fundamental diagrams
        fundamental_diagrams = cfg.get("fundamental_diagrams", {})
        default_fd_params = fundamental_diagrams.get("default", {})
        # load nodes
        if type(cfg["nodes"]) == dict:
            nodes = cfg["nodes"]
        elif type(cfg["nodes"]) == list:
            nodes = {n.get("id", i) if type(i) == dict else i: n for i, n in enumerate(cfg["nodes"])}
        else:
            raise ValueError("Invalid network file format. Nodes could not be parsed.")
        for nid, node in nodes.items():
            if type(node) == list:
                node = {"pos": node}
            if node.get("sink", False):
                nodes[nid] = SinkNode(node["pos"], id=nid)
            elif node.get("source", False):
                nodes[nid] = SourceNode(node["pos"], node.get("inflow", 0), id=nid)
            else:
                nodes[nid] = Node(node["pos"], id=nid)
        # load links
        if type(cfg["links"]) == dict:
            links = cfg["links"]
        elif type(cfg["links"]) == list:
            links = {l.get("id", i) if type(i) == dict else i: l for i, l in enumerate(cfg["links"])}
        else:
            raise ValueError("Invalid network file format. Links could not be parsed.")
        split_ratios = {}
        for lid, link in links.items():
            if type(link) == list:
                link = {"nodes": link}
            _id = link.pop("id", None)
            from_node, to_node = [nodes[n] for n in link.pop("nodes")]
            density = link.pop("density", 0)
            _ratios = link.pop("split_ratios", None)
            _fd = link.pop("fundamental_diagram", None)
            _fd_params = fundamental_diagrams.get(_fd, {})
            if "input_supply" in link:
                _input_supply = link.pop("input_supply")
                links[lid] = BottleneckLink(_input_supply, from_node, to_node,
                                            FundamentalDiagram(**{**default_fd_params, **_fd_params, **link}),
                                            density=density, id=_id)
            else:
                links[lid] = Link(from_node, to_node,
                                  FundamentalDiagram(**{**default_fd_params, **_fd_params, **link}),
                                  density=density, id=_id)
            if _ratios is not None:
                split_ratios[links[lid]] = _ratios
        for link, ratios in split_ratios.items():
            link.set_outgoing_split_ratios(ratios)
        return cls(nodes=nodes.values(), links=links.values())

    def get_node_by_id(self, id):
        for node in self._nodes:
            if node.id == id:
                return node
        raise IndexError("No node with id " + str(id) + " found in network.")

    def get_link_by_id(self, id):
        for link in self._links:
            if link.id == id:
                return link
        raise IndexError("No node with id " + str(id) + " found in network.")

    def insert_node(self, node):
        self._nodes.append(node)

    def insert_link(self, link):
        self._links.append(link)

    def step(self, dt):
        """
        Update the network state by the specified time step.

        :param dt: time step size (hours)
        :return: None
        """
        if dt > self._max_dt:
            raise UserWarning("Passed time step " + str(dt) + " greater than recommended maximum: " + str(self._max_dt))
        for node in self._nodes:
            node.compute_flows()
        for link in self._links:
            link.update_state(dt)

    def get_records(self):
        link_records = [{"link_id": link.id,
                         "density": link.density,
                         "flow": link.flow,
                         "speed": link.speed} for link in self._links]
        node_records = [{"node_id": node.id,
                         "split_ratios": node.split_ratio_matrix} for node in self._nodes]
        return link_records + node_records

    @property
    def colorbar_mappable(self):
        if self._mappable is not None:
            return self._mappable
        max_flow = max(*[link.flow_capacity for link in self._links])
        cmap = plt.get_cmap("RdYlGn")
        self._mappable = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=max_flow))
        return self._mappable

    def plot_colorbar(self, ax=None):
        if ax is None:
            ax = plt.gca()
        cb = plt.colorbar(self.colorbar_mappable, ax=ax)
        cb.set_label("Flow (veh/h)")

    def plot(self, ax=None, exaggeration=1):
        if ax is None:
            ax = plt.gca()  # type: plt.Axes
        artists = []
        for node in self._nodes:
            artists += node.plot(ax, exaggeration=exaggeration)
        for link in self._links:
            artists += link.plot(ax, exaggeration=exaggeration, color=self.colorbar_mappable.to_rgba(link.flow))
        ax.set_aspect("equal")
        ax.autoscale_view()
        return artists


class Simulation:
    def __init__(self, net, start_time=0, end_time=24, step_size=0.25, scenario_file=None):
        """
        Initialize a Simulation object with a network and optional scenario file.

        :param net: the Network object to use for the simulation
        :type net: Network
        :param start_time: start time for the simulation (hours)
        :type start_time: float
        :param end_time: end time for the simulation (hours)
        :type end_time: float
        :param step_size: step size for the simulation (hours)
        :type step_size: float
        :param scenario_file: (optional) path to a scenario file
        :type scenario_file: str
        """
        self.net = net
        self._start_time = start_time
        self._end_time = end_time
        self._time = start_time
        self._step_size = step_size
        self._all_steps = np.arange(start_time, end_time, step_size)
        self._step_iterator = iter(self._all_steps)
        self._records = []
        self._dynamic_inflows = EventManager()
        self._dynamic_split_ratios = EventManager()
        if scenario_file is not None:
            self.load_scenario_from_file(scenario_file)

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def step_size(self):
        return self._step_size

    @property
    def time(self):
        return self._time

    @property
    def time_steps(self):
        return np.arange(self.start_time, self.end_time, self.step_size)

    @property
    def records(self):
        return self._records

    def load_scenario_from_file(self, file):
        with open(file) as f:
            scenario = yaml.load(f, Loader=yaml.Loader)
        inflows = scenario.get("inflows", [])
        split_ratios = scenario.get("split_ratios", [])
        for inflow in inflows:
            start_time = inflow.pop("start_time")
            end_time = inflow.pop("end_time", np.inf)
            end_time = end_time if end_time is not None else np.inf
            self._dynamic_inflows.add(inflow, start_time=start_time, end_time=end_time)
        for splits in split_ratios:
            start_time = splits.pop("start_time")
            if "end_time" in splits:
                raise NotImplementedError("Specification of end_time for split ratios not supported.")
            self._dynamic_split_ratios.add(splits, start_time=start_time)

    def _update_dynamic_flows(self):
        newly_active_inflows, newly_inactive_inflows = self._dynamic_inflows.get_newly_active_and_inactive(self.time)
        for flow in newly_active_inflows:
            self.net.get_node_by_id(flow["node"]).inflow += flow["flow"]
        for flow in newly_inactive_inflows:
            self.net.get_node_by_id(flow["node"]).inflow -= flow["flow"]

    def _update_dynamic_split_ratios(self):
        newly_active, _ = self._dynamic_split_ratios.get_newly_active_and_inactive(self.time)
        for splits in newly_active:
            if "node" in splits and "link" not in splits:
                self.net.get_node_by_id(splits["node"]).split_ratio_matrix = splits["split_ratios"]
            elif "link" in splits and "node" not in splits:
                self.net.get_link_by_id(splits["link"]).set_outgoing_split_ratios(splits["split_ratios"])
            else:
                raise UserWarning("Invalid definition of split ratios in scenario file.")

    def step(self):
        self._time = next(self._step_iterator)
        self._update_dynamic_flows()
        self._update_dynamic_split_ratios()
        self.net.step(self.step_size)
        self._records += [{"time": self.time, **record} for record in self.net.get_records()]

    def __next__(self):
        return self.step()

    def plot(self, ax=None, timestamp_loc="upper left", exaggeration=1, **kwargs):
        if ax is None:
            ax = plt.gca()  # type: plt.Axes
        artists = []
        artists += self.net.plot(ax, exaggeration=exaggeration, **kwargs)
        if timestamp_loc is not None:
            h, m, s = int(self.time), int((self.time*60) % 60)%60, round((self.time*3600) % 60)%60
            artists.append(AnchoredText("{:02.0f}:{:02.0f}:{:02.0f}".format(h, m, s), loc=timestamp_loc))
            ax.add_artist(artists[-1])
        return artists


if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation
    net = Network.from_yaml("../test_net.yaml")

    def anim(t, ax, sim):
        artists = sim.plot(ax, exaggeration=1000)
        sim.step()
        return artists

    fig, ax = plt.subplots()
    net.plot_colorbar(ax)
    sim = Simulation(net, start_time=0, end_time=24, step_size=1/30)
    sim.load_scenario_from_file("../test_scenario.yaml")
    a = FuncAnimation(fig, anim, fargs=(ax, sim), frames=sim.time_steps, blit=True, repeat=False, interval=100)

    # net.plot()
    plt.show()
