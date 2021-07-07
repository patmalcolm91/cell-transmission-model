"""
Classes and utilities for defining a network at a more abstract level (roads and intersections, traffic signals, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from shapely.geometry import LineString, Point
from matplotlib.offsetbox import AnchoredText
from abc import abstractmethod
from typing import List
from CellTransmissionModel.ctm import Network, Node, SourceNode, SinkNode, IndependentDivergeNode, Link, FundamentalDiagram
from CellTransmissionModel._Util import signed_angle_from_three_points
import re
from copy import copy
import warnings


EPS = 1E-6  # epsilon (threshold for small values)


class AbstractRoad:
    def __init__(self, alignment, oneway=False, fundamental_diagram_a=None, fundamental_diagram_b=None,
                 max_link_length=10, *, id=None, node_class=Node, link_class=Link, use_intersection_node_class_at_end=True):
        """
        Initialize an AbstractRoad object, which can be used to easily generate consecutive CTM links.

        :param alignment: list or np.ndarray of n points making up alignment
        :param oneway: whether or not the road is one-way
        """
        self.id = id
        self.node_class = node_class
        self.link_class = link_class
        self.use_intersection_node_class_at_end = use_intersection_node_class_at_end
        self._fundamental_diagram_template_a = fundamental_diagram_a if fundamental_diagram_a is not None else FundamentalDiagram()
        self._fundamental_diagram_template_b = fundamental_diagram_b if fundamental_diagram_b is not None else FundamentalDiagram()
        self.alignment = alignment if isinstance(alignment, np.ndarray) else np.array(alignment)
        self._linestring = LineString(self.alignment)
        self._twin_links = {}  # dict mapping links to their "twins" (same portion of alignment, but opposite direction)
        self.oneway = oneway
        self._nodes = []  # type: list[Node]
        self._links = []  # type: list[Link]
        self.max_link_length = max_link_length
        self._from_intersection = None
        self._to_intersection = None

    def __repr__(self):
        return "AbstractRoad(id="+str(self.id)+")"

    @property
    def from_intersection(self):
        return self._from_intersection

    @from_intersection.setter
    def from_intersection(self, intersection):
        if self._from_intersection is not None:
            raise UserWarning("Trying to overwrite AbstractRoad's 'from_intersection', but it can only be set once.")
        self._from_intersection = intersection
        self._from_intersection.connect_road_outgoing(self)

    @property
    def to_intersection(self):
        return self._to_intersection

    @to_intersection.setter
    def to_intersection(self, intersection):
        if self._to_intersection is not None:
            raise UserWarning("Trying to overwrite AbstractRoad's 'to_intersection', but it can only be set once.")
        self._to_intersection = intersection
        self._to_intersection.connect_road_incoming(self)

    @classmethod
    def between_intersections(cls, from_intersection, to_intersection, oneway=False, max_link_length=10):
        from_pt, to_pt = from_intersection.location, to_intersection.location
        road = cls([from_pt, to_pt], oneway=oneway, max_link_length=max_link_length)
        road.from_intersection, road.to_intersection = from_intersection, to_intersection
        return road

    @property
    def n_links_per_direction(self):
        """Number of links in each travel direction into which the road is divided."""
        if len(self._links) == 0:
            raise UserWarning("Can't get number of links per direction on road before it has been baked.")
        return len(self._links) if self.oneway else len(self._links)//2

    def get_last_link_to_intersection(self, intersection):
        """
        Get the last link of the road from the perspective of the given intersection, i.e. the link that flows directly
        into the specified intersection.

        :param intersection: the reference intersection
        :type intersection: _AbstractJunction
        :return: Link object
        """
        if intersection == self._to_intersection:
            if self.oneway:
                return self._links[self.n_links_per_direction - 1]
            else:
                return self._links[(self.n_links_per_direction - 1)*2]
        elif intersection == self._from_intersection:
            if self.oneway:
                raise UserWarning("Road " + str(self.id) + " does not have an outgoing link to intersection" + str(intersection.id))
            return self._links[1]
        raise UserWarning("Road " + str(self.id) + " does not connect to intersection " + str(intersection.id))

    def get_first_link_from_intersection(self, intersection):
        """
        Get the first link of the road from the perspective of the given intersection, i.e. the link that flows directly
        out from the specified intersection.

        :param intersection: the reference intersection
        :type intersection: _AbstractJunction
        :return: Link object
        """
        if intersection == self.from_intersection:
            return self._links[0]
        elif intersection == self.to_intersection:
            if self.oneway:
                raise UserWarning("Road " + str(self.id) + " does not have an incoming link to intersection" + str(intersection.id))
            return self._links[self.n_links_per_direction*2 - 1]
        raise UserWarning("Road " + str(self.id) + " does not connect to intersection " + str(intersection.id))

    def bake(self):
        # check if connected
        if self.from_intersection is None or self.to_intersection is None:
            raise UserWarning("All AbstractRoad objects must start and end with either an intersection or an AbstractSourceSink object.")
        # split road geometry into segments
        ls = self._linestring
        if ls.length > self.max_link_length:
            ds = np.linspace(0, ls.length, int(np.ceil(ls.length/self.max_link_length)+1))
            pts = [ls.interpolate(d) for d in ds]
        else:
            pts = [p for p in ls.coords]
        # determine which node classes to use for the internal and external nodes
        _ncls = [self.node_class for _ in pts]
        if self.use_intersection_node_class_at_end:
            _ncls[0] = self.from_intersection.node_class
            _ncls[-1] = self.to_intersection.node_class
        # generate nodes
        self._nodes = [cls(pt, id=str(self.id)+"."+str(i)) for i, (pt, cls) in enumerate(zip(pts, _ncls))]
        # generate links
        self._links = []
        for i in range(len(self.nodes)-1):
            self._links.append(self.link_class(from_node=self._nodes[i], to_node=self._nodes[i+1],
                                               fundamental_diagram=copy(self._fundamental_diagram_template_a)))
            if not self.oneway:
                self._links.append(self.link_class(from_node=self._nodes[i+1], to_node=self._nodes[i],
                                                   fundamental_diagram=copy(self._fundamental_diagram_template_b)))
                self._twin_links[self._links[-2]] = self._links[-1]
                self._twin_links[self._links[-1]] = self._links[-2]
        # set split ratios (no u-turns)
        for from_link, to_link in self._twin_links.items():
            from_link.set_outgoing_split_ratios_by_reference({to_link: 0})

    def fundamental_diagram_into_intersection(self, intersection):
        """Get the fundamental diagram for the direction of the road leading into the specified intersection."""
        if self.from_intersection == intersection:
            if self.oneway:
                raise KeyError("Road "+str(self.id)+" does not connect to intersection "+str(intersection.id))
            return self._fundamental_diagram_template_b
        elif self.to_intersection == intersection:
            return self._fundamental_diagram_template_a
        else:
            KeyError("Road "+str(self.id)+" does not connect to intersection "+str(intersection.id))

    @property
    def nodes(self):
        return self._nodes

    @property
    def links(self):
        return self._links


class _AbstractJunction:
    def __init__(self, location, *, id=None, node_class=Node, link_class=Link):
        self.location = location if isinstance(location, np.ndarray) else np.array(location)
        self.id = id
        self.node_class = node_class
        self.link_class = link_class
        self._connecting_roads = []
        self._connecting_roads_ends = []  # 0 if the beginning of the road connects, -1 if the end of the road connects
        self._nodes = []
        self._links = []

    def __repr__(self):
        return self.__class__.__name__+"(id=" + str(self.id) + ")"

    @abstractmethod
    def bake(self):
        pass

    @property
    def nodes(self):
        return self._nodes

    @property
    def links(self):
        return self._links

    @property
    def incoming_roads(self):
        return [road for road, i in zip(self._connecting_roads, self._connecting_roads_ends) if i == -1 or not road.oneway]

    @property
    def outgoing_roads(self):
        return [road for road, i in zip(self._connecting_roads, self._connecting_roads_ends) if i == 0 or not road.oneway]

    def connect_road_incoming(self, road):
        self._connecting_roads.append(road)
        self._connecting_roads_ends.append(-1)

    def connect_road_outgoing(self, road):
        self._connecting_roads.append(road)
        self._connecting_roads_ends.append(0)

    def _road_node_incoming(self, road):
        idx = self._connecting_roads.index(road)
        end = self._connecting_roads_ends[idx]
        if end == 0 and road.oneway:
            return None
        return road.nodes[end]

    def _road_node_outgoing(self, road):
        idx = self._connecting_roads.index(road)
        end = self._connecting_roads_ends[idx]
        if end == -1 and road.oneway:
            return None
        return road.nodes[end]

    def step(self, dt):
        """Update the junction by one time step of size dt."""
        pass


class AbstractSourceSink(_AbstractJunction):
    def __init__(self, location, inflow=0, *, fundamental_diagram=None, id=None, node_class=Node, link_class=Link):
        super().__init__(location, id=id, node_class=node_class, link_class=link_class)
        self.inflow = inflow
        self.fundamental_diagram = fundamental_diagram if fundamental_diagram is not None else FundamentalDiagram()
        self.source_node = SourceNode(self.location, self.inflow, id=str(id)+".source")
        self.sink_node = SinkNode(self.location, id=str(id)+".sink")
        self._nodes = [self.source_node, self.sink_node]

    def bake(self):
        for road in self._connecting_roads:
            incoming_node = self._road_node_incoming(road)
            outgoing_node = self._road_node_outgoing(road)
            if incoming_node is not None:
                self._links.append(Link(from_node=incoming_node, to_node=self.sink_node,
                                        fundamental_diagram=self.fundamental_diagram))
            if outgoing_node is not None:
                self._links.append(Link(from_node=self.source_node, to_node=outgoing_node,
                                        fundamental_diagram=self.fundamental_diagram))
            if not road.oneway:
                self._links[-1].set_outgoing_split_ratios_by_reference({self._links[-2]: 0})  # disable u-turn


class AbstractIntersection(_AbstractJunction):
    def __init__(self, location, radius=8, *, id=None, node_class=Node, link_class=Link):
        super().__init__(location, id=id, node_class=node_class, link_class=link_class)
        self.radius = radius
        self._lsr_map = {}  # {incoming_road: {"l": outgoing_road, "s": outgoing_road, "r": outgoing_road}, ...}
        self._nodes_logic = {}  # {incoming_road: {"ls": node_ls, "l": node_l, "r": node_r}, ...}
        self._links_logic = {}  # {incoming_road: {"r_queue": rq, "ls_queue": lsq, "l_queue": lq, "l_mvmt": lm, "s_mvmt": sm, "r_mvmt": rm},  ...}

    @property
    def nodes(self):
        """All the internal nodes of the intersection as a list."""
        _nodes = []
        for rd_nodes in self._nodes_logic.values():
            for node in rd_nodes.values():
                _nodes.append(node)
        return _nodes

    @property
    def links(self):
        """All the internal links of the intersection as a list."""
        _links = []
        for rd_links in self._links_logic.values():
            for link in rd_links.values():
                _links.append(link)
        return _links

    def bake(self):
        incoming_roads, outgoing_roads = self.incoming_roads, self.outgoing_roads
        if len(incoming_roads) > 4 or len(outgoing_roads) > 4:
            raise NotImplementedError("AbstractIntersection only supports up to 4-way intersections.")
        elif len(incoming_roads) == 0 or len(outgoing_roads) == 0:
            raise UserWarning("AbstractIntersection must have at least one incoming and one outgoing road.")
        for incoming_road in incoming_roads:
            self._nodes_logic[incoming_road] = {}
            self._links_logic[incoming_road] = {}
            road_node = self._road_node_incoming(incoming_road)
            _vec = self.location - road_node.pos
            _perp = np.array([-_vec[1], _vec[0]])
            node_ls = self.node_class(road_node.pos + 0.4*_vec, id=str(self.id)+"_"+str(incoming_road.id)+"_ls")
            node_l = self.node_class(road_node.pos + 0.8*_vec, id=str(self.id)+"_"+str(incoming_road.id)+"_l")
            node_r = self.node_class(road_node.pos + 0.4*_vec-0.4*_perp, id=str(self.id)+"_"+str(incoming_road.id)+"_r")
            # classify connections as left, straight, or right
            self._lsr_map[incoming_road] = dict()
            for outgoing_road in outgoing_roads:
                if outgoing_road is incoming_road:
                    continue
                # TODO: this approach is restrictive. Should instead use sorted angles to determine lsr relationships?
                angle = signed_angle_from_three_points(road_node.pos, self.location, self._road_node_outgoing(outgoing_road).pos)
                if abs(angle) < np.pi/4:
                    if "s" in self._lsr_map[incoming_road]:
                        raise UserWarning("More than one straight movement detected from road " + str(incoming_road.id))
                    self._lsr_map[incoming_road]["s"] = outgoing_road
                elif angle < -np.pi/4:
                    if "r" in self._lsr_map[incoming_road]:
                        raise UserWarning("More than one right movement detected from road " + str(incoming_road.id))
                    self._lsr_map[incoming_road]["r"] = outgoing_road
                else:
                    if "l" in self._lsr_map[incoming_road]:
                        raise UserWarning("More than one left movement detected from road " + str(incoming_road.id))
                    self._lsr_map[incoming_road]["l"] = outgoing_road
            # add links and nodes as necessary
            fd = incoming_road.fundamental_diagram_into_intersection(self)
            if "l" in self._lsr_map[incoming_road] or "s" in self._lsr_map[incoming_road]:
                self._nodes_logic[incoming_road]["ls"] = node_ls
                self._links_logic[incoming_road]["ls_queue"] = self.link_class(from_node=road_node, to_node=node_ls,
                                                                               fundamental_diagram=copy(fd))
            if "l" in self._lsr_map[incoming_road]:
                self._nodes_logic[incoming_road]["l"] = node_l
                self._links_logic[incoming_road]["l_queue"] = self.link_class(from_node=node_ls, to_node=node_l,
                                                                              fundamental_diagram=copy(fd))
                to_node = self._road_node_outgoing(self._lsr_map[incoming_road]["l"])
                self._links_logic[incoming_road]["l_mvmt"] = self.link_class(from_node=node_l, to_node=to_node,
                                                                             fundamental_diagram=copy(fd))
            if "s" in self._lsr_map[incoming_road]:
                to_node = self._road_node_outgoing(self._lsr_map[incoming_road]["s"])
                self._links_logic[incoming_road]["s_mvmt"] = self.link_class(from_node=node_ls, to_node=to_node,
                                                                             fundamental_diagram=copy(fd))
            if "r" in self._lsr_map[incoming_road]:
                self._nodes_logic[incoming_road]["r"] = node_r
                self._links_logic[incoming_road]["r_queue"] = self.link_class(from_node=road_node, to_node=node_r,
                                                                              fundamental_diagram=copy(fd))
                to_node = self._road_node_outgoing(self._lsr_map[incoming_road]["r"])
                self._links_logic[incoming_road]["r_mvmt"] = self.link_class(from_node=node_r, to_node=to_node,
                                                                             fundamental_diagram=copy(fd))
        # disable internal u-turns
        for road in self.incoming_roads:
            road_node = self._road_node_incoming(road)
            for inc_link in road_node.incoming_links:
                for out_link in road_node.outgoing_links:
                    if inc_link in self.links and out_link in self.links:
                        inc_link.set_outgoing_split_ratios_by_reference({out_link: 0})

    def set_turning_ratios(self, incoming_road, left=None, straight=None, right=None):
        """
        Set the turning ratios at the intersection for the specified road. Provided values must sum to 1, and all
        available turning movements for the road must be specified. For example, if from an incoming road, left turns
        and straight movements are possible, then both ``left`` and ``straight`` must be given, even if one of them is
        zero, i.e. ``straight=1, left=0``.

        :param incoming_road: the incoming road for which to set the turning ratios
        :type incoming_road: AbstractRoad
        :param left: proportion of vehicles that should turn left
        :type left: float
        :param straight: proportion of vehicles that continue straight
        :type straight: float
        :param right: proportion of vehicles that should turn right
        :type right: float
        :return: None
        """
        road_node = self._road_node_incoming(incoming_road)  # type: Node
        inc_link = incoming_road.get_last_link_to_intersection(self)
        _nodes, _links = self._nodes_logic[incoming_road], self._links_logic[incoming_road]
        _l = left if left is not None else 0
        _s = straight if straight is not None else 0
        _r = right if right is not None else 0
        # check that arguments are valid
        if abs(1 - (_l + _s + _r)) > EPS:
            raise ValueError("Turning ratios must sum to 1.")
        _l_given, _s_given, _r_given = left is not None, straight is not None, right is not None
        _has_l, _has_s, _has_r = "l_mvmt" in _links, "s_mvmt" in _links, "r_mvmt" in _links
        if _l_given != _has_l or _s_given != _has_s or _r_given != _has_r:
            given_string = "".join([t for t, given in zip("lsr", [_l_given, _s_given, _r_given]) if given])
            has_string = "".join([t for t, has in zip("lsr", [_has_l, _has_s, _has_r]) if has])
            raise ValueError("Given turning ratios (" + given_string + ") do not exactly match available movements (" +
                             has_string+") for road "+str(incoming_road.id)+" into intersection "+str(self.id))
        # set the ratios at the incoming road node
        if _has_r and (_has_l or _has_s):
            road_node.set_split_ratios_by_reference({(inc_link, _links["r_queue"]): _r,
                                                     (inc_link, _links["ls_queue"]): _l+_s})
        # set ratios for right node
        if _has_r:
            _nodes["r"].set_split_ratios_by_reference({(_links["r_queue"], _links["r_mvmt"]): 1})
        # set ratios for left-straight node
        if _has_l or _has_s:
            if not _has_l:
                _nodes["ls"].set_split_ratios_by_reference({(_links["ls_queue"], _links["s_mvmt"]): 1})
            elif not _has_s:
                _nodes["ls"].set_split_ratios_by_reference({(_links["ls_queue"], _links["l_queue"]): 1})
            else:  # i.e. if both left and straight movements are available
                if _l_given:
                    _nodes["ls"].set_split_ratios_by_reference({(_links["ls_queue"], _links["l_queue"]): _l/(_l+_s)})
                if _s_given:
                    _nodes["ls"].set_split_ratios_by_reference({(_links["ls_queue"], _links["s_mvmt"]): _s/(_l+_s)})
        # set ratios for left node
        if _has_l:
            _nodes["l"].set_split_ratios_by_reference({(_links["l_queue"], _links["l_mvmt"]): 1})

    def step(self, dt):
        """Update the intersection by a time step of dt."""
        # TODO: implement reduced left turn capacities due to oncoming traffic
        warnings.warn("Note: reduced capacity for conflicting intersection movements not yet implemented.")


class SignalPhase:
    """
    Helper class for storing a signal phase, consisting of incoming roads and corresponding left, straight, and right
    movements. A movement is given as a (incoming_road, directions) tuple, where directions is a string containing the
    directions allowed during the phase (e.g. "l" for left, "lsr" for left-straight-right, "sr" for straight-right, and
    so on; the letters must be given in this order, i.e. "sl" is not allowed).
    """
    _DIRECTIONS_REGEX = re.compile(r"^[l]?[s]?[r]?$")  # regex for matching a valid movement directions string

    def __init__(self, duration, *movements):
        """
        Initialize a Signal Phase object.

        :param duration: duration of signal phase (s)
        :param movements: any number of (incoming_road, directions) tuples. See class documentation for details.
        """
        self.duration = duration
        self.movements = {}
        for incoming_road, directions in movements:
            self.add_movement(incoming_road, directions)

    def add_movement(self, incoming_road, directions):
        """
        Add movements from the incoming_road in the specified directions to a Signal Phase. See class documentation.

        :param incoming_road: the incoming road for the movement to be added
        :type incoming_road: AbstractRoad
        :param directions: string specifying the directions which are to be allowed from incoming_road during this phase
        :type directions: str
        :return:
        """
        if self._DIRECTIONS_REGEX.match(directions) is None:
            raise ValueError("Signal phase movement directions must consist only of l, s, and r, in that order.")
        if incoming_road not in self.movements:
            self.movements[incoming_road] = directions
        else:
            raise UserWarning("Road " + str(incoming_road.id) + " already specified in signal phase.")


class AbstractSignalizedIntersection(AbstractIntersection):
    def __init__(self, location, radius=8, *, id=None, phases: List[SignalPhase] = None,
                 node_class=IndependentDivergeNode, link_class=Link):
        super().__init__(location=location, radius=radius, id=id, node_class=node_class, link_class=link_class)
        self.phases = [] if phases is None else phases  # type: list[SignalPhase]
        self._current_phase_index = 0
        self._time_since_phase_change = 0  # in seconds
        self._update()

    @property
    def cycle_time(self):
        """Total cycle time for the signal plan."""
        return sum([phase.duration for phase in self.phases])

    @property
    def current_phase(self):
        """SignalPhase object corresponding to the currently active phase."""
        return self.phases[self._current_phase_index]

    @property
    def n_phases(self):
        """Number of phases in the signal plan."""
        return len(self.phases)

    def next_phase(self):
        """Move to the next phase in the signal plan."""
        self._current_phase_index = (self._current_phase_index + 1) % self.n_phases
        self._time_since_phase_change = 0
        self._update()

    def _update(self):
        """Update link capacities based on the current phase."""
        # skip update if intersection isn't yet fully built
        if self.n_phases == 0 or len(self.links) == 0:
            return
        # Disable all movements
        for incoming_road in self._links_logic:
            for link_label in ["ls_queue", "l_queue", "s_mvmt", "r_queue"]:
                if link_label in self._links_logic[incoming_road]:
                    self._links_logic[incoming_road][link_label].enabled = False
        # Enable the movements that are part of the current phase
        for incoming_road, directions in self.current_phase.movements.items():
            if "l" in directions or "s" in directions:
                self._links_logic[incoming_road]["ls_queue"].enabled = True
                if "l" in directions:
                    self._links_logic[incoming_road]["l_queue"].enabled = True
                if "s" in directions:
                    self._links_logic[incoming_road]["s_mvmt"].enabled = True
            if "r" in directions:
                self._links_logic[incoming_road]["r_queue"].enabled = True

    def step(self, dt):
        """Step the signalized intersection by time step dt (hours)."""
        if self._time_since_phase_change >= self.current_phase.duration:
            self.next_phase()
        self._time_since_phase_change += dt*3600

    def plot_current_phase_diagram(self, ax=None, cx=0, cy=0, scale=1, arrow_scale=1, autoscale_view=False):
        # TODO: make work with blitting
        warnings.warn("Note: plot_current_phase_diagram() does not currently work properly with blitting enabled.")
        artists = []
        if ax is None:
            ax = plt.gca()  # type: plt.Axes
        arrow_kwargs = dict(lw=4*arrow_scale, head_width=0.2*arrow_scale, head_length=0.2*arrow_scale, color="black")
        line_kwargs = dict(lw=4*arrow_scale, color="black")
        artists.append(Circle((cx, cy), 1.1*scale, edgecolor="black", linewidth=2*arrow_scale, facecolor="white"))
        ax.add_patch(artists[-1])
        for rd, dirs in self.current_phase.movements.items():
            direction_vector = rd.get_last_link_to_intersection(self).direction
            perp_vector = np.array([direction_vector[1], -direction_vector[0]])
            pt0 = np.array([cx, cy]) - scale*direction_vector
            if "s" in dirs:
                artists.append(plt.arrow(*pt0, *(0.7*scale*direction_vector), **arrow_kwargs))
            if "r" in dirs:
                rpt0 = pt0 + 0.2*scale*perp_vector
                bend_pt = rpt0 + 0.6*scale*direction_vector
                artists.append(Line2D(*zip(rpt0, bend_pt), **line_kwargs))
                ax.add_artist(artists[-1])
                artists.append(plt.arrow(*bend_pt, *(0.4*scale*perp_vector), **arrow_kwargs))
            if "l" in dirs:
                lpt0 = pt0 - 0.2*scale*perp_vector
                bend_pt = lpt0 + 0.6*scale*direction_vector
                artists.append(Line2D(*zip(lpt0, bend_pt), **line_kwargs))
                ax.add_artist(artists[-1])
                artists.append(plt.arrow(*bend_pt, *(-0.4*scale*perp_vector), **arrow_kwargs))
        if autoscale_view:
            ax.set_aspect("equal")
            ax.autoscale_view()
        return artists


class AbstractNetwork:
    def __init__(self, roads, intersections):
        self.roads = roads  # type: list[AbstractRoad]
        self.intersections = intersections  # type: list[AbstractIntersection]
        self._net = None
        self._bake()

    def _bake(self):
        nodes, links = [], []
        for road in self.roads:
            road.bake()
            nodes += road.nodes
            links += road.links
        for intersection in self.intersections:
            intersection.bake()
            nodes += intersection.nodes
            links += intersection.links
        self._net = Network(nodes=nodes, links=links)

    @property
    def net(self):
        if self._net is None:
            raise Exception("Net not yet compiled.")
        return self._net

    def step(self, dt):
        for intersection in self.intersections:
            intersection.step(dt)
        self.net.step(dt)


class AbstractSimulation:
    def __init__(self, abstract_network, start_time=0, end_time=24, step_size=0.25, scenario_file=None):
        self.abstract_network = abstract_network  # type: AbstractNetwork
        self.start_time = start_time
        self.time = start_time
        self.end_time = end_time
        self.step_size = step_size
        if scenario_file is not None:
            raise NotImplementedError("Scenario file for AbstractSimulation not yet implemented.")

    def step(self):
        self.time += self.step_size
        self.abstract_network.step(self.step_size)

    def plot(self, ax=None, timestamp_loc="upper left", exaggeration=1, half_arrows=True, **kwargs):
        ax = ax if ax is not None else plt.gca()
        artists = []
        artists += self.abstract_network.net.plot(ax, exaggeration=exaggeration, half_arrows=half_arrows, **kwargs)
        if timestamp_loc is not None:
            h, m, s = int(self.time), int((self.time*60) % 60)%60, round((self.time*3600) % 60)%60
            artists.append(AnchoredText("{:02.0f}:{:02.0f}:{:02.0f}".format(h, m, s), loc=timestamp_loc))
            ax.add_artist(artists[-1])
        return artists


if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation
    from CellTransmissionModel.ctm import Simulation
    fd = FundamentalDiagram(flow_capacity=1800, critical_density=36)
    sourcesink1 = AbstractSourceSink((-20, 0), 1000)
    sourcesink2 = AbstractSourceSink((170, 50), 0)
    sourcesink3 = AbstractSourceSink((100, 50), 0)
    rd1 = AbstractRoad([(0, 0), (85, 0)], id="rd1", fundamental_diagram_a=fd, fundamental_diagram_b=fd)
    intersection = AbstractSignalizedIntersection((100, 0), radius=15, phases=[SignalPhase(30, (rd1, "l")),
                                                                               SignalPhase(30, (rd1, "s"))])
    rd2 = AbstractRoad([(115, 0), (125, 0), (150, 50)], id="rd2", fundamental_diagram_a=fd, fundamental_diagram_b=fd)
    rd3 = AbstractRoad([(100, 15), (100, 35)], id="rd3", fundamental_diagram_a=fd, fundamental_diagram_b=fd)
    rd1.from_intersection = sourcesink1
    rd1.to_intersection = intersection
    rd2.from_intersection = intersection
    rd2.to_intersection = sourcesink2
    rd3.from_intersection = intersection
    rd3.to_intersection = sourcesink3
    anet = AbstractNetwork(roads=[rd1, rd2, rd3], intersections=[sourcesink1, intersection, sourcesink2, sourcesink3])
    intersection.set_turning_ratios(rd1, left=0.5, straight=0.5)
    intersection._update()
    asim = AbstractSimulation(anet, step_size=0.0001)

    def anim(t, ax, sim):
        plt.cla()
        artists = sim.plot(ax, exaggeration=1, half_arrows=True)
        artists += intersection.plot_current_phase_diagram(ax, 50, 50, scale=30, arrow_scale=0.6)
        asim.step()
        return artists

    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    a = FuncAnimation(fig, anim, fargs=(ax, asim), blit=False, interval=100)
    asim.abstract_network.net.plot_colorbar()
    # a.save("demo.gif", fps=2.8*3)
    # anet.net.plot()
    plt.show()
