"""
Classes and utilities for defining a network at a more abstract level (roads and intersections, traffic signals, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from abc import abstractmethod
from CellTransmissionModel.ctm import Network, Node, SourceNode, SinkNode, Link, FundamentalDiagram
from CellTransmissionModel._Util import signed_angle_from_three_points
from copy import copy


class AbstractRoad:
    def __init__(self, alignment, oneway=False, fundamental_diagram_a=None, fundamental_diagram_b=None, max_link_length=10, *, id=None):
        """
        Initialize an AbstractRoad object, which can be used to easily generate consecutive CTM links.

        :param alignment: list or np.ndarray of n points making up alignment
        :param oneway: whether or not the road is one-way
        """
        self.id = id
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

    def bake(self):
        if self.from_intersection is None or self.to_intersection is None:
            raise UserWarning("All AbstractRoad objects must start and end with either an intersection or an AbstractSourceSink object.")
        ls = self._linestring
        ds = np.linspace(0, ls.length, int(np.ceil(ls.length/self.max_link_length)+1))
        pts = [ls.interpolate(d) for d in ds]
        self._nodes = [Node(pt, id=str(self.id)+"."+str(i)) for i, pt in enumerate(pts)]
        self._links = []
        for i in range(len(self.nodes)-1):
            self._links.append(Link(from_node=self._nodes[i], to_node=self._nodes[i+1],
                                    fundamental_diagram=copy(self._fundamental_diagram_template_a)))
            if not self.oneway:
                self._links.append(Link(from_node=self._nodes[i+1], to_node=self._nodes[i],
                                        fundamental_diagram=copy(self._fundamental_diagram_template_b)))
                self._twin_links[self._links[-2]] = self._links[-1]
                self._twin_links[self._links[-1]] = self._links[-2]
        # set split ratios (no u-turns)
        for from_link, to_link in self._twin_links.items():
            from_link.set_outgoing_split_ratios_by_reference({to_link: 0})

    @property
    def nodes(self):
        return self._nodes

    @property
    def links(self):
        return self._links


class _AbstractJunction:
    def __init__(self, location, *, id=None):
        self.location = location if isinstance(location, np.ndarray) else np.array(location)
        self.id = id
        self._connecting_roads = []
        self._connecting_roads_ends = []  # 0 if the beginning of the road connects, -1 if the end of the road connects
        self._nodes = []
        self._links = []

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


class AbstractSourceSink(_AbstractJunction):
    def __init__(self, location, inflow=0, *, fundamental_diagram=None, id=None):
        super().__init__(location, id=id)
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
    def __init__(self, location, radius=8, *, id=None):
        super().__init__(location, id=id)
        self.radius = radius
        self._lsr_map = dict()

    def bake(self):
        incoming_roads, outgoing_roads = self.incoming_roads, self.outgoing_roads
        if len(incoming_roads) > 4 or len(outgoing_roads) > 4:
            raise NotImplementedError("AbstractIntersection only supports up to 4-way intersections.")
        elif len(incoming_roads) == 0 or len(outgoing_roads) == 0:
            raise UserWarning("AbstractIntersection must have at least one incoming and one outgoing road.")
        for incoming_road in incoming_roads:
            road_node = self._road_node_incoming(incoming_road)
            _vec = self.location - road_node.pos
            _perp = np.array([-_vec[1], _vec[0]])
            node_ls = Node(road_node.pos + 0.4*_vec, id=str(self.id)+"_"+str(incoming_road.id)+"_ls")
            node_l = Node(road_node.pos + 0.8*_vec, id=str(self.id)+"_"+str(incoming_road.id)+"_l")
            node_r = Node(road_node.pos + 0.4*_vec-0.4*_perp, id=str(self.id)+"_"+str(incoming_road.id)+"_r")
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
            if "l" in self._lsr_map[incoming_road] or "s" in self._lsr_map[incoming_road]:
                self.nodes.append(node_ls)
                self.links.append(Link(from_node=road_node, to_node=node_ls, fundamental_diagram=FundamentalDiagram()))
            if "l" in self._lsr_map[incoming_road]:
                self.nodes.append(node_l)
                self.links.append(Link(from_node=node_ls, to_node=node_l, fundamental_diagram=FundamentalDiagram()))
                to_node = self._road_node_outgoing(self._lsr_map[incoming_road]["l"])
                self.links.append(Link(from_node=node_l, to_node=to_node, fundamental_diagram=FundamentalDiagram()))
            if "s" in self._lsr_map[incoming_road]:
                to_node = self._road_node_outgoing(self._lsr_map[incoming_road]["s"])
                self.links.append(Link(from_node=node_ls, to_node=to_node, fundamental_diagram=FundamentalDiagram()))
            if "r" in self._lsr_map[incoming_road]:
                self.nodes.append(node_r)
                self.links.append(Link(from_node=road_node, to_node=node_r, fundamental_diagram=FundamentalDiagram()))
                to_node = self._road_node_outgoing(self._lsr_map[incoming_road]["r"])
                self.links.append(Link(from_node=node_r, to_node=to_node, fundamental_diagram=FundamentalDiagram()))
        # disable internal u-turns
        for road in self.incoming_roads:
            road_node = self._road_node_incoming(road)
            for inc_link in road_node.incoming_links:
                for out_link in road_node.outgoing_links:
                    if inc_link in self.links and out_link in self.links:
                        inc_link.set_outgoing_split_ratios_by_reference({out_link: 0})


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


if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation
    from CellTransmissionModel.ctm import Simulation
    sourcesink1 = AbstractSourceSink((-20, 0), 1000)
    sourcesink2 = AbstractSourceSink((170, 50), 0)
    sourcesink3 = AbstractSourceSink((100, 50), 0)
    rd1 = AbstractRoad([(0, 0), (85, 0)], id="rd1")
    intersection = AbstractIntersection((100, 0), radius=15)
    rd2 = AbstractRoad([(115, 0), (125, 0), (150, 50)], id="rd2")
    rd3 = AbstractRoad([(100, 15), (100, 35)], id="rd3")
    rd1.from_intersection = sourcesink1
    rd1.to_intersection = intersection
    rd2.from_intersection = intersection
    rd2.to_intersection = sourcesink2
    rd3.from_intersection = intersection
    rd3.to_intersection = sourcesink3
    anet = AbstractNetwork(roads=[rd1, rd2, rd3], intersections=[sourcesink1, intersection, sourcesink2, sourcesink3])
    sim = Simulation(anet.net, step_size=0.0001)

    def anim(t, ax, sim):
        artists = sim.plot(ax, exaggeration=1, half_arrows=True)
        sim.step()
        return artists

    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    a = FuncAnimation(fig, anim, fargs=(ax, sim), blit=True, interval=100)
    # anet.net.plot()
    plt.show()
