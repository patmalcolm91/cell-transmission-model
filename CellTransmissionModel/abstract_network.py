"""
Classes and utilities for defining a network at a more abstract level (roads and intersections, traffic signals, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from CellTransmissionModel.ctm import Network, Node, Link, FundamentalDiagram
from copy import copy


class AbstractRoad:
    def __init__(self, alignment, width=3.2, oneway=False, fundamental_diagram_a=None, fundamental_diagram_b=None, max_link_length=10, width_split=0.5, *, id=None):
        """
        Initialize an AbstractRoad object, which can be used to easily generate consecutive CTM links.

        :param alignment: list or np.ndarray of n points making up alignment
        :param width: width of road, or a list of n-1 widths corresponding to the segments between the alignment points
        :param oneway: whether or not the road is one-way
        """
        self.id = id
        self._fundamental_diagram_template_a = fundamental_diagram_a if fundamental_diagram_a is not None else FundamentalDiagram()
        self._fundamental_diagram_template_b = fundamental_diagram_b if fundamental_diagram_b is not None else FundamentalDiagram()
        self.alignment = alignment if isinstance(alignment, np.ndarray) else np.array(alignment)
        self._linestring = LineString(self.alignment)
        self._twin_links = {}  # dict mapping links to their "twins" (same portion of alignment, but opposite direction)
        if type(width) in [int, float]:
            self._widths = np.full(len(self.alignment)-1, width)
        else:
            if len(width) != len(self.alignment)-1:
                raise ValueError("Number of widths provided does not match number of segments in alignment.")
            self._widths = width if isinstance(width, np.ndarray) else np.array(width)
        self.oneway = oneway
        self._nodes = []  # type: list[Node]
        self._links = []  # type: list[Link]
        self.max_link_length =  max_link_length
        self.width_split = width_split
        self.from_intersection = None
        self.to_intersection = None

    @classmethod
    def between_intersections(cls, from_intersection, to_intersection, width=3.2, oneway=False, max_link_length=10, width_split=0.5):
        from_pt, to_pt = from_intersection.location, to_intersection.location
        road = cls([from_pt, to_pt], width=width, oneway=oneway, max_link_length=max_link_length, width_split=width_split)
        road.from_intersection, road.to_intersection = from_intersection, to_intersection
        return road

    def width_at_distance(self, distance):
        ls = self._linestring
        vertex_ds = np.cumsum([Point(c).distance(Point(c2)) for c, c2 in zip(ls.coords[1:], ls.coords[:-1])])
        return self._widths[np.searchsorted(vertex_ds, distance)-1]

    def bake(self):
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

    @property
    def nodes(self):
        return self._nodes

    @property
    def links(self):
        return self._links


class AbstractIntersection:
    def __init__(self, location, radius=8):
        self.location = location if isinstance(location, np.ndarray) else np.array(location)
        self.radius = radius
        self._nodes = []
        self._links = []

    def bake(self):
        # TODO: implement this
        pass

    @property
    def nodes(self):
        return self._nodes

    @property
    def links(self):
        return self._links


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
    rd = AbstractRoad([(0, 0), (100, 0), (150, 50)])
    anet = AbstractNetwork(roads=[rd], intersections=[])
    anet.net.plot()
    plt.show()