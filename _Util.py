"""
Various utility functions and classes.
"""

from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np


class LineDataUnits(Line2D):
    """
    A Line2D object, but with the linewidth and dash properties defined in data coordinates.
    """
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", kwargs.pop("lw", 1))
        _dashes_data = kwargs.pop("dashes", (1, 0))
        super().__init__(*args, **kwargs)
        if _dashes_data != (1, 0):
            self.set_linestyle("--")
        self._lw_data = _lw_data
        self._dashes_data = _dashes_data
        self._dashOffset = 0

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    def _get_dashes(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            dpu = (trans((1, 1)) - trans((0, 0)))[0]
            return tuple([u*dpu*ppd for u in self._dashes_data])
        else:
            return tuple((1, 0))

    def _set_dashes(self, dashes):
        self._dashes_data = dashes

    _linewidth = property(_get_lw, _set_lw)
    _dashSeq = property(_get_dashes, _set_dashes)


class CircleDataUnits(Circle):
    """
    A Circle patch, but with the linewidth and dash properties defined in data coordinates.
    """
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", kwargs.pop("lw", 1))
        _dashes_data = kwargs.pop("dashes", (1, 0))
        super().__init__(*args, **kwargs)
        if _dashes_data != (1, 0):
            self.set_linestyle("--")
        self._lw_data = _lw_data
        self._dashes_data = _dashes_data
        self._dashOffset = 0

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    def _get_dashes(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            dpu = (trans((1, 1)) - trans((0, 0)))[0]
            return tuple([u*dpu*ppd for u in self._dashes_data])
        else:
            return tuple((1, 0))

    def _set_dashes(self, dashes):
        self._dashes_data = dashes

    _linewidth = property(_get_lw, _set_lw)
    _dashSeq = property(_get_dashes, _set_dashes)


class EventManager:
    """Convenience class for managing and querying a list of abstract 'events' with start and end times."""
    def __init__(self):
        self._events = []  # list of (start_time, end_time, event) tuples
        self._prev_time = None

    def add(self, event, start_time, end_time=np.inf):
        """Add an event that is active between the specified start and end times"""
        self._events.append((start_time, end_time, event))

    def get_active(self, time):
        """Return all events which are active at the specified time."""
        return [e for t0, t1, e in self._events if t0 <= time < t1]

    def get_newly_active_and_inactive(self, time):
        """Return events which changed from inactive to active or vice versa since the last call to this function."""
        newly_active = [e for t0, t1, e in self._events if t0 <= time < t1 and (self._prev_time is None or self._prev_time < t0)]
        newly_inactive = [] if self._prev_time is None else [e for t0, t1, e in self._events if self._prev_time < t1 <= time]
        self._prev_time = time
        return newly_active, newly_inactive


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    ax.add_patch(CircleDataUnits((0, 0), 3, fc="blue", ec="black"))
    ax.add_patch(Circle((5, 5), 3, fc="orange", ec="black"))
    ax.set_aspect("equal")
    ax.autoscale_view()
    plt.show()
