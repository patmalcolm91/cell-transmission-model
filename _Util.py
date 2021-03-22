"""
Various utility functions and classes.
"""

from matplotlib.lines import Line2D
from matplotlib.patches import Circle


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
    ax.add_patch(CircleDataUnits((0, 0), 3, fc="blue", ec="black"))
    ax.add_patch(Circle((5, 5), 3, fc="orange", ec="black"))
    ax.set_aspect("equal")
    ax.autoscale_view()
    plt.show()
