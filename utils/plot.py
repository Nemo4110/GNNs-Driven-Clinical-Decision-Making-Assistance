import math
import numpy as np
import matplotlib.pyplot as plt
import os.path as path


class Ploter:
    """For plotting metric in figure and save it to file."""
    def __init__(self, row_col=(1, 1),
                 xscale='linear', yscale='linear',
                 xlabel=None, ylabel=None,
                 xlim: list=None, ylim: list=None,
                 fmts: tuple=None, legends: list=None, figsize=(3.5, 2.5),
                 save_dir=r"./results"):
        self.xscale = xscale; self.yscale = yscale
        self.xlabel = xlabel; self.ylabel = ylabel
        self.xlim, self.ylim = xlim, ylim
        self.fmts = fmts
        self.legends = legends
        self.save_dir = save_dir

        nrows, ncols = row_col[0], row_col[1]
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))

        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.Xs, self.Ys = None, None

    def add_point(self, xs, ys):
        """Add multiple data points into the figure"""
        if not hasattr(ys, "__len__"):
            ys = [ys]

        n = len(ys)
        if not hasattr(xs, "__len__"):
            xs = [xs] * n

        if not self.Xs:
            self.Xs = [[] for _ in range(n)]
        if not self.Ys:
            self.Ys = [[] for _ in range(n)]

        for i, (x, y) in enumerate(zip(xs, ys)):
            assert x is not None and y is not None
            self.Xs[i].append(x)
            self.Ys[i].append(y)

    def plotting(self, description: str):
        len_axes = len(self.axes)
        self.fmts = ['-'] * len(self.Xs) if self.fmts is None else self.fmts

        if len_axes == 1:
            for X, Y, fmt in zip(self.Xs, self.Ys, self.fmts):
                self.axes[0].plot(X, Y, fmt)
            self.set_axis(self.axes[0],
                          self.xlabel, self.ylabel,
                          self.xlim, self.ylim,
                          self.xscale, self.yscale,
                          self.legends)
        else:
            assert len_axes > 1

            self.xlabel = [None] * len_axes if self.xlabel is None else self.xlabel
            self.ylabel = [None] * len_axes if self.ylabel is None else self.ylabel
            self.xlim = [[None, None]] * len_axes if self.xlim is None else self.xlim
            self.ylim = [[None, None]] * len_axes if self.ylim is None else self.ylim
            self.legends = [None] * len_axes if self.legends is None else self.legends

            for i, (X, Y, fmt) in enumerate(zip(self.Xs, self.Ys, self.fmts)):
                self.axes[i].plot(X, Y, fmt)
                self.set_axis(self.axes[i],
                              self.xlabel[i], self.ylabel[i],
                              self.xlim[i], self.ylim[i],
                              self.xscale, self.yscale,
                              self.legends[i])

        assert description is not None
        fname = path.join(self.save_dir, description)
        # plt.savefig()
        self.fig.tight_layout()  # <https://github.com/matplotlib/matplotlib/issues/17118>
        self.fig.savefig(fname)

    def set_axis(self, axis, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axis for matplotlib."""
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xscale(xscale)
        axis.set_yscale(yscale)
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        if legend:
            axis.legend(legend)
        axis.grid(True, linestyle='--', linewidth=.87)


if __name__ == "__main__":
    t = np.arange(0, 2.5, 0.1)
    y_1 = map(math.sin, math.pi * t)
    y_2 = map(math.sin, math.pi * t + math.pi/2)
    y_3 = map(math.sin, math.pi * t - math.pi/2)

    ploter = Ploter(row_col=(1, 1), xlabel="t", legends=["sin", "sin + 1/2pi", "sin - 1/2pi"], save_dir=".")

    for t_, y_1_, y_2_, y_3_ in zip(t, y_1, y_2, y_3):
        ploter.add_point(t_, (y_1_, y_2_, y_3_))

    ploter.plotting(description="test")


