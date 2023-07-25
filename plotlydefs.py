"""
Handy plotly settings.
"""
from __future__ import annotations
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as plsub
# pio.kaleido.scope.mathjax = None
from .colordefs import (
    color_cycles,
    color_scales,
    opacity,
)

cc = color_cycles["whooie"]
cs = color_scales["sunset"]

def_size = {"width": 3.375 * 320, "height": 2.500 * 320}
template_whooie = go.layout.Template(
    data={
        "scatter": [
            go.Scatter(
                line_width=4,
            ),
        ],
        "scatterpolar": [
            go.Scatterpolar(
                line_width=4,
            ),
        ],
        "scatterternary": [
            go.Scatterternary(
                line_width=4,
            ),
        ],
    },
    layout=go.Layout(
        autosize=True,
        colorway=color_cycles["whooie"],
        colorscale={
            "diverging": color_scales["fire_ice"],
            "sequential": color_scales["sunset"],
            "sequentialminus": color_scales["vibrant"],
        },
        font=go.layout.Font(
            family="Myriad Pro",
            color="black",
            size=24.0,
        ),
        margin=go.layout.Margin(pad=5),
        paper_bgcolor='white',
        plot_bgcolor='#eaeaea',
        polar=go.layout.Polar(
            angularaxis=go.layout.polar.AngularAxis(
                gridcolor="white",
                gridwidth=3,
                linecolor="black",
                linewidth=3,
            ),
            radialaxis=go.layout.polar.RadialAxis(
                gridcolor="white",
                gridwidth=3,
                linecolor="black",
                linewidth=3,
            ),
            bgcolor="#eaeaea",
        ),
        title=go.layout.Title(
            x=0.05,
        ),
        xaxis=go.layout.XAxis(
            mirror="allticks",
            color="black",
            linecolor="black",
            linewidth=3,
            gridcolor="white",
            gridwidth=3,
            visible=True,
            zeroline=False,
        ),
        yaxis=go.layout.YAxis(
            mirror="allticks",
            color="black",
            linecolor="black",
            linewidth=3,
            gridcolor="white",
            gridwidth=3,
            visible=True,
            zeroline=False,
        ),
        modebar=go.layout.Modebar(
            add=[
                "drawline",
                "drawopenpath",
                "drawclosedpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
                "toggleHover",
                "resetViews",
                "toggleSpikelines",
                "resetViewMapbox",
                "zoom2d",
                "pan2d",
                "select2d",
                "lasso2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
                "zoom3d",
                "pan3d",
                "orbitRotation",
                "tableRotation",
                "handleDrag3d",
                "resetCameraDefault3d",
            ],
        ),
    )
)
pio.templates["whooie"] = template_whooie
pio.templates.default = "plotly+whooie"

def_show_config = dict(
    scrollZoom = True,
    displayModeBar = True,
    displaylogo = False,
    toImageButtonOptions = dict(
        width = 1080,
        height = 800,
        scale = 1.5,
    ),
)

class Plotler:
    def __init__(self, fig: go.Figure=None, nrows: int=1, ncols: int=1):
        self.fig = go.Figure() if fig is None else fig
        self.nrows = nrows
        self.ncols = ncols
        self._selector = (0, 0)
        self._figmethod = None
        self._figdir = set(dir(self.fig))

    @staticmethod
    def new(nrows: int=1, ncols: int=1, sharex: bool=False, sharey: bool=False,
            vspace: float=0.1, hspace=0.1, widths: list[float]=None,
            heights: list[float]=None, titles: list[str]=None,
            row_titles: list[str]=None, col_titles: list[str]=None,
            x_title: str=None, y_title: str=None) \
        -> Plotler:
        fig = plsub.make_subplots(
            rows=nrows, cols=ncols,
            shared_xaxes=sharex, shared_yaxes=sharey,
            vertical_spacing=vspace, horizontal_spacing=hspace,
            row_heights=heights, column_widths=widths,
            subplot_titles=titles,
            row_titles=row_titles, column_titles=col_titles,
            x_title=x_title, y_title=y_title
        )
        return Plotler(fig, nrows, ncols)

    def _verify_loc(self, loc: int | (int, int)) -> (int, int):
        if loc is None:
            return self._selector
        if not (
            (
                isinstance(loc, tuple)
                and all(isinstance(x, int) for x in loc)
                and len(loc) == 2
                and loc[0] in range(self.nrows)
                and loc[1] in range(self.ncols)
            )
            or (
                isinstance(loc, int)
                and loc < self.ncols * self.nrows
            )
        ):
            raise ValueError(
                "Index or indices must be within the ranges of the numbers of"
                " rows and columns"
            )
        if isinstance(loc, int):
            _selector = (loc // self.ncols, loc % self.ncols)
        else:
            _selector = loc
        return _selector

    def _loc_idx(self, loc: int | (int, int)) -> int:
        i, j = self._verify_loc(loc)
        return self.ncols * i + j

    def __getitem__(self, loc: int | (int, int)):
        self._selector = self._verify_loc(loc)
        return self

    def get_layout(self):
        return self.fig.layout

    def set_layout(self, overwrite: bool=False, **kwargs):
        if overwrite:
            self.fig.layout = go.Layout(**kwargs)
        else:
            self.fig.update_layout(**kwargs)
        return self

    def get_data(self, loc: int | (int, int)=None):
        _selector = self._verify_loc(loc)
        return self.fig.data[self.loc_idx(_selector)]

    def set_data(self, loc: int | (int, int)=None,
            datatype: plotly.basedatatypes.BaseTraceType=None,
            overwrite: bool=False, **kwargs):
        _selector = self._verify_loc(loc)
        _idx = self.loc_idx(_selector)
        if overwrite:
            self.data[_idx] \
                = (type(self.data[_idx]) if datatype is None else datatype)(
                    **kwargs)
        else:
            self.data[_idx].update(**kwargs)
        return self

    def add_data(self, data, loc: int | (int, int)=None, **kwargs):
        _selector = self._verify_loc(loc)
        self.fig.add_trace(
            data, row=_selector[0] + 1, col=_selector[1] + 1, **kwargs)
        return self

    def do(self, f: str, *args, **kwargs):
        return getattr(self, f)(*args, **kwargs)

    def __getattr__(self, attr: str):
        if attr in dir(self):
            return getattr(self, attr)
        elif attr in self._figdir:
            if isinstance(getattr(self.fig, attr), type(self.do)):
                self._figmethod = attr
                return self._process_figmethod
            else:
                return getattr(self.fig, attr)
        else:
            raise AttributeError

    def _process_figmethod(self, *args, **kwargs):
        if self._figmethod is None:
            raise Exception("figmethod is None")
        if self._figmethod[:4] == "add_":
            self._do_add(getattr(self.fig, self._figmethod), *args, **kwargs)
        else:
            getattr(self.fig, self._figmethod)(*args, **kwargs)
        self._figmethod = None
        return self

    def _set_rowcol(self, kwargs):
        _kwargs = kwargs.copy()
        _kwargs["row"] = kwargs.get("row", self._selector[0] + 1)
        _kwargs["col"] = kwargs.get("col", self._selector[1] + 1)
        return _kwargs

    def _do_add(self, f: str, *args, **kwargs):
        _kwargs = self._set_rowcol(kwargs)
        f(*args, **_kwargs)
        return self

    def set_xaxis(self, *args, **kwargs):
        _kwargs = self._set_rowcol(kwargs)
        self.fig.update_xaxes(**_kwargs)
        return self

    def set_yaxis(self, *args, **kwargs):
        _kwargs = self._set_rowcol(kwargs)
        self.fig.update_yaxes(**_kwargs)
        return self

    def set_zaxis(self, *args, **kwargs):
        rowcol = self._set_rowcol(dict())
        self.fig.update_scenes(
            zaxis=kwargs,
            domain=dict(
                row=rowcol["row"],
                column=rowcol["col"],
            )
        )
        return self

    def set_caxis(self, *args, **kwargs):
        _kwargs = self._set_rowcol(kwargs)
        self.fig.update_coloraxes(**_kwargs)
        return self

    def show(self, config: dict[str, ...]=None):
        self.fig.show(config=def_show_config if config is None else config)
        return self

    def write_html(self, filename: str, config: dict[str, ...]=None, **kwargs):
        self.fig.write_html(
            filename,
            def_show_config if config is None else config,
            **kwargs
        )
        return self

    def close(self):
        self.fig = None

