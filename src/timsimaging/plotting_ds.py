import datashader as ds
import numpy as np
import param
import panel as pn
import pandas as pd
import xarray as xr

from bokeh.models import (
    ColumnDataSource,
    DataTable,
    HoverTool,
    LogColorMapper,
    LinearColorMapper,
    NumericInput,
    CustomJS,
    TapTool,
    BoxSelectTool,
    WheelZoomTool,
    CrosshairTool,
    NumberFormatter,
    TableColumn,
    MetricLength,
    Range1d,
    ScaleBar,
)
from bokeh.palettes import Viridis256, Magma256

#from .spectrum import export_imzML

from bokeh.layouts import column, grid, row, layout, gridplot
from bokeh.models.annotations.dimensional import CustomDimensional
from bokeh.plotting import figure
from bokeh.transform import log_cmap, linear_cmap

from typing import Callable, TYPE_CHECKING, Tuple, Literal

if TYPE_CHECKING:
    from .spectrum import Frame, MSIDataset

__all__ = ["image", "scatterplot", "heatmap", "mobilogram", "spectrum", "MSIDashboard"]


def image(
    dataset: "MSIDataset",
    mz: slice = None,
    mobility: slice = None,
    results: dict = None,
    feature_index: int = None,
    normalization: Literal["TIC", "RMS", "none"] = "none",
) -> Tuple[figure, ColumnDataSource]:
    """Visualize images of a dataset.
    By default it is the TIC image. If mz and mobility provided,
    all intensities within the slices would be aggregated across pixels

    :param dataset: the dataset to visualize
    :type dataset: MSIDataset
    :param mz: the m/z slice to subset the data, defaults to None
    :type mz: slice, optional
    :param mobility: the mobility slice to subset the data, defaults to None
    :type mobility: slice, optional
    :param normalization: normalize pixel intensities, defaults to "TIC"
    :type normalization: Literal[&quot;TIC&quot;, &quot;RMS&quot;, &quot;none&quot;], optional
    :return: the image and its data source
    :rtype: Tuple[figure, ColumnDataSource]
    """

    source = ColumnDataSource(dataset.pos.reset_index())

    if mz is not None or mobility is not None:
        indices = dataset.data[:, mobility, 0, mz, "raw"]
        intensities = dataset.data.bin_intensities(indices, axis=["rt_values"])[1:]
        title = f"mz={mz.start:.4f}-{mz.stop:.4f}, 1/K_0={mobility.start:.3f}-{mobility.stop:.3f}"
    elif results is not None and isinstance(feature_index, int):
        intensities = results["intensity_array"].iloc[:, feature_index]
        mz, mobility = results["peak_list"].iloc[feature_index][["mz_values", "mobility_values"]]
        title = f"mz={mz:.4f}, 1/K_0={mobility:.3f}"
    else:
        intensities = dataset.tic()
        title = "TIC image"
    source.data["total_intensity"] = intensities

    if normalization == "none":
        source.data["normalized"] = intensities / intensities.max()
    elif normalization == "TIC":
        normalized = intensities / dataset.tic()
        source.data["normalized"] = normalized / normalized.max()
    elif normalization == "RMS":
        source.data["normalized"] = intensities / intensities.std()

    f = figure(
        title=title,
        match_aspect=True,
        toolbar_location="right",
        x_axis_label="X",
        y_axis_label="Y",
    )
    f.toolbar.active_scroll = f.select_one({"type": WheelZoomTool})

    # (0,0) is on the top-left for ion images
    f.y_range.flipped = True

    cmap = LinearColorMapper(
        palette=Viridis256, low=0, high=1, low_color="#440154", high_color="#FDE724"
    )
    # cmap.tags = ["color_mapper"]
    # cmap = linear_cmap(
    #     "normalized", palette="Viridis256", low=0, high=1, low_color="#440154", high_color="#FDE724"
    # )
    pixel_grid = f.rect(
        x="XIndexPos",
        y="YIndexPos",
        color={"field": "normalized", "transform": cmap},
        width=1,
        height=1,
        source=source,
        hover_line_alpha=1,
        hover_line_width=1,
        hover_line_color="gray",
    )
    pixel_grid.tags = ["image"]

    color_bar = pixel_grid.construct_color_bar()
    f.add_layout(color_bar, "right")

    # Scale bar
    # auxillary range in micrometer
    PIXEL_SIZE_UM = dataset.resolution["xy"]
    micron_range = Range1d(
        start=f.x_range.start * PIXEL_SIZE_UM,
        end=f.x_range.end * PIXEL_SIZE_UM,
    )

    f.x_range.js_on_change(
        "start",
        CustomJS(args=dict(r=micron_range, k=PIXEL_SIZE_UM), code="r.start = cb_obj.start * k"),
    )
    f.x_range.js_on_change(
        "end", CustomJS(args=dict(r=micron_range, k=PIXEL_SIZE_UM), code="r.end = cb_obj.end * k")
    )
    scale_bar = ScaleBar(
        range=micron_range,
        unit="µm",
        dimensional=MetricLength(),
        # title=f"resolution: {PIXEL_SIZE_UM}µm",
        location="top_left",
        background_fill_alpha=0.5,
        # dimensional=Metric(base_unit="px"),
        # label = "@{value} @{unit}",
    )

    f.add_layout(scale_bar)

    # hover tool
    hover = HoverTool(
        renderers=[pixel_grid],
        tooltips=[
            ("X", "@XIndexPos"),
            ("Y", "@YIndexPos"),
            ("Frame index", "@Frame"),
            ("Intensity", "@total_intensity"),
        ],
    )
    f.add_tools(hover)
    return f, source


def scatterplot(frame: "Frame") -> Tuple[figure, ColumnDataSource]:
    """Visualize a 2D spectogram in scatter plot, with better performance compared with heatmap

    :param frame: the frame to visualize
    :type frame: Frame
    :return: the 2D spectogram and its datasource
    :rtype: figure
    """

    df = frame.data
    source = ColumnDataSource(df)
    width, height = frame.resolution

    f = figure(
        title="2D Spectrogram",
        x_range=(frame.mz_domain.min(), frame.mz_domain.max()),
        y_range=(frame.mobility_domain.min(), frame.mobility_domain.max()),
        toolbar_location="right",
        background_fill_color="black",
        aspect_ratio=1,
        match_aspect=True,
    )

    # the 2D spectrogram

    cmap = log_cmap(
        "intensity_values",
        palette="Magma256",
        low=0.01,
        # high=df["intensity_values"].quantile(q=0.95),
        high=df["intensity_values"].max(),
    )

    spec2d = f.scatter(
        x="mz_values",
        y="mobility_values",
        color=cmap,
        source=source,
    )
    color_bar = spec2d.construct_color_bar()
    f.add_layout(color_bar, "right")

    crosshair = CrosshairTool(line_color="white")
    f.add_tools(crosshair)

    hover = HoverTool(
        renderers=[spec2d],
        tooltips=[
            ("m/z", "@mz_values{0.0000}"),
            ("1/K0", "@mobility_values{0.0000}"),
            ("intensity", "@intensity_values"),
            ("index", "$index"),
        ],
    )
    f.add_tools(hover)
    return f, source


def heatmap(frame: "Frame") -> Tuple[figure, ColumnDataSource]:
    """Visualize a 2D spectogram

    :param frame: the frame to visualize
    :type frame: Frame
    :return: the 2D spectogram and its data source
    :rtype: figure
    """

    df = frame.data
    source = ColumnDataSource(df)
    width, height = frame.resolution

    f = figure(
        title=r"Intensity heatmap",
        x_range=(frame.mz_domain.min(), frame.mz_domain.max()),
        y_range=(frame.mobility_domain.min(), frame.mobility_domain.max()),
        x_axis_label="m/z",
        y_axis_label=r"$$1/K_0$$",
        toolbar_location="right",
        background_fill_color="black",
        aspect_ratio=1,
        # height=600,
        # sizing_mode="fixed",
        match_aspect=True,
    )
    f.toolbar.active_scroll = f.select_one({"type": WheelZoomTool})
    # the 2D spectrogram

    cmap = log_cmap(
        "intensity_values",
        palette="Magma256",
        low=0.01,
        # high=df["intensity_values"].quantile(q=0.95),
        high=df["intensity_values"].max(),
    )

    spec2d = f.rect(
        x="mz_values",
        y="mobility_values",
        width=width,
        height=height,
        color=cmap,
        source=source,
    )
    color_bar = spec2d.construct_color_bar()
    f.add_layout(color_bar, "right")

    crosshair = CrosshairTool(line_color="white")
    f.add_tools(crosshair)

    hover = HoverTool(
        renderers=[spec2d],
        tooltips=[
            ("m/z", "@mz_values{0.0000}"),
            ("1/K0", "@mobility_values{0.0000}"),
            ("intensity", "@intensity_values"),
            ("index", "$index"),
        ],
    )
    f.add_tools(hover)
    return f, source


def spectrum(data: pd.DataFrame) -> Tuple[figure, ColumnDataSource]:
    """Visualize a classical mass spectrum using line plot

    :param data: the data to visualize, with "mz_values" column and "intensity_values" column
    :type data: pd.DataFrame
    :return: the 1D mass spectrum and its data source
    :rtype: Tuple[figure, ColumnDataSource]
    """
    source = ColumnDataSource(data)
    f = figure(
        title="MS1 spectrum",
        toolbar_location="right",
        x_axis_label="m/z",
        y_axis_label="intensity",
    )

    spec1d = f.line(
        x="mz_values",
        y="intensity_values",
        source=source,
        hover_alpha=0.5,
        line_width=1.5,
    )
    hover = HoverTool(
        renderers=[spec1d],
        tooltips=[
            ("m/z", "@mz_values{0.0000}"),
            ("intensity", "@intensity_values"),
        ],
    )
    f.add_tools(hover)
    return f, source


def mobilogram(data: pd.DataFrame, transposed: bool = False) -> Tuple[figure, ColumnDataSource]:
    """Visualize a mobilogram using line plot

    :param data: the data to visualize, with "mobility_values" column and "intensity_values" column
    :type data: pd.DataFrame
    :param transposed: swap x and y axis to set the mobilogram as y-marginal, defaults to False
    :type transposed: bool, optional
    :return: the mobilogram and its data source
    :rtype: Tuple[figure, ColumnDataSource]
    """
    source = ColumnDataSource(data)
    if transposed:
        f = figure(
            title="Mobilogram",
            toolbar_location="right",
            x_axis_label="intensity",
            y_axis_label=r"$$1/K_0$$",
        )

        mob = f.line(
            x="intensity_values",
            y="mobility_values",
            source=source,
            hover_alpha=0.5,
            line_width=1.5,
        )
    else:
        f = figure(
            title="Mobilogram",
            toolbar_location="right",
            x_axis_label=r"$$1/K_0$$",
            y_axis_label="intensity",
        )

        mob = f.line(
            x="mobility_values",
            y="intensity_values",
            source=source,
            hover_alpha=0.5,
            line_width=1.5,
        )

    hover = HoverTool(
        renderers=[mob],
        tooltips=[
            ("1/K_0", "@mobility_values{0.0000}"),
            ("intensity", "@intensity_values"),
        ],
    )
    f.add_tools(hover)
    return f, source


# def feature_list(data):
#     source = ColumnDataSource(data)
#     columns = [
#         # TableColumn(
#         #     field="index",
#         #     title="#",
#         #     ),
#         TableColumn(
#             field="mz_values",
#             title="m/z",
#             formatter=NumberFormatter(format="0.000"),
#         ),
#         TableColumn(
#             field="mobility_values",
#             title="1/K0",
#             formatter=NumberFormatter(format="0.000"),
#         ),
#         TableColumn(
#             field="total_intensity",
#             title="total peak intensity",
#             formatter=NumberFormatter(format="0.000"),
#         ),
#     ]
#     filtered_source = ColumnDataSource(data.copy())
#     table = DataTable(source=filtered_source, columns=columns, index_position=0)

#     intensity_threshold = NumericInput(
#         title="Relative intensity threshold",
#         placeholder="0-100% to the max intensity",
#         low=0,
#         high=100,
#         value=0,
#         mode="float",
#     )

#     intensity_filter_callback = CustomJS(
#         args=dict(source=source, filtered_source=filtered_source),
#         code="""
#         const data = source.data;
#         const indices = data['index'];
#         const mz = data['mz_values'];
#         const mobility = data['mobility_values'];
#         const intensity = data['total_intensity'];

#         const percent = parseFloat(cb_obj.value);

#         if (isNaN(percent) || percent < 0 || percent > 100) {
#             return;
#         }

#         // Compute threshold
#         const max = Math.max(...intensity);
#         const threshold = (percent / 100) * max;

#         const new_indices = [];
#         const new_mz = [];
#         const new_mobility = [];
#         const new_intensity = [];

#         for (let i = 0; i < intensity.length; i++) {
#             if (intensity[i] >= threshold) {
#                 new_indices.push(indices[i]);
#                 new_mz.push(mz[i]);
#                 new_mobility.push(mobility[i]);
#                 new_intensity.push(intensity[i]);
#             }
#         }

#         filtered_source.data = { 'index': new_indices, 'mz_values': new_mz, 'mobility_values': new_mobility, 'total_intensity': new_intensity };
#         filtered_source.change.emit();
#     """,
#     )

#     intensity_threshold.js_on_change("value", intensity_filter_callback)
#     return column([intensity_threshold, table], aspect_ratio=1), filtered_source


def feature_list(data: pd.DataFrame):
    """Feature list table using Panel Tabulator with intensity threshold filter.

    Returns a (pn.Column, pn.widgets.Tabulator) tuple.
    The Tabulator replaces the ColumnDataSource — selection is accessed via
    tabulator.param.watch(..., ["selection"]) or tabulator.param.selection.

    :param data: peak list DataFrame with mz_values, mobility_values, total_intensity columns
    :type data: pd.DataFrame
    :return: (panel layout, tabulator widget)
    :rtype: Tuple[pn.Column, pn.widgets.Tabulator]
    """

    df = data.reset_index(names="frame")
    tabulator = pn.widgets.Tabulator(
        df,
        titles={
            "frame": "Frame index",
            "mz_values": "m/z",
            "mobility_values": "1/K₀",
            "total_intensity": "Total Intensity",
        },
        formatters={
            "mz_values": NumberFormatter(format="0.0000"),
            "mobility_values": NumberFormatter(format="0.000"),
            "total_intensity": NumberFormatter(format="0.00"),
        },
        page_size=10,
        pagination="local",
        selectable=1,  # single row selection
        show_index=False,
        disabled=True,
        # sizing_mode="stretch_width",
        max_height=400,
    )

    # Intensity threshold filter — replaces the CustomJS NumericInput
    threshold_input = pn.widgets.FloatInput(
        name="Relative intensity threshold (%)",
        value=0.0,
        start=0.0,
        end=100.0,
        step=1.0,
    )

    def _apply_filter(event):
        pct = event.new
        if pct is None or pct < 0:
            tabulator.value = df.copy()
            return
        max_intensity = df["total_intensity"].max()
        threshold = (pct / 100.0) * max_intensity
        tabulator.value = df.loc[df["total_intensity"] >= threshold].copy()
        # Clear selection when filter changes
        tabulator.selection = []

    threshold_input.param.watch(_apply_filter, ["value"])

    layout = pn.Column(threshold_input, tabulator)
    return layout, tabulator


def heatmap_ds(
    frame: "Frame",
    plot_width: int = 600,
    plot_height: int = 600,
):
    """Datashader-backed heatmap using quadmesh + bokeh image glyph.

    Uses bokeh's `image` glyph with LogColorMapper instead of `image_rgba`,
    so position (x/y/dw/dh) is set directly on the glyph rather than as
    ColumnDataSource columns — eliminating drift on rerasterization.

    Private attributes stashed on the figure for MSIDashboard:
        _ds_rasterize(x0, x1, y0, y1, pw, ph) -> np.ndarray  float32
        _ds_glyph     GlyphRenderer (image glyph)
        _ds_source    ColumnDataSource  (image data only)
        _ds_cmap      LogColorMapper
        _ds_x_min/max, _ds_y_min/max   data domain bounds
    """
    X = frame.mz_domain
    Y = frame.mobility_domain
    intensity_mx = frame.to_dense_array()  # (len(Y), len(X))

    da = xr.DataArray(
        intensity_mx,
        coords=[Y, X],
        dims=["mobility_values", "mz_values"],
        name="intensity_values",
    )

    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())

    def rasterize(x0, x1, y0, y1, pw=plot_width, ph=plot_height):
        cvs = ds.Canvas(
            plot_width=pw,
            plot_height=ph,
            x_range=(x0, x1),
            y_range=(y0, y1),
        )
        agg = cvs.quadmesh(
            da,
            x="mz_values",
            y="mobility_values",
            agg=ds.mean("intensity_values"),
        )
        # Return raw float array — color mapping done by Bokeh LogColorMapper
        return agg.values.astype(np.float32)

    # ── initial render ─────────────────────────────────────────────────────
    init_agg = rasterize(x_min, x_max, y_min, y_max)
    global_max = float(np.nanmax(intensity_mx))

    source = ColumnDataSource({"image": [init_agg]})
    cmap = LogColorMapper(
        palette=Magma256,
        low=0.01,
        high=global_max,
        nan_color=(0, 0, 0, 0),  # transparent for empty bins
    )

    f = figure(
        title="Intensity heatmap",
        x_range=(x_min, x_max),
        y_range=(y_min, y_max),
        x_axis_label="m/z",
        y_axis_label="1/K\u2080",
        width=plot_width,
        height=plot_height,
        toolbar_location="right",
        background_fill_color="black",
        match_aspect=True,
    )
    f.toolbar.active_scroll = f.select_one({"type": WheelZoomTool})

    # Position set directly on the glyph — not as source columns
    # This prevents drift: source update and glyph position are always in sync
    img_glyph = f.image(
        image="image",
        x=x_min,
        y=y_min,
        dw=x_max - x_min,
        dh=y_max - y_min,
        color_mapper=cmap,
        source=source,
    )
    color_bar = img_glyph.construct_color_bar()

    f.add_layout(color_bar, "right")
    hover = HoverTool(
        tooltips=[
            ("m/z", "$x{0.0000}"),
            ("1/K₀", "$y{0.0000}"),
        ],
        renderers=[img_glyph],
    )
    crosshair = CrosshairTool(line_color="white")
    f.add_tools(crosshair, hover)

    # Stash references for MSIDashboard
    f._ds_rasterize = rasterize
    f._ds_glyph = img_glyph
    f._ds_source = source
    f._ds_cmap = cmap
    f._ds_x_min = x_min
    f._ds_x_max = x_max
    f._ds_y_min = y_min
    f._ds_y_max = y_max

    return f, source


class MSIDashboard(param.Parameterized):

    selected_frame = param.Integer(default=-1)  # start from 1
    selected_peak = param.Integer(default=-1)  # start from 1
    x_start = param.Number()
    x_end = param.Number()
    y_start = param.Number()
    y_end = param.Number()
    show_peaks = param.Boolean(default=False)
    show_grid = param.Boolean(default=True)
    # ccs_computed = param.Boolean(default=False)

    def __init__(
        self,
        dataset: "MSIDataset",
        mean_spectrum: "Frame",
        peak_list: pd.DataFrame,
        peak_extents: pd.DataFrame,
        intensity_array: pd.DataFrame | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.peak_list = peak_list
        self.peak_extents = peak_extents
        self.intensity_array = intensity_array
        self.mean_spec = mean_spectrum
        self._current_df = mean_spectrum.data  # store data of heatmap before rasterization

        self._init_figures()
        self._init_peak_boxes()
        self._wire_bokeh_callbacks()
        self._init_widgets()

    # ── Initialisation ────────────────────────────────────────────────────

    def _init_figures(self):
        # init plots
        df = self.mean_spec.data
        self.image_fig, self.image_source = image(self.dataset)
        self.heatmap_fig, self.heatmap_source = heatmap_ds(self.mean_spec)
        self.spec_fig, self.spec_source = spectrum(
            df.groupby("mz_values")["intensity_values"].sum().reset_index()
        )
        self.mob_fig, self.mob_source = mobilogram(
            df.groupby("mobility_values")["intensity_values"].sum().reset_index(),
            transposed=True,
        )

        # Link axes so spectrum / mobilogram pan/zoom together with heatmap
        self.spec_fig.x_range = self.heatmap_fig.x_range
        self.mob_fig.y_range = self.heatmap_fig.y_range
        self.spec_fig.height = self.heatmap_fig.height // 2
        self.mob_fig.width = self.heatmap_fig.width // 2

        self.param.update(
            x_start=df["mz_values"].min(),
            x_end=df["mz_values"].max(),
            y_start=df["mobility_values"].min(),
            y_end=df["mobility_values"].max(),
        )
        # peak table
        self.peak_table_widget, self.peak_table_source = feature_list(self.peak_list)

    def _init_peak_boxes(self):
        w, h = self.mean_spec.resolution
        ext = self.peak_extents
        self._peak_rect_source = ColumnDataSource(
            pd.DataFrame(
                {
                    "x": 0.5 * (ext["mz_values", "min"] + ext["mz_values", "max"]),
                    "y": 0.5 * (ext["mobility_values", "min"] + ext["mobility_values", "max"]),
                    "width": ext["mz_values", "max"] - ext["mz_values", "min"] + w,
                    "height": ext["mobility_values", "max"] - ext["mobility_values", "min"] + h,
                }
            )
        )
        self.peak_boxes = self.heatmap_fig.rect(
            x="x",
            y="y",
            width="width",
            height="height",
            source=self._peak_rect_source,
            color="steelblue",
            fill_alpha=0,
            line_width=1.5,
        )
        self.peak_boxes.visible = False

    # info flow: bokeh/widget callback -> param -> watcher callback
    def _wire_bokeh_callbacks(self):
        # ── ion image tap ──────────────────────────────────────────────────
        pixel_grid = self.image_fig.select(tags=["image"])[0]
        self.image_fig.add_tools(TapTool(renderers=[pixel_grid], mode="replace"))
        self.image_source.selected.on_change("indices", self._pixel_selected)

        # ── box selection (ROI, future use) ────────────────────────────────
        # box_select = BoxSelectTool(renderers=[pixel_grid])
        # self.image_fig.add_tools(box_select)

        # ── peak table row selection ───────────────────────────────────────
        self.peak_table_source.on_click(self._peak_selected)

        # ── heatmap range → param (projections only, no rerasterize) ──────
        self.heatmap_fig.x_range.on_change("start", self._range_changed)
        self.heatmap_fig.x_range.on_change("end", self._range_changed)
        self.heatmap_fig.y_range.on_change("start", self._range_changed)
        self.heatmap_fig.y_range.on_change("end", self._range_changed)

    def _init_widgets(self):
        self.param.watch(self._on_show_peaks, ["show_peaks"])
        self.param.watch(self._on_show_grid, ["show_grid"])

        self.peaks_toggle = pn.widgets.Checkbox.from_param(
            self.param.show_peaks, name="Show peak boxes"
        )
        self.grid_toggle = pn.widgets.Checkbox.from_param(
            self.param.show_grid, name="Show grid lines"
        )

        # ion image dynamic color mapping
        cmap = self.image_fig.select_one(LinearColorMapper)
        self.cmap_slider = pn.widgets.RangeSlider(
            name="Color range % to max", start=0, end=100, value=(0, 100)
        )

        def _update_mapper(event):
            cmap.low = event.new[0] * 0.01
            cmap.high = event.new[1] * 0.01

        self.cmap_slider.param.watch(_update_mapper, "value")

        self.tic_btn = pn.widgets.Button(name="Show TIC image", button_type="primary")
        self.meanspec_btn = pn.widgets.Button(name="Show mean spectrum", button_type="primary")
        # self.ccs_btn = pn.widgets.Button(name="Compute CCS", button_type="success")
        self.rerasterize_btn = pn.widgets.Button(name="Refresh heatmap", button_type="warning")
        self.tic_btn.on_click(self._show_tic)
        self.meanspec_btn.on_click(self._show_mean_spectrum)
        # self.ccs_btn.on_click(self._compute_ccs)
        self.rerasterize_btn.on_click(
            lambda e: self._rerasterize()
        )  # widget callback must accept an event param

        self.export_feature_btn = pn.widgets.FileDownload(
            callback=self._export_csv,
            filename="peak_list.csv",
            button_type="success",
            label="Export feature list",
        )
        self.export_intensity_btn = pn.widgets.FileDownload(
            callback=self._export_intensity_array,
            filename="intensity_array.csv",
            button_type="success",
            label="Export intensity array",
            disabled=self.intensity_array is None,
        )

    # ── Callbacks ─────────────────────────────────────────────────────

    def _pixel_selected(self, attr, old, new):
        # print(f"selected pixel index {new[0]}")
        # Bokeh glyph is 0-indexed, TIMSImaging frame is 1-indexed
        self.param.update(selected_frame=new[0] + 1 if new else -1)

    def _peak_selected(self, event):
        # event.row is integer location in current filtered view
        # Tabulator._processed is applied with filtering but not sorting
        peak_idx = int(self.peak_table_source._processed.iloc[event.row]["frame"])
        # print(f"selected row: {event}\nreal index: {peak_idx}")
        self.param.update(selected_peak=peak_idx)

    def _range_changed(self, attr, old, new):
        xr, yr = self.heatmap_fig.x_range, self.heatmap_fig.y_range
        self.param.update(
            x_start=xr.start,
            x_end=xr.end,
            y_start=yr.start,
            y_end=yr.end,
        )

    # ── Datashader rerasterization ────────────────────────────────────────

    def _rerasterize(self):
        """Rerasterize the heatmap at the current view bounds.
        Called explicitly by the Rerasterize button — no automatic triggering.
        Glyph position is updated directly to prevent drift.
        """
        xr = self.heatmap_fig.x_range
        yr = self.heatmap_fig.y_range
        x0, x1 = xr.start, xr.end
        y0, y1 = yr.start, yr.end
        if None in (x0, x1, y0, y1):
            return

        # Clamp to data domain
        x0 = max(x0, self.heatmap_fig._ds_x_min)
        x1 = min(x1, self.heatmap_fig._ds_x_max)
        y0 = max(y0, self.heatmap_fig._ds_y_min)
        y1 = min(y1, self.heatmap_fig._ds_y_max)
        if x0 >= x1 or y0 >= y1:
            return

        pw = self.heatmap_fig.inner_width or 600
        ph = self.heatmap_fig.inner_height or 600

        new_agg = self.heatmap_fig._ds_rasterize(x0, x1, y0, y1, pw, ph)

        # Update color mapper range to local data bounds for good contrast
        valid = new_agg[np.isfinite(new_agg) & (new_agg > 0)]
        if valid.size:
            self.heatmap_fig._ds_cmap.low = float(valid.min())
            self.heatmap_fig._ds_cmap.high = float(valid.max())

        # Update image data in source
        self.heatmap_fig._ds_source.data = {"image": [new_agg]}

        # Update glyph position directly — atomic, no ColumnDataSource column sync issue
        glyph = self.heatmap_fig._ds_glyph.glyph
        glyph.x = x0
        glyph.y = y0
        glyph.dw = x1 - x0
        glyph.dh = y1 - y0

    def _rerasterize_from_frame(self, frame: "Frame"):
        """Rebuild the datashader pipeline for a new frame (e.g. after pixel tap)."""
        X = frame.mz_domain
        Y = frame.mobility_domain
        intensity_mx = frame.to_dense_array()

        da = xr.DataArray(
            intensity_mx,
            coords=[Y, X],
            dims=["mobility_values", "mz_values"],
            name="intensity_values",
        )

        x_min, x_max = float(X.min()), float(X.max())
        y_min, y_max = float(Y.min()), float(Y.max())

        def rasterize(x0, x1, y0, y1, pw=600, ph=600):
            cvs = ds.Canvas(
                plot_width=pw,
                plot_height=ph,
                x_range=(x0, x1),
                y_range=(y0, y1),
            )
            agg = cvs.quadmesh(
                da,
                x="mz_values",
                y="mobility_values",
                agg=ds.mean("intensity_values"),
            )
            return agg.values.astype(np.float32)

        # Replace pipeline on the figure
        self.heatmap_fig._ds_rasterize = rasterize
        self.heatmap_fig._ds_x_min = x_min
        self.heatmap_fig._ds_x_max = x_max
        self.heatmap_fig._ds_y_min = y_min
        self.heatmap_fig._ds_y_max = y_max

        # Re-render at the current view
        self._rerasterize()

    # ── Param watchers ────────────────────────────────────────────────────

    @param.depends("selected_frame", watch=True)
    def _update_plots(self):
        if self.selected_frame < 0:
            return
        frame = self.dataset[self.selected_frame]
        self._current_df = frame.data
        self.heatmap_fig.title.text = f"Intensity heatmap — frame {self.selected_frame}"
        self._rerasterize_from_frame(frame)
        self._update_projections()

    @param.depends("x_start", "x_end", "y_start", "y_end", watch=True)
    def _update_projections(self):
        df = self._current_df
        view = df.loc[
            df["mz_values"].between(self.x_start, self.x_end)
            & df["mobility_values"].between(self.y_start, self.y_end)
        ]
        self.spec_source.data = dict(
            view.groupby("mz_values")["intensity_values"].sum().reset_index()
        )
        self.mob_source.data = dict(
            view.groupby("mobility_values")["intensity_values"].sum().reset_index()
        )

    @param.depends("selected_peak", watch=True)
    def _update_ion_image(self):
        if self.selected_peak < 0:
            return
        # peak_idx = self.peak_table_source.data["index"][self.selected_peak]
        peak_idx = self.selected_peak
        if isinstance(self.intensity_array, pd.DataFrame):
            peak_intensities = self.intensity_array.loc[:, peak_idx]
        elif isinstance(self.peak_extents, pd.DataFrame):
            ext = self.peak_extents.loc[peak_idx]

            mz_min, mz_max = ext["mz_values", "min"], ext["mz_values", "max"]
            mob_min, mob_max = ext["mobility_values", "min"], ext["mobility_values", "max"]

            indices = self.dataset.data[:, mob_min:mob_max, 0, mz_min:mz_max, "raw"]
            peak_intensities = self.dataset.data.bin_intensities(indices, axis=["rt_values"])[1:]
        else:
            raise NotImplementedError("return intensity in a fixed window in the future")

        self.image_source.data["total_intensity"] = peak_intensities
        self.image_source.data["normalized"] = peak_intensities / peak_intensities.max()

        mz, mob = self.peak_list.loc[peak_idx][["mz_values", "mobility_values"]]
        self.image_fig.title.text = f"Ion image - m/z: {mz:.4f}  1/K\u2080: {mob:.3f}"

    @param.depends("selected_peak", watch=True)
    def _zoom_to_peak(self):
        if self.selected_peak < 0:
            return
        # peak_idx = self.peak_table_source.data["index"][self.selected_peak]
        peak_idx = self.selected_peak
        ext = self.peak_extents.loc[peak_idx]

        mz_min, mz_max = ext["mz_values", "min"], ext["mz_values", "max"]
        mob_min, mob_max = ext["mobility_values", "min"], ext["mobility_values", "max"]

        w, h = self.mean_spec.resolution
        padding = 1.5
        cx = 0.5 * (mz_min + mz_max)
        cy = 0.5 * (mob_min + mob_max)
        mz_half = (mz_max - mz_min + w) * padding
        mob_half = (mob_max - mob_min + h) * padding

        self.heatmap_fig.x_range.start = cx - mz_half
        self.heatmap_fig.x_range.end = cx + mz_half
        self.heatmap_fig.y_range.start = cy - mob_half
        self.heatmap_fig.y_range.end = cy + mob_half
        self._rerasterize()

    def _on_show_peaks(self, event):
        self.peak_boxes.visible = event.new

    def _on_show_grid(self, event):
        self.heatmap_fig.xgrid.visible = event.new
        self.heatmap_fig.ygrid.visible = event.new

    # ── Button actions ────────────────────────────────────────────────────

    def _show_tic(self, event=None):
        intensities = self.dataset.tic()
        self.image_source.data["total_intensity"] = intensities
        self.image_source.data["normalized"] = intensities / intensities.max()
        self.image_fig.title.text = "TIC image"

    def _show_mean_spectrum(self, event=None):
        self._current_df = self.mean_spec.data
        self.heatmap_fig.title.text = "Intensity heatmap - mean spectrum"
        self._rerasterize_from_frame(self.mean_spec)
        self._update_projections()

    def _export_csv(self):
        from io import StringIO

        buf = StringIO()
        self.peak_list.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def _export_intensity_array(self):
        from io import StringIO

        buf = StringIO()
        self.intensity_array.to_csv(buf)
        buf.seek(0)
        return buf

    # ── Layout ────────────────────────────────────────────────────────────

    def view(self):
        grid = gridplot(
            [[self.heatmap_fig, self.mob_fig], [self.spec_fig, None]],
            toolbar_location="right",
        )

        controls = pn.WidgetBox(
            "# WidgetBox",
            self.cmap_slider,
            self.peaks_toggle,
            self.grid_toggle,
            self.tic_btn,
            self.meanspec_btn,
            self.rerasterize_btn,
            self.export_feature_btn,
            self.export_intensity_btn,
        )

        gspec = pn.GridSpec(
            sizing_mode="stretch_both",
            min_height=800,
            # allow_resize=True,
            # allow_drag=True,
        )

        # Row 0-5: top half
        # # Ion image spans left 6 columns, full height of top block
        # gspec[0:6, 0:6] = pn.pane.Bokeh(self.image_fig, sizing_mode="stretch_both")
        # #gspec[0:6, 0:6] = pn.Column(self.image_fig, self.cmap_slider, sizing_mode="stretch_both")
        # # Heatmap grid spans right 6 columns, full height of top block
        # gspec[0:6, 6:] = pn.pane.Bokeh(grid, sizing_mode="stretch_width")

        # # Row 6-9: bottom half
        # # Peak table spans left 8 columns
        # gspec[6:9, 0:8] = self.peak_table_widget

        # # Controls span right 4 columns
        # gspec[6:9, 8:12] = controls

        gspec[0:6, 0:8] = pn.pane.Bokeh(self.image_fig, sizing_mode="stretch_width")
        #gspec[0:6, 0:6] = pn.Column(self.image_fig, self.cmap_slider, sizing_mode="stretch_both")
        # Heatmap grid spans right 6 columns, full height of top block
        gspec[0:6, 8:14] = pn.pane.Bokeh(grid, sizing_mode="stretch_width")

        # Row 6-9: bottom half
        # Peak table spans left 8 columns
        gspec[6:9, 0:8] = self.peak_table_widget

        # Controls span right 4 columns
        gspec[0:9, 14:16] = controls

        return gspec
