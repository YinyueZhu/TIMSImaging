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
    """Visualize a spatial ion image of a dataset.

    By default renders the TIC image. If ``mz`` and/or ``mobility`` are given,
    intensities within those slices are summed per pixel. Alternatively, if
    ``results`` (from :meth:`MSIDataset.process`) and a ``feature_index`` are
    supplied, the pre-computed intensity array for that feature is used.

    The rendered pixel color is driven by the ``"normalized"`` field in the
    returned ``ColumnDataSource``, which is always scaled to [0, 1]. The raw
    summed intensities are also stored as ``"total_intensity"`` so hover
    tooltips can display physical values.

    A ``ScaleBar`` in µm is added via an auxiliary ``Range1d`` that is kept in
    sync with the figure's x-range through a ``CustomJS`` callback, so the bar
    remains correct after pan/zoom without a Python round-trip.

    The ``rect`` glyph renderer is tagged ``["image"]`` so that
    :class:`MSIDashboard` (and other callers) can retrieve it by tag to attach
    a :class:`~bokeh.models.tools.TapTool`.

    :param dataset: the dataset to visualize
    :type dataset: MSIDataset
    :param mz: m/z range to integrate per pixel; both ``mz`` and ``mobility``
        must be provided together when using slice-based extraction, defaults to None
    :type mz: slice, optional
    :param mobility: 1/K₀ range to integrate per pixel, defaults to None
    :type mobility: slice, optional
    :param results: output dictionary from :meth:`MSIDataset.process`, used
        together with ``feature_index`` to display a pre-computed ion image,
        defaults to None
    :type results: dict, optional
    :param feature_index: zero-based column index into
        ``results["intensity_array"]``, defaults to None
    :type feature_index: int, optional
    :param normalization: pixel intensity normalization strategy.
        ``"none"`` divides by the per-image maximum; ``"TIC"`` divides by the
        pixel TIC then rescales to [0, 1]; ``"RMS"`` divides by the standard
        deviation across pixels, defaults to ``"none"``
    :type normalization: Literal["TIC", "RMS", "none"], optional
    :return: the configured Bokeh figure and its ``ColumnDataSource``
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

    # (0,0) is at the top-left for ion images, matching stage coordinate convention
    f.y_range.flipped = True

    # LinearColorMapper is used directly so the color bar can be constructed from the glyph
    cmap = LinearColorMapper(
        palette=Viridis256, low=0, high=1, low_color="#440154", high_color="#FDE724"
    )
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
    # Tag used by MSIDashboard to retrieve this renderer for TapTool attachment
    pixel_grid.tags = ["image"]

    color_bar = pixel_grid.construct_color_bar()
    f.add_layout(color_bar, "right")

    # Auxiliary Range1d in µm that mirrors the pixel x-range via CustomJS,
    # providing a physical scale bar without a Python round-trip on pan/zoom
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
        location="top_left",
        background_fill_alpha=0.5,
    )
    f.add_layout(scale_bar)

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
    """Visualize a 2D spectrogram as a scatter plot.

    Each data point is plotted at its (m/z, 1/K₀) coordinate and colored by
    intensity on a logarithmic scale. Scatter rendering is faster than
    :func:`heatmap` for sparse frames because no ``rect`` glyph geometry needs
    to be computed, at the cost of losing the physical pixel-size context.

    :param frame: the frame to visualize
    :type frame: Frame
    :return: the configured Bokeh figure and its ``ColumnDataSource``
    :rtype: Tuple[figure, ColumnDataSource]
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

    cmap = log_cmap(
        "intensity_values",
        palette="Magma256",
        low=0.01,
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
    """Visualize a 2D spectrogram as a rect-glyph heatmap.

    Each data point is rendered as a rectangle whose width and height equal the
    minimum m/z and 1/K₀ strides of the frame (``frame.resolution``),
    effectively tiling the 2D space. Intensity is mapped to color on a
    logarithmic scale using the Magma256 palette.

    For large frames prefer :func:`heatmap_ds`, which uses Datashader to
    rasterize the data server-side and avoids sending every data point to the
    browser.

    :param frame: the frame to visualize
    :type frame: Frame
    :return: the configured Bokeh figure and its ``ColumnDataSource``
    :rtype: Tuple[figure, ColumnDataSource]
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
        match_aspect=True,
    )
    f.toolbar.active_scroll = f.select_one({"type": WheelZoomTool})

    cmap = log_cmap(
        "intensity_values",
        palette="Magma256",
        low=0.01,
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
    """Visualize a 1D mass spectrum as a line plot.

    :param data: DataFrame with ``"mz_values"`` and ``"intensity_values"`` columns
    :type data: pd.DataFrame
    :return: the configured Bokeh figure and its ``ColumnDataSource``
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
    """Visualize an ion mobilogram as a line plot.

    When ``transposed=True`` the axes are swapped so that 1/K₀ runs along the
    y-axis, matching the orientation of the heatmap y-axis. This is the
    preferred mode when the mobilogram is displayed as the right-hand marginal
    of a 2D heatmap layout.

    :param data: DataFrame with ``"mobility_values"`` and ``"intensity_values"`` columns
    :type data: pd.DataFrame
    :param transposed: if ``True``, plot 1/K₀ on y and intensity on x
        (y-marginal orientation), defaults to ``False``
    :type transposed: bool, optional
    :return: the configured Bokeh figure and its ``ColumnDataSource``
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


# Old Bokeh-only implementation of feature_list (kept for reference).
# Replaced by the Panel Tabulator version below, which provides server-side
# filtering without a CustomJS round-trip.
# def feature_list(data):
#     source = ColumnDataSource(data)
#     columns = [
#         TableColumn(field="mz_values",      title="m/z",              formatter=NumberFormatter(format="0.000")),
#         TableColumn(field="mobility_values", title="1/K0",             formatter=NumberFormatter(format="0.000")),
#         TableColumn(field="total_intensity", title="total peak intensity", formatter=NumberFormatter(format="0.000")),
#     ]
#     filtered_source = ColumnDataSource(data.copy())
#     table = DataTable(source=filtered_source, columns=columns, index_position=0)
#     intensity_threshold = NumericInput(title="Relative intensity threshold", placeholder="0-100% to the max intensity", low=0, high=100, value=0, mode="float")
#     intensity_filter_callback = CustomJS(args=dict(source=source, filtered_source=filtered_source), code="""...""")
#     intensity_threshold.js_on_change("value", intensity_filter_callback)
#     return column([intensity_threshold, table], aspect_ratio=1), filtered_source


def feature_list(data: pd.DataFrame) -> Tuple[pn.Column, pn.widgets.Tabulator]:
    """Build a Panel-based interactive peak-list table with an intensity threshold filter.

    The table is backed by a :class:`panel.widgets.Tabulator` widget, which
    replaces the earlier Bokeh-only ``DataTable`` + ``CustomJS`` approach.
    Row selection is single-row and is accessed via the tabulator's
    ``on_click`` callback or ``tabulator.param.selection``.

    The ``"frame"`` column in the displayed table holds the original DataFrame
    index (peak index from :meth:`Frame.peakPick`), which is used by
    :class:`MSIDashboard` to resolve the selected peak back to its entry in
    ``peak_list`` and ``peak_extents`` even after filtering changes the visible
    row order.

    The returned ``tabulator`` widget's ``.value`` is replaced (not mutated)
    each time the threshold filter fires, which clears the current selection
    automatically.

    :param data: peak list DataFrame with at minimum ``"mz_values"``,
        ``"mobility_values"``, and ``"total_intensity"`` columns; an optional
        ``"ccs_values"`` column is shown when present
    :type data: pd.DataFrame
    :return: a ``(pn.Column, pn.widgets.Tabulator)`` tuple — the Column bundles
        the threshold input above the table for embedding in a layout, while
        the Tabulator is returned separately so callers can attach row-click
        callbacks
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
            "ccs_values": "CCS",
        },
        formatters={
            "mz_values": NumberFormatter(format="0.0000"),
            "mobility_values": NumberFormatter(format="0.000"),
            "total_intensity": NumberFormatter(format="0.00"),
            "ccs_values": NumberFormatter(format="0.0"),
        },
        page_size=10,
        pagination="local",
        selectable=1,  # single-row selection
        show_index=False,
        disabled=True,
        max_height=400,
    )

    # FloatInput replaces the CustomJS NumericInput from the Bokeh-only version,
    # enabling pure-Python filtering without browser-side JS
    threshold_input = pn.widgets.FloatInput(
        name="Relative intensity threshold (%)",
        value=0.0,
        start=0.0,
        end=100.0,
        step=1.0,
    )

    def _apply_filter(event):
        # Recompute the visible rows whenever the threshold changes
        pct = event.new
        if pct is None or pct < 0:
            tabulator.value = df.copy()
            return
        max_intensity = df["total_intensity"].max()
        threshold = (pct / 100.0) * max_intensity
        tabulator.value = df.loc[df["total_intensity"] >= threshold].copy()
        # Clear selection so a stale index does not trigger a spurious peak callback
        tabulator.selection = []

    threshold_input.param.watch(_apply_filter, ["value"])

    layout = pn.Column(threshold_input, tabulator)
    return layout, tabulator


def heatmap_ds(
    frame: "Frame",
    plot_width: int = 600,
    plot_height: int = 600,
) -> Tuple[figure, ColumnDataSource]:
    """Build a Datashader-backed heatmap Bokeh figure for a single frame.

    Datashader rasterizes the full 2D intensity array into a fixed-resolution
    grid server-side, so only ``plot_width × plot_height`` pixels are sent to
    the browser regardless of how many data points the frame contains.

    **Rasterization pipeline**

    1. ``frame.to_dense_array()`` produces a ``(len(Y), len(X))`` NumPy array
       from the sparse COO representation.
    2. The array is wrapped in an :class:`xarray.DataArray` with m/z and 1/K₀
       coordinates so Datashader can re-aggregate on arbitrary sub-ranges.
    3. A closure ``rasterize(x0, x1, y0, y1, pw, ph)`` is defined over the
       ``DataArray`` and stashed as ``figure._ds_rasterize``.  Callers invoke
       it to produce a fresh ``float32`` array for the current viewport, then
       push the result into ``figure._ds_source`` and update the glyph position
       directly (see below).

    **Why glyph position is set directly (not via source columns)**

    Bokeh's ``image`` glyph accepts ``x``, ``y``, ``dw``, ``dh`` either as
    scalar properties on the glyph object or as columns in the
    ``ColumnDataSource``.  Using source columns creates a potential race
    condition: if ``source.data`` is replaced before the glyph property update
    arrives, Bokeh may briefly render the new pixel data at the old position
    (drift).  Setting ``glyph.x / .y / .dw / .dh`` as scalar properties and
    updating them atomically after the source update eliminates this.

    **Private attributes stashed on the returned figure**

    These are consumed by :class:`MSIDashboard` and should not be accessed
    directly from user code:

    - ``_ds_rasterize(x0, x1, y0, y1, pw, ph) -> np.ndarray`` — re-aggregation closure
    - ``_ds_glyph`` — the ``GlyphRenderer`` wrapping the ``image`` glyph
    - ``_ds_source`` — the ``ColumnDataSource`` holding ``{"image": [array]}``
    - ``_ds_cmap`` — the ``LogColorMapper``; its ``low``/``high`` are updated per render
    - ``_ds_x_min``, ``_ds_x_max``, ``_ds_y_min``, ``_ds_y_max`` — full data domain bounds
      used to clamp zoom extents before passing to Datashader

    :param frame: the frame to visualize
    :type frame: Frame
    :param plot_width: pixel width of the figure and the initial raster grid,
        defaults to 600
    :type plot_width: int, optional
    :param plot_height: pixel height of the figure and the initial raster grid,
        defaults to 600
    :type plot_height: int, optional
    :return: the configured Bokeh figure and its ``ColumnDataSource``
    :rtype: Tuple[figure, ColumnDataSource]
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
        # Aggregate using mean so zooming in does not artificially inflate intensity
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
        # Return float32 — color mapping is handled by Bokeh's LogColorMapper
        return agg.values.astype(np.float32)

    # Initial full-domain render
    init_agg = rasterize(x_min, x_max, y_min, y_max)
    global_max = float(np.nanmax(intensity_mx))

    source = ColumnDataSource({"image": [init_agg]})
    cmap = LogColorMapper(
        palette=Magma256,
        low=0.01,
        high=global_max,
        nan_color=(0, 0, 0, 0),  # transparent for empty/zero bins
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

    # Glyph position is set as scalar properties (not source columns) to
    # prevent position/data drift when the source is replaced on rerasterization
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

    # Stash pipeline references for MSIDashboard
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
    """Interactive Panel + Bokeh dashboard for exploring an MSI dataset.

    Combines a spatial ion image, a 2D intensity heatmap (Datashader-backed),
    1D spectral and mobilogram projections, and a peak-list table into a single
    linked layout.  The class uses ``param`` as the shared state bus: Bokeh
    callbacks translate low-level events (pixel tap, range change) into
    ``param`` updates, and ``param`` watchers propagate those updates to the
    relevant plots.

    **Control flow overview**

    User interaction → Bokeh/Panel callback → ``param`` update → watcher → plot update

    1. **Pixel tap** on the ion image:
       ``_pixel_selected`` (Bokeh ``on_change``) → ``selected_frame`` param
       → ``_update_plots`` (watcher) rebuilds the heatmap for that frame and
       refreshes the 1D projections.

    2. **Peak table row click**:
       ``_peak_selected`` (Panel ``on_click``) → ``selected_peak`` param
       → ``_update_ion_image`` zooms the heatmap to the peak extent and
       updates the ion image to show that peak's pixel-wise intensity.
       → ``_zoom_to_peak`` (separate watcher on the same param) centers the
       heatmap view on the selected peak and triggers a rerasterization.

    3. **Heatmap pan/zoom**:
       ``_range_changed`` (Bokeh ``on_change``) → ``x_start/x_end/y_start/y_end``
       params → ``_update_projections`` recomputes the 1D spectrum and
       mobilogram from the current viewport's raw data. Rerasterization is
       *not* triggered automatically on zoom to avoid excessive computation;
       the user must click the "Refresh heatmap" button.

    4. **Refresh heatmap button**:
       calls ``_rerasterize()`` directly at the current view bounds.

    5. **Show TIC / Show mean spectrum buttons**:
       ``_show_tic`` / ``_show_mean_spectrum`` update the image or heatmap
       source in-place without going through ``param``.

    **Datashader rasterization pipeline**

    The heatmap is built by :func:`heatmap_ds`, which stashes a ``rasterize``
    closure on the figure object (``figure._ds_rasterize``). The closure
    captures a fixed :class:`xarray.DataArray` for the current frame.

    - ``_rerasterize()`` — rerasterizes at the current view bounds using the
      existing closure; updates ``source.data`` and the glyph's scalar position
      properties atomically to prevent drift.
    - ``_rerasterize_from_frame(frame)`` — replaces the ``DataArray`` and
      closure entirely (e.g. after a pixel tap selects a new frame), then calls
      ``_rerasterize()``.

    The ``LogColorMapper`` ``low``/``high`` bounds are updated on each
    rerasterization to the min/max of the visible non-zero pixels, giving
    good local contrast at any zoom level.

    **Initialisation sequence**

    ``__init__`` calls four private setup methods in order:

    1. ``_init_figures`` — creates all Bokeh figures and links their ranges.
    2. ``_init_peak_boxes`` — draws the hidden peak-extent rectangles on the
       heatmap.
    3. ``_wire_bokeh_callbacks`` — attaches ``on_change`` / ``on_click``
       listeners on Bokeh objects.
    4. ``_init_widgets`` — creates Panel widgets and wires them to params and
       button callbacks.

    Call :meth:`view` to obtain the assembled ``pn.GridSpec`` layout for
    serving or embedding in a notebook.

    :param dataset: the full MSI dataset; used for TIC retrieval, per-pixel
        frame access, and peak intensity extraction
    :type dataset: MSIDataset
    :param mean_spectrum: pre-computed mean spectrum shown in the heatmap on
        startup
    :type mean_spectrum: Frame
    :param peak_list: DataFrame of detected peaks with ``"mz_values"``,
        ``"mobility_values"``, and ``"total_intensity"`` columns; index
        values are used as stable peak identifiers throughout
    :type peak_list: pd.DataFrame
    :param peak_extents: MultiIndex DataFrame produced by
        :meth:`Frame.peakPick` (``return_extents=True``) with m/z and 1/K₀
        min/max extents for each peak, used for bounding-box overlays and
        pixel intensity extraction
    :type peak_extents: pd.DataFrame
    :param intensity_array: optional pre-computed ``(n_pixels × n_peaks)``
        intensity matrix from :meth:`MSIDataset.integrate_intensity`; when
        provided, ion images are read directly from this array instead of
        being extracted on-demand from the raw data, defaults to ``None``
    :type intensity_array: pd.DataFrame | None, optional
    """

    # Negative sentinel (-1) means "nothing selected"
    selected_frame = param.Integer(default=-1)   # 1-indexed frame number
    selected_peak = param.Integer(default=-1)    # peak index from peak_list
    x_start = param.Number()   # current heatmap x_range.start (m/z)
    x_end = param.Number()     # current heatmap x_range.end   (m/z)
    y_start = param.Number()   # current heatmap y_range.start (1/K₀)
    y_end = param.Number()     # current heatmap y_range.end   (1/K₀)
    show_peaks = param.Boolean(default=False)  # toggles peak-box overlay visibility
    show_grid = param.Boolean(default=True)    # toggles heatmap grid lines

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
        # Raw frame DataFrame kept separately so projections can be recomputed
        # from sparse data without re-querying the heatmap raster
        self._current_df = mean_spectrum.data

        self._init_figures()
        self._init_peak_boxes()
        self._wire_bokeh_callbacks()
        self._init_widgets()

    # ── Initialisation ────────────────────────────────────────────────────

    def _init_figures(self):
        # Create all Bokeh figures and link shared axes
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

        # Share ranges so the spectrum and mobilogram stay aligned with the heatmap
        self.spec_fig.x_range = self.heatmap_fig.x_range
        self.mob_fig.y_range = self.heatmap_fig.y_range
        self.spec_fig.height = self.heatmap_fig.height // 2
        self.mob_fig.width = self.heatmap_fig.width // 2

        # Initialise range params to the full mean-spectrum domain
        self.param.update(
            x_start=df["mz_values"].min(),
            x_end=df["mz_values"].max(),
            y_start=df["mobility_values"].min(),
            y_end=df["mobility_values"].max(),
        )
        self.peak_table_widget, self.peak_table_source = feature_list(self.peak_list)

    def _init_peak_boxes(self):
        # Compute rect geometry (centre + size) from peak extents and add a
        # hidden overlay to the heatmap; visibility is toggled by show_peaks param
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

    def _wire_bokeh_callbacks(self):
        # Attach all Bokeh-side event listeners.
        # Control flow: Bokeh callback → param.update() → param watcher → plot update

        # Ion image tap: translate glyph index (0-based) to frame index (1-based)
        pixel_grid = self.image_fig.select(tags=["image"])[0]
        self.image_fig.add_tools(TapTool(renderers=[pixel_grid], mode="replace"))
        self.image_source.selected.on_change("indices", self._pixel_selected)

        # Box select tool (reserved for future ROI selection feature)
        # box_select = BoxSelectTool(renderers=[pixel_grid])
        # self.image_fig.add_tools(box_select)

        # Peak table: Panel Tabulator uses on_click rather than Bokeh's on_change
        self.peak_table_source.on_click(self._peak_selected)

        # Heatmap range changes update x/y params, which drive projection updates.
        # Rerasterization is intentionally NOT triggered here to avoid overhead
        # on every pan/zoom step; the user triggers it explicitly via the button.
        self.heatmap_fig.x_range.on_change("start", self._range_changed)
        self.heatmap_fig.x_range.on_change("end", self._range_changed)
        self.heatmap_fig.y_range.on_change("start", self._range_changed)
        self.heatmap_fig.y_range.on_change("end", self._range_changed)

    def _init_widgets(self):
        # Create Panel widgets, bind them to params, and wire button callbacks

        # Param watchers for boolean toggles (peak boxes, grid)
        self.param.watch(self._on_show_peaks, ["show_peaks"])
        self.param.watch(self._on_show_grid, ["show_grid"])

        self.peaks_toggle = pn.widgets.Checkbox.from_param(
            self.param.show_peaks, name="Show peak boxes"
        )
        self.grid_toggle = pn.widgets.Checkbox.from_param(
            self.param.show_grid, name="Show grid lines"
        )

        # RangeSlider drives the ion image LinearColorMapper low/high directly
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
        self.rerasterize_btn = pn.widgets.Button(name="Refresh heatmap", button_type="warning")
        self.tic_btn.on_click(self._show_tic)
        self.meanspec_btn.on_click(self._show_mean_spectrum)
        # Panel button callbacks receive an event argument; wrap _rerasterize in a lambda
        self.rerasterize_btn.on_click(lambda e: self._rerasterize())

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

    # ── Bokeh / Panel event callbacks ─────────────────────────────────────
    # These translate raw events into param updates; they do not touch plots
    # directly. Plot updates are handled by the param watchers below.

    def _pixel_selected(self, attr, old, new):
        # Bokeh glyph indices are 0-based; TIMSImaging frame indices are 1-based
        self.param.update(selected_frame=new[0] + 1 if new else -1)

    def _peak_selected(self, event):
        # event.row is the integer location in the currently filtered Tabulator view.
        # Tabulator._processed reflects filtering but not sorting, so we resolve
        # the original peak index through the "frame" column before updating param.
        peak_idx = int(self.peak_table_source._processed.iloc[event.row]["frame"])
        self.param.update(selected_peak=peak_idx)

    def _range_changed(self, attr, old, new):
        # Mirror heatmap view bounds into params so projection watchers fire
        xr, yr = self.heatmap_fig.x_range, self.heatmap_fig.y_range
        self.param.update(
            x_start=xr.start,
            x_end=xr.end,
            y_start=yr.start,
            y_end=yr.end,
        )

    # ── Datashader rerasterization ────────────────────────────────────────

    def _rerasterize(self):
        # Rerasterize at the current heatmap view bounds using the existing
        # DataArray closure. Clamps the viewport to the data domain to avoid
        # Datashader receiving out-of-range coordinates. Updates the
        # LogColorMapper to the local intensity range for optimal contrast.
        # Glyph position is set as scalar properties to prevent drift.
        xr = self.heatmap_fig.x_range
        yr = self.heatmap_fig.y_range
        x0, x1 = xr.start, xr.end
        y0, y1 = yr.start, yr.end
        if None in (x0, x1, y0, y1):
            return

        # Clamp to data domain before passing to Datashader
        x0 = max(x0, self.heatmap_fig._ds_x_min)
        x1 = min(x1, self.heatmap_fig._ds_x_max)
        y0 = max(y0, self.heatmap_fig._ds_y_min)
        y1 = min(y1, self.heatmap_fig._ds_y_max)
        if x0 >= x1 or y0 >= y1:
            return

        try:
            pw = self.heatmap_fig.inner_width
            ph = self.heatmap_fig.inner_height
        except Exception:
            pw, ph = 600, 600

        new_agg = self.heatmap_fig._ds_rasterize(x0, x1, y0, y1, pw, ph)

        # Adjust color mapper to the local non-zero range for per-view contrast
        valid = new_agg[np.isfinite(new_agg) & (new_agg > 0)]
        if valid.size:
            self.heatmap_fig._ds_cmap.low = float(valid.min())
            self.heatmap_fig._ds_cmap.high = float(valid.max())

        # Replace image data, then update glyph position atomically to prevent drift
        self.heatmap_fig._ds_source.data = {"image": [new_agg]}
        glyph = self.heatmap_fig._ds_glyph.glyph
        glyph.x = x0
        glyph.y = y0
        glyph.dw = x1 - x0
        glyph.dh = y1 - y0

    def _rerasterize_from_frame(self, frame: "Frame"):
        # Replace the Datashader DataArray and rasterize closure for a new frame,
        # then immediately rerasterize at the current view. Called when a pixel tap
        # switches the heatmap to a different frame.
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

        # Update all pipeline references on the figure
        self.heatmap_fig._ds_rasterize = rasterize
        self.heatmap_fig._ds_x_min = x_min
        self.heatmap_fig._ds_x_max = x_max
        self.heatmap_fig._ds_y_min = y_min
        self.heatmap_fig._ds_y_max = y_max

        self._rerasterize()

    # ── Param watchers ────────────────────────────────────────────────────
    # These run after param.update() and are responsible for all plot mutations.

    @param.depends("selected_frame", watch=True)
    def _update_plots(self):
        # Triggered by pixel tap; rebuilds the heatmap and projections for the
        # newly selected frame
        if self.selected_frame < 0:
            return
        frame = self.dataset[self.selected_frame]
        self._current_df = frame.data
        self.heatmap_fig.title.text = f"Intensity heatmap — frame {self.selected_frame}"
        self._rerasterize_from_frame(frame)
        self._update_projections()

    @param.depends("x_start", "x_end", "y_start", "y_end", watch=True)
    def _update_projections(self):
        # Recompute the 1D spectrum and mobilogram from the current viewport
        # using the raw (sparse) frame data, not the rasterized image
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
        # Update the spatial ion image for the selected peak. Uses the
        # pre-computed intensity_array if available; otherwise extracts on-demand
        # from the raw data using the peak's m/z and 1/K₀ extents.
        if self.selected_peak < 0:
            return
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
        # Center the heatmap view on the selected peak's bounding box with 1.5×
        # padding, then rerasterize at the new viewport
        if self.selected_peak < 0:
            return
        peak_idx = self.selected_peak
        ext = self.peak_extents.loc[peak_idx]

        mz_min, mz_max = ext["mz_values", "min"], ext["mz_values", "max"]
        mob_min, mob_max = ext["mobility_values", "min"], ext["mobility_values", "max"]

        w, h = self.mean_spec.resolution
        padding = 1.5
        cx = 0.5 * (mz_min + mz_max)
        cy = 0.5 * (mob_min + mob_max)
        mz_half = (mz_max - mz_min + w) * padding * w / h
        mob_half = (mob_max - mob_min + h) * padding

        self.heatmap_fig.x_range.start = cx - mz_half
        self.heatmap_fig.x_range.end = cx + mz_half
        self.heatmap_fig.y_range.start = cy - mob_half
        self.heatmap_fig.y_range.end = cy + mob_half
        self._rerasterize()

    def _on_show_peaks(self, event):
        # Toggle peak bounding-box overlay visibility on the heatmap
        self.peak_boxes.visible = event.new

    def _on_show_grid(self, event):
        # Toggle heatmap axis grid line visibility
        self.heatmap_fig.xgrid.visible = event.new
        self.heatmap_fig.ygrid.visible = event.new

    # ── Button action callbacks ───────────────────────────────────────────

    def _show_tic(self, event=None):
        # Reset the ion image to the dataset-wide TIC
        intensities = self.dataset.tic()
        self.image_source.data["total_intensity"] = intensities
        self.image_source.data["normalized"] = intensities / intensities.max()
        self.image_fig.title.text = "TIC image"

    def _show_mean_spectrum(self, event=None):
        # Reset the heatmap to the mean spectrum and refresh all projections
        self._current_df = self.mean_spec.data
        self.heatmap_fig.title.text = "Intensity heatmap - mean spectrum"
        self._rerasterize_from_frame(self.mean_spec)
        self._update_projections()

    def _export_csv(self):
        # Return a StringIO buffer of the peak list CSV for Panel FileDownload
        from io import StringIO
        buf = StringIO()
        self.peak_list.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    def _export_intensity_array(self):
        # Return a StringIO buffer of the intensity array CSV for Panel FileDownload
        from io import StringIO
        buf = StringIO()
        self.intensity_array.to_csv(buf)
        buf.seek(0)
        return buf

    # ── Layout ────────────────────────────────────────────────────────────

    def view(self) -> pn.GridSpec:
        """Assemble and return the full dashboard layout as a ``pn.GridSpec``.

        Layout (16-column grid, 9 rows):

        - Rows 0–5, cols 0–7:   Ion image (``image_fig``)
        - Rows 0–5, cols 8–13:  Heatmap + mobilogram + spectrum gridplot
        - Rows 6–8, cols 0–7:   Peak-list table with intensity filter
        - Rows 0–8, cols 14–15: Control panel (sliders, toggles, buttons)

        :return: the assembled Panel GridSpec ready for ``pn.serve`` or
            notebook display via ``.servable()``
        :rtype: pn.GridSpec
        """
        # Heatmap, mobilogram, and spectrum share a toolbar via gridplot
        heatmap_grid = gridplot(
            [[self.heatmap_fig, self.mob_fig], [self.spec_fig, None]],
            toolbar_location="right",
        )

        controls = pn.WidgetBox(
            "## Controls",
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
        )

        gspec[0:6, 0:8] = pn.pane.Bokeh(self.image_fig, sizing_mode="stretch_width")
        gspec[0:6, 8:14] = pn.pane.Bokeh(heatmap_grid, sizing_mode="stretch_width")
        gspec[6:9, 0:8] = self.peak_table_widget
        gspec[0:9, 14:16] = controls

        return gspec