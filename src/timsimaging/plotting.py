import pandas as pd
from bokeh.document import Document
from bokeh.layouts import column, grid, row, layout, gridplot
from bokeh.models import (
    Button,
    Checkbox,
    ColumnDataSource,
    DataTable,
    Div,
    HoverTool,
    NumberFormatter,
    TableColumn,
    TapTool,
    WheelZoomTool,
    BoxSelectTool,
    CrosshairTool,
    ScaleBar,
    NumericInput,
    CustomJS,
    Metric,
    Range1d,
)
from bokeh.models.annotations.dimensional import CustomDimensional
from bokeh.plotting import figure
from bokeh.transform import log_cmap, linear_cmap
from typing import Callable, TYPE_CHECKING, Tuple, Literal

if TYPE_CHECKING:
    from .spectrum import Frame, MSIDataset

__all__ = ["image", "scatterplot", "heatmap", "mobilogram", "heatmap_layouts", "dashboard"]


def image(
    dataset: "MSIDataset",
    mz: slice = None,
    mobility: slice = None,
    i: int = None,
    results = None,
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
    elif results is not None and isinstance(i, int):
        intensities = results["intensity_array"].iloc[:, i]
        mz, mobility = results["peak_list"].iloc[i][["mz_values", "mobility_values"]]
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
        # tools="pan,wheel_zoom,box_select,tap,hover,save,reset",
        toolbar_location="right",
        x_axis_label="X",
        y_axis_label="Y",
    )
    f.toolbar.active_scroll = f.select_one({"type": WheelZoomTool})

    # (0,0) is on the top-left for ion images
    f.y_range.flipped = True

    # cmap = LinearColorMapper(palette="Viridis256", low = 0, high = tic.max())
    cmap = linear_cmap("normalized", palette="Viridis256", low=0, high=1)
    pixel_grid = f.rect(
        x="XIndexPos",
        y="YIndexPos",
        width=1,
        height=1,
        source=source,
        # line_alpha=0,
        # line_width=0,
        hover_line_alpha=1,
        hover_line_width=1,
        hover_line_color="gray",
        color=cmap,
    )
    pixel_grid.tags = ["image"]

    color_bar = pixel_grid.construct_color_bar()
    f.add_layout(color_bar, "right")

    units = CustomDimensional(
        basis={
            "px": (1 / dataset.resolution["xy"], "px", "pixel"),
            # "mm": (1000/ dataset.resolution["xy"], "mm", "millimeter"),
        },
        ticks=[1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 250, 500, 750],
    )

    scale_bar = ScaleBar(
        range=f.x_range,
        location="top_left",
        unit="px",
        # unit="µm",
        dimensional=Metric(base_unit="px"),
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
        title=r"m/z vs 1/K_0 heatmap",
        x_range=(frame.mz_domain.min(), frame.mz_domain.max()),
        y_range=(frame.mobility_domain.min(), frame.mobility_domain.max()),
        x_axis_label="m/z",
        y_axis_label="1/K_0",
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
            y_axis_label="1/K_0",
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
            x_axis_label="1/K_0",
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


def feature_list(data):
    source = ColumnDataSource(data)
    columns = [
        # TableColumn(
        #     field="index",
        #     title="#",
        #     ),
        TableColumn(
            field="mz_values",
            title="m/z",
            formatter=NumberFormatter(format="0.000"),
        ),
        TableColumn(
            field="mobility_values",
            title="1/K0",
            formatter=NumberFormatter(format="0.000"),
        ),
        TableColumn(
            field="total_intensity",
            title="total peak intensity",
            formatter=NumberFormatter(format="0.000"),
        ),
    ]
    filtered_source = ColumnDataSource(data.copy())
    table = DataTable(source=filtered_source, columns=columns, index_position=0)

    intensity_threshold = NumericInput(
        title="Relative intensity threshold",
        placeholder="0-100% to the max intensity",
        low=0,
        high=100,
        value=0,
        mode="float",
    )

    intensity_filter_callback = CustomJS(
        args=dict(source=source, filtered_source=filtered_source),
        code="""
        const data = source.data;
        const indices = data['index'];
        const mz = data['mz_values'];
        const mobility = data['mobility_values'];
        const intensity = data['total_intensity'];

        const percent = parseFloat(cb_obj.value);

        if (isNaN(percent) || percent < 0 || percent > 100) {
            return;
        }

        // Compute threshold
        const max = Math.max(...intensity);
        const threshold = (percent / 100) * max;

        const new_indices = [];
        const new_mz = [];
        const new_mobility = [];
        const new_intensity = [];

        for (let i = 0; i < intensity.length; i++) {
            if (intensity[i] >= threshold) {
                new_indices.push(indices[i]);
                new_mz.push(mz[i]);  
                new_mobility.push(mobility[i]);              
                new_intensity.push(intensity[i]);
            }
        }

        filtered_source.data = { 'index': new_indices, 'mz_values': new_mz, 'mobility_values': new_mobility, 'total_intensity': new_intensity };
        filtered_source.change.emit();
    """,
    )

    intensity_threshold.js_on_change("value", intensity_filter_callback)
    return column([intensity_threshold, table], aspect_ratio=1), filtered_source


# make a function a Bokeh application
def bkapp(func):
    def wrapper(*args, **kwargs):
        def app(doc: Document):
            ui = func(*args, **kwargs)
            doc.add_root(ui)

        return app

    return wrapper


@bkapp
def heatmap_layouts(
    frame: "Frame",
    peak_list: pd.DataFrame = None,
    peak_extents: pd.DataFrame = None,
    **kwargs,
):
    """The interactive visualization of a frame.
    2D spectogram, 1D projections along mz and mobility dimension and peak list

    :param doc: _description_
    :type doc: Document
    :param frame: the frame to visualize
    :type frame: Frame
    """
    df = frame.data
    width, height = frame.resolution
    if peak_list is None:
        peak_list, peak_extents = frame.peakPick(
            return_extents=True,
            **kwargs,
        ).values()

    heatmap_figure, heatmap_source = heatmap(frame)

    # show/hide grid lines
    def show_grid_line(attr, old, new):
        heatmap_figure.xgrid.visible = new
        heatmap_figure.ygrid.visible = new

    grid_checkbox = Checkbox(label="Show grid lines", active=True)
    grid_checkbox.on_change("active", show_grid_line)

    # plot peak boxes
    # convert extents to location and size
    peak_rect_data = pd.DataFrame()
    peak_rect_data["x"] = 0.5 * (
        peak_extents["mz_values", "min"] + peak_extents["mz_values", "max"]
    )
    peak_rect_data["y"] = 0.5 * (
        peak_extents["mobility_values", "min"] + peak_extents["mobility_values", "max"]
    )
    # if min=max, the width is 1 unit
    peak_rect_data["width"] = (
        peak_extents["mz_values", "max"] - peak_extents["mz_values", "min"] + width
    )
    peak_rect_data["height"] = (
        peak_extents["mobility_values", "max"] - peak_extents["mobility_values", "min"] + height
    )
    peak_rect_source = ColumnDataSource(peak_rect_data)
    peak_boxes = heatmap_figure.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        source=peak_rect_source,
        color="steelblue",
        fill_alpha=0,
        line_width=1.5,
    )
    peak_boxes.visible = False

    # checkbox
    def plot_peak_box(attr, old, new):
        peak_boxes.visible = new

    peak_checkbox = Checkbox(label="Show peaks", active=False)
    peak_checkbox.on_change("active", plot_peak_box)

    debug_div = Div(
        text="""
        x_range:
        y_range:
    """
    )
    # peak list table
    peak_table, peak_table_source = feature_list(peak_list)

    # callback for centering the view of a peak
    def look_into_peak(attr, old, new):
        select_idx = new[0]
        peak_idx = peak_table_source.data['index'][select_idx]
        x, y, w, h = peak_rect_data.loc[peak_idx]
        heatmap_figure.x_range.start = x - 0.5 * h * width / height * 1.5
        heatmap_figure.x_range.end = x + 0.5 * h * width / height * 1.5
        heatmap_figure.y_range.start = y - 0.5 * h * 1.5
        heatmap_figure.y_range.end = y + 0.5 * h * 1.5
        debug_div.text = f"selected index: {select_idx}, original index: {peak_idx}"

    # when one entry is selected, call look_into_peak
    peak_table_source.selected.on_change("indices", look_into_peak)

    # 1d projections
    spec1d_df = df.groupby("mz_values")["intensity_values"].sum().reset_index()
    spec1d_figure, spec1d_source = spectrum(spec1d_df)
    spec1d_figure.x_range = heatmap_figure.x_range
    spec1d_figure.height = heatmap_figure.height // 2

    mob_df = df.groupby("mobility_values")["intensity_values"].sum().reset_index()
    mob_figure, mob_source = mobilogram(mob_df, transposed=True)
    mob_figure.y_range = heatmap_figure.y_range
    mob_figure.width = heatmap_figure.width // 2

    # callback for 1D projecting current range
    def range_callback(attr, old, new):
        # current data
        df_view = df.loc[
            lambda x: (
                (x["mz_values"] > heatmap_figure.x_range.start)
                & (x["mz_values"] < heatmap_figure.x_range.end)
                & (x["mobility_values"] > heatmap_figure.y_range.start)
                & (x["mobility_values"] < heatmap_figure.y_range.end)
            ),
        ]

        spec1d_source.data = df_view.groupby("mz_values")["intensity_values"].sum().reset_index()

        mob_source.data = df_view.groupby("mobility_values")["intensity_values"].sum().reset_index()
        debug_div.text = f"""
                            <h3>x_range: {heatmap_figure.x_range.start: .4f}-{heatmap_figure.x_range.end: .4f}</h3>
                            <h3>y_range: {heatmap_figure.y_range.start: .4f}-{heatmap_figure.y_range.end: .4f}</h3>
                            <h3>2d data length: {df_view.shape[0]}</h3>
                            <h3>1d spec data length: {spec1d_df.shape[0]}</h3>
                            <h3>1d mob data length:{mob_df.shape[0]}</h3>
        """

    heatmap_figure.x_range.on_change("start", range_callback)
    heatmap_figure.x_range.on_change("end", range_callback)
    heatmap_figure.y_range.on_change("start", range_callback)
    heatmap_figure.y_range.on_change("end", range_callback)

    # axis_range_button = Button(label="show axis ranges")
    # axis_range_button.js_on_click(
    #     CustomJS(
    #         args=dict(f=f1),
    #         code="""
    #                                         console.log(f.x_range.start)
    #                                         console.log(f.x_range.end)""",
    #     )
    # )
    # peak_table_source.selected.js_on_change("indices",
    #                                       CustomJS(code='''
    #                                       const idx = cb_obj.indices[0];
    #                                       console.log(idx);
    #                                       '''))
    # peak_data = ColumnDataSource(self.peakPick())
    # data_table = DataTable(source=source, columns=columns, width=400, height=280)

    # heatmap_layouts = (
    #     [f, column([peak_checkbox, peak_table]), axis_range_button],
    #     sizing_mode="stretch_both",
    # )
    layouts = row(
        [
            gridplot(
                [
                    [heatmap_figure, mob_figure],
                    [spec1d_figure, None],
                ]
            ),
            column([peak_checkbox, grid_checkbox, peak_table, debug_div]),
        ],
        sizing_mode="scale_width",
    )
    return layouts


@bkapp
def dashboard(
    dataset: "MSIDataset",
    sampling_ratio=1.0,
    intensity_threshold=0.05,
    normalization="TIC",
    **kwargs,
):
    """The interactive visualization for a dataset

    :param doc:
    :type doc: Document
    :param dataset: the dataset to visualize
    :type dataset: MSIDataset
    """

    mean_spec = dataset.mean_spectrum(
        sampling_ratio=sampling_ratio, intensity_threshold=intensity_threshold
    )
    width, height = mean_spec.resolution

    peak_list, group_labels, peak_extents = mean_spec.peakPick(
        return_labels=True,
        return_extents=True,
        **kwargs,
    )

    # image(initially TIC)
    image_figure, image_source = image(dataset)
    # heatmap(initially mean spectrum)
    heatmap_figure, heatmap_source = heatmap(mean_spec)

    def show_tic():
        intensities = dataset.tic()
        image_source.data["total_intensity"] = intensities
        image_source.data["normalized"] = intensities / intensities.max()

    tic_button = Button(label="Show TIC image", button_type="primary")
    tic_button.on_click(show_tic)

    def show_meanspec():
        heatmap_source.data = mean_spec.data

    meanspec_button = Button(label="Show mean spectrum", button_type="primary")
    meanspec_button.on_click(show_meanspec)

    # show/hide grid lines
    def show_grid_line(attr, old, new):
        heatmap_figure.xgrid.visible = new
        heatmap_figure.ygrid.visible = new

    grid_checkbox = Checkbox(label="Show grid lines", active=True)
    grid_checkbox.on_change("active", show_grid_line)

    df = mean_spec.data
    spec1d_df = df.groupby("mz_values")["intensity_values"].sum().reset_index()
    spec1d_figure, spec1d_source = spectrum(spec1d_df)
    spec1d_figure.x_range = heatmap_figure.x_range
    spec1d_figure.yaxis.major_label_orientation = "vertical"

    mob_df = df.groupby("mobility_values")["intensity_values"].sum().reset_index()
    mob_figure, mob_source = mobilogram(mob_df, transposed=True)
    mob_figure.y_range = heatmap_figure.y_range

    # add taptool
    pixel_grid = image_figure.select(tags=["image"])[0]
    tap = TapTool(renderers=[pixel_grid])
    image_figure.add_tools(tap)
    tap_div = Div(text="Selected index")

    def tap_callback(attr, old, new):
        selected_indices = new

        if selected_indices:
            selected_index = selected_indices[0]
            df = dataset[selected_index].data
            heatmap_source.data = df
            spec1d_source.data = df.groupby("mz_values")["intensity_values"].sum().reset_index()
            mob_source.data = df.groupby("mobility_values")["intensity_values"].sum().reset_index()
            tap_div.text = f"Selected index: {selected_indices}"
        # only one pixel could be selected once
        if len(selected_indices) > 1:
            image_source.selected.indices = selected_indices[1:]

    image_source.selected.on_change("indices", tap_callback)

    # box select

    # peak list
    peak_rect_data = pd.DataFrame()
    peak_rect_data["x"] = 0.5 * (
        peak_extents["mz_values", "min"] + peak_extents["mz_values", "max"]
    )
    peak_rect_data["y"] = 0.5 * (
        peak_extents["mobility_values", "min"] + peak_extents["mobility_values", "max"]
    )
    # if min=max, the width is 1 unit
    peak_rect_data["width"] = (
        peak_extents["mz_values", "max"] - peak_extents["mz_values", "min"] + width
    )
    peak_rect_data["height"] = (
        peak_extents["mobility_values", "max"] - peak_extents["mobility_values", "min"] + height
    )

    peak_rect_source = ColumnDataSource(peak_rect_data)
    peak_boxes = heatmap_figure.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        source=peak_rect_source,
        color="steelblue",
        fill_alpha=0,
        line_width=1.5,
    )
    peak_boxes.visible = False

    # checkbox
    def plot_peak_box(attr, old, new):
        peak_boxes.visible = new

    peak_checkbox = Checkbox(label="Show peaks", active=False)
    peak_checkbox.on_change("active", plot_peak_box)

    # peak table
    peak_table, peak_table_source = feature_list(peak_list)
    def look_into_peak(attr, old, new):
        select_idx = new[0]
        peak_idx = peak_table_source.data['index'][select_idx]
        x, y, w, h = peak_rect_data.loc[peak_idx]
        heatmap_figure.x_range.start = x - 0.5 * h * width / height * 1.5
        heatmap_figure.x_range.end = x + 0.5 * h * width / height * 1.5
        heatmap_figure.y_range.start = y - 0.5 * h * 1.5
        heatmap_figure.y_range.end = y + 0.5 * h * 1.5
        # peak info
        mz, mobility = peak_list.loc[peak_idx][["mz_values", "mobility_values"]]
        mz_min, mz_max, mob_min, mob_max = peak_extents.loc[peak_idx][
            ["mz_values", "mobility_values"]
        ]

        # pixel-wise intensity
        # image_data = dataset.data[:, mob_min:mob_max, 0, mz_min:mz_max]
        # peak_intensities = image_data.groupby("frame_indices")["intensity_values"].sum()
        indices = dataset.data[:, mob_min:mob_max, 0, mz_min:mz_max, "raw"]
        peak_intensities = dataset.data.bin_intensities(indices, axis=["rt_values"])[1:]

        image_source.data["total_intensity"] = peak_intensities
        if normalization == "TIC":
            image_source.data["normalized"] = peak_intensities / peak_intensities.max()
        elif normalization == "RMS":
            image_source.data["normalized"] = peak_intensities / peak_intensities.std()
        image_figure.title.text = f"MS Image m/z: {mz:.4f} 1/K_0: {mobility:.3f}"

    # peak_table_source = ColumnDataSource(peak_list)
    # columns = [
    #     TableColumn(
    #         field="mz_values",
    #         title="m/z",
    #         formatter=NumberFormatter(format="0.000"),
    #     ),
    #     TableColumn(
    #         field="mobility_values",
    #         title="1/K0",
    #         formatter=NumberFormatter(format="0.000"),
    #     ),
    #     TableColumn(
    #         field="total_intensity",
    #         title="total peak intensity",
    #         formatter=NumberFormatter(format="0.000"),
    #     ),
    #     # TableColumn(
    #     #     field="ccs_values",
    #     #     title="collision cross section(CCS)",
    #     #     formatter=NumberFormatter(format="0.000"),
    #     # ),
    # ]

    # when one entry is selected, call look_into_peak
    peak_table_source.selected.on_change("indices", look_into_peak)

    def range_callback(attr, old, new):
        # current data
        df_view = df.loc[
            lambda x: (
                (x["mz_values"] > heatmap_figure.x_range.start)
                & (x["mz_values"] < heatmap_figure.x_range.end)
                & (x["mobility_values"] > heatmap_figure.y_range.start)
                & (x["mobility_values"] < heatmap_figure.y_range.end)
            ),
        ]

        spec1d_source.data = df_view.groupby("mz_values")["intensity_values"].sum().reset_index()

        mob_source.data = df_view.groupby("mobility_values")["intensity_values"].sum().reset_index()

    heatmap_figure.x_range.on_change("start", range_callback)
    heatmap_figure.x_range.on_change("end", range_callback)
    heatmap_figure.y_range.on_change("start", range_callback)
    heatmap_figure.y_range.on_change("end", range_callback)

    # compute CCS
    ccs_button = Button(label="Compute CCS value", button_type="success")

    def compute_ccs():
        # already computed
        if "ccs_values" in peak_table_source.data:
            return
        else:
            calibrator = dataset.ccs_calibrator()
            ccs_values = calibrator.transform(
                peak_list["mz_values"], peak_list["mobility_values"], charge=1
            )
            peak_list["ccs_values"] = ccs_values
            peak_table_source.data["ccs_values"] = ccs_values
            peak_table.columns.append(
                TableColumn(
                    field="ccs_values",
                    title="CCS",
                    formatter=NumberFormatter(format="0.000"),
                ),
            )

    ccs_button.on_click(compute_ccs)

    # export peak list
    # TODO add more format options
    export_button = Button(label="Export peak list", button_type="success")

    def export_csv():
        csv_data = peak_list.to_csv()
        # csv_data = csv_data.replace("\n", "\\n").replace("\r", "\\r").replace('"', '\\"')
        filename = "peak_list.csv"
        export_button.js_on_click(
            CustomJS(
                args=dict(csv_data=csv_data, filename=filename),
                code="""
            const filetext = csv_data;
    
            const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });
    
            //addresses IE
            if (navigator.msSaveBlob) {
                navigator.msSaveBlob(blob, filename);
            } else {
                const link = document.createElement('a');
                if (link.download !== undefined) {
                    const url = URL.createObjectURL(blob);
                    link.setAttribute('href', url);
                    link.setAttribute('download', filename);
                    link.style.visibility = 'hidden';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }
            }
        """,
            )
        )

    # Assign the Python callback to the button
    export_button.on_click(export_csv)

    layouts = grid(
        [
            [image_figure, heatmap_figure, mob_figure],
            [
                peak_table,
                spec1d_figure,
                column(
                    peak_checkbox,
                    grid_checkbox,
                    tic_button,
                    meanspec_button,
                    ccs_button,
                    export_button,
                ),
            ],
        ],
        # sizing_mode="fixed",
    )
    # layouts = row(
    #     [
    #         column(image_figure, peak_table),
    #         column([heatmap_figure, spec1d_figure], sizing_mode="stretch_width"),
    #         column(mob_figure, peak_checkbox, button),
    #     ],
    # )
    # layouts = row([image_figure, heatmap_figure, column([peak_table, button])])
    return layouts


# sizing_mode="stretch_width"


# wrapper functions for Jupyter Notebook use, because bokeh.io.show() accepcts callable with only one doc arg
def plot(frame: "Frame", **kwargs) -> Callable[[Document], None]:
    """Plot a dashboard for a frame
    show(plot(<frame>, <params>), notebook_url=<...>)
    Any keyword arguments would be passed to peakPick()

    :param frame: the frame to visualize
    :type frame: Frame
    :return: a closure function
    :rtype: Callable[[Document], None]
    """
    return lambda doc: (heatmap_layouts(doc, frame=frame, **kwargs))


@bkapp
def _visualize(
    dataset: "MSIDataset",
    mean_spectrum: "Frame",
    peak_list: pd.DataFrame,
    peak_extents: pd.DataFrame,
):
    """A helper function to build a visualization App from processed data

    :param dataset: the original dataset, used for metadata and dynamic viz
    :type dataset: MSIDataset
    :param mean_spectrum: the mean spectrum, used for 2D heatmap
    :type mean_spectrum: Frame
    :param peak_list: mz, mobility center and intensity for each peak
    :type peak_list: pd.DataFrame
    :param peak_extents: mz and mobility range for each peak
    :type peak_extents: pd.DataFrame
    :return: a Bokeh layout, further wrapped into an App by the decorator
    :rtype: _type_
    """

    width, height = mean_spectrum.resolution

    # image(initially TIC)
    image_figure, image_source = image(dataset)
    # heatmap(initially mean spectrum)
    heatmap_figure, heatmap_source = heatmap(mean_spectrum)

    def show_tic():
        intensities = dataset.tic()
        image_source.data["total_intensity"] = intensities
        image_source.data["normalized"] = intensities / intensities.max()

    tic_button = Button(label="Show TIC image", button_type="primary")
    tic_button.on_click(show_tic)

    def show_meanspec():
        heatmap_source.data = mean_spectrum.data

    meanspec_button = Button(label="Show mean spectrum", button_type="primary")
    meanspec_button.on_click(show_meanspec)

    # show/hide grid lines
    def show_grid_line(attr, old, new):
        heatmap_figure.xgrid.visible = new
        heatmap_figure.ygrid.visible = new

    grid_checkbox = Checkbox(label="Show grid lines", active=True)
    grid_checkbox.on_change("active", show_grid_line)

    # spectrum and mobilogram
    df = mean_spectrum.data
    spec1d_df = df.groupby("mz_values")["intensity_values"].sum().reset_index()
    spec1d_figure, spec1d_source = spectrum(spec1d_df)
    spec1d_figure.x_range = heatmap_figure.x_range
    spec1d_figure.yaxis.major_label_orientation = "vertical"

    mob_df = df.groupby("mobility_values")["intensity_values"].sum().reset_index()
    mob_figure, mob_source = mobilogram(mob_df, transposed=True)
    mob_figure.y_range = heatmap_figure.y_range

    # add taptool
    pixel_grid = image_figure.select(tags=["image"])[0]
    tap = TapTool(renderers=[pixel_grid], mode="replace")
    image_figure.add_tools(tap)
    tap_div = Div(text="Selected index")

    def tap_callback(attr, old, new):
        selected_indices = new

        if selected_indices:
            selected_index = selected_indices[0]
            df = dataset[selected_index].data
            heatmap_source.data = df
            spec1d_source.data = df.groupby("mz_values")["intensity_values"].sum().reset_index()
            mob_source.data = df.groupby("mobility_values")["intensity_values"].sum().reset_index()
            tap_div.text = f"Selected index: {selected_indices}"
        # only one pixel could be selected once
        if len(selected_indices) > 1:
            image_source.selected.indices = selected_indices[1:]

    image_source.selected.on_change("indices", tap_callback)

    # add box selection tool(for future ROI selection)
    box_select = BoxSelectTool(renderers=[pixel_grid])
    image_figure.add_tools(box_select)
    box_select_div = Div(text="Selected index")

    def box_select_callback(attr, old, new):
        selected_indices = new
        if selected_indices:
            box_select_div.text = f"{len(selected_indices)} pixels selected"

    image_source.selected.on_change("indices", box_select_callback)

    # peak centering
    peak_rect_data = pd.DataFrame()
    peak_rect_data["x"] = 0.5 * (
        peak_extents["mz_values", "min"] + peak_extents["mz_values", "max"]
    )
    peak_rect_data["y"] = 0.5 * (
        peak_extents["mobility_values", "min"] + peak_extents["mobility_values", "max"]
    )
    # if min=max, the width is 1 unit
    peak_rect_data["width"] = (
        peak_extents["mz_values", "max"] - peak_extents["mz_values", "min"] + width
    )
    peak_rect_data["height"] = (
        peak_extents["mobility_values", "max"] - peak_extents["mobility_values", "min"] + height
    )

    peak_rect_source = ColumnDataSource(peak_rect_data)
    peak_boxes = heatmap_figure.rect(
        x="x",
        y="y",
        width="width",
        height="height",
        source=peak_rect_source,
        color="steelblue",
        fill_alpha=0,
        line_width=1.5,
    )
    peak_boxes.visible = False

    # checkbox
    def plot_peak_box(attr, old, new):
        peak_boxes.visible = new

    peak_checkbox = Checkbox(label="Show peaks", active=False)
    peak_checkbox.on_change("active", plot_peak_box)

    # peak list
    peak_table, peak_table_source = feature_list(peak_list)

    def look_into_peak(attr, old, new):
        select_idx = new[0]
        peak_idx = peak_table_source.data['index'][select_idx]
        x, y, w, h = peak_rect_data.loc[peak_idx]
        heatmap_figure.x_range.start = x - 0.5 * h * width / height * 1.5
        heatmap_figure.x_range.end = x + 0.5 * h * width / height * 1.5
        heatmap_figure.y_range.start = y - 0.5 * h * 1.5
        heatmap_figure.y_range.end = y + 0.5 * h * 1.5
        # peak info
        mz, mobility = peak_list.loc[peak_idx][["mz_values", "mobility_values"]]
        mz_min, mz_max, mob_min, mob_max = peak_extents.loc[peak_idx][
            ["mz_values", "mobility_values"]
        ]

        # pixel-wise intensity
        # image_data = dataset.data[:, mob_min:mob_max, 0, mz_min:mz_max]
        # peak_intensities = image_data.groupby("frame_indices")["intensity_values"].sum()

        indices = dataset.data[:, mob_min:mob_max, 0, mz_min:mz_max, "raw"]
        peak_intensities = dataset.data.bin_intensities(indices, axis=["rt_values"])[1:]
        image_source.data["total_intensity"] = peak_intensities
        image_source.data["normalized"] = peak_intensities / peak_intensities.max()
        image_figure.title.text = f"MS Image m/z: {mz:.4f} 1/K_0: {mobility:.3f}"

    peak_table_source.selected.on_change("indices", look_into_peak)

    # dynamic zooming
    def range_callback(attr, old, new):
        # current data
        df_view = df.loc[
            lambda x: (
                (x["mz_values"] > heatmap_figure.x_range.start)
                & (x["mz_values"] < heatmap_figure.x_range.end)
                & (x["mobility_values"] > heatmap_figure.y_range.start)
                & (x["mobility_values"] < heatmap_figure.y_range.end)
            ),
        ]

        spec1d_source.data = df_view.groupby("mz_values")["intensity_values"].sum().reset_index()

        mob_source.data = df_view.groupby("mobility_values")["intensity_values"].sum().reset_index()

    heatmap_figure.x_range.on_change("start", range_callback)
    heatmap_figure.x_range.on_change("end", range_callback)
    heatmap_figure.y_range.on_change("start", range_callback)
    heatmap_figure.y_range.on_change("end", range_callback)

    # compute CCS(to be changed to always compute)
    ccs_button = Button(label="Compute CCS value", button_type="success")

    def compute_ccs():
        # already computed
        if "ccs_values" in peak_table_source.data:
            return
        else:
            calibrator = dataset.ccs_calibrator()
            ccs_values = calibrator.transform(
                peak_list["mz_values"], peak_list["mobility_values"], charge=1
            )
            peak_list["ccs_values"] = ccs_values
            peak_table_source.data["ccs_values"] = ccs_values
            peak_table.columns.append(
                TableColumn(
                    field="ccs_values",
                    title="CCS",
                    formatter=NumberFormatter(format="0.000"),
                ),
            )

    ccs_button.on_click(compute_ccs)

    # export peak list
    # TODO add more format options
    export_button = Button(label="Export peak list", button_type="success")

    def export_csv():
        csv_data = peak_list.to_csv()
        # csv_data = csv_data.replace("\n", "\\n").replace("\r", "\\r").replace('"', '\\"')
        filename = "peak_list.csv"
        export_button.js_on_click(
            CustomJS(
                args=dict(csv_data=csv_data, filename=filename),
                code="""
            const filetext = csv_data;
    
            const blob = new Blob([filetext], { type: 'text/csv;charset=utf-8;' });
    
            //addresses IE
            if (navigator.msSaveBlob) {
                navigator.msSaveBlob(blob, filename);
            } else {
                const link = document.createElement('a');
                if (link.download !== undefined) {
                    const url = URL.createObjectURL(blob);
                    link.setAttribute('href', url);
                    link.setAttribute('download', filename);
                    link.style.visibility = 'hidden';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }
            }
        """,
            )
        )

    # Assign the Python callback to the button
    export_button.on_click(export_csv)

    layouts = layout(
        [
            [image_figure, heatmap_figure, mob_figure],
            [
                peak_table,
                spec1d_figure,
                column(
                    peak_checkbox,
                    grid_checkbox,
                    tic_button,
                    meanspec_button,
                    ccs_button,
                    export_button,
                    box_select_div,
                ),
            ],
        ],
        sizing_mode="fixed",
    )
    return layouts
