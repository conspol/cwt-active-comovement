import itertools

import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table as dt
from dash import html, dcc
from dash.dash_table.Format import Format, Scheme
from scipy.stats import ttest_rel, mannwhitneyu

from dataprep import *
from utils import get_color

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

COLOR_flight = 'rgb(0,66,10)'

set2_cmap = px.colors.qualitative.Set2
set3_cmap = px.colors.qualitative.Set3
dark2_cmap = px.colors.qualitative.Dark2
plotly_cmap = px.colors.qualitative.Plotly
spectral_cmap = px.colors.diverging.Spectral
brbg_cmap = px.colors.diverging.BrBG
pastel_cmap = px.colors.qualitative.Pastel
d3_cmap = px.colors.qualitative.D3
rdylgn_cmap = px.colors.diverging.RdYlGn
gray_cmap = px.colors.sequential.gray
plotly3_cmap = px.colors.sequential.Plotly3

grey_colors = {
    0: get_color(gray_cmap, 0.5),
    1: get_color(gray_cmap, 0.6),
    2: get_color(gray_cmap, 0.8),
    3: get_color(gray_cmap, 0.9),
    4: get_color(gray_cmap, 0.99),
}

distribution_colors = {
    'Log-normal': set3_cmap[3],
    'Stretched exponential': dark2_cmap[4],
    'Exponential': set3_cmap[6],
    'Power law': set3_cmap[9],
    'Truncated power law': set3_cmap[4],
}

celltype_colors = {
    'mef': get_color(spectral_cmap, 0.93),
    'mcf10a': get_color(rdylgn_cmap, 0.9),
    'ht': get_color(brbg_cmap, 0.15),
    'mda231': get_color(spectral_cmap, 0.1),
    'mcf7': get_color(spectral_cmap, 0.31),
}

comov_colors = {
    # 'nonp': get_color(plotly3_cmap, 1),
    # '80-20': get_color(plotly3_cmap, 0.66),
    # 'tmanp': get_color(plotly3_cmap, 0.35),
    # 'nonp': get_color(plotly3_cmap, 1),
    # '80-20': {NOCOMOV_SUFFIX: '#fdffc9', COMOV_SUFFIX: '#bebfa4'},
    # 'tmanp': {NOCOMOV_SUFFIX: '#f3c9ff', COMOV_SUFFIX: '#bfa1c7'},
    # '80-20': {NOCOMOV_SUFFIX: grey_colors[4], COMOV_SUFFIX: grey_colors[2]},
    # 'tmanp': {NOCOMOV_SUFFIX: grey_colors[4], COMOV_SUFFIX: grey_colors[2]},
    '80-20': 'white',
    'tmanp': get_color(gray_cmap, 0.7),
}

COLOR_Pastel_cycle = itertools.cycle(px.colors.qualitative.Pastel)


def get_colorcycle(pallette):
    return itertools.cycle(getattr(px.colors.qualitative, pallette))


def rgb2rgba_str(rgb_str, alpha='0.7'):
    rgb = rgb_str.lstrip('rgb(').rstrip(')').split(',')
    return 'rgba({}, {}, {}, {})'.format(rgb[0], rgb[1], rgb[2], alpha)


comov_mark_color = {
    'msd': rgb2rgba_str(pastel_cmap[0], '0.5'),
    'msd' + COMOV_SUFFIX: rgb2rgba_str(pastel_cmap[2], '0.5'),
}

comov_mark_symbol = {
    'a': 'circle-dot',
    'd': 'cross-dot',
}

comov_markline_color = {
    'msd': d3_cmap[9],
    'msd' + COMOV_SUFFIX: d3_cmap[1],
}


def get_markline_color(value, maxvalue=5):
    if value > maxvalue:
        maxvalue = value
    v = value / maxvalue
    return get_color(spectral_cmap, v)


def dd_graph(
        name,
        cells,
        graphname='summary',
        divstyle=None,
):
    if divstyle is None:
        divstyle = {}
    id_name = name + '-' + graphname

    but_conf = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': id_name,
            # 'scale': 1,
        }
    }
    return html.Div(
        [
            dcc.Dropdown(
                id=name + '-celltypes-dd',
                options=[
                    {'label': c_, 'value': c_}
                    for c_ in cells.keys()
                ],
                value=[c_ for c_ in cells.keys()],
                multi=True,
                style={
                    'marginLeft': '10px',
                    'marginRight': '10px',
                }
            ),
            dcc.Dropdown(
                id=name + '-treatment-dd',
                multi=True,
                style={
                    'marginLeft': '10px',
                    'marginRight': '10px',
                }
            ),
            dcc.Graph(id=id_name, config=but_conf),
        ],
        style={
            'display': 'inline-block',
            **divstyle,
            # 'width': '500px',
            # 'height': '1300px',
        }
    )


def dd_graph2(
        name,
        cells,
        graphname='summary',
):
    id_name = name + '-' + graphname + '2'
    but_conf = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': id_name,
            # 'scale': 1,
        }
    }

    ret = dd_graph(name, cells, graphname)
    ret.children.append(
        dcc.Graph(
            id=id_name,
            config=but_conf,
        ),
    )

    return ret


def dd_single_or_sum_radio(
        name,
        cells,
        graphname='summary',
):
    ret = dd_graph(name, cells, graphname)
    ret.children.insert(
        -1,
        html.Div([
            dcc.RadioItems(
                id=name + '-radio',
                options=[
                    {'label': 'All cells', 'value': 'sum'},
                    {'label': 'Separate cells', 'value': 'single'},
                ],
                value='sum',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                inputStyle={'margin-right': '7.5px'},
            )
        ])
    )

    return ret


def populate_treatments_base(
        cells,
        celltypes: List,
):
    if type(celltypes) == str:
        celltypes = [celltypes]

    lret = []
    unique_treatments = set()

    for ct_ in celltypes:
        for tr_ in cells[ct_].keys():
            if not (tr_ in unique_treatments):
                lret.append({'label': tr_, 'value': tr_})
                unique_treatments.add(tr_)

    return lret


def populate_cell_ids_base(
        cells,
        celltype: str,
        treatment: str,
):
    if type(celltype) == list:
        celltype = celltype[0]

    lret = []

    for s_ in cells[celltype][treatment]['samples']:
        lret.append({'label': s_['id'], 'value': s_['id']})

    return lret


def noplot_msg(text, size=20):
    return {
        "layout": {
            "xaxis": {
                "visible": False
            },
            "yaxis": {
                "visible": False
            },
            "annotations": [
                {
                    "text": text,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": size
                    }
                }
            ]
        }
    }


def make_slider_step(istep):
    slider_step = {
        'args': [
            [istep],
            {
                'frame': {"duration": 100, 'redraw': False},
                'mode': 'immediate',
                'transition': {'duration': 50}
            }
        ],
        'label': istep,
        'method': 'animate',
    }

    return slider_step


def update_singlecell_cdf_base(
        cells,
        celltypes: List,
        treatments: List,
        data_type: str,
):
    lret = list()
    if type(celltypes) == str:
        celltypes = [celltypes]

    if type(treatments) == str:
        treatments = [treatments]

    df_cdf, df_perst = get_singlecell_cdf_df(
        cells,
        celltypes,
        treatments,
        data_type=data_type,
    )

    grp_cdf = df_cdf.groupby(level=[0, 1, 2])

    for i_, g_ in grp_cdf:
        lret.append({
            'x': g_[0],
            'y': g_[1],
            'type': 'scatter',
            'name': str(i_),
            'loglog': True,
        })

    return {'data': lret,
            'layout': {
                'title': 'CDF',
                'xaxis': {'title': 'Time (s)', 'type': 'log'},
                'yaxis': {'title': 'CDF', 'type': 'log'},
            }}


def update_summary_cdf_base(
        cells,
        celltypes: List,
        treatments: List,
        data_type: str,
):
    lret = list()
    if type(celltypes) == str:
        celltypes = [celltypes]

    if type(treatments) == str:
        treatments = [treatments]

    df_cdf = get_summary_cdf_df(
        cells,
        celltypes,
        treatments,
        data_type=data_type,
    )

    grp_cdf = df_cdf.groupby(level=[0, 1])

    for i_, g_ in grp_cdf:
        lret.append({
            'x': g_[0],
            'y': g_[1],
            'type': 'scatter',
            'name': str(i_),
            'loglog': True,
        })

    titles = {
        'dir_t': 'Time (s)',
        'dir_l': 'Length (um)',
    }

    return {'data': lret,
            'layout': {
                'title': 'CDF',
                'xaxis': {'title': titles[data_type], 'type': 'log'},
                'yaxis': {'title': 'CDF', 'type': 'log'},
            }}


def update_act_sum_cdf_base(
        cells,
        celltypes: List,
        treatments: List,
        data_type: str,
):
    lret = list()
    if type(celltypes) == str:
        celltypes = [celltypes]

    if type(treatments) == str:
        treatments = [treatments]

    df_cdf, data_dict = get_act_cdf_df(
        cells,
        celltypes,
        treatments,
        data_type=data_type,
        from_summary=True,
    )

    grp_cdf = df_cdf.groupby(level=[0, 1])

    for i_, g_ in grp_cdf:
        dline = {
            # Celltype-specific color
            'color': celltype_colors[i_[0]],
        }
        linemode = 'lines'

        if i_[1] == 'nonp':
            # dline['dash'] = 'dash'
            dline['width'] = 2

        elif COMOV_SUFFIX in i_[1]:
            linemode = 'lines'
            dline['dash'] = 'dot'
            dline['width'] = 2

        if '_t' in data_type:
            xscale = FPS
        else:
            xscale = 1

        legend_name = f'{celltype_names[i_[0]]} {treatment_names_more[i_[1]]}'

        lret.append({
            'x': g_[0] / xscale,
            'y': g_[1],
            'type': 'scatter',
            'line': dline,
            'name': legend_name,
            'mode': linemode,
        })

    titles = {
        'active_t': 'Run time CCDF',
        'active_l': 'Run length CCDF',
        'run_t': 'Run time CCDF',
        'run_l': 'Run length CCDF',
        'flight_t': 'Flight time CCDF',
        'flight_l': 'Flight length CCDF',
    }

    xtitles = {
        'active_t': 'Time (s)',
        'active_l': 'Length (um)',
        'run_t': 'Time (s)',
        'run_l': 'Length (um)',
        'flight_t': 'Time (s)',
        'flight_l': 'Length (um)',
    }

    figdict = {
        'data': lret,
        'layout': {
            'font': {
                'family': 'Arial',
                'size': 16,
            },
            'legend': dict(
                # orientation='h',
                bgcolor='rgba(0,0,0,0)',
                yanchor='bottom',
                xanchor='left',
                y=0.03,
                x=0.03,
            ),
            'title': titles[data_type],
            'xaxis': {
                'title': xtitles[data_type],
                'tickformat': 'f',
                # 'dtick': 1,
                # 'type': 'log',
                'showline': True,
                # 'range': xranges[data_type],
                'range': [1, 6],
            },
            'yaxis': {
                'title': 'CCDF',
                'tickformat': 'f',
                # 'dtick': 1,
                # 'type': 'log',
                'showline': True,
                'range': [0, 0.15],
            },
        }
    }

    return figdict


def highlight_max_value_row(
        df,
        bgcolor='#42BFF5',
        textcolor='black',
        styles=None,
):
    if styles is None:
        styles = []

    numeric_columns = df.select_dtypes('number')

    for i, col in enumerate(numeric_columns.drop(['id'], axis=1).idxmax(axis=1)):
        styles.append({
            'if': {
                'filter_query': '{{id}} = {}'.format(i),
                'column_id': col,
            },
            'backgroundColor': bgcolor,
            'color': textcolor,
        })

    return styles


def shade_nonnumeric_cols(
        df,
        bgcolor='#EEEEEE',
        textcolor='black',
        styles=None,
):
    if styles is None:
        styles = []

    nonnumeric_columns = df.select_dtypes('O')

    for col in nonnumeric_columns.columns:
        styles.append({
            'if': {
                'column_id': col,
            },
            'backgroundColor': bgcolor,
            'color': textcolor,
        })

    return styles


def dash_all_aic_table(
        df,
        element_id='waic-table',
        tblkwgs: dict = None,
):
    if tblkwgs is None:
        tblkwgs = {}

    df = prepare_fits_waic_df_for_table(df)

    columns = []
    num_cols = df.select_dtypes('number').columns

    for dfcol in df.columns:
        if dfcol != 'id':
            tcol = {'name': dfcol, 'id': dfcol}

            if dfcol in num_cols:
                tcol.update({
                    'type': 'numeric',
                    'format': Format(precision=2, scheme=Scheme.decimal_or_exponent),
                })

            columns.append(tcol)

    styles = highlight_max_value_row(df)
    shade_nonnumeric_cols(df, styles=styles)

    # styles.append({
    #     'if': {'state': 'selected'},
    #     'backgroundColor': 'inherit !important',
    # })

    return dt.DataTable(
        id=element_id,
        data=df.to_dict('records'),
        columns=columns,
        style_data_conditional=styles,
        style_cell={'fontSize': 14, 'font-family': 'sans-serif'},
        # style_table={'width': '100rem'},
        style_header={
            'fontWeight': 'bold',
            'whiteSpace': 'normal',
            'heiht': 'auto',
        },
        **tblkwgs
    )


def multiple_aic_tables_by_datatype(
        dfs_dict: Dict,
        element_id_prefix='waic-table-datatype',
        dtypes2show: List = None,
):
    tables = []
    for dt_, df_ in dfs_dict.items():
        if dtypes2show and dt_ in dtypes2show:
            tbl_ = dash_all_aic_table(
                df_,
                element_id=element_id_prefix + '-' + dt_,
                tblkwgs={
                    'sort_action': 'native',
                    'sort_mode': 'multi',
                    'filter_action': 'native',
                }
            )
            div_ = html.Div([
                html.H3(datatype_names[dt_]),
                tbl_,
            ],
                style={
                    'width': '60em',
                    'display': 'inline-block',
                    'padding': '2em',
                    'margin': 'auto',
                }
            )
            tables.append(div_)

    return html.Div(
        tables,
        style={
            'display': 'inline-block',
            # 'flex-direction': 'row'
        },
    )


def plot_fits_ccdf(
        sample_idx,
        fits,
        element_id='fits-ccdf-plot',
        mainfit=None,
):
    """
    Plot fits from the powerlaw fit object.
    """
    fitobj = fits['fitobj']
    xs, data = fitobj.ccdf()
    mainfitkey = distr_names_reverse[mainfit]

    if fitobj.xmax is not None:
        xmax = fitobj.xmax
    else:
        xmax = np.max(xs)

    bins = np.linspace(fitobj.xmin, xmax, 115)

    fig = go.Figure()

    colors = get_colorcycle('Set1')

    for i_, (distr_, distrname_) in enumerate(distr_names.items()):
        distrobj_ = getattr(fitobj, distr_)
        ccdf_ = distrobj_.ccdf(bins)

        if distrname_ in distribution_colors:
            line = {'color': distribution_colors[distrname_]}
        else:
            line = {'color': next(colors)}

        if distrname_ == mainfit:
            line.update({'width': 3})
        else:
            line.update({
                'dash': 'dash',
                'width': 3,
            })

        fig.add_trace(go.Scatter(
            x=bins,
            y=ccdf_,
            name=distrname_,
            line=line,
        ))

    fig.add_trace(go.Scatter(
        x=xs,
        y=data,
        mode='lines',
        name='Data',
        line=dict(color='#9F1D35', width=4, dash='dot'),
    ))

    xtitles = {
        'active_t': 'Time (s)',
        'active_l': 'Length (um)',
        'run_t': 'Time (s)',
        'run_l': 'Length (um)',
        'flight_t': 'Time (s)',
        'flight_l': 'Length (um)',
    }
    xtitles.update({
        k+COMOV_SUFFIX: v for k, v in xtitles.items()
    })

    xranges = {
        'active_t': None,
        'active_l': None,
        'run_t': None,
        'run_l': np.log10([0.2, 20]),
        'flight_t': None,
        'flight_l': np.log10([0.2, 20]),
    }
    xranges.update({
        k+COMOV_SUFFIX: v for k, v in xranges.items()
    })
    yranges = {
        'active_t': None,
        'active_l': None,
        'run_t': None,
        'run_l': [-4, 0],
        'flight_t': None,
        'flight_l': [-4, 0],
    }
    yranges.update({
        k+COMOV_SUFFIX: v for k, v in xranges.items()
    })

    fig.update_layout({
        'title': f'{sample_idx2name(*sample_idx)}, <br>'
                 f' bestfit params: {fits["distr"][mainfitkey]["params"]}',
        'xaxis': {
            'title': xtitles[sample_idx[0]],
            'tickformat': 'f',
            'dtick': 1,
            'showline': True,
            'type': 'log',
            'range': xranges[sample_idx[0]],
        },
        'yaxis': {
            'title': 'CCDF',
            'tickformat': 'f',
            'dtick': 1,
            'showline': True,
            'type': 'log',
            'range': yranges[sample_idx[0]],
        },
    })

    return fig


def dash_active_t_ratio_table(
        df,
        element_id='act-t-ratio-table',
        tblkwgs: dict = None,
):
    if tblkwgs is None:
        tblkwgs = {}

    df = prepare_act_t_df4table(df)

    columns = []
    num_cols = df.select_dtypes('number').columns

    for dfcol in df.columns:
        if dfcol != 'id':
            tcol = {'name': dfcol, 'id': dfcol}

            if dfcol in active_t_ratio_names_more_reverse and \
                    'ratio' in active_t_ratio_names_more_reverse[dfcol]:

                tcol.update({
                    'type': 'numeric',
                    'format': Format(precision=2, scheme=Scheme.percentage),
                })

            elif dfcol in num_cols:
                tcol.update({
                    'type': 'numeric',
                })

            columns.append(tcol)

    styles = shade_nonnumeric_cols(df)

    # styles.append({
    #     'if': {'state': 'selected'},
    #     'backgroundColor': 'inherit !important',
    # })

    return dt.DataTable(
        id=element_id,
        data=df.to_dict('records'),
        columns=columns,
        style_data_conditional=styles,
        style_cell={'fontSize': 14, 'font-family': 'sans-serif'},
        # style_table={'width': '100rem'},
        style_header={
            'fontWeight': 'bold',
            'whiteSpace': 'normal',
            'heiht': 'auto',
        },
        **tblkwgs
    )


def plot_msdfits_scatter(
        cells,
):
    # fig = make_subplots(specs=[[{"secondary_x": True}]])
    fig = go.Figure()
    fig.update_layout(
        yaxis={
            'showgrid': True,
        },
        xaxis={
            'title': r'$\alpha$',
            'showgrid': True,
            'range': [1, 1.4],
        },
        xaxis2={
            'title': '$D$',
            'anchor': 'y',
            'overlaying': 'x',
            'side': 'top',
        })

    all_vals = {
        msdkey_: {
            v_:
                {
                    'names': [],
                    'vals': [],
                    'errs': [],
                } for v_ in ['a', 'd']
        } for msdkey_ in ['msd', 'msd' + COMOV_SUFFIX]
    }

    for (ct, tr), popul in popul_gen(cells):
        keylist = ['msd']
        if 'has_comov' in popul and popul['has_comov']:
            keylist.append('msd' + COMOV_SUFFIX)

        for msdkey, valkey in product(keylist, ['a', 'd']):
            # if msdkey == 'msd'+COMOV_SUFFIX:
            #     namesuf = f' {COMOV_SUFFIX}'
            # else:
            #     namesuf = ''

            all_vals[msdkey][valkey]['names'].append(f'{ct} {tr}')
            all_vals[msdkey][valkey]['vals'].append(
                popul['means'][msdkey][valkey])
            all_vals[msdkey][valkey]['errs'].append(
                popul['stds'][msdkey][valkey])

    keylist = ['msd', 'msd' + COMOV_SUFFIX]
    for msdkey, valkey in product(keylist, ['d', 'a']):
        fig.add_trace(go.Scatter(
            x=all_vals[msdkey][valkey]['vals'],
            y=all_vals[msdkey][valkey]['names'],
            name=f'{valkey} {msdkey}',
            mode='markers',
            marker=dict(
                size=18,
                symbol=comov_mark_symbol[valkey],
                color=comov_mark_color[msdkey],
                line=dict(
                    width=2,
                    color=comov_markline_color[msdkey],
                ),
            ),
        ))

        if valkey == 'd':
            fig.data[-1].update(xaxis='x2')

    return fig


def plot_ttest_boxplots(
        cells,
        celltypes,
        treatments,
        comovements=('all',),
        datatype_keys=('msd', 'd'),
        group_by='treatment',
        p_val_level=0,
        p_val_combinations=(
                ['mcf10a', 'mda231', 'mcf7'],
                ['mef', 'ht'],
                ['mef', 'mcf10a'],
        ),
        ttest='ind',
        pvals_position='bottom',
        df_data=None,
):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=f'{datatype_keys}'),
        # yaxis=dict(
        #     title=''
        # ),
    )
    slicer = pd.IndexSlice

    if df_data is None:
        if datatype_keys[0] == 'msd':
            if datatype_keys[1] == 'a':
                datatype = '_as'
            elif datatype_keys[1] == 'd':
                datatype = '_ds'
            else:
                datatype = None

            df_data = get_popul_msd_vals_df(cells, datatype)

        elif datatype_keys[0] == 'active':
            df_data = get_active_t_vals_df(cells)
            df_data = df_data.loc[:, slicer[:, datatype_keys[1]]]

    group_level = msd_df_index_order[group_by]
    grp = df_data.groupby(level=group_level, sort=False)

    level_inputs = {
        0: celltypes,
        1: treatments,
        2: comovements,
    }
    level_namings = {
        0: celltype_names,
        1: treatment_names_shorter,
        2: comovement_names
    }

    other_levels = sorted(set(range(3)).difference((group_level,)))

    firstiter = True

    used_idxs = []
    x_coord = 0
    x_coord_dict = {}

    for enum_, (i_, grp_) in enumerate(grp):
        if i_ in level_inputs[group_level]:

            # Adding dummy spaces between groups
            if not firstiter:
                # noinspection PyUnboundLocalVariable
                fig.add_trace(go.Box(
                    x=[[' ' * enum_], [' ']],
                    y=[np.mean(data)],
                    showlegend=False,
                    opacity=0,
                ))
                x_coord += 1

            keylist_ = list(itertools.product(*[level_inputs[li_] for li_ in other_levels]))
            keylist = []
            control_included = False

            for key in keylist_:
                if key[0] == 'nonp':
                    if not control_included:
                        keylist.append(('nonp', 'all'))
                        control_included = True
                    else:
                        continue

                else:
                    keylist.append(key)

            for subenum_, (key1, key2) in enumerate(keylist):
                if key1 == 'nonp':
                    key2 = 'all'
                try:
                    df_ = grp_.xs(
                        (key1, key2),
                        level=other_levels,
                    )
                except KeyError:
                    continue

                lidx_ = list(range(3))
                lidx_[other_levels[0]] = key1
                lidx_[other_levels[1]] = key2
                lidx_[group_level] = i_
                used_idxs.append(tuple(lidx_))

                if group_level == 0:
                    ct, tr, comov = i_, key1, key2
                elif group_level == 1:
                    ct, tr, comov = key1, i_, key2
                else:
                    ct, tr, comov = key1, key2, i_

                data = df_.loc[i_, datatype_keys]

                if group_level > 0:
                    r_, g_, b_ = celltype_colors[ct][4:-1].split(', ')
                    fillcolor = f'rgba({r_}, {g_}, {b_}, 0.8)'

                else:
                    if p_val_level == 1:
                        if subenum_ == 0:
                            r_, g_, b_ = celltype_colors[ct][4:-1].split(', ')
                            fillcolor = f'rgba({r_}, {g_}, {b_}, 0.66)'

                            fig.add_vrect(
                                x0=x_coord - 0.5,
                                x1=x_coord + 0.5,
                                y0=0,
                                y1=1,
                                fillcolor=fillcolor,
                                layer='below',
                            )
                            # fillcolor = f'rgba(255, 255, 255, 1)'
                            r_, g_, b_ = celltype_colors[ct][4:-1].split(', ')
                            fillcolor = f'rgba({r_}, {g_}, {b_}, 0.8)'

                            # r_, g_, b_ = celltype_colors[ct][4:-1].split(', ')
                            # fillcolor = f'rgba({r_}, {g_}, {b_}, 0.8)'

                        else:
                            if comov == COMOV_SUFFIX:
                                fillcolor = get_color(gray_cmap, 0.77)
                            else:
                                fillcolor = get_color(gray_cmap, 0.88)

                            fig.add_vrect(
                                x0=x_coord - 0.5,
                                x1=x_coord + 0.5,
                                y0=0,
                                y1=1,
                                fillcolor=fillcolor,
                                layer='below',
                            )
                            fillcolor = comov_colors[tr]

                    else:
                        if subenum_ == 0:
                            r_, g_, b_ = celltype_colors[ct][4:-1].split(', ')
                            fillcolor = f'rgba({r_}, {g_}, {b_}, 0.8)'
                        else:
                            fillcolor = grey_colors[subenum_]

                fig.add_trace(go.Box(
                    x=[
                        [level_namings[group_level][i_] for _ in data],
                        [
                            level_namings[other_levels[0]][key1]
                            + level_namings[other_levels[1]][key2]
                            for _ in data
                        ]
                    ],
                    y=data,
                    boxpoints='all',
                    jitter=0.5,
                    pointpos=-1.5,
                    line=dict(
                        width=2,
                        color='black',
                    ),
                    marker=dict(
                        size=7,
                        line=dict(
                            width=1,
                            # color=fillcolor,
                            color=celltype_colors[ct],
                        ),
                        color='rgba(0,0,0,0.13)',
                    ),
                    boxmean=True,
                    name=f'{key1} {key2}',
                    fillcolor=fillcolor,
                    showlegend=firstiter,
                    # line_color=treatment_colors[tr],
                    # marker_color=fillcolor,
                    # whiskerwidth=0.3,
                    offsetgroup=key1,
                ))
                x_coord_dict[(ct, tr, comov)] = x_coord
                x_coord += 1

            # To prevent legend duplicates.
            firstiter = False

    # Filling out the pairwise p-values for each treatment.
    # Also, save the highest value for each treatment to use
    # for calculating y coordinates of the annotations.
    dpvals = {}

    # Making dicts and converting them to DataFrames just to collect
    # the min and max values for each treatment might be a bit
    # excessive, but not too costly, so do it just in case we need it later.
    dmaxy = {}
    dminy = {}

    df_p = pd.DataFrame(index=pd.MultiIndex.from_tuples(used_idxs),
                        columns=pd.MultiIndex.from_tuples(used_idxs))

    # Here we compare the values only within the combination list
    # and only within the group_level (i.e. within treatmtent, celltype or comovement).
    # That way, later on we can just find the low p-values without worrying
    # about getting unwanted pairs, since they will have NaNs.
    for comp_comb in p_val_combinations:
        comp_idx = df_p.index.get_level_values(p_val_level).isin(comp_comb)
        comp_df = df_p.iloc[comp_idx, comp_idx]
        grpby1 = comp_df.groupby(level=group_level)
        for i_, grp_ in grpby1:
            grp_same_columns_as_i = grp_.loc[
                                    :, grp_.columns.get_level_values(group_level) == i_
                                    ]
            for irow_, row_ in grp_same_columns_as_i.iterrows():
                for irow2, row2 in row_.iteritems():
                    if ttest == 'ind':
                        data1 = df_data.loc[irow_, datatype_keys]
                        data2 = df_data.loc[irow2, datatype_keys]
                        df_p.loc[irow_, irow2] = mannwhitneyu(
                            data1,
                            data2,
                        ).pvalue
                    elif ttest == 'pairwise':
                        data1 = df_data.loc[irow_, datatype_keys]
                        data2 = df_data.loc[irow2, datatype_keys]
                        df_p.loc[irow_, irow2] = ttest_rel(
                            data1,
                            data2,
                        ).pvalue

                    dmaxy[irow_] = max(data1)
                    dminy[irow_] = min(data1)

    # Convert to the upper triangular matrix to avoid duplicates.
    df_p = df_p.where(np.triu(np.ones(df_p.shape)).astype(np.bool))
    signif_p_idxs = np.where(df_p.le(0.05))

    df_maxy = pd.DataFrame.from_dict(dmaxy, orient='index')
    df_miny = pd.DataFrame.from_dict(dminy, orient='index')
    df_maxy.index = pd.MultiIndex.from_tuples(df_maxy.index)
    df_miny.index = pd.MultiIndex.from_tuples(df_miny.index)

    # Change yaxis range to predetermined so that we can set the
    # proper y position of the annotations.
    yrange_min = df_miny.min().min()
    yrange_max = df_maxy.max().max()
    yrange_padding_coeff = 0.1
    yrange = yrange_max - yrange_min
    yrange_pad = yrange * yrange_padding_coeff

    # fig.update_layout(
    #     yaxis=dict(
    #         range=[yrange_min - yrange_pad,
    #                yrange_max + yrange_pad],
    #     ),
    # )

    annotation_padding_coeff = 0.05
    annotation_pad = yrange * annotation_padding_coeff
    annotation_height = annotation_pad * 0.3

    yshifts = {
        idx_: 0
        for idx_ in df_p.index.get_level_values(group_level).unique()
    }

    if pvals_position == 'bottom':
        for row, col in zip(*signif_p_idxs):
            idx1 = df_p.index[row]
            group_idx = idx1[group_level]
            idx2 = df_p.columns[col]

            x0pos = x_coord_dict[idx1]
            x1pos = x_coord_dict[idx2]

            y0pos = (df_maxy.xs(group_idx, level=group_level).max().max()
                     + yshifts[group_idx] + annotation_pad)
            y1pos = y0pos + annotation_height

            outliers_bot = {
                ('msd', 'a'): 3,
            }
            if datatype_keys in outliers_bot:
                nsmall = outliers_bot[datatype_keys]
                y0bot = df_miny[0].nsmallest(nsmall)[nsmall-1]
            else:
                y0bot = df_miny.min().min()

            y0bot -= annotation_pad
            y1bot = y0bot - annotation_height

            yshifts[group_idx] += annotation_height + annotation_pad

            # Making a bracket shape.
            # Horizontal line
            svgpath = f'M {x0pos:.6f},{y0pos:.6f} L {x0pos:.6f},{y1pos:.6f}' \
                      f' L {x1pos:.6f},{y1pos:.6f} L {x1pos:.6f},{y0pos:.6f}'

            svgpath_bot = f'M {x0pos:.6f},{y0bot:.6f} L {x0pos:.6f},{y1bot:.6f}' \
                          f' L {x1pos:.6f},{y1bot:.6f} L {x1pos:.6f},{y0bot:.6f}'

            # fig.add_shape(
            #     type='path',
            #     path=svgpath,
            #     line=dict(
            #         color='rgba(0,0,0,1)',
            #         width=3,
            #     ),
            #     fillcolor='rgba(0,0,0,0)',
            #     opacity=1,
            # )
            # fig.add_annotation(
            #     x=(x1pos + x0pos) / 2,
            #     y=y1pos,
            #     xref="x",
            #     yref="y",
            #     text=f'p={df_p.iloc[row, col]:.3f}',
            #     showarrow=True,
            #     font=dict(
            #         family="monospace",
            #         size=20,
            #         color="black"
            #     ),
            #     align="center",
            #     valign='top',
            #     arrowsize=1,
            #     ax=0,
            #     ay=-10,
            # )
            fig.add_shape(
                type='path',
                path=svgpath_bot,
                line=dict(
                    color='rgba(0,0,0,1)',
                    width=3,
                ),
                fillcolor='rgba(0,0,0,0)',
                opacity=1,
            )
            if df_p.iloc[row, col] <= 0.05:
                signif_text = f'<b><span style="font-size: 20px;">*</span></b><br>p={df_p.iloc[row, col]:.3f}'
            else:
                signif_text = f' ns<span style="font-size: 20px;"> </span>' \
                              f'<br>p={df_p.iloc[row, col]:.3f}'

            fig.add_annotation(
                x=(x1pos + x0pos) / 2,
                y=y1bot,
                xref="x",
                yref="y",
                text=signif_text,
                showarrow=True,
                font=dict(
                    family="monospace",
                    size=16,
                    color="black"
                ),
                align="center",
                valign='top',
                arrowsize=1,
                ax=0,
                ay=22,
            )

        # Compare everything with controls within the same group
        grpby1 = df_p.groupby(level=group_level)
        for i_, grp_ in grpby1:
            grp_same_columns_as_i = grp_.loc[
                                    :, grp_.columns.get_level_values(group_level) == i_
                                    ]
            row_ = grp_same_columns_as_i.loc(axis=0)[:, 'nonp']
            irow_ = row_.index[0]

            for irow2, row2 in row_.iteritems():
                if irow_ == irow2:
                    continue
                data1 = df_data.loc[irow_, datatype_keys]
                data2 = df_data.loc[irow2, datatype_keys]

                df_p.loc[irow_, irow2] = mannwhitneyu(
                    data1,
                    data2,
                ).pvalue

                p_val = df_p.loc[irow_, irow2]
                if p_val <= 0.05:
                    signif_text = f'<b>*</b>'
                    fontsize = 26
                else:
                    signif_text = f'ns'
                    fontsize = 18

                xpos = x_coord_dict[irow2]

                # Moving the p value labels below the outlier.
                outliers_top = {
                    ('active', 'ratio'): 2,
                    ('active', 'mean_run_l'): 2,
                    ('active', 'mean_flight_l'): 2,
                }
                if datatype_keys in outliers_top:
                    nlarge = outliers_top[datatype_keys]
                    ypos = df_maxy[0].nlargest(nlarge)[nlarge-1]
                else:
                    ypos = df_maxy[0].max()

                ypos += annotation_pad + annotation_height * 2

                fig.add_annotation(
                    x=xpos,
                    y=ypos,
                    xref="x",
                    yref="y",
                    text=signif_text,
                    showarrow=False,
                    font=dict(
                        family="monospace",
                        size=fontsize,
                        color="black",
                    ),
                    align="center",
                    valign='top',
                )

                if p_val < 0.001:
                    p_val_text = f'p<0.001'
                else:
                    p_val_text = f'p={p_val:.3f}'

                fig.add_annotation(
                    x=xpos,
                    y=ypos,
                    xref="x",
                    yref="y",
                    text=p_val_text,
                    showarrow=False,
                    font=dict(
                        family="monospace",
                        size=16,
                        color="black",
                    ),
                    # align='right',
                    # valign='top',
                    # yanchor='left',
                    yshift=50,
                    textangle=-90,
                )

    elif pvals_position == 'top':
        for row, col in zip(*signif_p_idxs):
            idx1 = df_p.index[row]
            group_idx = idx1[group_level]
            idx2 = df_p.columns[col]

            x0pos = x_coord_dict[idx1]
            x1pos = x_coord_dict[idx2]

            y0pos = (df_maxy.xs(group_idx, level=group_level).max().max()
                     + yshifts[group_idx] + annotation_pad)
            y1pos = y0pos + annotation_height

            yshifts[group_idx] += annotation_height + annotation_pad

            # Making a bracket shape.
            # Horizontal line
            svgpath = f'M {x0pos:.6f},{y0pos:.6f} L {x0pos:.6f},{y1pos:.6f}' \
                      f' L {x1pos:.6f},{y1pos:.6f} L {x1pos:.6f},{y0pos:.6f}'

            fig.add_shape(
                type='path',
                path=svgpath,
                line=dict(
                    color='rgba(0,0,0,1)',
                    width=3,
                ),
                fillcolor='rgba(0,0,0,0)',
                opacity=1,
            )
            fig.add_annotation(
                x=(x1pos + x0pos) / 2,
                y=y1pos,
                xref="x",
                yref="y",
                text=f'{df_p.iloc[row, col]}'[:6],
                showarrow=True,
                font=dict(
                    family="monospace",
                    size=20,
                    color="black"
                ),
                align="center",
                valign='top',
                arrowsize=1,
                ax=0,
                ay=-10,
            )

    # fig.show()

    return fig


def plot_popul_mean_heatmap(
        cells,
        key1='msd',
        key2='a',
        title='',
):
    df = popul_means_to_df(cells, key1, key2)

    df = df.reindex(celltypes_order)
    df.index = df.reset_index().loc[:, 'index'].replace(celltype_names)
    df = df.reindex(columns=[tr_ for tr_ in treatments_order if tr_ in df.columns])
    df.rename(columns=treatment_names_more, inplace=True)

    if key2 == 'ratio':
        percentcoeff = 100
        ticksuffix = '%'

    else:
        percentcoeff = 1
        ticksuffix = ''

    fig = go.Figure(
        data=go.Heatmap(
            z=df.values.T * percentcoeff,
            x=df.index.values,
            y=df.columns.to_list(),
            colorscale='Plasma',
            colorbar=dict(
                ticksuffix=ticksuffix,
            ),
        ),
    )

    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(
            # title='Cell type',
            showgrid=False,
            showline=True,
            mirror=True,
            type='category',
        ),
        yaxis=dict(
            type='category',
            # title='Treatment',
            showgrid=False,
            showline=True,
            mirror=True,
        ),
    )

    # fig.show()

    return fig


def make_popul_means_plots(
        cells,
        plotfunc,
        id_suffix='hm',
        width='37em',
        height='33em',
):
    lplots = []
    msdkeys = product(['msd'], ['a', 'd'])
    for key1, key2 in msdkeys:
        if key2 == 'd':
            width_ = '38em'
        else:
            width_ = width

        id_name = f'{key1}-{key2}-{id_suffix}'
        but_conf = {
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': id_name,
                # 'scale': 1,
            }
        }

        layout_ = html.Div([
            html.Button('Update plot', id=f'{key1}-{key2}-{id_suffix}-upd-btn'),
            dcc.Graph(
                id=id_name,
                figure=plotfunc(
                    cells, key1, key2,
                    title=f'{key1} {key2}',
                ),
                style={
                    'width': width_,
                    'height': height,
                },
                config=but_conf,
            )
        ], style=dict(display='inline-block'))
        lplots.append(layout_)

    key1 = 'active_t'
    key2 = 'ratio'
    id_name = f'{key1}-{key2}-{id_suffix}'
    but_conf = {
        'toImageButtonOptions': {
            'format': 'svg',
            'filename': id_name,
            # 'scale': 1,
        }
    }
    layout_ = html.Div([
        html.Button('Update plot', id=f'{key1}-{key2}-{id_suffix}-upd-btn'),
        dcc.Graph(
            id=id_name,
            figure=plotfunc(
                cells, key1, key2,
                title=f'{key1} {key2}',
            ),
            style={
                'width': width,
                'height': height,
            },
            config=but_conf,
        )
    ], style=dict(display='inline-block'))
    lplots.append(layout_)

    return lplots


def dash_msd_table(
        df,
        element_id='msd-sum-tbl',
        tblkwgs=None,
):
    if tblkwgs is None:
        tblkwgs = {}

    df = prepare_act_t_df4table(df)

    columns = []
    num_cols = df.select_dtypes('number').columns

    for dfcol in df.columns:
        if dfcol != 'id':
            tcol = {'name': dfcol, 'id': dfcol}

            if dfcol in active_t_ratio_names_more_reverse and \
                    'ratio' in active_t_ratio_names_more_reverse[dfcol]:

                tcol.update({
                    'type': 'numeric',
                    'format': Format(precision=2, scheme=Scheme.percentage),
                })

            elif dfcol in num_cols:
                tcol.update({
                    'type': 'numeric',
                })

            columns.append(tcol)

    styles = shade_nonnumeric_cols(df)

    # styles.append({
    #     'if': {'state': 'selected'},
    #     'backgroundColor': 'inherit !important',
    # })

    return dt.DataTable(
        id=element_id,
        data=df.to_dict('records'),
        columns=columns,
        style_data_conditional=styles,
        style_cell={'fontSize': 14, 'font-family': 'sans-serif'},
        # style_table={'width': '100rem'},
        style_header={
            'fontWeight': 'bold',
            'whiteSpace': 'normal',
            'heiht': 'auto',
        },
        **tblkwgs
    )


def plot_many_msdfits_scatter(
        manycells,
):
    fig = go.Figure()
    fig.update_layout(
        yaxis={
            'showgrid': True,
        },
        xaxis={
            'title': r'$\alpha$',
            'showgrid': True,
            'range': [1, 1.4],
        },
        xaxis2={
            'title': '$D$',
            'anchor': 'y',
            'overlaying': 'x',
            'side': 'top',
        })

    for kcells, cells in manycells.items():
        all_vals = {
            msdkey_: {
                v_:
                    {
                        'names': [],
                        'vals': [],
                        'errs': [],
                    } for v_ in ['a', 'd']
            } for msdkey_ in ['msd', 'msd' + COMOV_SUFFIX]
        }

        for (ct, tr), popul in popul_gen(cells):
            keylist = ['msd']
            if 'has_comov' in popul and popul['has_comov']:
                keylist.append('msd' + COMOV_SUFFIX)

            for msdkey, valkey in product(keylist, ['a', 'd']):
                # if msdkey == 'msd'+COMOV_SUFFIX:
                #     namesuf = f' {COMOV_SUFFIX}'
                # else:
                #     namesuf = ''

                all_vals[msdkey][valkey]['names'].append(f'{ct} {tr} {msdkey}')
                all_vals[msdkey][valkey]['vals'].append(
                    popul['means'][msdkey][valkey])
                all_vals[msdkey][valkey]['errs'].append(
                    popul['stds'][msdkey][valkey])

        keylist = ['msd', 'msd' + COMOV_SUFFIX]
        for msdkey, valkey in product(keylist, ['d', 'a']):
            fig.add_trace(go.Scatter(
                x=all_vals[msdkey][valkey]['vals'],
                y=all_vals[msdkey][valkey]['names'],
                name=f'{valkey} {msdkey} {kcells}',
                mode='markers',
                marker=dict(
                    size=18,
                    symbol=comov_mark_symbol[valkey],
                    color=get_markline_color(kcells),
                    line=dict(
                        width=2,
                        color=get_markline_color(kcells),
                    ),
                ),
            ))

            if valkey == 'd':
                fig.data[-1].update(xaxis='x2')

    return fig
