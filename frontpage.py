import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from trackedcell import TrackedCell
import pandas as pd
import base64
import io
from dataprep import *
from layout_utils import *
from loguru import logger as lg

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

DASH_PORT = 13888

cells = {}


"""
==================================================================
        Making the page layout.
==================================================================
"""
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Div([
        html.H3('Cell type'),
        dcc.Input(
            id='ct-input',
            type='text',
            value='celltype1',
        ),
    ]),
    html.Div([
        html.H3('Treatment'),
        dcc.Input(
            id='tr-input',
            type='text',
            value='treatment1',
        ),
    ]),
    html.Div([
        html.H3('Cell id'),
        dcc.Input(
            id='id-input',
            type='text',
            value='01',
        ),
    ]),

    html.Div([
        dcc.Upload([
            'Drag and Drop or ',
            html.A('select a .csv file')
        ],
            id='upload-data',
            multiple=False,
            style={
                'width': '50rem',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
            }),
    ],
        style=dict(padding='1.5rem'),
    ),

    html.Div(id='file-out-div'),
    html.Button('Process data', id='process-btn', disabled=True),
    html.Div(id='cell-out-div'),

    html.Div([
        dcc.Graph(id='msd-fig'),
    ],
        style=dict(
            padding='1.5rem',
            display='inline-block',
        ),
    ),

    html.Div([
        dcc.Graph(id='pairs-fig'),
    ],
        style=dict(
            padding='1.5rem',
            display='inline-block',
        ),
    ),

    html.Div([
        dcc.Graph(id='tracks-fig'),
    ],
        style=dict(
            padding='1.5rem',
            display='inline-block',
        ),
    ),

    dcc.Store(id='cells-store', data=dict()),
    dcc.Store(id='df-store'),
    dcc.Store(id='ct-tr-id-store')
])


"""
==================================================================
        Helper function to process the uploaded data.
==================================================================
"""


def parse_csv(contents):
    content_type, data_str = contents.split(',')

    decoded = base64.b64decode(data_str)
    # charcode = chardet.detect(decoded)

    df = pd.read_csv(io.StringIO(decoded.decode('unicode_escape')))
    if df.iloc[0, 0] == 'Label' and pd.isna(df.iloc[2, 0]):
        df = pd.read_csv(io.StringIO(decoded.decode('unicode_escape')),
                         skiprows=[1, 2, 3])

    return df


"""
==================================================================
        Callbacks for the interactive plots.
==================================================================
"""


@app.callback(
    Output('file-out-div', 'children'),
    Output('df-store', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
)
def calculate_out(
        csv_contents,
        filename: str,
):
    if filename is None or not filename.endswith('.csv'):
        return 'Please upload a .csv file', None

    df = parse_csv(csv_contents)

    return html.Span(f'File uploaded, the columns are: {df.columns.values}'), df.to_json(orient='split')


@app.callback(
    Output('process-btn', 'disabled'),
    Output('ct-tr-id-store', 'data'),
    Input('df-store', 'data'),
    State('ct-input', 'value'),
    State('tr-input', 'value'),
    State('id-input', 'value'),
)
def process_btn_state(
        df,
        ct,
        tr,
        cell_id,
):
    if df is None:
        return True, None
    else:
        return (
            False,
            {'ct': ct, 'tr': tr, 'id': cell_id}
        )


@app.callback(
    Output('cell-out-div', 'children'),
    Input('process-btn', 'n_clicks'),
    State('df-store', 'data'),
    State('ct-tr-id-store', 'data'),
    prevent_initial_call=True,
)
def process_data(n_clicks, df, dname):
    cname = f'{dname["ct"]}_{dname["tr"]}_{dname["id"]}'
    cell = TrackedCell(
        cname,
        df=pd.read_json(df, orient='split'),
        find_processed=False,
    )
    sample = {'tc': cell, 'id': dname['id']}
    cells.update({
        dname['ct']: {
            dname['tr']: {
                'samples': [sample],
            }
        }
    })

    df_lys = cell.df_lys
    df = df_lys.rename(columns={'posx': 'x', 'posy': 'y'})
    grp = df.groupby('id')
    lg.debug(f"calculating msd for [{cname}] ...")
    sample['analysis'] = {}
    sample['analysis']['msd'] = get_msd(grp)

    fit_msd_population(cells)

    return html.Span(f'Data processed.')


@app.callback(
    Output('msd-fig', 'figure'),
    Input('cell-out-div', 'children'),
)
def plot_msd(cells_str):
    if cells_str is None:
        return noplot_msg('Process data to show MSD plot')

    lret = []
    for (ct, tr), popul in popul_gen(cells):
        msd = popul['msd'].mean(axis=1)
        alph = popul['msd_fit'][0][1]
        difcoef = popul['msd_fit'][0][0]
        lret.append({
            'x': msd.index,
            'y': msd,
            'type': 'scatter',
            'name': f'{ct} {tr} {symbols["alpha"]}={alph:.3f}; D={difcoef:.5f}',
        })

    return {'data': lret,
            'layout': {
                'title': 'MSD',
                'xaxis': {
                    'title': 'Time (frames)',
                    'type': 'log',
                    'showline': True,
                    'tickformat': 'f',
                    'dtick': 1,
                },
                'yaxis': {
                    'title': 'MSD',
                    'type': 'log',
                    'showline': True,
                    'tickformat': 'f',
                    'dtick': 1,
                },
                'legend': {
                    'x': 0.5,
                    'y': 0.05,
                },
                'showlegend': True,
            }}


@app.callback(
    Output('pairs-fig', 'figure'),
    Input('cell-out-div', 'children'),
)
def plot_matches(cells_str):
    try:
        _, popul = next(popul_gen(cells))
    except StopIteration:
        return noplot_msg('Process data with co-movement to show co-moving pairs')

    tc = popul['samples'][0]['tc']
    if tc.imatch:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tc.imatch[1],
            y=tc.imatch[0],
            mode='markers',
        ))
        fig.update_layout(
            title='Co-moving pairs ids (click on a point to show tracks)',
        )

        return fig

    else:
        return noplot_msg('Co-moving pairs not found')


@app.callback(
    Output('tracks-fig', 'figure'),
    Input('pairs-fig', 'clickData'),
)
def plot_tracks(clickdata):
    if clickdata is None:
        return noplot_msg('Click on a point in the co-moving pairs to show tracks')

    point = clickdata['points'][0]
    id1 = point['y']
    id2 = point['x']

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 18},
            "prefix": "Frame: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {
            "duration": 100,
            # "easing": "cubic-in-out",
        },
        "pad": {"b": 10, "t": 30},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    fig = go.Figure(layout={
        'title': '<b>Tracks</b>',
        # 'height': 550,
        'updatemenus': [{
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 100, "redraw": False},
                            "fromcurrent": True,
                            # "transition": {
                            #     "duration": 70,
                            #     "easing": "quadratic-in-out",
                            # }
                        }],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        # "transition": {"duration": 0}
                    }],
                    "label": "Pause",
                    "method": "animate"
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    })

    _, popul = next(popul_gen(cells))
    tc = popul['samples'][0]['tc']

    idlys = tc.df_lys.index.get_level_values(0).unique()[id1]
    idnp = tc.df_np.index.get_level_values(0).unique()[id2]

    df_lys = tc.df_lys.loc[idlys]
    fig.add_trace(go.Scatter(
        x=df_lys.posx,
        y=df_lys.posy,
        mode='lines',
        name='Object 1',
        text=df_lys.index,
        hovertemplate='%{text}',
        line=dict(color='rgb(115, 115, 115)'),
    ))

    df_np = tc.df_np.loc[idnp]
    fig.add_trace(go.Scatter(
        x=df_np.posx,
        y=df_np.posy,
        mode='lines',
        name='Object 2',
        text=df_np.index,
        hovertemplate='%{text}',
    ))

    loverlap_itime = df_np.index.join(
        df_lys.index,
        how='inner',
    )

    if len(loverlap_itime) > 0:
        fig.add_trace(go.Scatter(
            x=[df_lys.posx.loc[loverlap_itime[0]], df_np.posx.loc[loverlap_itime[0]]],
            y=[df_lys.posy.loc[loverlap_itime[0]], df_np.posy.loc[loverlap_itime[0]]],
            mode='lines+markers+text',
            text=[None, ]
        ))
        lframes = []

        for it_ in loverlap_itime:
            frame = {'data': [], 'name': str(it_), 'traces': [2]}
            lx_ = df_lys.posx.loc[it_]
            ly_ = df_lys.posy.loc[it_]
            nx_ = df_np.posx.loc[it_]
            ny_ = df_np.posy.loc[it_]
            dist_ = np.linalg.norm([lx_ - nx_, ly_ - ny_])
            dframe_data = {
                'x': [lx_, nx_],
                'y': [ly_, ny_],
                'mode': 'lines+markers+text',
                'text': [None, dist_],
                'marker': {'size': 13, 'color': 'red'},
                'name': 'Frame',
            }

            frame['data'].append(dframe_data)
            lframes.append(frame)

            slider_step = make_slider_step(it_)
            sliders_dict['steps'].append(slider_step)

        fig.frames = lframes

    fig.layout.sliders = [sliders_dict]

    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    )
    fig.update_yaxes(scaleanchor='x', scaleratio=1, showline=True)

    return fig


if __name__ == '__main__':
    app.run_server(debug=False, port=DASH_PORT)
