import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from mlprops.util import read_json, lookup_meta, identify_all_correlations
from mlprops.index_and_rate import prop_dict_to_val
from mlprops.load_experiment_logs import find_sub_db
from mlprops.elex.util import ENV_SYMBOLS, RATING_COLORS, RATING_COLOR_SCALE, RATING_COLOR_SCALE_REV
from mlprops.elex.graphs import assemble_scatter_data, create_scatter_graph, add_rating_background


PLOT_WIDTH = 800
PLOT_HEIGHT = PLOT_WIDTH // 3
COLORS = ['#009ee3', '#983082', '#ffbc29', '#35cdb4', '#e82e82', '#59bdf7', '#ec6469', '#706f6f', '#4a4ad8', '#0c122b', '#ffffff']


def create_all(databases):
    os.chdir('paper_results')

    ####### DUMMY OUTPUT ####### for setting up pdf export of plotly
    fig=px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    time.sleep(0.5)
    os.remove("dummy.pdf")

    # imagenet env trades
    db_name, ds, task = 'ImageNetEff', 'imagenet', 'infer_imagenet'
    db, meta, metrics, xdef, ydef, bounds, _, _ = databases[db_name]
    envs = sorted([env for env in pd.unique(db['environment']) if 'Laptop' not in env])
    models = sorted(pd.unique(db['model']).tolist())
    traces = {}
    for env in envs:
        subdb = db[(db['environment'] == env) & (db['task'] == task)]
        avail_models = set(subdb['model'].tolist())
        traces[env] = [subdb[subdb['model'] == mod]['compound_index'].iloc[0] if mod in avail_models else None for mod in models]
    model_names = [f'{mod[:3]}..{mod[-5:]}' if len(mod) > 10 else mod for mod in models]
    fig = go.Figure(
        layout={'width': PLOT_WIDTH, 'height': PLOT_HEIGHT, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
                'yaxis':{'title': 'Compound score'}},
        data=[
            go.Scatter(x=model_names, y=vals, name=env, mode='markers',
            marker=dict(
                color=RATING_COLORS[i],
                symbol=ENV_SYMBOLS[i]
            ),) for i, (env, vals) in enumerate(traces.items())
        ]
    )
    fig.update_layout( legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5) )
    fig.show()
    fig.write_image(f'environment_changes.pdf')

    # scatter plots
    xaxis, yaxis = xdef[(ds, task)], ydef[(ds, task)]
    host_envs_map = { host: [env for env in envs if host in env] for host in ['Workstation', 'RasPi'] }
    for host, host_envs in host_envs_map.items():
        plot_data, axis_names, rating_pos = assemble_scatter_data(host_envs, db, 'index', xaxis, yaxis, meta, bounds)
        scatter = create_scatter_graph(plot_data, axis_names, dark_mode=False)
        rating_pos[0][0][0] = scatter.layout.xaxis.range[1]
        rating_pos[1][0][0] = scatter.layout.yaxis.range[1]
        add_rating_background(scatter, rating_pos, 'optimistic mean', dark_mode=False)
        scatter.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 25}, title_y=0.99, title_x=0.5, title_text=f'{host}')
        scatter.show()
        scatter.write_image(f"scatter_{host}.pdf")

if __name__ == '__main__':
    pass