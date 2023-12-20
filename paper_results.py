import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from strep.util import read_json, lookup_meta, identify_all_correlations
from strep.index_and_rate import prop_dict_to_val
from strep.load_experiment_logs import find_sub_db
from strep.elex.util import ENV_SYMBOLS, RATING_COLORS, RATING_COLOR_SCALE, RATING_COLOR_SCALE_REV
from strep.elex.graphs import assemble_scatter_data, create_scatter_graph, add_rating_background


def rbg_to_rgba(rgb, alpha):
    return rgb.replace('rgb', 'rgba').replace(')', f',{alpha})')


PLOT_WIDTH = 800
PLOT_HEIGHT = PLOT_WIDTH // 3
COLORS = ['#009ee3', '#983082', '#ffbc29', '#35cdb4', '#e82e82', '#59bdf7', '#ec6469', '#706f6f', '#4a4ad8', '#0c122b', '#ffffff']
SEL_DS_TASK = {
    'ImageNetEff': ('imagenet', 'infer'),
    'CocoEff': ('coco', 'infer'),
}
HOSTS = ['Desktop', 'Laptop', 'RasPi']
MODELS = ['MobileNetV2', 'DenseNet121', 'ResNet101V2']

def create_all(databases):
    os.chdir('paper_results')

    ####### DUMMY OUTPUT ####### for setting up pdf export of plotly
    fig=px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    time.sleep(0.5)
    os.remove("dummy.pdf")
    
    db_name, ds, task = 'ImageNetEff', 'imagenet', 'infer'
    db, meta, metrics, xdef, ydef, bounds, _, _ = databases[db_name]
    envs = sorted([env for env in pd.unique(db['environment'])])
    models = sorted(pd.unique(db['model']).tolist())

    power_draw_traces, acc_traces, compoound_traces = {}, {}, {}
    for env in envs:
        subdb = db[(db['environment'] == env) & (db['task'] == task)]
        avail_models = set(subdb['model'].tolist())
        acc_traces[env] = [subdb[subdb['model'] == mod]['accuracy_k1'].iloc[0]['value'] if mod in avail_models else None for mod in models]
        power_draw_traces[env] = [subdb[subdb['model'] == mod]['approx_USB_power_draw'].iloc[0]['value'] if mod in avail_models else None for mod in models]
        compoound_traces[env] = [subdb[subdb['model'] == mod]['compound_index'].iloc[0] if mod in avail_models else None for mod in models]
    model_names = [f'{mod[:3]}..{mod[-5:]}' if len(mod) > 10 else mod for mod in models]

    # environment profiling
    for metric, trace in zip(['Compound score', 'Accuracy [%]', 'Power draw per inference [Ws]'], [compoound_traces, acc_traces, power_draw_traces]):
        fig = go.Figure(
            layout={'width': PLOT_WIDTH, 'height': PLOT_HEIGHT, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
                    'yaxis':{'title': metric}},
        data=[
                go.Scatter(x=model_names, y=vals, name=env, mode='markers+lines',
                marker=dict(
                    color=COLORS[i],
                    symbol=ENV_SYMBOLS[i]
                ),) for i, (env, vals) in enumerate(trace.items())
            ]
        )
        fig.update_yaxes(type="log")
        fig.write_image(f'environment_{metric.split()[0].lower()}.pdf')

    # compound imagenet env trades
    fig = go.Figure(
        layout={'width': PLOT_WIDTH, 'height': PLOT_HEIGHT, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
                'yaxis':{'title': 'Compound score'}},
        data=[
            go.Scatter(x=model_names, y=vals, name=env, mode='markers',
            marker=dict(
                color=COLORS[i],
                symbol=ENV_SYMBOLS[i]
            ),) for i, (env, vals) in enumerate(compoound_traces.items())
        ]
    )
    fig.update_layout( legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5) )
    fig.show()
    fig.write_image(f'environment_changes.pdf')

    # scatter plots
    xaxis, yaxis = 'resource_index', 'quality_index'
    host_envs_map = { host: [env for env in envs if host in env] for host in HOSTS }
    for host, host_envs in host_envs_map.items():
        plot_data, axis_names, rating_pos = assemble_scatter_data(host_envs, db, 'index', xaxis, yaxis, meta, bounds)
        scatter = create_scatter_graph(plot_data, axis_names, marker_width=8, dark_mode=False)
        rating_pos[0][0][0] = scatter.layout.xaxis.range[1]
        rating_pos[1][0][0] = scatter.layout.yaxis.range[1]
        add_rating_background(scatter, rating_pos, 'optimistic mean', dark_mode=False)
        scatter.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                            #   title_y=0.99, title_x=0.5, title_text=f'{host}',
                              yaxis_range=[0.8, 1.02], legend=dict(x=1, y=1, orientation="v", xanchor="right", yanchor="top"))
        scatter.show()
        scatter.write_image(f"scatter_{host}.pdf")

    # Star Plots
    db, meta, metrics, _, _, _, _, _ = databases['ImageNetEff']
    metrics = metrics[SEL_DS_TASK['ImageNetEff']].tolist()
    metrics.append(metrics[0]) # fully connected stars
    metr_names = [lookup_meta(meta, metr, 'shortname', 'properties') for metr in metrics]

    for host, model in zip(HOSTS, MODELS):
        envs = host_envs_map[host]

        db = prop_dict_to_val(db, 'index')
        fig = go.Figure()
        for env, color in zip(envs, [RATING_COLORS[idx] for idx in [4, 2, 0]]):
            row = db[(db['model'] == model) & (db['environment'] == env)]
            if row.size > 0:
                fig.add_trace(go.Scatterpolar(
                    r=[row[col].iloc[0] for col in metrics], line={'color': color}, fillcolor=rbg_to_rgba(color, 0.3),
                    theta=metr_names, fill='toself', name=f'{env}: {row["compound_index"].values[0]:4.2f}'
                ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)), width=PLOT_WIDTH*0.25, height=PLOT_HEIGHT, title_y=1.0, title_x=0.5, title_text=f'{model} on {host}',
            legend=dict( yanchor="bottom", y=1.06, xanchor="center", x=0.5), margin={'l': 30, 'r': 30, 'b': 15, 't': 70}
        )
        fig.show()
        fig.write_image(f'true_best_{host}.pdf')
if __name__ == '__main__':
    pass