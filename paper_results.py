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


PLOT_WIDTH = 800
PLOT_HEIGHT = PLOT_WIDTH // 3
COLORS = ['#009ee3', '#983082', '#ffbc29', '#35cdb4', '#e82e82', '#59bdf7', '#ec6469', '#706f6f', '#4a4ad8', '#0c122b', '#ffffff']
SEL_DS_TASK = {
    'ImageNetEff': ('imagenet', 'infer'),
    'CocoEff': ('coco', 'infer'),
}

def create_all(databases):
    os.chdir('paper_results')

    ####### DUMMY OUTPUT ####### for setting up pdf export of plotly
    fig=px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    time.sleep(0.5)
    os.remove("dummy.pdf")
    
    db_name, ds, task = 'ImageNetEff', 'imagenet', 'infer'
    db, meta, metrics, xdef, ydef, bounds, _, _ = databases[db_name]
    envs = sorted([env for env in pd.unique(db['environment']) if 'Laptop' not in env])
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
                    color=RATING_COLORS[i],
                    symbol=ENV_SYMBOLS[i]
                ),) for i, (env, vals) in enumerate(trace.items())
            ]
        )
        fig.update_yaxes(type="log")
        fig.show()
        fig.write_image(f'environment_{metric.split()[0].lower()}.pdf')

    # # compound imagenet env trades
    # fig = go.Figure(
    #     layout={'width': PLOT_WIDTH, 'height': PLOT_HEIGHT, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
    #             'yaxis':{'title': 'Compound score'}},
    #     data=[
    #         go.Scatter(x=model_names, y=vals, name=env, mode='markers',
    #         marker=dict(
    #             color=RATING_COLORS[i],
    #             symbol=ENV_SYMBOLS[i]
    #         ),) for i, (env, vals) in enumerate(compoound_traces.items())
    #     ]
    # )
    # fig.update_layout( legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5) )
    # fig.show()
    # fig.write_image(f'environment_changes.pdf')

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

    # Star Plots
    for name, (db, meta, metrics, xdef, ydef, bounds, _, _) in databases.items():
        ds, task = SEL_DS_TASK[name]
        db = prop_dict_to_val(db, 'index')
        ds_name = lookup_meta(meta, ds, subdict='dataset')

        worst = db.sort_values('compound_index').iloc[0]
        best = db.sort_values('compound_index').iloc[-1]
        fig = go.Figure()
        for model, col, m_str in zip([best, worst], [RATING_COLORS[0], RATING_COLORS[4]], ['Best', 'Worst']):
            mod_name = lookup_meta(meta, model['model'], 'short', 'model')[:18]
            print(metrics)
            metr_names = [lookup_meta(meta, metr, 'shortname', 'properties') for metr in metrics[(ds, task)]]
            fig.add_trace(go.Scatterpolar(
                r=[model[col] for col in metrics[(ds, task)]], line={'color': col},
                theta=metr_names, fill='toself', name=f'{mod_name} ({m_str}): {model["compound_index"]:4.2f}'
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)), width=PLOT_WIDTH*0.25, height=PLOT_HEIGHT, title_y=1.0, title_x=0.5, title_text=ds_name,
            legend=dict( yanchor="bottom", y=1.06, xanchor="center", x=0.5), margin={'l': 30, 'r': 30, 'b': 15, 't': 70}
        )
        fig.write_image(f'true_best_{name}.pdf')
if __name__ == '__main__':
    pass