import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.express.colors import sample_colorscale

from strep.util import read_json, lookup_meta, identify_all_correlations
from strep.index_and_rate import prop_dict_to_val
from strep.load_experiment_logs import find_sub_db
from strep.elex.util import ENV_SYMBOLS, RATING_COLORS, RATING_COLOR_SCALE, RATING_COLOR_SCALE_REV
from strep.elex.graphs import assemble_scatter_data, add_rating_background


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
MODELS = [('ImageNetEff', 'ResNet50V2'), ('CocoEff', 'yolov8n-seg'), ('ImageNetEff', 'MobileNet')]


def extract_results(subdb, field, scale, all_models):
    avail_models = set(subdb['model'].tolist())
    results = []
    for mod in all_models:
        if mod in avail_models:
            value = subdb[subdb['model'] == mod][field].iloc[0]
            if isinstance(value, dict):
                value = value[scale]
            results.append(value)
        else:
            results.append(None)
    return results


def create_all(databases):
    os.chdir('paper_results')

    ####### DUMMY OUTPUT ####### for setting up pdf export of plotly
    fig=px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    time.sleep(0.5)
    os.remove("dummy.pdf")
    
    envs = sorted(pd.unique(databases['ImageNetEff'][0]['environment']).tolist())
    env_cols, env_symb = {}, {}
    for env in envs:
        if 'TPU' in env:
            env_cols[env] = RATING_COLORS[2]
        else:
            env_cols[env] = RATING_COLORS[0] if 'NCS' in env else RATING_COLORS[4]
        if 'Desktop' in env:
            env_symb[env] = ENV_SYMBOLS[0]
        else:
            env_symb[env] = ENV_SYMBOLS[1] if 'Laptop' in env else ENV_SYMBOLS[4]
    host_envs_map = { host: [env for env in envs if host in env] for host in HOSTS }
    db = databases['ImageNetEff'][0]
    models_cls = sorted([mod for mod in pd.unique(db['model']) if db[db['model'] == mod].shape[0] > 3])
    db = databases['CocoEff'][0]
    models_seg = sorted([mod for mod in pd.unique(db['model']) if db[db['model'] == mod].shape[0] > 3])
    models = models_cls + [None] + models_seg
    model_names = [f'{mod[:3]}..{mod[-5:]}' if mod is not None and len(mod) > 10 else mod for mod in models]

    acc_metrics = {'ImageNetEff': 'accuracy_k1', 'CocoEff': 'mAP50_M'}
    acc_metrics2 = {'ImageNetEff': 'accuracy_k5', 'CocoEff': 'mAP50_95_M'}
    time_acc_power = {
        'Running time UUUT [s]': {env: [] for env in envs},
        'Top1 / mAP50 UUUT [%]': {env: [] for env in envs},
        'Power draw UUUT [Ws]': {env: [] for env in envs},
        'Power draw SUUT [%]': {env: [] for env in envs},
        # 'Acc / mAP50 SUUT [%]': {env: [] for env in envs}
    }
    others = {
        'acc2': {env: [] for env in envs},
        'acc_rel' : {env: [] for env in envs},
        'comp': {env: [] for env in envs}
    }
    
    # power_traces, acc_traces, power_idx_traces, acc_idx_traces, compound_traces = {}, {}, {}, {}, {}
    for env in envs:
        for avail_models, (db_name, (db, meta, metrics, _, _, bounds, _, _)) in zip([models_cls, models_seg], databases.items()):
            subdb = db[(db['environment'] == env)]
            time_acc_power['Power draw UUUT [Ws]'][env] = time_acc_power['Power draw UUUT [Ws]'][env] + extract_results(subdb, 'approx_USB_power_draw', 'value', avail_models) + [None]
            time_acc_power['Power draw SUUT [%]'][env] = time_acc_power['Power draw SUUT [%]'][env] + extract_results(subdb, 'approx_USB_power_draw', 'index', avail_models) + [None]
            time_acc_power['Top1 / mAP50 UUUT [%]'][env] = time_acc_power['Top1 / mAP50 UUUT [%]'][env] + extract_results(subdb, acc_metrics[db_name], 'value', avail_models) + [None]
            time_acc_power['Running time UUUT [s]'][env] = time_acc_power['Running time UUUT [s]'][env] + extract_results(subdb, 'running_time', 'value', avail_models) + [None]
            others['acc2'][env] = others['acc2'][env] + extract_results(subdb, acc_metrics2[db_name], 'value', avail_models) + [None]
            others['acc_rel'][env] = others['acc_rel'][env] + extract_results(subdb, acc_metrics[db_name], 'index', avail_models) + [None]
            others['comp'][env] = others['comp'][env] + extract_results(subdb, 'compound_index', 'index', avail_models) + [None]

    # model / environment runtime vs accuracy vs power draw
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    for idx, (metric, trace) in enumerate(time_acc_power.items()):
        for i, (env, vals) in enumerate(trace.items()):
            fig.add_trace(
                go.Scatter(x=model_names, y=vals, name=env, mode='markers+lines', legendgroup=env, showlegend=idx==0,
                    marker=dict(
                        color=env_cols[env],
                        symbol=env_symb[env]
                    )), row=idx+1, col=1
            )
        fig.update_yaxes(title_text=metric, row=idx+1, col=1)
        if 'Ws' in metric:
            fig.update_yaxes(type="log", row=idx+1, col=1)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT * 3, margin={'l': 0, 'r': 0, 'b': 0, 't': 50},
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5))
    fig.show()
    fig.write_image(f'env_mod_time_acc_power.pdf')

    # model / environment redundancy - run time VS power_draw & acc1 vs acc5
    time_acc_power['Top5 / mAP50-95 UUUT [%]'] = others['acc2']
    time_acc_power['Top1 / mAP50 SUUT [%]'] = others['acc_rel']
    plot1 = ['Running time UUUT [s]', 'Power draw UUUT [Ws]', [-0.02, 0.8], [-1, 24]]
    plot2 = ['Top1 / mAP50 UUUT [%]', 'Top5 / mAP50-95 UUUT [%]', [0.56, 0.79], [0.75, 0.95]]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.02)
    for idx, (xaxis, yaxis, xrange, yrange) in enumerate([plot1 , plot2]):
        for env in envs:
            fig.add_trace(
                go.Scatter(x=time_acc_power[xaxis][env], y=time_acc_power[yaxis][env], name=env,
                        mode='markers', marker=dict(color=env_cols[env], symbol=env_symb[env]),
                        legendgroup=env, showlegend=idx==0), row=1, col=idx+1
            )
            fig.update_xaxes(title_text=xaxis, range=xrange, row=1, col=idx+1)
            fig.update_yaxes(title_text=yaxis, range=yrange, row=1, col=idx+1)
            if idx == 1:
                fig.update_yaxes(side="right", row=1, col=idx+1)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 50},
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5))
    fig.show()
    fig.write_image(f'env_trades.pdf')

    # model / environment benefit of SUUT
    plot1 = ['Power draw UUUT [Ws]', 'Top1 / mAP50 UUUT [%]', [-1, 20], [0.48, 0.8]]
    plot2 = ['Power draw SUUT [%]', 'Top1 / mAP50 SUUT [%]', [-0.02, 0.43], [0.75, 1.02]]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.02)
    for idx, (xaxis, yaxis, xrange, yrange) in enumerate([plot1 , plot2]):
        for env in envs:
            fig.add_trace(
                go.Scatter(x=time_acc_power[xaxis][env], y=time_acc_power[yaxis][env], name=env,
                        mode='markers', marker=dict(color=env_cols[env], symbol=env_symb[env]),
                        legendgroup=env, showlegend=idx==0), row=1, col=idx+1
            )
        fig.update_xaxes(title_text=xaxis, row=1, col=idx+1)
        fig.update_yaxes(title_text=yaxis, row=1, col=idx+1)
        if idx == 1:
            fig.update_yaxes(side="right", row=1, col=idx+1)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 50},
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5))
    fig.show()
    fig.write_image(f'env_suut_trades.pdf')
    

    # compound imagenet env trades
    fig = go.Figure(
        layout={'width': PLOT_WIDTH, 'height': PLOT_HEIGHT, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
                'yaxis':{'title': r'$\text{Overall performance } \bar M$'}},
        data=[
            go.Scatter(x=model_names, y=vals, name=env, mode='markers',
            marker=dict(
                color=env_cols[env],
                symbol=env_symb[env]
            ),) for i, (env, vals) in enumerate(others['comp'].items())
        ]
    )
    fig.update_layout( legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5) )
    fig.show()
    fig.write_image(f'environment_compound.pdf')

    # Star Plots

    for host, (db_name, model) in zip(HOSTS, MODELS):
        db, meta, metrics, _, _, _, _, _ = databases[db_name]
        metrics = metrics[SEL_DS_TASK[db_name]].tolist()
        metrics.append(metrics[0]) # fully connected stars
        metr_names = [lookup_meta(meta, metr, 'shortname', 'properties') for metr in metrics]
        envs = host_envs_map[host]
        db = prop_dict_to_val(db, 'index')
        fig = go.Figure()
        for env in envs:
            row = db[(db['model'] == model) & (db['environment'] == env)]
            if row.size > 0:
                fig.add_trace(go.Scatterpolar(
                    r=[row[col].iloc[0] for col in metrics], line={'color': env_cols[env]}, fillcolor=rbg_to_rgba(env_cols[env], 0.2),
                    theta=metr_names, fill='toself', name=r'$\text{{E}} : \bar M = {R}$'.replace('{E}', env.split()[1]).replace('{R}', f'{row["compound_index"].values[0]:4.2f}')
                ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)), width=PLOT_WIDTH*0.33, height=PLOT_HEIGHT, title_y=1.0, title_x=0.5, title_text=f'{model} on {host}',
            legend=dict( yanchor="bottom", y=1.02, xanchor="center", x=0.5), margin={'l': 30, 'r': 30, 'b': 5, 't': 80}
        )
        fig.show()
        fig.write_image(f'true_best_{host}.pdf')

    # scatter plots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    xaxis, yaxis, marker_width, ax_border = 'resource_index', 'quality_index', 6, 0.1
    for p_idx, (host, host_envs) in enumerate(host_envs_map.items()):
        plot_data, _, _ = assemble_scatter_data(host_envs, db, 'index', xaxis, yaxis, meta, bounds)
        i_min, i_max = min([min(vals['index']) for vals in plot_data.values()]), max([max(vals['index']) for vals in plot_data.values()])
        # link model scatter points across multiple environment
        models = set.union(*[set(data['names']) for data in plot_data.values()])
        x, y, text = [], [], []
        for model in models:
            avail = 0
            for data in plot_data.values():
                try:
                    idx = data['names'].index(model)
                    avail += 1
                    x.append(data['x'][idx])
                    y.append(data['y'][idx])
                except ValueError:
                    pass
            model_text = ['' if i != (avail - 1) // 2 else model for i in range(avail + 1)]
            text = text + model_text # place text near most middle node
            x.append(None)
            y.append(None)
        fig.add_trace(go.Scatter(x=x, y=y, text=text, mode='lines+markers+text', line={'color': 'black', 'width': marker_width / 5}, textposition='top center', showlegend=False), row=p_idx+1, col=1)
        for env_i, (env_name, data) in enumerate(plot_data.items()):
            index_vals = (np.array(data['index']) - i_min) / (i_max - i_min)
            node_col = sample_colorscale(RATING_COLOR_SCALE, [1-val for val in index_vals])
            fig.add_trace(go.Scatter(
                x=data['x'], y=data['y'], name=env_name.split()[1],
                mode='markers', marker_symbol=ENV_SYMBOLS[env_i],
                legendgroup=env_name.split()[1], marker=dict(color=node_col, size=marker_width),
                marker_line=dict(width=marker_width / 5, color='black'), showlegend=p_idx==0), row=p_idx+1, col=1
            )
        fig.update_yaxes(title_text=r'$\text{{H} - Quality average } \bar Q$'.replace('{H}', host), range=[0.82, 1.02], row=p_idx+1, col=1)
        if p_idx == len(host_envs_map) - 1:
            fig.update_xaxes(title_text=r'$\text{Resource average } \bar R$', range=[0.00, 1.02], row=p_idx+1, col=1)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT * 2.5, margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5))
    fig.show()
    fig.write_image(f"qual_res_all.pdf")

    
        # fig.update_layout(xaxis_title=axis_title[0], yaxis_title=axis_title[1])
        # fig.update_layout(legend=dict(x=.5, y=1, orientation="h", xanchor="center", yanchor="bottom",))
        # fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        # min_x, max_x = np.min([min(data['x']) for data in plot_data.values()]), np.max([max(data['x']) for data in plot_data.values()])
        # min_y, max_y = np.min([min(data['y']) for data in plot_data.values()]), np.max([max(data['y']) for data in plot_data.values()])
        # diff_x, diff_y = max_x - min_x, max_y - min_y
        # fig.update_layout(
        #     xaxis_range=[min_x - ax_border * diff_x, max_x + ax_border * diff_x],
        #     yaxis_range=[min_y - ax_border * diff_y, max_y + ax_border * diff_y],
        #     margin={'l': 10, 'r': 10, 'b': 10, 't': 10}
        # )
        # return fig




        # scatter = create_scatter_graph(plot_data, axis_names, marker_width=8, dark_mode=False)
        # scatter.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        #     xaxis=dict(showgrid=True), yaxis=dict(showgrid=True), yaxis_range=[0.82, 1.02], xaxis_range=[0.00, 1.02], 
        #     xaxis_title=r'$\text{Resource average } \bar R$', yaxis_title=r'$\text{Quality average } \bar Q$', 
        #     legend=dict(x=1, y=1, orientation="v", xanchor="right", yanchor="top"))

if __name__ == '__main__':
    pass