import os
import re
import warnings
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.template import plotly_template


def read_sb(filename):
    with open(filename) as f:
        d = f.read()
    rows = re.split('\n/|=', d)
    header = {rows[i]: rows[i+1] for i in range(1, len(rows)-1, 2)}
    fields = header['fields'].split(',')
    cols = pd.MultiIndex.from_tuples([(f'kd{f[7:]}', float(f[2:7])) if f.startswith('kd') else (f, '') for f in fields])
    df = pd.read_csv(StringIO(rows[-1].strip('end_header\n')), names=cols, na_values=-9999).copy()
    # Format date & time
    df.insert(1, ('datetime', ''), pd.to_datetime(df['date'].astype(str) + df['time'], format='%Y%m%d%H:%M:%S'))
    df.drop(columns=[('date', ''), ('time', '')], inplace=True)
    # Check fields comply
    for f in ['profile', 'date', 'time', 'lon', 'lat', 'quality']:
        if f not in fields:
            raise KeyError(f"{os.path.basename(f)}: missing field {f} in file.")
    for e in ['_unc', '_se', '_bincount']:
        try:
            if not (df['kd'].columns == df[f'kd{e}'].columns).all():
                raise KeyError(f"{os.path.basename(f)}: missing wavelength kd{e} in file.")
        except ValueError as ex:
            missing_cols = []
            for c in df['kd'].columns:
                if c not in df[f'kd{e}'].columns:
                    missing_cols.append(c)
            raise ValueError(f"{os.path.basename(filename)}: {ex} kd and kd{e} columns. Missing wavelength(s): {missing_cols}")
    return df, header


QUALITY_STR = {0: 'good', 1: 'questionable', 2: 'bad'}
QUALITY_COL = {0: "#2CA02C", 1: "#FF7F0E", 2: "#D62728"}

def make_plot(filenames):
    n, platform_ids = 0, []
    row2_max, row3_max = 0, 0
    # fig = go.Figure()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    for f in filenames:
        df, header = read_sb(f)
        platform_ids.append(header['platform_id'])
        wl = df.kd.columns.to_numpy(float)
        # Hide High Red Uncertainty
        m = df['kd_se'].loc[:, wl < 680].max().max()
        if not pd.isna(m) and m > row2_max:
            row2_max = m
        m = df['kd_unc'].loc[:, wl < 680].max().max()
        if not pd.isna(m) and m > row3_max:
            row3_max = m
        for _, r in df.iterrows():
            n += 1
            if r.kd.isna().all():
                continue
            p, q = r.profile.item(), r.quality.item()
            # fig.add_scatter(x=wl, y=r.kd + r.kd_se, mode='lines', opacity=0.5, legendgroup=f"{header['platform_id']}.{p}",
            #                 line=dict(width=0.5, color=QUALITY_COL[q]), showlegend=False, name=f"+se")
            # fig.add_scatter(x=wl, y=r.kd - r.kd_se, mode='lines', opacity=0.5, legendgroup=f"{header['platform_id']}.{p}",
            #                 line=dict(width=0.5, color=QUALITY_COL[q]), showlegend=False, fill='tonexty', name=f"-se")
            # fig.add_scatter(x=wl, y=r.kd + r.kd_unc, mode='lines', opacity=0.25, legendgroup=f"{header['platform_id']}.{p}",
            #                 line=dict(width=0.5, color=QUALITY_COL[q]), showlegend=False, name=f"+unc")
            # fig.add_scatter(x=wl, y=r.kd - r.kd_unc, mode='lines', opacity=0.25, legendgroup=f"{header['platform_id']}.{p}",
            #                 line=dict(width=0.5, color=QUALITY_COL[q]), showlegend=False, fill='tonexty', name=f"-unc")
            fig.add_scatter(x=wl, y=r.kd, mode='lines+markers', opacity=1, legendgroup=f"{header['platform_id']}.{p}",
                            marker_color=QUALITY_COL[q], showlegend=True,
                            name=f"{header['platform_id']} {p:03d} {QUALITY_STR[q]}", row=1, col=1)
            fig.add_scatter(x=wl, y=r.kd_se, mode='lines+markers', opacity=1, legendgroup=f"{header['platform_id']}.{p}",
                            marker_color=QUALITY_COL[q], showlegend=False,
                            name=f"{header['platform_id']} {p:03d} {QUALITY_STR[q]}", row=2, col=1)
            fig.add_scatter(x=wl, y=r.kd_unc, mode='lines+markers', opacity=1, legendgroup=f"{header['platform_id']}.{p}",
                            marker_color=QUALITY_COL[q], showlegend=False,
                            name=f"{header['platform_id']} {p:03d} {QUALITY_STR[q]}", row=3, col=1)
    fig.update_xaxes(title='Wavelength (nm)', row=3)
    fig.update_xaxes(showticklabels=True, showgrid=True)
    fig.update_yaxes(title=f'Kd (1/m)', showgrid=True, row=1)
    fig.update_yaxes(title=f'Kd_se (1/m)', showgrid=True, row=2)#, range=[0, row2_max*1.05])
    fig.update_yaxes(title=f'Kd_unc (1/m)', showgrid=True, row=3)#, range=[0, row3_max*1.05])
    fig.update_layout(title=f"{','.join(set(platform_ids))} \t n={n}")
                      #margin={'l': 0.05, 't': 0.1, 'b': 0.05, 'r': 0.05})
                      # legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99, tracegroupgap=0))
    return fig
