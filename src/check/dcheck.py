import pandas as pd
from plotly.subplots import make_subplots

import src.data_io.seabass as seabass
from config import QUALITY


QUALITY_COL = {0: "#2CA02C", 1: "#FF7F0E", 2: "#D62728"}

def make_plot(filenames):
    n, platform_ids = 0, []
    row2_max, row3_max = 0, 0
    # fig = go.Figure()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    for f in filenames:
        df, header = seabass.read(f)
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
                            name=f"{header['platform_id']} {p:03d} {QUALITY[q]}", row=1, col=1)
            fig.add_scatter(x=wl, y=r.kd_se, mode='lines+markers', opacity=1, legendgroup=f"{header['platform_id']}.{p}",
                            marker_color=QUALITY_COL[q], showlegend=False,
                            name=f"{header['platform_id']} {p:03d} {QUALITY[q]}", row=2, col=1)
            fig.add_scatter(x=wl, y=r.kd_unc, mode='lines+markers', opacity=1, legendgroup=f"{header['platform_id']}.{p}",
                            marker_color=QUALITY_COL[q], showlegend=False,
                            name=f"{header['platform_id']} {p:03d} {QUALITY[q]}", row=3, col=1)
    fig.update_xaxes(title='Wavelength (nm)', row=3)
    fig.update_xaxes(showticklabels=True, showgrid=True)
    fig.update_yaxes(title=f'Kd (1/m)', showgrid=True, row=1)
    fig.update_yaxes(title=f'Kd_se (1/m)', showgrid=True, row=2)#, range=[0, row2_max*1.05])
    fig.update_yaxes(title=f'Kd_unc (1/m)', showgrid=True, row=3)#, range=[0, row3_max*1.05])
    fig.update_layout(title=f"{','.join(set(platform_ids))} \t n={n}")
                      #margin={'l': 0.05, 't': 0.1, 'b': 0.05, 'r': 0.05})
                      # legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99, tracegroupgap=0))
    return fig
