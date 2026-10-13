import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from config import PATH_TO_MATCHUPS, PATH_TO_FIGURES, DOWNLOAD_DATE
from src.shared.regions import get_region_name
from src.data_io import processed

colors = px.colors.qualitative.D3

plotly_template = pio.templates["simple_white"]
plotly_template.layout.font.family = 'Helvetica'
plotly_template.layout.xaxis.mirror = True
plotly_template.layout.xaxis.exponentformat = 'power'
plotly_template.layout.yaxis.mirror = True
plotly_template.layout.yaxis.exponentformat = 'power'
plotly_template.layout.legend.bordercolor = 'Black'
plotly_template.layout.legend.borderwidth = 1
# pio.renderers.default = "chrome"
# pio.renderers.default = 'svg'
pio.templates.default = plotly_template


path_to_fig = PATH_TO_FIGURES / 'report'
path_to_fig.mkdir(parents=True, exist_ok=True)

# %% Load data from Charlotte
a = processed.read('Kd').copy()
# m = pd.read_csv(PATH_TO_MATCHUPS / 'merged_kd_loc_good_matchups.csv', parse_dates=['date', 'oci_datetime'])
# m = pd.read_csv(PATH_TO_MATCHUPS / 'matchups_PACE_Kd_L2_CBD.csv', parse_dates=['date', 'oci_datetime']).rename(columns={'date': 'datetime'}, inplace=True)
m = pd.read_csv(PATH_TO_MATCHUPS / 'PACE_OCI_KD.matchups.csv', parse_dates=['datetime', 'oci_datetime'])
m.rename(columns={'WMO': 'wmo'}, inplace=True)
m.time_diff = pd.to_timedelta(m.time_diff)
if 'profile' not in m.columns:
    m['wmo'] = m['wmo'].astype(str) + '_' + m['profile'].astype(str).str.zfill(3)
mu = m.loc[m['time_diff'].abs().groupby(m['wmo']).idxmin()]  # Unique matchups with minimum time difference per profile

# %% Region
float_id = a['wmo'].astype(str).str.split('_').str[0]
profile_number = pd.to_numeric(
    a['wmo'].astype(str).str.rsplit('_', n=1).str[-1], errors='coerce'
)
first_profiles = (
    a.assign(_float_id=float_id, _profile_number=profile_number)
    .sort_values(['_float_id', '_profile_number'], kind='stable')
    .drop_duplicates('_float_id')
)
region_names = get_region_name(
    first_profiles['lat'].to_numpy(), first_profiles['lon'].to_numpy()
)
region = {
    float_id: name if name is not None else 'Unknown'
    for float_id, name in zip(first_profiles['_float_id'], region_names)
}

reg_color = {v: colors[i % len(colors)] for i, v in enumerate(sorted(set(region.values())))}

# %% Plot All Kd

def break_lines_at_wavelength_gaps(wavelengths, values, max_gap_nm=20):
    wavelengths = np.asarray(wavelengths, dtype=float)
    values = np.asarray(values, dtype=float)
    # Index (in the original arrays) before which a break is needed
    gaps = np.flatnonzero(np.diff(wavelengths) > max_gap_nm) + 1
    return np.insert(wavelengths, gaps, np.nan), np.insert(values, gaps, np.nan)

cwl = sorted([k for k in a.columns if k.startswith('kd') and '_' not in k])
cwl_unc = sorted([k for k in a.columns if k.startswith('kd') and k.endswith('_unc')])
wl = np.array([float(k[2:]) for k in cwl])
group_names, group_colors = [], {}
fig = go.Figure()
for idx, r in a.iterrows():
    if r.quality not in ['good', 'questionable']:
        continue
    # Select wavelength with valid Kd
    sel = ~pd.isna(r[cwl].to_numpy())
    if not np.any(sel):
        continue
    # Insert None when gap in wavelengths is too large
    r_wl, r_kd = break_lines_at_wavelength_gaps(wl[sel], r[cwl][sel].to_numpy(), max_gap_nm=20)
    # Show only one legend entry per float (grouped by wmo)
    group_name = r.wmo.split('_')[0]
    show_legend = group_name not in group_names
    if group_name not in group_colors:
        group_colors[group_name] = colors[len(group_colors) % len(colors)]
    group_color = group_colors[group_name]
    fig.add_scatter(x=r_wl, y=r_kd, mode='lines+markers',
                    marker_color=group_color, opacity=0.6,
                    showlegend=show_legend, name=group_name, legendgroup=group_name)
    if show_legend:
        group_names.append(group_name)
    # fig.add_scatter(x=wl, y=idx[1][cwl] + idx[1][cwl_unc], mode='lines', line_color=colors[7], showlegend=False)
    # fig.add_scatter(x=wl, y=idx[1][cwl] - idx[1][cwl_unc], mode='lines', line_color=colors[7], showlegend=False)
    # break
fig.update_xaxes(title='Wavelength (nm)', showgrid=True)
fig.update_yaxes(title='K<sub>d</sub> (m<sup>-1</sup>)', showgrid=True, zeroline=True) #, type='log', exponentformat='power')
fig.update_layout(legend={#'yanchor': 'top', 'y': 0.99, 'xanchor': "right", 'x': 0.99,
                          'tracegroupgap': 0})
fig.show()
fig.write_image(path_to_fig / 'Kd.all.png', width=1920/3*1.3, height=1200/2*1.3)
fig.update_xaxes(range=[320, 700])
fig.update_yaxes(range=[0, 1])
fig.show()
fig.write_image(path_to_fig / 'Kd.all.zoomed.png', width=1920/3*1.3, height=1200/2*1.3)

# %% Plot Spectral Matchups
# Argo columns and Wavelength
cwl = np.array(sorted([k for k in m.columns if k.startswith('kd') and '_' not in k]))
cwl_unc = sorted([k for k in m.columns if k.startswith('kd') and k.endswith('_unc')])
wl = np.array([float(k[2:]) for k in cwl])
# OCI columns and Wavelength
ocwl = sorted([k for k in m.columns if k.startswith('oci_kd') and not k.endswith('_unc')])
owl = np.array([float(k[6:]) for k in ocwl])

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
shown = []
group_colors = {}
for idx, r in mu.iterrows():
# for idx, r in m.iterrows():
    # Spectrum
    sel = ~pd.isna(r[cwl].to_numpy())
    qc_sel = ~pd.isna(r[cwl[(490 <= wl) & (wl <= 530) ]].to_numpy())
    if np.sum(sel) < 2: # or np.sum(qc_sel) < 3:
        continue
    osel = ~pd.isna(r[ocwl].to_numpy())
    if not np.any(osel):
        continue
    # # Skip float with QC issue
    # if r.wmo.split('_')[0] in ['6990514']:
    #     continue
    reg = region[str(r.wmo).split('_')[0]]
    # color = reg_color[reg]
    # color = colors[idx % len(colors)]
    if r.wmo not in group_colors:
        group_colors[r.wmo] = colors[len(group_colors) % len(colors)]
    color = group_colors[r.wmo]
    if reg in shown:
        show = False
    else:
        show = True
        shown.append(reg)
    # if reg != 'North Atlantic':
    #     continue
    # Handle colors
    hours_diff = r.time_diff.total_seconds() / 3600
     # Argo
    r_wl, r_kd = break_lines_at_wavelength_gaps(wl[sel], r[cwl][sel].to_numpy(), max_gap_nm=20)
    fig.add_scatter(x=r_wl, y=r_kd, mode='lines', opacity=1, #legendgroup=idx,
                    marker_color=color, showlegend=True, legendgroup=r.wmo,
                    name=f'{r.wmo} {r.quality}', row=1, col=1)

    # OCI
    fig.add_scatter(x=owl[osel], y=r[ocwl][osel].to_numpy(), mode='lines+markers', opacity=0.5, #legendgroup=idx,
                    marker_color=color, showlegend=False, legendgroup=r.wmo,
                    name=f'OCI {hours_diff:.1f} hrs', row=1, col=1)

    # Delta
    y_oci = r[ocwl][osel].to_numpy()
    y_argo = np.interp(owl[osel], wl[sel], r[cwl][sel].to_numpy(float), left=np.nan, right=np.nan)
    fig.add_scatter(x=owl[osel], y=(y_oci - y_argo) / (0.5 * (y_argo + y_oci)) * 100, mode='lines+markers', #legendgroup=idx,
                    marker_color=color, showlegend=False, legendgroup=r.wmo,
                    name=f'OCI {hours_diff:.1f} hrs', row=2, col=1)

    # fig.add_scatter(x=wl, y=idx[1][cwl] + idx[1][cwl_unc], mode='lines', line_color=colors[7], showlegend=False)
    # fig.add_scatter(x=wl, y=idx[1][cwl] - idx[1][cwl_unc], mode='lines', line_color=colors[7], showlegend=False)
    # break

fig.update_xaxes(showgrid=True, row=1, col=1)
fig.update_xaxes(showticklabels=True, row=1, col=1)
fig.update_xaxes(title='Wavelength (nm)', showgrid=True, row=2, col=1)
fig.update_yaxes(title='K<sub>d</sub> (1/m)', range=[0, 0.8], showgrid=True, zeroline=True, row=1, col=1)
fig.update_yaxes(title='Relative Difference (%)', showgrid=True, zeroline=True, row=2, col=1)
fig.update_layout(legend={#'yanchor': 'bottom', 'y': 0.56, 'xanchor': "right", 'x': 0.99,
                          'tracegroupgap': 0})
fig.show()
fig.write_image(path_to_fig / 'Kd.matchups.png', width=1920/3*1.3, height=1200/2*1.3)

# %% Plot Map
fig = go.Figure()
fig.add_scattermap(
    lat=a['lat'], lon=a['lon'], text=a['wmo'], hovertemplate='WMO: %{text}<extra></extra>',
    mode='markers', marker=dict(color=colors[7]), name='All Profiles'
)
p = a[a.datetime > datetime(2024, 2, 8)]
fig.add_scattermap(
    lat=p['lat'], lon=p['lon'], text=p['wmo'], hovertemplate='WMO: %{text}<extra></extra>',
    mode='markers', marker=dict(color=colors[0]), name='Profiles since Feb 8, 2024'
)
fig.add_scattermap(
    lat=m['lat'], lon=m['lon'], text=m['wmo'], hovertemplate='WMO: %{text}<extra></extra>',
    mode='markers', marker=dict(color=colors[2]), name='PACE-OCI matchups'
)
fig.update_layout(
    title='VDIUP Matchups',
    # map_style="satellite",
    map_style="light", map_zoom=1.0, map_center={'lat': 20, 'lon': -40},
    # map_style="https://api.maptiler.com/maps/ocean/style.json?key=rLdmcaCX89R2SrWw2Qk8"
    margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
    legend={'yanchor': 'top', 'y': 0.985, 'xanchor': "right", 'x': 0.985, 'tracegroupgap': 0},
)
fig.show()
# fig.write_image(path_to_fig / f'map.{DOWNLOAD_DATE}.png', width=1920, height=1200)  # zoom=1.9
fig.write_image(path_to_fig / f'map.{DOWNLOAD_DATE}.png', width=1920, height=1080)  # zoom=1.0

# %% Stats
print(f'Number of profiles: {len(a)}')
print(f'Number of profiles since Feb 8, 2024: {len(a[a.datetime > datetime(2024, 2, 8)])}')
print(f'Number of unique floats: {len(set(a.wmo.astype(str).str.split("_").str[0]))}')
print(f'Number of unique floats since Feb 8, 2024: {len(set(a[a.datetime > datetime(2024, 2, 8)].wmo.astype(str).str.split("_").str[0]))}')
print(f'Number of good Kd: {len(a[a.quality == "good"])} ({len(a[a.quality == "good"]) / len(a) * 100:.1f}%)')
print(f'Number of questionable Kd: {len(a[a.quality == "questionable"])} ({len(a[a.quality == "questionable"]) / len(a) * 100:.1f}%)')
print(f'Number of bad Kd: {len(a[a.quality == "bad"])} ({len(a[a.quality == "bad"]) / len(a) * 100:.1f}%)')
print(f'Number of matchups: {len(m)}')
print(f'Number of matchups with complete spectra: {np.sum([False if (
    np.sum(~pd.isna(r[cwl].to_numpy())) < 15 or 
    np.sum(~pd.isna(r[cwl[(490 <= wl) & (wl <= 530) ]].to_numpy())) < 3
) else True for idx, r in m.iterrows()])}')
