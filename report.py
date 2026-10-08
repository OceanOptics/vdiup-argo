import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

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


# %% Load data from Charlotte
root = '/Volumes/SD/VDIUP/Argo/matchups/'
a = pd.read_csv(root + 'All_Kd_profiles.csv', parse_dates=['date'])
# m = pd.read_csv(root +'merged_kd_loc_good_matchups.csv', parse_dates=['date', 'oci_datetime'])
m = pd.read_csv(root +'matchups_PACE_Kd_L2.csv', parse_dates=['date', 'oci_datetime'])

# %% Region
region = {
    '1902637': 'North Atlantic',
    '1902695': 'North Atlantic',
    '4903739': 'North Atlantic',
    '4903740': 'North Atlantic',
    '1902601': 'Equatorial Atlantic',
    '1902685': 'North Pacific',
    '4903774': 'Baffin Bay',
    # '1902578': 'Labrador Sea',
    '4903660': 'Indian Ocean',
    '6990503': 'Indian Ocean',
    '6990514': 'Indian Ocean',
    '2903787': 'Southern Ocean',
    # '4903658': 'Southern Ocean',
    # '6903124': 'Mediterranean Sea',
    # '6903706': 'Baltic Sea',
}

reg_color = {v: colors[i] for i, v in enumerate(sorted(set(region.values())))}

# %% Plot All Kd
cwl = sorted([k for k in a.columns if k.startswith('kd') and not k.endswith('_unc')])
cwl_unc = sorted([k for k in a.columns if k.startswith('kd') and k.endswith('_unc')])
wl = np.array([float(k[2:]) for k in cwl])
fig = go.Figure()
for idx, r in a.iterrows():
    sel = ~pd.isna(r[cwl].to_numpy())
    if not np.any(sel):
        continue
    fig.add_scatter(x=wl[sel], y=r[cwl][sel].to_numpy(), mode='lines+markers', showlegend=True)
    # fig.add_scatter(x=wl, y=idx[1][cwl] + idx[1][cwl_unc], mode='lines', line_color=colors[7], showlegend=False)
    # fig.add_scatter(x=wl, y=idx[1][cwl] - idx[1][cwl_unc], mode='lines', line_color=colors[7], showlegend=False)
    # break
fig.update_xaxes(title='Wavelength (nm)')
fig.update_yaxes(title='Kd (1/m)')
fig.show()

# %% Plot Spectral Matchups
# Argo columns and Wavelength
cwl = np.array(sorted([k for k in m.columns if k.startswith('kd') and not k.endswith('_unc')]))
cwl_unc = sorted([k for k in m.columns if k.startswith('kd') and k.endswith('_unc')])
wl = np.array([float(k[2:]) for k in cwl])
# OCI columns and Wavelength
ocwl = sorted([k for k in m.columns if k.startswith('oci_kd') and not k.endswith('_unc')])
owl = np.array([float(k[6:]) for k in ocwl])

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
shown = []
for idx, r in m.iterrows():
    # Spectrum
    sel = ~pd.isna(r[cwl].to_numpy())
    qc_sel = ~pd.isna(r[cwl[(490 <= wl) & (wl <= 530) ]].to_numpy())
    if np.sum(sel) < 2: # or np.sum(qc_sel) < 3:
        continue
    osel = ~pd.isna(r[ocwl].to_numpy())
    if not np.any(osel):
        continue
    # # Skip float with QC issue
    # if r.WMO.split('_')[0] in ['6990514']:
    #     continue
    reg = region[r.WMO.split('_')[0]]
    color = reg_color[reg]
    # color = colors[idx % len(colors)]
    if reg in shown:
        show = False
    else:
        show = True
        shown.append(reg)
    # if reg != 'North Atlantic':
    #     continue
     # Argo
    fig.add_scatter(x=wl[sel], y=r[cwl][sel].to_numpy(), mode='lines+markers', opacity=1, #legendgroup=idx,
                    marker_color=color, showlegend=True, name=f'{r.WMO} {r.quality}', row=1, col=1)

    # OCI
    fig.add_scatter(x=owl[osel], y=r[ocwl][osel].to_numpy(), mode='lines+markers', opacity=0.5, #legendgroup=idx,
                    marker_color=color, showlegend=False, name=f'OCI', row=1, col=1)

    # Delta
    y_oci = r[ocwl][osel].to_numpy()
    y_argo = np.interp(owl[osel], wl[sel], r[cwl][sel].to_numpy(float), left=np.nan, right=np.nan)
    fig.add_scatter(x=owl[osel], y=(y_oci - y_argo) / (0.5 * (y_argo + y_oci)) * 100, mode='lines+markers', #legendgroup=idx,
                    marker_color=color, showlegend=False, name=f'OCI', row=2, col=1)

    # fig.add_scatter(x=wl, y=idx[1][cwl] + idx[1][cwl_unc], mode='lines', line_color=colors[7], showlegend=False)
    # fig.add_scatter(x=wl, y=idx[1][cwl] - idx[1][cwl_unc], mode='lines', line_color=colors[7], showlegend=False)
    # break

fig.update_xaxes(showgrid=True, row=1, col=1)
fig.update_xaxes(title='Wavelength (nm)', showgrid=True, row=2, col=1)
fig.update_yaxes(title='K<sub>d</sub> (1/m)', range=[0, 0.8], showgrid=True, zeroline=True, row=1, col=1)
fig.update_yaxes(title='Relative Difference (%)', showgrid=True, zeroline=True, row=2, col=1)
fig.update_layout(legend={'yanchor': 'bottom', 'y': 0.56, 'xanchor': "right", 'x': 0.99, 'tracegroupgap': 0})
fig.show()
# fig.write_image(os.path.join(root, 'spectrum.png'), width=1920/3*1.3, height=1200/2*1.3)

# %% Plot Map
fig = go.Figure()
fig.add_scattermap(lat=a['lat'], lon=a['lon'], mode='markers', marker=dict(color=colors[7]), name='All Profiles')
p = a[a.date > datetime(2024, 2, 8)]
fig.add_scattermap(lat=p['lat'], lon=p['lon'], mode='markers', marker=dict(color=colors[0]), name='Profiles since Feb 8, 2024')
fig.add_scattermap(lat=m['lat'], lon=m['lon'], text=m['WMO'], mode='markers', marker=dict(color=colors[2]), name='PACE-OCI matchups')
fig.update_layout(
    title='VDIUP Matchups',
    # map_style="satellite",
    map_style="light", map_zoom=1.0, map_center={'lat': 15, 'lon': -40},
    # map_style="https://api.maptiler.com/maps/ocean/style.json?key=rLdmcaCX89R2SrWw2Qk8"
    margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
    legend={'yanchor': 'top', 'y': 0.985, 'xanchor': "right", 'x': 0.985, 'tracegroupgap': 0},
)
fig.show()
# fig.write_image(os.path.join(root, 'map.20250213.png'), width=1920, height=1200)  # zoom=1.9
# fig.write_image(os.path.join(root, 'map.20250213.png'), width=1920/2, height=1200/2)  # zoom=1.0

# %% Stats
print(f'Number of profiles: {len(a)}')
print(f'Number of profiles since Feb 8, 2024: {len(a[a.date > datetime(2024, 2, 8)])}')
print(f'Number of matchups: {len(m)}')
print(f'Number of matchups with complete spectra: {np.sum([False if (
    np.sum(~pd.isna(r[cwl].to_numpy())) < 15 or 
    np.sum(~pd.isna(r[cwl[(490 <= wl) & (wl <= 530) ]].to_numpy())) < 3
) else True for idx, r in m.iterrows()])}')
