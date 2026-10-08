import plotly.io as pio


plotly_template = pio.templates["simple_white"]
plotly_template.layout.font.family = 'Helvetica'
plotly_template.layout.xaxis.mirror = True
plotly_template.layout.xaxis.exponentformat = 'power'
plotly_template.layout.yaxis.mirror = True
plotly_template.layout.yaxis.exponentformat = 'power'
plotly_template.layout.legend.bordercolor = 'Black'
plotly_template.layout.legend.borderwidth = 1
pio.templates.default = plotly_template
