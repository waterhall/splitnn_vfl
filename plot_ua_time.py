import plotly.graph_objects as go
import numpy as np
import os

np.random.seed(1)

N = 100
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
sz = np.random.rand(N) * 30

xaxis_title = "Segment"
yaxis_title = "Seconds"
CHART_CONTROL_COLOR = "black"
layout = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(
        b=0,
        t=0,
        l=0,
        r=0,
        pad=0
    ),
    legend=dict(
        x=0.995,
        y=0.99,
        orientation='v',
        yanchor='top',
        xanchor='right',
        bgcolor='rgba(254,254,254,0.9)',
        bordercolor=CHART_CONTROL_COLOR,
        borderwidth=1
        ),
    showlegend=True,
    xaxis=dict(
        title='<b>{}</b>'.format(xaxis_title),
        zeroline=True,
        showticklabels=True,
        mirror=True,
        ticks='outside',
        showline=True,
        rangemode = "tozero",
        linecolor=CHART_CONTROL_COLOR,
        gridcolor="#CCCCCC",
    ),
    yaxis=dict(
        title='<b>{}</b>'.format(yaxis_title),
        zeroline=True,
        showticklabels=True,
        mirror=True,
        ticks='outside',
        showline=True,
        rangemode = "tozero",
        linecolor=CHART_CONTROL_COLOR,
        gridcolor="#CCCCCC",
    ), 
    template="plotly_white",
    font=dict(
        color=CHART_CONTROL_COLOR
    )
)

metric_lbls = [ "align_recv_time", "align_send_time", "align_total_time"]

dataset_names = ["train", "test"]

fig = go.Figure(layout=layout)

from  mlflow.tracking import MlflowClient
client = MlflowClient()
experiments = client.list_experiments() 
exp_id = experiments[0].experiment_id

run_infos = client.list_run_infos(exp_id)
from collections import defaultdict
acc_dict = defaultdict(list)

for run_info in run_infos:
    run = client.get_run(run_info.run_id)
    print(run.data.metrics)
    
    for metric_lbl in metric_lbls:
        metric_val = 0
        for ea_db_name in dataset_names:
            metric_val += run.data.metrics[f"{ea_db_name}_{metric_lbl}"]
        acc_dict[f"{metric_lbl}"].append(metric_val)


compute_time = np.array(acc_dict["align_total_time"]) - np.array(acc_dict["align_send_time"])  - np.array(acc_dict["align_recv_time"])
#del acc_dict["align_total_time"]
acc_dict["align_compute_time"] = compute_time

pretty_lbl = {
    "align_recv_time": "Recieve", 
    "align_send_time": "Send",
    "align_compute_time": "UA Compute",
    "align_total_time": "Total"
}

seq_metric_lbls = [ "align_total_time", "align_compute_time", "align_send_time", "align_recv_time"]
x = []
y = []
error_y = []
text = []
total = 0
for ea_k in seq_metric_lbls:
    ea_v = acc_dict[ea_k]
    if pretty_lbl[ea_k] == "Total":
        total = np.mean(ea_v)
        text.append("100%")
    else:
        p = int(np.mean(ea_v) / total * 100)
        text.append(f"{p}%")

    x.append(pretty_lbl[ea_k])
    y.append(np.mean(ea_v))
    error_y.append(np.std(ea_v)*0.25)

fig.add_trace(go.Bar(
    name='MNIST',
    x=x, 
    y=y,
    error_y=dict(type='data', array=error_y),
    text=text,
    textposition='auto',
    
))

if not os.path.exists("images"):
    os.mkdir("images")


fig.write_image("images/ua_time.pdf", format='pdf', width=399, height=266)