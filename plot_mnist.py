import plotly.graph_objects as go
import numpy as np
import os
from  mlflow.tracking import MlflowClient
from collections import defaultdict

client = MlflowClient()
experiments = client.list_experiments() 
exp_id = experiments[0].experiment_id

run_infos = client.list_run_infos(exp_id)

metric_lbls = [ "accuracy", "loss" ]
name_seq = [ "tpe-network", "pe-network", "smpc-network",  "une-network",  "une-pytorch","une-framework" ]

run_time_dict = defaultdict(list)
metrics = defaultdict(lambda : defaultdict(list))
for run_info in run_infos:

    # Throw caution
    if not run_info.status == "FINISHED":
        print(run_info)
        continue

    run = client.get_run(run_info.run_id)
    run_name = run.data.tags['mlflow.runName']

    # Filter those unecessary ones away.
    if not run_name.startswith("MNIST"):
        continue

    if run_name.endswith("networked") and (not run_name.endswith("ba-networked")):
        continue

    run_name = run_name.replace('MNIST-', '')
    run_name = run_name.replace('-ba', '')
    run_name = run_name.replace('networked', 'network')
    # This is now the key for which the runs will merge upon

    # In seconds
    run_time = (run_info.end_time - run_info.start_time)/1000.0

    run_time_dict[run_name].append(run_time)

    print(run_name)
    print(run_time)
    print(run.data.metrics)
    
    for metric_lbl in metric_lbls:
        metric_hist = client.get_metric_history(run_info.run_id, metric_lbl)
        metric_hist = [ m.value for m in metric_hist ]
        metrics[metric_lbl][run_name].append(metric_hist)
   
# Plot the time taken
# ===================
print(run_time_dict)
def plot_timetaken():
    xaxis_title = "Implementation"
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
    fig = go.Figure(layout=layout)
    
    x = []
    y = []
    error_y = []
    text = []
    total = np.mean(run_time_dict["une-pytorch"])
    for ea_k in name_seq:
        ea_v = run_time_dict[ea_k]
        if ea_k == "une-pytorch":
            text.append("1x")
        else:
            p = np.mean(ea_v) / total
            text.append("{:.1f}x".format(p))

        x.append(ea_k)
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
    fig.write_image("images/mnist_time.pdf", format='pdf', width=399, height=266)

plot_timetaken()

colors = ['#636EFA',
 '#EF553B',
 '#00CC96',
 '#AB63FA',
 '#FFA15A',
 '#19D3F3',
 '#FF6692',
 '#B6E880',
 '#FF97FF',
 '#FECB52']
def plot_metric(metric_lbl):
    xaxis_title = "Iterations"
    yaxis_title = metric_lbl.capitalize()
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
            y=0.01,
            orientation='v',
            yanchor='bottom',
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
    fig = go.Figure(layout=layout)
    error_lst = []
    
    for s, ea_k in enumerate(name_seq):
        ea_v = metrics[metric_lbl][ea_k]

        x = list(range(250))
        y = np.mean(ea_v, axis=0)
        
        error_y = np.std(ea_v, axis=0) * 0.25
        stepping = int(len(x) / 10)
        
        err_x = []
        err_y = []
        err_bar = []
        for i in x:
            if i > 0 and i % stepping == 0:
                err_x.append(x[i])
                err_y.append(y[i])
                err_bar.append(error_y[i])

        fig.add_trace(go.Scatter(x=x, y=y, name=ea_k, line_color=colors[s]))
        fig.add_trace(
            go.Scatter(
                        x=err_x, 
                        y=err_y,
                        mode='markers',
                        name='Err-{}'.format(ea_k),
                        error_y=dict(
                            type='data',
                            visible=True,
                            array=err_bar,
                            thickness=1
                        ),
                        showlegend=False,
                        opacity=1,
                        marker=dict(size=1, opacity=0), 
                        line_color='grey'
                    )
        )

    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image(f"images/mnist_{metric_lbl}.pdf", format='pdf', width=399, height=266)

plot_metric("accuracy")
plot_metric("loss")