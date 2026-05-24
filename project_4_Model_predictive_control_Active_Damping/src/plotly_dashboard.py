# src/plotly_dashboard.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def build_dashboard(data_mpc, data_pid, save_path='dashboard.html'):
    t1 = data_mpc['t']
    t2 = data_pid['t']
    # Ошибка позиции основной массы
    err1 = np.abs(data_mpc['states'][:, 0] - data_mpc['ref'][0])
    err2 = np.abs(data_pid['states'][:, 0] - data_pid['ref'][0])

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Position error of main mass (|x1 - ref|)',
                                        'Control force'))

    fig.add_trace(go.Scatter(x=t1, y=err1, mode='lines', name='MPC error'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t2, y=err2, mode='lines', name='PID error'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t1, y=data_mpc['controls'], mode='lines', name='MPC force'), row=2, col=1)

    fig.update_layout(title='Active Vibration Damping: MPC vs PID', height=600)
    fig.write_html(save_path, include_plotlyjs='cdn')
    print(f"[plotly] dashboard saved to {save_path}")