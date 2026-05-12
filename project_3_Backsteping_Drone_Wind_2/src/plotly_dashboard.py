# src/plotly_dashboard.py
import numpy as np


def build_dashboard(data_pid, data_bs, label_pid='PID', label_bs='Backstepping',
                    target=None, save_path='dashboard.html',
                    title='1D Height Stabilization'):
    """
    Простой Plotly‑график: две траектории высоты + слайдер времени.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise RuntimeError("plotly required. Install with `pip install plotly`.")

    t_pid = data_pid['t']; z_pid = data_pid['states'][:, 0]
    t_bs  = data_bs['t'];  z_bs  = data_bs['states'][:, 0]

    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=['Drone height'],
                        specs=[[{"secondary_y": False}]])

    # Статические линии
    fig.add_trace(go.Scatter(x=t_pid, y=z_pid, mode='lines',
                             name=label_pid, line=dict(color='gray', dash='dot')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t_bs, y=z_bs, mode='lines',
                             name=label_bs, line=dict(color='blue')),
                  row=1, col=1)
    if target is not None:
        fig.add_hline(y=target, line_dash="dash", line_color="red",
                      annotation_text=f"z_des = {target:.1f} m")

    # Анимированные точки (последнее положение)
    fig.add_trace(go.Scatter(x=[t_pid[0]], y=[z_pid[0]], mode='markers',
                             marker=dict(size=8, color='gray'),
                             showlegend=False, name=f'{label_pid}_now'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[t_bs[0]], y=[z_bs[0]], mode='markers',
                             marker=dict(size=10, color='blue'),
                             showlegend=False, name=f'{label_bs}_now'),
                  row=1, col=1)

    # Слайдер
    frames = []
    n = max(len(t_pid), len(t_bs))
    t_uniform = np.linspace(0, max(t_pid[-1], t_bs[-1]), n)
    # интерполируем на общую сетку для слайдера (упрощённо – используем исходные точки с прореживанием)
    stride = max(1, n // 150)
    for i in range(0, n, stride):
        if i >= len(t_pid): i_pid = len(t_pid)-1
        else: i_pid = i
        if i >= len(t_bs): i_bs = len(t_bs)-1
        else: i_bs = i

        frames.append(go.Frame(
            data=[
                go.Scatter(x=[t_pid[i_pid]], y=[z_pid[i_pid]]),
                go.Scatter(x=[t_bs[i_bs]],   y=[z_bs[i_bs]])
            ],
            name=f't={t_uniform[i]:.2f}'
        ))

    fig.frames = frames

    fig.update_layout(
        title=title,
        xaxis_title='Time [s]',
        yaxis_title='Height [m]',
        updatemenus=[dict(type='buttons', showactive=False,
                          buttons=[
                              dict(label='▶', method='animate',
                                   args=[None, dict(frame=dict(duration=50, redraw=True),
                                                    fromcurrent=True)]),
                              dict(label='⏸', method='animate',
                                   args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                      mode='immediate')])
                          ])],
        sliders=[dict(steps=[dict(method='animate',
                                  args=[[f.name], dict(mode='immediate',
                                                       frame=dict(duration=0, redraw=True))],
                                  label=f.name) for f in frames])]
    )

    fig.write_html(save_path, include_plotlyjs='cdn', auto_play=False)
    print(f"[plotly] dashboard saved to {save_path}")
    return fig


def build_dashboard_2d(data_pd, data_bs, label_pd='PD + inversion', label_bs='Backstepping',
                       save_path='dashboard_2d.html', title='2D velocity tracking'):
    """Траектории x–z для двух регуляторов + слайдер по времени."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise RuntimeError("plotly required. Install with `pip install plotly`.")

    x1, z1 = data_pd['states'][:, 0], data_pd['states'][:, 1]
    x2, z2 = data_bs['states'][:, 0], data_bs['states'][:, 1]
    t1, t2 = data_pd['t'], data_bs['t']

    fig = make_subplots(rows=1, cols=1, subplot_titles=['x–z plane'])

    fig.add_trace(go.Scatter(x=x1, y=z1, mode='lines', name=label_pd,
                             line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=x2, y=z2, mode='lines', name=label_bs,
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[x1[0]], y=[z1[0]], mode='markers',
                             marker=dict(size=8, color='gray'), showlegend=False))
    fig.add_trace(go.Scatter(x=[x2[0]], y=[z2[0]], mode='markers',
                             marker=dict(size=10, color='blue'), showlegend=False))

    n = max(len(t1), len(t2))
    stride = max(1, n // 150)
    frames = []
    for i in range(0, n, stride):
        i1 = min(i, len(t1) - 1)
        i2 = min(i, len(t2) - 1)
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x1, y=z1, mode='lines', name=label_pd,
                           line=dict(color='gray', dash='dot')),
                go.Scatter(x=x2, y=z2, mode='lines', name=label_bs,
                           line=dict(color='blue')),
                go.Scatter(x=[x1[i1]], y=[z1[i1]], mode='markers',
                           marker=dict(size=8, color='gray'), showlegend=False),
                go.Scatter(x=[x2[i2]], y=[z2[i2]], mode='markers',
                           marker=dict(size=10, color='blue'), showlegend=False),
            ],
            name=f't={max(t1[i1], t2[i2]):.2f}',
        ))

    fig.frames = frames
    fig.update_xaxes(title_text='x [m]')
    fig.update_yaxes(title_text='z [m]', scaleanchor='x', scaleratio=1)

    fig.update_layout(
        title=title,
        updatemenus=[dict(type='buttons', showactive=False,
                          buttons=[
                              dict(label='▶', method='animate',
                                   args=[None, dict(frame=dict(duration=40, redraw=True),
                                                    fromcurrent=True)]),
                              dict(label='⏸', method='animate',
                                   args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                      mode='immediate')]),
                          ])],
        sliders=[dict(steps=[dict(method='animate',
                                  args=[[f.name], dict(mode='immediate',
                                                       frame=dict(duration=0, redraw=True))],
                                  label=f.name) for f in frames])],
    )

    fig.write_html(save_path, include_plotlyjs='cdn', auto_play=False)
    print(f"[plotly] 2D dashboard saved to {save_path}")
    return fig