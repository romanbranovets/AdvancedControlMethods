# src/visualization.py
"""
Здание с активным гасителем колебаний (TMD) на крыше + видимый фундамент.

Физическая интерпретация (как в схеме задачи):
    - Здание m₁ жёстко стоит на фундаменте, наполовину вкопанном в грунт.
    - Фундамент соединён с окружающим грунтом пружиной k₁ и демпфером c₁
      (Kelvin-Voigt, параллельно). Сам грунт неподвижен; пружины зарыты в землю.
    - Внешняя сила d(t) приложена непосредственно к фундаменту
      (сейсмический толчок / удар у основания), а не к стене здания.
    - На крыше — TMD m₂ на пружине k₂ и демпфере c₂.
    - Управление u(t) — сила, развиваемая актуатором между крышей и TMD.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyArrowPatch


# --- палитра ---------------------------------------------------------------
COLOR_BUILDING = '#4a7ab8'
COLOR_FOUNDATION = '#6e6e6e'
COLOR_TMD = '#d65f5f'
COLOR_SPRING = '#1f4f80'
COLOR_DAMPER = '#3a3a3a'
COLOR_ACTUATOR = '#ff9933'
COLOR_FORCE = '#c62828'
COLOR_BG = '#e9eef5'


# ---------------------------------------------------------------------------
# Графические примитивы
# ---------------------------------------------------------------------------
def _h_spring(x0, x1, y, num_coils=6, amp=0.06):
    if abs(x1 - x0) < 1e-4:
        return np.array([x0, x1]), np.array([y, y])
    n = num_coils * 4 + 2
    xs = np.linspace(x0, x1, n)
    ys = y + amp * np.sin(np.linspace(0, num_coils * 2 * np.pi, n))
    ys[0] = ys[-1] = y
    return xs, ys


def _dashpot_coords(anchor_x, attach_x, y, length, height, opens_right=True):
    h, L = height, length
    if opens_right:
        x_closed, x_open = anchor_x, anchor_x + L
        cyl_x = [x_open, x_closed, x_closed, x_open]
        cyl_y = [y + h / 2, y + h / 2, y - h / 2, y - h / 2]
        piston_x = x_closed + L * 0.6
        rod_x = [piston_x, attach_x]
    else:
        x_closed, x_open = anchor_x, anchor_x - L
        cyl_x = [x_open, x_closed, x_closed, x_open]
        cyl_y = [y + h / 2, y + h / 2, y - h / 2, y - h / 2]
        piston_x = x_closed - L * 0.6
        rod_x = [attach_x, piston_x]
    return {
        'cyl': (cyl_x, cyl_y),
        'piston': ([piston_x, piston_x], [y - h * 0.4, y + h * 0.4]),
        'rod': (rod_x, [y, y]),
    }


def _draw_ground(ax, xlim, y=0.0):
    ax.plot(xlim, [y, y], 'k-', lw=2, zorder=1)
    for xh in np.arange(xlim[0], xlim[1], 0.13):
        ax.plot([xh, xh - 0.07], [y, y - 0.09], 'k-', lw=0.6, zorder=1)


def _draw_buried_anchor(ax, x, y0, y1):
    """Подземный анкер — вертикальная штриховка слева в брауной полосе."""
    ax.plot([x, x], [y0, y1], 'k-', lw=3.5, zorder=2)
    for yh in np.arange(y0, y1, 0.08):
        ax.plot([x - 0.07, x], [yh, yh + 0.05], 'k-', lw=0.6, zorder=2)


# ---------------------------------------------------------------------------
# Анимация одной траектории
# ---------------------------------------------------------------------------
def animate_mechanical_system(data, save_path=None, target_fps=25, show=False,
                              building_h=2.2, building_w=0.6,
                              tmd_h=0.32, tmd_w=0.40, gap=0.28,
                              dpi=80, title='Building with active TMD on roof'):
    fig, ax = plt.subplots(figsize=(8, 7))
    art = _setup_building_axes([ax], [data['ref']], building_h, building_w,
                                tmd_h, tmd_w, gap, label=None)[0]
    fig.suptitle(title, fontsize=13, fontweight='bold')

    t = data['t']
    dt_sim = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    stride = max(1, int(round(1.0 / (target_fps * dt_sim))))
    idx = np.arange(0, len(t), stride)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)

    def update(frame):
        i = idx[frame]
        return _update_building(art, data, i, building_h, building_w,
                                tmd_h, tmd_w, gap)

    ani = FuncAnimation(fig, update, frames=len(idx),
                        interval=1000 / target_fps, blit=False, repeat=True)
    if save_path:
        ani.save(save_path, writer='pillow', fps=target_fps, dpi=dpi)
        print(f"[viz] saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return ani


# ---------------------------------------------------------------------------
# Сравнение нескольких контроллеров
# ---------------------------------------------------------------------------
def animate_comparison(data_dict, save_path=None, target_fps=25, show=False,
                       building_h=2.2, building_w=0.6,
                       tmd_h=0.32, tmd_w=0.40, gap=0.28,
                       title='Active TMD — force applied to the foundation',
                       dpi=80):
    labels = list(data_dict.keys())
    datas = list(data_dict.values())
    t = datas[0]['t']
    dt_sim = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    stride = max(1, int(round(1.0 / (target_fps * dt_sim))))
    idx = np.arange(0, len(t), stride)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)

    n = len(labels)
    fig = plt.figure(figsize=(4.4 * n, 8.5))
    gs = fig.add_gridspec(2, n, height_ratios=[3.4, 1.0], hspace=0.20)
    bld_axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    err_ax = fig.add_subplot(gs[1, :])

    artists_list = _setup_building_axes(bld_axes, [d['ref'] for d in datas],
                                         building_h, building_w, tmd_h, tmd_w,
                                         gap, label=labels)

    colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
    for i, (label, d) in enumerate(zip(labels, datas)):
        err_ax.plot(d['t'], d['states'][:, 0], color=colors[i % len(colors)],
                    label=label, lw=1.5)
    err_ax.axhline(0, ls='--', color='k', lw=0.8)
    err_ax.set_xlabel('t [s]')
    err_ax.set_ylabel(r'$x_1$ [m]')
    err_ax.grid(True, alpha=0.4)
    err_ax.legend(fontsize=10, loc='upper right', ncols=n)
    time_marker = err_ax.axvline(0.0, color='red', lw=1.5, alpha=0.7)
    time_text = err_ax.text(0.005, 0.85, '', transform=err_ax.transAxes, fontsize=10,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                       alpha=0.9, edgecolor='gray'))

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    def update(frame):
        i = idx[frame]
        time_marker.set_xdata([t[i], t[i]])
        time_text.set_text(f't = {t[i]:.2f} s')
        out = [time_marker, time_text]
        for label, d, art in zip(labels, datas, artists_list):
            out.extend(_update_building(art, d, i, building_h, building_w,
                                         tmd_h, tmd_w, gap))
        return out

    ani = FuncAnimation(fig, update, frames=len(idx),
                        interval=1000 / target_fps, blit=False, repeat=True)
    if save_path:
        ani.save(save_path, writer='pillow', fps=target_fps, dpi=dpi)
        print(f"[viz] saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return ani


# ---------------------------------------------------------------------------
# Геометрия сцены
# ---------------------------------------------------------------------------
FOUNDATION_W_RATIO = 1.5       # фундамент шире здания
FOUNDATION_TOP = 0.05          # верхний край фундамента над землёй
FOUNDATION_BOTTOM = -0.30      # нижний край (зарыт)


def _setup_building_axes(axes, refs, building_h, building_w, tmd_h, tmd_w, gap,
                          label=None):
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    labels = label if isinstance(label, list) else [label] * len(axes)

    xlim = (-2.0, 2.0)
    ylim = (FOUNDATION_BOTTOM - 0.20, building_h + gap + tmd_h + 0.7)
    n_floors = 7
    n_cols = 3
    floor_h = building_h / n_floors
    foundation_w = building_w * FOUNDATION_W_RATIO

    artists_all = []
    for ax, ref, lab in zip(axes, refs, labels):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_facecolor(COLOR_BG)
        if lab is not None:
            ax.set_title(lab, fontsize=14, fontweight='bold', pad=6)

        # Грунт (брауная полоса) и линия земли
        ax.axhspan(FOUNDATION_BOTTOM - 0.20, 0.0,
                    facecolor='#7a5a3a', alpha=0.30, zorder=0)
        _draw_ground(ax, xlim, y=0.0)

        # Подземный анкер пружины и демпфера слева, уходит ниже линии земли
        wall_x = xlim[0] + 0.20
        _draw_buried_anchor(ax, wall_x,
                              FOUNDATION_BOTTOM - 0.05, FOUNDATION_TOP + 0.05)

        # Опорная вертикаль x=0
        ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.5, zorder=1)

        # ---- Фундамент (вкопанный в землю) ---------------------------------
        foundation = Rectangle((-foundation_w / 2, FOUNDATION_BOTTOM),
                                 foundation_w, FOUNDATION_TOP - FOUNDATION_BOTTOM,
                                 fc=COLOR_FOUNDATION, ec='black', lw=1.5,
                                 hatch='///', zorder=3)
        ax.add_patch(foundation)
        foundation_label = ax.text(0, FOUNDATION_TOP - 0.10,
                                     'foundation', ha='center', va='center',
                                     fontsize=8, color='white', fontweight='bold',
                                     style='italic', zorder=5)

        # ---- Здание (стоит на фундаменте) ----------------------------------
        b_y0 = FOUNDATION_TOP
        building = Rectangle((-building_w / 2, b_y0), building_w, building_h,
                              fc=COLOR_BUILDING, ec='black', lw=1.5, zorder=4)
        ax.add_patch(building)
        cornice = Rectangle((-building_w / 2 - 0.04, b_y0 + building_h - 0.05),
                             building_w + 0.08, 0.05,
                             fc='#2c5a90', ec='black', lw=1.2, zorder=5)
        ax.add_patch(cornice)

        win_w = building_w * 0.18
        win_h = floor_h * 0.45
        window_offsets, windows = [], []
        for fl in range(n_floors):
            for cl in range(n_cols):
                off_x = (cl + 0.5) * building_w / n_cols - building_w / 2 - win_w / 2
                wy = b_y0 + 0.08 + fl * floor_h + (floor_h - win_h) / 2
                w = Rectangle((off_x, wy), win_w, win_h,
                               fc='#e7f1f8', ec='#1f4f80', lw=0.5, zorder=6)
                ax.add_patch(w)
                windows.append(w)
                window_offsets.append((off_x, wy))

        # Подпись m₁ на здании — будет двигаться вместе с ним
        mass1_label = ax.text(0, b_y0 + building_h * 0.5, r'$m_1$',
                                ha='center', va='center',
                                fontsize=16, color='white', fontweight='bold',
                                zorder=8)

        # ---- Рельсы и TMD на крыше ----------------------------------------
        rail_top, = ax.plot([], [], color='#333', lw=2, zorder=5)
        rail_bot, = ax.plot([], [], color='#333', lw=2, zorder=5)
        tmd_y0 = b_y0 + building_h + gap
        tmd = Rectangle((-tmd_w / 2, tmd_y0), tmd_w, tmd_h,
                         fc=COLOR_TMD, ec='black', lw=1.5, zorder=7)
        ax.add_patch(tmd)
        tmd_label = ax.text(0, tmd_y0 + tmd_h / 2, r'$m_2$',
                             ha='center', va='center', fontsize=12,
                             color='white', fontweight='bold', zorder=8)

        # ---- Фундамент ↔ грунт: пружина k₁ + демпфер c₁ параллельно --------
        # Заглублены в землю, соединяют фундамент с подземным анкером.
        spring_g_y = FOUNDATION_TOP - 0.06
        damper_g_y = FOUNDATION_BOTTOM + 0.10
        spring_g, = ax.plot([], [], color=COLOR_SPRING, lw=2.4, zorder=3)
        damper_g_cyl, = ax.plot([], [], color=COLOR_DAMPER, lw=1.8, zorder=3)
        damper_g_piston, = ax.plot([], [], color=COLOR_DAMPER, lw=2.4, zorder=3)
        damper_g_rod, = ax.plot([], [], color=COLOR_DAMPER, lw=2.0, zorder=3)
        label_k1 = ax.text(0, 0, r'$k_1$', fontsize=11, color=COLOR_SPRING,
                            ha='center', va='bottom', fontweight='bold', zorder=4)
        label_c1 = ax.text(0, 0, r'$c_1$', fontsize=11, color=COLOR_DAMPER,
                            ha='center', va='top', fontweight='bold', zorder=4)

        # ---- Крыша → TMD ---------------------------------------------------
        spring_r_y = tmd_y0 - gap * 0.30
        damper_r_y = tmd_y0 - gap * 0.72
        spring_r, = ax.plot([], [], color=COLOR_SPRING, lw=2.0, zorder=8)
        damper_r_cyl, = ax.plot([], [], color=COLOR_DAMPER, lw=1.6, zorder=8)
        damper_r_piston, = ax.plot([], [], color=COLOR_DAMPER, lw=2.2, zorder=8)
        damper_r_rod, = ax.plot([], [], color=COLOR_DAMPER, lw=1.8, zorder=8)
        label_k2 = ax.text(0, 0, r'$k_2$', fontsize=11, color=COLOR_SPRING,
                            ha='center', va='bottom', fontweight='bold', zorder=9)
        label_c2 = ax.text(0, 0, r'$c_2$', fontsize=11, color=COLOR_DAMPER,
                            ha='center', va='top', fontweight='bold', zorder=9)

        # ---- Актуатор -------------------------------------------------------
        act_box = Rectangle((0, 0), 0.08, 0.05, fc=COLOR_ACTUATOR,
                             ec='black', lw=1, zorder=10)
        ax.add_patch(act_box)
        act_label = ax.text(0, tmd_y0 + tmd_h + 0.20, '',
                             ha='center', fontsize=10, color='#cc5500',
                             fontweight='bold', zorder=10)

        # ---- Стрелка силы d(t) НА ФУНДАМЕНТ -------------------------------
        # Базовая стрелка; позиция и длина обновляются каждый кадр.
        force_arrow = FancyArrowPatch((0, FOUNDATION_BOTTOM + 0.10),
                                       (0.3, FOUNDATION_BOTTOM + 0.10),
                                       arrowstyle='-|>', mutation_scale=24,
                                       color=COLOR_FORCE, lw=3.2, zorder=11)
        ax.add_patch(force_arrow)
        force_label = ax.text(0, FOUNDATION_BOTTOM - 0.10, '',
                                fontsize=10, color=COLOR_FORCE,
                                fontweight='bold', ha='center', zorder=11)

        info = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.88, edgecolor='gray'))

        artists_all.append({
            'wall_x': wall_x,
            'foundation': foundation, 'foundation_label': foundation_label,
            'building': building, 'cornice': cornice, 'mass1_label': mass1_label,
            'windows': windows, 'window_offsets': window_offsets,
            'rail_top': rail_top, 'rail_bot': rail_bot,
            'tmd': tmd, 'tmd_label': tmd_label,
            'spring_g': spring_g,
            'damper_g_cyl': damper_g_cyl,
            'damper_g_piston': damper_g_piston,
            'damper_g_rod': damper_g_rod,
            'spring_g_y': spring_g_y, 'damper_g_y': damper_g_y,
            'label_k1': label_k1, 'label_c1': label_c1,
            'spring_r': spring_r,
            'damper_r_cyl': damper_r_cyl,
            'damper_r_piston': damper_r_piston,
            'damper_r_rod': damper_r_rod,
            'spring_r_y': spring_r_y, 'damper_r_y': damper_r_y,
            'label_k2': label_k2, 'label_c2': label_c2,
            'act_box': act_box, 'act_label': act_label,
            'force_arrow': force_arrow, 'force_label': force_label,
            'info': info,
            'b_y0': b_y0, 'tmd_y0': tmd_y0,
            'foundation_w': foundation_w,
        })
    return artists_all


def _update_building(art, data, i, building_h, building_w, tmd_h, tmd_w, gap):
    x1 = float(data['states'][i, 0])
    x2 = float(data['states'][i, 1])
    u = float(data['controls'][i])
    d_val = float(data['disturbances'][i]) if i < len(data['disturbances']) else 0.0

    b_y0 = art['b_y0']
    tmd_y0 = art['tmd_y0']
    foundation_w = art['foundation_w']
    foundation_left = x1 - foundation_w / 2
    foundation_right = x1 + foundation_w / 2
    foundation_center_y = (FOUNDATION_TOP + FOUNDATION_BOTTOM) / 2

    # --- Фундамент (двигается вместе со зданием как одно тело m₁) ---
    art['foundation'].set_xy((-foundation_w / 2 + x1, FOUNDATION_BOTTOM))
    art['foundation_label'].set_position((x1, FOUNDATION_TOP - 0.10))

    # --- Здание + окна + подпись m₁ ---
    art['building'].set_xy((-building_w / 2 + x1, b_y0))
    art['cornice'].set_xy((-building_w / 2 - 0.04 + x1, b_y0 + building_h - 0.05))
    art['mass1_label'].set_position((x1, b_y0 + building_h * 0.5))
    for w, (ox, oy) in zip(art['windows'], art['window_offsets']):
        w.set_xy((ox + x1, oy))

    # --- Рельсы на крыше ---
    rl = x1 - building_w / 2 - 0.05
    rr = x1 + building_w / 2 + 0.05
    art['rail_top'].set_data([rl, rr], [b_y0 + building_h + 0.005,
                                          b_y0 + building_h + 0.005])
    art['rail_bot'].set_data([rl, rr], [tmd_y0 - 0.005, tmd_y0 - 0.005])

    # --- TMD ---
    art['tmd'].set_xy((-tmd_w / 2 + x2, tmd_y0))
    art['tmd_label'].set_position((x2, tmd_y0 + tmd_h / 2))

    # --- Фундамент ↔ грунт: пружина k₁ и dashpot c₁ ---
    wall_x = art['wall_x']
    sp_y = art['spring_g_y']
    dp_y = art['damper_g_y']
    xs, ys = _h_spring(wall_x + 0.03, foundation_left, sp_y,
                        num_coils=6, amp=0.06)
    art['spring_g'].set_data(xs, ys)
    dpot = _dashpot_coords(wall_x + 0.04, foundation_left, dp_y,
                            length=0.42, height=0.14, opens_right=True)
    art['damper_g_cyl'].set_data(*dpot['cyl'])
    art['damper_g_piston'].set_data(*dpot['piston'])
    art['damper_g_rod'].set_data(*dpot['rod'])
    art['label_k1'].set_position(((wall_x + foundation_left) / 2, sp_y + 0.07))
    art['label_c1'].set_position(((wall_x + foundation_left) / 2, dp_y - 0.10))

    # --- Крыша → TMD: пружина k₂ и dashpot c₂ ---
    roof_anchor_x = x1 - building_w / 2 + 0.05
    tmd_left = x2 - tmd_w / 2
    sp_y2 = art['spring_r_y']
    dp_y2 = art['damper_r_y']
    xs, ys = _h_spring(roof_anchor_x, tmd_left, sp_y2, num_coils=4, amp=0.04)
    art['spring_r'].set_data(xs, ys)
    dpot2 = _dashpot_coords(roof_anchor_x + 0.01, tmd_left, dp_y2,
                             length=0.20, height=0.09, opens_right=True)
    art['damper_r_cyl'].set_data(*dpot2['cyl'])
    art['damper_r_piston'].set_data(*dpot2['piston'])
    art['damper_r_rod'].set_data(*dpot2['rod'])
    art['label_k2'].set_position(((roof_anchor_x + tmd_left) / 2, sp_y2 + 0.06))
    art['label_c2'].set_position(((roof_anchor_x + tmd_left) / 2, dp_y2 - 0.07))

    # --- Актуатор (рядом с TMD, размер ~ |u|) ---
    act_y = tmd_y0 - gap / 2
    act_x = x2 + tmd_w / 2 - 0.06
    u_norm = min(abs(u) / 8.0, 1.0)
    act_w = 0.08 + 0.08 * u_norm
    act_h = 0.05
    art['act_box'].set_xy((act_x - act_w / 2, act_y - act_h / 2))
    art['act_box'].set_width(act_w)
    art['act_box'].set_height(act_h)
    art['act_label'].set_position((x2, tmd_y0 + tmd_h + 0.20))
    art['act_label'].set_text(f'u = {u:+.2f} N')

    # --- d(t): красная стрелка ВНУТРИ фундамента, направление ~ sign(d) ----
    # Длина пропорциональна |d|. Стрелка двигается вместе с фундаментом.
    f_arrow_y = FOUNDATION_BOTTOM + 0.08
    half_len = (0.18 + 0.30 * min(abs(d_val) / 2.0, 1.0)) / 2
    direction = 1 if d_val >= 0 else (-1 if d_val < 0 else 1)
    cx = x1
    art['force_arrow'].set_positions(
        (cx - direction * half_len, f_arrow_y),
        (cx + direction * half_len, f_arrow_y),
    )
    art['force_label'].set_position((x1, FOUNDATION_BOTTOM - 0.10))
    art['force_label'].set_text(f'd(t) = {d_val:+.2f} N')

    # --- Информационная плашка ---
    art['info'].set_text(f'x₁ = {x1:+.3f} m\nx₂ = {x2:+.3f} m\nu = {u:+.2f} N')

    return [art['foundation'], art['foundation_label'],
            art['building'], art['cornice'], art['mass1_label'],
            art['tmd'], art['tmd_label'],
            art['rail_top'], art['rail_bot'],
            art['spring_g'], art['damper_g_cyl'], art['damper_g_piston'],
            art['damper_g_rod'], art['label_k1'], art['label_c1'],
            art['spring_r'], art['damper_r_cyl'], art['damper_r_piston'],
            art['damper_r_rod'], art['label_k2'], art['label_c2'],
            art['act_box'], art['act_label'],
            art['force_arrow'], art['force_label'], art['info']] + art['windows']
