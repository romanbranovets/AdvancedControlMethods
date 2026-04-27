# Adaptive Control of a Drone in Wind Disturbance

<p align="center">
  <img src="compare_flight.gif" alt="PID baseline (translucent grey) vs MRAC + sigma-modification (solid blue)" width="700"/>
</p>

> На анимации: серый «призрак» — PID baseline, синий — MRAC c σ-модификацией.
> Под одним и тем же двухчастотным ветром PID не доходит до цели за 25 с
> (final error 3.18 м), MRAC заходит в ε-окрестность за 14.45 с (final error
> 0.148 м) — улучшение 95%.

Цель проекта — построить **адаптивный регулятор** для квадрокоптера в
присутствии нестационарного ветрового возмущения, добиться устойчивого
слежения за заданной точкой и **показать преимущество адаптивного
управления над PID baseline**.

Реализация выбрана: **MRAC с σ-модификацией на скоростном контуре**, плюс
saturation freeze и projection для робастности. Полный математический
вывод и обсуждение — в [MRAC.md](MRAC.md).

---

## 1. Problem Definition

### 1.1 Control objective

Стабилизировать дрон в окрестности заданной 3D-цели $p_t$ под действием
неизвестного ветра $v_w(t)$, обеспечив:

- ограниченность всех сигналов замкнутой системы (Lyapunov);
- сходимость в **ε-окрестность** цели за конечное время:
  $\| p(t) - p_t \| < \varepsilon$;
- превосходство над PID baseline по метрикам **final error**, **IAE**,
  **time-to-target**.

### 1.2 Plant definition

Используется полная **12-state модель квадрокоптера** в ZYX-конвенции:

$$
x = [\,p^\top,\; v^\top,\; \eta^\top,\; \omega^\top\,]^\top \in \mathbb{R}^{12},
$$

где $p \in \mathbb{R}^3$ — положение в мировой СК, $v \in \mathbb{R}^3$ —
скорость, $\eta = [\phi, \theta, \psi]^\top$ — углы Эйлера (крен, тангаж,
курс), $\omega \in \mathbb{R}^3$ — угловые скорости в связанной СК.

Динамика ([src/system.py](src/system.py)):

$$
\dot p = v,
$$

где $\dot p$ — производная положения (скорость).

$$
m\dot v = R(\eta)\,T_{\text{body}} + F_{\text{drag}}(v - v_w) + m g,
$$

где $m = 0.5$ кг — масса, $R(\eta)$ — матрица поворота body→world,
$T_{\text{body}} = [0, 0, \sum_{i} u_i]^\top$ — суммарная тяга в body-СК,
$F_{\text{drag}} = -c_1(v - v_w) - c_2 \|v - v_w\|(v - v_w)$ —
аэродинамическое сопротивление (линейный + квадратичный drag), а
$g = [0, 0, -9.81]^\top$ — гравитация.

$$
I\dot\omega = \tau - \omega \times (I\omega),
$$

где $I$ — диагональная матрица инерции, а $\tau$ — суммарный момент,
формируемый разностью тяг моторов в X-конфигурации:

$$
\tau = \begin{bmatrix} \ell\,(u_2 - u_4)\\ \ell\,(-u_1 + u_3)\\ d\,(u_1 - u_2 + u_3 - u_4)\end{bmatrix},
$$

где $\ell = 0.2$ м — плечо, а $d = 0.01$ — коэффициент аэро-момента.

Кинематика углов Эйлера (Tait–Bryan):

$$
\dot\eta =
\begin{bmatrix}
1 & \sin\phi\tan\theta & \cos\phi\tan\theta \\
0 & \cos\phi & -\sin\phi \\
0 & \sin\phi/\cos\theta & \cos\phi/\cos\theta
\end{bmatrix}\omega,
$$

где $\phi$ — крен, $\theta$ — тангаж.

### 1.3 Assumptions and context

- **Масса $m$ известна**, инерции $I$ — известны.
- **Аэродинамика drag** — известна по форме (linear + quadratic),
  коэффициенты $c_1, c_2$ — фиксированы.
- **Ветер $v_w(t)$** — неизвестен заранее, но ограничен:
  $\|v_w(t)\|_\infty \le 5$ м/с, имеет высокочастотную (~3 рад/с)
  и низкочастотную (~0.4 рад/с) составляющие плюс bias.
- **Полное состояние** $x$ доступно для измерения (нет шума и
  задержек).
- **Управление насыщено**: $u_i \in [0, 5]$ Н, наклон $|\phi|, |\theta| \le 15°$.

### 1.4 Method class

**Direct MRAC** (Model Reference Adaptive Control) с **σ-модификацией**
(Ioannou, 1984), применённый к **скоростному (translational) контуру**.

Идея:
1. Вспомогательный эталон `v_m(t)` задаёт целевую динамику первого
   порядка `v̇_m = -a_m(v_m - v_des)`.
2. Регулятор формирует $u = a_m(v_{des} - v) - \hat\Theta^\top \Phi(v)$,
   где $\Phi$ — известный регрессор, $\hat\Theta$ — адаптивная оценка
   неизвестных параметров возмущения.
3. Закон обновления $\dot{\hat\Theta} = \gamma e \Phi - \sigma \hat\Theta$,
   где $e = v - v_m$.
4. Lyapunov-функция $V = \tfrac12 e^2 + \tfrac{1}{2\gamma}\|\tilde\Theta\|^2$
   гарантирует **uniformly ultimately bounded** (UUB).

Альтернативы, рассмотренные и отвергнутые:
- MRAC на угловой динамике — работает, но возмущение ветра туда не
  входит, улучшение всего +3%.
- Mass mismatch + MRAC — нарушает linear-in-parameters, MRAC проигрывает.

---

## 2. System Description

### 2.1 State variables

| Переменная | Размерность | Описание |
|---|---|---|
| $p = (x, y, z)$ | 3 | положение в мировой СК [м] |
| $v = (v_x, v_y, v_z)$ | 3 | скорость [м/с] |
| $\eta = (\phi, \theta, \psi)$ | 3 | углы Эйлера ZYX [рад] |
| $\omega = (p, q, r)$ | 3 | body rates [рад/с] |
| $\hat\Theta \in \mathbb{R}^{3\times 3}$ | 9 | адаптивные параметры (по 3 на каждую ось) |
| $v_m \in \mathbb{R}^3$ | 3 | состояние эталонной модели |

### 2.2 Control input

Командные тяги моторов $u = [u_1, u_2, u_3, u_4]^\top \in \mathbb{R}^4$,
формируемые через **mixer** из «виртуальных» сигналов
$U = [U_1, U_2, U_3, U_4]^\top$:

$$
\begin{aligned}
u_1 &= U_1/4 - U_3/(2\ell) + U_4/(4d) \\
u_2 &= U_1/4 + U_2/(2\ell) - U_4/(4d) \\
u_3 &= U_1/4 + U_3/(2\ell) + U_4/(4d) \\
u_4 &= U_1/4 - U_2/(2\ell) - U_4/(4d)
\end{aligned}
$$

где $U_1$ — суммарная тяга, $U_2, U_3, U_4$ — моменты по крену, тангажу,
курсу.

### 2.3 Unknown parameters

Возмущение по каждой оси скорости параметризуется через регрессор:

$$
\Delta_i(v_i) = {\Theta_i^*}^\top \Phi(v_i),\qquad i \in \{x, y, z\},
$$

где $\Phi(v) = [1,\, v,\, |v|\cdot v]^\top \in \mathbb{R}^3$ — известный
регрессор (bias + linear drag + quadratic drag), а $\Theta_i^* \in
\mathbb{R}^3$ — **неизвестный** истинный параметр для оси $i$.

В реальной модели $\Delta_i$ зависит от $v_w(t)$ и не точно линейна по
$\Phi(v)$ — это допущение нарушается (см. §11 в [MRAC.md](MRAC.md)).
σ-модификация компенсирует это и даёт UUB вместо точной сходимости
$\hat\Theta \to \Theta^*$.

### 2.4 Control bounds

| Сигнал | Лимит |
|---|---|
| Тяга мотора $u_i$ | $0 \le u_i \le 5$ Н |
| Углы наклона $\phi, \theta$ | $\le 15°$ |
| Желаемая скорость | $\|v_{des,xy}\| \le 2.5$ м/с, $\|v_{des,z}\| \le 2$ м/с |
| Желаемое ускорение | $\|a_{des,xy}\| \le 8$, $\|a_{des,z}\| \le 10$ м/с² |
| Тяга `thrust_d` | $0.2 \cdot mg \le T \le 3.0 \cdot mg$ |
| Адаптивные параметры | $\|\hat\Theta_i\|_2 \le 8$ (projection) |

При насыщении адаптация **замораживается** (Lavretsky §10.2) — без этого
σ-модификация недостаточна для устойчивости.

### 2.5 Dynamics

Полная замкнутая система (по осям, упрощённо для одной оси скорости):

$$
\dot v_i = u_i + \Theta_i^{*\top}\Phi(v_i),
$$

где $u_i$ — командное ускорение от MRAC, $\Phi$ — регрессор.

$$
\dot v_{m,i} = -a_m(v_{m,i} - v_{des,i}),
$$

где $v_{m,i}$ — эталонная скорость, $a_m > 0$ — частота эталона.

$$
\dot{\hat\Theta}_i = \gamma\, e_i\, \Phi(v_i) - \sigma\, \hat\Theta_i,
$$

где $e_i = v_i - v_{m,i}$ — tracking error, $\gamma > 0$ — скорость
адаптации, $\sigma > 0$ — leakage rate.

---

## 3. Mathematical Specification

### 3.1 Error dynamics

Подставляя закон управления $u_i = a_m(v_{des,i} - v_i) - \hat\Theta_i^\top \Phi$
в plant:

$$
\dot v_i = a_m(v_{des,i} - v_i) - \hat\Theta_i^\top\Phi + \Theta_i^{*\top}\Phi
        = -a_m(v_i - v_{des,i}) + \tilde\Theta_i^\top \Phi,
$$

где $\tilde\Theta_i = \Theta_i^* - \hat\Theta_i$ — ошибка параметров.

Вычитая $\dot v_{m,i}$:

$$
\dot e_i = -a_m\, e_i + \tilde\Theta_i^\top \Phi(v_i),
$$

где $e_i = v_i - v_{m,i}$ — tracking error.

### 3.2 Nominal adaptive control law

Полный закон управления MRAC по каждой оси (см. [src/controller.py](src/controller.py)):

$$
u_i = \underbrace{a_m(v_{des,i} - v_i)}_{\text{P-feedback на ошибке скорости}} - \underbrace{\hat\Theta_i^\top \Phi(v_i)}_{\text{adaptive disturbance compensation}}.
$$

В исходниках это `MRACAxis1D.update()` — $u$ далее насыщается на
$\pm u_{\max}$ и используется как **командное ускорение** для геометрического
блока, который пересчитывает его в наклон + тягу:

$$
\theta_d = \mathrm{atan2}(a_x^b,\ g + a_z),\qquad
\phi_d = -\mathrm{atan2}(a_y^b,\ g + a_z),\qquad
T_d = \frac{m(g + a_z)}{\cos\phi_d \cos\theta_d}.
$$

Внутренний угловой контур остаётся **обычным PID** на $(\phi, \theta, \psi)$
(он быстрее MRAC и не имеет существенной неопределённости).

---

## 4. Method Description

### 4.1 Control law

Каскадная структура:

```
target ──▶ outer P  (pos error → v_des)
              │
              ▼
          MRAC σ-mod  (v_des → a_cmd)         ←── adaptive layer
              │
              ▼
          tilt geometry (a_cmd → φ_d, θ_d, T_d)
              │
              ▼
          inner PID  (attitude error → torques)
              │
              ▼
          mixer  (U → motor thrusts)
              │
              ▼
            plant
```

Только **средний контур (velocity → acceleration)** заменён на MRAC.
Outer P-loop и inner PID остаются классическими — это даёт справедливое
сравнение.

### 4.2 Adaptation law

Закон обновления параметров **по каждой оси**:

$$
\dot{\hat\Theta}_i = \gamma\,e_i\,\Phi(v_i) - \sigma\,\hat\Theta_i,
$$

с σ-модификацией. **Если** управление насыщено, то $\dot{\hat\Theta}_i = 0$
(adaptation freeze).

После каждого шага применяется проекция:

$$
\hat\Theta_i \leftarrow \hat\Theta_i \cdot \min\left(1,\ \frac{\theta_{\max}}{\|\hat\Theta_i\|}\right).
$$

### 4.3 Idea of derivation

Lyapunov-кандидат для одной оси:

$$
V_i(e_i, \tilde\Theta_i) = \tfrac12 e_i^2 + \tfrac{1}{2\gamma}\|\tilde\Theta_i\|^2.
$$

Производная вдоль траектории даёт после подстановки закона обновления и
неравенства Юнга:

$$
\dot V_i \le -a_m\,e_i^2 - \tfrac{\sigma}{2\gamma}\|\tilde\Theta_i\|^2 + \tfrac{\sigma}{2\gamma}\|\Theta_i^*\|^2.
$$

Отсюда **UUB**: $(e_i, \tilde\Theta_i)$ остаются в ограниченном эллипсоиде
радиуса $\sim \sigma\|\Theta_i^*\|/\sqrt{a_m\gamma}$. Полный вывод —
в [MRAC.md §6–§7](MRAC.md).

---

## 5. Stability Proof

### 5.1 Closed-loop model

Объект (по оси) с подставленным управлением:

$$
\dot e_i = -a_m e_i + \tilde\Theta_i^\top \Phi(v_i),\qquad
\dot{\tilde\Theta}_i = -\dot{\hat\Theta}_i,
$$

где $\dot{\tilde\Theta}_i$ — производная ошибки параметров (так как $\Theta_i^*$
константна).

### 5.2 Lyapunov function

$$
V_i(e_i,\tilde\Theta_i) = \tfrac12 e_i^2 + \tfrac{1}{2\gamma} \tilde\Theta_i^\top \tilde\Theta_i.
$$

Положительно определённа, radially unbounded.

### 5.3 Derivative

$$
\dot V_i = e_i \dot e_i + \tfrac{1}{\gamma} \tilde\Theta_i^\top \dot{\tilde\Theta}_i
        = -a_m e_i^2 + e_i \tilde\Theta_i^\top \Phi - \tfrac{1}{\gamma} \tilde\Theta_i^\top \dot{\hat\Theta}_i.
$$

Подставляя $\dot{\hat\Theta}_i = \gamma e_i \Phi - \sigma \hat\Theta_i$:

$$
\dot V_i = -a_m e_i^2 + (\sigma/\gamma)\,\tilde\Theta_i^\top \hat\Theta_i.
$$

Используя $\hat\Theta_i = \Theta_i^* - \tilde\Theta_i$ и неравенство Юнга
$\tilde\Theta_i^\top \Theta_i^* \le \tfrac12\|\tilde\Theta_i\|^2 + \tfrac12\|\Theta_i^*\|^2$:

$$
\boxed{\dot V_i \le -a_m\, e_i^2 - \tfrac{\sigma}{2\gamma}\|\tilde\Theta_i\|^2 + \tfrac{\sigma}{2\gamma}\|\Theta_i^*\|^2.}
$$

### 5.4 Consequences

`V̇` отрицательна вне эллипсоида:

$$
\mathcal{B}_i = \left\{ (e_i, \tilde\Theta_i)\ :\ a_m e_i^2 + \tfrac{\sigma}{2\gamma}\|\tilde\Theta_i\|^2 \le \tfrac{\sigma}{2\gamma}\|\Theta_i^*\|^2 \right\}.
$$

Следствия:
- $V_i(t)$ ограничена ⇒ $e_i,\tilde\Theta_i$ ограничены **∀ t** (UUB);
- $e_i(t)$ сходится в окрестность нуля радиуса $\sim \sigma\|\Theta_i^*\|/a_m$;
- **сходимость $\hat\Theta_i \to \Theta_i^*$ НЕ гарантирована** — это
  требует *persistent excitation*, которого в задаче точечной стабилизации
  обычно нет (после захода в окрестность цели $v \approx 0$, регрессор
  вырождается).

### 5.5 If the wind is time-varying

Истинное возмущение — не константа, а $\Delta_i(v_i, t) =
{\Theta_i^*(t)}^\top \Phi(v_i)$ с медленно меняющимся $\Theta_i^*(t)$.

Тогда `V̇` приобретает дополнительный член $-\dot\Theta_i^* \tilde\Theta_i / \gamma$,
и в неравенстве появляется $\sigma/(2\gamma)(\|\Theta^*\|^2 + \|\dot\Theta^*\|^2/\sigma^2)$.

UUB сохраняется при ограниченности $\|\dot\Theta_i^*\|$. Размер
ultimate-области растёт — это видно на графиках 04 и 05 (см. §9).

Подробный анализ + альтернативы (e-modification, projection-based MRAC) —
в [MRAC.md §7, §12](MRAC.md).

---

## 6. Algorithm Listing

### Algorithm 1: Outer-loop MRAC with σ-modification

**Initialization** (в начале каждого запуска):
- $\hat\Theta_i \leftarrow 0,\ v_{m,i} \leftarrow 0$ для $i \in \{x, y, z\}$;
- сброс PID-интеграторов outer и inner контуров.

**Главный цикл** (период $\Delta t = 5$ мс):

1. Прочитать состояние $x = (p, v, \eta, \omega)$.
2. Outer P-loop: $v_{des,i} = K_{p,pos}(p_{t,i} - p_i)$, насыщение по
   $|v_{des,i}| \le v_{max,i}$.
3. Для каждой оси $i \in \{x, y, z\}$:
    1. Интегрировать эталон точно: $v_{m,i} \leftarrow v_{des,i} +
       (v_{m,i} - v_{des,i})\, e^{-a_m \Delta t}$.
    2. Tracking error: $e_i = v_i - v_{m,i}$.
    3. Регрессор: $\Phi_i = [1,\ v_i,\ |v_i|\,v_i]^\top$.
    4. Командное ускорение:
       $a_{cmd,i}^{\text{unsat}} = a_m(v_{des,i} - v_i) - \hat\Theta_i^\top \Phi_i$.
    5. Насыщение: $a_{cmd,i} = \mathrm{clip}(a_{cmd,i}^{\text{unsat}},
       \pm a_{max,i})$.
    6. **Если** не насыщено: $\hat\Theta_i \leftarrow \hat\Theta_i +
       \Delta t (\gamma e_i \Phi_i - \sigma \hat\Theta_i)$.
    7. Проекция: если $\|\hat\Theta_i\| > \theta_{\max}$, то
       $\hat\Theta_i \leftarrow \theta_{\max} \cdot \hat\Theta_i / \|\hat\Theta_i\|$.
4. Yaw-frame декаплинг: $a_x^b = c_\psi a_x + s_\psi a_y$,
   $a_y^b = -s_\psi a_x + c_\psi a_y$.
5. Tilt-команды: $\theta_d = \arctan(a_x^b / (g + a_z))$,
   $\phi_d = -\arctan(a_y^b / (g + a_z))$, обрезать до $\pm 15°$.
6. Тяга: $T_d = m(g + a_z) / (\cos\phi_d \cos\theta_d)$, обрезать до
   $[0.2 mg, 3 mg]$.
7. Inner PID: $\tau_\phi, \tau_\theta, \tau_\psi$ из ошибок углов.
8. Mixer: $U = (T_d, \tau_\phi, \tau_\theta, \tau_\psi)
   \to (u_1, u_2, u_3, u_4)$, обрезать снизу нулём.
9. Подать $u$ на plant; шаг RK4; перейти к шагу 1.

---

## 7. Experimental Setup

### 7.1 Simulation conditions

| Параметр | Значение |
|---|---|
| Начальное состояние | $p_0$ — random в кубе $[5, 15]^3$, $v_0 = \omega_0 = 0$, $\eta_0 = 0$ |
| Цель | $p_t$ — random в кубе $[5, 15]^3$, $\|p_t - p_0\| \ge 6$ м |
| Длительность | $T_{\max} = 25$ с |
| Шаг RK4 | $\Delta t = 5$ мс |
| Условие останова | $\|p - p_t\| < 0.15$ м (ε-окрестность) |
| Seed | 42 (для воспроизводимости) |

### 7.2 Reference trajectory

В этом проекте — **точечная стабилизация**: $p_r(t) \equiv p_t$,
$\dot p_r = \ddot p_r = 0$. Это упрощение задачи, ослабляющее MRAC
(меньше excitation), но не теряющее robustness-свойств.

### 7.3 Controller parameters

**Outer (position P)**: $K_p = (1.2, 1.2, 1.5)$.

**Middle PID (baseline)**: $k_p = (2.5, 2.5, 4)$, $k_i = (0.4, 0.4, 1.5)$,
$k_d = 0$.

**Middle MRAC (adaptive)**:

| Параметр | xy | z | Назначение |
|---|---|---|---|
| $a_m$ (ref-model rate) | 6.0 | 8.0 | время реакции эталона |
| $\gamma$ (adaptation rate) | 3.0 | 3.0 | скорость обновления $\hat\Theta$ |
| $\sigma$ (leakage) | 0.5 | 0.5 | защита от drift |
| $\theta_{\max}$ (projection) | 8.0 | 8.0 | hard limit на параметры |
| $n_{\text{basis}}$ | 3 | 3 | $[1, v, |v|v]$ |

Подробное обоснование — [MRAC.md §14](MRAC.md).

**Inner PID (attitude)**: $k_p = (10, 10, 3)$, $k_i = (0.2, 0.2, 0)$,
$k_d = (2, 2, 0.5)$, derivative-on-measurement.

### 7.4 Disturbance model

Двухчастотный ветер с bias (peaks ~5 м/с):

$$
v_w(t) = \begin{bmatrix}
2.5\sin(2.5t) + 1.5\sin(0.4t) + 1.5 \\
2.0\cos(2.0t) + 1.5\cos(0.3t) + 1.0 \\
0.9\sin(3.0t) + 0.6\sin(0.5t) + 0.4
\end{bmatrix}.
$$

Drag-коэффициенты: $c_1 = 0.22,\ c_2 = 0.10$ (в 1.5–2× выше базовых для
жёсткого сценария).

### 7.5 Baseline for comparison

PID baseline — тот же каскад (outer P + middle PI + inner PID), но **без
адаптивного слоя**. Сравниваем тот же объект, тот же ветер, ту же
начальную точку.

| Метрика | PID | MRAC + σ-mod | Δ |
|---|---|---|---|
| Final error (seed 42) | 3.180 м | **0.148 м** | **+95%** |
| Time-to-target | 25.00 с (timeout) | **14.45 с** | **−42%** |
| Mean final error (8 seeds, ε=0.15) | 3.19 м | 1.35 м | +58% |
| PID timeouts (8 seeds) | **7/8** | **2/8** | — |

---

## 8. Reproducibility

### 8.1 Dependencies

```
matplotlib >= 3.10.8
numpy      >= 2.4.4
plotly     >= 5.20
```

Установка через `uv` (как в проекте):

```bash
uv sync
```

или через pip:

```bash
pip install matplotlib numpy plotly
```

### 8.2 Run commands

**Полный пайплайн** (симуляция + статические графики + GIF +
Plotly dashboard + 3D animation):

```bash
python main.py
```

**Только статические графики для отчёта** (без анимации):

```bash
python scripts/generate_report.py             # seed 42
python scripts/generate_report.py --seed 7    # другой сценарий
```

### 8.3 Outputs

После `python main.py` в корне проекта появятся:

| Файл | Содержание |
|---|---|
| `results_pid.png` | 6-панельные графики PID-прогона (pos, vel, euler, body rates, моторы, ветер) |
| `results_mrac.png` | то же для MRAC |
| `compare.png` | overlay tracking error PID vs MRAC + таблица IAE/ISE |
| `adaptation.png` | трекинг эталона + эволюция $\hat\Theta_i$ по 3 осям |
| `compare_flight.gif` | 3D-анимация: PID-призрак (серый) + MRAC (синий) на одной сцене |
| `dashboard.html` | **Plotly интерактивный** — 3D с вращением + 2D X-Y + r vs ṙ phase |

После `python scripts/generate_report.py` в `figures/`:

| Файл | Содержание |
|---|---|
| `01_trajectory_3d.png` | 3D-траектория PID vs MRAC, ε-сфера wireframe |
| `02_xy_topdown.png` | top-down + cylindrical projection $\sqrt{x^2+y^2}$ vs $z$ |
| `03_state_signals.png` | pos / vel / euler PID (пунктир) vs MRAC (сплошной) |
| `04_lyapunov.png` | $V(t) = \tfrac12\|e\|^2 + \tfrac{1}{2\gamma}\|\hat\Theta\|^2$ + decomposition |
| `05_error_metrics.png` | $\rho(t)$ в symlog + кумулятивный IAE |
| `06_phase_portrait.png` | $r$ vs $\dot r$ (расстояние до цели и скорость сближения) |
| `07_wind_estimation.png` | $\hat\Theta^\top \Phi(v)$ vs истинный drag/m **с явным caveat** про PE |
| `08_adaptation.png` | по-осевая адаптация: $v$ vs $v_m$ vs $v_{des}$ + $\hat\Theta_i(t)$ |
| `09_control_signals.png` | тяги моторов с линией $u_{\max}$ |

### 8.4 Exact reproduction

- **Seed**: `np.random.default_rng(42)` в [main.py](main.py).
- **Python**: 3.12+.
- **Сценарий**: жёстко прошит в `make_wind_func` и `_make_wind` в
  [main.py](main.py) и [scripts/generate_report.py](scripts/generate_report.py)
  соответственно.

---

## 9. Results Summary

### 9.1 What works

- **MRAC + σ-mod на скоростном контуре** уверенно превосходит PID:
  - точность: 95% улучшение по final error (seed 42)
  - надёжность: 7→2 timeouts из 8 на жёстком ветре
  - время до цели: 25→14.45 с
- **Lyapunov-функция** $V(t)$ остаётся ограниченной (UUB) — см.
  [`figures/04_lyapunov.png`](figures/04_lyapunov.png).
- **Адаптивные параметры $\hat\Theta_i$** оседают в окрестности после
  переходного процесса, не уходят в drift благодаря σ-mod — см.
  [`figures/08_adaptation.png`](figures/08_adaptation.png).
- **Tracking error** $\rho(t)$ MRAC уверенно сходится в ε-окрестность,
  PID болтается на расстоянии 1–3 м — см.
  [`figures/05_error_metrics.png`](figures/05_error_metrics.png).

### 9.2 What remains limited

- **`Θ̂ → Θ*` НЕ гарантировано** теорией без persistent excitation. На
  практике — после захода в цель $v \approx 0$, регрессор $\Phi(v)$
  вырождается, оценка перестаёт уточняться. Графики 04 и 07 имеют
  явные caption-предупреждения.
- **Mass mismatch** ломает linear-in-parameters. MRAC на скорости
  предполагает $\dot v = u + \Theta^\top\Phi(v)$ с $\hat\Theta$
  не зависящим от $u$. Это нарушается при $m_{actual} \ne m_{nominal}$,
  и MRAC может проиграть PID. Решение — расширенный MRAC с оценкой
  input gain (выходит за рамки этого проекта).
- **Сильное насыщение** (наклон 15°, тяга 3·mg) в условиях очень
  сильного ветра (>5 м/с) делает обе схемы неэффективными.
- **Транзиент**: на первых 1–2 секундах MRAC может проиграть PID по
  IAE из-за инициализации $\hat\Theta = 0$. После warm-up разрыв
  закрывается.

### 9.3 Interpretation

Главный вывод: **σ-модифицированный MRAC на трансляционном контуре —
правильный приём для дрона с ветром**. Он:
1. **Доказуемо устойчив** (Lyapunov UUB).
2. **Численно превосходит PID** на жёстких сценариях.
3. **Простой по структуре** (3 независимых 1-D MRAC по осям).
4. Имеет **физически осмысленный регрессор** $\Phi(v) = [1, v, |v|v]$.

Документ [MRAC.md](MRAC.md) содержит полную защиту: вывод по Lyapunov,
обоснование σ-mod, обсуждение PE, разница с PID, литература.
Используется как «знание для защиты», на которое можно ссылаться.

---

## Структура репозитория

```
project_2_Adaptive_control_Drone_Wind/
├── README.md                    ← этот файл
├── MRAC.md                      ← глубокое описание MRAC (для защиты)
├── main.py                      ← полный пайплайн с анимацией
├── pyproject.toml
├── scripts/
│   └── generate_report.py       ← 9 PNG-графиков для отчёта
├── src/
│   ├── system.py                ← 12-state модель квадрокоптера
│   ├── controller.py            ← Controller (PID), MRACController, MRACAxis1D
│   ├── simulation.py            ← RK4 + условие остановки
│   ├── visualization.py         ← matplotlib 3D animation, compare-overlay
│   ├── plotly_dashboard.py      ← интерактивный HTML
│   └── plots.py                 ← time-series графики, plot_compare, plot_adaptation
├── figures/                     ← статические PNG для отчёта
└── animations/                  ← (опционально) GIF/MP4 файлы
```
