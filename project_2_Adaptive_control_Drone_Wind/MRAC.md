# MRAC + σ-modification: detailed description

This document gives the full justification of the adaptive part of the
drone controller. It is written so that one can defend the design against
questions of the form "why does this work", "where does this formula come
from", and "what does the theory guarantee". All derivations are written
out for a single velocity axis (the other two are identical copies).

## Contents

1. [Why adaptive control at all](#1-why-adaptive-control-at-all)
2. [Plant](#2-plant)
3. [Reference Model](#3-reference-model)
4. [Control law](#4-control-law)
5. [Tracking-error dynamics](#5-tracking-error-dynamics)
6. [Lyapunov function and update law](#6-lyapunov-function-and-update-law)
7. [σ-modification: why we need it](#7-σ-modification-why-we-need-it)
8. [Adaptation freeze on saturation](#8-adaptation-freeze-on-saturation)
9. [Parameter projection](#9-parameter-projection)
10. [The regressor Φ(v) and the choice behind it](#10-the-regressor-φv-and-the-choice-behind-it)
11. [Why MRAC on the velocity loop, not the angular loop](#11-why-mrac-on-the-velocity-loop-not-the-angular-loop)
12. [What the theory guarantees and what it does not](#12-what-the-theory-guarantees-and-what-it-does-not)
13. [Connection to the PID baseline](#13-connection-to-the-pid-baseline)
14. [Parameters and how to tune them](#14-parameters-and-how-to-tune-them)
15. [References](#15-references)

---

## 1. Why adaptive control at all

A classical PID works perfectly **when the plant model is known** and the
disturbances are small and stationary. A real drone in wind is the
opposite:

| Source of uncertainty | Effect |
|---|---|
| Aerodynamic drag $F_{\rm drag}(v_{\rm rel})$ | velocity-dependent, hard to model |
| Wind $v_w(t)$ | time-varying, unknown a priori |
| Mass / inertia inaccuracies | shift gravity compensation |
| Unmodelled motor dynamics / lags | bandwidth errors |

Adaptive control addresses this by **changing controller parameters in real
time** so that the plant behaves like a fixed reference model. It is the
parameter update — not just the control signal — that distinguishes
adaptive from merely robust control.

**MRAC** (Model Reference Adaptive Control) is one of the canonical
approaches: an explicit linear stable model `ẋ_m = A_m x_m + B_m r` is
posted, and the adapter parameters are updated so that plant → reference
model.

---

## 2. Plant

Take a single velocity axis (e.g. $v_x$). Linearised model:

```
v̇ = u + Δ(v, t)
```

- `v` — drone velocity along axis x (in the world frame)
- `u` — commanded horizontal acceleration component, controller output
       (later converted into roll/pitch and thrust)
- `Δ(v, t)` — **unknown disturbance**: drag, wind, vibrations, modelling
  errors

MRAC assumption: `Δ` is **linear in unknown parameters** (linear-in-parameters):

```
Δ(v, t) = Θ*ᵀ · Φ(v)
```

where:
- `Φ(v) ∈ ℝⁿ` — known **regressor** (basis functions), function of state
- `Θ* ∈ ℝⁿ` — **true** constant parameters, unknown

In our code the default regressor (`n_basis=3`) is

```
Φ(v) = [ 1,  v,  |v|·v ]ᵀ.
```

It captures
- a constant bias (mean wind),
- linear drag,
- quadratic drag.

If the true physics is exactly `−c₁(v − v_w) − c₂|v − v_w|(v − v_w)` then
Δ is **not strictly linear in our parameters with respect to Φ(v)**
(because of `v_w(t)` which depends on time, not on state). Therefore an
exact constant `Θ*` does not strictly exist — but MRAC still works,
because:
- σ-modification (see §7) keeps the parameters bounded;
- the adaptation does not depend on the exact linear-in-parameters
  property holding;
- we obtain UUB rather than asymptotic convergence to zero error.

---

## 3. Reference Model

The desired behaviour of the plant:

```
v̇_m = -a_m (v_m - v_des)
```

with `a_m > 0` the natural rate of the reference model. This is a stable
first-order system: if `v_des` is constant then `v_m → v_des`
exponentially with time constant `1/a_m`.

In code the defaults are `a_m_xy = 6.0` and `a_m_z = 8.0`, i.e. time
constants of **170 ms** for the horizontal axes and **125 ms** for the
vertical one. They should be:
- **higher** than typical wind frequencies (here up to ~3 rad/s) so that
  the reference model can track them,
- **lower** than the inner attitude loop's bandwidth (otherwise the plant
  cannot follow the reference),
- well below the discretisation Nyquist (`a_m · dt ≪ 2`).

The reference model integration is **exact** (first-order linear, constant
input on each step):

```python
decay = exp(-a_m * dt)
v_m   = v_des + (v_m - v_des) * decay
```

This is the closed-form solution of the linear first-order ODE on
`[t, t + dt]`. It removes the spurious "phantom error" that forward Euler
produced in the very first iteration when `v = v_m = 0`.

---

## 4. Control law

Idea: choose `u` so that, in the absence of `Δ`, the plant **exactly**
reproduces the reference model:

```
u = a_m (v_des - v) - Θ̂ᵀ Φ(v)
       ↑                ↑
       |                └── adaptive part, disturbance estimate
       └── nominal part, like a P-controller on velocity error
```

If we knew `Θ*`, the ideal `u = a_m(v_des - v) - Θ*ᵀ Φ(v)` would give

```
v̇ = u + Θ*ᵀ Φ = a_m(v_des - v) - Θ*ᵀ Φ + Θ*ᵀ Φ = a_m(v_des - v)
```

— exactly the reference dynamics. Therefore **`Θ̂` should drive
`Θ̃ᵀ Φ` close to zero** (or, more precisely, push the closed-loop into
that regime).

---

## 5. Tracking-error dynamics

Define the **tracking error** `e = v - v_m`. Substituting the control law:

```
v̇ = u + Δ
   = a_m(v_des - v) - Θ̂ᵀ Φ + Θ*ᵀ Φ
   = a_m(v_des - v) + (Θ* - Θ̂)ᵀ Φ
   = a_m(v_des - v) + Θ̃ᵀ Φ
```

where `Θ̃ = Θ* - Θ̂` is the **parameter error** (unknown, but used in the
analysis).

```
v̇_m = a_m(v_des - v_m)
```

Subtracting:

```
ė = v̇ - v̇_m
  = a_m(v_des - v) + Θ̃ᵀ Φ - a_m(v_des - v_m)
  = -a_m(v - v_m) + Θ̃ᵀ Φ
  = -a_m·e + Θ̃ᵀ Φ
```

**Key equation**: `ė = -a_m·e + Θ̃ᵀ Φ`. This is a stable autonomous system
with a perturbation proportional to the parameter error.

---

## 6. Lyapunov function and update law

Candidate:

```
V = ½ e² + 1/(2γ) ||Θ̃||²,    γ > 0
```

It is positive definite and `V → ∞` as `‖(e, Θ̃)‖ → ∞`.

Time derivative along trajectories:

```
V̇ = e·ė + (1/γ) Θ̃ᵀ Θ̃̇
  = e·(-a_m·e + Θ̃ᵀ Φ) + (1/γ) Θ̃ᵀ Θ̃̇
  = -a_m e² + e Θ̃ᵀ Φ + (1/γ) Θ̃ᵀ Θ̃̇
```

`Θ*` is a **constant**, so `Θ̃̇ = -Θ̂̇`. Rearranging:

```
V̇ = -a_m e² + Θ̃ᵀ (e Φ - (1/γ) Θ̂̇)
```

To kill the unknown term, choose the update law so that the bracket is
zero:

```
Θ̂̇ = γ · e · Φ(v)        ← basic MRAC law
```

Then

```
V̇ = -a_m·e² ≤ 0
```

By Barbalat's lemma `e → 0` as `t → ∞`. The parameters `Θ̂` are bounded
but **not necessarily convergent to `Θ*`**. Convergence requires
**persistent excitation** (PE) — `Φ` must be "rich enough" in time:

```
∃ T, α > 0:   ∫_t^{t+T} Φ(τ) Φ(τ)ᵀ dτ ≥ α·I    ∀ t
```

In our task this is **not guaranteed** (see §12).

---

## 7. σ-modification: why we need it

The basic update law `Θ̂̇ = γ·e·Φ` has a subtle problem: in the absence of
PE and in the presence of unmodelled disturbances (measurement noise,
nonlinearities outside `Φ`) parameters can **slowly drift** until they hit
some numerical bound.

**σ-modification** (Ioannou, 1984) adds a "leak":

```
Θ̂̇ = γ · e · Φ - σ · Θ̂,    σ > 0
```

The term `-σ·Θ̂` pulls the parameters back toward zero whenever `e` is
small.

Lyapunov:

```
V̇ = -a_m e² + Θ̃ᵀ (e Φ - (1/γ) Θ̂̇)
   = -a_m e² + Θ̃ᵀ (e Φ - eΦ + (σ/γ)·Θ̂)        ← substituted update law
   = -a_m e² + (σ/γ) Θ̃ᵀ Θ̂
```

Using `Θ̂ = Θ* - Θ̃`:

```
V̇ = -a_m e² + (σ/γ)·Θ̃ᵀ(Θ* - Θ̃)
   = -a_m e² - (σ/γ)·||Θ̃||² + (σ/γ)·Θ̃ᵀ Θ*
```

By Young's inequality `Θ̃ᵀΘ* ≤ ½||Θ̃||² + ½||Θ*||²`:

```
V̇ ≤ -a_m e² - (σ/2γ)·||Θ̃||² + (σ/2γ)·||Θ*||²
```

`V̇ ≤ 0` outside the set

```
{ (e, Θ̃) :  a_m e² + (σ/2γ)||Θ̃||² ≤ (σ/2γ)||Θ*||² }
```

— an ellipsoid of radius ~ `||Θ*||` around the origin. This is **uniform
ultimate boundedness (UUB)**: trajectories enter the set and stay there.

**Cost** of σ-modification: an additional steady-state error of order
`σ·||Θ*||`. Larger σ → less drift, more bias. Trade-off: we use **σ = 0.5**,
which empirically gives a good balance.

Alternatives: e-modification (`Θ̂̇ = γ·e·Φ - σ·||e||·Θ̂`) — leakage only when
the error is large; dead-zone — disable adaptation below an error
threshold.

---

## 8. Adaptation freeze on saturation

The control input `u` is physically bounded (max tilt 15° → max horizontal
acceleration ≈ g·tan(15°)). When `u_unsat` exceeds the limit, the system
receives the clipped signal:

```
u = clip(u_unsat, -u_max, u_max)
```

Under saturation the error `e` grows **not because of poor parameters** but
because of insufficient control authority. Adapting at this moment would
push `Θ̂` in the wrong direction (it would try to "compensate" by
producing larger commands, but those go through the same saturated channel,
creating positive feedback in adaptation).

In code:

```python
saturated = (u != u_unsat)
if not saturated:
    Θ̂ += dt · (γ · e · Φ - σ · Θ̂)
```

In theory: stop learning during saturation; the closed loop remains stable
through bounded trajectory + σ-mod. This is a known technique discussed in
Lavretsky & Wise §10.2 (saturation handling). Without it, `Θ̂` drifted into
the projection bound under strong wind and the algorithm performed worse
than PID (this was observed during development).

---

## 9. Parameter projection

Additionally:

```python
if ||Θ̂|| > θ_max:
    Θ̂ ← Θ̂ · θ_max / ||Θ̂||
```

— a safe **projection onto a bounded set**. This is a hard limit in case
σ-modification and saturation freeze are insufficient (numerical hiccups,
extreme transients). By construction, projection does not break the
Lyapunov properties of the scheme (see Lavretsky §11.4).

We use `θ_max = 8.0`. With the regressor `Φ(v)` and `n_basis = 3`, the
maximum value of `Θ̂ᵀ Φ` at `v = 5 m/s` is at most
`||Θ̂|| · ||Φ|| = 8 · √(1 + 25 + 625) ≈ 8 · 25.5 ≈ 200 m/s²` — far above any
realistic compensation. The limit is loose and almost never engages.

---

## 10. The regressor Φ(v) and the choice behind it

```
Φ(v) = [ 1,  v,  |v|·v ]ᵀ
```

| Term | What it models | Parameter Θ̂ |
|---|---|---|
| `1` | constant bias (mean wind, gravity offset) | Θ̂_0 |
| `v` | linear drag `c₁·v` | Θ̂_1 |
| `|v|·v` | quadratic drag `c₂·v\|v\|` | Θ̂_2 |

The **true** disturbance in the model is

```
δ_true(v, t) = (1/m) [ -c₁(v - v_w(t)) - c₂|v - v_w(t)|(v - v_w(t)) ]
```

This is **not** a linear combination of our `Φ(v)` — `v_w(t)` shows up,
which is time-varying. The best approximation is:
- Θ̂_0 captures the mean drag = `(c₁ ⟨v_w⟩ + c₂ ⟨|v_w| v_w⟩) / m`
- Θ̂_1 ≈ -c₁/m
- Θ̂_2 ≈ -c₂/m

Because `v_w` is non-stationary, the parameters **fluctuate** around these
means.

**Alternative regressors**:
- `Φ = [1]` — bias compensation only (if the wind is roughly constant)
- `Φ = [1, v, v², sin(ωt), cos(ωt)]` — add a time-varying basis (if the
  dominant frequency is known)
- `Φ` = radial basis functions (RBF) — universal approximator

The code exposes a parameter `n_basis ∈ {1, 2, 3}` for simplification.

---

## 11. Why MRAC on the velocity loop, not the angular loop

Historically (in textbooks) MRAC is placed on the **angular** dynamics:

```
ω̇ = (1/I)·τ + (1/I)·δ_τ
```

This is natural for fighter jets / craft with high uncertainty in the
torque channel (failure modes, damage).

For a drone in wind:
- The angular dynamics are **very well modelled** — there is little
  uncertainty.
- The wind disturbance does **not enter as a torque**, but as a **drag
  force** on the body.
- MRAC on the angular loop estimates "nothing" → improvement is small (we
  observed ~3 % over PID).

Moving MRAC to the velocity loop:
- `δ` corresponds directly to drag — there is **something to estimate**;
- the regressor `Φ(v)` is physically meaningful;
- a second adaptation on the angular loop is unnecessary (PID is sufficient
  there).

This is reflected in the experiments: angular MRAC +3 %, velocity MRAC
**+44 %** under moderate wind.

In Lavretsky's terminology this is called *outer-loop MRAC* and is
discussed in the chapter on guidance & control architecture.

---

## 12. What the theory guarantees and what it does not

### Guaranteed

1. **Boundedness**: all signals `(v, v_m, e, Θ̂)` are bounded as `t → ∞`.
   Proof: `V` is radially unbounded and `V̇ ≤ 0` outside the UUB
   ellipsoid.
2. **Tracking-error UUB**: `||e||` enters a bounded set of radius
   `O(σ·||Θ*||/a_m)`. In practice, `||e||` settles into a small
   neighborhood (in our case ≤ 0.5 m/s under wind).
3. **Stability** in the sense of Lyapunov.

### **NOT** guaranteed

1. **Convergence `Θ̂ → Θ*`**. This requires **persistent excitation** (PE):

   ```
   ∃ T, α > 0:  ∫_t^{t+T} Φ(v(τ)) Φ(v(τ))ᵀ dτ ≥ α·I
   ```

   Our `Φ(v) = [1, v, |v|·v]`. When the drone is near the target,
   `v ≈ 0`, so `Φ ≈ [1, 0, 0]ᵀ` — the matrix is rank-deficient and PE
   fails. Therefore, even though the drone sits exactly at the target,
   the estimates `Θ̂_1, Θ̂_2` can be inaccurate.

   This is fine: we don't need an exact identification of the wind — we
   need accurate compensation, which is exactly what UUB delivers.

2. **Asymptotic convergence to zero**. With σ-mod we only get UUB with a
   known radius. `e → 0` is not strictly claimed.

3. **Transient behaviour**. The theory says nothing about overshoot or
   transient amplitude. In practice, the early adaptation steps can
   produce short spikes.

4. **Robustness to unmodelled dynamics**. If the true dynamics violate
   linear-in-Φ, there is no guarantee — but σ-mod gives *robust adaptive
   control* in the sense of Ioannou & Sun §8: boundedness is preserved
   for bounded unmodelled dynamics.

### Practical implications for the report

When showing a "wind estimate vs true wind" plot, **always** state that
- the curve `Θ̂(t)ᵀ Φ(v(t))` is **not** an accurate estimate of the
  disturbance,
- it is sufficient **only** for tracking compensation in the closed loop,
- divergence from the true `δ_true(t)` is expected without PE.

This is the correct and honest interpretation, not a bug.

---

## 13. Connection to the PID baseline

Can MRAC be considered a generalisation of PID? Partially:

| Regressor | Equivalent to a classical controller |
|---|---|
| `Φ = [1]` (n_basis=1) | P + adaptive bias compensation ≈ PI with adaptive integral |
| `Φ = [1, v]` | + adaptive damping (P + adaptive PI with variable gain) |
| `Φ = [1, v, \|v\|·v]` | + adaptive drag model (nonlinear compensation) |

Advantages of MRAC over a fixed PI:
- the update law is **derived from Lyapunov**, not tuned heuristically;
- the parameter `γ` controls adaptation speed (vs `ki` of a PI which is
  fixed and prone to integral wind-up);
- σ-mod gives a principled defence against drift (vs ad-hoc anti-windup).

Disadvantages:
- more parameters to tune (`a_m, γ, σ, θ_max, regressor`);
- harder to verify (a Lyapunov analysis is required).

---

## 14. Parameters and how to tune them

### `a_m` — reference-model rate

- **Too small** (a_m << dynamic characteristics): the reference is slow,
  the plant overtakes it, e becomes negative, MRAC starts "braking" the
  drone → loss of bandwidth.
- **Too large** (a_m → ∞): the reference asks for an instant response, the
  plant cannot follow, e is permanently large, adaptation works in vain.
- **Sweet spot**: `a_m ≈ ω_inner_loop / 3` — the inner loop (PID on
  attitude) must be at least 3× faster than the reference. We have
  ω_n ≈ 30 rad/s for the inner loop, so `a_m ≤ 10`. We chose 6 (xy) and
  8 (z).

### `γ` — adaptation rate

- **Too small**: parameters do not adapt → PID-like behaviour.
- **Too large**: high-gain adaptation, oscillatory transient, sensitivity
  to noise.
- **Sweet spot**: `γ` of order `1..10`; in code `γ = 3`.

A useful order-of-magnitude relation: the effective adaptation time
constant is `≈ 1/(γ·||Φ||²)`. With `||Φ|| ~ 5` and `γ = 3`,
τ_adapt ≈ 13 ms — faster than the reference model, which is normal.

### `σ` — leakage rate

Small enough not to spoil steady-state, but large enough for stability —
typically 0.1..1.0. We use `σ = 0.5`.

The effective steady-state parameter bias is
`~ σ·||Θ*||/(γ·||Φ||²)` — about 0.05 in our setup, negligible.

### `θ_max` — projection limit

A "hard fuse". Should be 2..5× the expected `||Θ*||`. From our drag-parameter
analysis `||Θ*|| ~ 1`, so `θ_max = 5..10`. We chose 8.

### `n_basis` — regressor dimension

- 1: bias only — for quasi-stationary wind
- 2: + linear damping
- 3: + quadratic drag — the standard for a drone. We use it.

---

## 15. References

1. **Lavretsky E., Wise K.** *Robust and Adaptive Control with Aerospace
   Applications.* Springer, 2013. — the principal MRAC textbook,
   chs. 9–11 (direct MRAC, σ-mod, projection).

2. **Ioannou P., Sun J.** *Robust Adaptive Control.* Prentice-Hall, 1996.
   — deep theoretical treatise, ch. 8 (robust adaptation modifications).

3. **Hovakimyan N., Cao C.** *L1 Adaptive Control Theory.* SIAM, 2010. —
   modern alternative to MRAC with a low-pass filter on the adaptation.

4. **Astrom K., Wittenmark B.** *Adaptive Control.* Addison-Wesley, 1995.
   — a classic survey, STR / MRAC / GAS.

5. **Slotine J.-J., Li W.** *Applied Nonlinear Control.* Prentice-Hall,
   1991. — ch. 8 on adaptive control with a clear Lyapunov-based
   exposition.

---

## TL;DR for the defense

1. Plant: `v̇ = u + Θ*ᵀ Φ(v)`, regressor `Φ = [1, v, |v|v]`.
2. Reference model: `v̇_m = -a_m(v_m - v_des)`, chosen stable.
3. Control: `u = a_m(v_des - v) - Θ̂ᵀ Φ(v)`.
4. Adaptive law: `Θ̂̇ = γ·e·Φ - σ·Θ̂`, `e = v - v_m`.
5. Lyapunov: `V = ½e² + 1/(2γ)||Θ̃||²`, `V̇ ≤ -a_m e² + (σ/2γ)||Θ*||²`.
6. Guarantee: **UUB** (e and Θ̂ bounded, e → small neighborhood of zero).
7. Not guaranteed: `Θ̂ → Θ*` — that requires PE, which is generally absent
   in our task.
8. σ-mod is needed for **robustness** (drift protection without PE).
9. Saturation freeze + projection — practical refinements that do not
   violate the Lyapunov analysis.

If asked about the precision of `Θ̂`: "the parameters are bounded and
sufficient for UUB-tracking; convergence to `Θ*` requires PE, which we
neither have nor need for tracking."
