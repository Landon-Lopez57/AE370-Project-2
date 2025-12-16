import numpy as np
import matplotlib.pyplot as plt
from GP_2_Implementation import build_spherical_fv_system, trapezoid_step_fv


# ============================================================
# A1: MMS Convergence Test (Scheme A FV)
#   - Uses same FV operator and CN step as implementation
#   - Integrates to exact t_final (no rounding-time mismatch)
#   - SPACE test: oscillatory mode (n=8) to make spatial error visible
#   - TIME  test: smooth mode (n=2) + very fine N to isolate CN time order
# ============================================================

def fv_error_norms(T_num, T_ex, V_cells, Tref):
    e = T_num - T_ex
    l2_abs = np.sqrt(np.sum(V_cells * e**2) / np.sum(V_cells))
    linf   = np.max(np.abs(e))
    Tp     = T_ex - Tref
    l2_rel = np.sqrt(np.sum(V_cells * e**2) / np.sum(V_cells * Tp**2))
    return l2_abs, linf, l2_rel


# ---------- Manufactured solution ----------
# T(r,t) = Tref + A exp(-t/tau) f(r)
# f(r) = (r^2/R^2) sin(nπ r/R)  (smooth, regular at r=0)

def f_shape(r, R, n):
    w = n * np.pi / R
    return (r**2 / R**2) * np.sin(w * r)

def f_shape_r(r, R, n):
    w = n * np.pi / R
    return (2.0*r/R**2) * np.sin(w*r) + (r**2/R**2) * w * np.cos(w*r)

def lap_f_shape(r, R, n):
    # spherical laplacian: f'' + 2 f'/r
    w = n * np.pi / R
    s = w * r
    lap = (6.0/R**2)*np.sin(s) + (6.0*r*w/R**2)*np.cos(s) - (r**2*w**2/R**2)*np.sin(s)
    return np.where(r < 1e-14, 0.0, lap)

def T_exact(r, t, Tref, Aamp, tau, R, n):
    return Tref + Aamp * np.exp(-t/tau) * f_shape(r, R, n)

def q_in_exact(t, Aamp, tau, R, n, k):
    # FV forcing convention:
    #   q_in > 0 injects heat into solid
    # and q_in = k * dT/dr at r=R
    g = np.exp(-t/tau)
    dTdr_R = Aamp * g * f_shape_r(np.array([R]), R, n)[0]
    return k * dTdr_R

def qdot_exact(r, t, rho, cp, k, Tref, Aamp, tau, R, n):
    g = np.exp(-t/tau)
    f = f_shape(r, R, n)
    lapf = lap_f_shape(r, R, n)
    Tt = (-Aamp/tau) * g * f
    lapT = Aamp * g * lapf
    return rho*cp*Tt - k*lapT   # [W/m^3]


def run_mms(N, dt, t_final, R, rho, cp, k, Tref, Aamp, tau, nmode):
    layers = [(R, rho, cp, k)]
    system = build_spherical_fv_system(N, R, layers)
    (r_centers, r_faces, dr, rho_c, cp_c, k_c, V_cells, A_faces,
     M_diag, K_lower, K_diag, K_upper) = system

    T = T_exact(r_centers, 0.0, Tref, Aamp, tau, R, nmode)

    # integrate to EXACT t_final (last partial step)
    t = 0.0
    while t < t_final - 1e-14:
        dt_step = min(dt, t_final - t)
        t_np1   = t + dt_step

        q_n   = q_in_exact(t,     Aamp, tau, R, nmode, k)
        q_np1 = q_in_exact(t_np1, Aamp, tau, R, nmode, k)

        qdot_n   = qdot_exact(r_centers, t,     rho, cp, k, Tref, Aamp, tau, R, nmode)
        qdot_np1 = qdot_exact(r_centers, t_np1, rho, cp, k, Tref, Aamp, tau, R, nmode)

        T = trapezoid_step_fv(
            T, dt_step,
            M_diag, K_lower, K_diag, K_upper,
            V_cells, A_faces,
            q_n, q_np1,
            qdot_n=qdot_n, qdot_np1=qdot_np1
        )

        t = t_np1

    T_ex = T_exact(r_centers, t_final, Tref, Aamp, tau, R, nmode)
    l2_abs, linf_abs, l2_rel = fv_error_norms(T, T_ex, V_cells, Tref)

    return dict(N=N, dt=dt, r=r_centers, V_cells=V_cells,
                T_num=T, T_ex=T_ex,
                l2_abs=l2_abs, linf_abs=linf_abs, l2_rel=l2_rel)


def estimate_order(h1, e1, h2, e2):
    return np.log(e2/e1) / np.log(h2/h1)


# --------------------------
# Parameters
# --------------------------
R = 0.2
rho = 1.0
cp  = 1.0
k   = 1.0

Tref = 300.0
Aamp = 100.0
tau  = 0.5
t_final = 0.5


# ============================================================
# (1) SPACE convergence  (hard case, n=8)
# ============================================================
n_space = 8
dt_space = 1e-3
N_list = [30, 60, 120, 240, 480, 960]

outs = []
hs = []
es = []
for N in N_list:
    out = run_mms(N, dt_space, t_final, R, rho, cp, k, Tref, Aamp, tau, n_space)
    outs.append(out)
    hs.append(R/N)
    es.append(out["l2_abs"])

hs = np.array(hs)
es = np.array(es)

# Profile plot at a coarse N (shows why it looks "shifted")
out_coarse = outs[2]   # N=120
plt.figure()
plt.plot(out_coarse["r"], out_coarse["T_num"], label="numerical")
plt.plot(out_coarse["r"], out_coarse["T_ex"], "--", label="exact (MMS)")
plt.xlabel("r [m]")
plt.ylabel("T [K]")
plt.title("MMS (FV): final-time profile (space-test, oscillatory, N=120)")
plt.grid(True)
plt.legend()

# Profile plot at a fine N (should overlay much better)
out_fine = outs[-1]    # N=960
plt.figure()
plt.plot(out_fine["r"], out_fine["T_num"], label="numerical")
plt.plot(out_fine["r"], out_fine["T_ex"], "--", label="exact (MMS)")
plt.xlabel("r [m]")
plt.ylabel("T [K]")
plt.title("MMS (FV): final-time profile (space-test, oscillatory, N=960)")
plt.grid(True)
plt.legend()

print("Space convergence (vol-weighted abs L2):")
for i, N in enumerate(N_list):
    print(f"  N={N:4d}, h={hs[i]:.3e}, err={es[i]:.3e}")
for i in range(1, len(N_list)):
    p = estimate_order(hs[i-1], es[i-1], hs[i], es[i])
    print(f"  order between N={N_list[i-1]} and {N_list[i]}: p≈{p:.3f}")

plt.figure()
plt.loglog(hs, es, marker="o")
plt.gca().invert_xaxis()
plt.xlabel("h = R/N [m]")
plt.ylabel("abs L2 error [K]")
plt.title("MMS (FV): space convergence")
plt.grid(True, which="both")


# ============================================================
# (2) TIME convergence  (smooth case, n=2)
#     Use dt where time error dominates and avoid the tiny-error “cancellation” point.
# ============================================================
n_time = 2
N_time = 4000
dt_list = [0.16, 0.08, 0.04, 0.02]   # clean halving, shows ~2nd order for CN

dts = []
ets = []
for dtv in dt_list:
    outT = run_mms(N_time, dtv, t_final, R, rho, cp, k, Tref, Aamp, tau, n_time)
    dts.append(dtv)
    ets.append(outT["l2_abs"])

dts = np.array(dts)
ets = np.array(ets)

print("Time convergence (vol-weighted abs L2):")
for i, dtv in enumerate(dt_list):
    print(f"  dt={dtv:.3e}, err={ets[i]:.3e}")
for i in range(1, len(dt_list)):
    p = estimate_order(dts[i-1], ets[i-1], dts[i], ets[i])
    print(f"  order between dt={dt_list[i-1]} and {dt_list[i]}: p≈{p:.3f}")

plt.figure()
plt.loglog(dts, ets, marker="o")
plt.gca().invert_xaxis()
plt.xlabel("dt [s]")
plt.ylabel("abs L2 error [K]")
plt.title("MMS (FV): time convergence (CN should be ~2nd order)")
plt.grid(True, which="both")

plt.show()
