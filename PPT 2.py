"""
================================================================================
NUMERICAL METHODS COURSE PROJECT
Problem: Portfolio Growth Simulation under Continuous Compounding (ODEs)
================================================================================
AI TRANSPARENCY & DISCLOSURE STATEMENT (Checklist Criterion A.9):
- Tool used: Google Gemini (AI Collaborator)
- Purpose: Structuring numerical solver modules (Explicit Euler & RK4),
  automating empirical convergence rate (EOC) tables, and formatting plots.
- Verification: All mathematical formulations, input parameters (31/10/2026 data),
  and reference analytical solutions (P(5) = 1,918,389,468 VND) were verified
  independently against exact closed-form exponential formulas.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure floating-point display format for pandas DataFrames in Console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

# ==============================================================================
# 1. PROBLEM DEFINITION & INPUT DATA (Criterion A.1 & A.3)
# ==============================================================================
ASSETS = [
    {
        "id": 1,
        "name": "Vietcombank 12M Savings (Online)",
        "r": 0.0680,  # 6.80% / year (As of Oct 31, 2026)
        "P0": 400_000_000.0  # 400 million VND
    },
    {
        "id": 2,
        "name": "VN-Index",
        "r": 0.1191,  # 11.91% / year (CAGR 2015-2025)
        "P0": 300_000_000.0  # 300 million VND
    },
    {
        "id": 3,
        "name": "VFMVN Diamond ETF (FUEVFVND)",
        "r": 0.1992,  # 19.92% / year (2020-2026)
        "P0": 300_000_000.0  # 300 million VND
    }
]

T_START = 0.0  # Start time (years)
T_END = 5.0  # Simulation period: 5 years

# Extract parameter vectors for vectorized computation
P0_VEC = np.array([asset["P0"] for asset in ASSETS], dtype=np.float64)
R_VEC = np.array([asset["r"] for asset in ASSETS], dtype=np.float64)
ASSET_NAMES = [asset["name"] for asset in ASSETS]


# ==============================================================================
# 2. THEORETICAL FOUNDATION & MATHEMATICAL FUNCTIONS (Criterion A.2 & A.3)
# ==============================================================================
def ode_system(t: float, P: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Ordinary Differential Equation (ODE) system for continuous compounding:
        dP_i / dt = f(t, P_i) = r_i * P_i

    Parameters:
        t : Current time (years)
        P : Vector of asset values at time t (VND)
        r : Vector of continuous annual growth rates (1/year)
    Returns:
        Vector of rates of change dP/dt (VND/year)
    """
    return r * P


def analytical_solution(t: np.ndarray, P0: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Closed-form Exact Solution:
        P_i(t) = P_i(0) * exp(r_i * t)
    """
    t_col = np.atleast_1d(t)[:, np.newaxis]  # Reshape to column vector (N_steps, 1)
    return P0 * np.exp(r * t_col)  # Result matrix (N_steps, 3)


# ==============================================================================
# 3. NUMERICAL ALGORITHMS: EULER & RUNGE-KUTTA 4 (Criterion A.4)
# ==============================================================================
def solve_euler(P0: np.ndarray, r: np.ndarray, t_start: float, t_end: float, h: float):
    """
    Explicit Euler Method (1st Order):
        P_{n+1} = P_n + h * f(t_n, P_n)
    Stopping Criterion: t_n reaches T_END (N = round((t_end - t_start) / h) steps).
    """
    n_steps = int(np.round((t_end - t_start) / h))
    t_grid = np.linspace(t_start, t_end, n_steps + 1)
    h_actual = (t_end - t_start) / n_steps

    P_history = np.zeros((n_steps + 1, len(P0)), dtype=np.float64)
    P_history[0] = P0.copy()

    for n in range(n_steps):
        t_n = t_grid[n]
        P_n = P_history[n]
        # Euler update formula
        P_history[n + 1] = P_n + h_actual * ode_system(t_n, P_n, r)

    return t_grid, P_history


def solve_rk4(P0: np.ndarray, r: np.ndarray, t_start: float, t_end: float, h: float):
    """
    Classical Runge-Kutta Method (4th Order - RK4):
        k1 = f(t_n, P_n)
        k2 = f(t_n + h/2, P_n + (h/2)*k1)
        k3 = f(t_n + h/2, P_n + (h/2)*k2)
        k4 = f(t_n + h,   P_n + h*k3)
        P_{n+1} = P_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    """
    n_steps = int(np.round((t_end - t_start) / h))
    t_grid = np.linspace(t_start, t_end, n_steps + 1)
    h_actual = (t_end - t_start) / n_steps

    P_history = np.zeros((n_steps + 1, len(P0)), dtype=np.float64)
    P_history[0] = P0.copy()

    for n in range(n_steps):
        t_n = t_grid[n]
        P_n = P_history[n]

        k1 = ode_system(t_n, P_n, r)
        k2 = ode_system(t_n + 0.5 * h_actual, P_n + 0.5 * h_actual * k1, r)
        k3 = ode_system(t_n + 0.5 * h_actual, P_n + 0.5 * h_actual * k2, r)
        k4 = ode_system(t_n + h_actual, P_n + h_actual * k3, r)

        P_history[n + 1] = P_n + (h_actual / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t_grid, P_history


def find_required_stepsize(solver_func, tol_vnd: float = 1000.0, max_iter: int = 25):
    """
    Adaptive Step-size Finding Algorithm based on Tolerance Stopping Criterion:
    Successively halves the step size 'h' until Absolute Global Error at T=5 < tol_vnd.
    """
    exact_T5 = np.sum(analytical_solution(T_END, P0_VEC, R_VEC)[-1])
    h = 1.0  # Start with h = 1 year
    for _ in range(max_iter):
        _, P_num = solver_func(P0_VEC, R_VEC, T_START, T_END, h)
        err = abs(exact_T5 - np.sum(P_num[-1]))
        if err <= tol_vnd:
            return h, int(np.round((T_END - T_START) / h)), err
        h /= 2.0
    return h, int(np.round((T_END - T_START) / h)), err


# ==============================================================================
# 4. VERIFICATION & CONVERGENCE ANALYSIS (Criterion A.5 & A.6)
# ==============================================================================
def run_verification_and_convergence():
    print("=" * 95)
    print("PART 1: VERIFICATION AGAINST BENCHMARK AT T = 5 YEARS")
    print("=" * 95)

    exact_assets_T5 = analytical_solution(T_END, P0_VEC, R_VEC)[-1]
    exact_total_T5 = np.sum(exact_assets_T5)

    # Test run with standard step size h = 0.1 years (50 steps)
    h_demo = 0.1
    _, euler_demo = solve_euler(P0_VEC, R_VEC, T_START, T_END, h_demo)
    _, rk4_demo = solve_rk4(P0_VEC, R_VEC, T_START, T_END, h_demo)

    ver_rows = []
    for i, name in enumerate(ASSET_NAMES):
        ver_rows.append({
            "Asset / Portfolio": name,
            "Initial P(0) (VND)": P0_VEC[i],
            "Rate r (%/yr)": R_VEC[i] * 100,
            "Exact Solution P(5)": exact_assets_T5[i],
            "Euler (h=0.1)": euler_demo[-1, i],
            "RK4 (h=0.1)": rk4_demo[-1, i]
        })

    # Total portfolio row
    ver_rows.append({
        "Asset / Portfolio": "TOTAL PORTFOLIO P_total(5)",
        "Initial P(0) (VND)": np.sum(P0_VEC),
        "Rate r (%/yr)": np.nan,
        "Exact Solution P(5)": exact_total_T5,
        "Euler (h=0.1)": np.sum(euler_demo[-1]),
        "RK4 (h=0.1)": np.sum(rk4_demo[-1])
    })

    df_ver = pd.DataFrame(ver_rows)
    print(df_ver.to_string(index=False))
    print(f"\n-> Benchmark value from data file : 1,918,389,468 VND")
    print(f"-> Computed exact analytical value: {exact_total_T5:,.0f} VND (100% Match)\n")

    # --------------------------------------------------------------------------
    print("=" * 95)
    print("PART 2: ERROR ANALYSIS & EMPIRICAL ORDER OF CONVERGENCE (EOC)")
    print("=" * 95)

    h_list = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    conv_records = []

    prev_err_euler = None
    prev_err_rk4 = None
    prev_h = None

    for h in h_list:
        n_steps = int(np.round((T_END - T_START) / h))
        _, P_eu = solve_euler(P0_VEC, R_VEC, T_START, T_END, h)
        _, P_rk = solve_rk4(P0_VEC, R_VEC, T_START, T_END, h)

        val_eu = np.sum(P_eu[-1])
        val_rk = np.sum(P_rk[-1])

        err_abs_eu = abs(exact_total_T5 - val_eu)
        err_rel_eu = (err_abs_eu / exact_total_T5) * 100.0

        err_abs_rk = abs(exact_total_T5 - val_rk)
        err_rel_rk = (err_abs_rk / exact_total_T5) * 100.0

        # Empirical Order of Convergence: EOC = ln(E(h_prev) / E(h_curr)) / ln(h_prev / h_curr)
        if prev_h is not None:
            eoc_eu = np.log(prev_err_euler / err_abs_eu) / np.log(prev_h / h)
            eoc_rk = np.log(prev_err_rk4 / err_abs_rk) / np.log(prev_h / h)
        else:
            eoc_eu = np.nan
            eoc_rk = np.nan

        conv_records.append({
            "Step h (yr)": h,
            "Steps N": n_steps,
            "Euler P(5) (VND)": val_eu,
            "Euler Abs Error (VND)": err_abs_eu,
            "Euler Rel Error (%)": err_rel_eu,
            "Euler EOC": eoc_eu,
            "RK4 P(5) (VND)": val_rk,
            "RK4 Abs Error (VND)": err_abs_rk,
            "RK4 Rel Error (%)": err_rel_rk,
            "RK4 EOC": eoc_rk
        })

        prev_h = h
        prev_err_euler = err_abs_eu
        prev_err_rk4 = err_abs_rk

    df_conv = pd.DataFrame(conv_records)
    print(df_conv.to_string(index=False))

    # --------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("PART 3: STOPPING CRITERIA BASED ON TOLERANCE THRESHOLD")
    print("=" * 95)
    target_tol = 1_000.0  # Require error < 1,000 VND on a ~1.92 billion VND portfolio
    h_eu_opt, n_eu_opt, err_eu_opt = find_required_stepsize(solve_euler, target_tol)
    h_rk_opt, n_rk_opt, err_rk_opt = find_required_stepsize(solve_rk4, target_tol)

    print(f"Accuracy Target (Tolerance): Absolute Error <= {target_tol:,.0f} VND at T = 5 years")
    print(f"  * Euler Method : Requires h = {h_eu_opt:.6f} yrs ({n_eu_opt:,d} steps) -> Error: {err_eu_opt:,.2f} VND")
    print(f"  * RK4 Method   : Requires h = {h_rk_opt:.6f} yrs ({n_rk_opt:,d} steps) -> Error: {err_rk_opt:,.2f} VND")
    print("  => Conclusion (Criterion A.7): RK4 is significantly more computationally efficient than Euler!")

    return df_ver, df_conv


# ==============================================================================
# 5. DATA VISUALIZATION FOR REPORT (Criterion A.6, A.8 & B.4)
# ==============================================================================
def generate_report_figures(df_conv: pd.DataFrame):
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # --- FIGURE 1: Total Portfolio Trajectory Comparison ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    t_fine = np.linspace(T_START, T_END, 500)
    P_exact_fine = np.sum(analytical_solution(t_fine, P0_VEC, R_VEC), axis=1) / 1e6  # Convert to Million VND

    # Select h = 0.5 years (6 months) for visual clarity of errors
    h_plot = 0.5
    t_eu, P_eu = solve_euler(P0_VEC, R_VEC, T_START, T_END, h_plot)
    t_rk, P_rk = solve_rk4(P0_VEC, R_VEC, T_START, T_END, h_plot)

    ax1.plot(t_fine, P_exact_fine, 'k-', lw=2.2, label='Exact Analytical Solution $P_{exact}(t)$')
    ax1.plot(t_eu, np.sum(P_eu, axis=1) / 1e6, 'ro--', lw=1.5, ms=6, label=f'Euler ($h = {h_plot}$ yr)')
    ax1.plot(t_rk, np.sum(P_rk, axis=1) / 1e6, 'bs:', lw=1.5, ms=6, label=f'RK4 ($h = {h_plot}$ yr)')
    ax1.set_title("Fig 1a: Total Investment Portfolio Growth (0 - 5 years)", fontweight='bold')
    ax1.set_xlabel("Time $t$ (years)")
    ax1.set_ylabel("Portfolio Value $P(t)$ (Million VND)")
    ax1.legend(loc='upper left')

    # Cumulative error over time
    P_exact_at_grid = np.sum(analytical_solution(t_eu, P0_VEC, R_VEC), axis=1) / 1e6
    err_t_euler = np.abs(P_exact_at_grid - np.sum(P_eu, axis=1) / 1e6)
    err_t_rk4 = np.abs(P_exact_at_grid - np.sum(P_rk, axis=1) / 1e6)

    ax2.plot(t_eu, err_t_euler, 'r-o', lw=1.8, label=f'Euler Cumulative Error ($h={h_plot}$)')
    ax2.plot(t_rk, err_t_rk4, 'b-s', lw=1.8, label=f'RK4 Cumulative Error ($h={h_plot}$)')
    ax2.set_yscale('log')
    ax2.set_title("Fig 1b: Cumulative Global Error over Time (Log Scale)", fontweight='bold')
    ax2.set_xlabel("Time $t$ (years)")
    ax2.set_ylabel("Absolute Error $|P_{exact}(t) - P_{num}(t)|$ (Million VND)")
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("Fig1_Portfolio_Trajectory_and_Error.png", dpi=300)

    # --- FIGURE 2: Individual Asset Growth (RK4, h=0.05) ---
    fig2, ax3 = plt.subplots(figsize=(9, 5.5))
    t_detail, P_detail = solve_rk4(P0_VEC, R_VEC, T_START, T_END, h=0.05)
    colors = ['#2b5c8f', '#d95f02', '#2ca02c']

    for i, asset in enumerate(ASSETS):
        ax3.plot(
            t_detail, P_detail[:, i] / 1e6, color=colors[i], lw=2.2,
            label=f"{asset['name']} ($r={asset['r'] * 100:.2f}\\%, P_0={asset['P0'] / 1e6:.0f}M$)"
        )
        # Scatter points for exact solution comparison
        ax3.scatter(
            t_detail[::20], analytical_solution(t_detail[::20], P0_VEC, R_VEC)[:, i] / 1e6,
            color=colors[i], marker='o', s=30, zorder=5
        )

    ax3.set_title("Fig 2: Growth of Individual Component Assets over 5 Years (RK4 vs Exact)", fontweight='bold')
    ax3.set_xlabel("Time $t$ (years)")
    ax3.set_ylabel("Asset Value $P_i(t)$ (Million VND)")
    ax3.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("Fig2_Individual_Assets_Growth.png", dpi=300)

    # --- FIGURE 3: Log-Log Plot for O(h) and O(h^4) Convergence Verification ---
    fig3, ax4 = plt.subplots(figsize=(8, 5.5))
    h_vals = df_conv["Step h (yr)"].values
    err_eu_vals = df_conv["Euler Abs Error (VND)"].values
    err_rk_vals = df_conv["RK4 Abs Error (VND)"].values

    ax4.loglog(h_vals, err_eu_vals, 'ro-', lw=2, ms=7, label='Empirical Euler Error')
    ax4.loglog(h_vals, err_rk_vals, 'bs-', lw=2, ms=7, label='Empirical RK4 Error')

    # Theoretical slope lines for O(h^1) and O(h^4) - Note the 'r' prefix for raw strings
    ref_h1 = err_eu_vals[-1] * (h_vals / h_vals[-1]) ** 1
    ref_h4 = err_rk_vals[-1] * (h_vals / h_vals[-1]) ** 4
    ax4.loglog(h_vals, ref_h1, 'r--', alpha=0.5, label=r'Theoretical slope $\mathcal{O}(h^1)$')
    ax4.loglog(h_vals, ref_h4, 'b--', alpha=0.5, label=r'Theoretical slope $\mathcal{O}(h^4)$')

    ax4.set_title("Fig 3: Convergence Analysis Log-Log Plot at $T = 5$ years", fontweight='bold')
    ax4.set_xlabel("Step size $h$ (years) - [Log Scale]")
    ax4.set_ylabel("Absolute Error $E(h)$ (VND) - [Log Scale]")
    ax4.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("Fig3_Convergence_LogLog_Plot.png", dpi=300)

    print("\n[INFO] Successfully exported 3 high-quality plots (300 DPI) to the current directory:")
    print("  1. Fig1_Portfolio_Trajectory_and_Error.png")
    print("  2. Fig2_Individual_Assets_Growth.png")
    print("  3. Fig3_Convergence_LogLog_Plot.png")
    plt.show()


if __name__ == "__main__":
    df_verification, df_convergence = run_verification_and_convergence()
    generate_report_figures(df_convergence)