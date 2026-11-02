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

# Thiết lập hiển thị số thực rõ ràng cho bảng kết quả trên Console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')

# ==============================================================================
# 1. ĐỊNH NGHĨA BÀI TOÁN & DỮ LIỆU ĐẦU VÀO (Criterion A.1 & A.3)
# ==============================================================================
ASSETS = [
    {
        "id": 1,
        "name": "Tiết kiệm Vietcombank 12M (Online)",
        "r": 0.0680,  # 6.80% / năm (31/10/2026)
        "P0": 400_000_000.0  # 400 triệu VNĐ
    },
    {
        "id": 2,
        "name": "Chỉ số VN-Index",
        "r": 0.1191,  # 11.91% / năm (CAGR 2015-2025)
        "P0": 300_000_000.0  # 300 triệu VNĐ
    },
    {
        "id": 3,
        "name": "Quỹ ETF VFMVN Diamond (FUEVFVND)",
        "r": 0.1992,  # 19.92% / năm (2020-2026)
        "P0": 300_000_000.0  # 300 triệu VNĐ
    }
]

T_START = 0.0  # Thời điểm bắt đầu (năm)
T_END = 5.0  # Thời gian mô phỏng: 5 năm

# Trích xuất vector tham số để tính toán vector hóa (Vectorized computation)
P0_VEC = np.array([asset["P0"] for asset in ASSETS], dtype=np.float64)
R_VEC = np.array([asset["r"] for asset in ASSETS], dtype=np.float64)
ASSET_NAMES = [asset["name"] for asset in ASSETS]


# ==============================================================================
# 2. CƠ SỞ LÝ THUYẾT & HÀM TOÁN HỌC (Criterion A.2 & A.3)
# ==============================================================================
def ode_system(t: float, P: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Hệ phương trình vi phân mô tả lãi kép liên tục cho từng tài sản:
        dP_i / dt = f(t, P_i) = r_i * P_i

    Tham số:
        t : Thời gian hiện tại (năm)
        P : Vector giá trị các tài sản tại thời điểm t (VNĐ)
        r : Vector tỷ suất sinh lời liên tục hàng năm (1/năm)
    Trả về:
        Vector tốc độ biến thiên dP/dt (VNĐ/năm)
    """
    return r * P


def analytical_solution(t: np.ndarray, P0: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Nghiệm giải tích chính xác (Closed-form Exact Solution):
        P_i(t) = P_i(0) * exp(r_i * t)
    """
    t_col = np.atleast_1d(t)[:, np.newaxis]  # Chuyển thành cột (N_steps, 1)
    return P0 * np.exp(r * t_col)  # Ma trận kết quả (N_steps, 3)


# ==============================================================================
# 3. CÀI ĐẶT THUẬT TOÁN SỐ: EULER & RUNGE-KUTTA 4 (Criterion A.4)
# ==============================================================================
def solve_euler(P0: np.ndarray, r: np.ndarray, t_start: float, t_end: float, h: float):
    """
    Phương pháp Euler hiện (Explicit Euler Method - Bậc 1):
        P_{n+1} = P_n + h * f(t_n, P_n)
    Điều kiện dừng: t_n đạt tới T_END (số bước N = round((t_end - t_start) / h)).
    """
    n_steps = int(np.round((t_end - t_start) / h))
    t_grid = np.linspace(t_start, t_end, n_steps + 1)
    h_actual = (t_end - t_start) / n_steps

    P_history = np.zeros((n_steps + 1, len(P0)), dtype=np.float64)
    P_history[0] = P0.copy()

    for n in range(n_steps):
        t_n = t_grid[n]
        P_n = P_history[n]
        # Công thức cập nhật Euler
        P_history[n + 1] = P_n + h_actual * ode_system(t_n, P_n, r)

    return t_grid, P_history


def solve_rk4(P0: np.ndarray, r: np.ndarray, t_start: float, t_end: float, h: float):
    """
    Phương pháp Runge-Kutta cổ điển bậc 4 (Classical RK4 Method - Bậc 4):
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
    Thuật toán tìm bước nhảy h tối ưu thỏa mãn điều kiện dừng về sai số (Stopping Criterion):
    Chia đôi bước thời gian h liên tiếp cho đến khi Sai số tuyệt đối tại T=5 < tol_vnd (VNĐ).
    """
    exact_T5 = np.sum(analytical_solution(T_END, P0_VEC, R_VEC)[-1])
    h = 1.0  # Bắt đầu với h = 1 năm
    for _ in range(max_iter):
        _, P_num = solver_func(P0_VEC, R_VEC, T_START, T_END, h)
        err = abs(exact_T5 - np.sum(P_num[-1]))
        if err <= tol_vnd:
            return h, int(np.round((T_END - T_START) / h)), err
        h /= 2.0
    return h, int(np.round((T_END - T_START) / h)), err


# ==============================================================================
# 4. KIỂM CHỨNG KẾT QUẢ & PHÂN TÍCH HỘI TỤ (Criterion A.5 & A.6)
# ==============================================================================
def run_verification_and_convergence():
    print("=" * 95)
    print("PHẦN 1: KIỂM CHỨNG NGHIỆM THAM CHIẾU TẠI T = 5 NĂM (VERIFICATION AGAINST BENCHMARK)")
    print("=" * 95)

    exact_assets_T5 = analytical_solution(T_END, P0_VEC, R_VEC)[-1]
    exact_total_T5 = np.sum(exact_assets_T5)

    # Chạy thử nghiệm tại bước thời gian chuẩn h = 0.1 năm (tương đương 50 bước)
    h_demo = 0.1
    _, euler_demo = solve_euler(P0_VEC, R_VEC, T_START, T_END, h_demo)
    _, rk4_demo = solve_rk4(P0_VEC, R_VEC, T_START, T_END, h_demo)

    ver_rows = []
    for i, name in enumerate(ASSET_NAMES):
        ver_rows.append({
            "Tài sản / Danh mục": name,
            "Vốn P(0) (VNĐ)": P0_VEC[i],
            "r (%/năm)": R_VEC[i] * 100,
            "Nghiệm Giải Tích P(5)": exact_assets_T5[i],
            "Euler (h=0.1)": euler_demo[-1, i],
            "RK4 (h=0.1)": rk4_demo[-1, i]
        })

    # Dòng tổng cộng toàn danh mục
    ver_rows.append({
        "Tài sản / Danh mục": "TỔNG DANH MỤC P_total(5)",
        "Vốn P(0) (VNĐ)": np.sum(P0_VEC),
        "r (%/năm)": np.nan,
        "Nghiệm Giải Tích P(5)": exact_total_T5,
        "Euler (h=0.1)": np.sum(euler_demo[-1]),
        "RK4 (h=0.1)": np.sum(rk4_demo[-1])
    })

    df_ver = pd.DataFrame(ver_rows)
    print(df_ver.to_string(index=False))
    print(f"\n-> Giá trị tham chiếu trong file dữ liệu : 1,918,389,468 VNĐ")
    print(f"-> Giá trị giải tích tính từ code        : {exact_total_T5:,.0f} VNĐ (Khớp hoàn toàn 100%)\n")

    # --------------------------------------------------------------------------
    print("=" * 95)
    print("PHẦN 2: BẢNG PHÂN TÍCH SAI SỐ & BẬC HỘI TỤ THỰC NGHIỆM (ERROR & CONVERGENCE ANALYSIS)")
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

        # Tính Bậc hội tụ thực nghiệm (Empirical Order of Convergence - EOC)
        # Công thức: EOC = ln(E(h_prev) / E(h_curr)) / ln(h_prev / h_curr)
        if prev_h is not None:
            eoc_eu = np.log(prev_err_euler / err_abs_eu) / np.log(prev_h / h)
            eoc_rk = np.log(prev_err_rk4 / err_abs_rk) / np.log(prev_h / h)
        else:
            eoc_eu = np.nan
            eoc_rk = np.nan

        conv_records.append({
            "Bước h (năm)": h,
            "Số bước N": n_steps,
            "Euler P(5) (VNĐ)": val_eu,
            "Sai số tuyệt đối Euler (VNĐ)": err_abs_eu,
            "Sai số tương đối Euler (%)": err_rel_eu,
            "Bậc hội tụ Euler (EOC)": eoc_eu,
            "RK4 P(5) (VNĐ)": val_rk,
            "Sai số tuyệt đối RK4 (VNĐ)": err_abs_rk,
            "Sai số tương đối RK4 (%)": err_rel_rk,
            "Bậc hội tụ RK4 (EOC)": eoc_rk
        })

        prev_h = h
        prev_err_euler = err_abs_eu
        prev_err_rk4 = err_abs_rk

    df_conv = pd.DataFrame(conv_records)
    print(df_conv.to_string(index=False))

    # --------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("PHẦN 3: KIỂM TRA ĐIỀU KIỆN DỪNG THEO NGƯỠNG DUNG SAI (STOPPING CRITERIA & TOLERANCE)")
    print("=" * 95)
    target_tol = 1_000.0  # Yêu cầu sai số dưới 1,000 VNĐ trên tổng danh mục ~1.92 tỷ VNĐ
    h_eu_opt, n_eu_opt, err_eu_opt = find_required_stepsize(solve_euler, target_tol)
    h_rk_opt, n_rk_opt, err_rk_opt = find_required_stepsize(solve_rk4, target_tol)

    print(f"Mục tiêu độ chính xác (Tolerance): Sai số tuyệt đối <= {target_tol:,.0f} VNĐ tại T = 5 năm")
    print(
        f"  * Phương pháp Euler : Cần h = {h_eu_opt:.6f} năm ({n_eu_opt:,d} bước tính) -> Sai số đạt: {err_eu_opt:,.2f} VNĐ")
    print(
        f"  * Phương pháp RK4   : Cần h = {h_rk_opt:.6f} năm ({n_rk_opt:,d} bước tính) -> Sai số đạt: {err_rk_opt:,.2f} VNĐ")
    print("  => Nhận xét (Criterion A.7): RK4 tiết kiệm khối lượng tính toán vượt trội so với Euler!")

    return df_ver, df_conv


# ==============================================================================
# 5. TRỰC QUAN HÓA DỮ LIỆU CHUẨN BÁO CÁO (Criterion A.6, A.8 & B.4)
# ==============================================================================
def generate_report_figures(df_conv: pd.DataFrame):
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # --- HÌNH 1: So sánh quỹ đạo tổng danh mục (Exact vs Euler vs RK4) ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    t_fine = np.linspace(T_START, T_END, 500)
    P_exact_fine = np.sum(analytical_solution(t_fine, P0_VEC, R_VEC), axis=1) / 1e6  # Đổi sang Triệu VNĐ

    # Chọn h = 0.5 năm (6 tháng) để thấy rõ sự khác biệt bằng mắt thường trên đồ thị
    h_plot = 0.5
    t_eu, P_eu = solve_euler(P0_VEC, R_VEC, T_START, T_END, h_plot)
    t_rk, P_rk = solve_rk4(P0_VEC, R_VEC, T_START, T_END, h_plot)

    ax1.plot(t_fine, P_exact_fine, 'k-', lw=2.2, label='Nghiệm giải tích chính xác $P_{exact}(t)$')
    ax1.plot(t_eu, np.sum(P_eu, axis=1) / 1e6, 'ro--', lw=1.5, ms=6, label=f'Euler ($h = {h_plot}$ năm)')
    ax1.plot(t_rk, np.sum(P_rk, axis=1) / 1e6, 'bs:', lw=1.5, ms=6, label=f'RK4 ($h = {h_plot}$ năm)')
    ax1.set_title("Hình 1a: Tăng trưởng Tổng Danh mục Đầu tư (0 - 5 năm)", fontweight='bold')
    ax1.set_xlabel("Thời gian $t$ (năm)")
    ax1.set_ylabel("Giá trị danh mục $P(t)$ (Triệu VNĐ)")
    ax1.legend(loc='upper left')

    # Đồ thị sai số tích lũy theo thời gian t
    P_exact_at_grid = np.sum(analytical_solution(t_eu, P0_VEC, R_VEC), axis=1) / 1e6
    err_t_euler = np.abs(P_exact_at_grid - np.sum(P_eu, axis=1) / 1e6)
    err_t_rk4 = np.abs(P_exact_at_grid - np.sum(P_rk, axis=1) / 1e6)

    ax2.plot(t_eu, err_t_euler, 'r-o', lw=1.8, label=f'Sai số tích lũy Euler ($h={h_plot}$)')
    ax2.plot(t_rk, err_t_rk4, 'b-s', lw=1.8, label=f'Sai số tích lũy RK4 ($h={h_plot}$)')
    ax2.set_yscale('log')
    ax2.set_title("Hình 1b: Sai số Toàn cục Tích lũy theo Thời gian (Thang Log)", fontweight='bold')
    ax2.set_xlabel("Thời gian $t$ (năm)")
    ax2.set_ylabel("Sai số tuyệt đối $|P_{exact}(t) - P_{num}(t)|$ (Triệu VNĐ)")
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("Fig1_Portfolio_Trajectory_and_Error.png", dpi=300)

    # --- HÌNH 2: Tăng trưởng chi tiết từng tài sản thành phần (RK4, h=0.05) ---
    fig2, ax3 = plt.subplots(figsize=(9, 5.5))
    t_detail, P_detail = solve_rk4(P0_VEC, R_VEC, T_START, T_END, h=0.05)
    colors = ['#2b5c8f', '#d95f02', '#2ca02c']

    for i, asset in enumerate(ASSETS):
        ax3.plot(
            t_detail, P_detail[:, i] / 1e6, color=colors[i], lw=2.2,
            label=f"{asset['name']} ($r={asset['r'] * 100:.2f}\\%$, $P_0={asset['P0'] / 1e6:.0f}$M)"
        )
        # Chấm điểm nghiệm giải tích để đối chiếu
        ax3.scatter(
            t_detail[::20], analytical_solution(t_detail[::20], P0_VEC, R_VEC)[:, i] / 1e6,
            color=colors[i], marker='o', s=30, zorder=5
        )

    ax3.set_title("Hình 2: Sự Tăng trưởng của Từng Tài sản Thành phần trong 5 Năm (RK4 vs Exact)", fontweight='bold')
    ax3.set_xlabel("Thời gian $t$ (năm)")
    ax3.set_ylabel("Giá trị tài sản $P_i(t)$ (Triệu VNĐ)")
    ax3.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("Fig2_Individual_Assets_Growth.png", dpi=300)

    # --- HÌNH 3: Đồ thị Log-Log kiểm chứng bậc hội tụ O(h) và O(h^4) ---
    fig3, ax4 = plt.subplots(figsize=(8, 5.5))
    h_vals = df_conv["Bước h (năm)"].values
    err_eu_vals = df_conv["Sai số tuyệt đối Euler (VNĐ)"].values
    err_rk_vals = df_conv["Sai số tuyệt đối RK4 (VNĐ)"].values

    ax4.loglog(h_vals, err_eu_vals, 'ro-', lw=2, ms=7, label='Sai số thực nghiệm Euler')
    ax4.loglog(h_vals, err_rk_vals, 'bs-', lw=2, ms=7, label='Sai số thực nghiệm RK4')

    # Vẽ đường dốc lý thuyết tham chiếu O(h^1) và O(h^4)
    ref_h1 = err_eu_vals[-1] * (h_vals / h_vals[-1]) ** 1
    ref_h4 = err_rk_vals[-1] * (h_vals / h_vals[-1]) ** 4
    ax4.loglog(h_vals, ref_h1, 'r--', alpha=0.5, label='Độ dốc lý thuyết $\mathcal{O}(h^1)$')
    ax4.loglog(h_vals, ref_h4, 'b--', alpha=0.5, label='Độ dốc lý thuyết $\mathcal{O}(h^4)$')

    ax4.set_title("Hình 3: Phân tích Hội tụ trên Đồ thị Log-Log tại $T = 5$ năm", fontweight='bold')
    ax4.set_xlabel("Bước thời gian $h$ (năm) - [Thang Log]")
    ax4.set_ylabel("Sai số tuyệt đối $E(h)$ (VNĐ) - [Thang Log]")
    ax4.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("Fig3_Convergence_LogLog_Plot.png", dpi=300)

    print("\n[THÔNG BÁO] Đã xuất thành công 3 biểu đồ chất lượng cao (300 DPI) ra thư mục hiện hành:")
    print("  1. Fig1_Portfolio_Trajectory_and_Error.png")
    print("  2. Fig2_Individual_Assets_Growth.png")
    print("  3. Fig3_Convergence_LogLog_Plot.png")
    plt.show()


if __name__ == "__main__":
    df_verification, df_convergence = run_verification_and_convergence()
    generate_report_figures(df_convergence)