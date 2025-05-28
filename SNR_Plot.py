import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define the constants (original ones)
g_param = 3.0
ZRT_param = 50 + 0j
S_IT_param = 1e-18
N_na_param = 1e-16
k_param = 1.38e-23
T_param = 300

# 2. Implement the SNR formula (same as before)
def calculate_snr(ZL, g, Z_R, ZRT, S_IT, N_na, k, T):
    sum_Z_R_ZL = Z_R + ZL
    term_ZL_div_sum = ZL / sum_Z_R_ZL
    abs_term_ZL_div_sum_sq = np.abs(term_ZL_div_sum)**2
    abs_ZRT_sq = np.abs(ZRT)**2
    numerator = g**2 * abs_term_ZL_div_sum_sq * abs_ZRT_sq * S_IT
    
    term_ZR_div_sum = Z_R / sum_Z_R_ZL
    abs_term_ZR_div_sum_sq = np.abs(term_ZR_div_sum)**2
    Re_ZL = ZL.real
    noise_component_denominator = g**2 * abs_term_ZR_div_sum_sq * 2 * k * T * Re_ZL
    denominator = N_na + noise_component_denominator
    snr = numerator / denominator
    return snr

# 3. Grid of complex Z_L values (very large space, 150 div)
re_zl_min_v_large = 0.1 
re_zl_max_v_large = 500.0
im_zl_min_v_large = -400.0
im_zl_max_v_large = 400.0
num_points_v_large = 150

re_zl_values_v_large = np.linspace(re_zl_min_v_large, re_zl_max_v_large, num_points_v_large)
im_zl_values_v_large = np.linspace(im_zl_min_v_large, im_zl_max_v_large, num_points_v_large)

Re_ZL_mesh_v_large, Im_ZL_mesh_v_large = np.meshgrid(re_zl_values_v_large, im_zl_values_v_large)
ZL_complex_mesh_v_large = Re_ZL_mesh_v_large + 1j * Im_ZL_mesh_v_large

# --- New Case 1: Z_R = 50 + 200j ---
Z_R_new1 = 50 + 200j
SNR_values_new1 = calculate_snr(ZL_complex_mesh_v_large, g_param, Z_R_new1, ZRT_param, 
                               S_IT_param, N_na_param, k_param, T_param)
Log10_SNR_values_new1 = np.log10(SNR_values_new1)

# 3D Plot for Z_R = 50 + 200j
fig_3d_new1 = plt.figure(figsize=(12, 8))
ax_3d_new1 = fig_3d_new1.add_subplot(111, projection='3d')
surf_3d_new1 = ax_3d_new1.plot_surface(Re_ZL_mesh_v_large, Im_ZL_mesh_v_large, Log10_SNR_values_new1, cmap='viridis', edgecolor='none')
ax_3d_new1.set_xlabel('Re(Z_L) [Ohms]')
ax_3d_new1.set_ylabel('Im(Z_L) [Ohms]')
ax_3d_new1.set_zlabel('log10(SNR)')
ax_3d_new1.set_title(f'3D: log10(SNR) for Z_R = {Z_R_new1} ({num_points_v_large} div)')
fig_3d_new1.colorbar(surf_3d_new1, shrink=0.5, aspect=10, label='log10(SNR)')
plt.show()

# 2D Heatmap for Z_R = 50 + 200j
fig_2d_new1, ax_2d_new1 = plt.subplots(figsize=(10, 8))
im_new1 = ax_2d_new1.imshow(Log10_SNR_values_new1, aspect='auto', origin='lower', 
                            extent=[re_zl_min_v_large, re_zl_max_v_large, im_zl_min_v_large, im_zl_max_v_large],
                            cmap='viridis')
ax_2d_new1.set_xlabel('Re(Z_L) [Ohms]')
ax_2d_new1.set_ylabel('Im(Z_L) [Ohms]')
ax_2d_new1.set_title(f'2D Heatmap: log10(SNR) for Z_R = {Z_R_new1} ({num_points_v_large} div)')
fig_2d_new1.colorbar(im_new1, label='log10(SNR)')
ZL_conj_match_new1 = np.conjugate(Z_R_new1)
if re_zl_min_v_large <= ZL_conj_match_new1.real <= re_zl_max_v_large and \
   im_zl_min_v_large <= ZL_conj_match_new1.imag <= im_zl_max_v_large:
    ax_2d_new1.plot(ZL_conj_match_new1.real, ZL_conj_match_new1.imag, 'ro', markersize=5, label=f'Z_L = Z_R* ({ZL_conj_match_new1.real:.0f}{ZL_conj_match_new1.imag:+.0f}j)')
    ax_2d_new1.legend()
plt.show()

print(f"Summary for Z_R = {Z_R_new1}:")
print(f"  Min log10(SNR): {np.min(Log10_SNR_values_new1):.2f}")
print(f"  Max log10(SNR): {np.max(Log10_SNR_values_new1):.2f}")
if ZL_conj_match_new1.real > 0 : # Ensure positive real part for SNR calculation
    ZL_conj_match_val_new1 = calculate_snr(ZL_conj_match_new1, g_param, Z_R_new1, ZRT_param, S_IT_param, N_na_param, k_param, T_param)
    print(f"  log10(SNR) at Z_L=Z_R* ({ZL_conj_match_new1}): {np.log10(ZL_conj_match_val_new1):.2f}")


# --- New Case 2: Z_R = 200 + 50j ---
Z_R_new2 = 200 + 50j
SNR_values_new2 = calculate_snr(ZL_complex_mesh_v_large, g_param, Z_R_new2, ZRT_param, 
                               S_IT_param, N_na_param, k_param, T_param)
Log10_SNR_values_new2 = np.log10(SNR_values_new2)

# 3D Plot for Z_R = 200 + 50j
fig_3d_new2 = plt.figure(figsize=(12, 8))
ax_3d_new2 = fig_3d_new2.add_subplot(111, projection='3d')
surf_3d_new2 = ax_3d_new2.plot_surface(Re_ZL_mesh_v_large, Im_ZL_mesh_v_large, Log10_SNR_values_new2, cmap='viridis', edgecolor='none')
ax_3d_new2.set_xlabel('Re(Z_L) [Ohms]')
ax_3d_new2.set_ylabel('Im(Z_L) [Ohms]')
ax_3d_new2.set_zlabel('log10(SNR)')
ax_3d_new2.set_title(f'3D: log10(SNR) for Z_R = {Z_R_new2} ({num_points_v_large} div)')
fig_3d_new2.colorbar(surf_3d_new2, shrink=0.5, aspect=10, label='log10(SNR)')
plt.show()

# 2D Heatmap for Z_R = 200 + 50j
fig_2d_new2, ax_2d_new2 = plt.subplots(figsize=(10, 8))
im_new2 = ax_2d_new2.imshow(Log10_SNR_values_new2, aspect='auto', origin='lower',
                            extent=[re_zl_min_v_large, re_zl_max_v_large, im_zl_min_v_large, im_zl_max_v_large],
                            cmap='viridis')
ax_2d_new2.set_xlabel('Re(Z_L) [Ohms]')
ax_2d_new2.set_ylabel('Im(Z_L) [Ohms]')
ax_2d_new2.set_title(f'2D Heatmap: log10(SNR) for Z_R = {Z_R_new2} ({num_points_v_large} div)')
fig_2d_new2.colorbar(im_new2, label='log10(SNR)')
ZL_conj_match_new2 = np.conjugate(Z_R_new2)
if re_zl_min_v_large <= ZL_conj_match_new2.real <= re_zl_max_v_large and \
   im_zl_min_v_large <= ZL_conj_match_new2.imag <= im_zl_max_v_large:
    ax_2d_new2.plot(ZL_conj_match_new2.real, ZL_conj_match_new2.imag, 'ro', markersize=5, label=f'Z_L = Z_R* ({ZL_conj_match_new2.real:.0f}{ZL_conj_match_new2.imag:+.0f}j)')
    ax_2d_new2.legend()
plt.show()

print(f"\nSummary for Z_R = {Z_R_new2}:")
print(f"  Min log10(SNR): {np.min(Log10_SNR_values_new2):.2f}")
print(f"  Max log10(SNR): {np.max(Log10_SNR_values_new2):.2f}")
if ZL_conj_match_new2.real > 0: # Ensure positive real part for SNR calculation
    ZL_conj_match_val_new2 = calculate_snr(ZL_conj_match_new2, g_param, Z_R_new2, ZRT_param, S_IT_param, N_na_param, k_param, T_param)
    print(f"  log10(SNR) at Z_L=Z_R* ({ZL_conj_match_new2}): {np.log10(ZL_conj_match_val_new2):.2f}")

