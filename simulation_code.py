# -------------------------------------------------------------------
# Simulation Code for: Resetting Sensory Precision Enables Escape from Frozen Active Inference States
# Description: 
#   This script reproduces the computational dynamics of Proprioceptive 
#   Prediction Error Neglect (PPEN) and the mechanism of Specific 
#   Informational Perturbation (SIP) as shown in Figure 3.
#   It demonstrates emergent phase transitions using Langevin-like dynamics.
# -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- PNAS Scientific Style Setup ---
sns.set_context("paper", font_scale=1.5)
sns.set_style("ticks")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.linewidth'] = 1.5

def bistable_dynamics(y):
    """
    Derivative of the Double-Well Potential V(y).
    Defines the landscape with two stable attractors:
    1. Low state (y ~ 0.2): Maladaptive / Frozen State
    2. High state (y ~ 2.5): Adaptive / Optimized State
    """
    # The gradient is -dV/dy.
    # Constructed as a cubic function with roots at 0.2, 1.2, and 2.5.
    # y=1.2 acts as the unstable potential barrier.
    gradient = -(y - 0.2) * (y - 1.2) * (y - 2.5)
    return gradient

def run_emergent_simulation():
    T = 20.0
    dt = 0.01
    time = np.arange(0, T, dt)
    n_steps = len(time)

    # 1. Environment (Sine wave + Biological Noise)
    np.random.seed(101) # Seed for reproducibility
    
    # Generate smooth biological drift (Brownian motion)
    drift = np.cumsum(np.random.normal(0, 0.05, n_steps))
    drift = np.convolve(drift, np.ones(50)/50, mode='same')
    target = 1.5 * np.sin(time) + 0.5 * drift

    # 2. SIP Intervention (Implicit trigger)
    # Modeled as a magnitude-dependent impulse, not a hard-coded state switch.
    intervention_idx = int(n_steps * 0.45)
    sip_impulse = 6.0 * np.exp(-50 * ((time - time[intervention_idx])**2))

    sensory_input = target + sip_impulse

    # 3. System Variables
    mu = np.zeros(n_steps)             # Internal Model (Prediction)
    precision = np.zeros(n_steps)      # Real-time Total Precision
    tonic_gain = np.zeros(n_steps)     # Baseline State (System Parameter)
    pe_hist = np.zeros(n_steps)        # Prediction Error History

    # Initial State: Trapped in the "Low" potential well
    current_tonic = 0.2

    # --- Simulation Loop ---
    for t in range(1, n_steps):
        # A. Prediction Error Calculation
        pe = sensory_input[t] - mu[t-1]
        pe_hist[t] = pe

        # B. Phasic Response (Simulating LC-NE Burst)
        # Biological response: Sigmoidal activation based on error magnitude.
        phasic_response = 4.0 / (1 + np.exp(-3.0 * (np.abs(pe) - 1.5)))

        # C. Tonic Plasticity (Phase Transition Logic)
        # CORE MECHANISM: Evolution governed by differential equation.
        # d(tonic)/dt = (Natural Gradient) + (Phasic Forcing)

        # 1. Intrinsic flow (Restoring force towards local attractor)
        intrinsic_flow = bistable_dynamics(current_tonic)

        # 2. External forcing (Phasic burst overcomes the potential barrier)
        # Only strong phasic responses provide enough energy to cross the barrier.
        forcing = 0.8 * phasic_response

        # State Update (Euler Integration)
        d_tonic = intrinsic_flow + forcing
        current_tonic += d_tonic * dt

        # Boundary conditions (prevent numerical divergence)
        current_tonic = np.clip(current_tonic, 0.0, 3.0)
        tonic_gain[t] = current_tonic

        # Total Precision = Tonic (Baseline) + Phasic (Transient)
        precision[t] = current_tonic + phasic_response

        # D. Active Inference (Perceptual Update)
        # Higher precision leads to rapid synchronization (Kalman gain increase).
        k = 1.5 * precision[t]
        d_mu = k * pe
        mu[t] = mu[t-1] + d_mu * dt

    return time, target, mu, precision, tonic_gain, pe_hist

# --- Execution ---
time, target, mu, precision, tonic, pe = run_emergent_simulation()

# --- Visualization ---
fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
colors = ['#333333', '#D55E00', '#0072B2', '#009E73']

# Panel A: Synchronization Dynamics
axes[0].plot(time, target, color='gray', linestyle='--', linewidth=2, alpha=0.6, label='Environment')
axes[0].plot(time, mu, color=colors[2], linewidth=3, label='Internal Model ($\mu$)')
axes[0].set_ylabel("State Magnitude")
axes[0].set_title(r"$\mathbf{A.}$ Emergent Synchronization (Dynamical Locking)", loc='left', fontsize=16)
axes[0].legend(loc='upper left')

# Panel B: Prediction Error Dynamics
axes[1].plot(time, pe, color='k', linewidth=1.5)
axes[1].axhline(y=1.5, color='r', linestyle=':', alpha=0.5, label='Threshold')
axes[1].set_ylabel(r"Prediction Error ($\xi$)")
axes[1].set_title(r"$\mathbf{B.}$ Error Spike Drives State Transition", loc='left', fontsize=16)

# Panel C: Phase Transition (The Mechanism)
axes[2].plot(time, precision, color=colors[3], alpha=0.4, linewidth=1, label='Instantaneous Precision')
axes[2].plot(time, tonic, color=colors[3], linewidth=3, label='System State (Tonic Gain)')
axes[2].fill_between(time, tonic, 0, color=colors[3], alpha=0.1)

# Annotations explaining the Physics
axes[2].text(2, 0.4, "Metastable State 1\n(Low Gain)", ha='center', color='gray', fontsize=12)
axes[2].text(16, 2.7, "Stable State 2\n(High Gain)", ha='center', color=colors[3], fontweight='bold', fontsize=12)
axes[2].annotate('Phase Transition\n(Barrier Crossing)', xy=(9.0, 1.5), xytext=(6, 3.5),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)

axes[2].set_ylabel(r"Precision State ($\gamma$)")
axes[2].set_xlabel("Time (arbitrary units)")
axes[2].set_title(r"$\mathbf{C.}$ Bistable Phase Transition (Non-linear Dynamics)", loc='left', fontsize=16)
axes[2].legend(loc='lower right')

plt.tight_layout()

# Save figure (Optional but recommended for submission)
# plt.savefig("Figure3_Simulation_Dynamics.png", dpi=300)

plt.show()
