"""
DSMC-/Prozess-Skalierung fuer Versuch AJ30.

Trennt EXPERIMENT (Vc-invariant) und SIMULATIONS-AUSSCHNITT (der eine Knopf f):

  EXPERIMENT (fest):
    V_mixer_exp  = 1 L
    m_solid_exp  = 200.06 g Pural TH130, Primaerpartikel rund d0 = 34 um
    Pural inhaerent poroes -> Modell-Startporositaet 0, Masse mit rho_eff = rho_Boehmit*0.2
    Q_binder_exp = 106 g Wasser / 60 s
    -> n_real     = N_real / V_mixer_exp        [1/m^3]   Anzahldichte
    -> q_binder   = Q_binder_exp / V_mixer_exp  [1/s]     Binder-Ratendichte

  SIMULATION (Wahl):
    f      = SIM_FRACTION = Vc / V_mixer_exp
    N_comp = FEST (statistische Aufloesung der PSD; W ist nur Mengen-Buchhaltung)

    Vc                  = f * V_mixer_exp
    INITIAL_WEIGHT (W)  = n_real * Vc / N_comp  (= N_real * f / N_comp ; skaliert mit f)
    VOLUMETRIC_FLOW_RATE = q_binder * Vc        (= Q_binder_exp * f)

Aufruf:
    python studies/optimization/dataprep/compute_aj30_scaling.py                 # f = 1.0
    python studies/optimization/dataprep/compute_aj30_scaling.py --fraction 0.1
    python studies/optimization/dataprep/compute_aj30_scaling.py --fraction 0.5 --n-comp 2000
"""

from __future__ import annotations

import argparse
import numpy as np


# --- EXPERIMENT (fest, Vc-invariant) --------------------------------------
SOLID_MASS_KG      = 0.20006
PRIMARY_DIAMETER_M = 34e-6
RHO_BOEHMITE       = 3050.0
POROSITY_FACTOR    = 0.2
V_MIXER_EXP_M3     = 1.0e-3
BINDER_MASS_KG     = 0.106
BINDER_DURATION_S  = 60.0
RHO_WATER          = 1000.0
DROPLET_DIAMETER_M = 100e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", "-f", type=float, default=1.0,
                    help="SIM_FRACTION = Vc / V_mixer_exp  (Default 1.0 = voller Mischer).")
    ap.add_argument("--n-comp", type=int, default=1000,
                    help="Rechen-Partikel (FEST, Default 1000; statistische PSD-Aufloesung).")
    args = ap.parse_args()
    f = float(args.fraction)
    n_comp = int(args.n_comp)
    if not (0.0 < f <= 1.0):
        raise ValueError("--fraction muss in (0, 1] liegen.")

    # --- EXPERIMENT ------------------------------------------------------
    rho_eff = RHO_BOEHMITE * POROSITY_FACTOR
    v_primary = np.pi / 6.0 * PRIMARY_DIAMETER_M**3
    m_primary = rho_eff * v_primary
    n_real_count = SOLID_MASS_KG / m_primary                 # Partikel im vollen Mischer
    n_real_density = n_real_count / V_MIXER_EXP_M3           # 1/m^3   Vc-invariant
    binder_vol_exp = BINDER_MASS_KG / RHO_WATER
    q_binder_exp = binder_vol_exp / BINDER_DURATION_S        # m^3/s   (voller Mischer)
    q_binder_density = q_binder_exp / V_MIXER_EXP_M3         # 1/s     Vc-invariant
    phi_binder = binder_vol_exp / V_MIXER_EXP_M3             # -       Vc-invariant

    # --- SIMULATION (f) ------------------------------------------------
    Vc = f * V_MIXER_EXP_M3
    sum_W = n_real_density * Vc                              # = n_real_count * f
    W_init = sum_W / n_comp                                  # skaliert mit f
    flow_rate = q_binder_density * Vc                        # = q_binder_exp * f
    v_droplet = np.pi / 6.0 * DROPLET_DIAMETER_M**3
    n_droplets = (phi_binder * Vc) / v_droplet
    dw_over_w = 25.0 / W_init                                # AGG_DW_MAX / W  (Abbau-Grobheit)

    print("=" * 68)
    print(f"AJ30  Skalierung    SIM_FRACTION f = {f:g}    N_comp = {n_comp}  (fest)")
    print("=" * 68)
    print("EXPERIMENT (Vc-invariant)")
    print(f"  rho_eff (Boehmit * {POROSITY_FACTOR})        = {rho_eff:11.1f} kg/m^3")
    print(f"  m_primary (d0 = 34 um)             = {m_primary:11.4e} kg")
    print(f"  N_real (voller 1-L-Mischer)        = {n_real_count:11.4e}")
    print(f"  n_real   = N_real / V_mixer        = {n_real_density:11.4e} 1/m^3")
    print(f"  q_binder = Q_exp  / V_mixer        = {q_binder_density:11.4e} 1/s")
    print(f"  phi_binder = V_binder / V_mixer    = {phi_binder:11.4f}")
    print()
    print(f"SIMULATION (f = {f:g})")
    print(f"  Vc        = f * V_mixer_exp        = {Vc:11.4e} m^3")
    print(f"  N_comp                             = {n_comp:11d}   (fest)")
    print(f"  sum(W)   = n_real * Vc            = {sum_W:11.4e}")
    print(f"  INITIAL_WEIGHT = sum(W) / N_comp   = {W_init:11.4e}   (skaliert mit f)")
    print(f"  VOLUMETRIC_FLOW_RATE = q_binder*Vc = {flow_rate:11.4e} m^3/s")
    print(f"  LIQUID_ADDITION_DURATION           = {BINDER_DURATION_S:11.1f} s   (Vc-invariant)")
    print(f"  reale Tropfen im Vc (100 um)       = {n_droplets:11.4e}")
    print(f"  AGG_DW_MAX(25) / W                 = {dw_over_w:11.2e}   (Abbau-Grobheit; waechst ~1/f)")
    print()
    print("Kontrolle (muessen f-unabhaengig sein):")
    print(f"  sum(W) / Vc                        = {sum_W/Vc:11.4e} 1/m^3   (= n_real)")
    print(f"  flow_rate * Dauer / Vc            = {flow_rate*BINDER_DURATION_S/Vc:11.4f}      (= phi_binder)")
    print()
    print("Config-Eintrag:")
    print(f"  SIM_FRACTION = {f:g}")
    print(f"  _N_COMP      = {n_comp}")
    print("  (_VC, _W_INIT, _FLOW_RATE werden daraus in der Config berechnet)")


if __name__ == "__main__":
    main()
