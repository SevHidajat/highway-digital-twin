"""
Novel-view synthesis for evaluation.

Given a trained reconstruction (NeRF or 3DGS), render:
    1. Held-out real camera poses (for fidelity metrics vs. ground truth).
    2. Laterally-shifted "different lane" poses (tests extrapolation --
       a known weak point for both NeRF and 3DGS; characterizing the
       *degradation curve* as lateral offset increases is one of this
       project's more interesting empirical contributions).
    3. (Stretch) Weather/lighting-augmented renders, if a relighting
       module is added -- see LightSim (Waabi, 2023) for reference
       technique on lighting-aware digital twins.
"""
