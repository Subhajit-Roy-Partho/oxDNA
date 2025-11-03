"""
PyTorch implementation of oxDNA2 force field energies.

This module implements the oxDNA2 coarse-grained model for DNA simulations.
The oxDNA2 model includes the following energy terms:
- FENE backbone potential (bonded)
- Excluded volume interactions (bonded and non-bonded)
- Stacking interactions (bonded)
- Hydrogen bonding (non-bonded, Watson-Crick pairs)
- Cross-stacking (non-bonded)
- Coaxial stacking (non-bonded)
- Debye-Huckel electrostatics (non-bonded)

References:
- oxDNA: T. E. Ouldridge et al., J. Chem. Phys. 134, 085101 (2011)
- oxDNA2: B. E. K. Snodin et al., J. Chem. Phys. 142, 234901 (2015)

Author: Converted from oxDNA C++ implementation to PyTorch
"""

import torch
import torch.nn as nn
import numpy as np
import math

# Define constants (from src/model.h)
PI = math.pi

# Particle geometry - positions of interaction centers
POS_BACK = -0.4  # backbone position
POS_MM_BACK1 = -0.3400  # major-minor groove backbone 1
POS_MM_BACK2 = 0.3408   # major-minor groove backbone 2
POS_STACK = 0.34  # stacking site position
POS_BASE = 0.4    # base position

# FENE (backbone) parameters
FENE_EPS = 2.0
FENE_R0_OXDNA2 = 0.7564
FENE_DELTA = 0.25
FENE_DELTA2 = 0.0625

# Excluded volume parameters
EXCL_EPS = 2.0
EXCL_S1, EXCL_S2, EXCL_S3, EXCL_S4 = 0.70, 0.33, 0.515, 0.515
EXCL_R1, EXCL_R2, EXCL_R3, EXCL_R4 = 0.675, 0.32, 0.50, 0.50
EXCL_B1, EXCL_B2 = 892.016223343, 4119.70450017
EXCL_B3, EXCL_B4 = 1707.30627298, 1707.30627298
EXCL_RC1, EXCL_RC2 = 0.711879214356, 0.335388426126
EXCL_RC3, EXCL_RC4 = 0.52329943261, 0.52329943261

# Hydrogen bonding parameters
HYDR_EPS_OXDNA2 = 1.0678
HYDR_A = 8.0
HYDR_RC = 0.75
HYDR_R0 = 0.4
HYDR_BLOW, HYDR_BHIGH = -126.243, -7.87708
HYDR_RLOW, HYDR_RHIGH = 0.34, 0.7
HYDR_RCLOW, HYDR_RCHIGH = 0.276908, 0.783775

# Hydrogen bonding angular parameters
HYDR_THETA1_A, HYDR_THETA1_B = 1.5, 4.16038
HYDR_THETA1_T0, HYDR_THETA1_TS, HYDR_THETA1_TC = 0.0, 0.7, 0.952381
HYDR_THETA2_A, HYDR_THETA2_B = 1.5, 4.16038
HYDR_THETA2_T0, HYDR_THETA2_TS, HYDR_THETA2_TC = 0.0, 0.7, 0.952381
HYDR_THETA4_A, HYDR_THETA4_B = 0.46, 0.133855
HYDR_THETA4_T0, HYDR_THETA4_TS, HYDR_THETA4_TC = PI, 0.7, 3.10559
HYDR_THETA7_A, HYDR_THETA7_B = 4.0, 17.0526
HYDR_THETA7_T0, HYDR_THETA7_TS, HYDR_THETA7_TC = PI*0.5, 0.45, 0.555556

# Stacking parameters
STCK_BASE_EPS_OXDNA2 = 1.3523
STCK_FACT_EPS_OXDNA2 = 2.6717
STCK_A = 6.0
STCK_RC = 0.9
STCK_R0 = 0.4
STCK_BLOW, STCK_BHIGH = -68.1857, -3.12992
STCK_RLOW, STCK_RHIGH = 0.32, 0.75
STCK_RCLOW, STCK_RCHIGH = 0.23239, 0.956

# Stacking angular parameters (theta)
STCK_THETA4_A, STCK_THETA4_B = 1.3, 6.4381
STCK_THETA4_T0, STCK_THETA4_TS, STCK_THETA4_TC = 0.0, 0.8, 0.961538
STCK_THETA5_A, STCK_THETA5_B = 0.9, 3.89361
STCK_THETA5_T0, STCK_THETA5_TS, STCK_THETA5_TC = 0.0, 0.95, 1.16959

# Stacking angular parameters (phi)
STCK_PHI1_A, STCK_PHI1_B = 2.0, 10.9032
STCK_PHI1_XC, STCK_PHI1_XS = -0.769231, -0.65
STCK_PHI2_A, STCK_PHI2_B = 2.0, 10.9032
STCK_PHI2_XC, STCK_PHI2_XS = -0.769231, -0.65

# Cross-stacking parameters
CRST_R0, CRST_RC, CRST_K = 0.575, 0.675, 47.5
CRST_BLOW, CRST_BHIGH = -0.888889, -0.888889
CRST_RLOW, CRST_RHIGH = 0.495, 0.655
CRST_RCLOW, CRST_RCHIGH = 0.45, 0.7

# Cross-stacking angular parameters
CRST_THETA1_A, CRST_THETA1_B = 2.25, 7.00545
CRST_THETA1_T0, CRST_THETA1_TS, CRST_THETA1_TC = PI - 2.35, 0.58, 0.766284
CRST_THETA2_A, CRST_THETA2_B = 1.70, 6.2469
CRST_THETA2_T0, CRST_THETA2_TS, CRST_THETA2_TC = 1.0, 0.68, 0.865052
CRST_THETA4_A, CRST_THETA4_B = 1.50, 2.59556
CRST_THETA4_T0, CRST_THETA4_TS, CRST_THETA4_TC = 0.0, 0.65, 1.02564
CRST_THETA7_A, CRST_THETA7_B = 1.70, 6.2469
CRST_THETA7_T0, CRST_THETA7_TS, CRST_THETA7_TC = 0.875, 0.68, 0.865052

# Coaxial stacking parameters
CXST_R0, CXST_RC = 0.400, 0.6
CXST_K_OXDNA2 = 58.5
CXST_BLOW, CXST_BHIGH = -2.13158, -2.13158
CXST_RLOW, CXST_RHIGH = 0.22, 0.58
CXST_RCLOW, CXST_RCHIGH = 0.177778, 0.6222222

# Coaxial stacking angular parameters
CXST_THETA1_A, CXST_THETA1_B = 2.0, 10.9032
CXST_THETA1_T0_OXDNA2 = PI - 0.25
CXST_THETA1_TS, CXST_THETA1_TC = 0.65, 0.769231
CXST_THETA1_SA, CXST_THETA1_SB = 20.0, PI - 0.1*(PI - (PI - 0.25))
CXST_THETA4_A, CXST_THETA4_B = 1.3, 6.4381
CXST_THETA4_T0, CXST_THETA4_TS, CXST_THETA4_TC = 0.0, 0.8, 0.961538
CXST_THETA5_A, CXST_THETA5_B = 0.9, 3.89361
CXST_THETA5_T0, CXST_THETA5_TS, CXST_THETA5_TC = 0.0, 0.95, 1.16959

# Coaxial stacking phi parameters
CXST_PHI3_A, CXST_PHI3_B = 2.0, 10.9032
CXST_PHI3_XC, CXST_PHI3_XS = -0.769231, -0.65


class oxDNA2Energy(nn.Module):
    """
    PyTorch module for computing oxDNA2 force field energies.

    The model represents DNA nucleotides as rigid bodies with three interaction sites:
    - Backbone site (phosphate)
    - Stacking site (sugar)
    - Base site (nucleobase)

    Each nucleotide has a position and orientation (represented as a 3x3 rotation matrix).
    """

    def __init__(self, temperature=0.1, salt_concentration=0.5,
                 use_average_seq=True, grooving=True, hb_multiplier=1.0):
        """
        Initialize oxDNA2 energy calculator.

        Args:
            temperature: Temperature in simulation units (T=0.1 corresponds to ~300K)
            salt_concentration: Salt concentration in Molars
            use_average_seq: If True, use sequence-averaged parameters
            grooving: If True, use major-minor groove model
            hb_multiplier: Multiplier for hydrogen bonding strength (for special base types >= 300)
        """
        super(oxDNA2Energy, self).__init__()

        self.T = temperature
        self.salt_concentration = salt_concentration
        self.use_average_seq = use_average_seq
        self.grooving = grooving
        self.hb_multiplier = hb_multiplier

        # Initialize stacking parameters with sequence dependence (oxDNA2 values)
        # F1_EPS[STCK_F1][base_type_i][base_type_j] stores epsilon values
        # F1_SHIFT[STCK_F1][base_type_i][base_type_j] stores shift values
        # Base types: 0=A, 1=C, 2=G, 3=T, 4=dummy
        stck_eps_value = STCK_BASE_EPS_OXDNA2 + STCK_FACT_EPS_OXDNA2 * self.T
        self.stck_eps_matrix = torch.ones((5, 5)) * stck_eps_value
        self.stck_shift_matrix = torch.ones((5, 5)) * stck_eps_value * (1 - torch.exp(torch.tensor(-(STCK_RC - STCK_R0) * STCK_A)))**2

        # Initialize hydrogen bonding parameters with sequence dependence
        # F1_EPS[HYDR_F1][base_type_i][base_type_j] stores epsilon values
        # F1_SHIFT[HYDR_F1][base_type_i][base_type_j] stores shift values
        # Base types: 0=A, 1=C, 2=G, 3=T, 4=dummy
        self.hydr_eps_matrix = torch.ones((5, 5)) * HYDR_EPS_OXDNA2
        self.hydr_shift_matrix = torch.ones((5, 5)) * HYDR_EPS_OXDNA2 * (1 - torch.exp(torch.tensor(-(HYDR_RC - HYDR_R0) * HYDR_A)))**2

        # Note: If sequence-dependent parameters are needed, they should be loaded here
        # For now, using average values (oxDNA2 defaults)
        # In sequence-dependent mode, different values would be used for A-T vs G-C pairs

        # Debye-Huckel parameters
        self.dh_prefactor = 0.0543
        self.dh_lambdafactor = 0.3616455
        self.dh_half_charged_ends = True

        # Calculate Debye length
        lambda_dh = self.dh_lambdafactor * torch.sqrt(torch.tensor(self.T / 0.1)) / torch.sqrt(torch.tensor(self.salt_concentration))
        self.minus_kappa = -1.0 / lambda_dh

        # Debye-Huckel cutoff and smoothing parameters
        self.dh_rhigh = 3.0 * lambda_dh
        x = self.dh_rhigh
        q = self.dh_prefactor
        l = lambda_dh

        self.dh_B = -(torch.exp(-x / l) * q * q * (x + l) * (x + l)) / (-4.0 * x * x * x * l * l * q)
        self.dh_RC = x * (q * x + 3.0 * q * l) / (q * (x + l))

    def _f1(self, r, eps, shift, a, r0, rlow, rhigh, rclow, rchigh, blow, bhigh):
        """
        Modulated Morse potential (used for stacking and hydrogen bonding).

        This function implements the modulated Morse potential with smoothing regions.
        The potential has three regions:
        1. Low cutoff smoothing: r in (rclow, rlow]
        2. Main Morse region: r in (rlow, rhigh)
        3. High cutoff smoothing: r in [rhigh, rchigh)

        Args:
            r: distance tensor
            eps: epsilon (well depth) - can be a scalar or tensor for sequence dependence
            shift: shift value to ensure continuity - can be a scalar or tensor
            a, r0: Morse potential parameters
            rlow, rhigh, rclow, rchigh: cutoff parameters
            blow, bhigh: smoothing parameters

        Returns:
            Energy tensor
        """
        energy = torch.zeros_like(r)

        # Main Morse potential region
        mask_main = (r > rlow) & (r < rhigh)
        if torch.any(mask_main):
            tmp = 1 - torch.exp(-(r[mask_main] - r0) * a)
            energy[mask_main] = eps * tmp**2 - shift

        # Low cutoff smoothing
        mask_low = (r > rclow) & (r <= rlow)
        if torch.any(mask_low):
            energy[mask_low] = eps * blow * (r[mask_low] - rclow)**2

        # High cutoff smoothing
        mask_high = (r >= rhigh) & (r < rchigh)
        if torch.any(mask_high):
            energy[mask_high] = eps * bhigh * (r[mask_high] - rchigh)**2

        return energy

    def _f2(self, r, k, r0, rc, rlow, rhigh, rclow, rchigh, blow, bhigh):
        """
        Modulated harmonic potential (used for cross-stacking and coaxial stacking).

        Args:
            r: distance tensor
            k: spring constant
            r0: equilibrium distance
            rc: cutoff distance
            rlow, rhigh, rclow, rchigh: cutoff parameters
            blow, bhigh: smoothing parameters

        Returns:
            Energy tensor
        """
        energy = torch.zeros_like(r)

        # Main harmonic region
        mask_main = (r > rlow) & (r < rhigh)
        energy[mask_main] = (k / 2.0) * ((r[mask_main] - r0)**2 - (rc - r0)**2)

        # Low cutoff smoothing
        mask_low = (r > rclow) & (r <= rlow)
        energy[mask_low] = k * blow * (r[mask_low] - rclow)**2

        # High cutoff smoothing
        mask_high = (r >= rhigh) & (r < rchigh)
        energy[mask_high] = k * bhigh * (r[mask_high] - rchigh)**2

        return energy

    def _f4(self, theta, a, b, t0, ts, tc):
        """
        Modulated angular potential.

        Args:
            theta: angle tensor (in radians)
            a, b: potential parameters
            t0: equilibrium angle
            ts: smoothing start
            tc: cutoff angle

        Returns:
            Energy tensor
        """
        t = torch.abs(theta - t0)
        energy = torch.zeros_like(t)

        # Main parabolic region
        mask_main = (t < ts)
        energy[mask_main] = 1.0 - a * t[mask_main]**2

        # Smoothing region
        mask_smooth = (t >= ts) & (t < tc)
        energy[mask_smooth] = b * (tc - t[mask_smooth])**2

        return energy


    def _f5(self, cosphi, a, b, xc, xs):
        """
        Modulated dihedral-like potential.

        Args:
            cosphi: cosine of angle
            a, b: potential parameters
            xc, xs: cutoff parameters

        Returns:
            Energy tensor
        """
        energy = torch.zeros_like(cosphi)

        # Only compute if cosphi > xc
        mask_in_range = cosphi > xc

        if torch.any(mask_in_range):
            # Smoothing region: xc < cosphi < xs
            mask_smooth = mask_in_range & (cosphi < xs)
            if torch.any(mask_smooth):
                energy[mask_smooth] = b * (xc - cosphi[mask_smooth])**2

            # Main region: xs <= cosphi < 0
            mask_main = mask_in_range & (cosphi >= xs) & (cosphi < 0)
            if torch.any(mask_main):
                energy[mask_main] = 1.0 - a * cosphi[mask_main]**2

            # Region cosphi >= 0
            mask_high = mask_in_range & (cosphi >= 0)
            if torch.any(mask_high):
                energy[mask_high] = 1.0

        return energy

    def _repulsive_lj(self, r, sigma, rstar, b, rc):
        """
        Smoothed repulsive Lennard-Jones potential (excluded volume).

        Args:
            r: distance tensor
            sigma: LJ sigma parameter
            rstar: transition distance
            b: smoothing parameter
            rc: cutoff distance

        Returns:
            Energy tensor
        """
        energy = torch.zeros_like(r)

        # LJ region (r < rstar)
        mask_lj = r < rstar
        tmp = (sigma**2) / (r[mask_lj]**2)
        lj_part = tmp * tmp * tmp
        energy[mask_lj] = 4.0 * EXCL_EPS * (lj_part**2 - lj_part)

        # Smoothing region (rstar <= r < rc)
        mask_smooth = (r >= rstar) & (r < rc)
        rrc = r[mask_smooth] - rc
        energy[mask_smooth] = EXCL_EPS * b * rrc**2

        return energy

    def backbone_energy(self, r_backbone):
        """
        FENE potential for backbone connectivity.

        This implements the FENE (Finitely Extensible Nonlinear Elastic) potential
        for the backbone connectivity between adjacent nucleotides.

        The energy is:
        E = -(FENE_EPS/2) * log(1 - (r - r0)^2 / DELTA^2)

        where r is the distance between backbone sites, r0 is the equilibrium distance,
        and DELTA is the maximum extension.

        Args:
            r_backbone: backbone-backbone distance tensor (distance between backbone sites)

        Returns:
            Energy tensor
        """
        # Compute deviation from equilibrium
        rbackr0 = r_backbone - FENE_R0_OXDNA2

        # Check FENE range - the bond must satisfy |r - r0| < DELTA
        # If this condition is violated, the log term becomes undefined or positive
        if torch.any(torch.abs(rbackr0) >= FENE_DELTA):
            max_deviation = torch.max(torch.abs(rbackr0)).item()
            print(f"Warning: FENE bond stretched beyond acceptable range!")
            print(f"  Maximum deviation from r0: {max_deviation:.6f} (limit: {FENE_DELTA})")
            # Return a very large energy to indicate bond breakage
            return torch.full_like(r_backbone, 1e12)

        # FENE energy formula
        # The argument of log must be positive: 1 - (rbackr0^2 / DELTA^2) > 0
        # This is guaranteed by the check above
        energy = -(FENE_EPS / 2.0) * torch.log(1.0 - rbackr0**2 / FENE_DELTA2)

        return energy

    def bonded_excluded_volume(self, positions_p, positions_q, orientations_p, orientations_q):
        """
        Excluded volume between bonded neighbors.

        Args:
            positions_p, positions_q: COM positions of particles p and q
            orientations_p, orientations_q: 3x3 orientation matrices

        Returns:
            Total excluded volume energy
        """
        # Compute interaction site positions
        back_p = positions_p + orientations_p[:, :, 0] * POS_BACK
        back_q = positions_q + orientations_q[:, :, 0] * POS_BACK
        base_p = positions_p + orientations_p[:, :, 0] * POS_BASE
        base_q = positions_q + orientations_q[:, :, 0] * POS_BASE

        energy = torch.zeros(positions_p.shape[0])

        # Base-Base interaction
        r_bb = torch.norm(base_q - base_p, dim=1)
        energy += self._repulsive_lj(r_bb, EXCL_S2, EXCL_R2, EXCL_B2, EXCL_RC2)

        # Base(p)-Back(q) interaction
        r_pb_qb = torch.norm(back_q - base_p, dim=1)
        energy += self._repulsive_lj(r_pb_qb, EXCL_S3, EXCL_R3, EXCL_B3, EXCL_RC3)

        # Back(p)-Base(q) interaction
        r_bp_qb = torch.norm(base_q - back_p, dim=1)
        energy += self._repulsive_lj(r_bp_qb, EXCL_S4, EXCL_R4, EXCL_B4, EXCL_RC4)

        return energy

    def stacking_energy(self, positions_p, positions_q, orientations_p, orientations_q,
                       base_type_p, base_type_q):
        """
        Stacking interaction between bonded neighbors (nearest-neighbor base stacking).

        This implements the stacking interaction as described in the oxDNA model.
        The interaction energy depends on:
        - Radial distance between stacking sites (modulated Morse potential)
        - Angular modulations (theta4, theta5, theta6 for orientations; phi1, phi2 for dihedral)
        - Sequence-dependent epsilon values (different for different base pair combinations)

        Base types: 0=A, 1=C, 2=G, 3=T, 4=dummy

        Args:
            positions_p, positions_q: COM positions
            orientations_p, orientations_q: 3x3 orientation matrices
                v1 (first column): points from backbone to base (major groove direction)
                v2 (second column): perpendicular to v1 and v3
                v3 (third column): stacking direction (along helix axis)
            base_type_p, base_type_q: base type indices

        Returns:
            Stacking energy
        """
        # Extract orientation vectors
        a1 = orientations_p[:, :, 0]  # v1 of particle p
        a2 = orientations_p[:, :, 1]  # v2 of particle p
        a3 = orientations_p[:, :, 2]  # v3 of particle p
        b1 = orientations_q[:, :, 0]  # v1 of particle q
        b2 = orientations_q[:, :, 1]  # v2 of particle q
        b3 = orientations_q[:, :, 2]  # v3 of particle q

        # Compute stacking site positions
        stack_p = positions_p + a1 * POS_STACK
        stack_q = positions_q + b1 * POS_STACK

        r_stack = stack_q - stack_p
        r_stackmod = torch.norm(r_stack, dim=1, keepdim=True)
        r_stackdir = r_stack / (r_stackmod + 1e-10)

        # Reference backbone positions (for phi angles)
        rbackref = positions_q + b1 * POS_BACK - positions_p - a1 * POS_BACK
        rbackrefmod = torch.norm(rbackref, dim=1, keepdim=True)

        # Compute angles
        cost4 = torch.sum(a3 * b3, dim=1)  # theta4: angle between a3 and b3
        cost5 = torch.sum(a3 * r_stackdir.squeeze(1), dim=1)  # cost5 (will be negated in f4)
        cost6 = -torch.sum(b3 * r_stackdir.squeeze(1), dim=1)  # cost6
        cosphi1 = torch.sum(a2 * rbackref, dim=1) / (rbackrefmod.squeeze(1) + 1e-10)
        cosphi2 = torch.sum(b2 * rbackref, dim=1) / (rbackrefmod.squeeze(1) + 1e-10)

        # Get sequence-dependent stacking parameters
        # In C++: _f1(rstackmod, STCK_F1, q->type, p->type)
        # Uses F1_EPS[STCK_F1][q->type][p->type] and F1_SHIFT[STCK_F1][q->type][p->type]
        eps_values = self.stck_eps_matrix[base_type_q, base_type_p]
        shift_values = self.stck_shift_matrix[base_type_q, base_type_p]

        # Compute energy components with sequence-dependent radial part
        f1 = self._f1(r_stackmod.squeeze(1), eps_values, shift_values,
                      STCK_A, STCK_R0, STCK_RLOW, STCK_RHIGH,
                      STCK_RCLOW, STCK_RCHIGH, STCK_BLOW, STCK_BHIGH)

        f4_t4 = self._f4(torch.acos(torch.clamp(cost4, -1, 1)),
                         STCK_THETA4_A, STCK_THETA4_B, STCK_THETA4_T0,
                         STCK_THETA4_TS, STCK_THETA4_TC)

        # Note: C++ uses _custom_f4(-cost5, ...) so we negate cost5
        f4_t5 = self._f4(torch.acos(torch.clamp(-cost5, -1, 1)),
                         STCK_THETA5_A, STCK_THETA5_B, STCK_THETA5_T0,
                         STCK_THETA5_TS, STCK_THETA5_TC)

        # Note: C++ uses cost6 = -b3 * rstackdir, which we already computed
        f4_t6 = self._f4(torch.acos(torch.clamp(cost6, -1, 1)),
                         STCK_THETA5_A, STCK_THETA5_B, STCK_THETA5_T0,
                         STCK_THETA5_TS, STCK_THETA5_TC)

        f5_phi1 = self._f5(cosphi1, STCK_PHI1_A, STCK_PHI1_B, STCK_PHI1_XC, STCK_PHI1_XS)
        f5_phi2 = self._f5(cosphi2, STCK_PHI2_A, STCK_PHI2_B, STCK_PHI2_XC, STCK_PHI2_XS)

        energy = f1 * f4_t4 * f4_t5 * f4_t6 * f5_phi1 * f5_phi2

        return energy

    def nonbonded_excluded_volume(self, positions_p, positions_q, orientations_p, orientations_q):
        """
        Excluded volume between non-bonded particles.

        Args:
            positions_p, positions_q: COM positions
            orientations_p, orientations_q: orientation matrices

        Returns:
            Excluded volume energy
        """
        # Compute interaction site positions
        a1 = orientations_p[:, :, 0]
        b1 = orientations_q[:, :, 0]

        back_p = positions_p + a1 * POS_BACK
        back_q = positions_q + b1 * POS_BACK
        base_p = positions_p + a1 * POS_BASE
        base_q = positions_q + b1 * POS_BASE

        energy = torch.zeros(positions_p.shape[0])

        # Base-Base
        r_bb = torch.norm(base_q - base_p, dim=1)
        energy += self._repulsive_lj(r_bb, EXCL_S2, EXCL_R2, EXCL_B2, EXCL_RC2)

        # Back(p)-Base(q)
        r_bp_bq = torch.norm(base_q - back_p, dim=1)
        energy += self._repulsive_lj(r_bp_bq, EXCL_S4, EXCL_R4, EXCL_B4, EXCL_RC4)

        # Base(p)-Back(q)
        r_bq_bp = torch.norm(back_q - base_p, dim=1)
        energy += self._repulsive_lj(r_bq_bp, EXCL_S3, EXCL_R3, EXCL_B3, EXCL_RC3)

        # Back-Back
        r_back = torch.norm(back_q - back_p, dim=1)
        energy += self._repulsive_lj(r_back, EXCL_S1, EXCL_R1, EXCL_B1, EXCL_RC1)

        return energy

    def hydrogen_bonding_energy(self, positions_p, positions_q, orientations_p, orientations_q,
                                base_type_p, base_type_q):
        """
        Hydrogen bonding between Watson-Crick complementary bases.

        This implements the hydrogen bonding interaction as described in the oxDNA model.
        The interaction energy depends on:
        - Radial distance between base sites (modulated Morse potential)
        - Six angular modulations (theta1, theta2, theta3, theta4, theta7, theta8)
        - Sequence-dependent epsilon values (different for A-T vs G-C pairs)
        - Optional hb_multiplier for special base types (btype >= 300)

        Base types: 0=A, 1=C, 2=G, 3=T, 4=dummy
        Watson-Crick pairs: A-T (0+3=3) and G-C (2+1=3)

        Args:
            positions_p, positions_q: COM positions
            orientations_p, orientations_q: orientation matrices
            base_type_p, base_type_q: base type indices

        Returns:
            Hydrogen bonding energy
        """
        # Check if bases are Watson-Crick pairs (btype sum = 3)
        # This works for: A(0) + T(3) = 3, G(2) + C(1) = 3
        is_pair = (base_type_p + base_type_q) == 3

        # Calculate hb_multiplier for special base types (abs(btype) >= 300)
        # For regular bases, this is 1.0
        hb_multi = torch.ones(positions_p.shape[0])
        special_p = torch.abs(base_type_p) >= 300
        special_q = torch.abs(base_type_q) >= 300
        special_both = special_p & special_q
        if torch.any(special_both):
            hb_multi[special_both] = self.hb_multiplier

        # Extract orientation vectors
        a1 = orientations_p[:, :, 0]
        a3 = orientations_p[:, :, 2]
        b1 = orientations_q[:, :, 0]
        b3 = orientations_q[:, :, 2]

        # Base positions
        base_p = positions_p + a1 * POS_BASE
        base_q = positions_q + b1 * POS_BASE

        r_hydro = base_q - base_p
        r_hydromod = torch.norm(r_hydro, dim=1, keepdim=True)
        r_hydrodir = r_hydro / (r_hydromod + 1e-10)

        # Initialize energy
        energy = torch.zeros(positions_p.shape[0])

        # Check distance cutoff and Watson-Crick pairing
        in_range = (r_hydromod.squeeze(1) > HYDR_RCLOW) & (r_hydromod.squeeze(1) < HYDR_RCHIGH) & is_pair

        if torch.any(in_range):
            # Get sequence-dependent epsilon and shift values
            # In C++: F1_EPS[HYDR_F1][q->type][p->type]
            # q->type and p->type are base types (0-4)
            eps_values = self.hydr_eps_matrix[base_type_q[in_range], base_type_p[in_range]]
            shift_values = self.hydr_shift_matrix[base_type_q[in_range], base_type_p[in_range]]
            hb_multi_subset = hb_multi[in_range]

            # Compute angles
            cost1 = -torch.sum(a1 * b1, dim=1)[in_range]  # theta1
            cost2 = -torch.sum(b1[in_range] * r_hydrodir[in_range].squeeze(1), dim=1)  # theta2
            cost3 = torch.sum(a1[in_range] * r_hydrodir[in_range].squeeze(1), dim=1)  # theta3
            cost4 = torch.sum(a3[in_range] * b3[in_range], dim=1)  # theta4
            cost7 = -torch.sum(b3[in_range] * r_hydrodir[in_range].squeeze(1), dim=1)  # theta7
            cost8 = torch.sum(a3[in_range] * r_hydrodir[in_range].squeeze(1), dim=1)  # theta8

            # Radial part with sequence-dependent parameters
            # Apply hb_multiplier: f1 = hb_multi * _f1(...)
            f1 = hb_multi_subset * self._f1(r_hydromod[in_range].squeeze(1), eps_values, shift_values,
                         HYDR_A, HYDR_R0, HYDR_RLOW, HYDR_RHIGH,
                         HYDR_RCLOW, HYDR_RCHIGH, HYDR_BLOW, HYDR_BHIGH)

            # Angular parts
            f4_t1 = self._f4(torch.acos(torch.clamp(cost1, -1, 1)),
                            HYDR_THETA1_A, HYDR_THETA1_B, HYDR_THETA1_T0,
                            HYDR_THETA1_TS, HYDR_THETA1_TC)

            f4_t2 = self._f4(torch.acos(torch.clamp(cost2, -1, 1)),
                            HYDR_THETA2_A, HYDR_THETA2_B, HYDR_THETA2_T0,
                            HYDR_THETA2_TS, HYDR_THETA2_TC)

            # Note: theta3 uses HYDR_THETA1 parameters (same as theta1)
            # This is defined in model.h lines 88-92
            f4_t3 = self._f4(torch.acos(torch.clamp(cost3, -1, 1)),
                            HYDR_THETA1_A, HYDR_THETA1_B, HYDR_THETA1_T0,
                            HYDR_THETA1_TS, HYDR_THETA1_TC)

            f4_t4 = self._f4(torch.acos(torch.clamp(cost4, -1, 1)),
                            HYDR_THETA4_A, HYDR_THETA4_B, HYDR_THETA4_T0,
                            HYDR_THETA4_TS, HYDR_THETA4_TC)

            f4_t7 = self._f4(torch.acos(torch.clamp(cost7, -1, 1)),
                            HYDR_THETA7_A, HYDR_THETA7_B, HYDR_THETA7_T0,
                            HYDR_THETA7_TS, HYDR_THETA7_TC)

            # Note: theta8 uses HYDR_THETA7 parameters (same as theta7)
            # This is defined in model.h lines 106-110
            f4_t8 = self._f4(torch.acos(torch.clamp(cost8, -1, 1)),
                            HYDR_THETA7_A, HYDR_THETA7_B, HYDR_THETA7_T0,
                            HYDR_THETA7_TS, HYDR_THETA7_TC)

            # Total hydrogen bonding energy
            energy[in_range] = f1 * f4_t1 * f4_t2 * f4_t3 * f4_t4 * f4_t7 * f4_t8

        return energy

    def cross_stacking_energy(self, positions_p, positions_q, orientations_p, orientations_q):
        """
        Cross-stacking interaction between non-bonded bases.

        Args:
            positions_p, positions_q: COM positions
            orientations_p, orientations_q: orientation matrices

        Returns:
            Cross-stacking energy
        """
        # Extract orientation vectors
        a1 = orientations_p[:, :, 0]
        a3 = orientations_p[:, :, 2]
        b1 = orientations_q[:, :, 0]
        b3 = orientations_q[:, :, 2]

        # Base positions
        base_p = positions_p + a1 * POS_BASE
        base_q = positions_q + b1 * POS_BASE

        r_cstack = base_q - base_p
        r_cstackmod = torch.norm(r_cstack, dim=1, keepdim=True)
        r_cstackdir = r_cstack / (r_cstackmod + 1e-10)

        energy = torch.zeros(positions_p.shape[0])

        # Check distance cutoff
        in_range = (r_cstackmod.squeeze(1) > CRST_RCLOW) & (r_cstackmod.squeeze(1) < CRST_RCHIGH)

        if torch.any(in_range):
            # Compute angles
            cost1 = -torch.sum(a1[in_range] * b1[in_range], dim=1)
            cost2 = -torch.sum(b1[in_range] * r_cstackdir[in_range].squeeze(1), dim=1)
            cost3 = torch.sum(a1[in_range] * r_cstackdir[in_range].squeeze(1), dim=1)
            cost4 = torch.sum(a3[in_range] * b3[in_range], dim=1)
            cost7 = -torch.sum(b3[in_range] * r_cstackdir[in_range].squeeze(1), dim=1)
            cost8 = torch.sum(a3[in_range] * r_cstackdir[in_range].squeeze(1), dim=1)

            # Radial part
            f2 = self._f2(r_cstackmod[in_range].squeeze(1), CRST_K, CRST_R0, CRST_RC,
                         CRST_RLOW, CRST_RHIGH, CRST_RCLOW, CRST_RCHIGH,
                         CRST_BLOW, CRST_BHIGH)

            # Angular parts (note: some have symmetry, e.g., f4(cost4) + f4(-cost4))
            f4_t1 = self._f4(torch.acos(torch.clamp(cost1, -1, 1)),
                            CRST_THETA1_A, CRST_THETA1_B, CRST_THETA1_T0,
                            CRST_THETA1_TS, CRST_THETA1_TC)

            f4_t2 = self._f4(torch.acos(torch.clamp(cost2, -1, 1)),
                            CRST_THETA2_A, CRST_THETA2_B, CRST_THETA2_T0,
                            CRST_THETA2_TS, CRST_THETA2_TC)

            f4_t3 = self._f4(torch.acos(torch.clamp(cost3, -1, 1)),
                            CRST_THETA2_A, CRST_THETA2_B, CRST_THETA2_T0,
                            CRST_THETA2_TS, CRST_THETA2_TC)

            theta4 = torch.acos(torch.clamp(cost4, -1, 1))
            f4_t4 = self._f4(theta4, CRST_THETA4_A, CRST_THETA4_B, CRST_THETA4_T0,
                            CRST_THETA4_TS, CRST_THETA4_TC) + \
                    self._f4(-theta4, CRST_THETA4_A, CRST_THETA4_B, CRST_THETA4_T0,
                            CRST_THETA4_TS, CRST_THETA4_TC)

            theta7 = torch.acos(torch.clamp(cost7, -1, 1))
            f4_t7 = self._f4(theta7, CRST_THETA7_A, CRST_THETA7_B, CRST_THETA7_T0,
                            CRST_THETA7_TS, CRST_THETA7_TC) + \
                    self._f4(-theta7, CRST_THETA7_A, CRST_THETA7_B, CRST_THETA7_T0,
                            CRST_THETA7_TS, CRST_THETA7_TC)

            theta8 = torch.acos(torch.clamp(cost8, -1, 1))
            f4_t8 = self._f4(theta8, CRST_THETA7_A, CRST_THETA7_B, CRST_THETA7_T0,
                            CRST_THETA7_TS, CRST_THETA7_TC) + \
                    self._f4(-theta8, CRST_THETA7_A, CRST_THETA7_B, CRST_THETA7_T0,
                            CRST_THETA7_TS, CRST_THETA7_TC)

            energy[in_range] = f2 * f4_t1 * f4_t2 * f4_t3 * f4_t4 * f4_t7 * f4_t8

        return energy

    def coaxial_stacking_energy(self, positions_p, positions_q, orientations_p, orientations_q):
        """
        Coaxial stacking interaction (stacking between non-bonded neighbors).

        Args:
            positions_p, positions_q: COM positions
            orientations_p, orientations_q: orientation matrices

        Returns:
            Coaxial stacking energy
        """
        # Extract orientation vectors
        a1 = orientations_p[:, :, 0]
        a3 = orientations_p[:, :, 2]
        b1 = orientations_q[:, :, 0]
        b3 = orientations_q[:, :, 2]

        # Stack positions
        stack_p = positions_p + a1 * POS_STACK
        stack_q = positions_q + b1 * POS_STACK

        r_stack = stack_q - stack_p
        r_stackmod = torch.norm(r_stack, dim=1, keepdim=True)
        r_stackdir = r_stack / (r_stackmod + 1e-10)

        energy = torch.zeros(positions_p.shape[0])

        # Check distance cutoff
        in_range = (r_stackmod.squeeze(1) > CXST_RCLOW) & (r_stackmod.squeeze(1) < CXST_RCHIGH)

        if torch.any(in_range):
            # Compute angles
            cost1 = -torch.sum(a1[in_range] * b1[in_range], dim=1)
            cost4 = torch.sum(a3[in_range] * b3[in_range], dim=1)
            cost5 = torch.sum(a3[in_range] * r_stackdir[in_range].squeeze(1), dim=1)
            cost6 = -torch.sum(b3[in_range] * r_stackdir[in_range].squeeze(1), dim=1)

            # Radial part
            f2 = self._f2(r_stackmod[in_range].squeeze(1), CXST_K_OXDNA2, CXST_R0, CXST_RC,
                         CXST_RLOW, CXST_RHIGH, CXST_RCLOW, CXST_RCHIGH,
                         CXST_BLOW, CXST_BHIGH)

            # Angular parts with special handling for theta1 (oxDNA2 has additional harmonic term)
            theta1 = torch.acos(torch.clamp(cost1, -1, 1))
            t_sb = CXST_THETA1_SB
            f4_t1 = self._f4(theta1, CXST_THETA1_A, CXST_THETA1_B, CXST_THETA1_T0_OXDNA2,
                            CXST_THETA1_TS, CXST_THETA1_TC)
            # Add harmonic part for oxDNA2
            t_minus_sb = theta1 - t_sb
            harmonic_part = torch.where(t_minus_sb > 0, CXST_THETA1_SA * t_minus_sb**2,
                                       torch.zeros_like(t_minus_sb))
            f4_t1 = f4_t1 + harmonic_part

            f4_t4 = self._f4(torch.acos(torch.clamp(cost4, -1, 1)),
                            CXST_THETA4_A, CXST_THETA4_B, CXST_THETA4_T0,
                            CXST_THETA4_TS, CXST_THETA4_TC)

            theta5 = torch.acos(torch.clamp(cost5, -1, 1))
            f4_t5 = self._f4(theta5, CXST_THETA5_A, CXST_THETA5_B, CXST_THETA5_T0,
                            CXST_THETA5_TS, CXST_THETA5_TC) + \
                    self._f4(-theta5, CXST_THETA5_A, CXST_THETA5_B, CXST_THETA5_T0,
                            CXST_THETA5_TS, CXST_THETA5_TC)

            theta6 = torch.acos(torch.clamp(cost6, -1, 1))
            f4_t6 = self._f4(theta6, CXST_THETA5_A, CXST_THETA5_B, CXST_THETA5_T0,
                            CXST_THETA5_TS, CXST_THETA5_TC) + \
                    self._f4(-theta6, CXST_THETA5_A, CXST_THETA5_B, CXST_THETA5_T0,
                            CXST_THETA5_TS, CXST_THETA5_TC)

            energy[in_range] = f2 * f4_t1 * f4_t4 * f4_t5 * f4_t6

        return energy

    def debye_huckel_energy(self, positions_p, positions_q, orientations_p, orientations_q,
                           is_bonded, is_terminus_p, is_terminus_q):
        """
        Debye-Huckel electrostatic potential (salt-dependent).

        Args:
            positions_p, positions_q: COM positions
            orientations_p, orientations_q: orientation matrices
            is_bonded: boolean tensor indicating bonded pairs
            is_terminus_p, is_terminus_q: boolean tensors for chain termini

        Returns:
            Debye-Huckel energy
        """
        energy = torch.zeros(positions_p.shape[0])

        # Skip bonded pairs
        non_bonded = ~is_bonded

        if torch.any(non_bonded):
            # Backbone positions
            a1 = orientations_p[:, :, 0]
            b1 = orientations_q[:, :, 0]
            back_p = positions_p + a1 * POS_BACK
            back_q = positions_q + b1 * POS_BACK

            r_back = back_q - back_p
            r_backmod = torch.norm(r_back[non_bonded], dim=1)

            # Charge factor (half charge for terminus nucleotides)
            cut_factor = torch.ones(positions_p.shape[0])
            if self.dh_half_charged_ends:
                cut_factor[is_terminus_p] *= 0.5
                cut_factor[is_terminus_q] *= 0.5

            # Check distance cutoff
            in_range = (r_backmod < self.dh_RC) & non_bonded[non_bonded]

            if torch.any(in_range):
                r = r_backmod[in_range]

                # Debye-Huckel potential
                mask_low = r < self.dh_rhigh
                mask_high = (r >= self.dh_rhigh) & (r < self.dh_RC)

                e = torch.zeros_like(r)
                # Main exponential region
                e[mask_low] = torch.exp(r[mask_low] * self.minus_kappa) * (self.dh_prefactor / r[mask_low])
                # Smoothing region
                e[mask_high] = self.dh_B * (r[mask_high] - self.dh_RC)**2

                # Apply charge factor
                idx = torch.where(non_bonded)[0][torch.where(in_range)[0]]
                energy[idx] = e * cut_factor[idx]

        return energy

    def forward(self, positions_p, positions_q, orientations_p, orientations_q,
               is_bonded, base_type_p, base_type_q,
               is_terminus_p=None, is_terminus_q=None):
        """
        Compute total interaction energy between particles p and q.

        Args:
            positions_p, positions_q: (N, 3) tensors of COM positions
            orientations_p, orientations_q: (N, 3, 3) orientation matrices
            is_bonded: (N,) boolean tensor indicating bonded pairs
            base_type_p, base_type_q: (N,) int tensors with base types (0=A, 1=C, 2=G, 3=T)
            is_terminus_p, is_terminus_q: (N,) boolean tensors for terminus nucleotides

        Returns:
            total_energy: (N,) tensor of pairwise energies
        """
        if is_terminus_p is None:
            is_terminus_p = torch.zeros(positions_p.shape[0], dtype=torch.bool)
        if is_terminus_q is None:
            is_terminus_q = torch.zeros(positions_q.shape[0], dtype=torch.bool)

        total_energy = torch.zeros(positions_p.shape[0])

        # Bonded interactions
        if torch.any(is_bonded):
            # Backbone (FENE)
            a1_p = orientations_p[is_bonded, :, 0]
            b1_q = orientations_q[is_bonded, :, 0]
            back_p = positions_p[is_bonded] + a1_p * POS_BACK
            back_q = positions_q[is_bonded] + b1_q * POS_BACK
            r_backbone = torch.norm(back_q - back_p, dim=1)
            total_energy[is_bonded] += self.backbone_energy(r_backbone)

            # Bonded excluded volume
            total_energy[is_bonded] += self.bonded_excluded_volume(
                positions_p[is_bonded], positions_q[is_bonded],
                orientations_p[is_bonded], orientations_q[is_bonded])

            # Stacking
            total_energy[is_bonded] += self.stacking_energy(
                positions_p[is_bonded], positions_q[is_bonded],
                orientations_p[is_bonded], orientations_q[is_bonded],
                base_type_p[is_bonded], base_type_q[is_bonded])

        # Non-bonded interactions
        if torch.any(~is_bonded):
            # Excluded volume
            total_energy[~is_bonded] += self.nonbonded_excluded_volume(
                positions_p[~is_bonded], positions_q[~is_bonded],
                orientations_p[~is_bonded], orientations_q[~is_bonded])

            # Hydrogen bonding
            total_energy[~is_bonded] += self.hydrogen_bonding_energy(
                positions_p[~is_bonded], positions_q[~is_bonded],
                orientations_p[~is_bonded], orientations_q[~is_bonded],
                base_type_p[~is_bonded], base_type_q[~is_bonded])

            # Cross-stacking
            total_energy[~is_bonded] += self.cross_stacking_energy(
                positions_p[~is_bonded], positions_q[~is_bonded],
                orientations_p[~is_bonded], orientations_q[~is_bonded])

            # Coaxial stacking
            total_energy[~is_bonded] += self.coaxial_stacking_energy(
                positions_p[~is_bonded], positions_q[~is_bonded],
                orientations_p[~is_bonded], orientations_q[~is_bonded])

        # Debye-Huckel (all non-bonded pairs)
        total_energy += self.debye_huckel_energy(
            positions_p, positions_q, orientations_p, orientations_q,
            is_bonded, is_terminus_p, is_terminus_q)

        return total_energy


# Example usage
if __name__ == "__main__":
    # Initialize the energy model
    model = oxDNA2Energy(temperature=0.1, salt_concentration=0.5)

    # Example: two nucleotides
    batch_size = 2

    # Positions (COM)
    positions_p = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions_q = torch.tensor([[0.8, 0.0, 0.0], [2.0, 0.0, 0.0]])

    # Orientations (identity matrices for simplicity)
    orientations_p = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    orientations_q = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)

    # Base types (A and T are complementary)
    base_type_p = torch.tensor([0, 0])  # A
    base_type_q = torch.tensor([3, 3])  # T

    # First pair is bonded, second is not
    is_bonded = torch.tensor([True, False])

    # Compute energies
    energies = model(positions_p, positions_q, orientations_p, orientations_q,
                     is_bonded, base_type_p, base_type_q)

    print("Pairwise energies:", energies)
    print("Total energy:", energies.sum())
