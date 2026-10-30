"""campaign_config_v8.py -- Chapter 9 ball-socket campaign, BOX F.

SIGNED 2026-09-17 by Nigel C. Dhlamini (R-39).  Box E (``campaign_config_v7.py``, frozen
2026-09-01, config sha256
``9b24371fcad90db16c4a4e733ee8a77a82b8996728b2c769004959c51ae04c04``) is retained
unchanged and superseded, never edited.

WHY BOX F EXISTS.  Executing box E surfaced four things that change
``config_dict()`` without changing the campaign's question: the registered film
model (Patir-Cheng averaged Reynolds) was never applied; fork F2 was adjudicated
for a term the solver does not contain; the validity partition was extended
along a typed failure kind; and the design chart needs a model-sensitivity band
whose definition must be registered before it is computed.

No campaign is re-run.  See ``preregistration_v8_DRAFT.md``.
"""

from __future__ import annotations

import copy

# Box E, imported wholesale and NOT restated -- the same two hazards box E
# documents apply, and are handled the same way: box E is frozen and hashed (no
# later exports to absorb), and every inherited function that reads module state
# is overridden below or recorded as checked.
from .campaign_config_v7 import *                              # noqa: F401,F403
from . import campaign_config_v7 as _v7
from . import campaign_config_v6 as _v6

BOX_E_CONFIG_SHA256 = (
    "9b24371fcad90db16c4a4e733ee8a77a82b8996728b2c769004959c51ae04c04")


# ---------------------------------------------------------------------------
# The grid criterion's selection at the corrected bracket (R-35.5)
# ---------------------------------------------------------------------------
GRID_SELECTION = {
    "status": "OUTCOME OF BOX D's REGISTERED CRITERION, not a new value",
    "selected": {"Ntheta": 32, "Nphi": 24},
    "evidence": "reports/boxd_grid_audit_corrected_handback.md",
    "criterion_functionals": ["h_min_bnd", "T_max_bnd", "De_sq_bnd"],
    "change_32x24_to_48x36_p95_pct": {"h_min_bnd": 1.447, "T_max_bnd": 0.324,
                                      "De_sq_bnd": 0.091},
    "tolerance_pct": 2.0,
    "reported_as": "observed order p, not raw pairwise percentages (R-35.2)",
    "observed_order": {"h_min_bnd": 3.10, "De_sq_bnd": 1.98, "T_max_bnd": 0.67,
                       "h_min_legacy": 0.87, "P_loss": -0.20},
    "limitations": {
        "P_loss": "not grid-converged at any candidate; E6 inherits this",
        "hotspot_location": "unsatisfied at every pair (4/30); not a converged quantity",
        "T_max_bnd": "sub-first order (p = 0.67); do not assume second order",
    },
    "legacy_h_min_excluded": ("not a criterion functional: E14's term is "
                              "legacy(N) - corrected(N) at fixed grid, which needs "
                              "consistency, not independent convergence (R-35.4)"),
}


# ---------------------------------------------------------------------------
# KEY -- the film model (R-37.2, R-38.1)
# ---------------------------------------------------------------------------
FILM_MODEL = {
    "status": "REGISTERED",
    "film": "smooth-surface Reynolds",
    "contact": ("Greenwood-Tripp asperity pressure added to the load balance "
                "through the same projection quadrature as the film"),
    "averaged_patir_cheng": "NOT registered",
    "realised_as": "Spherical.flow_factors is None",
    "decision_rule": ("R-37.2, fixed BEFORE the measurement: if the shift is small "
                      "against E7's margin, register smooth with the measured "
                      "bound attached and do not re-run"),
    "measured_bound": {
        "evidence": "reports/boxe_E7_averaged_film_handback.md",
        "subsample": "300 stratified no-equilibrium samples, 32x24, at the cap",
        "film_ratio_median": (1.14, 1.21),
        "film_ratio_p95": (1.37, 1.53),
        "film_ratio_max": 1.94,
        "film_ratio_min": 0.872,
        "direction": "averaged carries more on 295 of 300 samples",
        "noeq_balanced_by_averaged_pct": (6.7, 9.0),
        "E7_under_averaged_pct": (41.5, 42.6),
        "E7_smooth_pct": 45.61,
        "conservative": True,
    },
    "quotation": ("Under the averaged (Patir-Cheng) formulation the fraction falls "
                  "to 41.5-42.6 %; the value reported is the conservative one."),
    "model_class_precedent": "Kazama (2008); positioned against by Lee, Shang & Zhang (2026)",
    "out_of_fit_domain": {
        "judged": "LOAD-WEIGHTED, not by cell count (R-38.4)",
        "why_not_cells": ("the out-of-fit strip is sub-cell at 32x24, so refining "
                          "the mesh adds cells below the floor -- a cell count "
                          "would make validity improve as the mesh got worse"),
        "measurement": "results/ch9/analysis_E7_load_weighted_domain.parquet",
        "values": {
            "prediction_refuted": ("the out-of-fit band is NOT load-free in the "
                                   "samples that have one"),
            "affected_samples": {"sigma_lo": "26/300", "sigma_hi": "78/300"},
            "affected_are": ("small clearance (c0 7-12 vs 22-24 um), nearest "
                             "balancing (film/target 0.45-0.54 vs 0.24-0.26), "
                             "largest ratio (1.38 vs 1.13-1.19), most flips"),
            "area_below_floor_median_max_pct": {"lo": (2.2, 7.0), "hi": (2.2, 14.6)},
            "load_from_it_median_p90_max_pct": {"lo": (32, 35, 64), "hi": (32, 54, 82)},
            "squeeze_load_from_it_median_pct": {"lo": 31, "hi": 31},
            "treatment_sensitivity_median_max_pct": {"lo": (4.0, 15.7), "hi": (3.9, 22.8)},
            "flips_differing_between_treatments": {"lo": 2, "hi": 3},
            "resolution_caveat": ("the 32x24 affected set (26/78 of 300) and its "
                                  "bimodal load fraction are a FIRST-RING artefact: "
                                  "the flag records whether the first cell centre "
                                  "lies below the floor.  Resolved (64x48, 96x72) "
                                  "the load fraction falls continuously with rim H "
                                  "and its edge moves toward the fit floor with "
                                  "refinement; the measure is grid-stable only deep "
                                  "in the tail.  Treatment sensitivity converges "
                                  "between grids and is the registered criterion."),
            "resolved_curve_96x72": {
                "evidence": "results/ch9/analysis_E7_mask_curve.parquet",
                "H_rim_0.15_0.18": {"load_from_below_pct": (36, 45), "treatment_sens_pct": (6, 8)},
                "H_rim_0.20_0.23": {"load_from_below_pct": (28, 31), "treatment_sens_pct": (3.5, 4.5)},
                "H_rim_0.24_0.31": {"load_from_below_pct": (17, 24), "treatment_sens_pct": (1.8, 3.1)},
                "H_rim_0.32_0.40": {"load_from_below_pct": (6, 9), "treatment_sens_pct": (0.3, 1.2)},
                "H_rim_ge_0.41": {"load_from_below_pct": (0, 0), "treatment_sens_pct": (0, 0)},
            },
            "grid_study_32_48_64_96": {
                "area_below_pct": (3.9, 3.3, 3.7, 3.2),
                "load_from_it_pct": (53, 40, 42, 44),
                "treatment_sensitivity_pct": (8.3, 7.3, 7.1, 6.7),
                "F_avg_rel_to_96x72_pct": (6.9, 2.5, 1.0, 0.0),
                "verdict": "area- and load-weighted measures are grid-stable"},
            "meaning": ("the squeeze peak sits in the thin band just inside the "
                        "Dirichlet rim; in small-clearance samples that band is "
                        "below the fit and carries ~1/3 of the load (up to 4/5). "
                        "The averaged form has no validated continuation there "
                        "(contact flow factor, Wu & Zhu 1989, not implemented)."),
        },
    },
}


# ---------------------------------------------------------------------------
# KEY -- registered model components, for gate 4 (sec 3.4)
# ---------------------------------------------------------------------------
MODEL_COMPONENTS = {
    "why": ("gate 4 enumerated solver_numerics only, so an unapplied registered "
            "model component was invisible to it (E-15).  R-39.6: enumerate EVERY "
            "registered model component, retrospectively, and assert each is "
            "reachable from the campaign entry -- measured by tracing a real "
            "campaign call, not asserted.  Three components were registered and "
            "never called (the contact path, eps_cap, the Patir-Cheng factors); "
            "each was found by something else asking the code a question."),
    "gate": "ball_socket.verification.preflight.model_components_reachable",
    "trace": ("preflight.trace_campaign_entry over expand_for_solving + "
              "evaluate_x_traversal + summarise, one arm of each provenance"),
    # Every config_dict() key has a role.  An unclassified key FAILS the gate,
    # so a new registration cannot pass by not being listed.
    "key_roles": {
        # model components -- reachability checked
        "geometry": "model_component", "materials": "model_component",
        "load_model": "model_component", "p_case_Pa": "model_component",
        "alpha_fixed": "model_component", "spin_ratio": "model_component",
        "squeeze_phase": "model_component", "clearance_closure": "model_component",
        "boundary_extremum_convention": "model_component",
        "constraint_extremum_registration": "model_component",
        "constraint_extremum_source": "model_component",
        "constraints": "model_component", "fluid_classes": "model_component",
        "sigma_star_band_m": "model_component", "film_model": "model_component",
        "patir_cheng": "model_component", "peklenik_gamma": "model_component",
        "greenwood_tripp": "model_component", "gt_param_band": "model_component",
        "hardness_conversion": "model_component",
        "summit_statistics": "model_component",
        "summit_statistics_derived": "model_component",
        "f5_branch_selection": "model_component",
        "contact_model_traversal": "model_component",
        "undetermined_row_policy": "model_component",
        "outcome_taxonomy": "model_component",
        "validity_partition": "model_component",
        # solver numerics -- gate 4's numeric half
        "solver_numerics": "solver_numeric", "eps_cap_margin": "solver_numeric",
        "grid_selection": "solver_numeric",
        # sampling
        "N_audit": "sampling", "N_bo": "sampling", "N_init": "sampling",
        "N_seeds": "sampling", "audit_seed": "sampling", "input_space": "sampling",
        "fork_subsample": "sampling", "oat_block": "sampling", "oat_n": "sampling",
        # reporting -- consumed by analysis and table generators, not the solve
        "analysis_convention": "reporting", "band_reporting": "reporting",
        "common_subbox_with_slipper": "reporting", "e5_fluid_triplet": "reporting",
        "e7_reporting_rules": "reporting", "escalation_criterion": "reporting",
        "feasibility_unchanged": "reporting", "frame_report_constants": "reporting",
        "hmin_floor_band_m": "reporting", "marginal_band": "reporting",
        "regime_thresholds": "reporting",
        "regime_vocabulary_delivered": "reporting",
        "regime_vocabulary_fine": "reporting", "validity_mask": "reporting",
        # fork records
        "open_forks": "fork_record", "reference_arm": "fork_record",
        "f2_withdrawal": "fork_record",
        # descoped
        "boundary_friction": "descoped", "frame_band": "descoped",
        # registration bookkeeping
        "DRAFT": "registration_meta", "FROZEN_ON": "registration_meta",
        "SIGNED_BY": "registration_meta", "box": "registration_meta",
        "blocking_unverified_at_registration": "registration_meta",
        "unverified_at_registration": "registration_meta",
        "box_d_errata": "registration_meta", "box_e_errata": "registration_meta",
        "campaign_artefacts": "registration_meta",
        "clearance_closure_sources": "registration_meta",
        "frame_report_sources": "registration_meta",
        "hardness_sources": "registration_meta",
        "peklenik_sources": "registration_meta",
        "regime_thresholds_sources": "registration_meta",
        "model_components": "registration_meta",
        "preflight_stop_conditions": "registration_meta",
        "registration_audit_convention": "registration_meta",
        "registration_lifecycle": "registration_meta",
        "reproduction_floor_measured": "registration_meta",
        "reproduction_gate": "registration_meta",
        "retained_control_arm": "registration_meta",
        "supersedes": "registration_meta",
    },
    "components": {
        "geometry": {"config_keys": ["geometry"], "status": "applied",
                     "consumers": ["ball_socket.geometry:BallSocketGeometry.pocket_relief_area",
                                   "ball_socket.spherical:Spherical.__init__"]},
        "materials": {"config_keys": ["materials"], "status": "applied",
                      "consumers": ["ball_socket.geometry:composite_modulus",
                                    "ball_socket.mixed:_E_STAR"]},
        "load_model": {"config_keys": ["load_model", "p_case_Pa"], "status": "applied",
                       "consumers": ["ball_socket.solver:BallSocketSolver._applied_load",
                                     "ball_socket.solver:BallSocketSolver._pocket_relief"]},
        "fluid": {"config_keys": ["alpha_fixed"], "status": "applied",
                  "consumers": ["ball_socket.solver:BallSocketSolver._fluid",
                                "ball_socket.reynolds:_solve_viscoelastic"]},
        "kinematics": {"config_keys": ["spin_ratio", "squeeze_phase"], "status": "applied",
                       "consumers": ["ball_socket.solver:BallSocketSolver._forward",
                                     "ball_socket.spherical:Spherical.gap_rate",
                                     "ball_socket.spherical:Spherical.surface_velocity"]},
        "clearance_closure": {"config_keys": ["clearance_closure"], "status": "applied",
                              "consumers": ["ball_socket.geometry:thermal_clearance",
                                            "ball_socket.solver:BallSocketSolver.solve_equilibrium",
                                            "ball_socket.thermal:solve_l2_spherical"]},
        "boundary_extremum": {"config_keys": ["boundary_extremum_convention",
                                              "constraint_extremum_registration",
                                              "constraint_extremum_source"],
                              "status": "applied",
                              "consumers": ["ball_socket.spherical:Spherical.h_min_boundary",
                                            "ball_socket.spherical:Spherical.gap_rate_max_boundary",
                                            "ball_socket.spherical:Spherical.field_max_boundary",
                                            "ball_socket.campaign:_extremum_triple"]},
        "constraints": {"config_keys": ["constraints", "fluid_classes", "sigma_star_band_m"],
                        "status": "applied",
                        "consumers": ["ball_socket.campaign:_labels"],
                        "value_identity": ("campaign reads ConstraintSet() defaults, CLASSES "
                                           "and BANDS rather than config_dict(); equality with "
                                           "the registered values is asserted by test")},
        "film_model_smooth": {"config_keys": ["film_model"], "status": "applied",
                              "consumers": ["ball_socket.spherical:Spherical._flow_factor_fields",
                                            "ball_socket.reynolds:solve_reynolds"],
                              "realised_column": "realised_film_model"},
        "patir_cheng_averaged": {"config_keys": ["patir_cheng", "peklenik_gamma"],
                                 "status": "not_applied",
                                 "why": "registered in box C; not the film model under box F (sec 3.1)",
                                 "consumers": ["ball_socket.mixed:phi_x", "ball_socket.mixed:phi_y",
                                               "ball_socket.mixed:Phi_s", "ball_socket.mixed:phi_s",
                                               "ball_socket.mixed:h_T_bar",
                                               "ball_socket.mixed:dh_T_bar_dh"]},
        "greenwood_tripp": {"config_keys": ["greenwood_tripp"], "status": "applied",
                            "consumers": ["ball_socket.mixed:F_n",
                                          "ball_socket.mixed:asperity_pressure_elastic",
                                          "ball_socket.mixed:asperity_pressure_plastic",
                                          "ball_socket.mixed:ContactModel.area_fraction"]},
        "gt_param_band": {"config_keys": ["gt_param_band"], "status": "applied",
                          "consumers": ["ball_socket.mixed:registered_contact_models",
                                        "ball_socket.mixed:eta_beta_sigma_band",
                                        "ball_socket.mixed:sigma_s_over_beta_a_band"]},
        "hardness": {"config_keys": ["hardness_conversion"], "status": "applied",
                     "consumers": ["ball_socket.mixed:H_ind_band",
                                   "ball_socket.mixed:plasticity_index"]},
        "summit_statistics": {"config_keys": ["summit_statistics", "summit_statistics_derived"],
                              "status": "applied",
                              "consumers": ["ball_socket.mixed:ContactModel.sigma_summit",
                                            "ball_socket.campaign_config_v7:summit_ratio_for",
                                            "ball_socket.campaign_config_v6:sigma_s_over_sigma_star"]},
        "f5_branch": {"config_keys": ["f5_branch_selection"], "status": "applied",
                      "consumers": ["ball_socket.mixed:ContactModel.psi_regime"]},
        "traversal": {"config_keys": ["contact_model_traversal", "undetermined_row_policy"],
                      "status": "applied",
                      "consumers": ["ball_socket.mixed:expand_for_solving",
                                    "ball_socket.campaign:evaluate_x_traversal",
                                    "ball_socket.mixed:ContactModel.pressure_arm"]},
        "outcome_taxonomy": {"config_keys": ["outcome_taxonomy"], "status": "applied",
                             "consumers": ["ball_socket.campaign:_evaluate_x_inner",
                                           "ball_socket.campaign:_ensure_schema"]},
        "validity_partition": {"config_keys": ["validity_partition"], "status": "applied",
                               "consumers": ["ball_socket.campaign:summarise"]},
    },
    "observed_2026_09_17": ("regime_thresholds has no consumer anywhere in "
                            "ball_socket/ or scripts/; it is reporting vocabulary, "
                            "outside this gate's scope, and is recorded rather than "
                            "silently classified"),
}


# ---------------------------------------------------------------------------
# KEY -- fork F2 withdrawn (R-38.5)
# ---------------------------------------------------------------------------
F2_WITHDRAWAL = {
    "status": "WITHDRAWN -- inapplicable",
    "reason": ("the registered film model contains no flow factors, so the "
               "question does not arise.  Withdrawn, not re-adjudicated: a "
               "question with no referent has no answer to change."),
    "trip_wire": "retired with the fork",
    "preserved_box_d_adjudication": copy.deepcopy(
        _v6.OPEN_FORKS["F2_memory_flow_factor"]),
    "pattern": ("third fork adjudicated into a void: F3 and F5 for a contact "
                "model not yet constructed, F2 for a flow factor never called"),
}


# ---------------------------------------------------------------------------
# KEY -- the design chart's model-sensitivity band (R-38.2)
# ---------------------------------------------------------------------------
MARGINAL_BAND = {
    "status": "REGISTERED BEFORE IT IS COMPUTED",
    "quantities_at_cap": {
        "F_film": "smooth film load, at the clearance the row reached",
        "F_asp": "asperity load (film-independent: the gap is fixed at the cap)",
        "F_target": "F_app - F_pocket",
    },
    "kappa": {"lo": 0.80, "hi": 2.50,
              "requirement": ("VALIDITY REQUIREMENT (R-39.4): kappa must not exclude, "
                              "by construction, any row whose feasibility could change "
                              "under either film model WITHIN THE VALIDATED DOMAIN.  "
                              "Observed there: 0.872 to 1.942; headroom 8 % and 29 %."),
              "not_a_bound": ("as H -> 0 the ratio Phi(H)/phi_x has no finite upper "
                              "limit, so kappa_hi is a sampled maximum with headroom, "
                              "NOT a bound.  Acceptable only because the region where "
                              "it could be exceeded is the VALIDITY_MASK, excluded on "
                              "validity grounds rather than by the band.  Measured "
                              "down to rim H = 0.05; the audit's minimum is 0.038."),
              "encloses_measured": (0.872, 1.942),
              "provenance": "FILM_MODEL.measured_bound, with a margin",
              "caveat": ("the upper edge is bounded only for the two below-floor "
                         "treatments measured; a contact-flow-factor continuation "
                         "is not measured and not claimed")},
    "marginal_iff": "kappa_lo*F_film + F_asp < F_target <= kappa_hi*F_film + F_asp",
    "non_marginal": "keeps its smooth verdict -- proven invariant over kappa, not assumed",
    "count_first": ("the band is counted and reported before any averaged solve; "
                    "converged rows need one smooth cap solve per sample"),
    "cost_gate_rows": 12000,
    "cost_gate_reports": ("R-39.5: if the band exceeds the gate, hand back its SIZE "
                          "and its DISTRIBUTION ACROSS THE DESIGN AXES before "
                          "stopping -- exceeding the gate is itself a finding about "
                          "how much of the design space the model ambiguity touches"),
    "re_solve": ("full equilibrium with thermal closure, averaged film on, at the "
                 "row's sigma* end, under BOTH below-floor treatments"),
    "stop_if": ("any re-solved row's averaged/smooth film ratio at the cap lies "
                "outside [kappa_lo, kappa_hi]: the band definition is unsound; "
                "stop, and do not widen it after the fact"),
    "drawn_as": ("a band between the two film models, not a line -- OUTSIDE the "
                 "VALIDITY_MASK only (R-39.3)"),
}


# ---------------------------------------------------------------------------
# KEY -- the validity mask (R-39.3)
# ---------------------------------------------------------------------------
VALIDITY_MASK = {
    "status": "REGISTERED BEFORE THE CHART IS COMPUTED (amends R-38.2)",
    "why": ("the marginal band sits where the averaged film is not validated: in "
            "small-clearance samples the squeeze peak lies in a thin band just "
            "inside the rim, below Patir-Cheng's fitted range, carrying a large "
            "share of the load.  A contour there would be bounded by a model with "
            "no validated form -- the error this project exists to avoid, in a "
            "figure."),
    "criterion": ("a row is MASKED iff it is MARGINAL (MARGINAL_BAND) AND its "
                  "analytic rim H at the cap is below H_mask"),
    "rim_H": ("H_rim = EPS_CAP_MARGIN * c / sigma*, exactly -- h(theta_min) = "
              "(1 - eps_cap cos theta_min) c = 0.01 c.  Grid-free; agrees with the "
              "solver's boundary value to 2.4e-14.  Evaluated at the row's own "
              "sigma* end; a pure-film sample carries one flag per end."),
    "H_mask": 0.32,
    "H_mask_basis": ("the rim H below which the continuation of the fit (extrapolate "
                     "vs freeze) moves the film load at the cap by more than the "
                     "project's registered 2 % numerical tolerance, read off the "
                     "RESOLVED curve (96x72): sensitivity >= 1.8 % up to H = 0.31, "
                     "<= 1.2 % from 0.32.  Two treatments bound the continuation "
                     "ambiguity from below only."),
    "in_clearance_terms": {"sigma_0.336um": "c < 10.7 um", "sigma_0.524um": "c < 16.8 um"},
    "not_masked": ("a non-marginal row keeps its smooth verdict everywhere, "
                   "including at small clearance: its verdict is invariant over "
                   "kappa, and kappa's measured range reaches rim H = 0.05"),
    "drawn_as": ("hatched, labelled as outside the validated domain of the "
                 "averaged-flow framework (H below Patir-Cheng's fitted range over "
                 "a load-bearing band; no contact flow factor implemented).  No "
                 "contour is drawn through it."),
    "context": ("Lee, Shang & Zhang (2026) begin their clearance study at 7.5 um, "
                "reached with elastic deformation flattening the gap rather than a "
                "rigid film at 0.01 c: the mask is where the two approaches diverge "
                "most and where neither is comfortable."),
}


# ---------------------------------------------------------------------------
# KEY -- the validity partition (R-32.4, extended by R-37.4)
# ---------------------------------------------------------------------------
VALIDITY_PARTITION = {
    "principle": "valid means the solve answered",
    "valid": ["converged", "contact_limit",
              "nonconverged[kind=blowup_film_closed]",
              "nonconverged[kind=blowup_capacity]"],
    "invalid": ["clearance_nonconverged", "error",
                "nonconverged[kind=picard_at_root]"],
    "untyped_nonconverged": "refused unless the caller names the convention",
    "box_c_denominator": "reported alongside, never replaced",
    "E7": {"numerator": 89672, "denominator": 196602, "pct": 45.61},
}


# ---------------------------------------------------------------------------
# Reporting rules for E7 (sec 7)
# ---------------------------------------------------------------------------
E7_REPORTING_RULES = (
    "film bound (FILM_MODEL.quotation)",
    "margin sensitivity: 45.4 -> 34.2 -> 28.8 % as the cap margin goes "
    "0.01 -> 0.003 -> 0.001 (9.1x -> 6.8x -> 5.8x); robust verdict, sensitive number",
    "geography once Phase 4 exists (R-36.2)",
    "rigid-joint qualifier in the stating sentence (R-34.3)",
    "rescue as a fraction ('under one per cent'), never the count 23 (R-36.3)",
    "arm invariance 99.46 %, measured (R-37.5)",
)


# ---------------------------------------------------------------------------
# Errata against box E -- box E is not edited
# ---------------------------------------------------------------------------
BOX_E_ERRATA = {
    "E-13": "eps_cap registered in box D and never applied; box D, box E and the grid audit ran at 0.99",
    "E-14": "the 0.99 grid-audit parquet destroyed by a re-run; remedy is the shared writer",
    "E-15": "averaged film registered in box C and never applied; F2 adjudicated for an absent term",
    "E-16": "NonConvergence carried two events under one outcome; failure rows omitted their clearance",
    "where": "reports/boxe_errata.md",
}

RETAINED_CONTROL_ARM = {
    "arm": "eps_cap = 0.99 (box B's bracket)",
    "artefacts": ["audit_boxD_N8192_seed20260727_48x36.parquet",
                  "audit_boxE_mixed_N8192_seed20260727_48x36.parquet"],
    "why": ("makes 'asperity contact contributes orders less than the deficit' a "
            "measurement rather than the signature of an unwired model (R-33.4, R-36.7)"),
}

CAMPAIGN_ARTEFACTS = {
    "pure_film": {"file": "audit_boxD_purefilm_N8192_seed20260727_32x24_corrected.parquet",
                  "sha256": "b121ab424eb54b41a4c2c7aa53e7e3a7880814ec2ec072d37e933fd38ce4e733"},
    "mixed": {"file": "audit_boxE_mixed_N8192_seed20260727_32x24_corrected.parquet",
              "sha256": "ea4893b571e869ba4ee7a97b5d61ac7ecffcff764886eeac87b7b8f4be3c4235"},
    "re_run": False,
}


DRAFT = False                 # SIGNED by Nigel C. Dhlamini, 2026-09-17 (R-39: "Assume signed")
FROZEN_ON = "2026-09-17"
SIGNED_BY = "Nigel C. Dhlamini"


# ---------------------------------------------------------------------------
# Overrides of inherited state readers (box E's documented hazard)
# ---------------------------------------------------------------------------
def phase_entry_check(phase: str) -> dict:
    """Box F's phase gate, reading BOX F's state (the inherited one reads v7's)."""
    if DRAFT:
        raise RuntimeError(
            f"box F is still DRAFT; phase {phase!r} may not be entered")
    return _v7.phase_entry_check(phase)


_INHERITED_STATE_READERS_CHECKED = {
    "freeze_and_hash": "OVERRIDDEN -- would have re-frozen box E",
    "phase_entry_check": "OVERRIDDEN -- would have read box E's DRAFT",
    "config_dict": "OVERRIDDEN -- box F's keys",
    "assert_box_b_invariants": "inherited: shared subset unchanged",
    "blocking_unverified": "inherited: box F adds no VERIFY constants",
}


def freeze_and_hash(path: str | None = None) -> str:
    """Digest box F's ``config_dict()`` and stamp ``campaign_frozen_v8.json``.

    Overridden for the reason box E documents: an inherited ``freeze_and_hash``
    reads its own module's DRAFT and config and writes its own module's frozen
    artefact.  Guards, all refusing:
      * ``DRAFT`` is True until signed;
      * the load-weighted out-of-domain values must be filled (sec 3.3);
      * box E's frozen hash must still verify before box F supersedes it.
    """
    import hashlib
    import json
    import os

    if DRAFT:
        raise RuntimeError("v8 (box F) is still DRAFT; refusing to freeze")
    if FILM_MODEL["out_of_fit_domain"]["values"] is None:
        raise RuntimeError("refusing to freeze box F: sec 3.3's load-weighted "
                           "values are not filled")
    missing = blocking_unverified()
    if missing:
        raise RuntimeError(f"refusing to freeze box F: still VERIFY: {missing}")

    e_now = hashlib.sha256(json.dumps(_v7.config_dict(), sort_keys=True,
                                      default=str).encode()).hexdigest()
    if e_now != BOX_E_CONFIG_SHA256:
        raise RuntimeError(f"box E now hashes to {e_now}, not "
                           f"{BOX_E_CONFIG_SHA256}; box F's supersession is "
                           f"invalid.  STOP.")
    assert_box_b_invariants()

    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "campaign_frozen_v8.json")
    from .immutable import guarded_path
    guarded_path(path, what="box-F frozen config")

    cfg = config_dict()
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    digest = hashlib.sha256(blob).hexdigest()
    with open(path, "w") as fh:
        json.dump({"sha256": digest, "config": cfg}, fh, indent=2, default=str)
    return digest


def config_dict() -> dict:
    """Box E's registered box, plus box F's keys.  Additive."""
    cfg = _v7.config_dict()
    forks = copy.deepcopy(cfg.get("open_forks", {}))
    if "F2_memory_flow_factor" in forks:
        forks["F2_memory_flow_factor"] = {
            "status": "WITHDRAWN -- inapplicable (box F)",
            "see": "f2_withdrawal, which preserves box D's adjudication verbatim"}
    cfg.update({
        "DRAFT": DRAFT, "box": "F",
        "FROZEN_ON": FROZEN_ON, "SIGNED_BY": SIGNED_BY,
        "open_forks": forks,
        "supersedes": {
            "box": "E",
            "module": "ball_socket/campaign_config_v7.py",
            "registration": "preregistration_v7_FROZEN.md",
            "frozen_config": "ball_socket/campaign_frozen_v7.json",
            "config_sha256": BOX_E_CONFIG_SHA256,
            "retained": "unchanged and superseded, never edited",
            "why": ("the registered film model was never applied, fork F2 was "
                    "adjudicated for an absent term, the validity partition was "
                    "extended, and the design chart's band must be registered "
                    "before it is computed"),
            "prior_chain": _v7.config_dict().get("supersedes"),
        },
        "grid_selection": copy.deepcopy(GRID_SELECTION),
        "film_model": copy.deepcopy(FILM_MODEL),
        "model_components": copy.deepcopy(MODEL_COMPONENTS),
        "f2_withdrawal": copy.deepcopy(F2_WITHDRAWAL),
        "marginal_band": copy.deepcopy(MARGINAL_BAND),
        "validity_mask": copy.deepcopy(VALIDITY_MASK),
        "validity_partition": copy.deepcopy(VALIDITY_PARTITION),
        "e7_reporting_rules": list(E7_REPORTING_RULES),
        "box_e_errata": dict(BOX_E_ERRATA),
        "retained_control_arm": copy.deepcopy(RETAINED_CONTROL_ARM),
        "campaign_artefacts": copy.deepcopy(CAMPAIGN_ARTEFACTS),
    })
    return cfg
