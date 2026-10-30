# Pre-registration — FROZEN

> ## ✎ REGISTERED — 2026-07-27
>
> **Signed by:** Nigel C. Dhlamini.
> **Registered content:** the text of `preregistration_FINAL_DRAFT_v3.md` as
> signed, retained unaltered alongside this file.
>
> | Artefact | SHA256 |
> |---|---|
> | Registered pre-registration content (`preregistration_FINAL_DRAFT_v3.md`) | `b413a5d1990185510991ef46e27c23a8fc88638226ab5e53c9da5ff981710e9d` |
> | Frozen campaign config (`ball_socket/campaign_frozen.json`) | `2468949e2855c45263b112482e55d3c14af551e59152bd8f84d04573703a41f6` |
>
> The pre-registration hash is taken over the **v3 file as signed**, not over
> this file, so that stamping the hashes cannot alter the value being stamped.
> Verify with `sha256sum preregistration_FINAL_DRAFT_v3.md`.
>
> **THIS DOCUMENT IS IMMUTABLE.** Any change to a registered expectation,
> threshold, convention or STOP condition requires a **new dated ledger entry**
> stating what changed, why, and that it post-dates the freeze. Amendments are
> permitted; silent edits are not. Recorded as ledger entry **L-7**
> (Chapter 6, `sec:falsification_ledger`).

**Chapter 9 ball-socket campaign.** Supersedes `preregistration_FINAL_DRAFT_v2.md`
(retained). Incorporates rulings **R1** (memory-operator mobility form),
**R2** (swept ν₁₀₀ E5) and **R3** (RE 92004 edition deferred to proof stage) of
2026-07-27, on top of the material-track and annular-domain revisions and the
A1–A6 flag rulings.

> Companions: `ball_socket/campaign_config.py` (`DRAFT = False`),
> `ball_socket/campaign_frozen.json`, `ball_socket/e5_fluid_spec.py`,
> `ball_socket/verification/`.
>
> **Reporting format is pre-registered too:** `ch9_reporting_plan.md` commits the
> prose structure, figure set and table set before any run, so the figure set
> cannot be chosen after the fact. Code's consistency review of that plan against
> the current chapter is in `reports/ch9_reporting_plan_review.md`.

---

## 0. Materials, axes, and comparability conventions

**Materials.** Literature-standard interface: hardened bearing-steel ball
(100Cr6 / AISI 52100 class; E = 210 GPa, ν = 0.30, α_th = 12×10⁻⁶ K⁻¹) on a
leaded tin bronze socket (CuSn12Pb / CuPb10Sn class; E = 100 GPa, ν = 0.34,
α_th = 18×10⁻⁶ K⁻¹, 80–120 HB), every constant **CITED**. Composite modulus
**E\* = 75.9 GPa**, reproducing the chapter's ≈76 GPa and unblocking Phase 5.
Material identity enters through exactly two channels: **α_th**
(Δα_th = 6.0×10⁻⁶ K⁻¹) and **σ\***. Proprietary identity, drawing dimensions
and the partner test programme are industry-report track and parked.

**Pre-freeze axis extension.** Before any sample was drawn: η₀ lower
5.0×10⁻³ → **4.0×10⁻³ Pa·s** (hot mineral oil at rated temperature is genuinely
4–5 mPa·s, so a floor excluding rated operation is mis-set); ω upper
470 → **475 rad/s** (the rated maximum corner is 471.2 rad/s).

**Comparability convention (mandatory).** The ball-socket η₀ axis is a strict
**superset** of the slipper's, so any cross-interface AR comparison is made
**only on the common sub-box η₀ ∈ [5.0×10⁻³, 2.0×10⁻¹] Pa·s**.

## 1. Domain: annular fed film

Annulus θ ∈ [θ_min, θ_max], Dirichlet p(θ_min) = p_ch at the fed-pocket edge
and p(θ_max) = p_case at the mouth. Representative class values:
**θ_max = 73°**, **θ_min ∈ [25°, 36°]**, **R_b = 6.5 mm**, **A_p = 2.0×10⁻⁴ m²**.
The pocket supplies relief F_pocket = p_ch·πR_b²sin²θ_min; the film carries
F_app − F_pocket. At rated pressure the relief is **of order a tenth to a
quarter** of the piston force (11.9 % at 25°, 22.9 % at 36°). The pole lies
outside the production domain; polar regularisation is exercised only in the
full-cap verification benchmarks.

## 2. Framework configuration (slipper-validated, carried unchanged)

Classifier-augmented BO-GPR; **Matérn-5/2 ARD**, **N_init = 18**; EIC with
entropy boundary targeting; **joint-posterior feasibility rule** as the named
safety instrument; **symmetric median-over-5-seeds**; **random-uniform arm at
matched budget, always**; **N_bo = 500** per arm per seed. **Two-tier
reporting**: film-thickness projections expected meaningful, power/leakage
projections pre-committed as expected extractor-limited; the matched-budget
oracle ceiling accompanies every IoU / d_H, **never bare**.

## 3. Modelling conventions (pre-registered)

- **Memory operator — MOBILITY FORM (ruling R1).** The Reynolds equation is
  mass conservation, ∂h/∂t + ∇·**q** = 0, with **q** the total
  depth-integrated flux. The squeeze-memory term is a **flux** contribution,
  **q**_mem = −(λḣ/12)(h²M/η̄²)∇p, so the whole of it — **ḣ included** — sits
  **inside the divergence**:
  **L_mem[p] = div_s( [−λḣ/12 · h²M/η̄²] grad_s p )**.
  Placing ḣ outside the divergence removes the memory contribution from mass
  balance and has no derivation behind it. The two placements **coincide when
  ḣ is spatially uniform**, as in planar parallel-plate squeeze, which is why
  the ambiguity remained latent until the spherical domain where
  ḣ = ė cos θ varies. Recorded as ledger entry **L-6**; Chapter 9's
  Eq. (Lmem_sph) as written is superseded (drop-in T5 prepared).
- **Load:** F_app = k_F_bs · W_sq_ref, W_sq_ref = 3π η₀ ω R_b⁴ / c₀²; the **3π
  prefactor is a convention constant, not physics**. k_F_bs ∈ [0.1, 30] log.
- **Kinematics:** squeeze-primary, ḣ ~ c ω ε; **squeeze_phase = 1.0**;
  ω_b = spin_ratio·ω, **spin_ratio = 1.0**; **α = 0.25**; **p_case = 1 bar**.
- **Thermal:** slipper L1 advective closure with the **Q_min guard**
  (Q_out ≤ Q_min = P_diss/(ρc_p·10 K) ⇒ L2) and **Brinkman**
  Br = η₀V_char²/(k_f·20), **V_char = ω_b R_b sin θ_max**; L2 is the spherical
  gap-averaged FVM with conduction to walls at the **datum T_in**.
- **De_sq = λ|ḣ|/c** (nominal clearance), **not** /h_min.

## 4. Constraints

- **g1:** Λ = h_min/σ\* ≥ 3, with the **wide** finish-class band
  **σ\* ∈ [0.336, 0.524] µm** → h_min floor **[1.01, 1.57] µm**. Binding counts
  reported at **both ends**.
- **g2:** mineral **110 °C**, water-glycol **60 °C** (Totten & De Negri, 2012).
- **g3:** De_sq ≤ 0.2.
- **Fourth outcome:** eccentricity blow-up ⇒ non-converged, never clipped.

## 5. Pre-registered expectations

- **E1** — g3 essentially never binds for HFC-E draws (De_sq ~ 0.02–0.06).
- **E2** — the binding geography **inverts**: **g1 dominates**, with the
  g1-critical region on the **outer land toward θ_max**, near load-reversal
  phases.
- **E3** — the envelope may genuinely bind g2 for water-glycol entries. Either
  outcome reportable; record which.
- **E4** — viscometric correction relatively larger than at the slipper,
  N1/hoop smaller, **squeeze memory the leading viscoelastic correction**.
  Inversion of this ordering is a **STOP**.
- **E5** *(re-based; severity by sweep, ruling R2)* — **at the rated maximum
  corner (500 bar, 4500 min⁻¹ for the 45 cm³ frame), the operating point lies
  outside the full-film feasible set with g1 binding, net of the partial
  hydrostatic relief through the fed pocket.** The result is reported as the
  **E5 verdict as a function of the hot viscosity ν₁₀₀**, identifying the
  **threshold hot viscosity above which E5 is refuted** — i.e. "E5 holds for
  all hot viscosities below X, and the rated corner is full-film only above
  it". The **thick end of the sweep is the severity-determining region**: a
  thinner film makes confirmation easier, so the thin end is the anti-severe
  case. **No single assumed viscosity is load-bearing anywhere.** The nominal
  corner (450 bar, 4300 min⁻¹) is reported alongside **without a registered
  direction** (exploratory). Refutation is a reportable **falsification**.
  - Point set: 2 grades (ISO VG32 at 100 °C, VG46 at 110 °C) × **10 swept
    ν₁₀₀ ∈ [4.5, 9.0] mm²/s** × 4 clearances (2, 5, 15, 30 µm) × 2 corners =
    **160 solves**.
  - η(T) from **ASTM D341 (Walther)** on ISO 3448 ν₄₀ (public, cited) with
    ν₁₀₀ swept. The viscosity index is **not used**: it was never
    load-bearing, and ASTM D2270's tables are paywalled and copyrighted, so
    reproducing them in a published thesis is not acceptable.
    Implementation: `ball_socket/e5_fluid_spec.py`.
  - **Box coverage:** 40 of the 160 points fall below the extended η₀ floor and
    are labelled **extrapolation** of the constructed region. All of them lie at
    the **thin, anti-severe** end; the **entire severity-determining thick arm
    is inside the box**.
- **E6** — the **L2 thermal routing fraction substantially exceeds the
  slipper's** (2,224 reroutes); report the fraction.

## 6. Mandatory sensitivity block (OAT)

32-sample Sobol prefix per value, reporting T_max, the **L2 routing fraction**,
and the **g1/g2 binding counts**.

| Parameter | Values | Rationale |
|---|---|---|
| **spin_ratio** | {0, 0.5, 1.0} | **Mandatory.** Friction-driven spin may lie anywhere in [0, ω]; dissipation ~ V². |
| **θ_max** | {66, 73, 80}° | Socket mouth, class range. |
| **θ_min** | {25, 36}° | Feed-pocket land; sets the relief share. |
| **α_th** | ±10 % | Robustness on the one material channel. |

**STOP for Chat review if the binding geography flips across ANY band.**

## 7. STOP conditions

Any expectation flip (E1–E6) or OAT geography flip; classifier AUC < 0.90 on
the audit set; feasible fraction < 2 % or > 95 % of converged; any BS-V gate
regression after a code change.

## 8. Verification status — all six gates PASS

Re-run in full on the annular production domain after the R1 operator change:

| Gate | Verdict | Key metric |
|---|---|---|
| BS-V1 | **PASS** | order 1.990 vs exact p = A ln tan(θ/2) + B |
| BS-V2 | **PASS** | MMS annular 2.0006, pole benchmark 2.0005 |
| BS-V3 | **PASS** | Newtonian limit exact; GNF Picard converges |
| BS-V4 | **PASS** | fitted 0.4969 vs closed-form prediction 0.4968 → **0.01 %** (tol 5 %); residual exponent 1.992 |
| BS-V5 | **PASS** | UCM/SOF → η_p; ball/slipper M ratio > 10³ |
| BS-V6 | **PASS** | L1 isothermal invariant exact; Q_min + Brinkman routing correct |

Full-cap literature benchmarks: **11/11**. Production-solver gate **BS-G0b
PASS** (commit `da5fdcf`), discharging the P0.2 conditional sign-off.

**Embargo (stands).** The geometric projection factor G is **domain-dependent**
(≈0.53–0.67 across the registered θ ranges), so **no single "geometry-modified
coefficient" belongs in prose**. Chapter 9 will instead carry the closed-form
expression; Chat drafts that text.

## 9. Open before the freeze signature

1. Final axis sign-off / freeze signature.

*Proof-stage items (RE 92004 edition, ASM handbook entry, Ch.9 label macros) are
tracked separately in `reports/proof_stage_checklist.md` and block nothing. No
item waits on the partner-questions list, which is industry-report scope only.*
