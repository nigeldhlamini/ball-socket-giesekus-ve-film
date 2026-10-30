"""R-64.3: Paper 3's two generated tables.  No number is typed: every value is read from the
frozen box-E configuration (ball_socket/campaign_frozen_v7.json, 'config') or from the modules
that registered it, and formatted here.

  paper3_table_inputs.tex     one row per sampled axis: symbol, range, units, sampling scale
  paper3_table_constants.tex  constants with value or band and source; a source is a \\cite{key}
                              present in paper3.bib, "derived from ...", or -- where the dossier
                              names a reference paper3.bib does not carry -- a plain description,
                              listed in MISSING for the hand-back (R-64.3: no bib entry is added
                              without full details in hand).

Both outputs pass the barred-token scan before they are written.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
FROZEN = ROOT / "ball_socket" / "campaign_frozen_v7.json"
OUT_DEFAULT = pathlib.Path("Z:/Papers/Paper 3/New")

AXES = {  # name -> (symbol, display units, factor from stored units)
    "omega": (r"$\omega$", r"rad\,s$^{-1}$", 1.0),
    "p_ch": (r"$p_{\mathrm{ch}}$", "bar", 1e-5),
    "c0": (r"$c_0$", r"$\mu$m", 1e6),
    "T_in": (r"$T_{\mathrm{in}}$", r"$^\circ$C", 1.0),
    "eta_0": (r"$\eta_0$", r"mPa\,s", 1e3),
    "lam": (r"$\lambda$", "s", 1.0),
    "eta_s_ratio": (r"$\eta_s/\eta_0$", "--", 1.0),
    "theta_min_deg": (r"$\theta_{\min}$", r"$^\circ$", 1.0),
}
WHAT = {"omega": "shaft speed", "p_ch": "chamber pressure", "c0": "radial clearance (cold)",
        "T_in": "inlet temperature", "eta_0": "zero-shear viscosity", "lam": "relaxation time",
        "eta_s_ratio": "solvent fraction", "theta_min_deg": "relief angle"}


def num(x: float) -> str:
    """Compact typeset number: 3 significant figures, scientific outside [1e-3, 1e4)."""
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e4:
        s = f"{x:.3g}"
        return f"{x:.0f}" if "e" in s else s
    m, e = f"{x:.2e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return rf"\ensuremath{{{m} \times 10^{{{int(e)}}}}}"


def inputs_table(cfg, bibkeys=frozenset()) -> str:
    rows = []
    names = [a["name"] for a in cfg["input_space"]]
    assert set(names) == set(AXES), names
    for a in cfg["input_space"]:
        sym, unit, f = AXES[a["name"]]
        rows.append(f"{sym} & {WHAT[a['name']]} & {num(a['lower'] * f)}--{num(a['upper'] * f)} & {unit} & "
                    f"{'logarithmic' if a['scale'] == 'log' else 'linear'} \\\\")
    # R-66.6.1: the analysed configuration (design inputs, not physical constants)
    g = cfg["geometry"]
    from ball_socket import representative as rep
    sig = rep.sigma_star_band()
    assert "LeeShangZhang2026" in bibkeys and "Oberg2020" in bibkeys
    conf = [rf"$R_b$ & ball radius & {num(g['R_b_m'] * 1e3)} & mm & fixed; cf. 6.475\,mm in \cite{{LeeShangZhang2026}} \\",
            rf"$A_p$ & piston area & {num(g['A_p_m2'] * 1e4)} & cm$^2$ & fixed \\",
            rf"$\theta_{{\max}}$ & outer land angle & {num(g['theta_max_deg'])} & deg & fixed \\",
            # R-68.0.1 / R-72.1.3: an input band, like the fluid band; self-contained source text
            rf"$\sigma^{{*}}$ & composite roughness & {num(sig[0] * 1e6)}--{num(sig[1] * 1e6)} & $\mu$m & "
            r"$R_a$ by process \cite{Oberg2020}; $R_q/R_a$: socket 1.1--1.3, ball $1.25 \approx \sqrt{\pi/2}$ "
            r"for a Gaussian surface (derived) \\"]
    return "\n".join([r"\begin{tabular}{llrlp{0.30\textwidth}}", r"\toprule",
                      r"Symbol & Input & Range & Units & Sampling \\", r"\midrule", *rows,
                      r"\midrule", r"\multicolumn{5}{l}{\emph{Analysed configuration}} \\", *conf,
                      r"\bottomrule", r"\end{tabular}", ""])


def constants_table(cfg, bibkeys):
    from ball_socket import representative as rep
    from src.thermal import ThermalProperties
    from ball_socket import e5_fluid_spec as E5
    th = ThermalProperties()
    import bs_thermal_props as _W
    Wm = _W.MINERAL
    g, m, s = cfg["geometry"], cfg["materials"], cfg["summit_statistics_derived"]["operating_band"]
    hb = cfg["hardness_conversion"]["H_ind_band_Pa"]["value"]
    gt = cfg["gt_param_band"]["eta_a_beta_a_sigma_s"]["value"]
    nah = json.loads((ROOT / "results" / "ch9" / "R57_thermal_by_route.report.json").read_text())["nahme"]
    wb = nah["walther_b_per_K_range_over_rows"]
    cor = E5.CORNERS["maximum"]
    missing = []

    def src(key=None, derived=None, plain=None, missing_ref=None):
        if key:
            for k in key.split(","):
                assert k in bibkeys, k
            return rf"\cite{{{key}}}"
        if missing_ref:
            missing.append(missing_ref)
        return derived or plain

    R = [
        (r"$p_{\mathrm{ch}}$, $n$", "rated maximum corner", f"{num(cor['p_ch_bar'])} bar, {num(cor['rpm'])} rpm",
         src(key="RE92004")),
        (r"$E_s$, $\nu_s$, $\alpha_s$", "socket, leaded tin bronze class",
         f"{num(m['socket_E_Pa'] / 1e9)} GPa, {num(m['socket_nu'])}, "
         + num(m['socket_alpha_th']) + r"\,K$^{-1}$",
         src(key="ASM1990v2")),
        (r"$E_b$, $\nu_b$, $\alpha_b$", "ball, bearing steel class",
         f"{num(m['ball_constants']['E'] / 1e9)} GPa, {num(m['ball_constants']['nu'])}, "
         + num(m['ball_constants']['alpha_th']) + r"\,K$^{-1}$",
         src(key="ASM1990v1")),
        (r"$\Delta\alpha$", "differential expansion", num(m['delta_alpha_th']) + r"\,K$^{-1}$",
         src(derived=r"derived from $\alpha_s - \alpha_b$")),
        (r"$H$", "socket indentation hardness", f"{num(hb[0] / 1e6)}--{num(hb[1] / 1e6)} MPa",
         src(key="ASM1990v2") + r" (80--120\,HB); $\times 9.807$ from kgf\,mm$^{-2}$, derived"),
        (r"$n_a\beta_a\sigma_s$", "asperity group (pair)", f"{num(gt[0])}--{num(gt[1])}", src(key="GreenwoodTripp1970")),
        (r"$\sigma_s/\sigma^{*}$", "summit to surface spread",
         f"{num(s['sigma_s_over_sigma_star'][0])}--{num(s['sigma_s_over_sigma_star'][1])}",
         src(key="Nayak1971,McCool1986") + " (derived from the asperity group)"),
        (r"$\phi_x, \phi_s$", "flow factors", "isotropic ($\\gamma = 1$)", src(key="PatirCheng1978,PatirCheng1979")),
        (r"$\alpha_G$", "Giesekus mobility", num(cfg["alpha_fixed"]), src(key="Giesekus1982")),
        (r"$b$", "viscosity--temperature coefficient", f"{num(wb[0])}--{num(wb[1])}\\,K$^{{-1}}$", src(key="ASTMD341")),
        # R-72.1.1: one row per fluid class, the D-8 thermal_props actually used (bs_thermal_props.MINERAL;
        # the water values of src/thermal.py as run for the water-glycol end)
        (r"$\rho$, $c_p$, $k_f$", "mineral oil",
         f"{Wm['rho']:g}\\,kg\\,m$^{{-3}}$, {num(Wm['c_p'])}\\,J\\,kg$^{{-1}}$K$^{{-1}}$, "
         f"{num(Wm['k_f'])}\\,W\\,m$^{{-1}}$K$^{{-1}}$",
         src(key="Totten2011")),
        (r"$\rho c_p$, $k_f$", "water--glycol",
         f"{num(th.rho * th.c_p / 1e6)}\\,MJ\\,m$^{{-3}}$K$^{{-1}}$, {num(th.k_f)}\\,W\\,m$^{{-1}}$K$^{{-1}}$",
         src(plain="water values, as run")),
        (r"$p_{\mathrm{case}}$", "case pressure", f"{num(cfg['p_case_Pa'] / 1e5)} bar", r"\pending{source}"),
    ]
    body = [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in R]
    tex = "\n".join([r"\begin{tabular}{p{0.16\textwidth}p{0.27\textwidth}p{0.25\textwidth}p{0.24\textwidth}}",
                     r"\toprule", r"Symbol & Quantity & Value or band & Source \\", r"\midrule", *body,
                     r"\bottomrule", r"\end{tabular}", ""])
    return tex, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    a = ap.parse_args()
    from ball_socket.provenance import require_clean_tree
    prov = require_clean_tree("R-64.3 tables")
    import barred_token_scan as B
    out = pathlib.Path(a.out)
    cfg = json.loads(FROZEN.read_text(encoding="utf-8"))["config"]
    bib = (out / "paper3_v7.bib").read_text(encoding="utf-8")   # R-72: v7 cites from paper3_v7.bib
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    t1 = inputs_table(cfg, keys)
    t2, missing = constants_table(cfg, keys)
    rx = B.patterns()
    for name, t in (("inputs", t1), ("constants", t2)):
        bad = [(k, line) for line in t.splitlines() for k, r in rx.items() if r.search(line)]
        assert not bad, (name, bad)
    head = f"% GENERATED by scripts/make_paper3_tables.py at {prov['commit'][:12]} (R-64.3, R-72.1).  Do not hand-edit.\n"
    head2 = ("% mineral-oil thermal row: Totten2011, citation_verified: false (page check pending, Nigel)\n"
             "% water-glycol row: the water values of src/thermal.py as run; no citation until confirmed\n")
    (out / "paper3_table_inputs.tex").write_text(head + t1, encoding="utf-8")
    (out / "paper3_table_constants.tex").write_text(head + head2 + t2, encoding="utf-8")
    print("wrote both tables; missing references:")
    for x in missing:
        print("  -", x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
