"""D-6 (R-58.1): the vendored sign law is Paper 2's, the src it runs on is asserted, and the
realised sign is right on both sides of Lambda_0."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from ball_socket import memory_sign as MS
from src.rheology import GiesekusFluid

P2 = pathlib.Path(r"C:\Users\Administrator\Documents\giesekus-paper2\scripts")
ALPHAS = (0.1, 0.25, 0.4)
LAMS = [float(v) for v in np.logspace(-2, 4, 25)]


@pytest.mark.skipif(not P2.exists(), reason="Paper 2 tree not on this machine")
def test_vendored_signed_M_equals_paper2_bitwise():
    """Paper 2's function is evaluated in a SEPARATE process (its module mutates sys.path,
    the matplotlib backend, stdout and the global warnings filter at import), and the
    vendored copy must return the same float64 bits on an (alpha, Lambda) grid."""
    code = ("import sys, json; sys.path.insert(0, r'%s');"
            "from paper2_appendixC_asymptotics import signed_M_over_etap as f;"
            "print(json.dumps({str(a): [float(v).hex() for v in f(a, %r)] for a in %r}))"
            % (P2, LAMS, ALPHAS))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(P2.parent), timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    theirs = json.loads(out.stdout.strip().splitlines()[-1])
    for a in ALPHAS:
        ours = [float(v).hex() for v in MS.signed_M_over_etap(a, LAMS)]
        assert ours == theirs[str(a)], a


def test_src_identity_asserted_and_armed(tmp_path, monkeypatch):
    MS.assert_src_identity()                       # this tree, identical to Paper 2's
    fake = tmp_path / "rheology.py"
    fake.write_text("# not the same file\n")
    monkeypatch.setattr(MS, "PAPER2_RHEOLOGY", fake)
    with pytest.raises(ImportError):               # control: a different src must fail
        MS.assert_src_identity()


def test_realised_sign_either_side_of_lambda0():
    """One cell below Lambda_0 = 1.2497 (Lambda = 1.0) and one above (Lambda = 2.0):
    signed M is + below and - above; the modulus is + on both."""
    base = GiesekusFluid(eta_s=0.01, eta_p=0.02, lambda_=1e-3, alpha=0.25)
    s = MS.wrap(base, "signed")
    g = np.array([1.0, 2.0]) / base.lambda_
    mod, sig = base.memory_viscosity(g), s.memory_viscosity(g)
    assert np.all(mod > 0)
    assert sig[0] > 0 and sig[1] < 0
    assert np.allclose(np.abs(sig), mod, rtol=0, atol=0)        # magnitude untouched
    a = MS.signed_M_over_etap(0.25, [1.2496, 1.2498])
    assert a[0] > 0 > a[1]                                       # the crossing is Lambda_0


def test_default_is_the_registered_modulus_path():
    from ball_socket.geometry import BallSocketGeometry
    from ball_socket.solver import BallSocketSolver, CampaignInput
    slv = BallSocketSolver(BallSocketGeometry(c0=5e-6))
    assert slv.memory_sign == "modulus"
    ci = CampaignInput(omega=300, p_ch=2e7, c0=5e-6, T_in=60, eta_0=0.02, lam=1e-3,
                       eta_s_over_eta0=0.5)
    assert type(slv._fluid(ci)) is GiesekusFluid                 # bit-for-bit path
    slv.memory_sign = "signed"
    assert type(slv._fluid(ci)).__name__ == "_SignedMemoryFluid"
    slv.memory_sign = "bogus"
    with pytest.raises(ValueError):
        slv._fluid(ci)
