# Ball-socket Giesekus viscoelastic squeeze film

This repository holds the code for Paper 3 of the thesis series: *Rim-pinned squeeze-film saturation and a pressure–viscosity criterion for full-film operation of the piston–slipper ball joint* (N. C. Dhlamini, W. H. Ho; manuscript in preparation).

It models the conforming spherical squeeze film between the piston ball and the slipper socket of an axial piston pump:
- the film follows a Reynolds equation on the spherical land, with a first-order Giesekus memory term in the mobility;
- the memory viscosity carries its sign;
- Patir–Cheng flow factors and Greenwood–Tripp asperity contact cover mixed lubrication;
- the thermal routes and the pressure–viscosity transform are bracketed.

The companion repositories are:

| Paper | Repository |
|---|---|
| 1, isothermal Giesekus squeeze film (slipper) | [isothermal-squeeze-film-giesekus](https://github.com/nigeldhlamini/isothermal-squeeze-film-giesekus) |
| 2, thermal extension of the Giesekus Reynolds equation | [thermal-extension-giesekus-reynolds-equation](https://github.com/nigeldhlamini/thermal-extension-giesekus-reynolds-equation) |
| 3, ball–socket joint (this repository) | ball-socket-giesekus-ve-film |

## Layout

| Path | Contents |
|---|---|
| `ball_socket/` | The ball–socket model: spherical mesh and Reynolds assembly (`spherical.py`), equilibrium solver (`solver.py`), signed memory viscosity (`memory_sign.py`), mixed lubrication (`mixed.py`), thermal routes (`thermal.py`), campaign configuration and gates, and its own tests (`ball_socket/tests/`). |
| `scripts/` | The runners (`run_R*.py`, `run_E*.py`), figure scripts (`fig_R*.py`), the table generator (`make_paper3_tables.py`) and the macro generator (`ch9_numbers.py`). Every number in the paper is emitted as a macro by the macro generator. |
| `src/` | The Giesekus rheology and shared solver modules that Paper 3 imports, vendored from the shared solver at the bytes Paper 3 ran against. `SRC_PINS.json` records each module's sha256. |
| `tests/` | Tests of the Paper 3 code. |
| `data/` | Digitised literature data used by the pressure–viscosity comparison. |
| `RELEASE_MANIFEST.json` | Every file in this tree with its sha256, and the commit of the working repository it was cut from. |

## Running

```bash
pip install numpy scipy pandas pyarrow matplotlib pytest
python -m pytest
```

## What is not here

**Withheld files.** `QUARANTINE_MANIFEST.json` lists the files withheld from this repository, each with its sha256 and the reason. They are earlier configuration versions and documents that carry industry-track or proprietary material. Some of the campaign runners need them. The tests that depend on them are not included.

**Campaign outputs.** The large parquet artefacts are not in the repository.

## Licence

MIT; see `LICENSE`.
