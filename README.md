# Misfit-line charge and polarization at relaxed nitride interfaces

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This repository holds the code, the inputs and the derived data for the paper
**"The charge of misfit dislocations decides the polarization charge at
relaxed nitride interfaces"** by Tanvir M. Mahim (Department of Electrical
and Electronic Engineering, BRAC University). The paper itself is not in
this repository.

- Repository: https://github.com/Tanvir-Mahmud-Mahim/nitride-misfit-polarization
- Archived on Zenodo, all versions: https://doi.org/10.5281/zenodo.24297966 (version 1.3.1: https://doi.org/10.5281/zenodo.24315036)
- Companion work (the GaN/AlN hole gas): T. M. Mahim, A. S. M. Mohsin and
  M. M. Rahman, Phys. Rev. B 114, L291304 (2026),
  https://doi.org/10.1103/zn44-nmx8; code at
  https://doi.org/10.5281/zenodo.21791358

No experiment was done for this work. Every experimental number is a
published value, and each is listed with its source in `data/`.

## 1. The idea in one minute

Nitride crystals such as GaN, AlN and AlGaN are electrically polarized. Where
two of them meet, the difference in polarization leaves a sheet of fixed
charge, and that charge pulls in the electrons (or holes) that make
transistors work. Computing it needs a reference crystal, and two references
are in use. They agree when the top layer is stretched to fit the layer below
(a coherent interface). They disagree when the top layer has partly relaxed,
and the disagreement can be larger than 10^13 charges per cm^2
(Wang et al., Appl. Phys. Lett. 124, 263502 (2024)).

A layer relaxes by inserting (or removing) extra rows of atoms, which end at
the interface as lines called misfit dislocations. This work shows that the
two references differ only in the charge they silently assume for those
lines: one assumes each line end is a plain cut through the crystal (a large
charge), the other assumes the lines are neutral. The real charge depends on
how the broken bonds at the line are filled. Electron counting says they are
nearly neutral. Published AlGaN/GaN data (Shen et al. 2000, with the
relaxation measured) show lines carrying about a third of the bare-end
charge, with an error of about 60 percent, and a second, independent series
(Hiroki et al. 2002) loses more electrons on relaxation than neutral lines
could remove. The mobility of the relaxed samples points the same way: for
lines in random directions it excludes bare ends, and for the straight
three-direction networks seen in AlGaN/GaN it still favors less than a bare
end, but no longer excludes it (`scripts/run_line_network_check.py`).

The measured charge may be bound to the line cores or carried by electrons
trapped on them. In a layer under tension (AlGaN on GaN) both are negative and
cannot be told apart. In a layer under compression (GaN on AlN) the bound part
changes sign and the trapped part does not, so measuring how the interface
charge of GaN on AlN changes as the GaN relaxes separates the two. The code
gives the expected rates for each case (`scripts/run_prediction.py`). The
test can be made at the same kind of interface as the Shen data: an AlGaN
barrier on a GaN channel thin enough to keep the lattice of AlN is compressed,
as in the series of Elwaradi et al. (2023), whose mobility already disfavors
bare line ends for misfit lines of this opposite sign.

A simpler form of the test needs no polarization constant at all. Every
contribution from polarization, the piezoelectric response and the charge bound
to the cores, changes sign with the mismatch; electrons trapped on the cores do
not. Divide the charge change of a stretched and of a compressed layer of the
same alloy by 2 R |eps_c| and add the two: polarization cancels and only the
trapped charge remains (`scripts/run_prediction.py`, key `odd_even_Al0.30`).

## 2. What is in this repository

| Path | What it is |
|---|---|
| `src/nmp/polarization.py` | Bound charge of coherent and relaxed interfaces; misfit-line density and charge limits |
| `src/nmp/sp1d.py` | One-dimensional Schroedinger-Poisson solver (quantum 2DEG, classical electrons elsewhere, Fermi-level pinning) |
| `src/nmp/lines.py` | Scattering of a 2D gas by charged lines in the interface (same kernel as the PRB paper) |
| `src/nmp/materials.py` | Material parameters, each with its published source |
| `src/nmp/qe.py` | Small driver for Quantum ESPRESSO `pw.x` |
| `scripts/dft_relax_bulk.py` | Relaxed AlN and GaN structures (PBEsol) |
| `scripts/dft_polarization.py` | Berry-phase spontaneous polarization, zinc-blende check, piezoelectric and elastic ratios |
| `scripts/dft_slab.py` | Field inside electron-counting slabs, and the dielectric constant from perturbation theory |
| `scripts/run_shen.py` | Analysis of the AlGaN/GaN series of Shen et al. (2000): line charge from density and from mobility, with sensitivity tests |
| `scripts/run_shen_uncertainty.py` | Joint fit of the calibration and the line charge with error bars, using [labplan](https://doi.org/10.5281/zenodo.22826765) |
| `scripts/run_shen_300K.py` | Tests whether one parallel channel explains the extra 300 K electrons of the Shen series |
| `scripts/run_hiroki.py`, `scripts/run_hiroki_map.py` | The independent series of Hiroki et al. (2002): relaxation each line charge needs, and the line charge allowed by their X-ray map |
| `scripts/run_elwaradi.py` | The series of Elwaradi et al. (2023): AlGaN/GaN/AlN on bulk AlN with measured relaxation; the barrier on the 20 nm channel relaxed under compression (misfit lines of the opposite sign); density and mobility tests |
| `scripts/run_jiang.py` | The Shubnikov-de Haas series of Jiang et al. (2001), same design as Shen et al., used as a control |
| `scripts/run_line_network_check.py` | Mobility limits for a straight three-direction misfit network: exact solution of the Boltzmann equation on the Fermi circle, checked against the random-line kernel |
| `scripts/run_shen_spacer.py`, `scripts/run_shen_spacer_local.py` | The spacer series of Shen et al. (their Fig. 3), a supporting check |
| `scripts/run_second_order_check.py` | Size of the second-order terms (exact area dilution, second-order piezoelectricity of Pal et al., Phys. Rev. B 84, 085211 (2011)) |
| `scripts/run_prediction.py` | The relaxed AlGaN/GaN interface, and the change of interface charge with relaxation for bound and for trapped line charge (GaN on AlN, AlN on GaN, AlGaN on GaN) |
| `scripts/wordcount.py` | Length of the Letter by the APS rules |
| `scripts/run_prb_link.py` | What the line charge would mean for the GaN/AlN hole gas of the PRB paper, for randomly oriented lines (kept for reference; no longer used in the paper) |
| `scripts/run_kp_check.py` | Recomputes the hole gas of the PRB paper with [kpenvelope](https://doi.org/10.5281/zenodo.22015269), an independent six-band solver |
| `scripts/make_numbers.py` | Writes every number used in the paper into `paper/numbers.tex` |
| `scripts/figures.py` | The three figures of the Letter and Fig. S1 (Fig. S2, the hole-gas link, is still drawn but no longer used), 1000 dpi, Times-metric font |
| `data/` | Digitized data and copied published values, each with its source |
| `dft/pseudo/` | Pseudopotentials and the `ld1.x` inputs that generate them |
| `dft/dft_runs.tar.gz` | Every DFT input and output file (`tar xzf dft/dft_runs.tar.gz -C dft` restores `dft/runs/`; wave functions and charge densities are not kept) |
| `results/` | All computed results as JSON |
| `tests/` | Automatic checks (`pytest`) |

## 3. Installation

Python 3.10 or newer:

```
pip install -r requirements.txt
pytest -q
```

The DFT part needs Quantum ESPRESSO 6.7 (`pw.x`, `ph.x`, `pp.x`,
`average.x`, `ld1.x`) and MPI. On Ubuntu 24.04 the packaged `pw.x` 6.7 aborts
with a false buffer-overflow report; `dft/snprintf_shim.c` is a small library,
loaded with `LD_PRELOAD`, that avoids it (build instructions in the file).
The pseudopotentials can be regenerated with `ld1.x < dft/pseudo/ld1_X.in`.

## 4. Reproducing the paper

```
bash scripts/run_all_dft.sh        # about 12 hours on two cores (the GaN slab alone takes about 4)
python3 scripts/run_shen.py        # about 1 hour
python3 scripts/run_shen_uncertainty.py   # about 1 hour (needs labplan)
python3 scripts/run_shen_uncertainty.py hse   # the same with the HSE constants, about 15 minutes
python3 scripts/run_shen_300K.py   # a few minutes
python3 scripts/run_hiroki.py      # about 30 minutes
python3 scripts/run_hiroki_map.py
python3 scripts/run_elwaradi.py    # about 2 hours (python3 scripts/run_elwaradi.py quick: one case)
python3 scripts/run_jiang.py       # about 10 minutes
python3 scripts/run_shen_spacer.py  # about 20 minutes
python3 scripts/run_shen_spacer_local.py   # about 20 minutes
python3 scripts/run_line_network_check.py && python3 scripts/run_line_network_check.py --shen
python3 scripts/run_second_order_check.py
python3 scripts/run_prediction.py
python3 scripts/run_prb_link.py
python3 scripts/run_kp_check.py    # about 10 minutes (needs kpenvelope)
python3 scripts/make_numbers.py
python3 scripts/figures.py
```

All results are already in `results/`, so `make_numbers.py` and
`figures.py` run in seconds without redoing anything. `labplan` and
`kpenvelope` are installed with `pip install labplan kpenvelope` (versions
0.6.0 and 0.16.0 were used). `scripts/make_numbers.py` writes `paper/numbers.tex`
and `paper/supp_tables.tex` (the numbers and tables quoted in the paper), and
`scripts/figures.py` writes `figures/`; Fig. 2 needs the slab potential
profiles, so unpack `dft/dft_runs.tar.gz` first.

## 5. Where the numbers come from

- Shen, Someya and Arakawa, Appl. Phys. Lett. 76, 2746 (2000),
  https://doi.org/10.1063/1.126463: the AlGaN/GaN densities and mobilities,
  read from their Fig. 2 (`data/shen2000_fig2_digitized.json`, with the
  method), and the structure, doping and relaxation from their text. Their
  Fig. 3, the spacer series, is in `data/shen2000_fig3_digitized.json`.
- Hiroki, Maeda and Kobayashi, J. Cryst. Growth 237-239, 956 (2002),
  https://doi.org/10.1016/S0022-0248(01)02020-6: densities of AlGaN/GaN
  barriers of increasing thickness (Figs. 2 and 3) and the X-ray map of
  Fig. 4 (`data/hiroki2002_digitized.json`, with the method).
- Elwaradi et al., J. Appl. Phys. 133, 145705 (2023),
  https://doi.org/10.1063/5.0147048: relaxation, density and mobility of
  their Tables I and II (`data/elwaradi2023_tables.json`, copied without
  change from the author version in HAL, hal-04086494).
- Jiang et al., Appl. Phys. Lett. 79, 374 (2001),
  https://doi.org/10.1063/1.1386620: subband densities and quantum mobilities
  of their Table I (`data/jiang2001_table1.json`, copied without change).
- Dreyer et al., Phys. Rev. X 6, 021038 (2016),
  https://doi.org/10.1103/PhysRevX.6.021038: hybrid-functional constants used
  as a sensitivity test (`data/dreyer2016_constants.json`).
- Vurgaftman and Meyer, J. Appl. Phys. 94, 3675 (2003),
  https://doi.org/10.1063/1.1600519, and Ambacher et al., J. Appl. Phys. 87,
  334 (2000), https://doi.org/10.1063/1.371866: lattice constants, band gaps,
  masses, dielectric constants and band offset (`src/nmp/materials.py`).
- Floro et al., J. Appl. Phys. 96, 7087 (2004),
  https://doi.org/10.1063/1.1812361: shape of the misfit network in
  AlGaN/GaN (long straight lines along three directions), used for the
  network mobility check.
- Pal et al., Phys. Rev. B 84, 085211 (2011),
  https://doi.org/10.1103/PhysRevB.84.085211: second-order piezoelectric
  coefficients of their Table I (in `scripts/run_second_order_check.py`).
- Mahim, Mohsin and Rahman, Phys. Rev. B 114, L291304 (2026): the line
  scattering fitted to the GaN/AlN hole gas (`data/prb2026_misfit_fit.json`,
  copied from its archived results), and its self-consistent hole
  distribution (`data/prb2026_well_profile.json`).

## 6. How to cite

Please cite the paper; the code is archived on Zenodo (see `CITATION.cff`).

## 7. License and contact

Apache License 2.0 (see `LICENSE`). Copyright 2026 Tanvir M. Mahim.
Contact: Tanvir M. Mahim, BRAC University, tanvir.mahim@bracu.ac.bd.
