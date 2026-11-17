# Changelog

## 1.3.1 (2026-11)

- `scripts/figures.py`: new overview figure (Fig. 1). Panel (a) draws the
  relaxed sample of Shen et al. as they report it, with its misfit network,
  and the charge per length of one misfit dislocation in each case; panel (b)
  lists the evidence, the findings and the consequences. Every number is read
  from `paper/numbers.tex` and cross-checked against the results and data
  files, and the script stops if any text leaves its box or overlaps other
  text. Run `scripts/make_numbers.py` first.
- The former Fig. 1(b), AlN on relaxed GaN, is now Fig. S1; the hole-gas
  figure is now Fig. S2. Figs. 2 and 3 are slightly shorter; Fig. 3(a) says
  "electron density".

## 1.3.0 (2026-11)

- `scripts/run_prediction.py`: splits the charge change on relaxation, per unit
  2R|eps|, into a part odd in the sign of the mismatch (piezoelectric response
  and bound core charge) and a part even in it (electrons trapped on the
  cores). For Al0.3Ga0.7N stretched on GaN and compressed on a GaN channel held
  at the lattice of AlN, the even part is zero for any bound charge and -q/b
  for trapped charge q; the sum of the two needs no polarization constant.
  Also gives the error of q and of Lambda for a given error of the measured
  interface charge.
- One more test (the sum rule).
- Figure text and table captions reworded.

## 1.2.0 (2026-11)

- New `scripts/run_elwaradi.py` and `data/elwaradi2023_tables.json`: a second
  series with measured relaxation (Elwaradi et al., J. Appl. Phys. 133,
  145705 (2023)), in which the AlGaN barrier on a 20 nm GaN channel relaxed
  under compression. Full AlN/GaN/AlN-spacer/AlGaN/GaN stack with electrons
  and holes, 72 calibrated variants; mobility bound and estimate for the
  barrier lines.
- `scripts/run_prediction.py`: new case, Al0.3Ga0.7N on a GaN channel held at
  the lattice of AlN (the bound-versus-trapped test at the same kind of
  interface as the Shen data). Fig. 3(d) now shows this case.
- One more test (interface charge of the stack against Eq. 2).

## 1.1.0 (2026-11)

- New `scripts/run_prediction.py`: the change of the interface charge with
  relaxation for bare-end, neutral, bound and trapped line charge, for GaN on
  AlN, AlN on GaN and AlGaN on GaN, with the PBEsol and HSE constants; the
  charge of fully relaxed Al0.3Ga0.7N/GaN, compared with Ye et al., Nat.
  Commun. 16, 3863 (2025); the line-charge fraction at which that charge
  changes sign.
- New `scripts/run_jiang.py` and `data/jiang2001_table1.json`: the
  Shubnikov-de Haas series of Jiang et al. (2001) as a control.
- `scripts/run_shen_uncertainty.py hse`: the joint fit with error bars using
  the HSE constants.
- `scripts/wordcount.py`: Letter length by the APS rules.
- Fig. 3(d) of the Letter now shows the bound-versus-trapped prediction; the
  hole-gas relaxation curve moved to Fig. S1.
- Two tests of the prediction (Eq. 2 slopes; agreement with Ye et al.).

## 1.0.0 (2026-11)

First public version, accompanying the paper "The charge of misfit
dislocations decides the polarization charge at relaxed nitride interfaces"
(T. M. Mahim, submitted).

- Berry-phase polarization constants of AlN and GaN (PBEsol), zinc-blende
  reference check, piezoelectric constants with relaxed ions, C13/C33.
- Electron-counting slab test of the face charge.
- Exact first-order decomposition of the bound charge at relaxed interfaces
  into coherent, proper-piezoelectric and misfit-line parts, with tests.
- Schroedinger-Poisson analysis of the AlGaN/GaN series of Shen et al. (2000),
  line charge from density and mobility, sensitivity tests.
- Joint fit of the calibration and the line charge with error bars (labplan),
  including the uncertainty of the measured relaxation.
- Test of a common parallel channel as the origin of the extra 300 K
  electrons of the Shen series, and the 300 K analysis with it removed.
- Analysis of the independent AlGaN/GaN series of Hiroki et al. (2002): the
  relaxation each line charge needs, and the line charge allowed by the
  relaxation read from their X-ray map.
- Electron-counting slabs of four and six formula units for both AlN and GaN.
- Link to the GaN/AlN hole gas of Mahim, Mohsin and Rahman, Phys. Rev. B 114,
  L291304 (2026), with an independent recomputation of its hole gas by
  kpenvelope.
