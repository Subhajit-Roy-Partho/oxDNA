# Comparison

Ground truth is the native oxDNA C++ `pair_energy` observable queried through the local analysis stack.

Compared implementations:
- `current-f32`: `subho/oxdna2_pytorch.py` with `compute_dtype="float32"`
- `current-f64`: `subho/oxdna2_pytorch.py` with `compute_dtype="float64"`
- `oxdna_torch`: installed `../oxdna-torch` package

All cases below are evaluated under oxDNA2 average-sequence settings so the three implementations are compared on the same force field.
The speed table below measures a single differentiable forward pass with a small warm-up (`5` runs) and `20` timed iterations per case.

## Summary

| Case | Class | N | Settings | Nonzero Terms | Current f32 Total Err | Current f64 Total Err | oxdna_torch Total Err | Closest Total |
|---|---:|---:|---|---|---:|---:|---:|---|
| DSDNA8 Duplex | Duplex | 16 | DNA2 avg-seq, T=20C, salt=0.5 M | FENE, STCK, HB, CRSTCK, DH | 0.000088% | 0.000045% | 0.000047% | current-f64 |
| FORCE_FIELD 3-Strand | Three-Strand Complex | 16 | DNA2 avg-seq, T=20C, salt=1.0 M | FENE, STCK, NEXC, HB, CRSTCK, CXSTCK, DH | 0.000396% | 0.000334% | 0.000332% | oxdna_torch |
| SSDNA15 MD | Single Strand | 15 | DNA2 avg-seq, T=300K, salt=0.5 M | FENE, STCK, DH | 0.000050% | 0.000012% | 0.000005% | oxdna_torch |

## DSDNA8 Duplex

- Source: `test/DNA/DSDNA8/dsdna8.top` and `test/DNA/DSDNA8/init.dat`
- Settings: `DNA2 avg-seq, T=20C, salt=0.5 M`
- Note: Matches the DNA2 duplex validation frame used earlier.

| Term | Ground Truth | Current float32 | Error | Current float64 | Error | oxdna_torch | Error | Closest |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `FENE` | 0.95588228 | 0.95586336 | 0.001980% | 0.95586891 | 0.001399% | 0.95586888 | 0.001402% | current-f64 |
| `BEXC` | 0.00000000 | 0.00000000 | 0 | 0.00000000 | 0 | 0.00000000 | 0 | tie |
| `STCK` | -15.83250500 | -15.83251286 | 0.000050% | -15.83251283 | 0.000049% | -15.83251267 | 0.000048% | oxdna_torch |
| `NEXC` | 0.00000000 | 0.00000000 | 0 | 0.00000000 | 0 | 0.00000000 | 0 | tie |
| `HB` | -4.35607100 | -4.35606194 | 0.000208% | -4.35605912 | 0.000273% | -4.35605960 | 0.000262% | current-f32 |
| `CRSTCK` | -1.87978280 | -1.87978196 | 0.000045% | -1.87978158 | 0.000065% | -1.87978152 | 0.000068% | current-f32 |
| `CXSTCK` | 0.00000000 | 0.00000000 | 0 | 0.00000000 | 0 | 0.00000000 | 0 | tie |
| `DH` | 0.07576585 | 0.07576576 | 0.000122% | 0.07576580 | 0.000074% | 0.07576579 | 0.000075% | current-f64 |
| `total` | -21.03670930 | -21.03672791 | 0.000088% | -21.03671882 | 0.000045% | -21.03671911 | 0.000047% | current-f64 |

## FORCE_FIELD 3-Strand

- Source: `test/DNA/FORCE_FIELD/init.top` and `test/DNA/FORCE_FIELD/init.dat`
- Settings: `DNA2 avg-seq, T=20C, salt=1.0 M`
- Note: Source topology uses the newer sequence-style format. It is converted to old-format connectivity for both Torch implementations.

| Term | Ground Truth | Current float32 | Error | Current float64 | Error | oxdna_torch | Error | Closest |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `FENE` | 1.22701958 | 1.22703362 | 0.001143% | 1.22703246 | 0.001049% | 1.22703237 | 0.001042% | oxdna_torch |
| `BEXC` | 0.00000000 | 0.00000000 | 0 | 0.00000000 | 0 | 0.00000000 | 0 | tie |
| `STCK` | -14.51683300 | -14.51683426 | 0.000009% | -14.51682035 | 0.000087% | -14.51682051 | 0.000086% | current-f32 |
| `NEXC` | 0.22877100 | 0.22881973 | 0.021300% | 0.22878661 | 0.006822% | 0.22878621 | 0.006649% | oxdna_torch |
| `HB` | -4.70846800 | -4.70845413 | 0.000295% | -4.70844134 | 0.000566% | -4.70844111 | 0.000571% | current-f32 |
| `CRSTCK` | -1.72013920 | -1.72012687 | 0.000717% | -1.72013103 | 0.000475% | -1.72013111 | 0.000470% | oxdna_torch |
| `CXSTCK` | -0.47412300 | -0.47412142 | 0.000333% | -0.47412279 | 0.000044% | -0.47412274 | 0.000056% | current-f64 |
| `DH` | 0.01846696 | 0.01846691 | 0.000295% | 0.01846696 | 0.000009% | 0.01846696 | 0.000009% | oxdna_torch |
| `total` | -19.94529616 | -19.94521713 | 0.000396% | -19.94522948 | 0.000334% | -19.94522992 | 0.000332% | oxdna_torch |

## SSDNA15 MD

- Source: `test/DNA/SSDNA15/MD/ssdna15.top` and `test/DNA/SSDNA15/MD/init.dat`
- Settings: `DNA2 avg-seq, T=300K, salt=0.5 M`
- Note: The frame comes from the legacy SSDNA15 test tree, but the comparison is run under oxDNA2 average-sequence settings.

| Term | Ground Truth | Current float32 | Error | Current float64 | Error | oxdna_torch | Error | Closest |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `FENE` | 1.08199669 | 1.08200896 | 0.001133% | 1.08200279 | 0.000563% | 1.08200298 | 0.000581% | current-f64 |
| `BEXC` | 0.00000000 | 0.00000000 | 0 | 0.00000000 | 0 | 0.00000000 | 0 | tie |
| `STCK` | -15.70920000 | -15.70922565 | 0.000163% | -15.70922844 | 0.000181% | -15.70922772 | 0.000176% | current-f32 |
| `NEXC` | 0.00000000 | 0.00000000 | 0 | 0.00000000 | 0 | 0.00000000 | 0 | tie |
| `HB` | 0.00000000 | 0.00000000 | 0 | 0.00000000 | 0 | 0.00000000 | 0 | tie |
| `CRSTCK` | 0.00000000 | 0.00000000 | 0 | 0.00000000 | 0 | 0.00000000 | 0 | tie |
| `CXSTCK` | 0.00000000 | 0.00000000 | 0 | 0.00000000 | 0 | 0.00000000 | 0 | tie |
| `DH` | 0.04901923 | 0.04901924 | 0.000005% | 0.04901919 | 0.000087% | 0.04901919 | 0.000087% | current-f32 |
| `total` | -14.57820477 | -14.57819748 | 0.000050% | -14.57820647 | 0.000012% | -14.57820555 | 0.000005% | oxdna_torch |

## Speed

Timings are average wall-clock milliseconds per forward pass. Current Torch timings use `compute_dtype="float32"` with autograd-enabled inputs. `oxdna_torch` currently cannot run on MPS here because its float64-first path is rejected by Metal, so the GPU column is marked as unsupported.
These are small test systems, so GPU launch overhead is part of the result and MPS can be slower than CPU on this workload.

| Case | Current CPU (ms) | Current MPS (ms) | MPS speedup vs CPU | oxdna_torch CPU (ms) | Current CPU speedup vs oxdna_torch | oxdna_torch MPS |
|---|---:|---:|---:|---:|---:|---|
| DSDNA8 Duplex | 1.611 | 118.938 | 0.01x | 1.453 | 0.90x | unsupported |
| FORCE_FIELD 3-Strand | 1.648 | 127.413 | 0.01x | 1.445 | 0.88x | unsupported |
| SSDNA15 MD | 0.805 | 41.095 | 0.02x | 0.679 | 0.84x | unsupported |

## Notes

- Relative errors are reported only when the C++ ground-truth term is nonzero.
- For zero-reference terms, the table shows either `0` or an absolute deviation in oxDNA energy units.
- The `FORCE_FIELD` case uses a temporary conversion from the newer sequence-style topology format to old-format connectivity so both PyTorch implementations can read the same frame.
- `current-f64` is the tightest match overall in this repo when CPU double precision is available.
- `oxdna_torch` does not currently provide a working MPS path on this machine without source changes because it materializes float64 tensors in MPS-bound code paths.

