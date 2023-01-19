# Parallel Tempering

Authors: Erik Popppleton & Petr Šulc
Last updated: Jan 2023

This example shows how to calculate a persistence length of a DNA duplex.

### Running the simulation
This directory contains initial configuration files for a 201 base pair dsDNA segment.  Run the simulation with
```
oxDNA input_persistence
```

For persistence length, one needs a large number of decorrelated states, so this simulation takes a while.  It can be sped up by running on a GPU rather than the default CPU.

### Analysis
There is a script for calculating the persistence length of a paired sequence of DNA in oxDNA_analysis_tools.  Make sure that you compiled oxDNA with `-DPython=1`, and then run
```
oat persistence_length trajectory.dat input_persistences 10 50 -p5
```

This script will calculate the persistence length, which should be around 120 nucleotides.

### Persistence length
Persistence length can be calculated from the average correlation between vectors tangent to a polymer.  In this case, we use vectors pointing from the midpoint of one base pair to the midpoint of the next base pair along the helix (note that this only works for DNA, not RNA). The persistence length can be found by performing an exponential fit to:
<p align=center>
&#9001;**n_k** &middot; **n_0** &#12297; = B * exp(x/L_ps )
</p> 