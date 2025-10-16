"""
Modified version of output_bonds.py that computes TRUE per-nucleotide energies.

This version splits pairwise energies equally between the two particles,
so that summing all per-nucleotide energies gives the total system energy
(without double counting).

Key differences from output_bonds.py:
1. Divides pairwise energies by 2 when assigning to nucleotides
2. Provides correct total energy (not double-counted)
3. Better for quantitative analysis

Usage:
    python output_per_nucleotide_energy.py input.dat trajectory.dat -v output.json

Author: Modified from oxDNA_analysis_tools.output_bonds
"""

from typing import List
import numpy as np
import argparse
from os import path
from collections import namedtuple
import sys

# Try to import oxDNA analysis tools
try:
    from oxDNA_analysis_tools.UTILS.logger import log, logger_settings
    from oxDNA_analysis_tools.UTILS.data_structures import TopInfo, TrajInfo
    from oxDNA_analysis_tools.UTILS.oat_multiprocesser import oat_multiprocesser
    from oxDNA_analysis_tools.UTILS.RyeReader import describe
    import oxpy
except ImportError:
    print("Error: This script requires oxDNA_analysis_tools and oxpy")
    print("Install with: pip install oxDNA-analysis-tools")
    sys.exit(1)

ComputeContext = namedtuple("ComputeContext", ["traj_info",
                                              "top_info",
                                              "input_file",
                                              "visualize",
                                              "conversion_factor",
                                              "n_potentials",
                                              "split_energies"])

def parse_header(e_txt: str) -> List[str]:
    """Parse the header line to extract potential names."""
    e_txt = e_txt.strip('#')
    e_txt = e_txt.split(',')[0]
    e_list = e_txt.split(' ')[2:]
    return e_list

def get_potentials(ctx) -> List[str]:
    """Get the list of energy potentials from the first configuration."""
    with oxpy.Context():
        inp = oxpy.InputFile()
        inp.init_from_filename(ctx.input_file)
        inp["list_type"] = "cells"
        inp["trajectory_file"] = ctx.traj_info.path
        inp["analysis_bytes_to_skip"] = str(0)
        inp["confs_to_analyse"] = str(1)
        inp["analysis_data_output_1"] = '''{
            name = stdout
            print_every = 1e10
            col_1 = {
                id = my_obs
                type = pair_energy
            }
        }'''

        backend = oxpy.analysis.AnalysisBackend(inp)
        backend.read_next_configuration()
        e_txt = backend.config_info().get_observable_by_id("my_obs").get_output_string(
            backend.config_info().current_step).strip().split('\n')
        pot_names = parse_header(e_txt[0])

    return pot_names

def compute(ctx: ComputeContext, chunk_size: int, chunk_id: int):
    """
    Compute per-nucleotide energies for a chunk of configurations.

    KEY MODIFICATION: Splits pairwise energies equally between particles.
    """
    with oxpy.Context():
        inp = oxpy.InputFile()
        inp.init_from_filename(ctx.input_file)
        inp["list_type"] = "cells"
        inp["trajectory_file"] = ctx.traj_info.path
        inp["analysis_bytes_to_skip"] = str(ctx.traj_info.idxs[chunk_id * chunk_size].offset)
        inp["confs_to_analyse"] = str(chunk_size)
        inp["analysis_data_output_1"] = '''{
            name = stdout
            print_every = 1e10
            col_1 = {
                id = my_obs
                type = pair_energy
            }
        }'''

        if (not inp["use_average_seq"] or inp.get_bool("use_average_seq")) and "RNA" in inp["interaction_type"]:
            log("Sequence dependence not set for RNA model, wobble base pairs will be ignored", level="warning")

        backend = oxpy.analysis.AnalysisBackend(inp)

        if ctx.visualize:
            energies = np.zeros((ctx.top_info.nbases, ctx.n_potentials))

        while backend.read_next_configuration():
            e_txt = backend.config_info().get_observable_by_id("my_obs").get_output_string(
                backend.config_info().current_step).strip().split('\n')

            if ctx.visualize:
                for e in e_txt[1:]:
                    if not e[0] == '#':
                        e = e.split()
                        p = int(e[0])
                        q = int(e[1])
                        l = np.array([float(x) for x in e[2:]]) * ctx.conversion_factor

                        # MODIFICATION: Split energy equally between particles
                        if ctx.split_energies:
                            energies[p] += l / 2.0
                            energies[q] += l / 2.0
                        else:
                            # Original behavior (double counting)
                            energies[p] += l
                            energies[q] += l
            else:
                # Print mode
                print(e_txt[0])
                for e in e_txt[1:]:
                    if not e[0] == '#':
                        e = e.split()
                        p = int(e[0])
                        q = int(e[1])
                        l = np.array([float(x) for x in e[2:]]) * ctx.conversion_factor
                        print(p, q, end=' ')
                        [print(v, end=' ') for v in l]
                        print()
                    else:
                        print(e)

        if ctx.visualize:
            return energies
        else:
            return

def output_per_nucleotide_energy(traj_info: TrajInfo, top_info: TopInfo, inputfile: str,
                                 visualize: bool = False, conversion_factor: float = 1,
                                 split_energies: bool = True, ncpus: int = 1):
    """
    Compute per-nucleotide energies in a trajectory.

    Parameters:
        traj_info (TrajInfo): Information about the trajectory.
        top_info (TopInfo): Information about the topology.
        inputfile (str): Path to the input file.
        visualize (bool): If True, compute averaged per-nucleotide energies for oxView.
                         If False, print pairwise energies to stdout.
        conversion_factor (float): Conversion factor (1 for oxDNA SU, 41.42 for pN·nm).
        split_energies (bool): If True, split pairwise energies equally (no double counting).
                              If False, use original behavior (adds full energy to both).
        ncpus (int): Number of CPUs to use.

    Returns:
        tuple: (energies, pot_names) if visualize=True, else (energies, None)
    """

    ctx = ComputeContext(traj_info, top_info, inputfile, visualize, conversion_factor, 0, split_energies)

    # Get potential names
    pot_names = get_potentials(ctx)
    ctx = ComputeContext(traj_info, top_info, inputfile, visualize, conversion_factor,
                        len(pot_names), split_energies)

    energies = np.zeros((ctx.top_info.nbases, len(pot_names)))

    def callback(i, r):
        nonlocal visualize, energies
        if visualize:
            energies += r
        else:
            print(r)

    oat_multiprocesser(traj_info.nconfs, ncpus, compute, callback, ctx)

    if visualize:
        return energies, pot_names
    else:
        return energies, None

def cli_parser(prog="output_per_nucleotide_energy.py"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Compute per-nucleotide energies (with correct accounting, no double counting)",
        epilog="This is a modified version of output_bonds.py that splits pairwise energies equally."
    )
    parser.add_argument('inputfile', type=str, nargs=1,
                       help="The input file used to run the simulation")
    parser.add_argument('trajectory', type=str, nargs=1,
                       help='The trajectory file to analyze')
    parser.add_argument('-v', '--view', type=str, nargs=1, dest='outfile',
                       help='Output average per-particle energy as oxView JSON')
    parser.add_argument('-p', '--parallel', metavar='num_cpus', nargs=1, type=int, dest='parallel',
                       help="(optional) Number of cores to use")
    parser.add_argument('-u', '--units', type=str, nargs=1, dest='units',
                       help="(optional) Energy units: 'pNnm' or 'oxDNA' (default: oxDNA)")
    parser.add_argument('--no-split', dest='no_split', action='store_true',
                       help="Use original behavior (double counting, same as output_bonds.py)")
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true',
                       help="Don't print INFO messages to stderr")
    return parser

def main():
    parser = cli_parser(path.basename(__file__))
    args = parser.parse_args()

    logger_settings.set_quiet(args.quiet)

    # Check dependencies
    try:
        from oxDNA_analysis_tools.config import check
        check(["python", "numpy", "oxpy"])
    except:
        pass

    traj_file = args.trajectory[0]
    inputfile = args.inputfile[0]

    top_info, traj_info = describe(None, traj_file)

    # Determine output mode
    try:
        outfile: str = args.outfile[0]
        visualize = True
    except:
        outfile = ''
        visualize = False

    # Number of CPUs
    if args.parallel:
        ncpus = args.parallel[0]
    else:
        ncpus = 1

    # Energy units
    if args.units:
        if args.units[0] == "pNnm":
            units = "pN nm"
            conversion_factor = 41.42
        elif args.units[0] == "oxDNA":
            units = "oxDNA su"
            conversion_factor = 1
        else:
            raise RuntimeError(f"Unrecognized units: {args.units[0]}\n"
                             f"Recognized options are 'pNnm' and 'oxDNA'.")
    else:
        units = "oxDNA su"
        conversion_factor = 1
        if not args.quiet:
            log("No units specified, assuming oxDNA su")

    # Split energies or not
    split_energies = not args.no_split

    if not args.quiet:
        if split_energies:
            log("Using split energy mode: pairwise energies divided equally between particles")
            log("Total system energy = sum of per-nucleotide energies (no double counting)")
        else:
            log("Using original mode: full pairwise energies added to both particles")
            log("Total system energy = (1/2) × sum of per-nucleotide energies")

    # Compute energies
    energies, potentials = output_per_nucleotide_energy(
        traj_info, top_info, inputfile, visualize, conversion_factor, split_energies, ncpus
    )

    if visualize:
        # Average over trajectory
        energies /= traj_info.nconfs

        # Write separate JSON file for each potential
        for i, potential in enumerate(potentials):
            if '.json' in outfile:
                fname = '.'.join(outfile.split('.')[:-1]) + "_" + potential + '.json'
            else:
                fname = outfile + "_" + potential + '.json'

            with open(fname, 'w+') as f:
                f.write("{{\n\"{} ({})\" : [".format(potential, units))
                f.write(', '.join([str(x) for x in energies[:, i]]))
                f.write("]\n}")

            if not args.quiet:
                log(f"Wrote oxView overlay to: {fname}")

        # Also write a summary
        total_energy = np.sum(energies[:, -1])  # Last column is total
        if not args.quiet:
            log(f"\nTotal system energy: {total_energy:.6f} {units}")
            log(f"Average per-nucleotide energy: {total_energy/len(energies):.6f} {units}")

            # Print energy breakdown
            log("\nEnergy breakdown:")
            for i, potential in enumerate(potentials[:-1]):  # Exclude 'total'
                pot_sum = np.sum(energies[:, i])
                log(f"  {potential:10s}: {pot_sum:12.6f} {units} ({pot_sum/total_energy*100:5.1f}%)")

if __name__ == "__main__":
    main()
