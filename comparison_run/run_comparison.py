import os
import subprocess

def run_simulation(backend, input_file, binary_path, output_prefix):
    print(f"Running {backend} simulation...")
    with open(input_file, 'w') as f:
        f.write(f"backend = {backend}\n")
        f.write("backend_precision = float\n") 
        f.write("sim_type = MD\n")
        f.write("steps = 1000\n")
        f.write("dt = 0.003\n") 
        f.write("print_energy_every = 100\n") 
        f.write("print_conf_every = 10000\n")
        f.write("interaction_type = DNA2\n")
        f.write("salt_concentration = 1.0\n")
        f.write("T = 300K\n")
        f.write("verlet_skin = 0.05\n")
        f.write("diff_coeff = 2.5\n")
        f.write("thermostat = john\n") # Brownian
        f.write("newtonian_steps = 103\n")
        f.write("energy_file = {}\n".format(output_prefix + "_energy.dat"))
        f.write("trajectory_file = {}\n".format(output_prefix + "_trajectory.dat"))
        f.write("conf_file = configuration.dat\n")
        f.write("topology = topology.top\n")
        f.write("restart_step_counter = 1\n")
        
    cmd = f"{binary_path} {input_file}"
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error running {backend}: {e}")
        # print first few lines of error
        print(e.stdout.decode()[:500])
        print(e.stderr.decode()[:500])

def parse_energy(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Time, Pot, Kin, Tot
                    step = float(parts[0])
                    pot = float(parts[1]) 
                    data.append([step, pot])
                except ValueError:
                    continue
    return data

def main():
    cpu_bin = "../build_cpu/bin/oxDNA"
    metal_bin = "../build_metal/bin/oxDNA"
    
    if not os.path.exists("configuration.dat"):
        print("Generating topology and configuration...")
        
        # Create topology (2 particles, 1 strand)
        with open("topology.top", "w") as f:
            f.write("2 1\n") 
            f.write("1 A -1 1\n")
            f.write("1 T 0 -1\n")
        
        # Create input for generator
        with open("input_gen", "w") as f:
            f.write("box_type = orthogonal\n")
            f.write("box_size = 20\n")
            f.write("step_amplitude = 0.5\n")
            f.write("N = 2\n") 
            f.write("topology = topology.top\n")
            f.write("particle_type = DNA2\n")
            f.write("energy_file = /dev/null\n")
            f.write("trajectory_file = configuration.dat\n")
            f.write("conf_file = /dev/null\n")
            # Minimal options
        
        # Run confGenerator with explicit box size if needed, though input file usually suffices if formatted right.
        # But the error 'Usage ...' suggests it strictly wants arguments.
        # Try passing box size.
        subprocess.run("../build_metal/bin/confGenerator input_gen 20", shell=True)
            
    # Remove faulty copy block
    # Check if configuration.dat exists now
    if not os.path.exists("configuration.dat"):
        print("Error: Configuration generation failed.")
        return

    # Check topology
    if not os.path.exists("topology.top"):
        print("Error: Topology missing.")
        return
    
    run_simulation("CPU", "input.cpu", cpu_bin, "cpu")
    run_simulation("Metal", "input.metal", metal_bin, "metal")
    
    try:
        cpu_data = parse_energy("cpu_energy.dat")
        metal_data = parse_energy("metal_energy.dat")
        
        print(f"{'Step':<10} {'CPU Pot':<15} {'Metal Pot':<15} {'Diff':<15}")
        print("-" * 55)
        
        n = min(len(cpu_data), len(metal_data))

        for i in range(n):
            step = cpu_data[i][0]
            pot_cpu = cpu_data[i][1]
            pot_metal = metal_data[i][1]
            diff = abs(pot_cpu - pot_metal)
            print(f"{int(step):<10} {pot_cpu:<15.6f} {pot_metal:<15.6f} {diff:<15.6f}")
            
    except Exception as e:
        print(f"Error parsing data: {e}")

if __name__ == "__main__":
    main()
