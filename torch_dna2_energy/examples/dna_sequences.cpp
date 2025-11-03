#include "dna2_energy_calculator.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <map>

using namespace dna2;

/**
 * @file dna_sequences.cpp
 * @brief Example demonstrating different DNA sequences and their effects
 * 
 * This example shows how to:
 * - Create different DNA sequences (random, specific patterns)
 * - Analyze sequence-dependent energy contributions
 * - Compare AT-rich vs GC-rich sequences
 * - Study sequence effects on stability and interactions
 */

enum class Base {
    A = 0,  // Adenine
    T = 1,  // Thymine
    G = 2,  // Guanine
    C = 3   // Cytosine
};

std::vector<Base> parse_sequence(const std::string& seq_str) {
    std::vector<Base> sequence;
    for (char c : seq_str) {
        switch (toupper(c)) {
            case 'A': sequence.push_back(Base::A); break;
            case 'T': sequence.push_back(Base::T); break;
            case 'G': sequence.push_back(Base::G); break;
            case 'C': sequence.push_back(Base::C); break;
            default: 
                std::cerr << "Warning: Unknown base '" << c << "' in sequence" << std::endl;
                break;
        }
    }
    return sequence;
}

std::string sequence_to_string(const std::vector<Base>& seq) {
    std::string result;
    for (Base b : seq) {
        switch (b) {
            case Base::A: result += 'A'; break;
            case Base::T: result += 'T'; break;
            case Base::G: result += 'G'; break;
            case Base::C: result += 'C'; break;
        }
    }
    return result;
}

std::vector<Base> get_complementary_sequence(const std::vector<Base>& seq) {
    std::vector<Base> comp;
    for (Base b : seq) {
        switch (b) {
            case Base::A: comp.push_back(Base::T); break;
            case Base::T: comp.push_back(Base::A); break;
            case Base::G: comp.push_back(Base::C); break;
            case Base::C: comp.push_back(Base::G); break;
        }
    }
    return comp;
}

void create_dna_from_sequence(DNAParticle& particles, const std::vector<Base>& sequence1,
                             const std::vector<Base>& sequence2, float separation = 2.0f) {
    const int N = particles.num_particles;
    const int strand_length = sequence1.size();
    
    if (sequence2.size() != strand_length) {
        throw std::invalid_argument("Sequences must have the same length");
    }
    
    for (int i = 0; i < N; ++i) {
        float x = (i % strand_length) * 0.7f;
        float y = (i < strand_length) ? 0.0f : separation;
        float z = 0.0f;
        
        particles.positions[i][0] = x;
        particles.positions[i][1] = y;
        particles.positions[i][2] = z;
        
        // Set strand IDs
        particles.strand_ids[i] = (i < strand_length) ? 0 : 1;
        
        // Set base types from sequences
        if (i < strand_length) {
            particles.types[i] = static_cast<int>(sequence1[i]);
            particles.btypes[i] = static_cast<int>(sequence1[i]);
        } else {
            particles.types[i] = static_cast<int>(sequence2[i - strand_length]);
            particles.btypes[i] = static_cast<int>(sequence2[i - strand_length]);
        }
        
        // Set up backbone bonding
        if (i < strand_length - 1) {
            particles.n3_neighbors[i] = i + 1;
            particles.n5_neighbors[i + 1] = i;
            particles.bonded_mask[i][i + 1] = true;
            particles.bonded_mask[i + 1][i] = true;
        } else if (i >= strand_length && i < N - 1) {
            particles.n3_neighbors[i] = i + 1;
            particles.n5_neighbors[i + 1] = i;
            particles.bonded_mask[i][i + 1] = true;
            particles.bonded_mask[i + 1][i] = true;
        }
        
        // Set up orientations
        particles.orientations[i] = torch::eye(3, torch::kFloat32);
    }
    
    particles.compute_interaction_centers();
}

void demonstrate_sequence_patterns() {
    std::cout << "\n=== DNA Sequence Patterns Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    const int strand_length = 12;
    const int N = 2 * strand_length;
    
    // Define different sequence patterns
    std::map<std::string, std::vector<Base>> sequences = {
        {"AT_rich", parse_sequence("ATATATATATAT")},
        {"GC_rich", parse_sequence("GCGCGCGCGCGC")},
        {"alternating", parse_sequence("AGCTAGCTAGCT")},
        {"random", parse_sequence("AGTCGATCGTAC")},
        {"polyA", parse_sequence("AAAAAAAAAAAA")},
        {"polyG", parse_sequence("GGGGGGGGGGGG")}
    };
    
    std::cout << "Testing different sequence patterns:" << std::endl;
    std::cout << std::setw(12) << "Pattern" << std::setw(20) << "Sequence" 
              << std::setw(15) << "Energy (kT)" << std::setw(15) << "GC Content" << std::endl;
    std::cout << std::string(62, '-') << std::endl;
    
    for (const auto& [name, seq] : sequences) {
        // Create complementary strand
        auto comp_seq = get_complementary_sequence(seq);
        
        // Create particle system
        DNAParticle particles(N, torch::kCPU);
        create_dna_from_sequence(particles, seq, comp_seq);
        
        // Compute energy
        torch::Tensor energy = calculator.compute_energy(particles);
        
        // Calculate GC content
        int gc_count = 0;
        for (Base b : seq) {
            if (b == Base::G || b == Base::C) gc_count++;
        }
        float gc_content = 100.0f * gc_count / seq.size();
        
        std::cout << std::setw(12) << name
                  << std::setw(20) << sequence_to_string(seq)
                  << std::setw(15) << std::fixed << std::setprecision(6) << energy.item<float>()
                  << std::setw(14) << std::setprecision(1) << gc_content << "%" << std::endl;
    }
}

void demonstrate_melting_temperature_analysis() {
    std::cout << "\n=== Melting Temperature Analysis Demo ===" << std::endl;
    
    const int strand_length = 10;
    const int N = 2 * strand_length;
    
    // Create sequences with different GC content
    std::vector<std::pair<std::string, std::vector<Base>>> test_sequences = {
        {"0% GC", parse_sequence("ATATATATAT")},
        {"25% GC", parse_sequence("ATATATATGC")},
        {"50% GC", parse_sequence("ATATGCGCGC")},
        {"75% GC", parse_sequence("GCGCGCGCAT")},
        {"100% GC", parse_sequence("GCGCGCGCGC")}
    };
    
    std::vector<float> temperatures = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.8f, 1.0f};
    
    std::cout << "Energy vs temperature for different GC contents:" << std::endl;
    std::cout << std::setw(8) << "Temp" << std::setw(12) << "0% GC" << std::setw(12) << "25% GC"
              << std::setw(12) << "50% GC" << std::setw(12) << "75% GC" << std::setw(12) << "100% GC" << std::endl;
    std::cout << std::string(68, '-') << std::endl;
    
    for (float temp : temperatures) {
        DNA2Parameters params;
        params.temperature = temp;
        params.initialize_tensors();
        DNA2EnergyCalculator calculator(params);
        
        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << temp;
        
        for (const auto& [name, seq] : test_sequences) {
            auto comp_seq = get_complementary_sequence(seq);
            DNAParticle particles(N, torch::kCPU);
            create_dna_from_sequence(particles, seq, comp_seq);
            
            torch::Tensor energy = calculator.compute_energy(particles);
            std::cout << std::setw(12) << std::setprecision(4) << energy.item<float>();
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nNote: Higher temperatures reduce binding energy, simulating melting behavior." << std::endl;
    std::cout << "GC-rich sequences maintain stronger binding at higher temperatures." << std::endl;
}

void demonstrate_mismatch_effects() {
    std::cout << "\n=== Mismatch Effects Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    const int strand_length = 10;
    const int N = 2 * strand_length;
    
    // Perfect match sequence
    std::vector<Base> perfect_seq = parse_sequence("ATGCGATCGA");
    std::vector<Base> perfect_comp = get_complementary_sequence(perfect_seq);
    
    // Create sequences with different types of mismatches
    std::vector<std::pair<std::string, std::vector<Base>>> mismatched_sequences = {
        {"Perfect match", perfect_comp},
        {"Single mismatch (A-C)", parse_sequence("TACGCTAGCT")},
        {"Double mismatch", parse_sequence("TTCGCTAGTT")},
        {"Bulge (insertion)", parse_sequence("TACGCTAGCTT")},
        {"Deletion", parse_sequence("TACGCTAGC")}
    };
    
    std::cout << "Effects of mismatches on binding energy:" << std::endl;
    std::cout << std::setw(20) << "Mismatch Type" << std::setw(20) << "Energy (kT)" 
              << std::setw(15) << "ΔE vs Perfect" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    // Compute perfect match energy as reference
    DNAParticle perfect_particles(N, torch::kCPU);
    create_dna_from_sequence(perfect_particles, perfect_seq, perfect_comp);
    torch::Tensor perfect_energy = calculator.compute_energy(perfect_particles);
    float perfect_e = perfect_energy.item<float>();
    
    for (const auto& [name, seq] : mismatched_sequences) {
        DNAParticle particles(N, torch::kCPU);
        create_dna_from_sequence(particles, perfect_seq, seq);
        
        torch::Tensor energy = calculator.compute_energy(particles);
        float e = energy.item<float>();
        float delta_e = e - perfect_e;
        
        std::cout << std::setw(20) << name
                  << std::setw(20) << std::fixed << std::setprecision(6) << e
                  << std::setw(15) << std::showpos << std::setprecision(6) << delta_e << std::endl;
    }
    
    std::cout << "\nNote: Positive ΔE indicates destabilization relative to perfect match." << std::endl;
}

void demonstrate_sequence_dependent_interactions() {
    std::cout << "\n=== Sequence-Dependent Interactions Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    const int strand_length = 8;
    const int N = 2 * strand_length;
    
    // Test all possible base pairs
    std::vector<std::pair<Base, Base>> base_pairs = {
        {Base::A, Base::T}, {Base::T, Base::A},  // AT pairs
        {Base::G, Base::C}, {Base::C, Base::G},  // GC pairs
        {Base::A, Base::C}, {Base::C, Base::A},  // Mismatches
        {Base::A, Base::G}, {Base::G, Base::A},  // Mismatches
        {Base::T, Base::C}, {Base::C, Base::T},  // Mismatches
        {Base::T, Base::G}, {Base::G, Base::T}   // Mismatches
    };
    
    std::cout << "Energy contributions of different base pairs:" << std::endl;
    std::cout << std::setw(10) << "Base Pair" << std::setw(15) << "Energy (kT)" 
              << std::setw(15) << "Type" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (const auto& [base1, base2] : base_pairs) {
        // Create homopolymer of this base pair
        std::vector<Base> seq1(strand_length, base1);
        std::vector<Base> seq2(strand_length, base2);
        
        DNAParticle particles(N, torch::kCPU);
        create_dna_from_sequence(particles, seq1, seq2);
        
        torch::Tensor energy = calculator.compute_energy(particles);
        float e_per_bp = energy.item<float>() / strand_length;
        
        std::string pair_name = sequence_to_string({base1}) + "-" + sequence_to_string({base2});
        std::string pair_type = ((base1 == Base::A && base2 == Base::T) || 
                                (base1 == Base::T && base2 == Base::A) ||
                                (base1 == Base::G && base2 == Base::C) || 
                                (base1 == Base::C && base2 == Base::G)) ? 
                               "Watson-Crick" : "Mismatch";
        
        std::cout << std::setw(10) << pair_name
                  << std::setw(15) << std::fixed << std::setprecision(6) << e_per_bp
                  << std::setw(15) << pair_type << std::endl;
    }
}

void demonstrate_realistic_sequences() {
    std::cout << "\n=== Realistic DNA Sequences Demo ===" << std::endl;
    
    auto calculator = create_default_calculator();
    
    // Some biologically relevant sequences
    std::map<std::string, std::string> biological_sequences = {
        {"Promoter", "TTGACAATATG"},
        {"Ribosome binding", "AGGAGG"},
        {"Restriction site", "GAATTC"},
        {"Random genomic", "ATCGATCGATCGATCG"},
        {"Highly repetitive", "ATATATATATATATAT"}
    };
    
    std::cout << "Energy analysis of biologically relevant sequences:" << std::endl;
    std::cout << std::setw(15) << "Sequence Type" << std::setw(20) << "Sequence" 
              << std::setw(12) << "Length" << std::setw(15) << "Energy (kT)" 
              << std::setw(12) << "Energy/bp" << std::endl;
    std::cout << std::string(74, '-') << std::endl;
    
    for (const auto& [type, seq_str] : biological_sequences) {
        auto seq = parse_sequence(seq_str);
        auto comp_seq = get_complementary_sequence(seq);
        
        const int N = 2 * seq.size();
        DNAParticle particles(N, torch::kCPU);
        create_dna_from_sequence(particles, seq, comp_seq);
        
        torch::Tensor energy = calculator.compute_energy(particles);
        float total_e = energy.item<float>();
        float e_per_bp = total_e / seq.size();
        
        std::cout << std::setw(15) << type
                  << std::setw(20) << seq_str
                  << std::setw(12) << seq.size()
                  << std::setw(15) << std::fixed << std::setprecision(6) << total_e
                  << std::setw(12) << std::setprecision(6) << e_per_bp << std::endl;
    }
}

int main() {
    std::cout << "DNA2 Energy Calculator - DNA Sequences Example" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        demonstrate_sequence_patterns();
        demonstrate_melting_temperature_analysis();
        demonstrate_mismatch_effects();
        demonstrate_sequence_dependent_interactions();
        demonstrate_realistic_sequences();
        
        std::cout << "\n✓ All DNA sequence examples completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}