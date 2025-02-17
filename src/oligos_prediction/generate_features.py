from dataclasses import dataclass
import numpy as np
from typing import List, Dict
from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import molecular_weight
import RNA
import re
import math
from collections import Counter

@dataclass
class OligoFeatures:
    """Class to store calculated oligonucleotide features"""
    length: int
    gc_content: float
    melting_temp: float
    synthesis_scale: float
    dinucleotides_count: dict
    max_gc_runs: int
    terminal_bases: Dict[str, str]
    purine_pyrimidine_ratio: float
    hairpin_score: float
    difficult_coupling_steps: int
    sequence_complexity: float
    repeating_motifs: int
    base_run_lengths: Dict[str, int]
    hydrophobicity_score: float
    coupling_efficiency: float
    deprotection_sensitivity: float
    purification_difficulty: float
    g_quadruplex_potential: float
    end_stability: Dict[str, float]
    molecular_weight: float
    charge_density: float
    gc_scale_product: float

class OligoFeatureCalculator:

    def __init__(self):
        self.purines = set('AG')
        self.pyrimidines = set('CT')
        self.hydrophobicity_values = {
            'A': 0.2, 'T': 0.3, 'G': 0.3, 'C': 0.1, 'U': 0.3
        }

    def calculate_all_features(self, sequence: str, synthesis_scale: float) -> OligoFeatures:
        """Calculate all features for a given sequence"""
        return OligoFeatures(
            length=self.calculate_length(sequence),
            synthesis_scale=synthesis_scale,
            gc_content=self.calculate_gc_content(sequence),
            melting_temp=self.calculate_melting_temp(sequence),
            dinucleotides_count = self.calculate_dinucleotides(sequence),
            max_gc_runs=self.calculate_max_gc_runs(sequence),
            terminal_bases=self.calculate_terminal_bases(sequence),
            purine_pyrimidine_ratio=self.calculate_purine_pyrimidine_ratio(sequence),
            hairpin_score=self.calculate_hairpin_score(sequence),
            difficult_coupling_steps=self.calculate_difficult_coupling_steps(sequence),
            sequence_complexity=self.calculate_sequence_complexity(sequence),
            repeating_motifs=self.calculate_repeating_motifs(sequence),
            base_run_lengths=self.calculate_base_run_lengths(sequence),
            hydrophobicity_score=self.calculate_hydrophobicity(sequence),
            coupling_efficiency=self.calculate_coupling_efficiency(sequence),
            deprotection_sensitivity=self.calculate_deprotection_sensitivity(sequence),
            purification_difficulty=self.calculate_purification_difficulty(sequence),
            g_quadruplex_potential=self.calculate_g_quadruplex_potential(sequence),
            end_stability=self.calculate_end_stability(sequence),
            molecular_weight=self.calculate_molecular_weight(sequence),
            charge_density=self.calculate_charge_density(sequence),
            gc_scale_product=self.calculate_gc_scale_product(sequence, synthesis_scale)
        )
    
    def calculate_length(self, sequence: str) -> int:
        """Calculate sequence length

        Args:
            sequence (str): sequence 

        Returns:
            int: length of sequence
        """
        return len(sequence)
    
    def calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content percentage

        Args:
            sequence (str): sequence

        Returns:
            float: GC content percentage
        """
        gc_count = sequence.count('G') + sequence.count('C')
        return (gc_count / len(sequence)) * 100
    
    def calculate_melting_temp(self, sequence: str, method = "NN") -> float:
        """Calculate estimated melting temperature using nearest neighbor method

        Args:
            sequence (str): sequence
            method (str, optional): . Defaults to "NN" or "Wallace" or "GC".

        Returns:
            float: estimated melting temperature
        """
        try:
            if method == "NN":
                return mt.Tm_NN(sequence)
            if method == "Wallace":
                return mt.Tm_Wallace(sequence)
            if method == "GC":
                return mt.Tm_GC(sequence)
        except Exception as e:
            print("Please check inputs")
    
    def calculate_dinucleotides(self, sequence: str) -> dict:
        """Frequency of dinucleotides (two-base combinations) in the sequence

        Args:
            sequence (str): sequence

        Returns:
            dict: two-base combinations
        """

        dinucleotides = [sequence[i: i+2] for i in range(len(sequence) - 1)]
        dinucleotides_count = Counter(dinucleotides)
        return dinucleotides_count
    
    def calculate_max_gc_runs(self, sequence: str) -> int:
        """Calculate longest run of consecutive G or C bases.

        Args:
            sequence (str): sequence

        Returns:
            int: _description_
        """
        gc_runs = re.findall(r'[GC]+', sequence)
        return max(len(run) for run in gc_runs) if gc_runs else 0
    
    def calculate_terminal_bases(self, sequence: str) -> Dict['str', 'str']:
        """Get 5' and 3' terminal bases.

        Args:
            sequence (str): sequence

        Returns:
            dict: 
        """
        return {
            "5_prime": sequence[0],
            "3_prime": sequence[-1]
        }
    
    def calculate_purine_pyrimidine_ratio(self, sequence: str) -> float:
        """Calculate purine/pyrimidine ratio

        Args:
            sequence (str): sequence

        Returns:
            float: _description_
        """
        purines_temp = sum(1 for base in sequence if base in self.purines)
        pyrimidines_temp = sum(1 for base in sequence if base in self.pyrimidines)

        return purines_temp / pyrimidines_temp if pyrimidines_temp > 0 else float('inf') 
    
    def calculate_hairpin_score(self, sequence: str) -> float:
        """Calculate hairpin formation potential score
        ΔG value (lower indicates stronger self-complementarity)

        Args:
            sequence (str): sequence

        Returns:
            float: _description_
        """
        fold_comp = RNA.fold(sequence)
        return fold_comp[1]
    
    def calculate_difficult_coupling_steps(self, sequence: str) -> int:
        """Count number of potentially difficult coupling steps

        Args:
            sequence (str): sequence

        Returns:
            int: _description_
        """
        difficult_patterns = ['GG', 'CC', 'GAG', 'CAC']
        count = 0
        for pattern in difficult_patterns:
            count += sequence.count(pattern)
        return count
    
    def calculate_sequence_complexity(self, sequence: str) -> float:
        """Calculate sequence complexity using Shannon entropy

        Args:
            sequence (str): sequence

        Returns:
            float: 
        """
        base_freq = Counter(sequence)
        entropy = 0
        for base in base_freq:
            p = base_freq[base] / len(sequence)
            entropy -= p * np.log2(p)
        return entropy
    
    def calculate_repeating_motifs(self, sequence: str) -> int:
        """_summary_

        Args:
            sequence (str): _description_

        Returns:
            int: _description_
        """
        motifs = 0
        for motif_len in range(2,7):
            for i in range(len(sequence) - motif_len + 1):
                motif = sequence[i: i + motif_len]
                if sequence.count(motif) > 1:
                    motifs += 1
        
        return motifs
    
    def calculate_base_run_lengths(self, sequence: str) -> Dict[str, int]:
        """Calculate maximum run length for each base

        Args:
            sequence (str): _description_

        Returns:
            Dict[str, int]: _description_
        """

        runs = {}
        for base in "ATGC":
            runs[base] = max(len(run) for run in re.findall(f'{base}+', sequence)) if re.findall(f'{base}+', sequence) else 0

        return runs 
    
    def calculate_hydrophobicity(self, sequence: str) -> float:
        """Calculate overall hydrophobicity score

        Args:
            sequence (str): sequence

        Returns:
            float: score
        """
        return sum(self.hydrophobicity_values[base] for base in sequence) / len(sequence)
    
    # this is prediction so not sure about this 
    def calculate_coupling_efficiency(self, sequence: str) -> float:
        """_summary_

        Args:
            sequence (str): _description_

        Returns:
            float: _description_
        """
        gc_penalty = self.calculate_gc_content(sequence= sequence) * 0.01
        length_penalty = len(sequence) * 0.005
        difficult_steps_penalty = self.calculate_difficult_coupling_steps(sequence= sequence)
        return 1 - (gc_penalty + length_penalty + difficult_steps_penalty)
    
    def calculate_deprotection_sensitivity(self, sequence: str) -> float:
        """_summary_

        Args:
            sequence (str): _description_

        Returns:
            float: _description_
        """
        sensitive_bases = {'G': 0.3, 'C': 0.2, 'A': 0.1}
        return sum(sensitive_bases.get(base, 0) for base in sequence)
    
    def calculate_purification_difficulty(self, sequence: str) -> float:
        """_summary_

        Args:
            sequence (str): _description_

        Returns:
            float: _description_
        """
        length_factor = len(sequence) / 20
        gc_factor = self.calculate_gc_content(sequence= sequence) / 50
        return (length_factor + gc_factor) / 2
    
    def calculate_g_quadruplex_potential(self, sequence: str) -> float:
        """Calculate G-quadruplex formation potential

        Args:
            sequence (str): sequence

        Returns:
            float: _description_
        """
        g_runs = len(re.findall(r'G{3}', sequence))
        return g_runs * (g_runs - 1) if g_runs >= 4 else 0
    
    def calculate_end_stability(self, sequence: str) -> dict:
        """Calculate stability of 5' and 3' ends

        Args:
            sequence (str): sequence

        Returns:
            dict: _description_
        """
        end_length = min(5, len(sequence))
        return {
            "5_prime": self.calculate_gc_content(sequence= sequence[:end_length]),
            "3_prime": self.calculate_gc_content(sequence= sequence[end_length:])
        }

    def calculate_molecular_weight(self, sequence: str) -> float:
        """Calculate molecular weight

        Args:
            sequence (str): _description_

        Returns:
            float: _description_
        """
        # Check if sequence contains both T and U
        has_t = 'T' in sequence
        has_u = 'U' in sequence
        
        if has_t and has_u:
            # Convert all U to T for consistency and use DNA mode
            sequence = sequence.replace('U', 'T')
            molecule_type = 'DNA'
            
        elif 'U' in sequence:
            molecule_type = 'RNA'
        else:
            molecule_type = 'DNA'
        return molecular_weight(sequence, molecule_type)
    
    def calculate_charge_density(self, sequence: str) -> float:
        """Calculate charge density (phosphate groups per base)

        Args:
            sequence (str): _description_

        Returns:
            float: _description_
        """
        return (len(sequence) - 1) / len(sequence)
    
    def calculate_gc_scale_product(self, sequence: str, synthesis_scale: float) -> float:
        """Calculate GC content × synthesis scale

        Args:
            sequence (str): _description_
            scale (float): _description_

        Returns:
            float: _description_
        """
        return self.calculate_gc_content(sequence= sequence) * synthesis_scale
    
