#!/usr/bin/env python3
"""
This single script can run one of three different molecule generation pipelines, depending on the user's choice:

1. GenMol1F: Single-factor search (CNNaffinity) with feasibility based on CNNaffinity.
2. GenMol1F with plus mode: Single-factor search (CNNaffinity) with extended feasibility test (CNNaffinity plus MolWt and SAS constraints).
3. GenMolMF: Multi-factor search (e.g. CNNaffinity, MolWt, SAS) with feasibility on all factors.
   - NOW USES BRANCH AND BOUND ALGORITHM instead of greedy search
0. To abort the run

Example run:
    python GenMol.py --protein DBH --target_size 1 --choice mf --context True --model gemini-2.5-pro --final_k 100
"""

import argparse
import random
import math
import numpy as np
from datetime import datetime
import os
import json
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import heapq  # For priority queue implementation

from env_utils import load_api_key
from search import Hypothesis, compute_Q, construct_file_paths
from LMLFStar import (
    generate_molecules_for_protein,
    generate_molecules_for_protein_with_context,
    generate_molecules_for_protein_multifactors,
    generate_molecules_for_protein_multifactors_with_context
)


# =========================
# Helper: Environment Setup
# =========================
def setup_environment(protein, results_subdir, data_path="data", model_engine="gemini-2.5-pro"):
    """
    Sets up common parameters and directories.
    Returns a dictionary with:
      - date_time (timestamp)
      - labelled_data and unlabelled_data (from CSV files)
      - api_key, model_engine, gnina_path, config_path, temp_dir, output_dir
    """
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    labelled_file, unlabelled_file = construct_file_paths(data_path, protein)
    labelled_data = pd.read_csv(labelled_file).to_dict(orient="records")
    unlabelled_data = pd.read_csv(unlabelled_file).to_dict(orient="records")
    api_key = load_api_key()
    gnina_path = "./docking"
    config_path = f"./docking/{protein}/{protein}_config.txt"
    temp_dir = "/tmp/molecule_generation"
    output_dir = f"results/{results_subdir}/{protein}/{model_engine}/{date_time}"
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return {
        "date_time": date_time,
        "labelled_data": labelled_data,
        "unlabelled_data": unlabelled_data,
        "api_key": api_key,
        "model_engine": model_engine,
        "gnina_path": gnina_path,
        "config_path": config_path,
        "temp_dir": temp_dir,
        "output_dir": output_dir
    }


# ====================================
# Pipeline 1: GenMol1F (Single-Factor)
# ====================================
def GenMol1F(seed=0, protein="DBH", target_size=5, final_k=20, context=False, model_engine="gpt-4o", plus_mode=False):
    """
    Single-factor search for CNNaffinity.
    Checks feasibility solely by verifying that the molecule's CNNaffinity
    lies within the current search interval.
    """
    print("[Function Outdated.] This function is removed from the current version.")
    return


# ========================================================
# Pipeline 2: GenMolMF (Multi-Factor) - BRANCH AND BOUND
# ========================================================
def GenMolMF(seed=0, protein="DBH", target_size=5, final_k=20, context=False, model_engine="gemini-2.5-pro"):
    """
    Multi-factor search using BRANCH AND BOUND algorithm.
    The algorithm searches for optimal parameter ranges for multiple properties
    (e.g. CNNaffinity, MolWt, SAS) and verifies that each molecule satisfies
    the corresponding constraints.
    
    Key differences from greedy search:
    - Uses a max-priority queue ordered by Upper Bound Q-score
    - Explores nodes with highest potential first
    - Prunes branches when UB <= best_w (current best solution)
    - Guarantees optimal solution within search space
    """
    random.seed(seed)
    np.random.seed(seed)

    env = setup_environment(protein, "GenMolMF_BranchBound", model_engine=model_engine)
    labelled_data = env["labelled_data"]
    unlabelled_data = env["unlabelled_data"]
    api_key = env["api_key"]
    gnina_path = env["gnina_path"]
    config_path = env["config_path"]
    temp_dir = env["temp_dir"]
    output_dir = env["output_dir"]

    def branch_and_bound_LMLFStar(protein, labelled_data, unlabelled_data, initial_intervals,
                                   api_key, model_engine, gnina_path, config_path, temp_dir,
                                   output_dir, s=4, n=10, max_samples=5, final_k=100, 
                                   target_size=5, context=False, neg_threshold=-0.5):
        """
        Branch and Bound implementation of LMLFStar.
        
        Parameters:
        - s: branching factor (number of children per node)
        - n: maximum iterations
        - neg_threshold: domain-specific cutoff for Q-score (line 14 in pseudocode)
        """
        
        # Initialize
        factors = [lambda x, p=param: x.get(p) for param in initial_intervals.keys()]
        e_root = [initial_intervals[param] for param in initial_intervals]
        
        theta_ext_h_default = len(unlabelled_data) / (len(labelled_data) + len(unlabelled_data))
        
        # Compute upper bound for root (optimistic estimate)
        root_UB = compute_upper_bound_optimistic(e_root, labelled_data, factors, 
                                                   theta_ext_h_default, initial_intervals)
        
        # Priority queue: stores (-UB, node_id, interval) 
        # Negative UB because heapq is min-heap, we want max-heap
        PQ = []
        heapq.heappush(PQ, (-root_UB, 0, e_root))
        
        # Best solution tracking
        best_w = -float('inf')
        best_M = []
        best_interval = e_root
        
        # Iteration tracking
        k = 0
        node_counter = 0
        
        # Logging
        search_tree = []
        iteration_numbers = []
        current_Q_history = []
        best_Q_history = []
        intermediate_csv = os.path.join(output_dir, "intermediate.csv")
        intermediate_data = []
        
        print("\n" + "="*70)
        print("BRANCH AND BOUND ALGORITHM STARTED")
        print("="*70)
        print(f"Root Upper Bound: {root_UB:.4f}")
        print(f"Initial Interval: {e_root}")
        print(f"Branching factor (s): {s}")
        print(f"Max iterations (n): {n}")
        print(f"Negative threshold: {neg_threshold}")
        print("="*70 + "\n")
        
        # Main Branch and Bound loop
        while PQ and k < s:  # Line 8: while PQ not empty and k < s
            k += 1
            
            # Line 9: Pop node with largest upper bound
            neg_UB_e, node_id, e = heapq.heappop(PQ)
            UB_e = -neg_UB_e
            
            print(f"\n{'='*70}")
            print(f"Iteration {k}: Exploring node {node_id}")
            print(f"  Upper Bound: {UB_e:.4f}")
            print(f"  Interval: {e}")
            print(f"  Best W so far: {best_w:.4f}")
            print(f"  Priority Queue size: {len(PQ)}")
            print(f"{'='*70}")
            
            # Line 10: Pruning condition
            if UB_e <= best_w:
                print(f"  ✂️  PRUNED: UB ({UB_e:.4f}) <= best_w ({best_w:.4f})")
                print(f"  Remaining nodes in queue cannot improve solution")
                continue
            
            # Line 12: Create hypothesis and compute exact Q
            h = Hypothesis(factors, e)
            q = compute_Q(h, "Background Knowledge", labelled_data, 
                         epsilon=0.1, theta_ext_h_approx=theta_ext_h_default)
            
            print(f"  Exact Q-score: {q:.4f}")
            
            # Line 14: Check if Q meets threshold (domain-specific cutoff)
            if q <= neg_threshold:
                print(f"  ❌ Q-score ({q:.4f}) below threshold ({neg_threshold})")
                print(f"  Node rejected, not generating molecules")
                continue
            
            # Line 15: Generate child intervals (BranchGenerate)
            print(f"\n  📊 Generating {s} child intervals...")
            children = generate_child_intervals(e, s, initial_intervals, seed + k)
            
            print(f"  Generated {len(children)} children:")
            for i, ec in enumerate(children):
                print(f"    Child {i+1}: {ec}")
            
            # Line 16-19: Evaluate each child
            promising_children = []
            for idx, ec in enumerate(children):
                node_counter += 1
                
                # Compute upper bound for child
                ubc = compute_upper_bound_optimistic(ec, labelled_data, factors, 
                                                      theta_ext_h_default, initial_intervals)
                
                print(f"\n  Child {idx+1} evaluation:")
                print(f"    Upper Bound: {ubc:.4f}")
                
                # Line 18: Only add to queue if ubc > best_w
                if ubc > best_w:
                    heapq.heappush(PQ, (-ubc, node_counter, ec))
                    promising_children.append((ubc, ec))
                    print(f"    ✅ Added to queue (UB > best_w)")
                else:
                    print(f"    ❌ Pruned (UB <= best_w)")
            
            # Line 22: Generate molecules using LMLFStar
            print(f"\n  🧪 Generating molecules for current interval...")
            parameter_ranges = {param: e[i] for i, param in enumerate(initial_intervals.keys())}
            
            if context:
                generate_molecules_for_protein_multifactors_with_context(
                    protein=protein,
                    input_csv=f"data/{protein}.txt",
                    output_dir=output_dir,
                    api_key=api_key,
                    model_engine=model_engine,
                    gnina_path=gnina_path,
                    config_path=config_path,
                    temp_dir=temp_dir,
                    parameter_ranges=parameter_ranges,
                    target_size=target_size,
                    max_iterations=1,
                    max_samples=max_samples
                )
            else:
                generate_molecules_for_protein_multifactors(
                    protein=protein,
                    input_csv=f"data/{protein}.txt",
                    output_dir=output_dir,
                    api_key=api_key,
                    model_engine=model_engine,
                    gnina_path=gnina_path,
                    config_path=config_path,
                    temp_dir=temp_dir,
                    parameter_ranges=parameter_ranges,
                    target_size=target_size,
                    max_iterations=1,
                    max_samples=max_samples
                )
            
            # Check if molecules were generated
            gen_csv = f"{output_dir}/generated.csv"
            M = []
            
            if os.path.exists(gen_csv) and os.path.getsize(gen_csv) > 0:
                properties_df = pd.read_csv(gen_csv)
                
                # Filter by current interval
                for param, bounds in parameter_ranges.items():
                    properties_df = properties_df[
                        (properties_df[param] >= bounds[0]) &
                        (properties_df[param] <= bounds[1])
                    ]
                
                if len(properties_df) > 0:
                    M = properties_df.to_dict(orient="records")
                    print(f"  ✅ Generated {len(M)} feasible molecules")
                else:
                    print(f"  ⚠️  No molecules passed interval filters")
            else:
                print(f"  ⚠️  No molecules generated")
            
            # Line 23: Compute w (weighted Q-score with molecule indicator)
            if M:
                w = q * 1.0  # Indicator(M != ∅) = 1
            else:
                w = q * 0.0  # Indicator(M != ∅) = 0
            
            print(f"  Weighted score (w): {w:.4f}")
            
            # Line 24-26: Update best solution if improved
            if w > best_w:
                print(f"  🎉 NEW BEST SOLUTION FOUND!")
                print(f"    Previous best: {best_w:.4f}")
                print(f"    New best: {w:.4f}")
                best_w = w
                best_M = M
                best_interval = e
                
                # Save intermediate results
                if M:
                    intermediate_data.extend(M)
                    interm_df = pd.DataFrame(intermediate_data).drop_duplicates()
                    
                    # Filter by best interval
                    for param, bounds in zip(initial_intervals.keys(), best_interval):
                        interm_df = interm_df[
                            (interm_df[param] >= bounds[0]) &
                            (interm_df[param] <= bounds[1])
                        ]
                    
                    intermediate_data = interm_df.to_dict(orient="records")
                    interm_df.to_csv(intermediate_csv, index=False)
                    print(f"    Saved {len(interm_df)} molecules to {intermediate_csv}")
            
            # Logging for visualization
            iteration_numbers.append(k)
            current_Q_history.append(q)
            best_Q_history.append(best_w)
            
            search_tree.append({
                "iteration": k,
                "node_id": node_id,
                "interval": e,
                "Q_score": q,
                "upper_bound": UB_e,
                "weighted_score": w,
                "molecules_found": len(M),
                "children": [{"interval": ec, "UB": ubc} for ubc, ec in promising_children]
            })
        
        # Algorithm complete
        print(f"\n{'='*70}")
        print("BRANCH AND BOUND ALGORITHM COMPLETED")
        print(f"{'='*70}")
        print(f"Total iterations: {k}")
        print(f"Nodes explored: {node_counter}")
        print(f"Final best W: {best_w:.4f}")
        print(f"Best interval: {best_interval}")
        print(f"Total feasible molecules: {len(best_M)}")
        print(f"{'='*70}\n")
        
        # Generate final molecules with best interval
        if best_interval:
            print("🎯 Generating final molecule set with optimal interval...")
            final_parameter_ranges = {param: best_interval[i] 
                                      for i, param in enumerate(initial_intervals)}
            
            if context:
                generate_molecules_for_protein_multifactors_with_context(
                    protein=protein,
                    input_csv=f"data/{protein}.txt",
                    output_dir=output_dir,
                    api_key=api_key,
                    model_engine=model_engine,
                    gnina_path=gnina_path,
                    config_path=config_path,
                    temp_dir=temp_dir,
                    parameter_ranges=final_parameter_ranges,
                    target_size=target_size,
                    max_iterations=1,
                    max_samples=final_k
                )
            else:
                generate_molecules_for_protein_multifactors(
                    protein=protein,
                    input_csv=f"data/{protein}.txt",
                    output_dir=output_dir,
                    api_key=api_key,
                    model_engine=model_engine,
                    gnina_path=gnina_path,
                    config_path=config_path,
                    temp_dir=temp_dir,
                    parameter_ranges=final_parameter_ranges,
                    target_size=target_size,
                    max_iterations=1,
                    max_samples=final_k
                )
        
        # Visualization
        if iteration_numbers:
            plt.figure(figsize=(12, 6))
            
            # Plot 1: Q-score progression
            plt.subplot(1, 2, 1)
            plt.plot(iteration_numbers, current_Q_history, marker='o', 
                    label='Current Q Score', color='blue')
            plt.plot(iteration_numbers, best_Q_history, marker='x', 
                    linestyle='--', label='Best W Score', color='red', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Score')
            plt.title('Branch & Bound: Score Progression')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Nodes explored
            plt.subplot(1, 2, 2)
            nodes_explored = list(range(1, len(iteration_numbers) + 1))
            plt.plot(nodes_explored, best_Q_history, marker='s', color='green')
            plt.xlabel('Nodes Explored')
            plt.ylabel('Best W Score')
            plt.title('Branch & Bound: Best Solution vs Nodes')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf_path = os.path.join(output_dir, "branch_bound_progress.pdf")
            plt.savefig(pdf_path, dpi=300)
            plt.close()
            print(f"📈 Visualization saved to: {pdf_path}")
        
        # Save detailed log
        log_lines = []
        log_lines.append("="*70)
        log_lines.append("BRANCH AND BOUND SEARCH TREE")
        log_lines.append("="*70)
        log_lines.append(f"Algorithm: Branch and Bound")
        log_lines.append(f"Branching factor (s): {s}")
        log_lines.append(f"Max iterations (n): {n}")
        log_lines.append(f"Negative threshold: {neg_threshold}")
        log_lines.append(f"Total nodes explored: {node_counter}")
        log_lines.append("")
        
        for node in search_tree:
            log_lines.append(f"Iteration {node['iteration']} (Node {node['node_id']}):")
            log_lines.append(f"  Interval: {node['interval']}")
            log_lines.append(f"  Upper Bound: {node['upper_bound']:.4f}")
            log_lines.append(f"  Exact Q-score: {node['Q_score']:.4f}")
            log_lines.append(f"  Weighted score (w): {node['weighted_score']:.4f}")
            log_lines.append(f"  Molecules found: {node['molecules_found']}")
            log_lines.append(f"  Children generated: {len(node['children'])}")
            for i, child in enumerate(node['children']):
                log_lines.append(f"    Child {i+1}: UB={child['UB']:.4f}, Interval={child['interval']}")
            log_lines.append("")
        
        log_lines.append("="*70)
        log_lines.append(f"FINAL RESULTS")
        log_lines.append("="*70)
        log_lines.append(f"Best weighted score: {best_w:.4f}")
        log_lines.append(f"Best interval: {best_interval}")
        log_lines.append(f"Total feasible molecules: {len(best_M)}")
        
        log_str = "\n".join(log_lines)
        log_file_path = os.path.join(output_dir, "branch_bound_log.txt")
        with open(log_file_path, "w") as log_file:
            log_file.write(log_str)
        print(f"📝 Detailed log saved to: {log_file_path}")

    # Initial intervals and params
    initial_intervals = {"CNNaffinity": [3, 10], "MolWt": [200, 700], "SAS": [0, 7.0]}
    
    # Branch and Bound specific parameters
    search_params = {
        "s": 4,              # Branching factor (number of children per node)
        "n": 10,             # Max iterations
        "max_samples": 2,    # Molecules per iteration
        "final_k": final_k,  # Final molecule count
        "context": context,
        "neg_threshold": -0.5  # Domain-specific Q-score cutoff
    }

    branch_and_bound_LMLFStar(
        protein=protein,
        labelled_data=labelled_data,
        unlabelled_data=unlabelled_data,
        initial_intervals=initial_intervals,
        api_key=api_key,
        model_engine=model_engine,
        gnina_path=gnina_path,
        config_path=config_path,
        temp_dir=temp_dir,
        output_dir=output_dir,
        s=search_params["s"],
        n=search_params["n"],
        max_samples=search_params["max_samples"],
        final_k=search_params["final_k"],
        target_size=target_size,
        context=search_params["context"],
        neg_threshold=search_params["neg_threshold"]
    )

    config_data = {
        "algorithm": "Branch and Bound",
        "protein": protein,
        "target_size": target_size,
        "context": context,
        "model_engine": model_engine,
        "search_intervals": initial_intervals,
        "search_params": search_params
    }
    config_file_path = os.path.join(output_dir, "config.json")
    with open(config_file_path, "w") as f:
        json.dump(config_data, f, indent=4)
    print(f"⚙️  Configuration saved to: {config_file_path}")
    print("\n✅ DONE [GenMolMF - Branch and Bound]")


# Helper Functions for Branch and Bound


def compute_upper_bound(interval, labelled_data, factors, theta_ext_h_approx, initial_intervals):
    """
    TP-based theoretical upper bound.
    
    Theory: Upper bound is the best Q-score achievable if we:
    1. Keep all current true positives (TP)
    2. Eliminate all false positives (FP → 0)
    3. Maximize precision while maintaining coverage
    
    This is optimistic but realistic based on actual data.
    
    Pros: Theoretically sound, uses actual TP/FP
    Cons: Slightly slower than naive
    """
    # Compute current hypothesis
    h = Hypothesis(factors, interval)
    q_actual = compute_Q(h, "Background Knowledge", labelled_data, 
                        epsilon=0.1, theta_ext_h_approx=theta_ext_h_approx)
    
    # Get examples covered by this interval
    covered_examples = []
    for example in labelled_data:
        if h.covers(example):
            covered_examples.append(example)
    
    # Compute confusion matrix
    TP = sum(1 for ex in covered_examples if ex.get('Label') == 1)
    FP = sum(1 for ex in covered_examples if ex.get('Label') == 0)
    
    total_positives = sum(1 for ex in labelled_data if ex.get('Label') == 1)
    total_negatives = sum(1 for ex in labelled_data if ex.get('Label') == 0)
    
    FN = total_positives - TP
    TN = total_negatives - FP
    
    # Optimistic scenario: What if we could eliminate all FP?
    # Best case: Keep all TP, remove all FP
    TP_optimistic = TP  # Keep current true positives
    FP_optimistic = 0   # Optimistically assume we can filter out all FP
    FN_optimistic = FN  # Same false negatives (we're not finding new positives)
    TN_optimistic = total_negatives  # All negatives correctly rejected
    
    # Compute optimistic Q-score
    # Using standard Q formula: Precision × Recall - α × FPR
    if TP_optimistic + FN_optimistic > 0:
        recall_opt = TP_optimistic / (TP_optimistic + FN_optimistic)
    else:
        recall_opt = 0.0
    
    if TP_optimistic + FP_optimistic > 0:
        precision_opt = TP_optimistic / (TP_optimistic + FP_optimistic)
    else:
        precision_opt = 1.0  # Perfect precision when no predictions
    
    if FP_optimistic + TN_optimistic > 0:
        fpr_opt = FP_optimistic / (FP_optimistic + TN_optimistic)
    else:
        fpr_opt = 0.0
    
    # Q_optimistic = F1-like score
    if precision_opt + recall_opt > 0:
        q_optimistic = 2 * (precision_opt * recall_opt) / (precision_opt + recall_opt)
    else:
        q_optimistic = 0.0
    
    # Penalty for false positive rate
    q_optimistic = q_optimistic - 0.5 * fpr_opt
    
    # Upper bound is the better of actual or optimistic
    upper_bound = max(q_actual, q_optimistic)
    
    # Add small bonus for larger intervals (potential to improve)
    interval_volume = 1.0
    for i, param in enumerate(initial_intervals.keys()):
        interval_size = interval[i][1] - interval[i][0]
        full_size = initial_intervals[param][1] - initial_intervals[param][0]
        if full_size > 0:
            interval_volume *= (interval_size / full_size)
    
    # Smaller intervals have less potential for improvement
    potential_bonus = 0.05 * interval_volume
    upper_bound = upper_bound + potential_bonus
    
    return upper_bound


def generate_child_intervals(parent_interval, n, initial_intervals, seed):
    """
    Generate n child intervals by splitting the parent interval.
    This is line 15 in the pseudocode: BranchGenerate(e, n)
    
    Uses Latin Hypercube Sampling to create diverse child intervals.
    """
    lhs_samples = scipy.stats.qmc.LatinHypercube(d=len(initial_intervals), seed=seed).random(n=n)
    children = []
    
    for sample in lhs_samples:
        new_intervals = []
        for i, param in enumerate(initial_intervals.keys()):
            parent_min, parent_max = parent_interval[i]
            
            # Split the parent interval
            # For CNNaffinity: keep max fixed, split min
            # For MolWt, SAS: keep min fixed, split max
            if param == "CNNaffinity":
                quantiles = np.linspace(parent_min, parent_max, n + 1)
                index = min(max(int(sample[i] * n), 0), n - 1)
                new_intervals.append([float(quantiles[index]), float(parent_max)])
            elif param in ["MolWt", "SAS"]:
                quantiles = np.linspace(parent_min, parent_max, n + 1)
                index = min(max(int(sample[i] * n), 0), n - 1)
                new_intervals.append([float(parent_min), float(quantiles[index + 1])])
            else:
                # Generic: split somewhere in the middle
                split_point = parent_min + sample[i] * (parent_max - parent_min)
                new_intervals.append([float(parent_min), float(split_point)])
        
        children.append(new_intervals)
    
    return children


# ================================
# Main: Parsing arguments and run
# ================================
def main():
    date_time = datetime.now().strftime("%d%m%y_%H%M")
    
    print("="*63)
    print(f"   TARGET-SPECIFIC LEAD DISCOVERY USING AN LLM [{date_time}]")
    print(f"   Algorithm: BRANCH AND BOUND")
    print("="*63)
    
    parser = argparse.ArgumentParser(
        description="TARGET-SPECIFIC LEAD DISCOVERY USING AN LLM (Branch & Bound)"
    )
    parser.add_argument("--choice", type=str, required=True,
                        help="Choice of pipeline: '1' (or '1f') for GenMol1F; '2' (or '1fplus') for GenMol1F with plus mode; '3' (or 'mf') for GenMolMF; '0' to abort")
    parser.add_argument("--protein", type=str, default="DBH", help="Target protein")
    parser.add_argument("--target_size", type=int, default=5, help="Target size for molecule generation")
    parser.add_argument("--context", type=str, default="False", help="Use context (True/False)")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Model engine to use")
    parser.add_argument("--final_k", type=int, default=20, help="Number of molecules to generate in the final step")
    args = parser.parse_args()
    
    context = args.context.lower() in ("true", "1", "yes")
    
    choice = args.choice.lower()
    print(args)

    if choice in ["1", "1f"]:
        print("Calling GenMol1F ...")
        GenMol1F(seed=0, 
                 protein=args.protein, 
                 target_size=args.target_size, 
                 final_k=args.final_k, 
                 context=context, 
                 model_engine=args.model)
    elif choice in ["2", "1fplus"]:        
        print("Calling GenMol1F with plus mode ...")
        GenMol1F(seed=0, 
                 protein=args.protein, 
                 target_size=args.target_size, 
                 final_k=args.final_k, 
                 context=context, 
                 model_engine=args.model,
                 plus_mode=True)
    elif choice in ["3", "mf"]:
        print("Calling GenMolMF (Branch and Bound) ...")
        GenMolMF(seed=0, 
                 protein=args.protein,
                 target_size=args.target_size, 
                 final_k=args.final_k, 
                 context=context, 
                 model_engine=args.model)
    else:
        print(f"Choice {args.choice} is invalid. Aborting...")
        return 1

if __name__ == "__main__":
    main()
