#!/usr/bin/env python3
"""
Plot improvement curves to compare solution discovery speed of different methods.
Usage:
    python plot_improvements.py <file1> [<file2> ...] [--output <output_file>] [--title <title>]
    python plot_improvements.py --pattern "tai125e01*improvements.txt" --output comparison.png
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import glob
import os
from pathlib import Path

def parse_improvement_file(filename):
    """Parse improvement file and return (times, objectives) arrays."""
    times = []
    objectives = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        time = float(parts[0])
                        obj = float(parts[1])
                        times.append(time)
                        objectives.append(obj)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Warning: File {filename} not found, skipping...")
        return None, None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None
    
    if len(times) == 0:
        print(f"Warning: No data found in {filename}")
        return None, None
    
    return np.array(times), np.array(objectives)

def get_method_name(filename):
    """Extract method name from filename."""
    basename = os.path.basename(filename)
    # Remove common suffixes
    if 'rots' in basename:
        return 'Ro-TS'
    if 'bma' in basename:
        return 'BMA'
    # Default: use filename without extension
    return os.path.splitext(basename)[0]

def plot_improvements(files, output_file=None, title=None, normalize=False, 
                     log_scale=False, xlim=None, ylim=None):
    """Plot improvement curves for multiple files."""
    
    if len(files) == 0:
        print("Error: No files provided")
        return
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(files)))
    linestyles = ['-', '--', '-.', ':']

    for idx, filename in enumerate(files):
        times, objectives = parse_improvement_file(filename)
        
        if times is None or objectives is None:
            continue
        
        method_name = get_method_name(filename)
        title = filename.split('_')[0]
        
        # Normalize objectives if requested (normalize to [0, 1] range)
        if normalize:
            if len(objectives) > 0:
                min_obj = objectives.min()
                max_obj = objectives.max()
                if max_obj > min_obj:
                    objectives = (objectives - min_obj) / (max_obj - min_obj)
                else:
                    objectives = np.zeros_like(objectives)
        
        # Plot the curve
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        
        plt.plot(times, objectives, label=method_name, 
                color=color, linestyle=linestyle, linewidth=2, marker='o', 
                markersize=4, markevery=max(1, len(times)//20))
    
    plt.xlabel('Time (seconds)', fontsize=12)
    if normalize:
        plt.ylabel('Normalized Objective Value', fontsize=12)
    else:
        plt.ylabel('Objective Value', fontsize=12)
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title('Solution Discovery Speed Comparison', fontsize=14, fontweight='bold')
    
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Plot improvement curves to compare solution discovery speed',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific files
  python plot_improvements.py tai125e01_improvements.txt tai125e01_bma_improvements.txt
  
  # Compare all improvement files for a specific instance
  python plot_improvements.py --pattern "tai125e01*improvements.txt" --output comparison.png
  
  # Compare with normalized objectives
  python plot_improvements.py file1.txt file2.txt --normalize
        """
    )
    
    parser.add_argument('files', nargs='*', help='Improvement files to plot')
    parser.add_argument('--pattern', '-p', help='Glob pattern to match files (e.g., "*improvements.txt")')
    parser.add_argument('--output', '-o', help='Output file name (e.g., comparison.png)')
    parser.add_argument('--title', '-t', help='Plot title')
    parser.add_argument('--normalize', '-n', action='store_true', 
                       help='Normalize objective values to [0,1] range for comparison')
    parser.add_argument('--log', action='store_true', 
                       help='Use logarithmic scale for y-axis')
    parser.add_argument('--xlim', nargs=2, type=float, 
                       help='X-axis limits (e.g., --xlim 0 100)')
    parser.add_argument('--ylim', nargs=2, type=float, 
                       help='Y-axis limits (e.g., --ylim 0 50000)')
    
    args = parser.parse_args()
    
    files = []
    
    # Collect files from arguments
    if args.files:
        files.extend(args.files)
    
    # Collect files from pattern
    if args.pattern:
        matched_files = glob.glob(args.pattern)
        files.extend(matched_files)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    if len(unique_files) == 0:
        print("Error: No files found. Please provide files or use --pattern")
        parser.print_help()
        sys.exit(1)
    
    print(f"Plotting {len(unique_files)} file(s):")
    for f in unique_files:
        print(f"  - {f}")
    
    plot_improvements(unique_files, 
                     output_file=args.output,
                     title=args.title,
                     normalize=args.normalize,
                     log_scale=args.log,
                     xlim=args.xlim,
                     ylim=args.ylim)

if __name__ == '__main__':
    main()
