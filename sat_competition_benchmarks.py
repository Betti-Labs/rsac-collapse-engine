#!/usr/bin/env python3
"""
SAT Competition Benchmark Runner for RSAC
Downloads and tests RSAC against real-world SAT competition instances.
"""

import os
import requests
import gzip
import time
import argparse
from pathlib import Path
import subprocess
import tempfile

from src.rsac.sat import eval_clause
from src.rsac.bucket_search import bucket_search_no_oracle


def parse_dimacs_cnf(content):
    """Parse DIMACS CNF format."""
    lines = content.strip().split('\n')
    clauses = []
    n_vars = 0
    
    for line in lines:
        line = line.strip()
        if line.startswith('c'):  # comment
            continue
        elif line.startswith('p cnf'):  # problem line
            parts = line.split()
            n_vars = int(parts[2])
            n_clauses = int(parts[3])
        elif line and not line.startswith('%'):  # clause line
            clause = [int(x) for x in line.split() if x != '0']
            if clause:  # non-empty clause
                clauses.append(clause)
    
    return clauses, n_vars


def download_sat_instance(url, filename):
    """Download a SAT instance from URL."""
    print(f"Downloading {filename}...")
    response = requests.get(url)
    response.raise_for_status()
    
    # Handle gzipped files
    if filename.endswith('.gz'):
        content = gzip.decompress(response.content).decode('utf-8')
        filename = filename[:-3]  # remove .gz extension
    else:
        content = response.text
    
    with open(filename, 'w') as f:
        f.write(content)
    
    return filename


def get_small_sat_instances():
    """Get a curated list of small SAT competition instances suitable for RSAC."""
    # These are small instances from various SAT competitions
    # We focus on instances with <= 20 variables for RSAC feasibility
    instances = [
        {
            'name': 'aim-50-1_6-yes1-1.cnf',
            'url': 'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/AIM/aim-50-1_6-yes1-1.cnf',
            'description': 'AIM series - 50 vars, satisfiable'
        },
        {
            'name': 'aim-50-1_6-no-1.cnf', 
            'url': 'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/AIM/aim-50-1_6-no-1.cnf',
            'description': 'AIM series - 50 vars, unsatisfiable'
        },
        {
            'name': 'dubois20.cnf',
            'url': 'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/DUBOIS/dubois20.cnf',
            'description': 'Dubois series - 60 vars, unsatisfiable'
        }
    ]
    
    return instances


def create_small_test_instances():
    """Create some small test instances for RSAC evaluation."""
    instances_dir = Path('sat_instances')
    instances_dir.mkdir(exist_ok=True)
    
    # Create a few small handcrafted instances
    test_instances = []
    
    # Simple satisfiable 3-SAT
    with open(instances_dir / 'simple_sat_8.cnf', 'w') as f:
        f.write("""c Simple satisfiable 8-variable 3-SAT instance
p cnf 8 12
1 2 3 0
-1 4 5 0
-2 -4 6 0
-3 -5 -6 0
7 8 1 0
-7 2 -8 0
3 -1 4 0
-2 5 -3 0
6 -4 7 0
-5 -6 8 0
1 -7 -8 0
-1 2 3 0
""")
    test_instances.append({
        'name': 'simple_sat_8.cnf',
        'path': instances_dir / 'simple_sat_8.cnf',
        'description': 'Simple 8-var satisfiable instance'
    })
    
    # Pigeonhole principle (unsatisfiable)
    with open(instances_dir / 'pigeonhole_3_2.cnf', 'w') as f:
        f.write("""c Pigeonhole principle: 3 pigeons, 2 holes (unsatisfiable)
c Variables: x_ij means pigeon i goes in hole j
c x11, x12, x21, x22, x31, x32 = variables 1-6
p cnf 6 9
1 2 0
3 4 0  
5 6 0
-1 -3 0
-1 -5 0
-3 -5 0
-2 -4 0
-2 -6 0
-4 -6 0
""")
    test_instances.append({
        'name': 'pigeonhole_3_2.cnf',
        'path': instances_dir / 'pigeonhole_3_2.cnf', 
        'description': '3-pigeon 2-hole (unsatisfiable)'
    })
    
    return test_instances


def run_minisat(cnf_file, timeout=60):
    """Run MiniSat on a CNF file for comparison."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False) as tmp:
            result = subprocess.run(
                ['minisat', str(cnf_file), tmp.name],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Check if satisfiable
            if result.returncode == 10:  # SAT
                return 'SAT', result.stderr
            elif result.returncode == 20:  # UNSAT
                return 'UNSAT', result.stderr
            else:
                return 'UNKNOWN', result.stderr
                
    except subprocess.TimeoutExpired:
        return 'TIMEOUT', f'Timeout after {timeout}s'
    except FileNotFoundError:
        return 'NO_MINISAT', 'MiniSat not found'
    except Exception as e:
        return 'ERROR', str(e)


def run_rsac_on_instance(cnf_file, max_vars=20):
    """Run RSAC on a SAT instance."""
    print(f"\nTesting RSAC on {cnf_file}")
    
    # Parse the instance
    with open(cnf_file, 'r') as f:
        content = f.read()
    
    try:
        clauses, n_vars = parse_dimacs_cnf(content)
        print(f"  Variables: {n_vars}, Clauses: {len(clauses)}")
        
        if n_vars > max_vars:
            print(f"  SKIPPED: Too many variables ({n_vars} > {max_vars})")
            return None
        
        # Run RSAC
        start_time = time.time()
        result = bucket_search_no_oracle(clauses, n_vars, eval_clause)
        rsac_time = time.time() - start_time
        
        if result[0] is not None:
            print(f"  RSAC: SAT in {result[1]} checks ({rsac_time:.3f}s)")
            return {
                'instance': cnf_file.name,
                'n_vars': n_vars,
                'n_clauses': len(clauses),
                'rsac_result': 'SAT',
                'rsac_checks': result[1],
                'rsac_time': rsac_time
            }
        else:
            print(f"  RSAC: UNSAT after {result[1]} checks ({rsac_time:.3f}s)")
            return {
                'instance': cnf_file.name,
                'n_vars': n_vars,
                'n_clauses': len(clauses),
                'rsac_result': 'UNSAT',
                'rsac_checks': result[1],
                'rsac_time': rsac_time
            }
            
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def benchmark_comparison(instances, max_vars=20):
    """Run comprehensive benchmark comparison."""
    results = []
    
    print("="*60)
    print("SAT COMPETITION BENCHMARK COMPARISON")
    print("="*60)
    
    for instance_info in instances:
        if 'path' in instance_info:
            cnf_file = instance_info['path']
        else:
            # Download if needed
            cnf_file = Path('sat_instances') / instance_info['name']
            if not cnf_file.exists():
                cnf_file = download_sat_instance(instance_info['url'], str(cnf_file))
                cnf_file = Path(cnf_file)
        
        print(f"\n{'='*40}")
        print(f"Instance: {instance_info['name']}")
        print(f"Description: {instance_info['description']}")
        print(f"{'='*40}")
        
        # Run RSAC
        rsac_result = run_rsac_on_instance(cnf_file, max_vars)
        
        # Run MiniSat for comparison
        minisat_result, minisat_output = run_minisat(cnf_file)
        print(f"  MiniSat: {minisat_result}")
        
        if rsac_result:
            rsac_result['minisat_result'] = minisat_result
            results.append(rsac_result)
    
    return results


def print_summary(results):
    """Print benchmark summary."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    if not results:
        print("No results to summarize.")
        return
    
    print(f"{'Instance':<20} {'Vars':<5} {'RSAC':<8} {'Checks':<8} {'Time(s)':<8} {'MiniSat':<8}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['instance']:<20} {result['n_vars']:<5} {result['rsac_result']:<8} "
              f"{result['rsac_checks']:<8} {result['rsac_time']:<8.3f} {result['minisat_result']:<8}")
    
    # Statistics
    sat_instances = [r for r in results if r['rsac_result'] == 'SAT']
    if sat_instances:
        avg_checks = sum(r['rsac_checks'] for r in sat_instances) / len(sat_instances)
        avg_time = sum(r['rsac_time'] for r in sat_instances) / len(sat_instances)
        print(f"\nSAT Instances: {len(sat_instances)}")
        print(f"Average checks: {avg_checks:.1f}")
        print(f"Average time: {avg_time:.3f}s")


def main():
    parser = argparse.ArgumentParser(description='Run RSAC on SAT competition benchmarks')
    parser.add_argument('--max-vars', type=int, default=20, help='Maximum variables for RSAC')
    parser.add_argument('--download', action='store_true', help='Download external instances')
    parser.add_argument('--local-only', action='store_true', help='Only test local instances')
    
    args = parser.parse_args()
    
    # Create instances directory
    Path('sat_instances').mkdir(exist_ok=True)
    
    # Get test instances
    instances = create_small_test_instances()
    
    if args.download and not args.local_only:
        instances.extend(get_small_sat_instances())
    
    # Run benchmarks
    results = benchmark_comparison(instances, args.max_vars)
    
    # Print summary
    print_summary(results)
    
    # Save results
    import json
    with open('sat_competition_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to sat_competition_results.json")


if __name__ == '__main__':
    main()