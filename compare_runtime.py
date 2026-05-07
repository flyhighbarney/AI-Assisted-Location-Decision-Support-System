"""
Runtime Comparison Tool for migration_script.py and migration_v2.py
Place this file in the same directory as your migration scripts and run it.
"""

import time
import subprocess
import sys

def run_script_and_measure(script_name, iterations=5):
    """
    Run a Python script multiple times and measure its execution time.
    
    Args:
        script_name: Name of the Python script to run
        iterations: Number of times to run the script
    
    Returns:
        List of execution times in milliseconds
    """
    print(f"\nTesting {script_name}...")
    times = []
    
    for i in range(iterations):
        start_time = time.perf_counter()
        
        # Run the script as a subprocess
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True
        )
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        
        if result.returncode == 0:
            times.append(elapsed_ms)
            print(f"  Run {i + 1}: {elapsed_ms:.2f} ms")
        else:
            print(f"  Run {i + 1}: Failed with error")
            print(f"  Error: {result.stderr}")
            return None
    
    return times

def calculate_stats(times):
    """Calculate statistics from timing data."""
    if not times:
        return None
    
    avg = sum(times) / len(times)
    minimum = min(times)
    maximum = max(times)
    
    # Calculate standard deviation
    if len(times) > 1:
        variance = sum((x - avg) ** 2 for x in times) / len(times)
        std_dev = variance ** 0.5
    else:
        std_dev = 0
    
    return {
        'average': avg,
        'min': minimum,
        'max': maximum,
        'std_dev': std_dev
    }

def main():
    """Main comparison function."""
    print("=" * 70)
    print("MIGRATION SCRIPT RUNTIME COMPARISON")
    print("=" * 70)
    
    script1 = "migration_script.py"
    script2 = "migration_v2.py"
    iterations = 5
    
    print(f"\nRunning each script {iterations} times to measure performance...\n")
    
    # Run first script
    times1 = run_script_and_measure(script1, iterations)
    if times1 is None:
        print(f"\nFailed to run {script1}. Please check for errors.")
        return
    
    # Run second script
    times2 = run_script_and_measure(script2, iterations)
    if times2 is None:
        print(f"\nFailed to run {script2}. Please check for errors.")
        return
    
    # Calculate statistics
    stats1 = calculate_stats(times1)
    stats2 = calculate_stats(times2)
    
    # Display results
    print("\n" + "=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)
    
    print(f"\n{script1}:")
    print(f"  Average Time: {stats1['average']:.2f} ms")
    print(f"  Min Time:     {stats1['min']:.2f} ms")
    print(f"  Max Time:     {stats1['max']:.2f} ms")
    print(f"  Std Dev:      {stats1['std_dev']:.2f} ms")
    
    print(f"\n{script2}:")
    print(f"  Average Time: {stats2['average']:.2f} ms")
    print(f"  Min Time:     {stats2['min']:.2f} ms")
    print(f"  Max Time:     {stats2['max']:.2f} ms")
    print(f"  Std Dev:      {stats2['std_dev']:.2f} ms")
    
    # Determine which is faster
    print("\n" + "-" * 70)
    print("WINNER")
    print("-" * 70)
    
    avg1 = stats1['average']
    avg2 = stats2['average']
    
    if avg1 < avg2:
        improvement = ((avg2 - avg1) / avg2) * 100
        print(f"\n✓ {script1} is FASTER by {improvement:.2f}%")
        print(f"  Time saved: {avg2 - avg1:.2f} ms per execution")
    elif avg2 < avg1:
        improvement = ((avg1 - avg2) / avg1) * 100
        print(f"\n✓ {script2} is FASTER by {improvement:.2f}%")
        print(f"  Time saved: {avg1 - avg2:.2f} ms per execution")
    else:
        print("\n≈ Both scripts have similar performance")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
