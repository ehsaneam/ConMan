import random
import math
import matplotlib.pyplot as plt
from pathlib import Path
import copy
import torch
import re

# ==================== GENERATION FUNCTIONS ====================

def generate_app_a(intensity, samples=100):
    """CPU-intensive: high constant CPU, cache miss sinusoidal, mem BW load-store pattern"""
    scale = intensity / 5.0
    cpu_base = 20 + (80 * scale)
    cpu = [int(cpu_base)] * samples
    
    peak_miss = 10 * scale
    miss = []
    for i in range(samples):
        if i < 8:
            miss.append(int(peak_miss * i/7))
        else:
            sin_val = int(peak_miss + (peak_miss * 0.3) * math.sin((i-8) * math.pi / 6))
            miss.append(sin_val)
    
    mem_bw_val = int(5 + (15 * scale))
    mem = [mem_bw_val] * 8 + [0] * (samples - 13) + [mem_bw_val] * 5
    
    return cpu, miss, mem

def generate_app_b(intensity, samples=100):
    """Mem BW intensive: low random CPU/cache, square-wave mem BW"""
    scale = intensity / 5.0
    mem_high = int(20 + (70 * scale))
    
    cpu = [random.randint(2, int(5 + 10*scale)) for _ in range(samples)]
    cache = [random.randint(5, int(20 + 30*scale)) for _ in range(samples)]
    
    mem = []
    pattern = [mem_high]*14 + [0]*6
    while len(mem) < samples:
        mem.extend(pattern)
    mem = mem[:samples]
    
    return cpu, cache, mem

def generate_app_c(intensity, samples=100):
    """Cache miss heavy: CPU triangular, mem BW like A, cache miss very high"""
    scale = intensity / 5.0
    
    cpu_peak = int(30 + (40 * scale))
    cpu = []
    for i in range(samples):
        if (i % 20) < 10:
            cpu.append(int(cpu_peak * ((i % 20)/10)))
        else:
            cpu.append(int(cpu_peak * ((20 - (i % 20))/10)))
    
    miss_peak = int(30 + (15 * scale))
    miss = []
    for i in range(samples):
        if i < 8:
            miss.append(int(miss_peak * (i/8)))
        else:
            miss.append(int(miss_peak/2 + random.randint(int(miss_peak*0.7), int(miss_peak*1.3))/2))
    
    mem_bw_val = int(3 + (12 * scale))
    mem = [mem_bw_val] * 8 + [0] * (samples - 13) + [mem_bw_val] * 5
    
    return cpu, miss, mem

def generate_app_d(intensity, samples=100):
    """Balanced: CPU random, cache miss triangular+random, mem BW rectangular+random"""
    scale = intensity / 5.0
    
    cpu = [random.randint(int(10 + 10*scale), int(30 + 60*scale)) for _ in range(samples)]
    
    miss_peak = int(5 + (20 * scale))
    miss = []
    for i in range(samples):
        if i < 10:
            base = int(miss_peak * ((i%20)/10))
        else:
            base = int(miss_peak * ((20-(i%20))/10))
        miss.append(base + random.randint(-int(miss_peak*0.1), int(miss_peak*0.1)))
    
    mem_high = int(10 + (40 * scale))
    mem = []
    for i in range(samples):
        if i % 12 < 8:
            mem.append(mem_high + random.randint(-5, 5))
        else:
            mem.append(random.randint(0, 5))
    
    return cpu, miss, mem

def generate_app_e(intensity, samples=100):
    """Low activity: CPU low like B, mem BW like A, no cache miss"""
    scale = intensity / 5.0
    
    cpu = [random.randint(1, int(5 + 15*scale)) for _ in range(samples)]
    cache = [random.randint(0, 2) for _ in range(samples)]
    
    mem_bw_val = int(2 + (8 * scale))
    mem = [mem_bw_val] * 8 + [0] * (samples - 13) + [mem_bw_val] * 5
    
    return cpu, cache, mem

def generate_app_f(intensity, samples=100):
    """Spiky: CPU/cache spikes, mem BW rectangular"""
    scale = intensity / 5.0
    
    cpu = []
    for i in range(samples):
        if i % 5 == 3:
            cpu.append(random.randint(70, 100))
        else:
            cpu.append(random.randint(5, 20))
    
    cache = []
    for i in range(samples):
        if i % 5 == 3:
            cache.append(random.randint(50, 80))
        else:
            cache.append(random.randint(1, 3))
    
    mem_val = int(10 + (40 * scale))
    mem = [mem_val] * samples
    
    return cpu, cache, mem

def write_trace_file(filename):
    """Generate and write trace file"""
    apps = ['A', 'B', 'C', 'D', 'E', 'F']
    generators = [
        generate_app_a, generate_app_b, generate_app_c,
        generate_app_d, generate_app_e, generate_app_f
    ]
    
    with open(filename, 'w') as f:
        for app, gen in zip(apps, generators):
            for intensity in range(1, 6):
                cpu, cache, mem = gen(intensity)
                
                f.write(f"######################## application {app} - intensity {intensity} #######################\n")
                f.write(f"core cpu usage: {','.join(map(str, cpu))}\n")
                f.write(f"cache miss: {','.join(map(str, cache))}\n")
                f.write(f"mem bw: {','.join(map(str, mem))}\n")
                f.write("\n")
        
        f.write("##########################################################\n")
    
    print(f"Trace file '{filename}' generated successfully!")

def get_slowdown_factor(cpu_sum, mem_bw_sum, cache_miss_sum):
    """Calculate combined slowdown factor based on rules"""
    
    # CPU slowdown (only if exceeds 100%)
    if cpu_sum > 100:
        cpu_slowdown = cpu_sum / 100
    else:
        cpu_slowdown = 1.0
    
    # Memory bandwidth slowdown (only if exceeds 100, causes stall)
    if mem_bw_sum > 100:
        mem_slowdown = mem_bw_sum / 100
        mem_stall = True
    else:
        mem_slowdown = 1.0
        mem_stall = False
    
    # Cache miss slowdown (corrected step function)
    if cache_miss_sum > 80:
        cache_slowdown = 3.0
    elif cache_miss_sum > 60:
        cache_slowdown = 2.0
    elif cache_miss_sum > 40:
        cache_slowdown = 1.5
    elif cache_miss_sum > 10:
        cache_slowdown = 1.1
    else:
        cache_slowdown = 1.0
    
    # Product of all slowdown factors
    total_slowdown = cpu_slowdown * mem_slowdown * cache_slowdown
    
    return total_slowdown, mem_stall

def generate_colocated_traces(isolated_data, num_combinations, output_file):
    apps = ['A', 'B', 'C', 'D', 'E', 'F']
    intensities = [1, 2, 3, 4, 5]
    
    all_configs = []
    for app in apps:
        for intensity in intensities:
            all_configs.append(f"{app}{intensity}")
    
    combinations = []
    
    for _ in range(num_combinations):
        size = random.randint(2, 6)
        combo = random.sample(all_configs, size)
        combinations.append(combo)
    
    with open(output_file, 'w') as f:
        # Write isolated data first in colocated format
        for app in apps:
            for intensity in intensities:
                config_name = f"{app}{intensity}"
                trace = isolated_data[app][intensity]
                
                f.write(f"######################## {config_name} #######################\n")
                f.write(f"{config_name} +0 offset\n")
                f.write(f"core cpu usage: {','.join(map(str, trace['cpu']))}\n")
                f.write(f"cache miss: {','.join(map(str, trace['cache_miss']))}\n")
                f.write(f"mem bw: {','.join(map(str, trace['mem_bw']))}\n")
                f.write(f"**********************************************************************\n")
        
        # Generate and write colocated combinations
        for combo_idx, combo in enumerate(combinations):
            # Header for this combination
            combo_name = '-'.join(combo)
            f.write(f"######################## {combo_name} #######################\n")
            
            # Parse each config in the combination
            configs = []
            for config in combo:
                app = config[0]
                intensity = int(config[1])
                configs.append({
                    'name': config,
                    'app': app,
                    'intensity': intensity,
                    'trace': isolated_data[app][intensity],
                    'samples': len(isolated_data[app][intensity]['cpu']),
                    'delay': random.randint(0, 100),  # Random delay in samples (0.5s units)
                    'current_sample': 0,
                    'finished': False,
                    'mem_stalled': False
                })
            
            # Sort by delay and subtract minimum delay
            configs.sort(key=lambda x: x['delay'])
            min_delay = min([c['delay'] for c in configs])
            for config in configs:
                config['delay'] -= min_delay
            
            # Prepare result traces as lists
            result_traces = {}
            for config in configs:
                result_traces[config['name']] = {
                    'cpu': [],
                    'cache_miss': [],
                    'mem_bw': []
                }
            
            # Simulation variables
            time = 0
            all_finished = False
            
            # Continue until all applications finish
            while True:
                # Determine which apps are active at this time (started but not finished)
                active_apps = []
                inactive_apps = []
                for config in configs:
                    if not config['finished'] and time >= config['delay']:
                        active_apps.append(config)
                    else:
                        inactive_apps.append(config)
                
                # Calculate current metrics sum from active apps at their respective sample indices
                cpu_sum = 0
                mem_bw_sum = 0
                cache_miss_sum = 0
                
                app_states = []
                for app_state in active_apps:
                    sample_idx = app_state['current_sample']
                    if sample_idx < app_state['samples']:
                        cpu_val = app_state['trace']['cpu'][sample_idx]
                        mem_val = app_state['trace']['mem_bw'][sample_idx]
                        cache_val = app_state['trace']['cache_miss'][sample_idx]
                        
                        cpu_sum += cpu_val
                        mem_bw_sum += mem_val
                        cache_miss_sum += cache_val
                        
                        app_states.append({
                            'config': app_state,
                            'cpu_val': cpu_val,
                            'mem_val': mem_val,
                            'cache_val': cache_val
                        })
                
                # Calculate slowdown for this interval
                slowdown, mem_stall_global = get_slowdown_factor(cpu_sum, mem_bw_sum, cache_miss_sum)
                
                # Determine per-app mem stall status
                for state in app_states:
                    if mem_stall_global and state['mem_val'] > 0:
                        state['config']['mem_stalled'] = True
                    elif not mem_stall_global:
                        state['config']['mem_stalled'] = False
                
                # Apply slowdown to duration
                effective_duration = int(slowdown)
                
                # Record values for each time unit in this duration
                for _ in range(effective_duration):
                    # Record values for currently active apps only
                    for state in app_states:
                        config = state['config']
                        if config['mem_stalled']:
                            result_traces[config['name']]['cpu'].append(0)
                        else:
                            result_traces[config['name']]['cpu'].append(state['cpu_val'])
                        result_traces[config['name']]['cache_miss'].append(state['cache_val'])
                        result_traces[config['name']]['mem_bw'].append(state['mem_val'])
                
                for app_state in inactive_apps:
                    result_traces[app_state['name']]['cpu'].append(0)
                    result_traces[app_state['name']]['cache_miss'].append(0)
                    result_traces[app_state['name']]['mem_bw'].append(0)

                # Advance current_sample for active apps
                for app_state in configs:
                    if not app_state['finished']:
                        app_state['current_sample'] += 1
                        if app_state['current_sample'] >= app_state['samples']:
                            app_state['finished'] = True
                
                time += 1
                    
                # Check if all finished
                if all([c['finished'] for c in configs]):
                    all_finished = True
                    break
                
            # Ensure all traces have the same length
            max_length = max([len(result_traces[config['name']]['cpu']) for config in configs])
            for config in configs:
                while len(result_traces[config['name']]['cpu']) < max_length:
                    result_traces[config['name']]['cpu'].append(0)
                    result_traces[config['name']]['cache_miss'].append(0)
                    result_traces[config['name']]['mem_bw'].append(0)
            
            # Write traces for this combination
            for config in configs:
                name = config['name']
                trace = result_traces[name]
                delay = config['delay']
                delay_seconds = delay
                
                f.write(f"{name} +{delay_seconds} offset\n")
                f.write(f"core cpu usage: {','.join(map(str, trace['cpu']))}\n")
                f.write(f"cache miss: {','.join(map(str, trace['cache_miss']))}\n")
                f.write(f"mem bw: {','.join(map(str, trace['mem_bw']))}\n")
                f.write(f"**********************************************************************\n")
                
    print(f"Generated {num_combinations} colocated combinations in {output_file}")

def augment_trace(trace, max_shift):
    T, C = trace.shape

    zero_prefix = 0
    skip_start = 0
    zero_suffix = 0

    mode = random.choice(["cold", "late", "early", "combo"])

    if mode == "cold":
        zero_prefix = random.randint(1, max_shift)
        out = torch.cat([torch.zeros(zero_prefix, C), trace], dim=0)

    elif mode == "late":
        skip_start = random.randint(1, T - 1)
        out = trace[skip_start:]

    elif mode == "early":
        end = random.randint(1, T - 1)
        zero_suffix = T - end
        out = torch.cat([trace[:end], torch.zeros(zero_suffix, C)], dim=0)

    elif mode == "combo":
        zero_prefix = random.randint(1, max_shift // 2)
        skip_start = random.randint(1, T // 2)
        end = random.randint(skip_start + 1, T)
        zero_suffix = T - end

        if zero_prefix > 0:
            skip_start = 0

        out = torch.cat([
            torch.zeros(zero_prefix, C),
            trace[skip_start:end],
            torch.zeros(zero_suffix, C)
        ], dim=0)

    meta = {
        "zero_prefix": zero_prefix,
        "skip_start": skip_start,
        "zero_suffix": zero_suffix
    }

    return out, meta

def write_traces(blocks, out_path):
    with open(out_path, "w") as f:
        for blk in blocks:
            f.write(f"{blk['name']}\n")

            meta = blk.get("meta")
            if meta is not None:
                f.write(
                    f"@zero_prefix={meta['zero_prefix']} "
                    f"skip_start={meta['skip_start']} "
                    f"zero_suffix={meta['zero_suffix']}\n"
                )
            else:
                f.write("@zero_prefix=0 skip_start=0 zero_suffix=0\n")

            cpu = ",".join(f"{int(x)}" for x in blk["trace"][:, 0])
            cache = ",".join(f"{int(x)}" for x in blk["trace"][:, 1])
            mem = ",".join(f"{int(x)}" for x in blk["trace"][:, 2])

            f.write(f"core cpu usage: {cpu}\n")
            f.write(f"cache miss: {cache}\n")
            f.write(f"mem bw: {mem}\n")
            f.write("*" * 70 + "\n")

# ==================== READ FUNCTION ====================

def read_isolated_traces(filename):
    data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('######################## application'):
            parts = line.split()
            app = parts[2]  # 'A', 'B', etc.
            intensity = int(parts[5])  # 1-5
            
            cpu_line = lines[i+1].strip()
            cache_line = lines[i+2].strip()
            mem_line = lines[i+3].strip()
            
            cpu_values = [int(x) for x in cpu_line.split(':')[1].strip().split(',')]
            cache_values = [int(x) for x in cache_line.split(':')[1].strip().split(',')]
            mem_values = [int(x) for x in mem_line.split(':')[1].strip().split(',')]
            
            if app not in data:
                data[app] = {}
            data[app][intensity] = {
                'cpu': cpu_values,
                'cache_miss': cache_values,
                'mem_bw': mem_values
            }
            
            i += 4
        else:
            i += 1
    
    print(f"Loaded data for {len(data)} applications")
    return data

def read_colocated_traces(filename):
    """Read colocated trace file"""
    combinations = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('########################') and 'next combination' not in line:
            # Extract combination name
            combo_name = line.replace('########################', '').strip().replace('#######################', '').strip()
            
            apps_data = []
            
            # Read until next combination or end
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('########################'):
                if lines[i].strip() and ('+' in lines[i].strip() or 'offset' in lines[i].strip()):
                    # Parse app name and delay from format "A5 +2.0s offset"
                    app_line = lines[i].strip()
                    app_name = app_line.split('+')[0].strip()
                    delay = float(app_line.split('+')[1].split('s')[0].strip())
                    
                    cpu_line = lines[i+1].strip()
                    cache_line = lines[i+2].strip()
                    mem_line = lines[i+3].strip()
                    
                    cpu_values = [int(x) for x in cpu_line.split(':')[1].strip().split(',')]
                    cache_values = [int(x) for x in cache_line.split(':')[1].strip().split(',')]
                    mem_values = [int(x) for x in mem_line.split(':')[1].strip().split(',')]
                    
                    apps_data.append({
                        'app': app_name,
                        'delay': delay,
                        'cpu': cpu_values,
                        'cache_miss': cache_values,
                        'mem_bw': mem_values
                    })
                    
                    i += 5  # Skip the separator line
                else:
                    i += 1
            
            combinations.append({
                'combination': combo_name,
                'applications': apps_data
            })
        else:
            i += 1
    
    print(f"Loaded {len(combinations)} colocated combinations")
    return combinations

def parse_traces_txt(path):
    blocks = []
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("*"):
            i += 1
            continue

        # workload name (single token)
        wname = line

        cpu = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", lines[i + 1])))
        cache = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", lines[i + 2])))
        mem = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", lines[i + 3])))

        T = min(len(cpu), len(cache), len(mem))
        trace = torch.tensor(
            list(zip(cpu[:T], cache[:T], mem[:T])),
            dtype=torch.float32
        )

        blocks.append({
            "name": wname,
            "trace": trace
        })

        i += 4

    return blocks

# ==================== PLOT FUNCTIONS ====================

def plot_application(data, app_name, intensity=None, save_path=None):
    """
    Plot metrics for a specific application and intensity.
    
    Args:
        data: dict from read_trace_file()
        app_name: 'A', 'B', etc.
        intensity: 1-5, or None to plot all intensities
        save_path: if provided, save figure to this path
    """
    if app_name not in data:
        print(f"Application {app_name} not found in data")
        return
    
    if intensity is not None:
        # Plot single intensity
        if intensity not in data[app_name]:
            print(f"Intensity {intensity} not found for app {app_name}")
            return
        
        metrics = data[app_name][intensity]
        time_points = [i * 0.5 for i in range(len(metrics['cpu']))]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].plot(time_points, metrics['cpu'], 'b-', linewidth=2)
        axes[0].set_ylabel('CPU Usage (%)')
        axes[0].set_title(f'Application {app_name} - Intensity {intensity}')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(time_points, metrics['cache_miss'], 'r-', linewidth=2)
        axes[1].set_ylabel('Cache Miss (count)')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(time_points, metrics['mem_bw'], 'g-', linewidth=2)
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Mem BW (GB/s)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    else:
        # Plot all intensities on same figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        
        for intensity in range(1, 6):
            if intensity in data[app_name]:
                metrics = data[app_name][intensity]
                time_points = [i * 0.5 for i in range(len(metrics['cpu']))]
                
                axes[0].plot(time_points, metrics['cpu'], color=colors[intensity-1], 
                           label=f'Intensity {intensity}', linewidth=1.5)
                axes[1].plot(time_points, metrics['cache_miss'], color=colors[intensity-1], 
                           label=f'Intensity {intensity}', linewidth=1.5)
                axes[2].plot(time_points, metrics['mem_bw'], color=colors[intensity-1], 
                           label=f'Intensity {intensity}', linewidth=1.5)
        
        axes[0].set_ylabel('CPU Usage (%)')
        axes[0].set_title(f'Application {app_name} - All Intensities')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_ylabel('Cache Miss (count)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Mem BW (GB/s)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


def plot_all_applications(data, intensity=3, save_path=None):
    """
    Plot all applications at a specific intensity for comparison.
    
    Args:
        data: dict from read_trace_file()
        intensity: 1-5
        save_path: if provided, save figure to this path
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    apps = ['A', 'B', 'C', 'D', 'E', 'F']
    
    for idx, app in enumerate(apps):
        row, col = idx // 3, idx % 3
        
        if app in data and intensity in data[app]:
            metrics = data[app][intensity]
            time_points = [i * 0.5 for i in range(len(metrics['cpu']))]
            
            axes[row, col].plot(time_points, metrics['cpu'], 'b-', label='CPU', alpha=0.7)
            axes[row, col].plot(time_points, metrics['cache_miss'], 'r-', label='Cache Miss', alpha=0.7)
            axes[row, col].plot(time_points, metrics['mem_bw'], 'g-', label='Mem BW', alpha=0.7)
            axes[row, col].set_title(f'App {app} - Intensity {intensity}')
            axes[row, col].set_xlabel('Time (s)')
            axes[row, col].legend(fontsize=8)
            axes[row, col].grid(True, alpha=0.3)
        else:
            axes[row, col].text(0.5, 0.5, f'No data for App {app}', 
                               ha='center', va='center')
            axes[row, col].set_title(f'App {app}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_metric_comparison(data, metric='cpu', intensity=3, save_path=None):
    """
    Compare a specific metric across all applications.
    
    Args:
        data: dict from read_trace_file()
        metric: 'cpu', 'cache_miss', or 'mem_bw'
        intensity: 1-5
        save_path: if provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    apps = ['A', 'B', 'C', 'D', 'E', 'F']
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    
    for app, color in zip(apps, colors):
        if app in data and intensity in data[app]:
            values = data[app][intensity][metric]
            time_points = [i * 0.5 for i in range(len(values))]
            ax.plot(time_points, values, color=color, label=f'App {app}', linewidth=1.5)
    
    metric_labels = {'cpu': 'CPU Usage (%)', 'cache_miss': 'Cache Miss (count)', 'mem_bw': 'Mem BW (GB/s)'}
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.set_title(f'{metric_labels.get(metric, metric)} Comparison - Intensity {intensity}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_colocated_combination(combination_data, save_path=None):
    """Plot a single colocated combination"""
    combo_name = combination_data['combination']
    apps = combination_data['applications']
    
    num_apps = len(apps)
    fig, axes = plt.subplots(num_apps, 3, figsize=(15, 3*num_apps))
    
    if num_apps == 1:
        axes = axes.reshape(1, -1)
    
    for idx, app in enumerate(apps):
        time_points = [i * 0.5 for i in range(len(app['cpu']))]
        
        axes[idx, 0].plot(time_points, app['cpu'], 'b-', linewidth=1)
        axes[idx, 0].set_ylabel('CPU (%)')
        axes[idx, 0].set_title(f'{app["app"]} - CPU')
        axes[idx, 0].grid(True, alpha=0.3)
        
        axes[idx, 1].plot(time_points, app['cache_miss'], 'r-', linewidth=1)
        axes[idx, 1].set_ylabel('Cache Miss')
        axes[idx, 1].set_title(f'{app["app"]} - Cache Miss')
        axes[idx, 1].grid(True, alpha=0.3)
        
        axes[idx, 2].plot(time_points, app['mem_bw'], 'g-', linewidth=1)
        axes[idx, 2].set_ylabel('Mem BW')
        axes[idx, 2].set_title(f'{app["app"]} - Mem BW')
        axes[idx, 2].grid(True, alpha=0.3)
        
        if idx == num_apps - 1:
            axes[idx, 0].set_xlabel('Time (s)')
            axes[idx, 1].set_xlabel('Time (s)')
            axes[idx, 2].set_xlabel('Time (s)')
    
    plt.suptitle(f'Colocated Combination: {combo_name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# ==================== MAIN EXAMPLE ====================

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate trace file
    # write_trace_file("../traces/traces.txt")
    
    # Read the trace file
    isolated_data = read_isolated_traces("traces/traces.txt")

    # Generate colocated combinations
    generate_colocated_traces(isolated_data, num_combinations=100, output_file="traces/colocated_traces.txt")

    # Read colocated traces
    # colocated_data = read_colocated_traces("../traces/colocated_traces.txt")
    
    # Example plots:
    
    # 1. Plot single application at specific intensity
    # plot_application(trace_data, app_name='A', intensity=3)
    
    # 2. Plot single application with all intensities
    # plot_application(trace_data, app_name='C', intensity=None)
    
    # 3. Plot all applications at intensity 3
    # plot_all_applications(trace_data, intensity=3)
    
    # 4. Compare CPU usage across all apps at intensity 4
    # plot_metric_comparison(trace_data, metric='cpu', intensity=4)
    
    # 5. Compare cache miss across all apps at intensity 5
    # plot_metric_comparison(trace_data, metric='cache_miss', intensity=5)
    
    # To save plots instead of showing:
    # plot_application(trace_data, app_name='A', intensity=3, save_path="app_A_int3.png")

    # inp = Path("traces/traces.txt")
    # out = Path("traces/traces_augmented.txt")
    # originals = parse_traces_txt(inp)

    # augmented = []

    # AUG_PER_TRACE = 10
    # MAX_SHIFT = 100

    # for blk in originals:
    #     augmented.append(blk)  # keep original

    #     for k in range(AUG_PER_TRACE):
    #         aug_trace, meta = augment_trace(blk["trace"], MAX_SHIFT)
    #         augmented.append({
    #             "name": f"{blk['name']}",
    #             "trace": aug_trace,
    #             "meta": meta
    #         })

    # write_traces(augmented, out)

    # print(f"Original workloads: {len(originals)}")
    # print(f"Total after augmentation: {len(augmented)}")