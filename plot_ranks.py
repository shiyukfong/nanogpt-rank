import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def parse_and_plot_ranks(log_filename, output_filename):
    # --- Configuration ---
    # Define the expected stages (Chunks)
    chunk_labels = ["Step 0 (Init)", "Step 1%", "Step 25%", "Step 50%", "Step 75%", "Step 100%"]
    
    # Structure of the log info
    layers_per_pass = 12
    components_per_layer = 4
    lines_per_chunk = layers_per_pass * components_per_layer # 48 lines

    # --- 1. Read and Parse Log File ---
    data = []
    pattern = re.compile(r'\[(\d+)\.(.*?)\]\s+Effective rank.*?:\s+(\d+)/(\d+)', re.IGNORECASE)
    
    print(f"Reading {log_filename}...")
    try:
        with open(log_filename, 'r') as f:
            # Filter only relevant lines to avoid noise
            lines = [line for line in f if "Effective rank" in line]
    except FileNotFoundError:
        print(f"Error: Log file '{log_filename}' not found.")
        return
        

    print(f"Found {len(lines)} rank entries.")
    
    for i, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            layer = int(match.group(1))
            component = match.group(2)
            rank_val = int(match.group(3))
            dim = int(match.group(4))
            
            # Determine which chunk (stage) this line belongs to
            chunk_idx = i // lines_per_chunk
            
            if chunk_idx < len(chunk_labels):
                stage_name = chunk_labels[chunk_idx]
            else:
                stage_name = f"Unknown Chunk {chunk_idx}"

            data.append({
                'Stage': stage_name,
                'StageIdx': chunk_idx,
                'Layer': layer,
                'Component': component,
                'Rank (%)': rank_val / dim * 100,  # Convert to percentage
            })

    df = pd.DataFrame(data)

    # --- 2. Aggregate Data (Average over 10 passes) ---
    # We group by Stage, Layer, and Component, then take the mean of the Rank
    df_avg = df.groupby(['Stage', 'StageIdx', 'Layer', 'Component'])['Rank (%)'].mean().reset_index()

    # --- 3. Plotting ---
    # Setup a 2x3 grid for 6 chunks
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True, sharex=True)
    axes = axes.flatten()
    
    components = sorted(df_avg['Component'].unique())
    # Define consistent styles
    styles = {
        'attn.c_attn': {'color': 'blue', 'marker': 'o', 'label': 'attn.c_attn'},
        'attn.c_proj': {'color': 'orange', 'marker': '^', 'label': 'attn.c_proj'},
        'mlp.c_fc':    {'color': 'green', 'marker': 's', 'label': 'mlp.c_fc'},
        'mlp.c_proj':  {'color': 'red', 'marker': 'D', 'label': 'mlp.c_proj'}
    }

    # Iterate through the 6 expected stages
    for i, stage in enumerate(chunk_labels):
        ax = axes[i]
        
        # Check if we actually have data for this stage
        if i in df_avg['StageIdx'].unique():
            subset = df_avg[df_avg['StageIdx'] == i]
            
            for comp in components:
                comp_data = subset[subset['Component'] == comp].sort_values('Layer')
                style = styles.get(comp, {})
                ax.plot(
                    comp_data['Layer'], 
                    comp_data['Rank (%)'], 
                    color=style.get('color'), 
                    marker=style.get('marker'), 
                    label=style.get('label'),
                    linewidth=2, 
                    markersize=5,
                    alpha=0.8
                )
            ax.set_title(stage, fontsize=20, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)
        else:
            ax.set_title(f"{stage}\n(No Data)", fontsize=20, color='gray', fontweight='bold')
            ax.text(0.5, 0.5, 'Data Not Available', ha='center', va='center', transform=ax.transAxes, color='gray')

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(16)
            label.set_fontweight('bold')

        # Axis labels
        if i >= 3: # Bottom row
            ax.set_xlabel("Layer Depth", fontsize=20, fontweight='bold')
        if i % 3 == 0: # Left column
            ax.set_ylabel("Avg Effective Rank (%)", fontsize=18, fontweight='bold')

    # Legend (only need one for the whole figure)
    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
    if all_handles:
        seen, uniq_handles, uniq_labels = set(), [], []
        for h, l in zip(all_handles, all_labels):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)
        fig.legend(uniq_handles, uniq_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, prop={'size': 18, 'weight': 'bold'}, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Parse the name of the log file to plot from command line")
    args.add_argument("log_file", type=str, nargs='?', default="log.log", help="Path to the log file containing the rank information")
    args.add_argument("output_file", type=str, nargs='?', default="nanogpt_ranks.pdf", help="Path to the output file for the plot (e.g. plot_ranks.pdf)")
    parsed_args = args.parse_args()
    parse_and_plot_ranks(parsed_args.log_file, parsed_args.output_file)