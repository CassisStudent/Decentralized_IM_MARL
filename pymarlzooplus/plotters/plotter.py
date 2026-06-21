import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 1. Zoek automatisch naar alle tfevents (.th) bestanden in de 'runs/' map
log_dir = "../results/tb_logs/"  # Pas dit aan naar de exacte map waar je logs worden weggeschreven
event_files = glob.glob(os.path.join(log_dir, "**/events.out.tfevents.*"), recursive=True)

if not event_files:
    print(f"Geen TensorBoard event-bestanden gevonden in: {log_dir}")
    print("Controleer of je training al gestart is en logs aanmaakt.")
    exit()

print(f"Gevonden logbestanden ({len(event_files)} stuks):")
for f in event_files:
    print(f" - {f}")

# We pakken de meest recente of de eerste file om uit te lezen
target_file = event_files[3]
print(f"\nBezig met uitlezen van: {target_file}")

# Laad de binaire data in (we zetten de limiet hoog om alle data te pakken)
ea = event_accumulator.EventAccumulator(target_file, size_guidance={event_accumulator.SCALARS: 0})
ea.Reload()

# Laat zien welke metrieken we succesvol hebben opgevangen
available_tags = ea.Tags()['scalars']
print("Gevonden metrics in bestand:", available_tags)

agents_found = sorted(list(set([t.split('/')[0] for t in available_tags if t.startswith("Agent_")])))
print(f"Gevonden agents in logfile: {agents_found}")

# Functie om een specifieke metric om te zetten naar een Pandas DataFrame
def extract_metric(tag_name):
    if tag_name in available_tags:
        events = ea.Scalars(tag_name)
        return pd.DataFrame({'step': [e.step for e in events], 'value': [e.value for e in events]})
    return None

# 2. Plot de resultaten in een 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

colors = plt.cm.tab10.colors


# Loop door alle gevonden agents heen en plot ze over elkaar heen in de subplots
for idx, agent_prefix in enumerate(agents_found):
    color = colors[idx % len(colors)]
    
    # Grafiek 1: World Model Loss per Agent
    df_wm = extract_metric(f'{agent_prefix}/World_Model_Loss')
    if df_wm is not None:
        axs[0, 0].plot(df_wm['step'], df_wm['value'], color=color, alpha=0.8, linewidth=1.5, label=agent_prefix)
    
    # Grafiek 2: Intrinsieke Reward Mean per Agent
    df_ir = extract_metric(f'{agent_prefix}/Intrinsic_Reward_Raw_Mean')
    if df_ir is not None:
        axs[0, 1].plot(df_ir['step'], df_ir['value'], color=color, alpha=0.8, linewidth=1.5, label=agent_prefix)
        
    # Grafiek 3: Policy Entropy per Agent
    df_ent = extract_metric(f'{agent_prefix}/Entropy')
    if df_ent is not None:
        axs[1, 0].plot(df_ent['step'], df_ent['value'], color=color, alpha=0.8, linewidth=1.5, label=agent_prefix)
        
    # Grafiek 4: Actor Loss (doorgetrokken lijn) & Critic Loss (gestreepte lijn) per Agent
    df_actor = extract_metric(f'{agent_prefix}/Actor_Loss')
    df_critic = extract_metric(f'{agent_prefix}/Critic_Loss')
    if df_actor is not None:
        axs[1, 1].plot(df_actor['step'], df_actor['value'], color=color, linestyle='-', alpha=0.7, label=f'{agent_prefix} Actor')
    if df_critic is not None:
        axs[1, 1].plot(df_critic['step'], df_critic['value'], color=color, linestyle='--', alpha=0.5, label=f'{agent_prefix} Critic')

# --- OPMAAK AS-LABELS EN LEGENDAS ---
axs[0, 0].set_title('World Model Loss (MSE)', fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel('Tijdstappen')
axs[0, 0].grid(True, linestyle='--', alpha=0.5)
axs[0, 0].legend()

axs[0, 1].set_title('Intrinsic Reward Raw Mean (Variance)', fontsize=12, fontweight='bold')
axs[0, 1].set_xlabel('Tijdstappen')
axs[0, 1].grid(True, linestyle='--', alpha=0.5)
axs[0, 1].legend()

axs[1, 0].set_title('PPO Policy Entropy', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('Tijdstappen')
axs[1, 0].grid(True, linestyle='--', alpha=0.5)
axs[1, 0].legend()

axs[1, 1].set_title('PPO Network Losses', fontsize=12, fontweight='bold')
axs[1, 1].set_xlabel('Tijdstappen')
axs[1, 1].grid(True, linestyle='--', alpha=0.5)
axs[1, 1].legend(ncol=2, fontsize='small') # Kolommen splitsen vanwege de hoeveelheid lijnen

plt.tight_layout()
plt.savefig("marl_all_agents_progress.png", dpi=300)
print("\nSucces! De gecombineerde grafieken zijn opgeslagen als 'marl_all_agents_progress.png'.")
    
    
    
    
    
    
    
    
    
    
    
"""
# Grafiek 1: World Model Loss
df_wm = extract_metric('Agent_0/World_Model_Loss')
if df_wm is not None:
    axs[0, 0].plot(df_wm['step'], df_wm['value'], color='tab:red', linewidth=1.5)
    axs[0, 0].set_title('World Model Loss (MSE)')
    axs[0, 0].set_xlabel('Tijdstappen')
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
else:
    print("Could not find Exploration/World_Model_Loss")

# Grafiek 2: Intrinsieke Reward Mean
df_ir = extract_metric('Agent_0/Intrinsic_Reward_Raw_Mean')
if df_ir is not None:
    axs[0, 1].plot(df_ir['step'], df_ir['value'], color='tab:orange', linewidth=1.5)
    axs[0, 1].set_title('Intrinsic Reward Raw Mean')
    axs[0, 1].set_xlabel('Tijdstappen')
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
else:
    print("Could not find Exploration/Intrinsic_Reward_Raw_Mean")

# Grafiek 3: Policy Entropy
df_ent = extract_metric('Agent_0/Entropy')
if df_ent is not None:
    axs[1, 0].plot(df_ent['step'], df_ent['value'], color='tab:purple', linewidth=1.5)
    axs[1, 0].set_title('PPO Policy Entropy')
    axs[1, 0].set_xlabel('Tijdstappen')
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
else:
    print("could not find PPO/Policy_Entropy")

# Grafiek 4: Actor & Critic Loss gecombineerd
df_actor = extract_metric('Agent_0/Actor_Loss')
df_critic = extract_metric('Agent_0/Critic_Loss')
if df_actor is not None:
    axs[1, 1].plot(df_actor['step'], df_actor['value'], label='Actor Loss', color='tab:blue', alpha=0.7)
if df_critic is not None:
    axs[1, 1].plot(df_critic['step'], df_critic['value'], label='Critic Loss', color='tab:green', alpha=0.7)
axs[1, 1].set_title('PPO Losses')
axs[1, 1].set_xlabel('Tijdstappen')
axs[1, 1].legend()
axs[1, 1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("marl_training_progress.png", dpi=300)
print("\nSucces! De grafieken zijn opgeslagen als 'marl_training_progress.png'.")
"""