import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

run_name = "moa_critic20260715-145301"
# 1. Zoek naar je logbestand
log_dir = "../results/tb_logs/" + run_name
event_files = glob.glob(os.path.join(log_dir, "**/events.out.tfevents.*"), recursive=True)

if not event_files:
    print(f"Geen logbestanden gevonden in: {log_dir}")
    exit()

target_file = event_files[0] # We pakken hetzelfde bestand als voorheen [6] is belangrijk ofzo
print(f"Bezig met uitlezen van: {target_file}")

ea = event_accumulator.EventAccumulator(target_file, size_guidance={event_accumulator.SCALARS: 0})
ea.Reload()

available_tags = ea.Tags()['scalars']

# Automatisch agents detecteren
agents_found = sorted(list(set([t.split('/')[0] for t in available_tags if t.startswith("Agent_")])))

def extract_metric(tag_name):
    if tag_name in available_tags:
        events = ea.Scalars(tag_name)
        return pd.DataFrame({'step': [e.step for e in events], 'value': [e.value for e in events]})
    return None

# 2. Maak een lay-out: 2 rijen van 2 voor agent-specifieke zaken, en 1 grote rij onderop voor de Extrinsieke Return
fig = plt.figure(figsize=(14, 14))
grid = plt.GridSpec(4, 2, hspace=0.3, wspace=0.2)

ax_wm = fig.add_subplot(grid[0, 0])
ax_ir = fig.add_subplot(grid[0, 1])
ax_ent = fig.add_subplot(grid[1, 0])
ax_loss = fig.add_subplot(grid[1, 1])
ax_ext = fig.add_subplot(grid[3, :]) # Grote balk onderaan voor de echte score!
ax_last_explained_var = fig.add_subplot(grid[2, 0])
ax_moa_loss = fig.add_subplot(grid[2,1])

colors = plt.cm.tab10.colors

# Loop door agents voor de bovenste 4 grafieken
for idx, agent_prefix in enumerate(agents_found):
    color = colors[idx % len(colors)]
    
    # Grafiek 1: World Model Loss
    df_wm = extract_metric(f'{agent_prefix}/World_Model_Loss')
    if df_wm is not None:
        ax_wm.plot(df_wm['step'], df_wm['value'], color=color, alpha=0.8, label=agent_prefix)
    
    # Grafiek 2: Intrinsieke Reward Mean
    df_ir = extract_metric(f'{agent_prefix}/Intrinsic_Reward_Raw_Mean')
    if df_ir is not None:
        ax_ir.plot(df_ir['step'], df_ir['value'], color=color, alpha=0.8, label=agent_prefix)
        
    # Grafiek 3: Policy Entropy
    df_ent = extract_metric(f'{agent_prefix}/Entropy')
    if df_ent is not None:
        ax_ent.plot(df_ent['step'], df_ent['value'], color=color, alpha=0.8, label=agent_prefix)
        
    # Grafiek 4: PPO Losses
    df_actor = extract_metric(f'{agent_prefix}/Actor_Loss')
    df_critic = extract_metric(f'{agent_prefix}/Critic_Loss')
    if df_actor is not None:
        ax_loss.plot(df_actor['step'], df_actor['value'], color=color, linestyle='-', alpha=0.7, label=f'{agent_prefix} Actor')
    if df_critic is not None:
        ax_loss.plot(df_critic['step'], df_critic['value'], color=color, linestyle='--', alpha=0.5, label=f'{agent_prefix} Critic')
    
    #explained var
    df_var = extract_metric(f'{agent_prefix}/Explained_Variance')
    if df_ent is not None:
        ax_last_explained_var.plot(df_var['step'], df_var['value'], color=color, alpha=0.8, label=agent_prefix)
        
    df_moa = extract_metric(f'{agent_prefix}/MOA_Loss')
    if df_moa is not None:
        ax_moa_loss.plot(df_moa['step'], df_moa['value'], color=color, alpha=0.8, label=agent_prefix)

# --- GRAFIEK 5: DE ECHTE EXTRINSIEKE REWARD (OMGEVINGSPUNTEN) ---
# We zoeken flexibel naar de reward-tag (omdat deze vaak niet per agent maar globaal wordt opgeslagen)
reward_tags = ["Train/Mean_Episode_Return"]

if reward_tags:
    # We pakken de eerste match
    df_ext = extract_metric(reward_tags[0])
    if df_ext is not None:
        ax_ext.plot(df_ext['step'], df_ext['value'], color='black', linewidth=2.0, label='Extrinsieke Score')
        ax_ext.fill_between(df_ext['step'], df_ext['value'], color='gray', alpha=0.1)
    ax_ext.set_title(f"Echte Omgevingspunten / Episodic Return (Tag: {reward_tags[0]})", fontsize=12, fontweight='bold')
else:
    ax_ext.set_title("Echte Omgevingspunten (Geen 'return' of 'reward' tag gevonden in log file)", fontsize=12, fontweight='bold')
    print("Beschikbare tags waren:", available_tags) # Helpt je de juiste tag te vinden als de zoektocht faalt



# --- TITELS EN LABELS OPMAAK ---
ax_wm.set_title('World Model Loss (MSE)', fontweight='bold')
ax_wm.grid(True, linestyle='--', alpha=0.5)
ax_wm.legend()

ax_ir.set_title('Intrinsic Reward Raw Mean', fontweight='bold')
ax_ir.grid(True, linestyle='--', alpha=0.5)
ax_ir.legend()

ax_ent.set_title('PPO Policy Entropy', fontweight='bold')
ax_ent.grid(True, linestyle='--', alpha=0.5)
ax_ent.legend()

ax_loss.set_title('PPO Network Losses', fontweight='bold')
ax_loss.grid(True, linestyle='--', alpha=0.5)
ax_loss.legend(ncol=2, fontsize='small')

ax_ext.set_xlabel('Tijdstappen', fontsize=12)
ax_ext.set_ylabel('Echte Punten', fontsize=12)
ax_ext.grid(True, linestyle='--', alpha=0.5)
ax_ext.legend()

ax_last_explained_var.set_title('expl. var', fontweight='bold')
ax_last_explained_var.grid(True, linestyle='--', alpha=0.5)
ax_last_explained_var.legend()

ax_moa_loss.set_title('MOA LOSS', fontweight='bold')
ax_moa_loss.grid(True, linestyle='--', alpha=0.5)
ax_moa_loss.legend()

if not os.path.exists("images/" + run_name):
    os.makedirs("images/" + run_name)

plt.savefig("images/" + run_name + "/marl_validation_with_extrinsic.png", dpi=300, bbox_inches='tight')
print("\nSucces! De uitgebreide validatie-grafiek is opgeslagen als 'marl_validation_with_extrinsic.png'.")
