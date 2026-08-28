import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

run_name = "ablation_only_influence_909620260810-020934"
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
fig = plt.figure(figsize=(30, 36))
grid = plt.GridSpec(7, 2, hspace=0.3, wspace=0.2)

ax_ir = fig.add_subplot(grid[0, 0])
ax_ent = fig.add_subplot(grid[0, 1])
ax_last_explained_var = fig.add_subplot(grid[1, 0])
ax_intrinsic_stats = fig.add_subplot(grid[1, 1])

ax_ext = fig.add_subplot(grid[2, :])

ax_total_bumps = fig.add_subplot(grid[3, :])
ax_wall_bumps = fig.add_subplot(grid[4, 0])
ax_success_rate = fig.add_subplot(grid[4, 1])


ax_collisions = fig.add_subplot(grid[5, 0])
ax_collisions_stat = fig.add_subplot(grid[5, 1])

ax_correct_plate = fig.add_subplot(grid[6, 0])
ax_wrong_plate = fig.add_subplot(grid[6, 1])

colors = plt.cm.tab10.colors

# Loop door agents voor de bovenste 4 grafieken
for idx, agent_prefix in enumerate(agents_found):
    color = colors[idx % len(colors)]
    
    # Grafiek 1: Intrinsieke Reward Mean
    df_ir = extract_metric(f'{agent_prefix}/Intrinsic_Reward_Raw_Mean')
    if df_ir is not None:
        ax_ir.plot(df_ir['step'], df_ir['value'], color=color, alpha=0.8, label=agent_prefix)
        
    # Grafiek 2: Policy Entropy
    df_ent = extract_metric(f'{agent_prefix}/Entropy')
    if df_ent is not None:
        ax_ent.plot(df_ent['step'], df_ent['value'], color=color, alpha=0.8, label=agent_prefix)
        
    
    #3a explained var
    df_var = extract_metric(f'{agent_prefix}/Explained_Variance')
    if df_var is not None:
        df_var['smoothed'] = df_var['value'].ewm(alpha=0.05, adjust=False).mean()

        ax_last_explained_var.plot(df_var['step'], df_var['value'], color=color, alpha=0.15)
        ax_last_explained_var.plot(df_var['step'], df_var['smoothed'], color=color, alpha=1.0, linewidth=2.0, label=agent_prefix)
        
    #3b social_influence reward
    df_influence = extract_metric(f'{agent_prefix}/Total_Full_Influence_Mean')
    if df_influence is not None:
        df_influence['smoothed'] = df_influence['value'].ewm(alpha=0.05, adjust=False).mean()

        ax_intrinsic_stats.plot(df_influence['step'], df_influence['value'], color=color, alpha=0.15)
        ax_intrinsic_stats.plot(df_influence['step'], df_influence['smoothed'], color=color, alpha=1.0, linewidth=2.0, label=agent_prefix)
    
    #5 wall_bumps
    df_wall_bumps = extract_metric(f'Eval_Metrics_{agent_prefix}/01_Wall_Bumps')
    if df_wall_bumps is not None:
        df_wall_bumps['smoothed'] = df_wall_bumps['value'].ewm(alpha=0.05, adjust=False).mean()
        
        # 1. Plot de ruwe data heel licht op de achtergrond
        ax_wall_bumps.plot(df_wall_bumps['step'], df_wall_bumps['value'], color=color, alpha=0.15)
        
        # 2. Plot de schone trendlijn dik op de voorgrond voor je thesis
        ax_wall_bumps.plot(df_wall_bumps['step'], df_wall_bumps['smoothed'], color=color, alpha=1.0, linewidth=2.0, label=agent_prefix)
        
    #6 collisions
    df_collisions_dyn = extract_metric(f'Eval_Metrics_{agent_prefix}/02_Partner_Collisions_Dynamic')
    df_collisions_stat = extract_metric(f'Eval_Metrics_{agent_prefix}/03_Partner_Collisions_Stationary')
    
    if df_collisions_dyn is not None:
        df_collisions_dyn['smoothed'] = df_collisions_dyn['value'].ewm(alpha=0.05, adjust=False).mean()
        ax_collisions.plot(df_collisions_dyn['step'], df_collisions_dyn['value'], color=color, linestyle='-', alpha=0.15)
        ax_collisions.plot(df_collisions_dyn['step'], df_collisions_dyn['smoothed'], color=color, linestyle='-', alpha=1.0, linewidth=2.0, label=f'{agent_prefix}(dyn)')

    if df_collisions_stat is not None:
        df_collisions_stat['smoothed'] = df_collisions_stat['value'].ewm(alpha=0.05, adjust=False).mean()
        
        ax_collisions_stat.plot(df_collisions_stat['step'], df_collisions_stat['value'], color=color, linestyle='--', alpha=0.15)
        ax_collisions_stat.plot(df_collisions_stat['step'], df_collisions_stat['smoothed'], color=color, linestyle='--', alpha=1.0, linewidth=2.0, label=f'{agent_prefix}(dyn)')
    
    
    df_correct_plate = extract_metric(f'Eval_Metrics_{agent_prefix}/04_Steps_On_Correct_Plate')
    if df_correct_plate is not None:
        df_correct_plate['smoothed'] = df_correct_plate['value'].ewm(alpha=0.05, adjust=False).mean()
        
        # 1. Plot de ruwe data heel licht op de achtergrond
        ax_correct_plate.plot(df_correct_plate['step'], df_correct_plate['value'], color=color, alpha=0.15)
        
        # 2. Plot de schone trendlijn dik op de voorgrond voor je thesis
        ax_correct_plate.plot(df_correct_plate['step'], df_correct_plate['smoothed'], color=color, alpha=1.0, linewidth=2.0, label=agent_prefix)
    
    
    df_wrong_plate = extract_metric(f'Eval_Metrics_{agent_prefix}/05_Steps_Plate_Blocked_By_Wrong')
    if df_wrong_plate is not None:
        df_wrong_plate['smoothed'] = df_wrong_plate['value'].ewm(alpha=0.05, adjust=False).mean()

        ax_wrong_plate.plot(df_wrong_plate['step'], df_wrong_plate['value'], color=color, alpha=0.15)
        ax_wrong_plate.plot(df_wrong_plate['step'], df_wrong_plate['smoothed'], color=color, alpha=1.0, linewidth=2.0, label=f'plate {idx}')
    
    
    if (df_wall_bumps is not None) and (df_collisions_dyn is not None) and (df_collisions_stat is not None):
        df_total_bumps = df_wall_bumps.copy()
        df_total_bumps['value'] = df_wall_bumps['value'] + df_collisions_dyn['value'] + df_collisions_stat['value']
        df_total_bumps['smoothed'] = df_total_bumps['value'].ewm(alpha=0.05, adjust=False).mean()

        ax_total_bumps.plot(df_total_bumps['step'], df_total_bumps['value'], color=color, alpha=0.15)
        ax_total_bumps.plot(df_total_bumps['step'], df_total_bumps['smoothed'], color=color, alpha=1.0, linewidth=2.0, label=agent_prefix)
    
    

# --- GRAFIEK 4: DE ECHTE EXTRINSIEKE REWARD (OMGEVINGSPUNTEN) ---
# We zoeken flexibel naar de reward-tag (omdat deze vaak niet per agent maar globaal wordt opgeslagen)
reward_tags = ["Train/Mean_Episode_Return"]

if reward_tags:
    # We pakken de eerste match
    df_ext = extract_metric(reward_tags[0])
    if df_ext is not None:
        ax_ext.plot(df_ext['step'], df_ext['value'], color='black', linewidth=2.0, label='Extrinsieke Score')
        ax_ext.fill_between(df_ext['step'], df_ext['value'], color='gray', alpha=0.1)
    
    df_eval_extrinsic = extract_metric(f'Eval/Mean_Episode_Return')
    if df_eval_extrinsic is not None:
        ax_ext.plot(df_eval_extrinsic['step'], df_eval_extrinsic['value'], color='red', linewidth=2.0, alpha=0.5, label='Extrinsieke Score (Eval)')
        ax_ext.fill_between(df_eval_extrinsic['step'], df_eval_extrinsic['value'], color='gray', alpha=0.1)
    
    ax_ext.set_title(f"Echte Omgevingspunten / Episodic Return (Tag: {reward_tags[0]})", fontsize=12, fontweight='bold')
else:
    ax_ext.set_title("Echte Omgevingspunten (Geen 'return' of 'reward' tag gevonden in log file)", fontsize=12, fontweight='bold')
    print("Beschikbare tags waren:", available_tags) # Helpt je de juiste tag te vinden als de zoektocht faalt

df_success_rate = extract_metric(f'Eval/Success_Rate_Percentage')
if df_success_rate is not None:
    ax_success_rate.plot(df_success_rate['step'], df_success_rate['value'], color='black', linewidth=2.0, label='success_Rate')

    
    

ax_ir.set_title('Intrinsic Reward Raw Mean', fontweight='bold')
ax_ir.grid(True, linestyle='--', alpha=0.5)
ax_ir.legend()

ax_ent.set_title('PPO Policy Entropy', fontweight='bold')
ax_ent.grid(True, linestyle='--', alpha=0.5)
ax_ent.legend()

ax_intrinsic_stats.set_ylim([0, 1])
ax_intrinsic_stats.set_title(
    "Intrinsic Reward Statistics",
    fontweight='bold'
)
ax_intrinsic_stats.grid(True, linestyle='--', alpha=0.5)
ax_intrinsic_stats.legend()



#ax_ext.set_xlabel('Tijdstappen', fontsize=12)
ax_ext.set_ylabel('Echte Punten', fontsize=12)
ax_ext.grid(True, linestyle='--', alpha=0.5)
ax_ext.legend()


ax_total_bumps.set_title('Total Bumps', fontsize=12, fontweight='bold')
ax_total_bumps.grid(True, linestyle='--', alpha=0.5)
ax_total_bumps.legend()

ax_last_explained_var.set_title('expl. var', fontweight='bold')
ax_last_explained_var.grid(True, linestyle='--', alpha=0.5)
ax_last_explained_var.legend()

ax_wall_bumps.set_title('Wall bumps per Agent', fontweight='bold')
ax_wall_bumps.grid(True, linestyle='--', alpha=0.5)
ax_wall_bumps.legend()

ax_success_rate.set_title('Success Rate', fontweight='bold')
ax_wall_bumps.grid(True, linestyle='--', alpha=0.5)
ax_wall_bumps.legend()

ax_collisions.set_title('Collisions with other agents', fontweight='bold')
ax_collisions.grid(True, linestyle='--', alpha=0.5)
ax_collisions.legend()

ax_collisions_stat.set_title('Collisions with other agents (stat)', fontweight='bold')
ax_collisions_stat.grid(True, linestyle='--', alpha=0.5)
ax_collisions_stat.legend()

ax_correct_plate.set_title('Number of steps on correct plate', fontweight='bold')
ax_correct_plate.grid(True, linestyle='--', alpha=0.5)
ax_correct_plate.legend()

ax_wrong_plate.set_title('Number of steps on wrong plate', fontweight='bold')
ax_wrong_plate.grid(True, linestyle='--', alpha=0.5)
ax_wrong_plate.legend()
                              


if not os.path.exists("images/" + run_name):
    os.makedirs("images/" + run_name)

plt.savefig("images/" + run_name + "/marl_validation_why_" + run_name +".png", dpi=300, bbox_inches='tight')
print("\nSucces! De uitgebreide validatie-grafiek is opgeslagen als 'marl_validation_why.png'.")
