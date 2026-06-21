import os
import glob
import json
from tensorboard.backend.event_processing import event_accumulator

# 1. Zoek naar je TensorBoard logbestanden
log_dir = "../results/tb_logs/"  # Pas dit aan naar de exacte map waar je logs worden weggeschreven
event_files = glob.glob(os.path.join(log_dir, "**/events.out.tfevents.*"), recursive=True)

if not event_files:
    print(f"Geen logbestanden gevonden in {log_dir}")
    exit()

# Pak het eerste/meest recente bestand
target_file = event_files[3]
print(f"Bezig met converteren van: {target_file}")

# 2. Laad de binaire data in
ea = event_accumulator.EventAccumulator(target_file, size_guidance={event_accumulator.SCALARS: 0})
ea.Reload()

# Dit wordt onze JSON-structuur
raw_data_dict = {}

# 3. Loop door alle beschikbare metrieken (tags) heen
for tag in ea.Tags()['scalars']:
    raw_data_dict[tag] = []
    events = ea.Scalars(tag)
    
    # Voeg per stap het stapnummer en de exacte ruwe waarde toe
    for e in events:
        raw_data_dict[tag].append({
            'step': int(e.step),
            'value': float(e.value)
        })

# 4. Schrijf de data weg naar een JSON-bestand
output_json = "raw_training_metrics.json"
with open(output_json, "w") as f:
    json.dump(raw_data_dict, f, indent=4)

print(f"\nSucces! Alle ruwe getallen zijn opgeslagen in '{output_json}'")
