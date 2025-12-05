# create_multiturn_ragbench.py
import random
import pandas as pd

df = pd.read_csv("data/train.csv")

# Group by dataset to sample realistic histories
multiturn = []

for name, group in df.groupby('dataset'):
    examples = group.to_dict('records')
    for i, ex in enumerate(examples):
        # Create history: 1–3 previous turns from same domain
        history_len = 2
        history = random.sample(
            [e for e in examples if e['original_id'] != ex['original_id']],
            k=min(history_len, len(examples)-1)
        )
        context = " || ".join([f"User: {h['user_query']} Assistant: {h['true_a_i']}" for h in history])
        
        multiturn.append({
            **ex,
            'dialogue_context': context.strip(),
            'turn_number': history_len + 1
        })

pd.DataFrame(multiturn).to_csv("data/train_multiturn.csv", index=False)
print(f"Created {len(multiturn):,} multi-turn examples")