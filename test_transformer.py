from HIVPredictor import HIVPredictor
import pandas as pd

testset_path = "dataset/HIV_test.csv"

model_path = "models/hiv_transformer_sampled_20epoch.pth"
predictor = HIVPredictor(model_path)

df = pd.read_csv(testset_path)

active_correct = 0
active_incorrect = 0
inactive_correct = 0
inactive_incorrect = 0
invalid = 0

for index, row in df.iterrows():
    smiles = row['smiles']
    valid_class = int(row['HIV_active'])

    model_output = predictor.predict(smiles)
    if 'error' in model_output:
        invalid += 1
        continue
    is_active_predicted = model_output['is_active']

    if is_active_predicted:
        if valid_class == 1: active_correct += 1
        else: inactive_incorrect += 1
    else:
        if valid_class == 1: active_incorrect += 1
        else: inactive_correct += 1

print(f"active correct: {active_correct}")
print(f"active incorrect: {active_incorrect}")
print(f"inactive correct: {inactive_correct}")
print(f"inactive incorrect: {inactive_incorrect}")
print(f"invalid: {invalid}")