import pandas as pd


def split_dataset(file_path):
    print(f"Loading data from {file_path}...")

    df = pd.read_csv(file_path)

    df = df.drop(columns=['activity'], errors='ignore')

    df.dropna(subset=['smiles', 'HIV_active'], inplace=True)
    df['HIV_active'] = df['HIV_active'].astype(int)

    print(f"Original dataset size: {len(df)} rows.")

    df_active = df[df['HIV_active'] == 1].reset_index(drop=True)
    df_inactive = df[df['HIV_active'] == 0].reset_index(drop=True)

    print(f"Active molecules: {len(df_active)}")
    print(f"Inactive molecules: {len(df_inactive)}")

    test_rows = []
    val_rows = []
    train_rows = []

    def assign_splits(subset_df, split_name):
        for i, row in subset_df.iterrows():
            if (i % 10) == 9:
                test_rows.append(row)
            elif (i % 9) == 8:
                val_rows.append(row)
            else:
                train_rows.append(row)

    assign_splits(df_active, "Active")
    assign_splits(df_inactive, "Inactive")

    df_train = pd.DataFrame(train_rows).reset_index(drop=True)
    df_val = pd.DataFrame(val_rows).reset_index(drop=True)
    df_test = pd.DataFrame(test_rows).reset_index(drop=True)

    print(f"Training Set: {len(df_train)} rows ({len(df_train) / len(df):.1%} of total)")
    print(f"Validation Set: {len(df_val)} rows ({len(df_val) / len(df):.1%} of total)")
    print(f"Test Set: {len(df_test)} rows ({len(df_test) / len(df):.1%} of total)")

    return df_train, df_val, df_test


# --- Execution ---
FILE_PATH = 'HIV.csv'
train_file = 'HIV_train.csv'
val_file = 'HIV_validation.csv'
test_file = 'HIV_test.csv'

df_train, df_val, df_test = split_dataset(FILE_PATH)
df_train.to_csv(train_file, index=False)
df_val.to_csv(val_file, index=False)
df_test.to_csv(test_file, index=False)
