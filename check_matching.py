import os
import pandas as pd

# --- Step 1: Load Excel file ---
excel_path = "dieases_solution.xlsx"
df = pd.read_excel(excel_path)
print(f"✅ Loaded Excel file with {len(df)} rows.")

# --- Step 2: Load folder names ---
folders = [f for f in os.listdir() if os.path.isdir(f)]
print(f"📁 Found {len(folders)} folders in dataset.\n")

# --- Step 3: Clean text function ---
def clean_name(name):
    return str(name).strip().lower().replace("__", "_").replace(" ", "_")

# --- Step 4: Extract disease names from Excel ---
excel_diseases = df["Disease"].dropna().apply(clean_name).tolist()

# --- Step 5: Compare both lists ---
folder_clean = [clean_name(f) for f in folders]

matches = []
missing_in_excel = []
missing_in_folders = []

for f in folder_clean:
    if any(f_part in f for f_part in excel_diseases):
        matches.append(f)
    else:
        missing_in_excel.append(f)

for d in excel_diseases:
    if not any(d_part in d for d_part in folder_clean):
        missing_in_folders.append(d)

print("✅ Matching completed!\n")

print("📦 Matches found:", len(matches))
print("❌ Missing in Excel:", len(missing_in_excel))
print("⚠️ Missing in folders:", len(missing_in_folders))
print("\n--- Few Example Matches ---")
print(matches[:10])

if missing_in_excel:
    print("\n--- Example Missing in Excel ---")
    print(missing_in_excel[:10])

if missing_in_folders:
    print("\n--- Example Missing in Folders ---")
    print(missing_in_folders[:10])
