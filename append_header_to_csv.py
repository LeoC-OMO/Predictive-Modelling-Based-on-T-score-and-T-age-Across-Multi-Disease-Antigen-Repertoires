import pandas as pd
import os
from pathlib import Path

# ====================== CONFIGURATION ======================
# input_folder = r"C:\Users\ASUS\OneDrive\桌面\FYP_resources\TCR_data_new_csv\disease"
# output_folder = r"C:\Users\ASUS\OneDrive\桌面\FYP_resources\TCR_data_new_csv\disease_modified"

# Create output folder if it doesn't exist

def append_header(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Expected header
    expected_header = [
        "Number", "GeneV", "GeneJ", "Protein", "Percent", "Total",
        "annotation", "Epitope_species_DB_V1.0 sapiens/HomoSapiens/Wheat sapiens/Wheat"
    ]

    num_headers = len(expected_header)

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in the input folder.")
        exit()

    print(f"Found {len(csv_files)} CSV files. Processing...\n")

    for csv_file in csv_files:
        try:
            # Step 1: Read first few rows to detect header
            df_sample = pd.read_csv(csv_file, nrows=5, header=None, dtype=str)
            
            first_row = df_sample.iloc[0].astype(str).str.strip().tolist()
            
            # Clean expected header for comparison
            expected_clean = [h.strip() for h in expected_header]
            
            # Check if first row looks like our expected header (case-insensitive, allow minor differences)
            has_header = False
            if len(first_row) >= num_headers - 2:  # allow small column difference
                # Compare first few important columns
                match_count = sum(
                    any(word in cell or cell in word for word in ["Number", "GeneV", "GeneJ", "Protein", "Percent"])
                    for cell in first_row[:5]
                )
                if match_count >= 3:  # quite strict
                    has_header = True

            # Now read the full file properly
            if has_header:
                df = pd.read_csv(csv_file, header=0, dtype=str)
                print(f"  → Detected header in {csv_file.name}")
            else:
                df = pd.read_csv(csv_file, header=None, dtype=str)
                print(f"  → No header detected in {csv_file.name}")

            current_cols = df.shape[1]

            # Handle column count mismatch
            if current_cols < num_headers:
                for i in range(current_cols, num_headers):
                    df[i] = ""
                print(f"  → Padded {csv_file.name} with {num_headers - current_cols} empty columns")

            elif current_cols > num_headers:
                df = df.iloc[:, :num_headers]
                print(f"  → Truncated {csv_file.name} from {current_cols} to {num_headers} columns")

            # Assign the clean expected header
            df.columns = expected_header

            # Save the modified file
            output_file = output_path / csv_file.name
            df.to_csv(output_file, index=False)

            print(f"✓ Successfully processed: {csv_file.name} "
                f"({current_cols} cols → {num_headers} cols, header={'yes' if has_header else 'no'})")

        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {e}")

    print("\n=== All done! ===")
    print(f"Modified files have been saved to:\n{output_folder}")
    
    
import pandas as pd
from pathlib import Path


def append_header_overwrite(input_folder):
    # Expected header
    expected_header = [
        "Number", "GeneV", "GeneJ", "Protein", "Percent", "Total",
        "annotation", "Epitope_species_DB_V1.0 sapiens/HomoSapiens/Wheat sapiens/Wheat"
    ]

    num_headers = len(expected_header)

    input_path = Path(input_folder)

    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in the input folder.")
        return

    print(f"Found {len(csv_files)} CSV files. Processing...\n")

    for csv_file in csv_files:
        try:
            # Step 1: Peek at first few rows to detect if header exists
            df_sample = pd.read_csv(csv_file, nrows=5, header=None, dtype=str)
            first_row = df_sample.iloc[0].astype(str).str.strip().tolist()

            # Check if it looks like it already has a header
            has_header = False
            if len(first_row) >= num_headers - 2:
                match_count = sum(
                    any(word in cell or cell in word 
                        for word in ["Number", "GeneV", "GeneJ", "Protein", "Percent"])
                    for cell in first_row[:5]
                )
                if match_count >= 3:
                    has_header = True

            if has_header:
                print(f"  → Already has header: {csv_file.name} (skipping)")
                continue  # Skip files that already look correct

            # === No header detected → process and overwrite ===
            print(f"  → No header detected in {csv_file.name} → adding header and overwriting")

            # Read the full file without header
            df = pd.read_csv(csv_file, header=None, dtype=str)

            current_cols = df.shape[1]

            # Handle column count mismatch
            if current_cols < num_headers:
                for i in range(current_cols, num_headers):
                    df[i] = ""
                print(f"    → Padded with {num_headers - current_cols} empty columns")

            elif current_cols > num_headers:
                df = df.iloc[:, :num_headers]
                print(f"    → Truncated from {current_cols} to {num_headers} columns")

            # Assign the new header
            df.columns = expected_header

            # OVERWRITE the original file
            df.to_csv(csv_file, index=False)

            print(f"✓ Successfully updated: {csv_file.name} "
                  f"({current_cols} cols → {num_headers} cols)")

        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {e}")

    print("\n=== All done! ===")
    print(f"Only files without headers were overwritten in:\n{input_folder}")

