
import os
import csv

def extract_7k(input_folder, output_folder, top_n, sort_column):
# Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    def to_float(value):
        """Convert to float for sorting; non-numeric becomes -inf so it sinks in descending sort."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(".csv"):
            continue  # skip non-CSV files

        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + "_cropped.csv"
        output_path = os.path.join(output_folder, output_filename)
        print(f"Handling file: {input_path}")

        # Read the file
        with open(input_path, "r", encoding="utf-8", errors="ignore", newline="") as infile:
            reader = csv.DictReader(infile)
            rows = list(reader)
            fieldnames = reader.fieldnames

        # Validate presence of sort column
        if sort_column not in (fieldnames or []):
            print(f"  Skipped (column '{sort_column}' not found): {filename}")
            continue

        # Sort rows according to column 'Number' (descending)
        rows.sort(key=lambda row: to_float(row.get(sort_column)), reverse=True)

        # Keep only the top N entries
        top_rows = rows[:top_n]

        # Write to CSV
        with open(output_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(top_rows)

        print(f"  Wrote {len(top_rows)} rows → {output_path}")

    print(f"All CSV files in '{input_folder}' have been processed and saved to '{output_folder}'.")
    

def extract_top_n(input_folder, top_n, sort_column):
    def to_float(value):
        """Convert to float for sorting; non-numeric becomes -inf so it sinks in descending sort."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    # Get all CSV files
    csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".csv")]
    
    if not csv_files:
        print("No CSV files found in the input folder.")
        return

    print(f"Found {len(csv_files)} CSV files. Processing...\n")

    for filename in csv_files:
        input_path = os.path.join(input_folder, filename)
        print(f"Handling file: {filename}")

        try:
            # Read the file
            with open(input_path, "r", encoding="utf-8", errors="ignore", newline="") as infile:
                reader = csv.DictReader(infile)
                rows = list(reader)
                fieldnames = reader.fieldnames

            # Validate presence of sort column
            if sort_column not in (fieldnames or []):
                print(f"  Skipped (column '{sort_column}' not found): {filename}")
                continue

            # Sort rows descending by the sort_column
            rows.sort(key=lambda row: to_float(row.get(sort_column)), reverse=True)

            # Keep only the top N rows
            top_rows = rows[:top_n]

            # OVERWRITE the original file
            with open(input_path, "w", newline="", encoding="utf-8") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(top_rows)

            print(f"✓ Overwritten with top {len(top_rows)} rows: {filename}")

        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")

    print("\n=== All done! ===")
    print(f"Original files have been overwritten with top {top_n} rows in:\n{input_folder}")

