
import os
import csv

# Set the folder containing your MS-DOS files
# input_folder = r"C:\Users\ASUS\OneDrive\桌面\FYP_resources\tcr_csv_new_verification\disease" # input folder path
# output_folder = r"C:\Users\ASUS\OneDrive\桌面\FYP_resources\tcr_csv_new_verification_7k\disease" # output folder path

# Create the output folder if it doesn't exist

def msdos_to_csv(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".csv"
        output_path = os.path.join(output_folder, output_filename)
        print("handling file: " + output_filename)

        # Read the file
        with open(input_path, "r", encoding="utf-8", errors="ignore") as infile:
            lines = infile.readlines()

        # Write to CSV
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for line in lines:
                writer.writerow(line.strip().split('\t'))

    print(f"All files in '{input_folder}' have been converted to CSV format in '{output_folder}'.")
