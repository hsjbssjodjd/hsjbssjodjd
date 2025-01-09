import os
import shutil
import pandas as pd


def copy_bug_files(csv_file, bug_file_column, source_base_dir, output_dir):
    """
    Extract 'bug file' paths from a CSV file, search for them in the source directory,
    and copy the found files to a new directory.
    
    Args:
        csv_file (str): Path to the CSV file containing bug file information.
        bug_file_column (str): Name of the column that contains the bug file paths.
        source_base_dir (str): Base directory to search for the bug files.
        output_dir (str): Directory where the copied files will be stored.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        if bug_file_column not in df.columns:
            print(f"Column '{bug_file_column}' not found in the CSV file.")
            return

        # Get the bug file column values
        bug_file_entries = df[bug_file_column].dropna()

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Process each bug file entry
        for index, entry in enumerate(bug_file_entries, start=1):
            file_paths = entry.split(",")  # Split multiple paths separated by commas
            for file_path in file_paths:
                file_path = file_path.strip()  # Remove leading/trailing whitespace
                full_path = os.path.join(source_base_dir, file_path)

                if os.path.isfile(full_path):
                    # Copy the file to the output directory, preserving subdirectory structure
                    relative_path = os.path.relpath(full_path, source_base_dir)
                    dest_path = os.path.join(output_dir, relative_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(full_path, dest_path)
                    print(f"Copied: {full_path} -> {dest_path}")
                else:
                    print(f"File not found: {full_path}")

        print(f"All files processed. Copied files are in {output_dir}.")
    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    # Input CSV file and column name
    csv_file = "/media/oscar6/6F682A90B86D8F9F/wkb/FaultLocSim/IverilogRepository.csv"  # Replace with the path to your CSV file
    bug_file_column = "bug file"  # Replace with the column name containing file paths

    # Base directory to search for the bug files
    source_base_dir = "/media/oscar6/6F682A90B86D8F9F/wkb/FaultLocSim/project/iverilog"  # Replace with the root directory of your files

    # Output directory to store the copied files
    output_dir = "Iverilog/code"  # Replace with your desired output directory

    # Run the script
    copy_bug_files(csv_file, bug_file_column, source_base_dir, output_dir)
