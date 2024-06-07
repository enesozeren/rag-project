import bz2
import json
import random

def subsample_data(input_path, output_path, sample_size):
    """
    Subsample data from a .jsonl.bz2 file and write the subsample to another .jsonl.bz2 file.

    Parameters:
    - input_path (str): Path to the input .jsonl.bz2 file.
    - output_path (str): Path to the output .jsonl.bz2 file.
    - sample_size (int): Number of samples to retain.
    """
    # Read the entire dataset
    data = []
    try:
        with bz2.open(input_path, "rt") as file:
            for line in file:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    print("Warning: Failed to decode a line.")
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        return
    except IOError as e:
        print(f"Error: An error occurred while reading the file {input_path}: {e}")
        return

    # set a seed 
    random.seed(42)
    
    # Subsample the data
    if sample_size < len(data):
        subsampled_data = random.sample(data, sample_size)
    else:
        subsampled_data = data

    # Write the subsampled data to a new file
    try:
        with bz2.open(output_path, "wt") as output_file:
            for item in subsampled_data:
                output_file.write(json.dumps(item) + "\n")
    except IOError as e:
        print(f"Error: An error occurred while writing to the file {output_path}: {e}")

# Example usage
sample_size = 333  # Adjust the sample size as needed
# File paths
input_file_path = 'example_data/crag_task_1_dev_v3_release.jsonl.bz2'
output_file_path = "example_data/subsampled_crag_task_1_dev_v3_release.jsonl.bz2"
subsample_data(input_file_path, output_file_path, sample_size)
