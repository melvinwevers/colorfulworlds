#!/bin/bash
#Set job requirements
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --cpus-per-task=16

# Set paths
data_home=$HOME/data/Colors/OrientalColorData #adjust to your own path
data_tmp="$TMPDIR"/images_all
output_dir="$TMPDIR"/output_dir
output_home=$HOME

# Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0

# Define functions
copy_to_tmp() {
    echo "copying files"
    cp -r "$1" "$2"
    echo "files copied!"
}

create_output_dir() {
    mkdir -p "$1"
}

execute_python() {
    echo "calculating buckets!"
    python $HOME/couleur_locale/src/calculate_buckets.py --meta_data "$data_home" --data_path "$data_tmp" --model_path "$output_dir" --n_colors 8 --n_pixels_dim 8
}

copy_to_home() {
    cp -r "$1" "$2"
}

# Main Execution
copy_to_tmp "$data_home/images_all" "$data_tmp"
create_output_dir "$output_dir"
execute_python
copy_to_home "$output_dir" "$output_home"
