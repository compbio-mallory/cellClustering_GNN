src_root='./clone_4'
for rep in {1..5}; do
    echo "### Output for rep${rep}" >> output_011325.txt
    python train.py --CNA_path ./$src_root/rep${rep}/cell_adj_cosine.tsv --SNV_path ./$src_root/rep${rep}/input_genotype.tsv --labels_path ./$src_root/rep${rep}/cells_groups.tsv>> output.txt 2>&1
    echo -e "\n" >> output.txt
done

# src_root='./simulation_20250102'
# for subfolder in $src_root/*; do
#     if [ -d "$subfolder" ]; then
#         output_file="$subfolder/output_010925.txt"
#         for rep in {1..5}; do
#             echo "### Output for rep${rep}" >> "$output_file"
#             python train.py --CNA_path "$subfolder/rep${rep}/cell_adj_cosine.tsv" --SNV_path "$subfolder/rep${rep}/input_genotype.tsv" --labels_path "$subfolder/rep${rep}/cells_groups.tsv" >> "$output_file" 2>&1
#             echo -e "\n" >> "$output_file"
#         done
#     fi
# done