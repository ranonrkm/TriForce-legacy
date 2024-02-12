import os
os.system(f"ls {dir_path} > {dir_path}file_list.txt")

with open(f"{dir_path}file_list.txt", 'r') as file:
    files = file.read().strip().split('\n')
filtered_files = [file for file in files if not file.endswith(('.json', '.lock'))]