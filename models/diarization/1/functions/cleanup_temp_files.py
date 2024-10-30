import os


def cleanup_temp_files(data_dir, temp_files):
    for temp_file_path in temp_files:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    manifest_path = os.path.join(data_dir, "input_manifest.json")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
