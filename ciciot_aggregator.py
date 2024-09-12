import os
import pandas as pd

# you can download the .tar file of the dataset from https://www.unb.ca/cic/datasets/iotdataset-2022.html
root_directory = '.\\CSV files.tar\\CSV files\\CIC Device Type'

dfs = []
for folder in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder)
    if os.path.isdir(folder_path):
        for subdir, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".csv"):
                    csv_path = os.path.join(subdir, file)
                    
                    df = pd.read_csv(csv_path)
                    df['Label'] = folder # for devicetype (camera, audio, etc. as label)
                    #df['Label'] = os.path.basename(subdir)  # for device (actual specific devices as labels)
                    
                    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv('aggregated_data_devicetype.csv', index=False)