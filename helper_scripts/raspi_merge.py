import os
import pandas as pd
import json
import argparse

def merge_directory(target_dir)
path = target_dir
for (dirpath, dirnames, filenames) in os.walk(path):
    for dirname in dirnames:
        val_result_list = []
        for (dirpath1, dirnames1, filenames1) in os.walk(path+'/'+dirname):

            for filename in filenames1:
                if filename == 'emissions.csv':
                    emissions_df = pd.read_csv(path+'/'+dirname+'/'+filename)
                    emissions_df_merged = emissions_df.iloc[0]

                    energysum = emissions_df['energy_consumed'].sum() 
                    durationsum = emissions_df['duration'].sum() 
                    emissionssum = emissions_df['emissions'].sum() 

                    emissions_df_merged['energy_consumed'] = energysum
                    emissions_df_merged['duration'] = durationsum
                    emissions_df_merged['emissions'] = emissionssum  
                    emissions_df_merged = emissions_df_merged.to_frame().transpose()
                    emissions_df_merged.drop(columns=emissions_df_merged.columns[0], axis=1,  inplace=True)
                    
                    os.rename(path+'/'+dirname+'/'+'emissions.csv',path+'/'+dirname+'/'+'emissions_unmerged.csv')
                    emissions_df_merged.to_csv(path+'/'+dirname+'/'+'emissions.csv')

                elif filename.startswith('validation_results'):
                    with open(path+'/'+dirname+'/'+filename) as json_file:
                        val_result_dict = json.load(json_file)
                        val_result_list.append(val_result_dict)
        val_result_df = pd.DataFrame(val_result_list)
        result_dict = {}
        val_result_df_merged = val_result_df.iloc[0]
        avg_durationsum = val_result_df['avg_duration_ms'].mean() 
        k1_mean = val_result_df['accuracy_k1'].mean() 
        k3_mean = val_result_df['accuracy_k3'].mean() 
        k5_mean = val_result_df['accuracy_k5'].mean() 
        k10_mean = val_result_df['accuracy_k10'].mean() 
        size_sum = val_result_df['validation_size'].sum()

        val_result_df_merged['avg_duration_ms'] = avg_durationsum
        val_result_df_merged['accuracy_k1'] = k1_mean
        val_result_df_merged['accuracy_k3'] = k3_mean
        val_result_df_merged['accuracy_k5'] = k5_mean
        val_result_df_merged['accuracy_k10'] = k10_mean
        val_result_df_merged['validation_size'] = size_sum

        val_result_df_merged.to_json(path+'/'+dirname+'/'+'validation_results.json')
        #print(val_result_df.head)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', '--directory', default = 'mnt/data/staay/raspi_test')
    args = parser.parse_args()

    merge_directory(args.directory)


