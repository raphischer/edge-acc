import os
import pandas as pd
import json
import argparse

pd.options.mode.chained_assignment = None  # default='warn'

path = os.getcwd()+'/mnt_data/staay/final_final_raspi'
for (dirpath, dirnames, filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        val_result_list = []
        move_on = True
        for (dirpath1, dirnames1, filenames1) in os.walk(path+'/'+dirname):

            for filename in filenames1:
                if filename == 'emissions.csv':
                    emissions_df = pd.read_csv(path+'/'+dirname+'/'+filename)
                    emissions_df_merged = emissions_df.iloc[0]

                    energysum = emissions_df['energy_consumed'].sum() 
                    durationsum = emissions_df['duration'].sum() 
                    emissionssum = emissions_df['emissions'].mean() 

                    emissions_df_merged.loc['energy_consumed'] = energysum
                    emissions_df_merged.loc['duration'] = durationsum
                    emissions_df_merged.loc['emissions'] = emissionssum  
                    emissions_df_merged = emissions_df_merged.to_frame().transpose()
                    emissions_df_merged.drop(columns=emissions_df_merged.columns[0], axis=1,  inplace=True)
                    
                    os.rename(path+'/'+dirname+'/'+'emissions.csv',path+'/'+dirname+'/'+'emissions_unmerged.csv')
                    emissions_df_merged.to_csv(path+'/'+dirname+'/'+'emissions.csv')

                elif filename.startswith('validation_results') and filename!= 'validation_results.json':
                    with open(path+'/'+dirname+'/'+filename) as json_file:
                        val_result_dict = json.load(json_file)
                        val_result_list.append(val_result_dict)
                elif filename == 'validation_results.json':
                    
                    move_on = False

        if move_on:
            try:                
                val_result_df = pd.DataFrame(val_result_list)
                result_dict = {}
                val_result_df_merged = val_result_df.iloc[0]
                avg_durationsum = val_result_df['avg_duration_ms'].mean() 
                k1_mean = val_result_df['accuracy_k1'].mean() 
                k3_mean = val_result_df['accuracy_k3'].mean() 
                k5_mean = val_result_df['accuracy_k5'].mean() 
                k10_mean = val_result_df['accuracy_k10'].mean() 
                size_sum = val_result_df['validation_size'].sum()

                val_result_df_merged.loc['avg_duration_ms'] = avg_durationsum
                val_result_df_merged.loc['accuracy_k1'] = k1_mean
                val_result_df_merged.loc['accuracy_k3'] = k3_mean
                val_result_df_merged.loc['accuracy_k5'] = k5_mean
                val_result_df_merged.loc['accuracy_k10'] = k10_mean
                val_result_df_merged.loc['validation_size'] = size_sum

                val_result_df_merged.to_json(path+'/'+dirname+'/'+'validation_results.json')
                #print(val_result_df.head)
            except:
                pass


