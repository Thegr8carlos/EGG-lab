from pathlib import Path
from shared.fileUtils import get_file_extension, is_folder_not_empty, get_files_by_extensions, get_Data_filePath, get_Label_filePath

import mne, sys
import numpy as np 
import os, re
import pandas as pd

class Dataset:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.extensions_enabled = [".bdf", ".edf"]

    def upload_dataset(self,path_to_folder):
        print("Entering upload dataset ")
        print("getting all the files with .bdf, .edf extensions, just by now ....")

        if is_folder_not_empty:
            print("Okay so know we are getting the files ")
            files = get_files_by_extensions(path_to_folder,self.extensions_enabled)
            for i in files: 
                print(i)
            if len(files) == 0 : 
                return {"status": 400, "message": f"No se han encontrado archivos con extension {self.extensions_enabled}"} 

            print("reading the files ")
            for file in files:
                
                
                if get_file_extension(file) == ".bdf":
                
                    raw_data = self.read_bdf(str(file))
                    print(f"se ha completado la lectura de {str(file)}, {raw_data.info.ch_names}")
                    events = mne.find_events(raw_data, stim_channel='Status')
                    print(events)
                    
                if get_file_extension(file) == ".edf":
                    print("using .edf")
                    raw_data = self.read_edf(str(file))
                    
                    
                    data, times = raw_data.get_data(return_times = True)
                    
                    
                    print(raw_data.info['ch_names'])
                    events = mne.find_events(raw_data, stim_channel='Trigger')
                    
                    
                    events = events[events[:,2] == 65380]
                    
                    current_file = os.path.basename(str(file))
                    
                    match = re.search(r'run-(\d+)', current_file)   
                    
                    if match: 
                        run_number = int(match.group(1))
                    else:
                        run_number =None
                    
                    print(f"Run number is: {run_number}")
                    
                    
                    
                    #Temporary code, fix later
                    if run_number is not None:
                        label_file = os.path.join("Data/ciscoEEG/ds005170-1.1.2/textdataset", f"split_data_{run_number}.xlsx")
                        labels_df = pd.read_excel(label_file)
    
    
                            
                        label_list = labels_df.iloc[:, 0].tolist()  


                        if len(label_list) != len(events):
                            print(f"Warning: Number of labels ({len(label_list)}) != Number of events ({len(events)}). Truncating to min length.")
                            min_len = min(len(label_list), len(events))
                            label_list = label_list[:min_len]
                            events = events[:min_len]
                    else:
                        label_list = []  # or raise an error

                    # Prepare label array (object dtype to store strings)
                    label_array = np.zeros((1, data.shape[1]), dtype=object)

                    label_duration_sec = 3.2  # User-defined parameter
                    sfreq = raw_data.info['sfreq']
                    label_duration_samples = int(label_duration_sec * sfreq)

                    # Fill label array by mapping each event's sample window to the sentence label
                    for event, label in zip(events, label_list):
                        sample_idx = event[0]
                        start = sample_idx
                        end = min(sample_idx + label_duration_samples, data.shape[1])
                        label_array[0, start:end] = label

                    print("Data shape:", data.shape)
                    print("Label array shape:", label_array.shape)


                    print(label_array)
                    
                    
                    label_array = label_array.astype(str)

                    unique_labels, counts = np.unique(label_array, return_counts=True)
                    print("Unique labels and counts:")
                    for ul, ct in zip(unique_labels, counts):
                        print(f"{ul}: {ct}")
                                        
                                        
                    
                    print(f"File name is: {file}")
                 
             



                    auxFilePath = get_Data_filePath(str(file))
                    
                    auxLabelPath = get_Label_filePath(str(file))  
                    
                    print(f"Aux file is: {auxFilePath}")
                    print(f"Aux label is: {auxLabelPath}")
             
             
                    #Ensure the path structure for the file exists
                    
                    os.makedirs(os.path.dirname(auxFilePath), exist_ok=True)
                    os.makedirs(os.path.dirname(auxLabelPath), exist_ok=True)
             
                    #Now that we've ensured that the path was created we can now write the .npy
                    
                    
                    np.save(auxFilePath, data)
                    np.save(auxLabelPath, label_array)


                    
            print("ending")     
                    
                
            return {"status": 200, "message": f"Se han encontrado {len(files)}  sesiones ", "files" : files}

        else:
            return {"status": 400, "message": "Se ha seleccionado una carpeta Vacia "}

    def find_datasets():
        print("finding datasets")

    def get_info(self):
        return f"Info about the dataset hehe "

    def read_bdf(self, path):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
        # MNE espera una cadena, no un Path
        raw = mne.io.read_raw_bdf(str(file_path), verbose=True, infer_types=True)
        return raw


    def read_edf(self,path):
        
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
        # MNE espera una cadena, no un Path
        # raw = mne.io.read_raw_bdf(str(file_path), verbose=True, infer_types=True)
        raw = mne.io.read_raw_edf(str(file_path), verbose = True, infer_types = True, preload= True)
        return raw
