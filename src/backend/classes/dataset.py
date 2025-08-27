from pathlib import Path
from shared.fileUtils import get_file_extension, is_folder_not_empty, get_files_by_extensions, get_Data_filePath, get_Label_filePath
from dash import no_update
from sklearn.preprocessing import LabelEncoder
import mne, sys
import numpy as np 
import os, re
import pandas as pd


LABELS = {31: "arriba", 32: "abajo", 33: "derecha", 34: "izquierda"}
CUE_IDS = set(LABELS.keys())

def _inner_speech_cues(events):
    """
    events: array Nx3 de mne.find_events (col3 = event_id)
    Devuelve lista de (sample, event_id) SOLO para trials dentro de runs de inner speech.
    """
    # 1) quita el evento espurio 65536
    events = events[events[:,2] != 65536]

    in_inner_run = False
    cues = []
    for sample, _, eid in events:
        if eid == 15:         # start of run
            in_inner_run = False
        elif eid == 22:       # start of inner speech run
            in_inner_run = True
        elif eid == 16:       # end of run
            in_inner_run = False
        elif in_inner_run and eid in CUE_IDS:  # cue de clase dentro del run inner
            cues.append((int(sample), int(eid)))
    return cues


class Dataset:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.extensions_enabled = [".bdf", ".edf"]

    def upload_dataset(self,path_to_folder):
        print("Entering upload dataset ")
        print("getting all the files with .bdf, .edf extensions, just by now ....")

        if is_folder_not_empty(path_to_folder):
            print("Okay so know we are getting the files ")
            files = get_files_by_extensions(path_to_folder,self.extensions_enabled)
            for i in files: 
                print(i)
            if len(files) == 0 : 
                return {"status": 400, "message": f"No se han encontrado archivos con extension {self.extensions_enabled}"} 

            print("reading the files ")
            all_entries = []

            for file in files:
                if get_file_extension(file) == ".bdf":
                    raw_data = self.read_bdf(str(file))
                    print(f"se ha completado la lectura de {str(file)}, {raw_data.info.ch_names}")

                    events = mne.find_events(raw_data, stim_channel='Status', shortest_event=1, verbose=True)
                    inner_cues = _inner_speech_cues(events)

                    # Data raw
                    data, _ = raw_data.get_data(return_times=True)

                    # Crear label_array vacío
                    label_array = np.zeros((1, data.shape[1]), dtype=object)

                    # Duración estimada de cada etiqueta (en segundos)
                    label_duration_sec = 3.2
                    sfreq = raw_data.info['sfreq']
                    label_duration_samples = int(label_duration_sec * sfreq)

                    # Asignar etiquetas a cada segmento
                    for sample_idx, eid in inner_cues:
                        start = sample_idx
                        end = min(sample_idx + label_duration_samples, data.shape[1])
                        label_array[0, start:end] = LABELS[eid]

                    label_array = label_array.astype(str)

                    # Guardar en la carpeta _aux
                    auxFilePath = get_Data_filePath(str(file))
                    auxLabelPath = get_Label_filePath(str(file))

                    os.makedirs(os.path.dirname(auxFilePath), exist_ok=True)
                    os.makedirs(os.path.dirname(auxLabelPath), exist_ok=True)

                    np.save(auxFilePath, data)
                    np.save(auxLabelPath, label_array)

                    # Resumen por archivo
                    counts = {LABELS[k]: 0 for k in LABELS}
                    for _, eid in inner_cues:
                        counts[LABELS[eid]] += 1
                    print(f"Inner-speech en {Path(file).name}: {counts}")

                    # Guardar entradas (opcional, por si las necesitas de vuelta)
                    all_entries.extend(
                        [{"sample": s, "event_id": eid, "clase": LABELS[eid], "file": str(file)}
                         for (s, eid) in inner_cues]
                    )

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
                        run_number = None

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

            return {"status": 200, "message": f"Se han encontrado {len(files)}  sesiones ", "files" : files, "entries": all_entries}

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
    

    
    def load_signal_data(selected_file_path):
        
        
        
        
        print(f"Selected path is: {selected_file_path}")
        base = Path("Data")
        full_path = base / selected_file_path
        #We check if file passed is valid 
        if not selected_file_path:
            
            
            print("Invalid or missing file.")
            return no_update, True  # Keep interval disabled
        
        
        #Now we check if file is a valid format
        if not selected_file_path.endswith(".npy"):
            
            #We check if there's a corresponding .npy in Aux 
            
            
            
            #We check that the path is actually valid as an absolute one of /Data
            if not os.path.exists(f"Data/{selected_file_path}"):
                return no_update, True

            mappedFilePath = get_Data_filePath(f"Data/{selected_file_path}")
            
            if os.path.exists(mappedFilePath):
                
                print(mappedFilePath)
                
                signal = np.load(mappedFilePath, mmap_mode = 'r')
                full_path = Path(mappedFilePath)
            else: 
                return no_update, True
                
        else:
            
            # Load the signal
            signal = np.load(full_path, mmap_mode='r')
            
            

            
        
        

        #we want to extract the parent path and the file name to obtain the label that is in the parent directory and in a folder named labels with the same file name 
        
        labels_path = full_path.parent / "Labels" / full_path.name
        print("Buscando etiquetas en:", labels_path, "| Existe?", labels_path.exists())

        
        

        if not labels_path.exists():
            print("Labels don't exist")
            # We create dummy labels for no crash the application
            labels = np.zeros(signal.shape[0], dtype=int)
        else:
            labels = np.load(labels_path, allow_pickle=True)

        if signal.shape[0] < signal.shape[1]:
            signal = signal.T

        length_of_segment = 60
        
        for i in range(0,signal.shape[0],length_of_segment):
            randint = np.random.randint(0,10)
            if randint < 3: 
                
                labels[i:i+length_of_segment] = np.random.randint(1,5)
            
        labels = labels.astype(str)
        unique_labels = np.unique(labels)
        
        label_color_map  = {}
        
        for idx, label in enumerate(unique_labels):
            hue = (idx* 47) % 360
            label_color_map[str(label)] = f"hsl({hue}, 70%, 50%)"        

        
        
        
        
        

        '''
            Change this later when optimizing right now it's as is because the file wont load in time and the server times out, we need to optimize to load in chunks 
            or something like that, therefore we only load the first 50k points
        
        '''    
        
        signal = signal[:5000,:]
        labels = labels.reshape(-1)[:5000]
        
        print(f"signal shape: {signal.shape}")  
        print(f"labels shape: {labels.shape}")  
        
            
        # We encode the vector 
        
        encoder = LabelEncoder() 
        labels = encoder.fit_transform(labels.ravel())
        
        
        
        print(f"One hot encoded labels shape: {labels.shape}")
        
        
        
        
            
        # Serialize the signal (convert to list to make it JSON serializable)
        signal_dict = {
            "data": signal.tolist(),
            "num_channels": signal.shape[1],
            "num_timepoints": signal.shape[0], 
            "labels": labels.tolist(), 
            "label_color_map" : label_color_map
        }

        return signal_dict, False  # Enable interval
    
