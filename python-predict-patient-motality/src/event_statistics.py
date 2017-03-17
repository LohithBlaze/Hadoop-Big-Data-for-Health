import time
import pandas as pd
import numpy as np
from datetime import datetime

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')
    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    merged=pd.merge(events, mortality, on='patient_id', suffixes=['_x','_y'],how='left')
    dead=merged[merged['label']==1]
    alive=merged[merged['label']!=1]
    dead_count=dead.groupby('patient_id')['event_id'].count()
    alive_count=alive.groupby('patient_id')['event_id'].count()
    
    avg_dead_event_count = np.mean(dead_count)
    max_dead_event_count = max(dead_count)
    min_dead_event_count = min(dead_count)
    avg_alive_event_count = np.mean(alive_count)
    max_alive_event_count = max(alive_count)
    min_alive_event_count = min(alive_count)

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    merged=pd.merge(events, mortality, on='patient_id', suffixes=['_x','_y'],how='left')
    dead=merged[merged['label']==1]
    alive=merged[merged['label']!=1]
    dead_encounter_count=dead.groupby('patient_id')['timestamp_x'].nunique()
    alive_encounter_count=alive.groupby('patient_id')['timestamp_x'].nunique()
    
    avg_dead_encounter_count = np.mean(dead_encounter_count)
    max_dead_encounter_count = max(dead_encounter_count)
    min_dead_encounter_count = min(dead_encounter_count)
    avg_alive_encounter_count = np.mean(alive_encounter_count)
    max_alive_encounter_count = max(alive_encounter_count)
    min_alive_encounter_count = min(alive_encounter_count)

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    merged=pd.merge(events, mortality, on='patient_id', suffixes=['_x','_y'],how='left')
    dead=merged[merged['label']==1]
    alive=merged[merged['label']!=1]
    
    dead_len=dead.groupby('patient_id')['timestamp_x'].apply(lambda x: pd.to_datetime(x).max()-pd.to_datetime(x).min())
    dead_len=dead_len.astype('timedelta64[D]')
    alive_len=alive.groupby('patient_id')['timestamp_x'].apply(lambda x: pd.to_datetime(x).max()-pd.to_datetime(x).min())
    alive_len=alive_len.astype('timedelta64[D]')
  
    avg_dead_rec_len = np.mean(dead_len)
    max_dead_rec_len = max(dead_len)
    min_dead_rec_len = min(dead_len)
    avg_alive_rec_len = np.mean(alive_len)
    max_alive_rec_len = max(alive_len)
    min_alive_rec_len = min(alive_len)
    
    
    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    You may change the train_path variable to point to your train data directory.
    OTHER THAN THAT, DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following line to point the train_path variable to your train data directory
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute event count metrics: " + str(end_time - start_time) + "s")
    print event_count

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute encounter count metrics: " + str(end_time - start_time) + "s")
    print encounter_count

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print("Time to compute record length metrics: " + str(end_time - start_time) + "s")
    print record_length
    
if __name__ == "__main__":
    main()
