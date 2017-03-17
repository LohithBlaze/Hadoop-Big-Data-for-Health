import utils
import pandas as pd
import numpy as np
from datetime import datetime
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    merged=pd.merge(events, mortality, on='patient_id', suffixes=['_x','_y'],how='left')
    alive_ind=merged['label']!=1
    dead_ind=merged['label']==1
    alive_date=merged.groupby(merged['patient_id'][alive_ind.values],sort=False)['timestamp_x'].max()
    dead_date=merged.groupby(merged['patient_id'][dead_ind.values],sort=False)['timestamp_y'].max()
    alive=pd.DataFrame({'indx_date':alive_date}).reset_index()
    dead=pd.DataFrame({'indx_date':dead_date}).reset_index()
    dead['indx_date']=pd.to_datetime(dead['indx_date'])-pd.Timedelta(days=30)
    indx_date=alive.append(dead,ignore_index=True)
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    return indx_date
    

def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    merged=pd.merge(events,indx_date,on='patient_id',how='inner')
    filtered_index1=pd.to_datetime(merged['timestamp'])<=pd.to_datetime(merged['indx_date'])
    filtered_index2=pd.to_datetime(merged['timestamp'])>=pd.to_datetime(merged['indx_date'])-pd.Timedelta(days=2000)
    filtered_events=merged.loc[filtered_index1 & filtered_index2]
    filtered_events = filtered_events[['patient_id','event_id','value']]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    copy=filtered_events_df
    copy['event_id']=filtered_events_df['event_id'].map(feature_map_df.set_index('event_id')['idx'])
    #copy.loc[copy.event_id.isin(feature_map_df.event_id)]=feature_map_df['idx']
    #copy['event_id']=filtered_events_df['event_id'].replace(feature_map_df.set_index('event_id')['idx'])
    copy=copy.dropna()
    lastDIAGid=sum(feature_map_df['event_id'].str.contains('DIAG'))
    lastDRUGid=lastDIAGid+sum(feature_map_df['event_id'].str.contains('DRUG'))
    lastLABid=lastDRUGid+sum(feature_map_df['event_id'].str.contains('LAB'))

    notLAB= copy[copy['event_id']<=lastDRUGid].groupby(['patient_id','event_id'],sort=False)['value'].sum()
    LAB=copy[copy['event_id']>lastDRUGid].groupby(['patient_id','event_id'],sort=False).count()
    notLAB= pd.DataFrame(notLAB)
    LAB=pd.DataFrame(LAB)
    aggregated_events=notLAB.append(LAB).reset_index()
    # print aggregated_events
    aggregated_events.columns=['patient_id', 'feature_id', 'feature_value']
    
    normalized = aggregated_events.groupby(['feature_id'],sort=False)['feature_value'].transform(lambda x:x/x.max())
    aggregated_events['feature_value']=normalized
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    # print aggregated_events
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    
    merged=pd.merge(events,mortality,on='patient_id', suffixes=['_x','_y'],how='left')
    merged.fillna(0,inplace=True)
    patient_features=aggregated_events.groupby('patient_id')[['feature_id','feature_value']].apply(lambda x: [tuple(x) for x in x.values]).to_dict()
    mortality=merged.groupby('patient_id')['label'].apply(lambda x: x.unique()[0]).to_dict()
    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')    
    sorted_keys=sorted(patient_features.keys())
    
    d1=d2=''
    for i in sorted_keys:
        label=mortality[i]
        #d1+=str(label)
        #d2+=str(int(i))+' '+str(label)
        deliverable1.write(str(int(label)))
        deliverable2.write(str(int(i))+' '+str(int(label)))
        others=sorted(patient_features[i])
        for j in others:
            #d1+=' '+str(int(j[0]))+':'+str(format(j[1],'.6f'))
            #d2+=' '+str(int(j[0]))+':'+str(format(j[1],'.6f'))
            deliverable1.write(' '+str(int(j[0]))+':'+'%.6f' % (j[1]))
            deliverable2.write(' '+str(int(j[0]))+':'+'%.6f' % (j[1]))
        deliverable1.write(' \n');
        deliverable2.write(' \n');

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()