from datetime import datetime
#from sklearn.preprocessing import LabelEncoder

import os

    
def save(folder_name, all_data=None):
    
    
    if all_data is None:
        		all_data = {
        		'person_id': [folder_name],
        	
        		}
        	
    else:
        		all_data['person_id'].append(folder_name)
        
        	
    return all_data