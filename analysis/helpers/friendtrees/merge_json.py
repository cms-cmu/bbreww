from copy import deepcopy
import json


#### This file is to merge the results of classifier evaluation json files for different years ####

files = ['/home/aniket/result_2022_preEE.json', '/home/aniket/result_2022EE.json', '/home/aniket/result_2023_preBPix.json',
        '/home/aniket/result_2023_BPix.json']

# Load all files
all_data = [json.load(open(f)) for f in files]

# Group predictions by offset (or whatever key you want to match on)
merged_by_offset = {}

for data in all_data:
    for pred in data['predictions']:
        offset = pred['metadata']['offset']
        
        if offset not in merged_by_offset:
            # First time seeing this offset - deep copy it
            merged_by_offset[offset] = deepcopy(pred)
        else:
            # Already have this offset - merge the data arrays
            existing = merged_by_offset[offset]
            
            for i in range(len(pred['outputs'][0]['output'])):
                if len(pred['outputs'][0]['output'][i]) > 0:
                    new_data = pred['outputs'][0]['output'][i][0]['data']
                    existing['outputs'][0]['output'][i][0]['data'].extend(new_data)

# Now flatten index 0 and 1 together
for offset, pred in merged_by_offset.items():
    outputs = pred['outputs'][0]['output']
    
    if len(outputs) >= 2 and len(outputs[0]) > 0 and len(outputs[1]) > 0:
        # Combine data from index 0 and index 1
        combined_data = outputs[0][0]['data'] + outputs[1][0]['data']
        
        # Keep index 0 structure, update its data
        outputs[0][0]['data'] = combined_data
        
        # Remove index 1
        pred['outputs'][0]['output'] = [outputs[0]]

# modify structure slightly and separate 3 k-fold models
merged = [{'SvB': merged_by_offset[0]['outputs'][0]['output'][0][0]},
          {'SvB': merged_by_offset[1]['outputs'][0]['output'][0][0]},
          {'SvB': merged_by_offset[2]['outputs'][0]['output'][0][0]}]


# save output to 3 json files for 3 k-folds
for i in range(3):
    with open(f"SvB_result_kfold{i}.json", "w") as outfile:
        json.dump(merged[i], outfile, indent=4)