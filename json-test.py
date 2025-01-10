import json
'''
JSON Play Format:
A Playbook is a set of 
A Play is a dictionary with the following keys:
    * play_type: Run or Pass (str)
    * play_name: Descriptive play name (str)
    * formation_name: Formation the play is run out of (str)
    * formation: Players involved: [locations] (dict)
    * target_locations: Target locations for all players involved, [0,0] for players without target locations (dict)
'''
sample_play = {"play_type":"run",
               "play_name":"sample_play",
               "formation_name":"trey right shotgun",
                "target_locations": {
                    "WR_1": [70,35],
                    "WR_2": [70,8],
                    "WR_3": [70,30],
                    "OL_1": [0,0],
                    "OL_2": [0,0],
                    "OL_3": [0,0],
                    "OL_4": [0,0],
                    "OL_5": [0,0],
                    "QB_1": [0,0],
                    "RB_1": [5,32],
                    "TE_1": [0,0]
                }}

rel_formation = {"formation_name":"trey right shotgun",
                "formation": {
                    "WR_1": [-1,35],
                    "WR_2": [0,8],
                    "WR_3": [-1,17],
                    "OL_1": [-1,33],
                    "OL_2": [-1,31],
                    "OL_3": [0,30],
                    "OL_4": [-1,28],
                    "OL_5": [-1,27],
                    "QB_1": [-5,30],
                    "RB_1": [-6,32],
                    "TE_1": [-2,25]
                }}

cover_1 = {"play_name":"Cover 1",
           "personnel":"Base 4-3",
           "locations": {
               "DL_1": [1,33],
               "DL_2": [1,31],
               "DL_3": [1,27],
               "DL_4": [1,24],
               "LB_1": [5,34],
               "LB_2": [5,29],
               "LB_3": [5,25],
               "DB_1": [4,38],
               "DB_2": [13,21],
               "DB_3": [3,17],
               "DB_4": [6,8]
           }}

off_file = "sample_playbook_off.json"
def_file = "sample_playbook_def.json"

sample_playbook_off = {"sample_play":sample_play,
                   "trey right shotgun":rel_formation}

sample_playbook_def = {"test_play":cover_1}

# Write dictionary to JSON file
with open(off_file, "w") as f:
    json.dump(sample_playbook_off, f, ensure_ascii=False, indent=4)

with open(def_file, "w") as f:
    json.dump(sample_playbook_def, f, ensure_ascii=False, indent=4)

'''# Read data from JSON to dictionary
with open(json_file) as json_data:
    data = json.load(json_data)'''