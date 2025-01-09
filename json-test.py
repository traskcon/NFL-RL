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

json_file = "sample_playbook.json"

sample_playbook = {"sample_play":sample_play,
                   "trey right shotgun":rel_formation}

# Write dictionary to JSON file
with open(json_file, "w") as f:
    json.dump(sample_playbook, f, ensure_ascii=False, indent=4)

# Read data from JSON to dictionary
with open(json_file) as json_data:
    data = json.load(json_data)