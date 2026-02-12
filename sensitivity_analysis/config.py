import json
import copy

with open('sensitivity_analysis/settings.json', 'r') as f:
    CONFIG = json.load(f)
def get_sensitivity_config():
    return copy.deepcopy(CONFIG["sensitivity_analysis"])
