from sensitivity_analysis.config import CONFIG

def get_marvy_acq_settings():
    return CONFIG['marvy']['acquisition']

def get_marvy_sampling_settings():
    return CONFIG['marvy']['sampling']

def get_acq_settings_vfa():
    return CONFIG['vfa']['acquisition']

def get_sampling_settings_vfa():
    return CONFIG['vfa']['sampling']
