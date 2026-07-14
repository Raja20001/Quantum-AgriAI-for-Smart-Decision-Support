import yaml, os
def load_config(p=None):
    if p is None:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
    with open(p) as f: return yaml.safe_load(f)
