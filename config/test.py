import yaml

with open('mpii.yaml', 'r') as f:
    out = yaml.safe_load(f)
    print(out['LOGGER'])