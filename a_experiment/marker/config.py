import json
import os


def load() -> dict:
    conf = None
    path = os.path.realpath(__file__)
    path = path[:path.rindex(os.path.sep)]
    with open(path + "/config.json", 'r', encoding='utf-8') as f:
        conf = json.load(f)
    print(conf)
    return conf
