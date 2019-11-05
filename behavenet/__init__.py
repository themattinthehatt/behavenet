
def get_params_dir():
    from pathlib import Path
    return Path.home()


def setup():
    """Set up the user's data, results, and figure directories; store info in user's home dir"""
    import os
    import json

    dirs = {}

    prompt = "Enter base data directory: "
    dirs['data_dir'] = input(prompt)

    prompt = "Enter base results directory: "
    dirs['save_dir'] = input(prompt)

    prompt = "Enter base figures directory: "
    dirs['fig_dir'] = input(prompt)

    params_dir = os.path.join(get_params_dir(), '.behavenet')
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)
    params_file = os.path.join(params_dir, 'directories')
    with open(params_file, 'w') as f:
        json.dump(dirs, f, sort_keys=False, indent=4)

    print('Directories are now stored in %s' % params_file)


def add_dataset():
    pass
