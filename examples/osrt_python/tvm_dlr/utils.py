import os
import requests


def download_models(mpath, models):
    headers = {
    'User-Agent': 'My User Agent 1.0',
    'From': 'aid@ti.com'  # This is another valid field
    }
    # Check whether the specified path exists or not
    isExist = os.path.exists(mpath)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(mpath)
    for model_name in models:
        model_path = models[model_name]['dir'] + model_name
        if(not os.path.isfile(model_path)):
            print("Downloading  ", model_name)
            url = models[model_name]['url']
            try:
                r = requests.get(url, allow_redirects=True, headers=headers)
            except requests.exceptions.RequestException as e:  # This is the correct syntax
                raise SystemExit(e)
            open(model_path, 'wb').write(r.content)
