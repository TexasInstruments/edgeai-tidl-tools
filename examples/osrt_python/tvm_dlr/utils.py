import os
import requests


def download_models(mpath, models):
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
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)
