import os
import requests
import yaml

def gen_param_yaml(model_dir, config, new_height, new_width):
    resize = []
    crop = []
    resize.append(new_width)
    resize.append(new_height)
    crop.append(new_width)
    crop.append(new_height)
    if(config['model_type'] == "classification"):
        model_type = "classification"
    elif(config['model_type'] == "od"):
        model_type = "detection"
    elif(config['model_type'] == "seg"):
        model_type = "segmentation"
    dict_file =[]
    dict_file.append( {'session' :  {'artifacts_folder': '',
                                     'model_path': '',
                                     'session_name': 'tvmdlr'} ,
                      'task_type' : model_type,
                      'target_device': 'pc',
                      'postprocess':{'data_layout' : config['data_layout']},
                      'preprocess' :{'data_layout' : config['data_layout'],
                                    'mean':config['mean'],
                                    'scale':config['std'],
                                    'resize':config['resize'],
                                    'crop':config['crop']
                                     } })
    

    with open(model_dir+"/param.yaml", 'w') as file:
        documents = yaml.dump(dict_file[0], file)

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
