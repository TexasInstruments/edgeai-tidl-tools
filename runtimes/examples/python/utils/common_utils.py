SOC_MAP = {
    'AM62'   : ['AM62','AM62X'],
    'AM62A'  : ['AM62A','AM62AX'],
    'J722S'  : ['J722S','AM67A','TDA4AEN'],
    'J721E'  : ['J721E','AM68PA','TDA4VM'],
    'J721S2' : ['J721S2','AM68A','TDA4VL'],
    'J784S4' : ['J784S4','AM69A','TDA4VH'],
}

def get_soc(soc: str) -> str:
    soc = soc.strip().upper()
    for key,val in SOC_MAP.items():
        if soc in val:
            return key.strip().upper()
    return ''