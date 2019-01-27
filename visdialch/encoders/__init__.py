from visdialch.encoders.bu_lf import BottomUpLateFusionEncoder
from visdialch.encoders.lf import LateFusionEncoder

def Encoder(model_config, *args):
    name_enc_map = {
        'lf': LateFusionEncoder,
        'bu_lf': BottomUpLateFusionEncoder
    }
    return name_enc_map[model_config["encoder"]](model_config, *args)
