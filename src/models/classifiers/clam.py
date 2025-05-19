import os 
import sys 
import torch
# base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) 
# clam_path = os.path.join(base_path,"src/externals/CLAM")
clam_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../externals/CLAM"))
sys.path.append(clam_path)
print("CLAM path added:", clam_path)
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_mil import MIL_fc, MIL_fc_mc
from utils.utils import print_network



def load_clam_model(args, ckpt_path, device='cuda'):
    print('[INFO] Initializing CLAM model from checkpoint:', ckpt_path)

    model_dict = {
        "dropout": args.drop_out,
        "n_classes": args.n_classes,
        "embed_dim": args.embed_dim
    }

    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})

    # Select model type
    if args.model_type == 'clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type == 'clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'mil':
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    else:
        raise NotImplementedError(f"Unknown model type: {args.model_type}")

    print_network(model)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean[key.replace('.module', '')] = ckpt[key]

    model.load_state_dict(ckpt_clean, strict=True)
    model.to(device)
    model.eval()

    return model