import os

import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

from .moge.model import MoGeModel
from .utils3d.numpy import image_mesh, image_uv, depth_edge
import trimesh
import numpy as np

from pathlib import Path
import uuid
import tempfile
from PIL import Image
from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    is_accelerate_available = False
    pass

script_directory = os.path.dirname(os.path.abspath(__file__))

    
#region ModelLoading
class DownloadAndLoadMoGeModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [   
                        "MoGe_ViT_L_fp16.safetensors",
                        "MoGe_ViT_L_fp32.safetensors",
                    ],
                    {"tooltip": "Downloads from 'https://huggingface.co/Kijai/MoGe_safetensors' to 'models/MoGe'", },
                ),
               
                 "precision": (["fp16", "fp32", "bf16"],
                    {"default": "fp32", "tooltip": "The precision to use for the model weights. Has no effect with GGUF models"},),
            },
        }

    RETURN_TYPES = ("MOGEMODEL",)
    RETURN_NAMES = ("mochi_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "MoGe"
    DESCRIPTION = "Downloads and loads the selected MoGe model from Huggingface"

    def loadmodel(self, model, precision):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_download_path = os.path.join(folder_paths.models_dir, 'MoGe')
        model_path = os.path.join(model_download_path, model)
   
        repo_id = "kijai/MoGE_safetensors"
        
        if not os.path.exists(model_path):
            log.info(f"Downloading mochi model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{model}*"],
                local_dir=model_download_path,
                local_dir_use_symlinks=False,
            )
        model_config = {
            'encoder': 'dinov2_vitl14', 
            'remap_output': 
            'exp', 
            'output_mask': True, 
            'split_head': True, 
            'intermediate_layers': 4, 
            'dim_upsample': [256, 128, 64], 
            'dim_times_res_block_hidden': 2, 
            'num_res_blocks': 2, 
            'trained_area_range': [250000, 500000], 
            'last_conv_channels': 32, 
            'last_conv_size': 1
            }
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            model = MoGeModel(**model_config)
        model_sd = load_torch_file(model_path)
        if is_accelerate_available:
            for key in model_sd:
                set_module_tensor_to_device(model, key, dtype=dtype, device=device, value=model_sd[key])
        else:
            model.load_state_dict(model_sd, strict=True)
            model.to(dtype).to(device)
        model.eval()
        del model_sd

        return (model,)
    
class MoGeProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "model": ("MOGEMODEL",),
                 "image": ("IMAGE",),
                 "resolution_level": ("INT", {"default": 9}),
                 "remove_edge": ("BOOLEAN", {"default": True}),
                 "metallic_factor": ("FLOAT", {"default": 0.5, "step": 0.01}),
                 "roughness_factor": ("FLOAT", {"default": 1.0, "step": 0.01}),
                 "output_format": (["glb", "ply"], {"default": "glb"}),
                 "filename_prefix": ("STRING", {"default": "3D/MoGe"}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("depth", )
    FUNCTION = "process"
    CATEGORY = "MoGe"
    OUTPUT_NODE = True
    DESCRIPTION = "Runs the MoGe model on the input image"

    def process(self, model, image, resolution_level, remove_edge, metallic_factor, roughness_factor, output_format, filename_prefix):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        B, H, W, C = image.shape
  
        input_tensor = image.permute(0, 3, 1, 2).to(device)
        
        # model infer
        output = model.infer(input_tensor[0], resolution_level=resolution_level, apply_mask=True)
        # tensor outputs
        points_tensor = output['points']
        depth_tensor = output['depth']
        mask_tensor = output['mask']
        # convert to np
        points_np = points_tensor.cpu().numpy()
        depth_np = depth_tensor.cpu().numpy()
        mask_np = mask_tensor.cpu().numpy()
        input_np = image.cpu().numpy().astype(np.float32)

        print(input_np[0].shape)
        faces, vertices, vertex_colors, vertex_uvs = image_mesh(
            points_np,
            input_np[0],
            image_uv(width=W, height=H),
            mask=mask_np & ~depth_edge(depth_np, mask=mask_np, rtol=0.02) if remove_edge else mask_np,
            tri=True
        )
        vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]

        run_id = str(uuid.uuid4())

        tempdir = folder_paths.get_temp_directory()
       
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())

        if output_format == 'glb':
            output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.glb')
            output_glb_path.parent.mkdir(exist_ok=True)
            trimesh.Trimesh(
                vertices=vertices * [-1, 1, -1],    # No idea why Gradio 3D Viewer' default camera is flipped
                faces=faces, 
                visual = trimesh.visual.texture.TextureVisuals(
                    uv=vertex_uvs, 
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorTexture=Image.fromarray((input_np[0] * 255).astype(np.uint8)),
                        metallicFactor=metallic_factor,
                        roughnessFactor=roughness_factor
                    )
                ),
                process=False
            ).export(output_glb_path)
        elif output_format == 'ply':
            output_ply_path = Path(tempdir, f'{run_id}.ply')
            output_ply_path.parent.mkdir(exist_ok=True)
            trimesh.Trimesh(
                vertices=vertices, 
                faces=faces, 
                vertex_colors=vertex_colors,
                process=False
            ).export(output_ply_path)
        counter += 1
       
        depth_min = depth_tensor.min()
        depth_max = depth_tensor.max()
        depth_tensor = 1 - ((depth_tensor - depth_min) / (depth_max - depth_min))
        depth_tensor = depth_tensor.unsqueeze(0).cpu().float()

        return depth_tensor,
    

#endregion
#region NodeMappings
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadMoGeModel": DownloadAndLoadMoGeModel,
    "MoGeProcess": MoGeProcess,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadMoGeModel": "(Down)load MoGe Model",
    "MoGeProcess": "MoGe Process",
    }
