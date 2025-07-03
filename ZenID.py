import torch
import os
import comfy.utils
import folder_paths
import numpy as np
import math
import cv2
import PIL.Image
from .resampler import Resampler
from .CrossAttentionPatch import Attn2Replace, instantid_attention
from .utils import tensor_to_image

from insightface.app import FaceAnalysis

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

import torch.nn.functional as F
from . import node_tools as node_helpers

MODELS_DIR = os.path.join(folder_paths.models_dir, "instantid")
if "instantid" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["instantid"]
folder_paths.folder_names_and_paths["instantid"] = (current_paths, folder_paths.supported_pt_extensions)

FACE_FUSION_DIR = os.path.join(folder_paths.models_dir, "face_fusion")
if "face_fusion" not in folder_paths.folder_names_and_paths:
    current_paths = [FACE_FUSION_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["face_fusion"]
folder_paths.folder_names_and_paths["face_fusion"] = (current_paths, folder_paths.supported_pt_extensions)

INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    h, w, _ = image_pil.shape
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def draw_kps_new(kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    h = 1024
    w = 1024
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

class InstantID(torch.nn.Module):
    def __init__(self, instantid_model, cross_attention_dim=1280, output_cross_attention_dim=1024, clip_embeddings_dim=512, clip_extra_context_tokens=16):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens

        self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(instantid_model["image_proj"])
        self.ip_layers = To_KV(instantid_model["ip_adapter"])

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4
        )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        #image_prompt_embeds = clip_embed.clone().detach()
        image_prompt_embeds = self.image_proj_model(clip_embed)
        #uncond_image_prompt_embeds = clip_embed_zeroed.clone().detach()
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)

        return image_prompt_embeds, uncond_image_prompt_embeds

class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
    
class ResidualBlock(torch.nn.Module):
    def __init__(self, dim, use_layernorm=True, use_relu=True):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, dim)
        self.use_layernorm = use_layernorm
        self.use_relu = use_relu

        if use_layernorm:
            self.norm1 = torch.nn.LayerNorm(dim)
            self.norm2 = torch.nn.LayerNorm(dim)
        if use_relu:
            self.relu = torch.nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        if self.use_layernorm:
            out = self.norm1(out)
        if self.use_relu:
            out = self.relu(out)
        out = self.linear2(out)
        if self.use_layernorm:
            out = self.norm2(out)
        out += identity
        return out

class EncoderNet(torch.nn.Module):
    def __init__(self, dim=512, hidden_dim=1024, depth=4, use_batchnorm=False):
        super().__init__()
        layers = []
        self.linear_layers = []
        self.norm_layers = []
        self.use_batchnorm = use_batchnorm
        
        for i in range(depth):
            in_dim = dim if i == 0 else hidden_dim
            out_dim = dim if i == depth - 1 else hidden_dim
            linear = torch.nn.Linear(in_dim, out_dim)
            layers.append(linear)
            self.linear_layers.append(linear)

            if i != depth - 1 and self.use_batchnorm:
                norm = torch.nn.BatchNorm1d(out_dim)
                layers.append(norm)
                self.norm_layers.append(norm)

        self.net = torch.nn.Sequential(*layers)


    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class FaceFusionNet(torch.nn.Module):
    def __init__(self, residual_depth=6, encoder_depth=4,
                 dim=512, hidden_dim=1024,
                 use_layernorm=True, use_relu=False, use_batchnorm=False):
        super().__init__()

        self.encoder = EncoderNet(dim=dim, hidden_dim=hidden_dim,
                                  depth=encoder_depth, use_batchnorm=use_batchnorm)
        
        self.input_linear = torch.nn.Linear(hidden_dim, dim, bias=False)


        self.residual_blocks = torch.nn.Sequential(*[
            ResidualBlock(dim, use_layernorm=use_layernorm, use_relu=use_relu)
            for _ in range(residual_depth)
        ])

        self.output_linear = torch.nn.Linear(dim, dim)

    def forward(self, a, b):
        out_a = self.encoder(a)  
        out_b = self.encoder(b) 

        x = torch.cat([out_a, out_b], dim=-1)  

        x = self.input_linear(x)

        x = self.residual_blocks(x) 

        x = self.output_linear(x) 

        return x

class To_KV(torch.nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = torch.nn.ModuleDict()
        for key, value in state_dict.items():
            k = key.replace(".weight", "").replace(".", "_")
            self.to_kvs[k] = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[k].weight.data = value

def _set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    else:
        to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()
    
    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = Attn2Replace(instantid_attention, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(instantid_attention, **patch_kwargs)

class ZenInstantIDModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "instantid_file": (folder_paths.get_filename_list("instantid"), )}}

    RETURN_TYPES = ("INSTANTID",)
    FUNCTION = "load_model"
    CATEGORY = "ZenID"

    def load_model(self, instantid_file):
        ckpt_path = folder_paths.get_full_path("instantid", instantid_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        model = InstantID(
            model,
            cross_attention_dim=1280,
            output_cross_attention_dim=model["ip_adapter"]["1.to_k_ip.weight"].shape[1],
            clip_embeddings_dim=512,
            clip_extra_context_tokens=16,
        )

        return (model,)

def extractFeatures(insightface, image, extract_kps=False):
    face_img = tensor_to_image(image)
    out = []

    insightface.det_model.input_size = (640,640) # reset the detection size

    for i in range(face_img.shape[0]):
        for size in [(size, size) for size in range(640, 128, -64)]:
            insightface.det_model.input_size = size # TODO: hacky but seems to be working
            face = insightface.get(face_img[i])
            if face:
                face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]

                if extract_kps:
                    out.append(draw_kps(face_img[i], face['kps']))
                else:
                    out.append(torch.from_numpy(face['embedding']).unsqueeze(0))

                if 640 not in size:
                    print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                break

    if out:
        if extract_kps:
            out = torch.stack(T.ToTensor()(out), dim=0).permute([0,2,3,1])
        else:
            out = torch.stack(out, dim=0)
    else:
        out = None

    return out

class ZenInstantIDFaceAnalysis:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM", "CoreML"], ),
            },
        }

    RETURN_TYPES = ("FACEANALYSIS",)
    FUNCTION = "load_insight_face"
    CATEGORY = "ZenID"

    def load_insight_face(self, provider):
        model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',]) # alternative to buffalo_l
        model.prepare(ctx_id=0, det_size=(640, 640))

        return (model,)

def add_noise(image, factor):
    seed = int(torch.sum(image).item()) % 1000000007
    torch.manual_seed(seed)
    mask = (torch.rand_like(image) < factor).float()
    noise = torch.rand_like(image)
    noise = torch.zeros_like(image) * (1-mask) + noise * mask

    return factor*noise

class ApplyZenID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET", ),
                "model": ("MODEL", ),
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "image_source": ("IMAGE", ),
                "image_face" : ("IMAGE", ),
                "instantid_file": (folder_paths.get_filename_list("instantid"), ),
                "insightface": (["CPU", "CUDA", "ROCM", "CoreML"], ),
                "weight": ("FLOAT", {"default": .8, "min": 0.0, "max": 5.0, "step": 0.01, }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "blur_kernel": ("INT", {"default": 51, "min": 1, "max": 101, "step": 2, }),
            },
            "optional": {
                "mask": ("MASK", ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING","LATENT")
    RETURN_NAMES = ("MODEL", "positive", "negative", "latent")
    FUNCTION = "apply_instantid"
    CATEGORY = "ZenID"

    def load_insight_face(self, provider):
        model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',]) # alternative to buffalo_l
        model.prepare(ctx_id=0, det_size=(640, 640))

        return model
    
    def load_model(self, instantid_file):
        ckpt_path = folder_paths.get_full_path("instantid", instantid_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        model = InstantID(
            model,
            cross_attention_dim=1280,
            output_cross_attention_dim=model["ip_adapter"]["1.to_k_ip.weight"].shape[1],
            clip_embeddings_dim=512,
            clip_extra_context_tokens=16,
        )

        return model

    def encode_prompt(self, clip, text):
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return [[cond, output]]
    
    def prepare_condition_inpainting(self, positive, negative, pixels, vae, mask):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {}

        out_latent["samples"] = orig_latent
        out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                                    "concat_mask": mask})
            out.append(c)
        return out[0], out[1], out_latent
    
    def prepare_mask(self, insightface, image, blur_kernel):
        image = tensor_to_image(image)
        insightface.det_model.input_size = (640,640)

        face = insightface.get(image[0])
        face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]

        mask = np.zeros((image.shape[1], image.shape[2]))
        x1, y1, x2, y2 = face['bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        mask[y1:y2, x1:x2] = 1
        mask =cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return mask

    def apply_instantid(self, instantid_file, insightface, control_net, model, clip, vae, image_source, image_face, start_at, end_at, weight=.8,blur_kernel = 51, ip_weight=None, cn_strength=None, noise=0.35, image_kps=None, mask=None, combine_embeds='average'):
        
        #load instantid model
        instantid = self.load_model(instantid_file)
        insightface = self.load_insight_face(insightface)
        
        #define positive and negative
        positive = " "
        negative = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"
        image = image_face

        #Encode prompt
        positive = self.encode_prompt(clip, positive)
        negative = self.encode_prompt(clip, negative)

        #Prepare mask
        if mask == None:
            mask = self.prepare_mask(insightface, image_source, blur_kernel) if mask is None else mask

        #Prepare conditioning for inpainting
        positive, negative, latent = self.prepare_condition_inpainting(positive, negative, image_source, vae, mask)
 
        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        
        self.dtype = dtype
        self.device = comfy.model_management.get_torch_device()

        ip_weight = weight if ip_weight is None else ip_weight
        cn_strength = weight if cn_strength is None else cn_strength

        face_embed = extractFeatures(insightface, image)
        if face_embed is None:
            raise Exception('Reference Image: No face detected.')

        # if no keypoints image is provided, use the image itself (only the first one in the batch)
        face_kps = extractFeatures(insightface, image_kps if image_kps is not None else image_source[0].unsqueeze(0), extract_kps=True)

        if face_kps is None:
            face_kps = torch.zeros_like(image) if image_kps is None else image_kps
            print(f"\033[33mWARNING: No face detected in the keypoints image!\033[0m")

        clip_embed = face_embed
        # InstantID works better with averaged embeds (TODO: needs testing)
        if clip_embed.shape[0] > 1:
            if combine_embeds == 'average':
                clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
            elif combine_embeds == 'norm average':
                clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
            #clip_embed_zeroed = add_noise(clip_embed, noise)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        # 1: patch the attention
        self.instantid = instantid
        self.instantid.to(self.device, dtype=self.dtype)

        image_prompt_embeds, uncond_image_prompt_embeds = self.instantid.get_image_embeds(clip_embed.to(self.device, dtype=self.dtype), clip_embed_zeroed.to(self.device, dtype=self.dtype))

        image_prompt_embeds = image_prompt_embeds.to(self.device, dtype=self.dtype)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(self.device, dtype=self.dtype)

        work_model = model.clone()

        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        if mask is not None:
            mask = mask.to(self.device)

        patch_kwargs = {
            "ipadapter": self.instantid,
            "weight": ip_weight,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
            "mask": mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
        }

        number = 0
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                _set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                _set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, ("middle", 1, index))
            number += 1

        # 2: do the ControlNet
        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        cnets = {}
        cond_uncond = []

        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype)

                if mask is not None and is_cond:
                    d['mask'] = mask
                    d['set_area_to_bounds'] = False

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False

        return(work_model, cond_uncond[0], cond_uncond[1], latent)
    
class ZenIDCombineFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net": ("CONTROL_NET", ),
                "model": ("MODEL", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "image_1": ("IMAGE", ),
                "image_2": ("IMAGE", ),
                "instantid_file": (folder_paths.get_filename_list("instantid"), ),
                "fusion_file": (folder_paths.get_filename_list("face_fusion"), ),
                "insightface": (["CPU", "CUDA", "ROCM", "CoreML"], ),
                "balance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, }),
                "weight": ("FLOAT", {"default": .8, "min": 0.0, "max": 5.0, "step": 0.01, }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("MODEL", "positive", "negative", )
    FUNCTION = "combine_face"
    CATEGORY = "ZenID"

    def load_insight_face(self, provider):
        model = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',]) # alternative to buffalo_l
        model.prepare(ctx_id=0, det_size=(640, 640))

        return model

    def load_fusion_model(self, fusion_file):
        ckpt_path = folder_paths.get_full_path("face_fusion", fusion_file)
        model = FaceFusionNet()
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        return model
    
    def load_model(self, instantid_file):
        ckpt_path = folder_paths.get_full_path("instantid", instantid_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model

        model = InstantID(
            model,
            cross_attention_dim=1280,
            output_cross_attention_dim=model["ip_adapter"]["1.to_k_ip.weight"].shape[1],
            clip_embeddings_dim=512,
            clip_extra_context_tokens=16,
        )

        return model

    def combine_face(self, instantid_file, insightface, fusion_file, control_net, image_1, balance, image_2, model, positive, negative, start_at, end_at, weight=.8, ip_weight=None, cn_strength=None, noise=0.35, image_kps=None, mask=None, combine_embeds='average'):
        #load instantid model
        instantid = self.load_model(instantid_file)
        insightface = self.load_insight_face(insightface)
        fusion_model = self.load_fusion_model(fusion_file)

        dtype = comfy.model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        
        self.dtype = dtype
        self.device = comfy.model_management.get_torch_device()

        ip_weight = weight if ip_weight is None else ip_weight
        cn_strength = weight if cn_strength is None else cn_strength

        face_embed_1 = extractFeatures(insightface, image_1)
        face_embed_2 = extractFeatures(insightface, image_2)


        label_embed = face_embed_1 * balance + face_embed_2 * (1 - balance)

        if face_embed_1 is None:
            raise Exception('Reference Image: No face detected.')
        
        if face_embed_2 is None:
            raise Exception('Reference Image: No face detected.')

        face_embed_1 = face_embed_1 * balance * 2
        face_embed_2 = face_embed_2 * (1 - balance) * 2

        face_embed = fusion_model(face_embed_1, face_embed_2)

        ratio = 8.0
        dst = arcface_dst * ratio
        out = []
        image_kps = draw_kps_new(dst)
        out.append(image_kps)
        out = torch.stack(T.ToTensor()(out), dim=0).permute([0,2,3,1])
        face_kps = out

        clip_embed = face_embed
        # InstantID works better with averaged embeds (TODO: needs testing)
        if clip_embed.shape[0] > 1:
            if combine_embeds == 'average':
                clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
            elif combine_embeds == 'norm average':
                clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
            #clip_embed_zeroed = add_noise(clip_embed, noise)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        # 1: patch the attention
        self.instantid = instantid
        self.instantid.to(self.device, dtype=self.dtype)

        image_prompt_embeds, uncond_image_prompt_embeds = self.instantid.get_image_embeds(clip_embed.to(self.device, dtype=self.dtype), clip_embed_zeroed.to(self.device, dtype=self.dtype))

        image_prompt_embeds = image_prompt_embeds.to(self.device, dtype=self.dtype)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(self.device, dtype=self.dtype)

        work_model = model.clone()

        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        if mask is not None:
            mask = mask.to(self.device)

        patch_kwargs = {
            "ipadapter": self.instantid,
            "weight": ip_weight,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
            "mask": mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
        }

        number = 0
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                _set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                _set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, ("middle", 1, index))
            number += 1

        # 2: do the ControlNet
        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        cnets = {}
        cond_uncond = []

        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device(), dtype=c_net.cond_hint_original.dtype)

                if mask is not None and is_cond:
                    d['mask'] = mask
                    d['set_area_to_bounds'] = False

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False

        return(work_model, cond_uncond[0], cond_uncond[1], )



NODE_CLASS_MAPPINGS = {
    "InstantIDModelLoader": ZenInstantIDModelLoader,
    "InstantIDFaceAnalysis": ZenInstantIDFaceAnalysis,
    "ApplyZenID": ApplyZenID,
    "ZenIDCombineFace": ZenIDCombineFace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantIDModelLoader": "Load InstantID Model",
    "InstantIDFaceAnalysis": "InstantID Face Analysis",
    "ApplyZenID": "ZenID FaceSwap",
    "ZenIDCombineFace": "ZenID Combine Face",
}
