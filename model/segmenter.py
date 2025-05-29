import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from PIL import Image
import cv2
import time

from tqdm import tqdm
import os
import openai_clip as openai_clip
import pickle
from glog import G

from operator import mod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
#from model.extractor import ViTExtractor

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from torch import autocast

from torchvision import transforms as T   

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import spacy

nlp = spacy.load("en_core_web_sm")

def extract_subject(text):
    doc = nlp(text)
    subject = None
    
    for token in doc:
        if "ROOT" in token.dep_ or "subj" in token.dep_:
            subject = token
            break
    
    if subject is not None:
        subject_span = subject.subtree
        subject_text = " ".join([token.text for token in subject_span])
        return str(subject)#subject_text
    else:
        return text

def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model



#torch.set_grad_enabled(False)

class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sam_checkpoint = cfg.sam_ckpt
        model_type = cfg.sam_type

        self.device = "cuda"
        self.cache_dir = cfg.cache_dir
        self.use_cache = cfg.use_cache
        self.overwrite_cache = cfg.overwrite_cache

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)

        self.predictor = SamAutomaticMaskGenerator(model=sam,)

        self.clip_encoder, _ = openai_clip.clip.load(cfg.clip_type, self.device)
        self.stride = int(cfg.clip_type[-2:])
        self.filter_size_ratio = cfg.filter_size_ratio
        self.filter_size_value = cfg.filter_size_value
        self.topk = cfg.topk
        #self.extractor = ViTExtractor('dino_vit_b16', self.stride, is_pretrained=True)


        self.strength = cfg.diffusion_strength 
        self.ddim_steps = cfg.diffusion_step 
        self.scale = cfg.diffusion_scale
        self.eta = cfg.diffusion_eta 
        self.a_prompt = cfg.diffusion_prompt_pos if cfg.diffusion_prompt_pos else ""
        self.n_prompt = cfg.diffusion_prompt_neg if cfg.diffusion_prompt_neg else ""
        self.diffusion_image_size = cfg.diffusion_image_size
        self.diffusion_pad = cfg.diffusion_pad
        self.diffusion_pad_value = cfg.diffusion_pad_value

        self.config = OmegaConf.load(cfg.diffusion_config)
        self.sdmodel = load_model_from_config(self.config, cfg.diffusion_ckpt, torch.device(self.device))
        self.sdddim_sampler = DDIMSampler(self.sdmodel, torch.device(self.device))

        self.sdddim_sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.eta, verbose=False)

        self.hard_text_weight = cfg.hard_text_weight
        self.hard_core_weight = cfg.hard_core_weight
        self.soft_text_weight = cfg.soft_text_weight
        self.soft_core_weight = cfg.soft_core_weight

        self.soft_weight = cfg.soft_weight
        self.hard_weight = cfg.hard_weight
        self.diff_weight = cfg.diff_weight

        
        
    def get_proposal(self, image_npy):
        all_masks = [ann['segmentation'] for ann in self.predictor.generate(image_npy)]

        if len(all_masks) == 0:
            all_masks = [np.ones_like(image_npy[:,:,0])]
        return all_masks

    def filter_proposal(self, all_masks):
        image_size = all_masks[0].shape[-1] * all_masks[0].shape[-2]
        areas = np.array([mask.sum() for mask in all_masks])
        filtered = np.zeros(shape=(len(all_masks)))

        num = min(self.topk, len(all_masks))
        filtered[np.argpartition(-areas, num-1)[:num]] = 1
        
        for idx in range(len(all_masks)):
            if areas[idx] <= image_size * self.filter_size_ratio or areas[idx] <= self.filter_size_value:   #  This is an important hyper-para
                filtered[idx] = 0
        return filtered

    def get_textfeat(self, text):
        input_ids = openai_clip.clip.tokenize(text).to(self.device)
        key_obj = extract_subject(text[0])
        new_text = text[0][:text[0].find(key_obj) + len(key_obj)]
        pref_ids = openai_clip.clip.tokenize([new_text]).to(self.device)
        sent_num = np.where(input_ids.cpu()[0] != 0)[0][-1] - 1
        core_num = np.where(pref_ids.cpu()[0] != 0)[0][-1] - 1
        if core_num > sent_num:
            core_num = sent_num
        core_ids = openai_clip.clip.tokenize([key_obj]).to(self.device)
        noun_tokens = self.clip_encoder.encode_text(input_ids)
        core_tokens = self.clip_encoder.encode_text(core_ids)
        noun_tokens /= noun_tokens.clone().norm(dim=-1, keepdim=True)
        core_tokens /= core_tokens.clone().norm(dim=-1, keepdim=True)
        return noun_tokens, core_tokens, key_obj, core_num

    def get_diffattn(self, image_pil, text, select_idx):
        cond = self.sdmodel.get_learned_conditioning([text[0] + ', ' + self.a_prompt] * 1)
        un_cond = self.sdmodel.get_learned_conditioning([self.n_prompt] * 1)

        
        self.sdmodel.control_scales = ([1.0] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        #raw_image = raw_image #input_image.convert("RGB")
        square_size = max(image_pil.size)

        if not self.diffusion_pad:
            scale_ratio = self.diffusion_image_size / square_size
            resize_w = int(image_pil.size[0] * scale_ratio // 64 * 64)
            resize_h = int(image_pil.size[1] * scale_ratio // 64 * 64)
            sdimage = image_pil.resize((resize_w, resize_h), Image.BICUBIC)
        else:
            background = Image.new('RGB', (square_size, square_size), (0, 0, 0))

            x_offset = (square_size - image_pil.size[0]) // 2
            y_offset = (square_size - image_pil.size[1]) // 2

            resize_w = self.diffusion_image_size
            resize_h = self.diffusion_image_size

            background.paste(image_pil, (x_offset, y_offset))
            sdimage = background.resize((resize_w, resize_h), Image.LANCZOS)

        attn_w = resize_w // 32
        attn_h = resize_h // 32

        G.images = {}
        G.index = select_idx
        G.attn_w = attn_w
        G.attn_h = attn_h

        
        sdimage = np.array(sdimage).astype(np.float32) / 255.0
        sdimage = sdimage[None].transpose(0, 3, 1, 2)
        sdimage = torch.from_numpy(sdimage).cuda()
        sdimage = 2. * sdimage - 1.
        precision_scope = autocast
        with precision_scope("cuda"), self.sdmodel.ema_scope():
            init_latent = self.sdmodel.get_first_stage_encoding(self.sdmodel.encode_first_stage(sdimage)) 
            t_enc = int(self.strength * self.ddim_steps)
            z_enc = self.sdddim_sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(self.device))
            samples = self.sdddim_sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=self.scale,
                                         unconditional_conditioning=un_cond, )

        m24 = (np.array(G.images[attn_w]).mean(axis=0)*255).astype(np.uint8)
        m24 = Image.fromarray(m24)
        #m24.save('mask.png')
        m24 = m24.resize((square_size, square_size), Image.LANCZOS)
        #m24.save('maskr.png')
        m24 = np.array(m24) / 255

        w,h = image_pil.size
        if w > h:
            m24 = m24[(w-h)//2:(w-h)//2+h, :]
        else:
            m24 = m24[:, (h-w)//2:(h-w)//2+w]
        #cv2.imwrite('maskr2.png', m24*255)
        m24 = m24[:,:,0]
        maxvalue = m24.max()
        minvalue = m24.min()
        m24 = (m24 - minvalue) / (maxvalue - minvalue + 1e-8)

        return m24

    def apply_position(self, image_npy, text):
        image_npy_pos = image_npy
        H, W, C = image_npy_pos.shape
        C = 3
        if "left" in text[0]:
            #print("!")
            W1 = W * 2 // 3
            w_vector = np.concatenate([np.linspace(1, 0, W1), np.zeros((W-W1))], axis=-1)
            w_vector = np.broadcast_to(w_vector.reshape(1, W, 1), (H, W, C))
            image_npy_pos = image_npy_pos*w_vector
        elif "right" in text[0]:
            W1 = W * 2 // 3
            w_vector = np.concatenate([np.zeros((W-W1)), np.linspace(0, 1, W1)], axis=-1)
            w_vector = np.broadcast_to(w_vector.reshape(1, W, 1), (H, W, C))
            image_npy_pos = image_npy_pos*w_vector
        #elif "top" in text[0]:
        #    H1 = H * 2 // 3
        #    w_vector = np.concatenate([np.linspace(1, 0, H1), np.zeros((H-H1))], axis=-1)
        #    w_vector = np.broadcast_to(w_vector.reshape(H, 1, 1), (H, W, C))
        #    image_npy_pos = image_npy_pos*w_vector
        #elif "bottom" in text[0]:
        #    H1 = H * 2 // 3
        #    w_vector = np.concatenate([np.zeros((H-H1)), np.linspace(0, 1, H1)], axis=-1)
        #    w_vector = np.broadcast_to(w_vector.reshape(H, 1, 1), (H, W, C))
        #    image_npy_pos = image_npy_pos*w_vector
        image_pil_pos = Image.fromarray((image_npy_pos*255).astype(np.uint8))
        return image_npy_pos, image_pil_pos

    def get_cliphard(self, masks, image_npy):
        cropped_images = []
        for mask in masks:
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
            new_img = image_npy * mask * 1.0 + image_npy * 0
            new_img = new_img[ymin:ymax+1, xmin:xmax+1,:]
            new_img = new_img.astype(np.uint8)
            cropped_images.append(new_img)
        clip_image = [Image.fromarray(img) for img in cropped_images]
        transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        clip_images = [transform(img).to(self.device).unsqueeze(0) for img in clip_image]
        tokens = torch.cat([self.clip_encoder.get_mask_token(clip_image, num_tokens=1, cls_tokens=None)[:, 0:1] for clip_image in clip_images], dim=0)
        obj_tokens = F.normalize(tokens, p=2, dim=-1)
        return obj_tokens

    def calc_soft(self, soft, text_feat, core_feat):
        full_feat = text_feat * self.soft_text_weight + core_feat * self.soft_core_weight
        attn_weights = soft @ full_feat.T
        score = attn_weights.view(-1)
        return score

    def calc_hard(self, hard, text_feat, core_feat):
        full_feat = text_feat * self.hard_text_weight + core_feat * self.hard_core_weight
        attn_weights = hard @ full_feat.T
        score = attn_weights.view(-1)
        return score

    def get_clipsoft(self, masks, image_pil):
        H, W = masks[0].shape

        attn_obj_masks = F.interpolate(torch.Tensor(np.array(masks)).to(self.device).view(len(masks), 1, H, W), (H // self.stride, W // self.stride), mode='nearest').long()
        attn_obj_masks = attn_obj_masks.permute(1, 2, 3, 0).view(1, -1, len(masks))  # B x (HxW) x N
        B, HW, N = attn_obj_masks.shape
        # Patch tokens cannot see cls token
        T11 = torch.ones(B, N, N, device=self.device, dtype=torch.float)
        T11 = T11 - torch.eye(N, device=self.device, dtype=torch.float).unsqueeze(0).repeat(B, 1, 1)
        M13 = 1 - attn_obj_masks.permute(0, 2, 1)
        T21 = torch.ones(B, HW, N, device=self.device, dtype=torch.float)
        F22 = torch.zeros(B, HW, HW, device=self.device, dtype=torch.float)
        M = torch.cat([
            torch.cat([T11, M13], dim=2),
            torch.cat([T21, F22], dim=2),
        ], dim=1)
        M[M==1] = float("-inf")
        # Make attn masks
        for i, block in enumerate(self.clip_encoder.visual.transformer.resblocks):
            block.attn_mask = M
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        clip_image = transform(image_pil).unsqueeze(0).to(self.device)
        tokens = self.clip_encoder.get_mask_token(clip_image, num_tokens=len(masks), cls_tokens=None)

        for i, block in enumerate(self.clip_encoder.visual.transformer.resblocks):
            block.attn_mask = None
        contain_obj_mask = attn_obj_masks.sum(dim=1)
        contain_obj_mask[contain_obj_mask > 1] = 1  # B x N
        obj_tokens = F.normalize(tokens[:, 0:len(masks)], p=2, dim=-1)
        return obj_tokens

    def calc_diffattn(self, masks, attn):
        score = []
        total = attn.shape[0] * attn.shape[1]
        #m = attn.mean()
        #attn_m = (attn > m) * attn ** 5
        #m_2 = attn_m.mean()
        #atten = (attn_m > m_2) * attn_m
        for mask in masks:
            #score.append(((attn_m - mask) ** 2).sum()/total)
            score.append((attn * mask).sum()/(mask.sum()+1e-6)-(attn * ~mask).sum()/((~mask).sum()+1e-6))
        score = torch.Tensor(score).to(self.device)
        return score

    def forward_with_cache(self, cache_name, sample_name, func, args):
        cache_file = os.path.join(self.cache_dir, cache_name, sample_name + ".pkl")
        if cache_name in self.use_cache and not cache_name in self.overwrite_cache and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        else:
            result = func(*args)
            if cache_name in self.use_cache:
                dir_name = os.path.dirname(cache_file)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            return result

    def forward(self, img, word, raw_img, raw_text, img_name, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        with torch.no_grad():
            image_pil = raw_img.convert("RGB")
            image_npy = np.array(image_pil)

            masks = self.forward_with_cache("proposal", img_name[0], self.get_proposal, [image_npy])
            filtered = self.filter_proposal(masks)
            text_feat, core_feat, core_text, select_idx = self.forward_with_cache("text", img_name[0], self.get_textfeat, [raw_text])
            image_npy_pos, image_pil_pos = self.forward_with_cache("pos", img_name[0], self.apply_position, [image_npy, raw_text])
            attn = self.forward_with_cache("attn", img_name[0], self.get_diffattn, [image_pil, raw_text, select_idx])
            hard = self.forward_with_cache("hard", img_name[0], self.get_cliphard, [masks, image_npy_pos])
            soft = self.forward_with_cache("soft", img_name[0], self.get_clipsoft, [masks, image_pil])

            #cv2.imwrite("mask.png", (attn*255).astype(np.uint8))
            
            score_hard = self.calc_hard(hard, text_feat, core_feat)
            score_soft = self.calc_soft(soft, text_feat, core_feat)
            score_diff = self.calc_diffattn(masks, attn)

            score = self.hard_weight * score_hard + self.soft_weight * score_soft + self.diff_weight * score_diff
            score = score.cpu() * filtered
            class_ids = score.argmax(dim=-1)
            mask = masks[class_ids]

        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w)
        return mask_image

        #if self.training:
        #    # resize mask
        #    if pred.shape[-2:] != mask.shape[-2:]:
        #        mask = F.interpolate(mask, pred.shape[-2:],
        #                             mode='nearest').detach()
        #    loss = F.binary_cross_entropy_with_logits(pred, mask)
        #    return pred.detach(), mask, loss
        #else:
        #    return pred.detach()
