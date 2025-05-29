import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
from timm.models.layers import trunc_normal_

from model.clip import build_model
from model.extractor import ViTExtractor, StegoSegmentationHead
import openai_clip as openai_clip

from .layers import FPN, Projector, TransformerDecoder

class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        #clip_model = torch.jit.load(cfg.clip_pretrain,
        #                            map_location="cpu").eval()
        #self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        self.backbone, _ = openai_clip.clip.load('ViT-B/16', 'cuda')
        self.extractor = ViTExtractor('dino_vit_b16', 16, is_pretrained=True)

        #self.backbone.eval()

        vision_size = self.extractor.model.embed_dim
        stego_head_dim = 90
        self.num_classes = 27

        self.stego_head = StegoSegmentationHead(vision_size, stego_head_dim, self.num_classes)
        self.stego_head.load_state_dict(torch.load("pretrain/vitb16_stego_head.ckpt"))

        self.learnable_cls_tokens = nn.Parameter(torch.zeros(1, self.num_classes, self.extractor.model.embed_dim))
        trunc_normal_(self.learnable_cls_tokens, std=.02)

        # Multi-Modal FPN
        #self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # Decoder
        #self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
        #                                  d_model=cfg.vis_dim,
        #                                  nhead=cfg.num_head,
        #                                  dim_ffn=cfg.dim_ffn,
        #                                  dropout=cfg.dropout,
        #                                  return_intermediate=cfg.intermediate)
        # Projector
        #self.text_proj = nn.Linear(512, 1024)
        #self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        #vis = self.backbone.encode_image(img)
        #word, state = self.backbone.encode_text(word)

        #with torch.no_grad():
        #all_vis, all_words, sents = self.backbone.forward2(img, word)
        #vis = all_vis[0]
        #B, D, W, H = vis.shape
        sents = self.backbone.encode_text(word)
        sents /= sents.clone().norm(dim=-1, keepdim=True)
        B = img.shape[0]
        sents = sents.view(B, -1, 1)

        with torch.no_grad():
            patch_tokens, _ = self.extractor.extract_descriptors(img,\
                                                layers=[11],
                                                facet='token',
                                                return_saliency=False)  # B x (HxW+1) x C1
            flip_patch_tokens, _ = self.extractor.extract_descriptors(img.clone().flip(dims=[3]),\
                                                layers=[11],
                                                facet='token',
                                                return_saliency=False)
            attn_obj_masks = self.stego_head(patch_tokens[:, 1:], flip_patch_tokens[:, 1:], \
                self.extractor.num_patches, single=False)  # B x H x W
            obj_masks = self.stego_head(patch_tokens[:, 1:], flip_patch_tokens[:, 1:], \
                    self.extractor.num_patches, resize=img.shape[-2::], single=False)  # B x H x W

        #print(all_words.shape)
        #print(sents.shape)

        #cluster_map = obj_masks
        attn_obj_masks = attn_obj_masks.view(B, -1).long()  # B x (HxW)
        attn_obj_masks = F.one_hot(attn_obj_masks, num_classes=self.num_classes)  # B x (HxW) x N
        B, HW, N = attn_obj_masks.shape
        # Patch tokens cannot see cls token
        T11 = torch.ones(B, N, N, device=sents.device, dtype=sents.dtype)
        T11 = T11 - torch.eye(N, device=sents.device, dtype=sents.dtype).unsqueeze(0).repeat(B, 1, 1)
        M13 = 1 - attn_obj_masks.permute(0, 2, 1)
        T21 = torch.ones(B, HW, N, device=sents.device, dtype=sents.dtype)
        F22 = torch.zeros(B, HW, HW, device=sents.device, dtype=sents.dtype)
        M = torch.cat([
            torch.cat([T11, M13], dim=2),
            torch.cat([T21, F22], dim=2),
        ], dim=1)
        M[M==1] = float("-inf")
        # Make attn masks
        for i, block in enumerate(self.backbone.visual.transformer.resblocks):
            block.attn_mask = M
        tokens = self.backbone.get_mask_token(img, num_tokens=self.num_classes, \
            cls_tokens=self.learnable_cls_tokens)
        
        for i, block in enumerate(self.backbone.visual.transformer.resblocks):
            block.attn_mask = None
        contain_obj_mask = attn_obj_masks.sum(dim=1)
        contain_obj_mask[contain_obj_mask > 1] = 1  # B x N
        obj_tokens = F.normalize(tokens[:, 0:self.num_classes], p=2, dim=-1)
        attn_weights = torch.matmul(obj_tokens, sents)
        #print(obj_tokens.shape, sents.shape)
        #print(attn_weights.shape)
        #print(obj_tokens.shape)
        #print(noun_tokens.shape)
        #print(attn_weights.shape)
        #print(attn_weights.max(), attn_weights.mean(), attn_weights.min())
        #class_ids = attn_weights[:,:,0] > attn_weights[:,:,1:].max(dim=-1)[0] #attn_weights.view(B, -1).mean(dim=-1, keepdims=True) #.argmax(dim=-1)
        pred_id = attn_weights[:,:,0]
        #obj_masks = obj_masks.view(B, 1, WW, HH)
        _, W, H = obj_masks.shape
        pred_masks = torch.zeros_like(obj_masks.view(B, 1, W, H)).float()
        for index in range(self.num_classes):
            selected = obj_masks == index
            #print(selected.shape)
            #selected_float = F.interpolate(selected.float(), pred_raw.shape[-2:], mode='bilinear')
            #num = (selected_float * pred_raw).view(B, 1, -1).sum(dim=-1) / (selected_float.view(B, 1, -1).sum(dim=-1) + 1e-6)
            num = pred_id[:,index]
            _, W, H = selected.shape
            selected = selected.view(B, 1, W, H)
            pred_masks += selected.float() * num.view(B, 1, 1, 1)

        pred = pred_masks

        #pred = (pred_masks + pred_raw) / 2
        #pred = pred_masks

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask)
            return pred.detach(), mask, loss
        else:
            return pred.view(B, W, H).detach()
