import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from model.clip import build_model
from model.extractor import ViTExtractor, StegoSegmentationHead

from .layers import FPN, Projector, TransformerDecoder

class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        self.extractor = ViTExtractor('dino_vit_b16', 16, is_pretrained=True)

        self.backbone.eval()

        vision_size = self.extractor.model.embed_dim
        stego_head_dim = 90
        self.num_classes = 27

        self.stego_head = StegoSegmentationHead(vision_size, stego_head_dim, self.num_classes)
        self.stego_head.load_state_dict(torch.load("pretrain/vitb16_stego_head.ckpt"))
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # Projector
        self.text_proj = nn.Linear(512, 1024)
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

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

        with torch.no_grad():
            all_vis, all_words, sents = self.backbone.forward2(img, word)
            vis = all_vis[0]

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

        B = sents.shape[0]
        B, WW, HH = obj_masks.shape
        #vis_p = vis.permute(0,2,3,1).view(B, -1, D)
        words_p = sents.view(B, -1, 1)
        
        # b, 512, 26, 26 (C4)
        fq = self.neck(all_vis, self.text_proj(sents))
        #b, c, h, w = fq.size()
        B, D, W, H = fq.shape
        fq = self.decoder(fq, all_words, pad_mask)
        #fq_p = fq.view(B, D, -1).permute(0,2,1)
        fq = fq.reshape(B, D, W, H)

        #pred_raw = torch.matmul(fq_p, words_p)
        #pred_raw = pred_raw.view(B, 1, W, H)
        #pred_raw = einops.repeat(pred_raw, "B D W H -> B D (W W1) (H H1)", W1=WW // W, H1=HH // H)

        # b, 1, 104, 104
        pred_raw = self.proj(fq, self.text_proj(sents))

        #B, WW, HH = obj_masks.shape
        obj_masks = obj_masks.view(B, 1, WW, HH)
        #
        pred_masks = torch.zeros_like(pred_raw).float()
        for index in range(self.num_classes):
            selected = obj_masks == index
            selected_float = F.interpolate(selected.float(), pred_raw.shape[-2:], mode='bilinear')
            num = (selected_float * pred_raw).view(B, 1, -1).sum(dim=-1) / (selected_float.view(B, 1, -1).sum(dim=-1) + 1e-6)
            pred_masks += selected_float * num.view(B, 1, 1, 1)

        #pred = pred_raw

        pred = (pred_masks + pred_raw) / 2
        #pred = pred_masks

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask)
            return pred.detach(), mask, loss
        else:
            return pred.detach()
