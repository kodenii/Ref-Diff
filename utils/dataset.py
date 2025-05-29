import os
from typing import List, Union

import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset
from PIL import Image
import random

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

info = {
    'refcoco': {
        'train': 10000,#42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 10000,#42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 10000,#42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 10000,#44822,
        'val': 5000,
        'val-test': 5000
    }
}
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class RefDataset(Dataset):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size,
                 word_length, max_sample=None):
        super(RefDataset, self).__init__()
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.length = info[dataset][split]
        if max_sample is not None:
            self.length = min(max_sample, self.length)
        self.env = None
        self.prompt = []
        with open("./datasets/imaginarynet/outputs/prompt.txt") as reader:
            lines = reader.readlines()
            for line in lines:
                line = line.strip().strip("a professional photo of ")
                txt, c = line.split("\t")
                self.prompt.append((txt, c))

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir,
                             subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        if self.split != 'train':
            img_name = f"{self.dataset}_{self.split}_{index}"
            ref = loads_pyarrow(byteflow)
            # img
            ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8),
                                   cv2.IMREAD_COLOR)
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            img_size = img.shape[:2]
            # mask
            seg_id = ref['seg_id']
            mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')
            # sentences
            idx = np.random.choice(ref['num_sents'])
            sents = ref['sents']
            # transform
            mat, mat_inv = self.getTransformMat(img_size, True)
            img = cv2.warpAffine(
                img,
                mat,
                self.input_size,
                flags=cv2.INTER_CUBIC,
                borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        if self.mode == 'train':
            # sentence -> vector
            def read_img_mask(index):
                ori_img = cv2.imread(f"./datasets/imaginarynet/outputs/image/{index:0>6d}.jpg", cv2.IMREAD_COLOR)
                img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
                img_size = img.shape[:2]

                # mask transform
                mask = cv2.imread(f"./datasets/imaginarynet/outputs/mask/{index:0>6d}.png", cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (img_size[1], img_size[0]))
                
                return img, mask

            while True:
                if random.random() >= 0.5:
                    index2 = index
                else:
                    index2 = random.randint(0, info[self.dataset][self.split])
                img1, mask1 = read_img_mask(index)
                if np.max(mask1) == 0:
                    print(f"Error {index}")
                    index = random.randint(0, info[self.dataset][self.split])
                    continue
                img2, mask2 = read_img_mask(index2)
                if np.max(mask2) == 0:
                    print(f"Error {index2}")
                    continue
                img_size = img1.shape[:2]

                img1 = Image.fromarray(img1)
                img2 = Image.fromarray(img2)

                # 分割物体
                #mask1 = Image.fromarray(mask1)
                #mask2 = Image.fromarray(mask2)

                # 在img1中找到物体的包围盒
                bbox1 = [np.min(np.where(mask1)[1]), np.min(np.where(mask1)[0]),
                         np.max(np.where(mask1)[1]), np.max(np.where(mask1)[0])]
                w1, h1 = bbox1[2]-bbox1[0], bbox1[3]-bbox1[1]

                bbox2 = [np.min(np.where(mask2)[1]), np.min(np.where(mask2)[0]),
                         np.max(np.where(mask2)[1]), np.max(np.where(mask2)[0])]
                cw2, ch2 = bbox2[2]-bbox2[0]*0.5, bbox2[3]-bbox2[1]*0.5

                # 随机生成缩放因子和旋转角度
                scale_factor = random.uniform(0.7, 1)
                rotation_angle = random.uniform(-10, 10)

                # 对物体进行缩放和旋转
                obj1 = img1.crop(bbox1).rotate(rotation_angle, resample=Image.BICUBIC)
                w2, h2 = int(w1*scale_factor), int(h1*scale_factor)
                obj1 = obj1.resize((w2, h2), resample=Image.BICUBIC)

                resized_mask1 = Image.fromarray(mask1)
                resized_mask1 = resized_mask1.crop(bbox1).rotate(rotation_angle, resample=Image.BICUBIC)
                resized_mask1 = resized_mask1.resize((w2, h2), resample=Image.BICUBIC)

                # 随机生成物体的放置位置
                y_offset = random.randint(0, int(img2.height*0.7))
                x_offset = random.randint(0, int(img2.width*0.7))

                img3 = img2.copy()
                img3.paste(obj1, (x_offset, y_offset), resized_mask1)
                img3 = np.array(img3)
                img3 = img3[:,:,::-1]

                mask3_1 = np.zeros_like(mask2)
                mask3_1 = Image.fromarray(mask3_1)
                mask3_1.paste(resized_mask1, (x_offset, y_offset), resized_mask1)
                mask3_1 = np.array(mask3_1)
                mask3_2_bool = np.logical_xor(mask2.astype(bool), np.logical_and(mask2.astype(bool), mask3_1.astype(bool)))
                mask3_2 = mask3_2_bool.astype(np.uint8) * mask2

                #mask3_1.astype(np.uint8)
                #mask3_2.astype(np.uint8)

                #cv2.imwrite("img3.png", img3)
                #cv2.imwrite("mask3_1.png", mask3_1)
                #cv2.imwrite("mask3_2.png", mask3_2)

                if np.max(mask3_1) == 0:
                    print(f"Error {index} and {index2}")
                    continue
                bbox3 = [np.min(np.where(mask3_1)[1]), np.min(np.where(mask3_1)[0]),
                         np.max(np.where(mask3_1)[1]), np.max(np.where(mask3_1)[0])]
                cw3, ch3 = bbox3[2]-bbox3[0]*0.5, bbox3[3]-bbox3[1]*0.5

                img = img3
                if random.random() >= 0.5:
                    mask = mask3_1
                    c = random.random()
                    if c >= 0.66:
                        sent = self.prompt[index][0]
                    elif c >= 0.33:
                        if cw3 < cw2:
                            sent = np.random.choice(["left", "the left", "left one", "the left one", f"the left {self.prompt[index][1]}", f"{self.prompt[index][1]} on the left"], 1)[0]
                        else:
                            sent = np.random.choice(["right", "the right", "right one", "the right one", f"the right {self.prompt[index][1]}", f"{self.prompt[index][1]} on the right"], 1)[0]
                    else:
                        sent = self.prompt[index][1]
                else:
                    mask = mask3_2
                    c = random.random()
                    if c >= 0.66:
                        sent = self.prompt[index2][0]
                    elif c >= 0.33:
                        if cw2 < cw3:
                            sent = np.random.choice(["left", "the left", "left one", "the left one", f"the left {self.prompt[index2][1]}", f"{self.prompt[index2][1]} on the left"], 1)[0]
                        else:
                            sent = np.random.choice(["right", "the right", "right one", "the right one", f"the right {self.prompt[index2][1]}", f"{self.prompt[index2][1]} on the right"], 1)[0]
                    else:
                        sent = self.prompt[index2][1]
                
                mat, mat_inv = self.getTransformMat(img_size, True)
                img = cv2.warpAffine(
                    img,
                    mat,
                    self.input_size,
                    flags=cv2.INTER_CUBIC,
                    borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
                mask = cv2.warpAffine(mask,
                                          mat,
                                          self.input_size,
                                          flags=cv2.INTER_LINEAR,
                                          borderValue=0.)
                mask = mask / 255.
                img, mask = self.convert(img, mask)
                word_vec = tokenize(sent, self.word_length, True).squeeze(0)
                return img, word_vec, mask
        elif self.mode == 'val':
            # sentence -> vector
            sent = sents[0]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img = self.convert(img)[0]
            params = {
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'img_name': img_name,
                'ori_size': np.array(img_size)
            }
            return img, word_vec, ori_img, sent, params
        else:
            # sentence -> vector
            img = self.convert(img)[0]
            params = {
                'ori_img': ori_img,
                'seg_id': seg_id,
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'img_name': img_name,
                'ori_size': np.array(img_size),
                'sents': sents
            }
            return img, params

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def convert(self, img, mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"db_path={self.lmdb_dir}, " + \
            f"dataset={self.dataset}, " + \
            f"split={self.split}, " + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}"

    # def get_length(self):
    #     return self.length

    # def get_sample(self, idx):
    #     return self.__getitem__(idx)
