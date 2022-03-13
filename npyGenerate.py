
import imp
from importlib.abc import Loader
from xmlrpc.client import Boolean
from net import *
import yaml
import argparse
from pathlib import Path
import os

import logging
from PIL import Image

import numpy as np 
from tqdm import tqdm
import torch.nn as nn
from extraction import *

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('--yaml',type=Path,default=Path(os.getcwd())/"config"/"cfg.yml")
parser.add_argument('--input_dir',type=Path,required=True)
# parser.add_argument('--ms',type=bool,action="store_true" ,help="Multi-scale allowed or not, load config from yaml")
parser.add_argument("-ms",action="store_true")
parser.add_argument('--out',type=Path,default="out.npy")
parser.add_argument('--outName',type=Path,default="out_name.npy")


def main():
    args = parser.parse_args()
    logging.info(f"Yaml file: {args.yaml}")
    with open(args.yaml,'r') as stream:
        cfg = yaml.safe_load(stream)
    print(cfg)
    names = []
    out = []
    dir  = args.input_dir
    logging.info(f"Finding pano folders: {os.listdir(dir)}")
    for dirname in tqdm(os.listdir(dir)):
        if len(dirname) !=3: #TODO: hardcode change later
            continue
        imagelist = dir / dirname
        images = os.listdir(imagelist)
        for image in images:
            path = imagelist / image
            img = Image.open(path)
            width, height = img.size
            img= img.crop((0,height/3,width,height/3*2)) # for this case, it's hard-coded 1920x320
            aggreated_vector = np.zeros([2048]) # output demension of GEM pooling layer
            for img_crop in sliding_window(img,cfg['stepSize'],cfg['windowSize']):
                if not args.ms:
                    tensor = torch.unsqueeze (transform(img_crop),0).cuda()
                    vector = extract_vector(net,tensor)
                    aggreated_vector = aggreated_vector + vector.numpy()
                else:
                    ms = eval(cfg['ms'])
                    ms_vector = torch.zeros([2048])
                    tensor = torch.unsqueeze (transform(img_crop),0).cuda()
                    for s in ms:
                        tensor = nn.functional.interpolate(tensor, scale_factor=s, mode='bilinear', align_corners=False)
                        vector = extract_vector(net,tensor)
                        ms_vector += vector
                    aggreated_vector = aggreated_vector + ms_vector.numpy()
            out.append(aggreated_vector)
            names.append(path.as_posix())
    out_array = np.stack(out,axis=0)
    with open(args.out,'wb') as f1:
        np.save(f1,out_array)
    with open(args.outName,'wb') as f2:
        np.save(f2,names)
if __name__ == "__main__":
    main()