from ast import arg
import imp
from net import *
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--inputName',type=Path,required=True)
parser.add_argument('--queryDir',type=Path,required=True)

parser.add_argument("--saveCrop",action="store_true")
parser.add_argument("--vis",action="store_true")
args = parser.parse_args()
out_array = np.load(args.input)
names = np.load(args.inputName)


dir  = args.queryDir
list =  os.listdir(dir)

aggreated_vector = np.zeros([2048]) # output demension of GEM pooling layer
for e in tqdm(list):
    path = dir / e

    img = Image.open(path)
    width,height = img.size
    new_width = height
    new_height = height
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = img.crop((left, top, right, bottom))
    im = im.resize((320,320))
    
    tensor = torch.unsqueeze (transform(im),0).cuda()
    vector = extract_vector(net,tensor).numpy()

    aggreated_vector = aggreated_vector + vector

result = np.dot(out_array,aggreated_vector)
ranks = np.argsort(-result, axis=0)
print(f"For query Dir: {args.queryDir}")

# this have an issue that only for npy trained on the same machine, otherwise the path doesn't match
for i in range(0,5):
    print(f"name: {names[ranks[i]]}, score: {result[ranks[i]]}")
    pic = Image.open(names[ranks[i]])
    plt.imshow(pic)
    plt.show()