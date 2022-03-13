from ast import arg
import imp
from net import *
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--inputName',type=Path,required=True)
parser.add_argument('--query',type=Path,required=True)

parser.add_argument("--saveCrop",action="store_true")
parser.add_argument("--vis",action="store_true")
args = parser.parse_args()
out_array = np.load(args.input)
names = np.load(args.inputName)


path  = args.query
img = Image.open(path)
#center crop and resize to 320x320
width,height = img.size
new_width = height
new_height = height
left = (width - new_width)/2
top = (height - new_height)/2
right = (width + new_width)/2
bottom = (height + new_height)/2
im = img.crop((left, top, right, bottom))
im = im.resize((320,320))
if args.saveCrop:
    im.save("CropImage.jpg")
tensor = torch.unsqueeze (transform(im),0).cuda()
vector = extract_vector(net,tensor).numpy()
result = np.dot(out_array,vector)
ranks = np.argsort(-result, axis=0)
print(f"For query Image: {args.query}")

# this have an issue that only for npy trained on the same machine, otherwise the path doesn't match
for i in range(0,5):
    print(f"name: {names[ranks[i]]}, score: {result[ranks[i]]}")
    pic = Image.open(names[ranks[i]])
    plt.imshow(pic)
    plt.show()