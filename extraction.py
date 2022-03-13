import torch
from torch.utils.data import DataLoader
from PIL import Image

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
    width,height = image.size
    for x in range(0, width, stepSize):
        # yield the current window
        if x + windowSize > width:
            break;
        # print((x, height ,x + windowSize,height))
        yield image.crop((x, 0 ,x + windowSize,height) )

def extract_vector(net,img):
    net.cuda()
    net.eval()
    with torch.no_grad():
        
        vector = net(img).cpu().data.squeeze()
    return vector


if __name__ == "__main__":
    pass
        
        