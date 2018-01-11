import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

"""
Object detection function. This function will apply on each frame of the input video
Args:
    frame: the original frame
    net: the NN (SSD for this case)
    transform: transformation to transform frame to match NN format
Return:
    return the frame with detected objects (rectangle border + label)
"""
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    
    # transform the original frame to torch Variable -> feed into the NN. 4 steps
    # step 1
    frame_t = transform(frame)[0]
    
    # step 2: numpy array -> torch tensor
    x = torch.from_numpy(frame_t).permute(2,0,1)
    
    # step 3: create fake dimension for the batch since torch NN only accept inputs as batches
    x = x.unsqueeze(0)
    
    # step 4: torch tensor -> torch Variable
    x = Variable(x)
    
    # feed x into NN and get the result
    y = net(x)
    scale = torch.Tensor([width, height, width, height])
    detections = y.data # detections tensor contains following information [batch, #classes, #occurence, (score/threshold,x0,y0,x1,y1)]
    
    # loop through the detections tensor to detect all objects
    threshold = 0.6
    for cls in range(detections.size(1)):
        occurence = 0
        while detections[0, cls, occurence, 0] >= threshold:
            point = detections[0, cls, occurence, 1:] * scale
            point = point.numpy()
            x0 = int(point[0])
            y0 = int(point[1])
            x1 = int(point[2])
            y1 = int(point[3])
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255,0,0), 2)
            cv2.putText(frame, labelmap[cls-1], (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
            occurence += 1
    return frame
    
# Create NN/SSD
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc:storage)) # Load the pre-trained model   

# Create the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Prepare input and output
reader = imageio.get_reader('INPUT.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4')

# Start detecting
for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()









    
    
    