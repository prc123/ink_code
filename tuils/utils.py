from sklearn.metrics import accuracy_score
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
import glob
import torch
import torch.utils.data as data
import matplotlib.patches as patches
from sklearn.metrics import accuracy_score,precision_score,recall_score
def get_stack_images(PREFIX,Z_START,Z_DIM,BUFFER,rect,DEVICE,step=1):
    mask = np.array(Image.open(PREFIX+"mask.png").convert('1'))
    label = torch.from_numpy(np.array(Image.open(PREFIX+"inklabels.png"))).gt(0).float()

    # for filename in tqdm(sorted(glob.glob(PREFIX+"surface_volume/*.tif"))[Z_START:Z_START+Z_DIM]):
    #     print(np.array(Image.open(filename), dtype=np.float32)/65535.0 )
    images = [np.array(Image.open(filename), dtype=np.float32)/65535.0 \
              for filename in tqdm(sorted(glob.glob(PREFIX+"surface_volume/*.tif"))[Z_START:Z_START+Z_DIM])]
    
    image_stack = torch.stack([torch.from_numpy(image) for image in images], dim=0)
    
    not_border = np.zeros(mask.shape, dtype=bool)
    not_border[BUFFER:mask.shape[0]-BUFFER, BUFFER:mask.shape[1]-BUFFER] = True
    arr_mask = np.array(mask) * not_border
    inside_rect = np.zeros(mask.shape, dtype=bool) * arr_mask
    # Sets all indexes with inside_rect array to True
    inside_rect[rect[1]:rect[1]+rect[3]+1:step, rect[0]:rect[0]+rect[2]+1:step] = True
    # 间隔取值
    outside_rect = np.zeros(mask.shape, dtype=bool)
    outside_rect[::step,::step] = True
    outside_rect[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1] = False
    outside_rect =outside_rect* arr_mask
    return mask,label,image_stack,inside_rect,outside_rect

def get_out_image(model , dataSet,BATCH_SIZE,DEVICE):
    dataload = data.DataLoader(dataSet, batch_size=BATCH_SIZE, shuffle=False)

    pixels_inside_rect = dataSet.pixels
    BUFFER = dataSet.buffer
    DEVICE = DEVICE
    label = dataSet.label
    output = torch.zeros_like(label).float()
    output2 = torch.zeros_like(label).float()
    with torch.no_grad():
        for i, (subvolumes, tmp_lable) in enumerate(tqdm(dataload)):
            for j, value in enumerate(model(subvolumes.to(DEVICE))):
                #self.label[y-BUFFER:y+BUFFER, x-BUFFER:x+BUFFER]
                y,x = pixels_inside_rect[i*BATCH_SIZE+j]
                output[y-BUFFER:y+BUFFER, x-BUFFER:x+BUFFER]+=value.cpu().view(BUFFER*2,BUFFER*2)
                output2[y-BUFFER:y+BUFFER, x-BUFFER:x+BUFFER]+=1
    return output,output2,label
def image_acc(image,lable,piexle,gt):
    target_list=[]
    value_list = []
    image = image.gt(gt)
    for i,data in enumerate(piexle):
        y,x = data
        value = image[y,x]
        target = lable[y,x]
        target_list.append(target)
        value_list.append(value)
    target_list = np.asarray(target_list)
    value_list = np.asarray(value_list)
    acc =accuracy_score(target_list,value_list)
    return acc 

def get_sorce(train_lable,tmpvalue,tmp_rect):
    tmp_lable = train_lable[tmp_rect[1]:tmp_rect[1]+tmp_rect[3]+1, tmp_rect[0]:tmp_rect[0]+tmp_rect[2]].reshape(-1)
    tmpvalue = tmpvalue[tmp_rect[1]:tmp_rect[1]+tmp_rect[3]+1, tmp_rect[0]:tmp_rect[0]+tmp_rect[2]].reshape(-1)

    tmp_pre = precision_score(tmp_lable.reshape(-1),tmpvalue)
    tmp_recall = recall_score(tmp_lable.reshape(-1),tmpvalue)
    #acc_tmp = accuracy_score(tmp_lable.view(-1).cpu(),tmpvalue)
    acc_tmp = accuracy_score(tmp_lable.reshape(-1),tmpvalue)
    F5_tmp =  (1+0.5**2)*tmp_pre*tmp_recall /(0.5**2*tmp_pre + tmp_recall)
    print("get acc and loss ...")
    print("acc:{}".format(acc_tmp))
    print("precision:{}".format(tmp_pre))
    print("recall:{}".format(tmp_recall))
    print("f5:{}".format(F5_tmp))


def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    # pixels = (pixels >= thr).astype(int)
    
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)