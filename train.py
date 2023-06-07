import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from dataSetLoad.dataSetLoad import SubvolumeDatasetUnet,SubvolumeDataset,SubvolumeDatasetUnetMutil
from model.simpleModel import U_Net,simpleModel
from config import config 
import numpy as np
import PIL.Image as Image
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score
from tuils.utils import image_acc,get_stack_images,get_out_image,get_sorce
import matplotlib.patches as patches

def evalUnet(model,eval_dataset,criterion,DEVICE,BATCH_SIZE):
    model.eval()
    eval_loader = data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        acc = []
        precision= []
        loss = []
        recall=[]
        F5=[]
        for i, (subvolumes, lable) in tqdm(enumerate(eval_loader)):
            value = model(subvolumes.to(DEVICE))
            # for j, value in enumerate(model(subvolumes.to(DEVICE))):
            #     lable_i = lable[j]           
            tmp_loss = criterion((value), lable.to(DEVICE))
            loss.append(tmp_loss.item())
            tmpvalue = value.view(-1).cpu()>0.5            
            tmplable = lable.view(-1).cpu()
            tmp_pre = precision_score(tmplable,tmpvalue)
            tmp_recall = recall_score(tmplable,tmpvalue)
            acc_tmp = accuracy_score(tmplable,tmpvalue)
            F5_tmp=1 
            #F5_tmp =  (1+0.5**2)*tmp_pre*tmp_recall /(0.5**2*tmp_pre + tmp_recall)
            precision.append(tmp_pre) 
            recall.append(tmp_recall)
            F5. append(F5_tmp)  
            acc.append(acc_tmp)
        print("get acc and loss ...")

        print("acc:{}".format(np.array(acc).mean()))
        print("precision:{}".format(np.array(precision).mean()))
        print("recall:{}".format(np.array(recall).mean()))
        print("F5:{}".format(np.array(F5).mean()))
        print("loss:{}".format(np.array(loss).mean()))
    return acc,precision,loss,recall

def get_UNet_data_DatasetUnet(PREFIX, Z_START,Z_DIM,BUFFER,rect_train,DEVICE,step,inside=True):
    image_info_1 = get_stack_images(PREFIX, Z_START,Z_DIM,BUFFER,rect_train,DEVICE,step)
    mask_1,label_1,images_1,inside_rect_1,outside_rect_1 = image_info_1
    pixels_inside_rect_1 = np.argwhere(inside_rect_1)
    pixels_outside_rect_1 = np.argwhere(outside_rect_1)
    if inside == True:
        train_dataset_1 = SubvolumeDatasetUnet(images_1, label_1, pixels_inside_rect_1,BUFFER,Z_DIM)
    else:
        train_dataset_1 = SubvolumeDatasetUnet(images_1, label_1, pixels_outside_rect_1,BUFFER,Z_DIM)
    return train_dataset_1,pixels_inside_rect_1,pixels_outside_rect_1


def train_Unet():
    #prepay data
    model_type = "Unet"
    #model_type = "base"
    print("train start。。。。")
    PREFIX = config.PREFIX
    DEVICE = config.DEVICE
    Z_START =config.Z_START
    Z_DIM = config.Z_DIM
    BATCH_SIZE = config.BATCH_SIZE
    BUFFER = config.BUFFER
    EPCHO = config.EPCHO
    rect_val = (1100, 2500, 400, 550)
    rect_train = (1200, 3000, 900, 750)
    
    # mask = np.array(Image.open(PREFIX+"mask.png").convert('1'))
    # label = torch.from_numpy(np.array(Image.open(PREFIX+"inklabels.png"))).gt(0).float().to(DEVICE)

    # # for filename in tqdm(sorted(glob.glob(PREFIX+"surface_volume/*.tif"))[Z_START:Z_START+Z_DIM]):
    # #     print(np.array(Image.open(filename), dtype=np.float32)/65535.0 )
    # images = [np.array(Image.open(filename), dtype=np.float32)/65535.0 \
    #           for filename in tqdm(sorted(glob.glob(PREFIX+"surface_volume/*.tif"))[Z_START:Z_START+Z_DIM])]
    # image_stack = torch.stack([torch.from_numpy(image) for image in images], dim=0).to(DEVICE)
# eval
    step_1  = int(BUFFER/4) 
    image_info_0 = get_stack_images(config.PREFIX_list[0], Z_START,Z_DIM,BUFFER,config.rect_train_list[0],DEVICE,step=step_1)
    #image_info_0_in = get_stack_images(config.PREFIX_list[0], Z_START,Z_DIM,BUFFER,config.rect_train_list[0],DEVICE,step=1)
    #image_info_0 = get_stack_images(PREFIX, Z_START,Z_DIM,BUFFER,rect_train,DEVICE,step=int(BUFFER/2))
    mask,label,images,inside_rect,outside_rect = image_info_0
    pixels_inside_rect = np.argwhere(inside_rect)
    pixels_outside_rect = np.argwhere(outside_rect)
    train_dataset_0 = SubvolumeDatasetUnet(images, label, pixels_outside_rect,BUFFER,Z_DIM)
    # _,label_in,imgaes_in,inside_rect_in,_=image_info_0_in
    # pixels_inside_rect = np.argwhere(inside_rect)
    #train_dataset_0_in = SubvolumeDatasetUnet(images, label, pixels_inside_rect,BUFFER,Z_DIM)
    train_dataset_0_in,_,_ = get_UNet_data_DatasetUnet(config.PREFIX_list[0], Z_START,Z_DIM,BUFFER,config.rect_train_list[0],DEVICE,step=step_1)

    image_info_1 = get_stack_images(config.PREFIX_list[1], Z_START,Z_DIM,BUFFER,config.rect_train_list[1],DEVICE,step=step_1)
    mask_1,label_1,images_1,inside_rect_1,outside_rect_1 = image_info_1
    pixels_inside_rect_1 = np.argwhere(inside_rect_1)
    pixels_outside_rect_1 = np.argwhere(outside_rect_1)
    train_dataset_1 = SubvolumeDatasetUnet(images_1, label_1, pixels_outside_rect_1,BUFFER,Z_DIM)

    image_info_2 = get_stack_images(config.PREFIX_list[2], Z_START,Z_DIM,BUFFER,config.rect_train_list[2],DEVICE,step=step_1)
    mask_2,label_2,images_2,inside_rect_2,outside_rect_2 = image_info_2
    pixels_inside_rect_2 = np.argwhere(inside_rect_2)
    pixels_outside_rect_2 = np.argwhere(outside_rect_2)
    train_dataset_2 = SubvolumeDatasetUnet(images_2, label_2, pixels_outside_rect_2,BUFFER,Z_DIM)
# train

    
    
    train_dataset_all = SubvolumeDatasetUnetMutil([train_dataset_0])
    train_loader = data.DataLoader(train_dataset_all, batch_size=BATCH_SIZE, shuffle=True)
# eval 
    eval_dataset = SubvolumeDatasetUnet(images_2, label_2, pixels_inside_rect_2,BUFFER,Z_DIM)
    #eval_dataset = SubvolumeDatasetUnet(images_2, label_2, pixels_inside_rect_2,BUFFER,Z_DIM)
    eval_loader = data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #eval_loader = data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #create model
    model = U_Net(Z_DIM)


    model=model.to(DEVICE)
    learning_rate = config.LEARNING_RATE
    TRAINING_STEPS = config.TRAINING_STEPS
    TRAINING_STEPS = config.EPCHO*len(train_loader)
    EVAL_STEPS = config.EVAL_STEPS
    #create criterion optimizer scheduler
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=TRAINING_STEPS)
    #train
    train_step = 0
    #criterion = nn.CrossEntropyLoss

    model.train()
    running_loss = 0.0
    loss_record=[]
    train_on= False
    if train_on :
        for epcho in  range (EPCHO):
            with tqdm(total=len(train_loader)) as t:
                t.set_description('Epoch %i' % epcho)
                for i, (subvolumes, inklabels) in enumerate(train_loader):
                    if train_step >= TRAINING_STEPS:
                        break
                    train_step+=1
                    optimizer.zero_grad()
                    outputs = model(subvolumes.to(DEVICE))
                    loss = criterion(outputs, inklabels.to(DEVICE))
                    #loss = criterion(nn.Sigmoid()(outputs), inklabels.to(DEVICE))
                    #acc = accuracy_score(outputs.view(-1).cpu(),inklabels.view(-1).cpu())
                    t.set_postfix(loss=loss.item())
                    t.update(1)
                    loss.backward()
                    optimizer.step()
                    scheduler.step() 
            #evalUnet(model,train_dataset_0_in,criterion,DEVICE,BATCH_SIZE)
            torch.save(model ,"./save.pt")
    model = torch.load("./save.pt")
    output = torch.zeros_like(label_2).float()
    output2 = torch.zeros_like(label_2).float()
    model.eval()
    
    with torch.no_grad():
        for i, (subvolumes, tmp_lable) in enumerate(tqdm(eval_loader)):
            for j, value in enumerate(model(subvolumes.to(DEVICE))):
                if  model_type == "base"  :      
                    output[tuple(pixels_inside_rect_2[i*BATCH_SIZE+j])] = value
                else:
                #self.label[y-BUFFER:y+BUFFER, x-BUFFER:x+BUFFER]
                    y,x = pixels_inside_rect_2[i*BATCH_SIZE+j]
                    output[y-BUFFER:y+BUFFER, x-BUFFER:x+BUFFER]+=(value.cpu()).view(BUFFER*2,BUFFER*2)
                    output2[y-BUFFER:y+BUFFER, x-BUFFER:x+BUFFER]+=1
                    

    train_image,num_image,train_lable = get_out_image(model,train_dataset_0_in,BATCH_SIZE,DEVICE)

    rect_val = config.rect_train_list[2]
    rect_train = config.rect_train_list[0]
    patch_val = patches.Rectangle((rect_val[0], rect_val[1]), rect_val[2], rect_val[3], linewidth=2, edgecolor='r', facecolor='none')
    patch_val2 = patches.Rectangle((rect_val[0], rect_val[1]), rect_val[2], rect_val[3], linewidth=2, edgecolor='r', facecolor='none')
    patch_trian = patches.Rectangle((rect_train[0], rect_train[1]), rect_train[2], rect_train[3], linewidth=2, edgecolor='b', facecolor='none')
    

    tmp_rect = config.rect_train_list[0]

    train_image_out = train_image[tmp_rect[1]:tmp_rect[1]+tmp_rect[3]+1, tmp_rect[0]:tmp_rect[0]+tmp_rect[2]]/ num_image[tmp_rect[1]:tmp_rect[1]+tmp_rect[3]+1, tmp_rect[0]:tmp_rect[0]+tmp_rect[2]]
    print(train_image_out.shape)
    train_image_out = train_image_out.reshape(-1)
    print(train_image_out.shape)
    train_image_out = np.nan_to_num(train_image_out)
    #tmpvalue = output.reshape(-1)>0.5
    tmpvalue = train_image_out>0.5
    
    #tmpvalue =tmp_lable.view(-1).cpu()
   
    tmp_lable = train_lable[tmp_rect[1]:tmp_rect[1]+tmp_rect[3]+1, tmp_rect[0]:tmp_rect[0]+tmp_rect[2]]
    print(tmp_lable.shape) 

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
    fig, ax = plt.subplots(2, 2)

   # output_1 = output/output2
    output_1 =  output/output2 >0.5
    get_sorce(label_2.cpu(),output_1,config.rect_train_list[2])

    ax[0][0].imshow(output, cmap='gray')
    #patch_trian = patches.ma
    #ax2.add_patch((patch_val,patch_trian))

    ax[0][1].imshow(label_2.cpu(), cmap='gray')
    ax[0][1].add_patch(patch_val)
    ax[1][0].imshow(train_image.cpu(), cmap='gray')
    #patch_trian = patches.ma
    #ax2.add_patch((patch_val,patch_trian))
    ax[1][1].imshow(train_lable.cpu(), cmap='gray')
    ax[1][1].add_patch(patch_trian)
    plt.show()
if __name__ == '__main__':
    train_Unet()