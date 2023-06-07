import torch.utils.data as data


class SubvolumeDataset(data.Dataset):
    def __init__(self, image_stack, label, pixels,buffer,z_dim):
        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels
        self.BUFFER = buffer
        self.Z_DIM = z_dim
    def __len__(self):
        return len(self.pixels)
    def __getitem__(self, index):
        y, x = self.pixels[index]
        subvolume = self.image_stack[:, y-self.BUFFER:y+self.BUFFER+1, x-self.BUFFER:x+self.BUFFER+1].view(1, self.Z_DIM, self.BUFFER*2+1, self.BUFFER*2+1)
        inklabel = self.label[y, x].view(1)
        return subvolume, inklabel

class SubvolumeDatasetUnet(data.Dataset):
    def __init__(self, image_stack, label, pixels,buffer,z_dim):
        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels
        self.buffer = buffer
        self.z_dim = z_dim
    def __len__(self):
        return len(self.pixels)
    def __getitem__(self, index):
        y, x = self.pixels[index]
        subvolume = self.image_stack[:, y-self.buffer:y+self.buffer, x-self.buffer:x+self.buffer].view(self.z_dim, self.buffer*2, self.buffer*2)
        inklabel = self.label[y-self.buffer:y+self.buffer, x-self.buffer:x+self.buffer].view(1,self.buffer*2, self.buffer*2)
        return subvolume, inklabel  
    
class SubvolumeDatasetUnetMutil(data.Dataset):
    def __init__(self, dataSetList:SubvolumeDatasetUnet):
        self.len_list = [0]
        tmp_len = 0
        for data_set in dataSetList:
            tmp_len+=len(data_set)
            self.len_list.append(tmp_len)
        self.dataSetList = dataSetList
        
    def __len__(self):
        return self.len_list[-1]
        
    def __getitem__(self, index):
        target_id = len(self.len_list)-1
        for i,index_target in enumerate(self.len_list[::-1]):
            if index<index_target:
                target_id =target_id- i
                subvolume, inklabel = self.dataSetList[target_id-1][index -self.len_list[target_id-1]]
                return subvolume, inklabel
        return None