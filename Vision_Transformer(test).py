import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#==========================================================================================
# Convolution 구조 없이 patch 사이즈로 분할 하는 법
# x = torch.randn(8, 3, 224, 224)
# print(x.shape) # torch.Size([8, 3, 224, 224])



# patch_size = 16 # 16 pixels

# print('x :', x.shape) # x : torch.Size([8, 3, 224, 224])
# patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
# b c (h s1) (w s2) = 8 3 (14*16) (14*16)
# b (h w) (s1 s2 c) = 8 (14*14) (16*16*3)
# print('patches :', patches.shape) # patches : torch.Size([8, 196, 768])
# einops의 rearrange를 통해 Batch,C,H,W의 텐서를 
# Batch,N,Psize로 바꿔줄 수 있다.
#==========================================================================================



#==========================================================================================
# 하지만 실제의 ViT에서는 einops같은 Linear Embedding이 아니라 
# kernal size와 stride size를 patch size로 갖는 Convolutional 2D Layer를 
# 이용한 후 flatten 시켜줍니다. 이렇게 하면 performance gain이 있다고 저자는 말합니다.
# x = torch.randn(8, 3, 224, 224)
# patch_size = 16
# in_channels = 3
# emb_size = 768

# projection = nn.Sequential(
#             # using a conv layer instead of a linear one -> performance gains
#             nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
#             Rearrange('b e (h) (w) -> b (h w) e'),
#         )

# # print(projection(x).shape) # torch.Size([8, 196, 768])


# emb_size = 768
# img_size = 224
# patch_size = 16

# # 이미지를 패치사이즈로 나누고 flatten
# projected_x = projection(x)
# print('Projected X shape :', projected_x.shape)
# # Projected X shape : torch.Size([8, 196, 768])

# # cls_token과 pos encoding Parameter 정의
# cls_token = nn.Parameter(torch.randn(1,1, emb_size))
# positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
# print('Cls Shape :', cls_token.shape, ', Pos Shape :', positions.shape)
# # Cls Shape : torch.Size([1, 1, 768]) , Pos Shape : torch.Size([197, 768])

# # cls_token을 반복하여 배치사이즈의 크기와 맞춰줌
# batch_size = 8
# cls_tokens = repeat(cls_token, '() n e -> b n e', b=batch_size)
# print('Repeated Cls shape :', cls_tokens.shape)
# # Repeated Cls shape : torch.Size([8, 1, 768])

# # cls_token과 projected_x를 concatenate
# cat_x = torch.cat([cls_tokens, projected_x], dim=1)
# # dim 이 0일 경우 행을 기준으로 붙인다 ex) (2,3) + (2,3) = (4,3)
# # dim 이 1일 경우 열을 기준으로 붙인다 ex) (2,3) + (2,3) = (2,6)

# # position encoding을 더해줌
# cat_x += positions
# print('output : ', cat_x.shape)
# # output :  torch.Size([8, 197, 768])
#==========================================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions

        return x
      
# emb_size = 768
# num_heads = 8

# keys = nn.Linear(emb_size, emb_size)
# queries = nn.Linear(emb_size, emb_size)
# values = nn.Linear(emb_size, emb_size)
# # print('key :',keys, 'queries :',queries, 'values :',values)
# # key :     Linear(in_features=768, out_features=768, bias=True) 
# # queries : Linear(in_features=768, out_features=768, bias=True) 
# # values :  Linear(in_features=768, out_features=768, bias=True)

# queries = rearrange(queries(x), "b n (h d) -> b h n d", h=num_heads)
# keys = rearrange(keys(x), "b n (h d) -> b h n d", h=num_heads)
# values  = rearrange(values(x), "b n (h d) -> b h n d", h=num_heads)

# print('shape :', queries.shape, keys.shape, values.shape)
# shape : 
# torch.Size([8, 8, 197, 96]) 
# torch.Size([8, 8, 197, 96]) 
# torch.Size([8, 8, 197, 96])

# # Queries * Keys
# energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
# print('energy :', energy.shape)
# # energy : torch.Size([8, 8, 197, 197])

# # Get Attention Score
# scaling = emb_size ** (1/2)
# att = F.softmax(energy, dim=-1) / scaling
# print('att :', att.shape)
# # att : torch.Size([8, 8, 197, 197])

# # Attention Score * values
# out = torch.einsum('bhal, bhlv -> bhav ', att, values)
# print('out :', out.shape)
# # out : torch.Size([8, 8, 197, 96])

# # Rearrage to emb_size
# out = rearrange(out, "b h n d -> b n (h d)")
# print('out2 : ', out.shape)
# # out2 :  torch.Size([8, 197, 768])

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
      
    
# print(x.shape) # torch.Size([8, 197, 768])
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

x = torch.randn(8, 3, 224, 224)
patches_embedded = PatchEmbedding()(x)
# print(TransformerEncoderBlock()(patches_embedded).shape)
# torch.Size([8, 197, 768
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))


class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from torchvision import utils
import numpy as np
import time
import copy
path2data = 'D:\study_data\_data\\test108/'

# if not exists the path, make the directory
if not os.path.exists(path2data):
    os.mkdir(path2data)

# load dataset
train_ds = datasets.STL10(path2data, split='train', download=False, transform=transforms.ToTensor())
val_ds = datasets.STL10(path2data, split='test', download=False, transform=transforms.ToTensor())

print(len(train_ds))
print(len(val_ds))

# define transformation
transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(224),
                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# apply transformation to dataset
train_ds.transform = transformation
val_ds.transform = transformation

# make dataloade
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=True)

# check sample images
def show(img, y=None):
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg_tr)

    if y is not None:
        plt.title('labels:' + str(y))

np.random.seed(10)
torch.manual_seed(0)

grid_size=4
rnd_ind = np.random.randint(0, len(train_ds), grid_size)

x_grid = [train_ds[i][0] for i in rnd_ind]
y_grid = [val_ds[i][1] for i in rnd_ind]

x_grid = utils.make_grid(x_grid, nrow=grid_size, padding=2)
plt.figure(figsize=(10,10))
show(x_grid, y_grid)

x = torch.randn(16,3,224,224).to(device)
model = ViT().to(device)
output = model(x)
print(output.shape)

loss_func = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=0.01)

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()
    
    return loss_b.item(), metric_b


# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b
        
        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data
    return loss, metric
# function to start training
def train_val(model, params):
    num_epochs=params['num_epochs']
    loss_func=params['loss_func']
    opt=params['optimizer']
    train_dl=params['train_dl']
    val_dl=params['val_dl']
    sanity_check=params['sanity_check']
    lr_scheduler=params['lr_scheduler']
    path2weights=params['path2weights']

    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr= {}'.format(epoch, num_epochs-1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print('Copied best model weights!')

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %(train_loss, val_loss, 100*val_metric, (time.time()-start_time)/60))
        print('-'*10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history
# define the training parameters
params_train = {
    'num_epochs':100,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_dl,
    'val_dl':val_dl,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/weights.pt',
}

# check the directory to save weights.pt
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')
createFolder('./models')

model, loss_hist, metric_hist = train_val(model, params_train)



