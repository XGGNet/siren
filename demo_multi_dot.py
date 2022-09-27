import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
import cv2
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time

import pdb

def get_mgrid(sidelen, dim=2, img_idx=0):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = torch.cat((mgrid, img_idx * torch.ones((sidelen, sidelen, 1))),-1)
    mgrid = mgrid.reshape(-1, dim+1)
    return mgrid

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
    
def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_tensor(sidelength, img):    
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

class ImageFitting(Dataset):
    def __init__(self, sidelength, path, pics):
        super().__init__()
        # img = get_cameraman_tensor(sidelength)
        self.num = len(pics)
        self.grid = []
        self.target = []
        img_idx = 0
        for pic in pics:
            # img = cv2.imread((path+'/'+pic), cv2.IMREAD_GRAYSCALE)
            # img = Image.fromarray(img)
            img = Image.open((path+'/'+pic))
            img = get_tensor(sidelength, img)
            self.grid.append(get_mgrid(sidelength, 2, img_idx))
            self.target.append(img.permute(1, 2, 0).view(-1, 1))
            img_idx += 1

    def __len__(self):
        return self.num

    def __getitem__(self, idx): 
        return self.grid[idx], self.target[idx]

size = 256
path ="pic"
pics =os.listdir(path) 
pics.sort()

img_grid = ImageFitting(size, path, pics)

# dataloader = DataLoader(img_grid, batch_size=1, pin_memory=True, num_workers=0)

img_siren = Siren(in_features=3, out_features=1, hidden_features=256, 
                  hidden_layers=3, outermost_linear=True)
img_siren.cuda()

total_steps = 1000 # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 50

optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

input, pixel_ground_truth = img_grid.__getitem__(0)
input, pixel_ground_truth = input.unsqueeze(0).cuda(), pixel_ground_truth.unsqueeze(0).cuda()
for i in range(4):
    model_input, ground_truth = img_grid.__getitem__(i+1)
    model_input, ground_truth = model_input.unsqueeze(0).cuda(), ground_truth.unsqueeze(0).cuda()
    input = torch.cat((input, model_input), 0)
    pixel_ground_truth = torch.cat((pixel_ground_truth, ground_truth), 0)

pdb.set_trace()

for step in range(total_steps):

    output, coords = img_siren(input)    
    loss = ((output - pixel_ground_truth)**2).mean()
    
    if not step % steps_til_summary:
        print("Step %d, Total loss %0.6f" % (step, loss))
        input_dot = input
        for i in range(5):
            input_dot[i,:,2] = 4-i
        output_new, coords_new = img_siren(input_dot)
        for i in range(5):
            file_path = 'result_dot_reverse/'+ str(step) + '/'
            file_name = file_path + str(i+1) +'.jpg'
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            plt.imshow(output_new[i,:,:].cpu().view(size,size).detach().numpy())
            plt.savefig(file_name)
        # img_grad = gradient(model_output, coords)
        # img_laplacian = laplace(model_output, coords)

        # fig, axes = plt.subplots(1,3, figsize=(18,6))
        # plt.imsave(model_output.cpu().view(size,size).detach().numpy())
        # axes[1].imshow(img_grad.norm(dim=-1).cpu().view(size,size).detach().numpy())
        # axes[2].imshow(img_laplacian.cpu().view(size,size).detach().numpy())
        # plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()