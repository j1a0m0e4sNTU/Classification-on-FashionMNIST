import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.relu  = nn.ReLU(inplace=True)        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(16, 16, kernel_size= 3, padding= 1)
        self.fc1   = nn.Linear(16*14*14, 32)
        self.fc2   = nn.Linear(32, 10)
    
    def name(self):
        return 'base_01'

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)   
        x = self.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x) 
        return x
        
def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unit_test():
    batch_size = 8
    img_batch = torch.zeros(batch_size, 1, 28, 28)
    model = Model()
    out = model(img_batch)
    print('Input size: ', img_batch.size())
    print('Output size:', out.size())

def show_info():
    model = Model()
    
    print('Parameter number: ',parameter_number(model))
    print('Parameter structure:')
    print(model)
    
if __name__ == '__main__':
    unit_test()
    print('-- Pass unit test --\n')
    show_info()