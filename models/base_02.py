import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace= True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace= True)
        )
        
        self.pool  = nn.MaxPool2d((2, 2), stride=(2, 2))
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace= True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace= True)
        )
        self.block_3 = nn.Sequential(
            nn.Linear(16*14*14, 32),
            nn.ReLU(inplace= True),
            nn.Linear(32, 10)
        )
    
    def name(self):
        return 'base_02'

    def forward(self, x):
        x = self.block_1(x)
        x = self.pool(x)
        x = self.block_2(x)
        x = x.view(x.size(0), -1)
        x = self.block_3(x) 
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