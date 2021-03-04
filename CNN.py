import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import torch.utils.data as Data
import torchvision


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())

Epoch = 50
Batch_Size = 1024
LR = 5e-3
Restore_net = 0  #  0: not restore; 1:restore net; 2:restore params;
Freeze = False
Gamma = 0.95

# define Dataload class
class LoadData():
    def dataload(self, root='./mnist', train=True, download=True):
        train_sets = torchvision.datasets.MNIST(
            root=root,
            train=train,  #if True: trainsets, False: testsets;
            transform=torchvision.transforms.ToTensor(),  # [0-255] to [0-1]
            download=download
        )
        train_data = Data.DataLoader(dataset=train_sets, batch_size=Batch_Size, shuffle=True, num_workers=0)
        return train_data


# define network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, padding=2)   #(1,28,28)
        self.Re1 = nn.ReLU()
        self.M1 = nn.MaxPool2d(kernel_size=2, stride=2)   #(24,14,14)

        self.conv2 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5, padding=2)
        self.Re2 = nn.ReLU()
        self.M2 = nn.MaxPool2d(2, 2)        #(48,7,7)

        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2)
        self.Re3 = nn.ReLU()
        self.M3 = nn.MaxPool2d(2, 2)        #(64,3,3)

        self.Fc = nn.Linear(64 * 3 * 3, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.Re1(x)
        x = self.M1(x)

        x = self.conv2(x)
        x = self.Re2(x)
        x = self.M2(x)

        x = self.conv3(x)
        x = self.Re3(x)
        x = self.M3(x)
        x = x.view(x.size(0), -1)

        x = self.Fc(x)
        return x


if Restore_net == 1:
    net = torch.load('net.pkl')
elif Restore_net == 2:
    net = CNN()
    net.load_state_dict(torch.load('net_params.pkl'))
else:
    net = CNN()

net = net.cuda(0)

print(net)

# define optimizer
optimizer = torch.optim.Adam(params=net.parameters(), lr=LR)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=Gamma)

# define loss function
loss_func = nn.CrossEntropyLoss()

# load traindata and test data
train_data = LoadData().dataload(root='./mnist', train=True, download=True)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
with torch.no_grad():
    test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)/255.
    test_y = test_data.test_labels

# train and Test
for epoch in range(Epoch):

    # Train
    ACC = 0
    Loss = 0
    if Freeze:
        break

    for step, (batch_x, batch_y) in enumerate(train_data):
        x = Variable(batch_x).cuda(0)
        y = Variable(batch_y).cuda(0)
        output = net(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            equal = torch.eq(torch.max(output, 1)[1], y)
            accuracy = torch.mean(equal.float())
        ACC += accuracy.item()
        Write_acc = ACC / (step + 1)
        Loss += loss.item()
        Write_Loss = Loss / (step + 1)
    print(f"Epoch={epoch} | loss={Write_Loss} | lr={optimizer.param_groups[0]['lr']} | Train ACC={Write_acc}")

    # test
    x = test_x.cuda(0)
    test_output = net(x)
    pred_y = torch.max(test_output.to('cpu'), 1)[1].data.numpy()
    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    print(f'Epoch={epoch} | test ACC: {accuracy}')

    # save train acc and loss in txt
    trainACCTXT = open("./Train_Acc.txt", 'a')
    trainACCTXT.write(str(Write_acc))
    trainACCTXT.write('\n')
    trainACCTXT.close()
    trainLossTXT = open("./Train_Loss.txt", 'a')
    trainLossTXT.write(str(Write_Loss))
    trainLossTXT.write('\n')
    trainLossTXT.close()
    # save test acc
    trainACCTXT = open("./Test_Acc.txt", 'a')
    trainACCTXT.write(str(accuracy))
    trainACCTXT.write('\n')
    trainACCTXT.close()

    # save model
    if Write_acc == 1.0:
        torch.save(net, 'net.pkl')   # both save net and params
        torch.save(net.state_dict(), 'net_params.pkl')   # only save params
        Freeze = True
    lr_scheduler.step()






