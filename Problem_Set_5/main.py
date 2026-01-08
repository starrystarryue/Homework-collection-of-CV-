import torch
import torch.nn as nn
import argparse

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class LinearClassifier(nn.Module):
    # define a linear classifier
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # out_channels: number of categories. For CIFAR-10, it's 10
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        return self.linear(x)


class FCNN(nn.Module):
    # def a full-connected neural network classifier
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        # inchannels: dimenshion of input data. For example, a RGB image [3x32x32] is converted to vector [3 * 32 * 32], so dimenshion=3072
        # hidden_channels
        # out_channels: number of categories. For CIFAR-10, it's 10

        # full connected layer
        # activation function
        # full connected layer
        # ......
        hidden = [512, 256, 128, 64,32]
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.bn2 = nn.BatchNorm1d(hidden[1])
        self.bn3 = nn.BatchNorm1d(hidden[2])
        self.bn4 = nn.BatchNorm1d(hidden[3])
        self.bn5 = nn.BatchNorm1d(hidden[4])

        self.fc1 = nn.Linear(in_channels, hidden[0])
        self.fc2= nn.Linear(hidden[0], hidden[1])
        self.fc3= nn.Linear(hidden[1], hidden[2])
        self.fc4= nn.Linear(hidden[2], hidden[3])
        self.fc5= nn.Linear(hidden[3], hidden[4])
        self.fc6= nn.Linear(hidden[4], out_channels)

    def forward(self, x: torch.Tensor): 
        x=self.fc1(x)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.fc2(x)
        x=self.bn2(x)
        x=self.relu(x)

        x=self.fc3(x)
        x=self.bn3(x)
        x=self.relu(x)

        x=self.fc4(x)
        x=self.bn4(x)
        x=self.relu(x)

        x=self.fc5(x)
        x=self.bn5(x)
        x=self.relu(x)
        x=self.fc6(x)
        return x

class CNNClassifier(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2=nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3=nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1=nn.Linear(128*4*4, 256)
        self.fc2=nn.Linear(256, out_channels)

        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=0.5)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self,x:torch.Tensor):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)

        x=self.conv3(x)
        x=self.relu(x)
        x=self.pool(x)

        x=x.view(x.size(0), -1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x

def train(model, optimizer, scheduler, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: Cross-entropy loss
        optimizer: Adamw or SGD
        scheduler: step or cosine
        args: configuration
    '''
    ##连接虚拟机
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    writer = SummaryWriter(log_dir='runs/{}_{}_{}'.format(args.model, args.optimizer, args.scheduler))
    # create dataset
    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.12, 0.12)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 128
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    # for-loop 
    criterion = nn.CrossEntropyLoss()
    for epoch in range(50):
        running_loss=0.0
        correct=0.0
        total=0.0
        for i,data in enumerate(trainloader,0):
        # train
            # get the inputs; data is a list of [inputs, labels]
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            if isinstance(model, CNNClassifier):    ##卷积
                outputs = model(inputs)
            else:
                outputs = model(inputs.view(inputs.size(0), -1))
            # loss backward
            loss = criterion(outputs, labels)
            loss.backward()
            # optimize
            optimizer.step()
        
            running_loss += loss.item()
            # acc
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        # adjust learning rate
        scheduler.step()
        epoch_loss = running_loss / len(trainloader)
        train_accuracy = 100* correct / total
        print(f"Epoch:{epoch}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # test
            # forward
            # calculate accuracy
        model.eval()
        test_loss=0.0
        test_correct=0.0
        test_total=0.0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                if isinstance(model, CNNClassifier):
                    outputs = model(inputs)
                else:
                    outputs = model(inputs.view(inputs.size(0), -1))
                
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)
        test_accuracy = 100* test_correct / test_total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        model.train()
    writer.close()

    # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': epoch_loss,
}, f'checkpoint_epoch_{epoch+1}_fcnn_new.pth')
    
def test(model, args):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        loss_function: Cross-entropy loss
    '''
    ##连接虚拟机
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    checkpoint = torch.load('checkpoint_epoch_50_fcnn_new.pth',map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # create testing dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)
    # test
        # forward
        # calculate accuracy
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            if isinstance(model, CNNClassifier):
                outputs = model(inputs)
            else:
                outputs = model(inputs.view(inputs.size(0), -1))

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    test_accuracy =100*  test_correct / test_total
    test_loss = test_loss / len(testloader)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.4f}")
    model.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The configs')

    parser.add_argument('--run', type=str, default='train')
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='step')
    args = parser.parse_args()

    # create model
    if args.model == 'linear':
        model = LinearClassifier(in_channels=3*32*32, out_channels=10)
    elif args.model == 'fcnn':
        model = FCNN(in_channels=3*32*32,hidden_channels=512,out_channels=10)
    elif args.model == 'cnn':   ##卷积
        model = CNNClassifier(in_channels=3,out_channels=10)
    else: 
        raise AssertionError

    # create optimizer
    if args.optimizer == 'adamw':
        # create Adamw optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.007)
    elif args.optimizer == 'sgd':
        # create SGD optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0071, momentum=0.92)
    else:
        raise AssertionError
    
    # create scheduler
    if args.scheduler == 'step':
        # create torch.optim.lr_scheduler.StepLR scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.scheduler == 'cosine':
        # create torch.optim.lr_scheduler.CosineAnnealingLR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0)
    else:
        raise AssertionError

    if args.run == 'train':
        train(model, optimizer, scheduler, args)
    elif args.run == 'test':
        test(model, args)
    else: 
        raise AssertionError
    
# You need to implement training and testing function that can choose model, optimizer, scheduler and so on by command, such as:
# python main.py --run=train --model=fcnn --optimizer=adamw --scheduler=step
