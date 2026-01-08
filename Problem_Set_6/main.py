import torch
import torch.nn.functional as F
from models import VGG, ResNet, ResNext
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

def train(model, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    batch_size = 128
    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    transform_base = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_base)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    #optimizer, scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.07)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0.0)

    writer=SummaryWriter(log_dir='runs/{}_adamw_cosine'.format(args.model))
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(40): 
        running_loss=0.0
        correct=0.0
        total=0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
           
            optimizer.step()
        
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        epoch_loss = running_loss / len(trainloader)
        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        #【测试集评估】
        test_loss=0.0
        test_correct=0.0
        test_total=0.0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                
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
    
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': epoch_loss,
}, f'checkpoint_epoch_{epoch+1}_{model.__class__.__name__}_2.0.pth')


def test(model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    checkpoint = torch.load(f'checkpoint_epoch_40_{model.__class__.__name__}_2.0.pth',map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)
    test_loss = 0.0
    test_correct = 0.0
    test_total = 0.0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    test_accuracy = 100 * test_correct / test_total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='The configs')
    parser.add_argument('--model',type=str,default='vgg',choices=['vgg','resnet','resnext'])
    parser.add_argument('--run', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()

    if args.model == 'vgg':
        model = VGG()   
    elif args.model == 'resnet':
        model = ResNet()
    elif args.model == 'resnext':
        model = ResNext()   
    else:
        raise AssertionError
    
    if args.run == 'train':
        train(model,args)
    elif args.run == 'test':
        test(model, args)
    else: 
        raise AssertionError
