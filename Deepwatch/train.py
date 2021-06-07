import torch
import torch.nn as nn
import signal
import glob
import sys
from model import BehaviorCloneNet, CarModel
from torchvision.transforms import Compose, ToTensor, Normalize
from logloader import LogLoader


device = torch.device("cuda:0")wwas

def main():
    model = BehaviorCloneNet(12)
    #model = CarModel()
    model.to(device)
    ckpt_path = './checkpoints/deepwatch_13.pth'
    checkpoint = torch.load(ckpt_path)
    log_dir = './logging/'
    framesz = 360
    batch_size = 32
    lr = 0.001
    epochs = 100

    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    transform_train = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    
    train_data = LogLoader(log_dir, framesz, transform_train)
    val_data = LogLoader(log_dir, framesz, transform_train, is_val=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                               shuffle=True, num_workers=16, 
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, 
                                               shuffle=False, num_workers=16, 
                                               pin_memory=True)

    criterion = nn.MSELoss().to(device)
    #criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    print("Begining training...")
    for epoch in range(0, epochs): #checkpoint['epoch']
        train(train_loader, model, criterion, optimizer, epoch)
        validate(val_loader, model, criterion, epoch)

        if (epoch + 1) % 5 == 0:
            print("Saving Checkpoint")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, (frame_tensor, detect_tensor, target) in enumerate(train_loader, 0):
        frame_tensor = frame_tensor.to(device, dtype=torch.float)
        detect_tensor = detect_tensor.to(device, dtype=torch.float)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(frame_tensor, detect_tensor)
        #output = model(frame_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print("Train Epoch: [{},{}], Loss: {}".format(epoch + 1, i + 1, running_loss))
            running_loss = 0.0

def validate(val_loader, model, criterion, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (frame_tensor, detect_tensor, target) in enumerate(val_loader, 0):
            frame_tensor = frame_tensor.to(device, dtype=torch.float)
            detect_tensor = detect_tensor.to(device, dtype=torch.float)
            target = target.to(device)

            output = model(frame_tensor, detect_tensor)
            #output = model(frame_tensor)
            loss = criterion(output, target)

            pred = output.cpu().to(device)
            pred = pred.t()
