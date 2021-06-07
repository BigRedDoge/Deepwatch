import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BehaviorCloneNet(nn.Module):
    def __init__(self, num_actions):
        super(BehaviorCloneNet, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.15)
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4)
        )
        #self.conv1d = nn.Conv1d(1, 28, kernel_size=2, stride=1)

        self.fc_det = nn.Sequential(
            nn.Linear(24, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.15),
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            #nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            #nn.Dropout(p=0.15),
            nn.Linear(128, num_actions),
            #nn.Sigmoid()
        )
        self.fc1 = nn.Linear(256*41*41, 256)
        self.batch1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(24, 256)
        self.fc4 = nn.Linear(256, 128)
        self.batch2 = nn.BatchNorm1d(128)
        self.batch3 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(128, 64)
        self.batch4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, num_actions)
        
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.15)
        self.sigmoid = nn.Sigmoid()

    def forward(self, obs, det):
        obs = self.cnn1(obs)
        obs = self.cnn2(obs)
        obs = self.cnn3(obs)
        obs = self.cnn4(obs)
        obs = obs.view(obs.size(0), -1)
        #print(obs.shape)
        obs = self.fc1(obs)
        #print(obs.size())
        #obs = obs.flatten()
        #obs = self.relu(obs)
        #obs = self.batch1(obs)
        #obs = self.drop(obs)
        #obs = self.fc4(obs)
        #obs = self.relu(obs)
        #obs = self.batch2(obs)
        #obs = self.drop(obs)
        
        det = self.fc_det(det)
        #print(det.size())
        #det = det.flatten()
        #det = self.fc2(det)
        #det = self.relu(det)
        #det = self.batch3(det)
        #det = self.drop(det)
        #det = self.fc4(det)
        #det = self.relu(det)
        #det = self.batch2(det)
        #det = self.drop(det)
    

        combined = torch.cat((obs, det), dim=1)
        #print(combined.size())

        out = self.fc_combined(combined)
        split = torch.split(out, [2, 10], 1)
        mouse_out = split[0]
        actions_out = split[1]
        actions_out = self.sigmoid(actions_out)
        out = torch.cat((mouse_out, actions_out), 1)
        #out = self.fc4(combined)
        #out = self.relu(out)
        #out = self.batch2(out)
        #out = self.drop(out)
        #out = self.fc6(out)
        #out = self.relu(out)
        #out = self.batch4(out)
        #out = self.drop(out)
        #out = self.fc5(out)
        return out


class CarModel(nn.Module):
    def __init__(self):
        super(CarModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 66 x 200
            nn.Conv2d(3, 24, 5, stride=2, bias=False),
            #nn.ELU(0.2, inplace=True),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(36),
            
            nn.Conv2d(36, 48, 5, stride=2, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(48),
            
            nn.Conv2d(48, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, 3, stride=1, bias=False),
            nn.ELU(),
            nn.Dropout(p=0.4)
        )
        self.linear_layers = nn.Sequential(
            #input from sequential conv layers
            nn.Linear(in_features=92416, out_features=100, bias=False),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50, bias=False),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=25, bias=False),
            nn.ELU(),
            nn.Linear(in_features=25, out_features=12, bias=False))
        self._initialize_weights()
        
    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight, mean=1, std=0.02)
                init.constant_(m.bias, 0)

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output