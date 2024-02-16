from torch import nn
class ClassifyHead(nn.Module):
    def __init__(self, in_places, num_class):
        super(ClassifyHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(in_places),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=in_places, out_features=128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_class),
        )
    def forward(self,x):
        return self.classifier(x)