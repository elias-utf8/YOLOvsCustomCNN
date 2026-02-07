import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiObjectDetector(nn.Module):
    """
    CNN avec deux têtes :
    - Classification : présence/absence de chaque classe (cylindre, cube)
    - Régression : coordonnées des bounding boxes (max 2 objets)
    """
    
    def __init__(self, num_classes=2):  # cylindre=0, cube=1
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone partagé

        # En entrée images de taille : [B, 3, 224, 224]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)  # 224 -> 112
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)  # 112 -> 56
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)  # 56 -> 28
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2)  # 28 -> 14
        
        # Taille après backbone: [B, 128, 14, 14] = 25088
        flatten_size = 128 * 14 * 14
        
        # Tête de classification (présence de chaque classe)
        self.cls_head = nn.Sequential(
            nn.Linear(flatten_size, 256),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(256, num_classes)  # Sortie: [cylindre_present, cube_present]
        )

        # Tête de régression (2 bboxes max: une par classe)
        # Sortie: [x1, y1, w1, h1, x2, y2, w2, h2] = 8 valeurs
        self.reg_head = nn.Sequential(
            nn.Linear(flatten_size, 256),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(256, num_classes * 4)  # 2 classes × 4 coords = 8
        )

        self._init_reg_head()

    
    def forward(self, x):
        # Backbone activation
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Deux sorties
        cls_output = self.cls_head(x)           # [B, 2] logits
        reg_output = self.reg_head(x)           # [B, 8] coords

        reg_output = torch.sigmoid(reg_output)  # Sigmoid pour normaliser dans [0, 1]
        
        # Reshape reg_output pour avoir une dimension par classe: [B, num_classes, 4]
        reg_output = reg_output.view(-1, self.num_classes, 4)
        
        return cls_output, reg_output
    

    def _init_reg_head(self):
        for module in self.reg_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.001)
                nn.init.zeros_(module.bias)
