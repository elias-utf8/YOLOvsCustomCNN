import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    """
    Combine deux losses :
    - Classification : BCE pour présence/absence
    - Régression : MSE uniquement sur les objets présents
    """
    
    def __init__(self, cls_weight=1.0, reg_weight=5.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.reg_loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self, cls_pred, reg_pred, cls_target, reg_target):
        """
            cls_pred: [B, num_classes] logits
            reg_pred: [B, num_classes, 4] coords prédites
            cls_target: [B, num_classes] présence (0 ou 1)
            reg_target: [B, num_classes, 4] coords réelles
        """
        # Loss de classification
        cls_loss = self.cls_loss_fn(cls_pred, cls_target)
        
        # Loss de régression (seulement si objet présent)
        
        # [1, 0] Si le cylindre est présent par exemple
        mask = cls_target.unsqueeze(-1)  
        
        # La loss de régression est calculée pour tous les éléments, mais on ne garde que ceux où il y a un objet
        reg_loss_all = self.reg_loss_fn(reg_pred, reg_target)  # [B, num_classes, 4]


        # Exemple : 
        # [0.02, 0.05, 0.01, 0.03] × [1, 1, 1, 1] = [0.02, 0.05, 0.01, 0.03]   gardé
        # [0.80, 0.90, 0.70, 0.60] × [0, 0, 0, 0] = [0.00, 0.00, 0.00, 0.00]   effacé
        reg_loss_masked = reg_loss_all * mask  # Zéro si pas d'objet
        
        # Moyenne sur les objets présents uniquement
        num_objects = mask.sum() + 1e-6  # Éviter division par 0
        reg_loss = reg_loss_masked.sum() / num_objects
        
        # Loss totale
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        
        return total_loss, cls_loss, reg_loss