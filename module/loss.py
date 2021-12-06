import torch.nn as nn
import torch.nn.functional as F
import torch

class attn_loss(nn.Module):
    def __init__(self, option):
        super(attn_loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.option = option
        self.loss_list = self.option.result['train']['attn_loss']
    
    def forward(self, output, label, loss_cls1, attn_features=None):
        loss = 0.
        
        loss_cls1 = torch.tensor(loss_cls1).cuda()
        
        # Classification Loss with Attention
        loss_cls2 = self.CE(output, label)
        
        if 'CE' in self.loss_list:
            loss += loss_cls2
        
        # Superior Loss
        if 'Superior' in self.loss_list:
            loss_superior = loss_cls2 - loss_cls1
            loss += loss_superior       
        
        # Revisit Loss
        if attn_features is not None:    
            if 'Revisit' in self.loss_list:
                for attn in attn_features:
                    if attn[0] is not None:
                        c_attn = torch.mean(attn[0], dim=[2,3])
                        c_loss = torch.mean(1 - c_attn)
                        loss += c_loss
                    
                    if attn[1] is not None:
                        s_attn = torch.mean(attn[1], dim=1)
                        s_loss = torch.mean(1 - s_attn)
                        loss += s_loss

        return loss