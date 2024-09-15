from jittor import nn
from jittor import attention
import jittor as jt
import math

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.attention = attention.MultiheadAttention(dim, num_heads)
    
    def execute(self, x):
        x = x.unsqueeze(0)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.squeeze(0)
        return attn_output
    
class Adapter_MLP(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter_MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU()
        )

    def execute(self, x):
        x = self.fc(x)
        return x

class Adapter_Attention(nn.Module):
    def __init__(self, c_in, reduction=4, num_heads=8):
        super(Adapter_Attention, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(),
            Attention(c_in // reduction, num_heads),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU()
        )
    
    def execute(self, x):
        x = self.fc(x)
        return x
    
class Adapter_Conv(nn.Module):
    def __init__(self, c_in):
        super(Adapter_Conv, self).__init__()
        self.conv1 =  nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(1, 5), stride=(1,2), padding=(0,2), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 9, kernel_size=(1, 5), stride=(1,2), padding=(0,2), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(9),
        )
        fc_number = 9*math.ceil(math.ceil(c_in / 2) / 2)
        self.fc = nn.Sequential(
            nn.Linear(fc_number , c_in, bias=False),
            nn.ReLU(),
            nn.Linear(c_in, c_in, bias=False),
            nn.ReLU()
        )

    def execute(self, x):
        x = x.unsqueeze(1).unsqueeze(2)
        x = self.conv1(x)  
        x = x.view(x.size(0), -1)  
        x = self.fc(x)  
        return x
    
class Adapted_Clip(nn.Module):
    
    def __init__(self, clip_model, num_adapters, ratio=0.2, adapter_type=None):
        assert adapter_type in ['mlp', 'attn', 'conv']
        
        super(Adapted_Clip, self).__init__()
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.ratio = ratio
        if adapter_type == 'mlp':
            self.adapters = [Adapter_MLP(512) for _ in range(num_adapters)]
        elif adapter_type == 'attn':
            self.adapters = [Adapter_Attention(512) for _ in range(num_adapters)]
        elif adapter_type == 'conv':
            self.adapters = [Adapter_Conv(512) for _ in range(num_adapters)]
    
    def execute(self, image, text_features, adapter_index):
        image_features = self.image_encoder(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        adapter = self.adapters[adapter_index]
        x = adapter(image_features)
        x /= x.norm(dim=-1, keepdim=True)
        image_features = self.ratio * x + (1 - self.ratio) * image_features
        
        logits = (100 * image_features @ text_features.transpose(0, 1)).softmax(dim=-1)
        
        return logits
    
    def boost_execute(self, image, text_features, alphas):
        logits = jt.zeros((image.shape[0], text_features.shape[0]))
        for i in range(len(alphas)):
            logits += alphas[i] * self.execute(image, text_features, i)
        logits = logits.softmax(dim=-1)
        return logits