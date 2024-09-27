import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from gluefactory.models.backbones.vision_transformer import vit_base
from torchvision.transforms import Resize
from time import time
import math
from gluefactory.models.base_model import BaseModel
import numpy as np


class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """ 
    def __init__(self, mode = 'bicubic', align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)

class BasicLayer(nn.Module):
	"""
	  Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
	"""
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),
									  nn.ReLU(inplace = True),
									)

	def forward(self, x):
	    return self.layer(x)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)
    

class ResNetFPN_8_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self):
        super().__init__()
        block = BasicBlock
        initial_dim = 128
        block_dims = [128, 128, 128]

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        return x3_out, x1_out


class DinoDetector(BaseModel):
    
    default_conf = {
        "model_name": "dino-detector",
        "max_num_keypoints": -1,
        "detection_threshold": 0.05,
        "force_num_keypoints": False,
        "pretrained": True,
        "kernel_size": 5,
        "softmax_temp": 1.0
    }
    
    checkpoint_url = '/home/gochen/Documents/glue-factory/dino_extractor.pth'

    n_limit_max = 20000
    
    required_data_keys = ["image"]
    
    def _init(self, conf):

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if conf.force_num_keypoints:
            assert conf.detection_threshold <= 0 and conf.max_num_keypoints > 0
        
        self.norm = nn.InstanceNorm2d(1)
        self.interpolator = InterpolateSparse2d('bicubic')
        # build the model
        dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth", map_location="cpu")
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )
        
        dinov2_vitl14 = vit_base(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(dinov2_weights)
        self.register_buffer('dummy_buffer', torch.tensor(0))
        dinov2_vitl14.device = self.dummy_buffer.device
        self.vit = [dinov2_vitl14]

        self.image_reszie = Resize((518, 518))
        ########### ⬇️ CNN Backbone & Heads ⬇️ ###########

        self.resnet_fpn = ResNetFPN_8_2()

        self.block_fusion =  nn.Sequential(
                                        BasicLayer(128, 128, stride=1),
                                        BasicLayer(128, 128, stride=1),
                                        nn.Conv2d (128, 128, 1, padding=0)
                                        )

        self.keypoint_head = nn.Sequential(
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        nn.Conv2d (64, 65, 1),
                                    )
        
        self.heatmap_head = nn.Sequential(
                                        BasicLayer(256, 128, 1, padding=0),
										BasicLayer(128, 128, 1, padding=0),
										nn.Conv2d (128, 1, 1),
										nn.Sigmoid()
                                    )

        self.vit_head = nn.Sequential(
                                        BasicLayer(768, 256, 1, padding=0),
                                        BasicLayer(256, 128, 1, padding=0)
                                    )

        if conf.pretrained:
            state_dict = torch.load(self.checkpoint_url, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
        
    def _unfold2d(self, x, ws = 2):
        """
            Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape
        x = x.unfold(2,  ws , ws).unfold(3, ws,ws)                             \
            .reshape(B, C, H//ws, W//ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)


    def extract(self, x):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map

        """

        B, C, H, W = x.shape
        H_c, W_c = 518, 518
        if C != 3:
            x = x.repeat(1,3,1,1)

        x_vit = self.image_reszie(x)
        
        #dont backprop through normalization
        with torch.no_grad():
            x = x.mean(dim=1, keepdim = True)
            x = self.norm(x)
        
        #vit backbone
        with torch.no_grad():
            if self.vit[0].device != self.dummy_buffer.device:
                self.vit[0] = self.vit[0].to(self.dummy_buffer.device)
            dinov2_features_16 = self.vit[0].forward_features(x_vit)
            vit_feats = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,768,H_c//14, W_c//14)
            del dinov2_features_16

        #main backbone
        x3, x1 = self.resnet_fpn(x)

        #pyramid fusion
        x1 = F.interpolate(x1, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        vit_feats = F.interpolate(vit_feats, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        vit_feats = self.vit_head(vit_feats)

        feats = self.block_fusion( x3 + x1)

        feats = torch.cat([feats, vit_feats], dim=1)

        #heads
        
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits
        heatmap = self.heatmap_head(feats)
        
        return feats, keypoints, heatmap
    
    def get_kpts_heatmap(self, kpts):
        softmax_temp = self.conf.softmax_temp
        scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
        return heatmap

    def NMS(self, x):
        threshold = self.conf.detection_threshold
        kernel_size = self.conf.kernel_size
        B, _, H, W = x.shape
        pad=kernel_size//2
        local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
        pos = (x == local_max) & (x > threshold)
        pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

        pad_val = max([len(x) for x in pos_batched])
        pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

        #Pad kpts and build (B, N, 2) tensor
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b]

        return pos
    
    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        if isinstance(x, np.ndarray) and len(x.shape) == 3:
            x = torch.tensor(x).permute(2,0,1)[None]
        x = x.to(self.dev).float()

        H, W = x.shape[-2:]
        _H, _W = (H//32) * 32, (W//32) * 32
        rh, rw = H/_H, W/_W

        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw
    
    def _forward(self, data):
        image = data["image"]
        
        if self.conf.force_num_keypoints:
            top_k = self.conf.max_num_keypoints
        else:
            top_k = self.n_limit_max
        
        detection_threshold = self.conf.detection_threshold
        
        x, rh1, rw1 = self.preprocess_tensor(image)

        B, _, _H1, _W1 = x.shape

        feats, keypoints, heatmap = self.extract(x)
        feats = F.normalize(feats, dim=1)

        #Convert logits to heatmap and extract kpts
        K1h = self.get_kpts_heatmap(keypoints)
        mkpts = self.NMS(K1h)

        #Compute reliability scores
        _nearest = InterpolateSparse2d('nearest')
        _bilinear = InterpolateSparse2d('bilinear')
        scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(heatmap, mkpts, _H1, _W1)).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        #Select top-k features
        idxs = torch.argsort(-scores)
        mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)[:, :top_k]
        mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)[:, :top_k]
        mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, :top_k]

        #Interpolate descriptors at kpts positions
        feats = self.interpolator(feats, mkpts, H = _H1, W = _W1)

        #L2-Normalize
        feats = F.normalize(feats, dim=-1)

        #Correct kpt scale
        mkpts = mkpts * torch.tensor([rw1,rh1], device=mkpts.device).view(1, 1, -1)
        
        return {
            "keypoints": mkpts,
            "scores": scores,
            "descriptors": feats,
        }
    
    def loss(self, pred, data):
            raise NotImplementedError