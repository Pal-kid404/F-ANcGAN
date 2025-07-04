from einops import rearrange
from torch import einsum
import torch.nn as nn
import functools
import torch
import torch.nn.functional as F


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        #Construct a PatchGAN discriminator
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        sequence += [Residual(PreNorm(ndf, LinearAttention(ndf)))]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps=1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class en_conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.LeakyReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

class style_dc_conv_block(nn.Module):
    def __init__(self, in_c, out_c,latent_dim=128):
        super().__init__()

        self.to_style1 = nn.Linear(latent_dim, in_c)
        self.to_noise1 = nn.Linear(1, out_c)
        self.conv1 = Conv2DMod(in_c, out_c, 3)

        self.to_style2 = nn.Linear(latent_dim, out_c)
        self.to_noise2 = nn.Linear(1, out_c)
        self.conv2 = Conv2DMod(out_c, out_c, 3)
        self.activation = leaky_relu()

    def forward(self, x,istyle, inoise):
        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 1, 2))
        noise2 = self.to_noise2(inoise).permute((0, 3, 1, 2))
        style1 = self.to_style1(istyle)

        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = en_conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
 
    
################################################### Deepseek
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Gate Module.
        Args:
            F_g (int): Number of channels in the decoder feature map.
            F_l (int): Number of channels in the encoder feature map.
            F_int (int): Intermediate number of channels.
        """
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Forward pass for the Attention Gate.
        Args:
            g (Tensor): Decoder feature map (lower resolution).
            x (Tensor): Encoder feature map (higher resolution).
        Returns:
            Tensor: Attention-weighted encoder feature map.
        """
        g1 = self.W_g(g)  # Transform decoder features
        x1 = self.W_x(x)  # Transform encoder features
        psi = self.relu(g1 + x1)  # Add and apply ReLU
        psi = self.psi(psi)  # Sigmoid to get attention coefficients
        #Temp = 0.4
        #psi = self.psi(psi/Temp)
        #output = torch.sigmoid(x*psi+x)
        
        #print(f"Attention coefficients min: {psi.min().item()}, max: {psi.max().item()}")
        
        #return output, psi
        return x*psi, psi # Apply attention coefficients to encoder features

 ##################################################
  


################################################
class encoder_attention_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder_attention_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        s = self.conv(x)  # Convolutional block
        p = self.pool(s)  # Max pooling
        return s, p


#############################################

#################################################### Chatgpt  
class AttentionGate2(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionGate2, self).__init__()
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # Modify this part to handle different channel sizes
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        f = self.relu(theta_x + phi_g)
        psi_f = self.sigmoid(self.psi(f))
        return x * psi_f
##########################################################################


class style_decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Input channels: in_channels (from previous) + out_channels (skip connection)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x, skip, style, noise):
        # Step 1: Upsample
        x = self.up(x)
        
        # Step 2: Ensure spatial dimensions match skip connection
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Step 3: Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)
      
class style_decoder_attention_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.attention_gate = AttentionGate(F_g=out_c, F_l=out_c, F_int=out_c // 2)  # Add attention gate
        self.conv = style_dc_conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip, istyle, inoise):
        x = self.up(inputs)  # Upsample the input feature map
        # Apply attention gate to the skip connection
        skip,psi = self.attention_gate(g=x, x=skip)
        x = torch.cat([x, skip], axis=1)  # Concatenate upsampled feature map and attention-weighted skip connection
        x = self.conv(x, istyle, inoise)  # Pass through the style and noise convolutional block
        return x
    
class StyleUnetGenerator(nn.Module):
    def __init__(self,style_latent_dim=128,style_depth=3,style_lr_mul=0.1,output_nc=3):
        super().__init__()

        """ StyleNet """
        self.latent_dim=style_latent_dim
        self.StyleNet = StyleVectorizer(emb=style_latent_dim, depth=style_depth, lr_mul=style_lr_mul)

        # Encoder (with your original channels)
        self.e1 = encoder_block(1, 64)       # -> [B, 64, H/2, W/2]
        self.e2 = encoder_block(64, 128)     # -> [B, 128, H/4, W/4]
        self.e3 = encoder_block(128, 256)    # -> [B, 256, H/8, W/8]
        self.e4 = encoder_block(256, 512)    # -> [B, 512, H/16, W/16]
       # self.e5 = encoder_block(512, 512)    # -> [B, 512, H/32, W/32] (additional)
        
        # Bottleneck with 1024 channels (as requested)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU(0.2, True)
        )
        
        # Decoder with channel matching
        self.d1 = style_decoder_block(1024, 512)  # Input:1024 + Skip:512 → Output:512
        self.d2 = style_decoder_block(512, 256)    # Input:512 + Skip:256 → Output:256
        self.d3 = style_decoder_block(256, 128)    # Input:256 + Skip:128 → Output:128
        self.d4 = style_decoder_block(128, 64)     # Input:128 + Skip:64 → Output:64

        """ Classifier """
        self.outputs = nn.Conv2d(64, output_nc, kernel_size=1, padding=0)

    def latent_to_w(self, style_vectorizer, latent_descr):
    # Debug print to verify input
    # print(f"latent_to_w input: {len(latent_descr)} tuples")
    
    # Process each z through StyleVectorizer
     styles = []
     for z, num_layers in latent_descr:
        w = style_vectorizer(z)
        #print(f"StyleVectorizer output shape: {w.shape}")  # Should be [batch, latent_dim]
        styles.append((w, num_layers))
    
     return styles

    def styles_def_to_tensor(self, styles_def):
    # Debug print
     #print(f"styles_def input: {len(styles_def)} styles")
    
    # Stack all style vectors
     style_tensors = [t for t, n in styles_def]
     stacked = torch.stack(style_tensors, dim=1)  # [batch, num_styles, latent_dim]
    
    # print(f"Stacked styles shape: {stacked.shape}")
     return stacked
    
    def forward(self, inputs,style, input_noise):
         # Verify input counts
        
        """ StyleNet """
        # Process styles
        w_space = self.latent_to_w(self.StyleNet, style)
        w_styles = self.styles_def_to_tensor(w_space)  # [batch, 4, latent_dim]
     
      # Split into 4 discrete style vectors
        style_list = [w_styles[:, i] for i in range(4)]  # Each [batch, latent_dim]
        
        # Encoder
        s1, p1 = self.e1(inputs)  # s1: [B,64,H,W], p1: [B,64,H/2,W/2]
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
       # s5, p5 = self.e5(p4)  # Additional encoder layer
        
        # Bottleneck
        b = self.bottleneck(p4)  # [B,1024,H/32,W/32]
        
        # Decoder
        d1 = self.d1(b, s4, style[0],input_noise)
        d2 = self.d2(d1, s3, style[1],input_noise)
        d3 = self.d3(d2, s2, style[2],input_noise)
        d4 = self.d4(d3, s1, style[3],input_noise)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs

class DeepSeaUp(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1=self.double_conv(x)
        x2 = self.conv1d(x)
        x=x1+x2
        x=self.bn(x)
        x=self.relu(x)
        return x


class DeepSea(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DeepSea, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.res1=ResBlock(n_channels,64)
        self.down1 = nn.MaxPool2d(2)
        self.res2 = ResBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.res3 = ResBlock(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.res4 = ResBlock(256, 512)
        self.up1 = DeepSeaUp(512, 256)
        self.res5 = ResBlock(768, 256)
        self.up2 = DeepSeaUp(256, 128)
        self.res6 = ResBlock(384, 128)
        self.up3 = DeepSeaUp(128, 64)
        self.res7 = ResBlock(192, 64)
        self.conv3 = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x1=self.res1(x)
        x2=self.down1(x1)
        x2 = self.res2(x2)
        x3 = self.down2(x2)
        x3 = self.res3(x3)
        x4 = self.down3(x3)
        x4 = self.res4(x4)

        x5=self.up1(x4,x3)
        x5 = self.res5(x5)
        x6 = self.up2(x5,x2)
        x6 = self.res6(x6)
        x7 = self.up3(x6, x1)
        x7 = self.res7(x7)
        logits=self.conv3(x7)
        return logits



class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb=256, depth=4, lr_mul = 0.1):
        super().__init__()
        self.proj = EqualLinear(emb, 256, lr_mul)  # Projection to 256D
        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.LeakyReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

############################################################## Deepseek
class decoder_attention_block(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(decoder_attention_block, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Add dropout
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention_gate = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)

    def forward(self, x, skip):
        x = self.up(x)  # Upsample
        if x.size()[2:] != skip.size()[2:]:  # Ensure spatial dimensions match
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=True)
        if self.use_attention:
            skip,psi = self.attention_gate(x, skip)  # Apply attention gate
        x = torch.cat([x, skip], dim=1)  # Concatenate with skip connection
        x = self.conv(x)  # Convolutional block
        return x,psi

##########################################################################


class UnetSegmentation(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UnetSegmentation, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        """ Encoder """
        self.e1 = encoder_block(n_channels, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = en_conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)
        
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)
        #print("Unet No of classes",self.n_classes)
        return outputs

####################################################### Deepseek

class AttentionUnetSegmentation(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, use_attention=True):
        super(AttentionUnetSegmentation, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_attention = use_attention

        """ Encoder """
        self.e1 = encoder_attention_block(n_channels, 64)
        self.e2 = encoder_attention_block(64, 128)
        self.e3 = encoder_attention_block(128, 256)
        self.e4 = encoder_attention_block(256, 512)

        """ Bottleneck """
        
        self.b = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        """ Decoder """
        self.d1 = decoder_attention_block(1024, 512, use_attention=use_attention)
        self.d2 = decoder_attention_block(512, 256, use_attention=use_attention)
        self.d3 = decoder_attention_block(256, 128, use_attention=use_attention)
        self.d4 = decoder_attention_block(128, 64, use_attention=use_attention)

        """ Classifier """
        self.outputs = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1,psi1 = self.d1(b, s4)
        d2,psi2 = self.d2(d1, s3)
        d3,psi3 = self.d3(d2, s2)
        d4,psi4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)
        return outputs,psi4

class EdgeAttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # Manual Sobel kernels
        self.register_buffer('sobel_kernel_x', torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_kernel_y', torch.tensor([
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ]).view(1, 1, 3, 3))
        
        # Edge processing
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, channels//4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels//4, channels, 3, padding=1)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.Sigmoid()
        )
    
    def apply_sobel(self, x):
        # Convert to grayscale if needed
        if x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)
        
        # Pad input
        x = F.pad(x, (1,1,1,1), mode='reflect')
        
        # Apply Sobel filters
        gx = F.conv2d(x, self.sobel_kernel_x)
        gy = F.conv2d(x, self.sobel_kernel_y)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)  # Magnitude
    
    def forward(self, x):
        # Get edges
        edges = self.apply_sobel(x)
        
        # Process edges
        edge_feats = self.edge_conv(edges)
        
        # Create attention
        attention = self.attention(torch.cat([x, edge_feats], dim=1))
        
        # Apply
        return x * attention + edge_feats

class StyleUnetGeneratorWithEdgeAttention(nn.Module):
    def __init__(self, style_latent_dim=256, style_depth=4, style_lr_mul=0.01, output_nc=3):
        super().__init__()
        self.latent_dim = style_latent_dim
        
        # Style network
        self.StyleNet = StyleVectorizer(emb=style_latent_dim, depth=style_depth, lr_mul=style_lr_mul)
        
        # Encoder
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        
        # Bottleneck
        self.b = en_conv_block(512, 1024)
        
        # Decoder with Edge Attention
        self.d1 = StyleDecoderBlockWithEdge(1024, 512)
        self.d2 = StyleDecoderBlockWithEdge(512, 256)
        self.d3 = StyleDecoderBlockWithEdge(256, 128)
        self.d4 = StyleDecoderBlockWithEdge(128, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, output_nc, 3, padding=1)
        
    def forward(self, inputs, style, input_noise):
        # Style processing
        w_space = [(self.StyleNet(z), n) for z, n in style]
        w_styles = torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in w_space], dim=1)
        
        # Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        # Bottleneck
        b = self.b(p4)
        
        # Decoder
        styles = w_styles.transpose(0, 1)
        d1 = self.d1(b, s4, styles[0], input_noise[0])
        d2 = self.d2(d1, s3, styles[1], input_noise[1])
        d3 = self.d3(d2, s2, styles[2], input_noise[2])
        d4 = self.d4(d3, s1, styles[3], input_noise[3])
        
        return self.out_conv(d4)


class StyleDecoderBlockWithEdge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            EdgeAttentionModule(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
    
    def forward(self, x, skip, style, noise):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        style = style.view(style.size(0), -1, 1, 1)
        return self.conv((x + noise) * (style + 1))
