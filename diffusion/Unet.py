import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTwice(nn.Module): #make a convolution twice and keep the output image in same shape (but variable channels)
    def __init__(self, c_in, c_out, c_mid = None, residual = False):
        super().__init__()
        if not c_mid:
            c_mid = c_out
        self.residual = residual

        self.conv_layer = nn.Sequential(nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1),
                                        nn.GroupNorm(1, c_mid),
                                        nn.GELU(),
                                        nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1),
                                        nn.GroupNorm(1, c_out))

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.conv_layer(x))
        else:
            return self.conv_layer(x)


class DownSample(nn.Module): #downsample to half the width and length of the image, and concat the embedded t data
    def __init__(self, c_in, c_out, embedding_dimension = 256):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2),
                                  ConvTwice(c_in, c_in, residual=True),
                                  ConvTwice(c_in, c_out))

        self.embed_t = nn.Sequential(nn.SiLU(),
                                     nn.Linear(embedding_dimension, c_out))

    def forward(self, x, t):
        x = self.down(x) # down sample x
        t = self.embed_t(t) #here, t has two dims, we need replicate the values to match to x dimensions
        t = t[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # copy x's two last dims to t for matching the shapes
        return x + t

class UpSample(nn.Module):
    def __init__(self, c_in, c_out, embedding_dimension = 256):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners= True)
        self.conv_layer = nn.Sequential(ConvTwice(c_in, c_in, residual=True),
                                        ConvTwice(c_in, c_out, c_in//2))
        self.embed_t = nn.Sequential(nn.SiLU(),
                                     nn.Linear(embedding_dimension, c_out))

    def forward(self, x, x_res, t):
        x = self.upsample(x)
        x = torch.cat((x, x_res), dim = 1) # concatenate x and the residue over the channel axis
        x = self.conv_layer(x)
        t = self.embed_t(t)
        t = t[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # copy x's two last dims to t for matching the shapes
        return x + t


class SelfAttention(nn.Module):
    def __init__(self, c_in, im_size):
        super().__init__()
        self.c_in = c_in
        self.im_size = im_size
        self.attention = nn.MultiheadAttention(c_in, 4, batch_first = True)
        self.layer_norm = nn.LayerNorm([im_size*im_size, c_in])
        self.network = nn.Sequential(nn.LayerNorm([im_size*im_size, c_in]),
                                     nn.Linear(c_in, c_in),
                                     nn.GELU(),
                                     nn.Linear(c_in, c_in))

    def forward(self, x):
        x_original = x.view(-1, self.c_in, self.im_size * self.im_size).swapaxes(1,2)
        x = self.layer_norm(x_original)
        attention, w = self.attention(x, x, x)
        attention = attention + x_original
        attention = self.network(attention) + attention
        attention = attention.swapaxes(2, 1).view(-1, self.c_in, self.im_size, self.im_size)
        return attention


class Unet(nn.Module): # im_size must be a power of 2, larger than 8.
    def __init__(self, c_in, c_out, im_size, device = "cuda", time_emb_dim = 256):
        super().__init__()
        im_size_tracker = im_size ## keep in track of the image size for the self attention inputs
        self.device = device
        self.time_emb_dim = time_emb_dim
        self.im_size = im_size
        self.c_in = c_in
        self.c_out = c_out

        # the downsampling components:
        self.conv1 = ConvTwice(c_in, 64)
        self.down1 = DownSample(64, 128)
        im_size_tracker = im_size_tracker//2
        self.att1 = SelfAttention(128, im_size_tracker)
        self.down2 = DownSample(128, 256)
        im_size_tracker = im_size_tracker//2
        self.att2 = SelfAttention(256, im_size_tracker)
        self.down3 = DownSample(256, 256)
        im_size_tracker = im_size_tracker//2
        self.att3 = SelfAttention(256, im_size_tracker)

        # now the bottleneck components
        self.mid1 = ConvTwice(256, 512)
        self.mid2 = ConvTwice(512, 512)
        self.mid3 = ConvTwice(512, 256)

        #the upsampling components:
        self.up1 = UpSample(512, 128) #notice that the "Upsample" function gets the c_in argument as twice the input, as defined. here the input is 256 so the 1st argument is 512
        im_size_tracker = im_size_tracker * 2
        self.att4 = SelfAttention(128, im_size_tracker)
        self.up2 = UpSample(256, 64)
        im_size_tracker = im_size_tracker * 2
        self.att5 = SelfAttention(64, im_size_tracker)
        self.up3 = UpSample(128, 64)
        im_size_tracker = im_size_tracker * 2
        self.att6 = SelfAttention(64, im_size_tracker)
        self.conv2 = nn.Conv2d(64, c_out, kernel_size=1)

    def t_positional_encoding(self, t, enc_dim):
        omega = torch.arange(0, enc_dim, 2, device=self.device).float()/enc_dim
        omega = 1 / (10000 ** omega)
        sin = torch.sin(t[:, None].repeat(1, enc_dim//2) * omega)
        cos = torch.cos(t[:, None].repeat(1, enc_dim//2) * omega)
        return torch.cat((sin, cos), dim = -1).type(torch.float)

    def forward(self, x, t):
        t = self.t_positional_encoding(t, self.time_emb_dim)

        #downsample:
        c1 = self.conv1(x)
        d1 = self.down1(c1, t)
        a1 = self.att1(d1)
        d2 = self.down2(a1, t)
        a2 = self.att2(d2)
        d3 = self.down3(a2, t)
        a3 = self.att3(d3)

        # bottleneck (from now without saving each outout to a different variable):
        x = self.mid1(a3)
        x = self.mid2(x)
        x = self.mid3(x)

        #upsample :
        x = self.up1(x, a2, t)
        x = self.att4(x)
        x = self.up2(x, a1, t)
        x = self.att5(x)
        x = self.up3(x, c1, t)
        x = self.att6(x)
        x = self.conv2(x)
        return x

