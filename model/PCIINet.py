import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as resnet_model

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()

        img_size = pair(img_size)
        patch_size = pair(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() 

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_sizes, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0., new_shape=None):
        super().__init__()

        self.patch_embeddings = nn.ModuleList([
            PatchEmbed(img_size=size, patch_size=patch_size, in_c=channel, embed_dim=dim)
            for size, channel, patch_size in zip(image_size, channels, patch_sizes)
        ])

        # total_patches = sum([(size // patch_size) ** 2 for size, patch_size in zip(image_size, patch_sizes)])
        self.pos_embedding = nn.ParameterList([
            nn.Parameter(torch.randn(1, (size // patch_size) ** 2, dim))
            for size, patch_size in zip(image_size, patch_sizes)
        ])

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim*4, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.pool = pool
        self.to_latent = nn.Identity()
        self.conv_head = nn.Conv2d(dim*4, num_classes, kernel_size=1) if num_classes is not None else nn.Identity()
        self.upsample = nn.Upsample(size=(new_shape, new_shape), mode='bilinear', align_corners=False) if new_shape is not None else nn.Identity()

    def forward(self, img):
        patch_embeddings = [self.patch_embeddings[i](img[i]) + self.pos_embedding[i] for i in range(len(img))]
        x = torch.cat(patch_embeddings, dim=2)

        x = self.dropout(x)
        x = self.transformer(x)

        new_dim = int((x.shape[1]) ** 0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=new_dim, w=new_dim)

        x = self.upsample(x)
        x = self.conv_head(x)

        return x

        
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UPConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DepthwiseSeparableConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class GSRS(nn.Module):
    def __init__(self):
        super(GSRS, self).__init__()

    def forward(self, x):

        batch_size, _, height, _ = x.shape

        for i in range(batch_size):
            for row in range(height):
                if row % 2 == 0:
                    even_indices = torch.arange(0, 14, 2)
                    shuffled = x[i, :, row, even_indices].clone()
                    shuffled = shuffled[:, torch.randperm(shuffled.shape[1])]
                    x[i, :, row, even_indices] = shuffled
                else:
                    odd_indices = torch.arange(1, 14, 2)
                    shuffled = x[i, :, row, odd_indices].clone()
                    shuffled = shuffled[:, torch.randperm(shuffled.shape[1])]
                    x[i, :, row, odd_indices] = shuffled

        return x


class PCIINet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, big_kernel=7, big_padding=3, small_kernel=5, small_padding=2, channels=[24, 40, 64, 96, 192]):
        super(PCIINet, self).__init__()
        
        # ResNet
        resnet = resnet_model.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # first down
        self.dc1_1 = DoubleConv(in_channels, channels[0], small_kernel, small_padding)
        self.down1_1 = DownConv(channels[0], channels[0])
        self.dc1_2 = DoubleConv(channels[0], channels[1], small_kernel, small_padding)
        self.down1_2 = DownConv(channels[1], channels[1])
        self.dc1_3 = DoubleConv(channels[1], channels[2], small_kernel, small_padding)
        self.down1_3 = DownConv(channels[2], channels[2])
        self.dc1_4 = DoubleConv(channels[2], channels[3], small_kernel, small_padding)
        self.down1_4 = DownConv(channels[3], channels[3])

        self.dc2_1 = DoubleConv(in_channels, channels[0], big_kernel, big_padding)
        self.down2_1 = DownConv(channels[0], channels[0])
        self.dc2_2 = DoubleConv(channels[0], channels[1], big_kernel, big_padding)
        self.down2_2 = DownConv(channels[1], channels[1])
        self.dc2_3 = DoubleConv(channels[1], channels[2], big_kernel, big_padding)
        self.down2_3 = DownConv(channels[2], channels[2])
        self.dc2_4 = DoubleConv(channels[2], channels[3], big_kernel, big_padding)
        self.down2_4 = DownConv(channels[3], channels[3])

        # first bottleneck
        self.bottleneck_1 = DoubleConv(channels[4]+512, channels[4]*2)

        # first up
        self.up1_1 = UPConv(channels[4], channels[3])
        self.up2_1 = UPConv(channels[4], channels[3])
        self.dc3_1 = DoubleConv(channels[4], channels[3], small_kernel, small_padding)
        self.dc4_1 = DoubleConv(channels[4], channels[3], big_kernel, big_padding)
        self.up1_2 = UPConv(channels[3], channels[2])
        self.up2_2 = UPConv(channels[3], channels[2])

        # second down
        self.dc5_1 = DoubleConv(channels[2]*2, channels[2], small_kernel, small_padding)
        self.dc6_1 = DoubleConv(channels[2]*2, channels[2], big_kernel, big_padding)
        self.down_3_1 = DownConv(channels[2], channels[3])
        self.down_4_1 = DownConv(channels[2], channels[3])
        self.dc5_2 = DoubleConv(channels[4], channels[3], small_kernel, small_padding)
        self.dc6_2 = DoubleConv(channels[4], channels[3], big_kernel, big_padding)
        self.down_3_2 = DownConv(channels[3], channels[3])
        self.down_4_2 = DownConv(channels[3], channels[3])

        # second bottleneck
        self.bottleneck_2 = DoubleConv(channels[4]+512, channels[4]*2)

        # second up
        self.up3_1 = UPConv(channels[4], channels[3])
        self.up4_1 = UPConv(channels[4], channels[3])
        self.dc7_1 = DoubleConv(channels[4], channels[3], small_kernel, small_padding)
        self.dc8_1 = DoubleConv(channels[4], channels[3], big_kernel, big_padding)
        self.up3_2 = UPConv(channels[3], channels[2])
        self.up4_2 = UPConv(channels[3], channels[2])

        # third down
        self.dc9_1 = DoubleConv(channels[2]*2, channels[2], small_kernel, small_padding)
        self.dc10_1 = DoubleConv(channels[2]*2, channels[2], big_kernel, big_padding)
        self.down_5_1 = DownConv(channels[2], channels[3])
        self.down_6_1 = DownConv(channels[2], channels[3])
        self.dc9_2 = DoubleConv(channels[4], channels[3], small_kernel, small_padding)
        self.dc10_2 = DoubleConv(channels[4], channels[3], big_kernel, big_padding)
        self.down_5_2 = DownConv(channels[3], channels[3])
        self.down_6_2 = DownConv(channels[3], channels[3])

        # third bottleneck
        self.bottleneck_3 = DoubleConv(channels[4], channels[4]*2)
        self.bottleneck_3_vit = DoubleConv(channels[4]*3, channels[4]*2)

        # third up
        self.up5_1 = UPConv(channels[4], channels[3])
        self.up6_1 = UPConv(channels[4], channels[3])
        self.dc11_1 = DoubleConv(channels[3]*3+256, channels[3], small_kernel, small_padding)
        self.dc12_1 = DoubleConv(channels[3]*3+256, channels[3], big_kernel, big_padding)
        self.up5_2 = UPConv(channels[3], channels[2])
        self.up6_2 = UPConv(channels[3], channels[2])
        self.dc11_2 = DoubleConv(channels[2]*3+128, channels[2], small_kernel, small_padding)
        self.dc12_2 = DoubleConv(channels[2]*3+128, channels[2], big_kernel, big_padding)
        self.up5_3 = UPConv(channels[2], channels[1])
        self.up6_3 = UPConv(channels[2], channels[1])
        self.dc11_3 = DoubleConv(channels[1]*3+64, channels[1], small_kernel, small_padding)
        self.dc12_3 = DoubleConv(channels[1]*3+64, channels[1], big_kernel, big_padding)
        self.up5_4 = UPConv(channels[1], channels[0])
        self.up6_4 = UPConv(channels[1], channels[0])
        self.dc11_4 = DoubleConv(channels[0]*3+64, channels[0], small_kernel, small_padding)
        self.dc12_4 = DoubleConv(channels[0]*3+64, channels[0], big_kernel, big_padding)

        # ViT part
        self.up_vit_big_1 = UPConv(channels[4], channels[3])
        self.up_vit_small_1 = UPConv(channels[4], channels[3])
        self.up_vit_big_2 = UPConv(channels[3], channels[2])
        self.up_vit_small_2 = UPConv(channels[3], channels[2])
        self.up_vit_big_3 = UPConv(channels[2], channels[1])
        self.up_vit_small_3 = UPConv(channels[2], channels[1])
        self.up_vit_big_4 = UPConv(channels[1], channels[0])
        self.up_vit_small_4 = UPConv(channels[1], channels[0])

        # final
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.dws = DepthwiseSeparableConv(channels[0]*2, channels[0])

        # ViT
        self.vit = ViT(image_size=[224, 112, 56, 28], patch_sizes=[16, 8, 4, 2], channels=channels, dim=192, num_classes=channels[4], depth=3, heads=12, mlp_dim=512, dropout=0.1, emb_dropout=0.1, new_shape=None)
        self.dws_vit = DepthwiseSeparableConv(channels[4]*2, channels[4]*1)


    def forward(self, x, channels=[24, 40, 64, 96, 192]):

        # ResNet
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e1 = self.encoder1(e0)  # torch.Size([4, 64, 112, 112])
        e2 = self.encoder2(e1)  # torch.Size([4, 128, 56, 56])
        e3 = self.encoder3(e2)  # torch.Size([4, 256, 28, 28])
        e4 = self.encoder4(e3)  # torch.Size([4, 512, 14, 14])


        big_kernel_features = []
        small_kernel_features = []
        # first down
        x_1_1 = self.dc1_1(x)
        small_kernel_features.append(x_1_1)
        x_1_1_down = self.down1_1(x_1_1)
        x_1_2 = self.dc1_2(x_1_1_down)
        small_kernel_features.append(x_1_2)
        x_1_2_down = self.down1_2(x_1_2)
        x_1_3 = self.dc1_3(x_1_2_down)
        small_kernel_features.append(x_1_3)
        x_1_3_down = self.down1_3(x_1_3)
        x_1_4 = self.dc1_4(x_1_3_down)
        small_kernel_features.append(x_1_4)
        x_1_4_down = self.down1_4(x_1_4)

        x_2_1 = self.dc2_1(x)
        big_kernel_features.append(x_2_1)
        x_2_1_down = self.down2_1(x_2_1)
        x_2_2 = self.dc2_2(x_2_1_down)
        big_kernel_features.append(x_2_2)
        x_2_2_down = self.down2_2(x_2_2)
        x_2_3 = self.dc2_3(x_2_2_down)
        big_kernel_features.append(x_2_3)
        x_2_3_down = self.down2_3(x_2_3)
        x_2_4 = self.dc2_4(x_2_3_down)
        big_kernel_features.append(x_2_4)
        x_2_4_down = self.down2_4(x_2_4)

        # first bottleneck
        x_big = self.vit(big_kernel_features)
        x_small = self.vit(small_kernel_features)

        x_vit = torch.cat((x_big, x_small), dim=1)
        x_vit = self.dws_vit(x_vit)

        x_bottleneck_1 = torch.cat((x_1_4_down, x_2_4_down, e4), dim=1)
        x_bottleneck_1 = CustomChannelShuffle()(x_bottleneck_1)
        x_bottleneck_1 = self.bottleneck_1(x_bottleneck_1)

        # first up
        x_up1_1, x_up2_1 = torch.split(x_bottleneck_1, channels[4], dim=1)
        x_up1_1_up = self.up1_1(x_up1_1)
        x_up2_1_up = self.up2_1(x_up2_1)
        x_up1_2 = self.dc3_1(torch.cat((x_up1_1_up, x_1_4), dim=1))
        x_up2_2 = self.dc4_1(torch.cat((x_up2_1_up, x_2_4), dim=1))
        x_up1_2_up = self.up1_2(x_up1_2)
        x_up2_2_up = self.up2_2(x_up2_2)

        # second down
        x_5_1 = self.dc5_1(torch.cat((x_up1_2_up, x_1_3), dim=1))
        x_6_1 = self.dc6_1(torch.cat((x_up2_2_up, x_2_3), dim=1))
        x_5_1_down = self.down_3_1(x_5_1)
        x_6_1_down = self.down_4_1(x_6_1)
        x_5_2 = self.dc5_2(torch.cat((x_5_1_down, x_up1_2), dim=1))
        x_6_2 = self.dc6_2(torch.cat((x_6_1_down, x_up2_2), dim=1))
        x_5_2_down = self.down_3_2(x_5_2)
        x_6_2_down = self.down_4_2(x_6_2)

        # second bottleneck
        x_bottleneck_2 = torch.cat((x_5_2_down, x_6_2_down, e4), dim=1)
        x_bottleneck_2 = CustomChannelShuffle()(x_bottleneck_2)
        x_bottleneck_2 = self.bottleneck_2(x_bottleneck_2)

        # second up
        x_up3_1, x_up4_1 = torch.split(x_bottleneck_2, channels[4], dim=1)
        x_up3_1_up = self.up3_1(x_up3_1)
        x_up4_1_up = self.up4_1(x_up4_1)
        x_up3_2 = self.dc7_1(torch.cat((x_up3_1_up, x_5_2), dim=1))
        x_up4_2 = self.dc8_1(torch.cat((x_up4_1_up, x_6_2), dim=1))
        x_up3_2_up = self.up3_2(x_up3_2)
        x_up4_2_up = self.up4_2(x_up4_2)


        # third down
        x_9_1 = self.dc9_1(torch.cat((x_up3_2_up, x_5_1), dim=1))
        x_10_1 = self.dc10_1(torch.cat((x_up4_2_up, x_6_1), dim=1))
        x_9_1_down = self.down_5_1(x_9_1)
        x_10_1_down = self.down_6_1(x_10_1)
        x_9_2 = self.dc9_2(torch.cat((x_9_1_down, x_up3_2), dim=1))
        x_10_2 = self.dc10_2(torch.cat((x_10_1_down, x_up4_2), dim=1))
        x_9_2_down = self.down_5_2(x_9_2)
        x_10_2_down = self.down_6_2(x_10_2)

        # third bottleneck
        x_bottleneck_3 = torch.cat((x_9_2_down, x_10_2_down), dim=1)
        x_bottleneck_3 = CustomChannelShuffle()(x_bottleneck_3)
        x_bottleneck_3 = self.bottleneck_3(x_bottleneck_3)
        x_bottleneck_3_vit = torch.cat((x_bottleneck_3, x_vit), dim=1)
        x_bottleneck_3 = self.bottleneck_3_vit(x_bottleneck_3_vit) + x_bottleneck_3

        # third up
        x_up5_1, x_up6_1 = torch.split(x_bottleneck_3, channels[4], dim=1)

        x_up5_1_up = self.up5_1(x_up5_1)
        x_up6_1_up = self.up6_1(x_up6_1)
        x_big_1 = self.up_vit_big_1(x_big)
        x_small_1 = self.up_vit_small_1(x_small)
        x_up5_2 = self.dc11_1(torch.cat((x_up5_1_up, x_9_2, x_small_1, e3), dim=1))
        x_up6_2 = self.dc12_1(torch.cat((x_up6_1_up, x_10_2, x_big_1, e3), dim=1))

        x_up5_2_up = self.up5_2(x_up5_2)
        x_up6_2_up = self.up6_2(x_up6_2)
        x_big_2 = self.up_vit_big_2(x_big_1)
        x_small_2 = self.up_vit_small_2(x_small_1)
        x_up5_3 = self.dc11_2(torch.cat((x_up5_2_up, x_9_1, x_small_2, e2), dim=1))
        x_up6_3 = self.dc12_2(torch.cat((x_up6_2_up, x_10_1, x_big_2, e2), dim=1))

        x_up5_3_up = self.up5_3(x_up5_3)
        x_up6_3_up = self.up6_3(x_up6_3)
        x_big_3 = self.up_vit_big_3(x_big_2)
        x_small_3 = self.up_vit_small_3(x_small_2)
        x_up5_4 = self.dc11_3(torch.cat((x_up5_3_up, x_1_2, x_small_3, e1), dim=1))
        x_up6_4 = self.dc12_3(torch.cat((x_up6_3_up, x_2_2, x_big_3, e1), dim=1))

        x_up5_4_up = self.up5_4(x_up5_4)
        x_up6_4_up = self.up6_4(x_up6_4)
        x_big_4 = self.up_vit_big_4(x_big_3)
        x_small_4 = self.up_vit_small_4(x_small_3)
        e0 = F.interpolate(e0, scale_factor=2, mode='bilinear', align_corners=True)
        x_up5_5 = self.dc11_4(torch.cat((x_up5_4_up, x_1_1, x_small_4, e0), dim=1))
        x_up6_5 = self.dc12_4(torch.cat((x_up6_4_up, x_2_1, x_big_4, e0), dim=1))

        # final
        temp = torch.cat((x_up5_5, x_up6_5), dim=1)
        temp = self.dws(temp)
        x = x_up5_5 + x_up6_5 + temp
        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

    x = torch.randn(4, 3, 224, 224).cuda()
    model = PCIINet().cuda()
    out = model(x)
    print(out.shape)
    
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print(f'Flops: {flops}, params: {params}')