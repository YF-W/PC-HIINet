import torchvision.models as resnet_model
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import torch.nn.init as init

torch.cuda.empty_cache()
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as resnet_model
from dropblock import DropBlock2D


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(out_features)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj):
        support = torch.matmul(features, self.weight)
        output = torch.matmul(adj, support)
        output = self.norm(output)
        output = self.activation(output)
        return output


class CompoundPoissonQKV(nn.Module):
    def __init__(self, d_model, lambda_poisson, mean_secondary, std_secondary, alpha, dropout=0.):
        super(CompoundPoissonQKV, self).__init__()
        self.d_model = d_model
        self.lambda_poisson = lambda_poisson
        self.mean_secondary = mean_secondary
        self.std_secondary = std_secondary
        self.alpha = alpha
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model))
        self.W_K = nn.Parameter(torch.randn(d_model, d_model))
        self.W_V = nn.Parameter(torch.randn(d_model, d_model))
        self.similarity_weight = nn.Parameter(torch.tensor(0.5))
        self.scale = 1.0 / (d_model ** 0.5)
        self.attend = nn.Softmax(dim=-1)

        self.gcn_0 = GCNLayer(d_model, d_model)
        self.gcn_1 = GCNLayer(d_model, d_model)
        self.gcn_2 = GCNLayer(d_model, d_model)
        self.gcn_3 = GCNLayer(d_model, d_model)

        init.xavier_uniform_(self.W_Q)
        init.xavier_uniform_(self.W_K)
        init.xavier_uniform_(self.W_V)

    def forward(self, X):
        Q = F.normalize(torch.matmul(X, self.W_Q), p=2, dim=-1)  # Normalizing Q
        K = F.normalize(torch.matmul(X, self.W_K), p=2, dim=-1)  # Normalizing K
        V = torch.matmul(X, self.W_V)

        dots = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        topk_values, topk_indices = torch.topk(dots, k=49, dim=-1)
        adj = torch.zeros_like(dots).scatter_(-1, topk_indices, topk_values)

        adj_0 = adj[0]
        adj_1 = adj[1]
        adj_2 = adj[2]
        adj_3 = adj[3]

        V_0 = V[0]
        V_1 = V[1]
        V_2 = V[2]
        V_3 = V[3]

        gcn_output_0 = self.gcn_0(V_0, adj_0)
        gcn_output_1 = self.gcn_1(V_1, adj_1)
        gcn_output_2 = self.gcn_2(V_2, adj_2)
        gcn_output_3 = self.gcn_3(V_3, adj_3)

        gcn_output = torch.stack([gcn_output_0, gcn_output_1, gcn_output_2, gcn_output_3])

        scores = self.attend(dots * self.scale)

        attention_output = torch.matmul(scores, gcn_output)

        return attention_output


class Attention(nn.Module):
    def __init__(self, dim, dropout=0., lambda_poisson=1.0, mean_secondary=0.0, std_secondary=1.0, alpha=0.5):
        super().__init__()
        self.compound_poisson_qkv = CompoundPoissonQKV(dim, lambda_poisson, mean_secondary, std_secondary, alpha)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        attn_out = self.compound_poisson_qkv(x)
        return self.to_out(attn_out)


class RegionAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, H, W):
        B, N, C = x.shape

        # 将特征图分成四个区域
        region_size = N // 4
        regions = torch.split(x, region_size, dim=1)

        # 对每个区域进行自注意力机制
        region_outs = []
        for region in regions:
            region = self.norm(region)
            qkv = self.to_qkv(region).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            region_outs.append(out)

        # 将区域特征合并
        out = torch.cat(region_outs, dim=1)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                RegionAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, H, W):
        for attn, ff in self.layers:
            x = attn(x, H, W) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_sizes, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., new_shape=None, lambda_poisson=1.0, mean_secondary=0.0,
                 std_secondary=1.0, alpha=0.5):
        super().__init__()

        self.patch_embeddings = nn.ModuleList([
            PatchEmbed(img_size=size, patch_size=patch_size, in_c=channel, embed_dim=dim)
            for size, channel, patch_size in zip(image_size, channels, patch_sizes)
        ])

        total_patches = (image_size[0] // patch_sizes[0]) * (image_size[0] // patch_sizes[0])
        self.pos_embedding = nn.Parameter(torch.randn(1, total_patches, dim * 4))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim * 4, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.conv_head = nn.Conv2d(dim * 4, num_classes, kernel_size=1) if num_classes is not None else nn.Identity()
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, img):
        patch_embeddings = [self.patch_embeddings[i](img[i]) for i in range(len(img))]
        x = torch.cat(patch_embeddings, dim=2)

        x += self.pos_embedding
        x = self.dropout(x)

        # 获取特征图的高度和宽度
        H, W = int(x.size(1) ** 0.5), int(x.size(1) ** 0.5)

        x = self.transformer(x, H, W)

        new_dim = int((x.shape[1]) ** 0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=new_dim, w=new_dim)

        x = self.down(x)
        x = self.conv_head(x)

        return x


class PatchEmbed(nn.Module):
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
    def __init__(self, dim, hidden_dim, dropout=0.):
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


class FeatureSplitter(nn.Module):
    def __init__(self, pool_kernel_size=2, pool_stride=2):
        super(FeatureSplitter, self).__init__()

    def forward(self, input_tensor):
        batch_size, channels, height, width = input_tensor.shape
        zero_tensor = torch.zeros_like(input_tensor)

        center_mask = zero_tensor.clone()
        border_mask = zero_tensor.clone()

        transition_width = height // 4
        center_mask[:, :, transition_width:height - transition_width, transition_width:width - transition_width] = 1
        border_mask[:, :, :transition_width, :] = 1
        border_mask[:, :, height - transition_width:, :] = 1
        border_mask[:, :, transition_width:height - transition_width, :transition_width] = 1
        border_mask[:, :, transition_width:height - transition_width, width - transition_width:] = 1

        center_feature = input_tensor * center_mask
        border_feature = input_tensor * border_mask

        return center_feature, border_feature


class DROPBlock(nn.Module):
    def __init__(self, drop_prob=0.3):
        super(DROPBlock, self).__init__()
        self.drop_prob = drop_prob
        self.activation = nn.ReLU()

    def forward(self, x):
        block_size = max(1, min(x.size(2), x.size(3)) // 10)
        dropblock = DropBlock2D(drop_prob=self.drop_prob, block_size=block_size)
        y = dropblock(x)
        return y


class Poolblock(nn.Module):
    def __init__(self):
        super(Poolblock, self).__init__()
        self.dropblock = DROPBlock()
        self.AvgPool = nn.AvgPool2d(2, 2)
        self.MaxPool = nn.MaxPool2d(2, 2)
        self.activation = nn.ReLU()

    def forward(self, center_feature, border_feature):
        center_feature_1 = self.MaxPool(center_feature)
        border_feature_1 = self.AvgPool(border_feature)
        AVG_OUT = center_feature_1 + border_feature_1
        AVG_OUT_b = self.dropblock(AVG_OUT)

        center_feature_2 = self.AvgPool(center_feature)
        border_feature_2 = self.MaxPool(border_feature)
        AVG_IN = center_feature_2 + border_feature_2
        AVG_IN_b = self.dropblock(AVG_IN)

        y = AVG_OUT + AVG_IN + AVG_IN * 2

        return y


class RotatingTriangularPooling:
    def __init__(self, num_rotations, mode='average'):
        self.num_rotations = num_rotations
        self.mode = mode
        self.masks_cache = None

    def _create_triangular_mask(self, shape, center, angle, pool_size):
        device = center.device
        mask = torch.zeros(shape, dtype=torch.float32, device=device)
        half_height, half_width = pool_size[0] // 2, pool_size[1] // 2

        vertices = torch.tensor([
            [center[0] - half_width, center[1] + half_height],
            [center[0] + half_width, center[1] + half_height],
            [center[0], center[1] - half_height]
        ], dtype=torch.float32, device=device)

        angle_rad = torch.deg2rad(torch.tensor(angle, dtype=torch.float32, device=device))
        rotation_matrix = torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad)],
            [torch.sin(angle_rad), torch.cos(angle_rad)]
        ], dtype=torch.float32, device=device)

        rotated_vertices = torch.matmul(vertices - center, rotation_matrix) + center

        rr, cc = torch.meshgrid(torch.arange(shape[0], device=device), torch.arange(shape[1], device=device))
        rr = rr.float()
        cc = cc.float()

        v0 = rotated_vertices[0]
        v1 = rotated_vertices[1]
        v2 = rotated_vertices[2]

        edge1 = (v1[1] - v0[1]) * (cc - v0[0]) - (v1[0] - v0[0]) * (rr - v0[1])
        edge2 = (v2[1] - v1[1]) * (cc - v1[0]) - (v2[0] - v1[0]) * (rr - v1[1])
        edge3 = (v0[1] - v2[1]) * (cc - v2[0]) - (v0[0] - v2[0]) * (rr - v2[1])

        mask[(edge1 >= 0) & (edge2 >= 0) & (edge3 >= 0)] = 1
        return mask

    def _precompute_masks(self, shape, pool_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        center = torch.tensor([shape[1] // 2, shape[0] // 2], dtype=torch.float32, device=device)
        angles = torch.linspace(0, 360, steps=self.num_rotations, device=device, dtype=torch.float32)
        masks = torch.stack([self._create_triangular_mask(shape, center, angle, pool_size) for angle in angles])
        return masks

    def pool(self, image):
        batch_size, channels, h, w = image.shape
        pool_size = (h // 2, w // 2)

        if self.masks_cache is None:
            self.masks_cache = self._precompute_masks((h, w), pool_size)

        masks = self.masks_cache.to(image.device)
        pooled_height, pooled_width = pool_size

        image_expanded = image.unsqueeze(2)  # shape (batch_size, channels, 1, height, width)
        masks_expanded = masks.unsqueeze(0).unsqueeze(1)  # shape (1, 1, num_rotations, height, width)

        regions = image_expanded * masks_expanded  # shape (batch_size, channels, num_rotations, height, width)

        pooled_regions = F.adaptive_max_pool2d(regions.view(-1, h, w), output_size=(pooled_height, pooled_width))
        pooled_regions = pooled_regions.view(batch_size, channels, self.num_rotations, pooled_height, pooled_width)

        if self.mode == 'max':
            pooled_image = pooled_regions.max(dim=2)[0]
        elif self.mode == 'average':
            pooled_image = pooled_regions.mean(dim=2)

        return pooled_image


class FWP(nn.Module):
    def __init__(self, in_channels, out_channels, num_rotations=6, mode='max'):
        super(FWP, self).__init__()
        self.pooling = RotatingTriangularPooling(num_rotations, mode)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, bias=False)

    def forward(self, x):
        pooled_tensor = self.pooling.pool(x)
        x = self.up(pooled_tensor)
        # 确保输出图像尺寸是输入图像尺寸的一半
        expected_height = x.size(2) // 2
        expected_width = x.size(3) // 2
        if pooled_tensor.size(2) != expected_height or pooled_tensor.size(3) != expected_width:
            pooled_tensor = F.interpolate(pooled_tensor, size=(expected_height, expected_width), mode='bilinear',
                                          align_corners=False)

        return pooled_tensor


class BWP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BWP, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.convv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 2, 2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.feature_splitter = FeatureSplitter()
        self.poolblock = Poolblock()

    def forward(self, x):
        # 分离特征为中心区域和边界区域
        y1, y2 = self.feature_splitter(x)

        # 应用卷积
        x1 = self.convv(x)
        x2 = self.poolblock(y1, y2)

        # 结合特征
        x = x1 + x2

        # 最后的卷积层保持输出尺寸
        x = self.conv(x)

        # 确保输出图像尺寸是输入图像的一半
        input_size = x.shape[2] * 2
        output_size = x.shape[2]
        if output_size != input_size // 2:
            # 计算所需的目标尺寸
            target_size = (input_size // 2, input_size // 2)
            # 调整输出图像的尺寸
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x


class UPConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UPConv, self).__init__()
        self.feature_splitter = FeatureSplitter()
        self.poolblock = Poolblock()
        self.dropblock = DROPBlock()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, bias=False)
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DepthwiseSeparableConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


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


class Adaptive_Pooling_Layout(nn.Module):
    def __init__(self, in_channels, out_channels, num_rotations=6, mode='average', num_modules=1):
        super(Adaptive_Pooling_Layout, self).__init__()
        self.pooling = RotatingTriangularPooling(num_rotations, mode)

        # Define a fully connected layer for global feature extraction
        self.fc = nn.Linear(in_channels * 2 * 2, out_channels)  # Assuming the input image is resized to 2x2

        # Define a transpose convolution layer if you need to upsample later
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, bias=False)

    def forward(self, x):
        # Apply pooling
        pooled_tensor = self.pooling.pool(x)

        # Ensure output image size is half of the input image size
        expected_height = x.size(2) // 2
        expected_width = x.size(3) // 2
        if pooled_tensor.size(2) != expected_height or pooled_tensor.size(3) != expected_width:
            pooled_tensor = F.interpolate(pooled_tensor, size=(expected_height, expected_width), mode='bilinear',
                                          align_corners=False)

        # Flatten the pooled tensor to extract global features
        global_features = pooled_tensor.view(pooled_tensor.size(0), -1)

        # Pass the flattened features through the fully connected layer
        global_features = self.fc(global_features)

        return global_features


class ConditionNetwork(nn.Module):
    def __init__(self, input_dim, num_modules=2):
        super(ConditionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_modules)
        self.activation = nn.Softmax(dim=1)  # Softmax across the number of modules

    def forward(self, global_features):
        x = F.relu(self.fc1(global_features))
        weights = self.activation(self.fc2(x))
        return weights


class PC_HIINet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, big_kernel=7, big_padding=3, small_kernel=5, small_padding=2,
                 channels=[24, 40, 64, 96, 192]):
        super(PC_HIINet, self).__init__()

        # ResNet
        resnet = resnet_model.resnet34(pretrained=True)
        self.global_feature_extractor=Adaptive_Pooling_Layout
        self.condition_network=ConditionNetwork
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # first down
        self.dc1_1 = DoubleConv(in_channels, channels[0], small_kernel, small_padding)
        self.down1_1 = FWP(channels[0], channels[0])
        self.dc1_2 = DoubleConv(channels[0], channels[1], small_kernel, small_padding)
        self.down1_2 = FWP(channels[1], channels[1])
        self.dc1_3 = DoubleConv(channels[1], channels[2], small_kernel, small_padding)
        self.down1_3 = BWP(channels[2], channels[2])
        self.dc1_4 = DoubleConv(channels[2], channels[3], small_kernel, small_padding)
        self.down1_4 = BWP(channels[3], channels[3])

        self.dc2_1 = DoubleConv(in_channels, channels[0], big_kernel, big_padding)
        self.down2_1 = FWP(channels[0], channels[0])
        self.dc2_2 = DoubleConv(channels[0], channels[1], big_kernel, big_padding)
        self.down2_2 = FWP(channels[1], channels[1])
        self.dc2_3 = DoubleConv(channels[1], channels[2], big_kernel, big_padding)
        self.down2_3 = BWP(channels[2], channels[2])
        self.dc2_4 = DoubleConv(channels[2], channels[3], big_kernel, big_padding)
        self.down2_4 = BWP(channels[3], channels[3])

        # first bottleneck
        self.bottleneck_1 = DoubleConv(channels[4] + 512, channels[4] * 2)

        # first up
        self.up1_1 = UPConv(channels[4], channels[3])
        self.up2_1 = UPConv(channels[4], channels[3])
        self.dc3_1 = DoubleConv(channels[4], channels[3], small_kernel, small_padding)
        self.dc4_1 = DoubleConv(channels[4], channels[3], big_kernel, big_padding)
        self.up1_2 = UPConv(channels[3], channels[2])
        self.up2_2 = UPConv(channels[3], channels[2])

        # second down
        self.dc5_1 = DoubleConv(channels[2] * 2, channels[2], small_kernel, small_padding)
        self.dc6_1 = DoubleConv(channels[2] * 2, channels[2], big_kernel, big_padding)
        self.down_3_1 = BWP(channels[2], channels[3])
        self.down_4_1 = BWP(channels[2], channels[3])
        self.dc5_2 = DoubleConv(channels[4], channels[3], small_kernel, small_padding)
        self.dc6_2 = DoubleConv(channels[4], channels[3], big_kernel, big_padding)
        self.down_3_2 = BWP(channels[3], channels[3])
        self.down_4_2 = BWP(channels[3], channels[3])

        # second bottleneck
        self.bottleneck_2 = DoubleConv(channels[4] + 512, channels[4] * 2)

        # second up
        self.up3_1 = UPConv(channels[4], channels[3])
        self.up4_1 = UPConv(channels[4], channels[3])
        self.dc7_1 = DoubleConv(channels[4], channels[3], small_kernel, small_padding)
        self.dc8_1 = DoubleConv(channels[4], channels[3], big_kernel, big_padding)
        self.up3_2 = UPConv(channels[3], channels[2])
        self.up4_2 = UPConv(channels[3], channels[2])

        # third down
        self.dc9_1 = DoubleConv(channels[2] * 2, channels[2], small_kernel, small_padding)
        self.dc10_1 = DoubleConv(channels[2] * 2, channels[2], big_kernel, big_padding)
        self.down_5_1 = BWP(channels[2], channels[3])
        self.down_6_1 = BWP(channels[2], channels[3])
        self.dc9_2 = DoubleConv(channels[4], channels[3], small_kernel, small_padding)
        self.dc10_2 = DoubleConv(channels[4], channels[3], big_kernel, big_padding)
        self.down_5_2 = BWP(channels[3], channels[3])
        self.down_6_2 = BWP(channels[3], channels[3])

        # third bottleneck
        self.bottleneck_3 = DoubleConv(channels[4], channels[4] * 2)
        self.bottleneck_3_vit = DoubleConv(channels[4] * 3, channels[4] * 2)

        # third up
        self.up5_1 = UPConv(channels[4], channels[3])
        self.up6_1 = UPConv(channels[4], channels[3])
        self.dc11_1 = DoubleConv(channels[3] * 3 + 256, channels[3], small_kernel, small_padding)
        self.dc12_1 = DoubleConv(channels[3] * 3 + 256, channels[3], big_kernel, big_padding)
        self.up5_2 = UPConv(channels[3], channels[2])
        self.up6_2 = UPConv(channels[3], channels[2])
        self.dc11_2 = DoubleConv(channels[2] * 3 + 128, channels[2], small_kernel, small_padding)
        self.dc12_2 = DoubleConv(channels[2] * 3 + 128, channels[2], big_kernel, big_padding)
        self.up5_3 = UPConv(channels[2], channels[1])
        self.up6_3 = UPConv(channels[2], channels[1])
        self.dc11_3 = DoubleConv(channels[1] * 3 + 64, channels[1], small_kernel, small_padding)
        self.dc12_3 = DoubleConv(channels[1] * 3 + 64, channels[1], big_kernel, big_padding)
        self.up5_4 = UPConv(channels[1], channels[0])
        self.up6_4 = UPConv(channels[1], channels[0])
        self.dc11_4 = DoubleConv(channels[0] * 3 + 64, channels[0], small_kernel, small_padding)
        self.dc12_4 = DoubleConv(channels[0] * 3 + 64, channels[0], big_kernel, big_padding)

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
        self.dws = DepthwiseSeparableConv(channels[0] * 2, channels[0])

        # ViT
        self.vit = ViT(image_size=[224, 112, 56, 28], patch_sizes=[8, 4, 2, 1], channels=channels, dim=192,
                       num_classes=channels[4], depth=3, heads=12, mlp_dim=512, dropout=0.1, emb_dropout=0.1,
                       new_shape=None)
        self.dws_vit = DepthwiseSeparableConv(channels[4] * 2, channels[4] * 1)

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
        x_bottleneck_1 = GSRS()(x_bottleneck_1)
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
        x_bottleneck_2 = GSRS()(x_bottleneck_2)
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
        x_bottleneck_3 = GSRS()(x_bottleneck_3)
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

    x = torch.randn(1, 3, 224, 224)
    model = PC_HIINet()
    out = model(x)
    print(out.shape)

    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print(f'Flops: {flops}, params: {params}')
