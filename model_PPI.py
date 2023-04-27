import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=10, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

#esm kernel=3 best 2023.1.3
class DGCNN(nn.Module):
    def __init__(self):
        super(DGCNN, self).__init__()
        self.k = 10

        self.bn_seq = nn.BatchNorm1d(32)
        self.bn_dipole = nn.BatchNorm1d(16)
        self.bn_gram = nn.BatchNorm1d(16)
        self.bn_position = nn.BatchNorm1d(32)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv_encoder = nn.Sequential(
            nn.Conv1d(in_channels=1280,
                      out_channels=32,
                      kernel_size=3, stride=1,
                      padding=3 // 2, dilation=1, groups=1,
                      bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.5)  # it was .2
        )

        self.conv_seq = nn.Sequential(nn.Conv1d(21, 32, kernel_size=3, padding=3//2, bias=False),
                                   self.bn_seq,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5))
        self.conv_dipole = nn.Sequential(nn.Conv1d(8, 16, kernel_size=3, padding=3//2, bias=False),
                                   self.bn_dipole,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5))
        self.conv_gram = nn.Sequential(nn.Conv1d(5, 16, kernel_size=3, padding=3//2, bias=False),
                                   self.bn_gram,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5))
        self.conv_position = nn.Sequential(nn.Conv1d(20, 32, kernel_size=3, padding=3//2, bias=False),
                                   self.bn_position,
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Dropout(0.5))
        self.conv1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

        self.linear1 = nn.Linear(1216, 512)
        self.linear2 = nn.Linear(512, 256)

        self.outLayer = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid())
    #预训练参数对应，迁移时pssm和position含义调换
    def forward(self, seq,esm_features, dipole, gram, position, index):
        seq = seq.squeeze(1).permute(0, 2, 1)
        dipole = dipole.squeeze(1).permute(0, 2, 1)
        gram = gram.squeeze(1).permute(0, 2, 1)
        position = position.squeeze(1).permute(0, 2, 1)
        seq = self.conv_seq(seq)
        dipole = self.conv_dipole(dipole)
        gram = self.conv_gram(gram)
        position = self.conv_position(position)
        esm_features = esm_features.squeeze(1).permute(0, 2, 1)
        esm_features = self.conv_encoder(esm_features) #(32,32,500)
        x = torch.cat((seq,esm_features,dipole,gram,position),dim=1)

        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        feature = x[0][:,index[0][0]]
        feature = feature.unsqueeze(0)

        for i in range(1,len(index)):
            temp = x[i][:,index[i][0]].unsqueeze(0)
            feature = torch.cat((feature, temp),dim=0)

        feature = F.leaky_relu(self.bn7(self.linear1(feature))) # (batch_size, emb_dims*2) -> (batch_size, 512)
        feature = self.dp1(feature)
        feature = F.leaky_relu(self.bn8(self.linear2(feature))) # (batch_size, 512) -> (batch_size, 256)
        feature = self.dp2(feature)
        output = self.outLayer(feature)

        return output