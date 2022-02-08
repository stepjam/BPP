import torch


class PointFeatCNN(torch.nn.Module):
    def __init__(self, batchnorm=False):
        super(PointFeatCNN, self).__init__()
        if batchnorm:
            self.net = torch.nn.Sequential(
                    torch.nn.Conv1d(6, 64, kernel_size=1),
                    torch.nn.BatchNorm1d(64),
                    torch.nn.PReLU(),
                    torch.nn.Conv1d(64, 128, kernel_size=1),
                    torch.nn.BatchNorm1d(128),
                    torch.nn.PReLU(),
                    torch.nn.Conv1d(128, 1024, kernel_size=1),
                    torch.nn.AdaptiveMaxPool1d(output_size=1)
            )
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(6, 64, kernel_size=1),
                torch.nn.PReLU(),
                torch.nn.Conv1d(64, 128, kernel_size=1),
                torch.nn.PReLU(),
                torch.nn.Conv1d(128, 1024, kernel_size=1),
                torch.nn.AdaptiveMaxPool1d(output_size=1)
            )
    def forward(self, x):
        x = self.net(x)
        return x.squeeze()


class PointNet(torch.nn.Module):
    def __init__(self, features_dim, normalize_output=False, batchnorm=False):
        super(PointNet, self).__init__()
        self.feat_net = PointFeatCNN(batchnorm=batchnorm)
        self.normalize_output = normalize_output
        self.head = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.PReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.PReLU(),
            torch.nn.Linear(128, features_dim)
        )

    def forward(self, x):

        # Decompose input into two point clouds
        if x.dim() < 4:
            x = x.unsqueeze(dim=0)

        x_1 = x[:, 0, :, :]
        x_2 = x[:, 1, :, :]

        feats_12 = self.feat_net(torch.cat([x_1, x_2], dim=1))

        if feats_12.dim() < 2:
            feats_12 = feats_12.unsqueeze(dim=0)

        out = self.head(feats_12)

        if self.normalize_output:
            out = out / out.norm(dim=1, keepdim=True)

        return out