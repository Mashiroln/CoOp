import torch
from torch import nn
from tqdm import tqdm


class PseudoSampler(nn.Module):
    def __init__(self, out_size, penaly=1000):
        super(PseudoSampler, self).__init__()
        self.x = torch.nn.Parameter(torch.randn(out_size))
        self.penaly = penaly
        self.image_features = None
        self.text_feature = None
        self.delta_text_feature = None

    @torch.no_grad()
    def build_image_feature(self, image_feature):
        image_feature = torch.unsqueeze(image_feature, dim=0)
        if self.image_features is None:
            self.image_features = image_feature
        else:
            self.image_features = torch.cat((self.image_features, image_feature), dim=0)
        self.image_features = self.image_features.detach()

    @torch.no_grad()
    def build_text_feature(self, normal_feature, anomaly_feature):
        self.text_feature = torch.cat((normal_feature, anomaly_feature), dim=0).permute(1, 0)
        self.delta_text_feature = self.text_feature[:, 0] - self.text_feature[:, 1]
        self.delta_text_feature = self.delta_text_feature.detach()

    def cal_constraints(self):
        expanded_x = self.x.unsqueeze(0).expand(self.image_features.size(0), -1)
        return torch.sum((self.image_features + expanded_x) @ self.delta_text_feature)

    def loss_fn(self):
        obj = -self.objective()
        constraint_penaly = self.penaly * torch.relu(self.cal_constraints())
        return obj + constraint_penaly

    def objective(self):
        l2 = torch.norm(self.x, 2)
        return l2


class PseudoTrainer:
    def __init__(self, out_size, data, epochs=100, lr=0.01, wd=0.0001):
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.model = PseudoSampler(out_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.01)
        self.train_loader = data['train_loader']
        self.val_loader = data['val_loader']

    def train(self):
        for epoch in range(0, self.epochs):
            self.scheduler.step()

            train_loss = 0.0
            valid_loss = 0.0

            self.model.train()
            for batch, (img, target) in enumerate(tqdm(self.train_loader)):
                self.optimizer.zero_grad()
                loss = self.model(img)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * img.size(0)

            self.model.eval()
            for batch, (img, target) in enumerate(self.val_loader):
                loss = self.model(img)
                valid_loss += loss.item() * img.size(0)

            train_loss /= len(self.train_loader.dataset)
            valid_loss /= len(self.val_loader.dataset)
            print(f'Epoch: {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
