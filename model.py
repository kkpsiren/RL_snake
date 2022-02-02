from re import L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        print(f"using linear net")

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, filename="model.pth"):
        if len(filename.rsplit("/")) > 1:
            model_folder_path = filename.rsplit("/")[-1]
            if not os.path.exists(model_folder_path):
                os.makedirs(model_folder_path)
        # filename = f"{model_folder_path}/{filename}"
        torch.save(self.state_dict(), filename)

    def load(self, filename="model.pth"):
        self.load_state_dict(torch.load(filename))
        print("model loaded!")


class ConvQNet(nn.Module):
    """
    apply padding to get corners
    max-pooling or max-summing
    """

    # img (n_samples, 2 channels, 20, 20)
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # 20 - 2 + 2 * 1
        self.linear_out = nn.Linear(6 * 5 * 5, 3)
        print(f"using linear net")

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6 * 5 * 5)
        x = self.linear_out(x)
        return x

    def save(self, filename="model.pth"):
        if len(filename.rsplit("/")) > 1:
            model_folder_path = filename.rsplit("/")[-1]
            if not os.path.exists(model_folder_path):
                os.makedirs(model_folder_path)
        # filename = f"{model_folder_path}/{filename}"
        torch.save(self.state_dict(), filename)

    def load(self, filename="model.pth"):
        self.load_state_dict(torch.load(filename))
        print("model loaded!")


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=12,
        output_size=3,
        num_layers=1,
        bidirectional=True,
        dropout=0.5,
    ):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq = input_size // 4  # input_size // 4
        self.embedding = nn.Embedding(input_size, self.seq)
        self.bidirectional = bidirectional
        self.lstm = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(p=dropout)

        self.fc = nn.Linear(hidden_size * (1 + int(self.bidirectional)), output_size)
        print(f"using LSTM class")

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        x = x.view(-1, self.input_size, self.seq)
        x = torch.swapaxes(x, 1, 2)
        h0 = torch.zeros(
            self.num_layers + int(self.bidirectional), x.size(0), self.hidden_size
        )
        # c0 = torch.zeros(self.num_layers + 1, x.size(0), self.hidden_size)
        # out, _ = self.lstm(x, (h0, c0))
        out, _ = self.lstm(x, h0)
        # print(out.shape)
        out = out[:, -1, :]
        # print(out.shape)
        out = self.fc(out)
        return out

    def save(self, filename="model.pth"):
        if len(filename.rsplit("/")) > 1:
            model_folder_path = filename.rsplit("/")[-1]
            if not os.path.exists(model_folder_path):
                os.makedirs(model_folder_path)
        # filename = f"{model_folder_path}/{filename}"
        torch.save(self.state_dict(), filename)

    def load(self, filename="model.pth"):
        self.load_state_dict(torch.load(filename))
        print("model loaded!")


class ConvQTrainer:
    def __init__(self, model, learning_rate, gamma) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done,)

        # Bellman
        # 1. predicted Q with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )
            else:
                Q_new = reward[idx]
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2. Q_new = r+ y * max(next_predicted Q-value)
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()


class QTrainer:
    def __init__(self, model, learning_rate, gamma) -> None:
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done,)

        # Bellman
        # 1. predicted Q with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )
            else:
                Q_new = reward[idx]
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2. Q_new = r+ y * max(next_predicted Q-value)
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
