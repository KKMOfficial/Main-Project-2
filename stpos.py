import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class StPosAE(nn.Module):
    def __init__(self, input_ch, hidden_ch, x_dim, z_dim):
        super(StPosAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.pre_transform = nn.Sequential(
            nn.Conv1d(in_channels=input_ch, out_channels=hidden_ch//4, kernel_size=9, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=hidden_ch//4, out_channels=hidden_ch//2, kernel_size=7, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=hidden_ch//2, out_channels=hidden_ch, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.skip_transform = nn.Conv1d(in_channels=input_ch, out_channels=hidden_ch, kernel_size=183, stride=1, padding=0)

        self.transform = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=2*z_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=2*z_dim, out_features=z_dim)
        )
    
    def forward(self, x):
        return self.transform(self.pre_transform(x)+self.skip_transform(x)).unsqueeze(1)

def train_stposae(model, train_data_loader, validation_data_loader, num_epochs, learning_rate, device, check_point_path, model_path):
    best_validation_loss = -1
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_history = []
    train_val_history = []

    # try:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for bidex,batch in enumerate(train_data_loader):
            x_batch = batch[0].to(device)
            z_batch = batch[1].to(device)
            optimizer.zero_grad()
            # first order loss
            z_hat = model(x_batch)
            loss = F.mse_loss(z_hat, z_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # print(f'Batch Index [{bidex + 1}/{num_epochs}], Partial Loss: {loss.item():.4f}')
            train_history += [ loss.item()/batch[0].shape[0] ]

        avg_loss = total_loss / len(train_data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in validation_data_loader:
                x_batch = batch[0].to(device)
                z_batch = batch[1].to(device)
                # first order loss
                z_hat = model(x_batch)
                loss = F.mse_loss(z_hat, z_batch)
                print(f'Partial Validation Loss: {loss.item():.4f}')  

            avg_val_loss = total_val_loss / len(validation_data_loader)
            print(f'>>>>>>>      Validation Loss: {avg_val_loss:.4f}')

            if avg_val_loss < best_validation_loss:
                print(f'saving model...')
                torch.save(model.state_dict(), model_path)
                best_validation_loss = avg_val_loss
            else:
                print(f'saving check-point...')
                torch.save(model.state_dict(), check_point_path)
            # log information
            train_val_history += [ [avg_loss, avg_val_loss] ]
    # except Exception as e:
    #     print(e)
        # print(f'saving model...')
        # torch.save(model.state_dict(), model_path)
        # return train_history
    return train_val_history, train_history

