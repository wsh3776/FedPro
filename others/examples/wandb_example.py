import torch
import wandb
from tqdm import tqdm

# 初始化wandb
wandb.init(project="X101", name="demo", mode="run")

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out))
loss_fn = torch.nn.MSELoss(reduction='sum')

lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

pbar = tqdm(range(500), ncols=100)
for t in pbar:
    pbar.set_description(f"Processing {t + 1}-th iteration")
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    # 添加wandb图表
    wandb.log({"Train/loss": loss.item(), "round": t})

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
