import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 定义一个简单的GCN模型
class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 构建简单的图数据
edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
x = torch.randn(3, 5)  # 3个节点，每个节点有5个特征
target = torch.tensor([0, 1, 0], dtype=torch.long)  # 根据具体任务提供标签

data = Data(x=x, edge_index=edge_index)

print("x = ", x)

# 初始化并训练模型
model = SimpleGCN(in_channels=5, hidden_channels=16, out_channels=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    out = model(data.x, data.edge_index)
    if epoch % 10 == 0:
        print("out = ", out)
    loss = criterion(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 提取节点嵌入
with torch.no_grad():
    model.eval()
    node_embeddings = model.conv1(data.x, data.edge_index)

print("x = ", x)

# node_embeddings即为提取得到的节点嵌入
print("Node Embeddings:")
print(node_embeddings)
