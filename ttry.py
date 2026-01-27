import torch
import torch.nn.functional as F

x = torch.tensor([[1., 1.,1],
                  [1., 0.,0]])

y = F.normalize(x, p=2, dim=1)
print(y)
print("row norms:", y.norm(p=2, dim=1))