import torch

print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)


# Other init
x = torch.empty(size=(3, 3))
x= torch.zeros((3, 3))
x = torch.rand((3, 3))
x = torch.ones((3, 3))
x = torch.eye(3, 3)

x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)

x = torch.diag(torch.ones(3))

x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))


batch = 32
n = 10
m = 20
p = 30

tensor_1 = torch.rand((batch, n, m))
tensor_2 = torch.rand((batch, m, p))

out = torch.bmm(tensor_1, tensor_2)
# print(out)

#Broadcasting (expandes dimention to 5 by copying
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2
z = x1 ** x2

sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)

z = torch.clamp(x1, min=0.2, max=0.3) #RELU
print(z)

