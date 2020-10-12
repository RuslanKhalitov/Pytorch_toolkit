import torch

x = torch.arange(9)
#
# x3x3 = x.view(3, 3).t()
# print(x3x3)
#
# x3x32 = x.reshape(3, 3)
# print(x3x32)
#
# x3x32 = x.reshape(9)
# print(x3x32)
#
# print(x3x3.contiguous().view(9))

x1 = torch.randn(2, 3)
x2 = torch.randn(1, 3)

print(torch.cat([x1, x2], dim=0))
print(torch.cat([x1, x2], dim=0).view(-1))

batch = 32
x = torch.randn((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

print(x.permute(0, 2, 1))

x = torch.arange(10)
print(x.shape)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = x.unsqueeze(0).unsqueeze(1)

print(x.squeeze(1).shape)

