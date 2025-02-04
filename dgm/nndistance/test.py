import torch
from modules.nnd import NNDModule
from torch.autograd import Variable

dist = NNDModule()

p1 = torch.rand(100, 10000, 3)
p2 = torch.rand(100, 15000, 3)
points1 = Variable(p1, requires_grad=True)
points2 = Variable(p2)
dist1, dist2 = dist(points1, points2)
print(dist1, dist2)
loss = torch.sum(dist1)
print(loss)
loss.backward()
print(points1.grad, points2.grad)


points1 = Variable(p1.cuda(), requires_grad=True)
points2 = Variable(p2.cuda())
dist1, dist2 = dist(points1, points2)
print(dist1, dist2)
loss = torch.sum(dist1)
print(loss)
loss.backward()
print(points1.grad, points2.grad)
