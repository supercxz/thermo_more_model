import torch
from torch import nn
from d2l import torch as d2l
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 返回训练损失和训练精度
def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)

        y_mean = y.mean()
        tss = ((y - y_mean) ** 2).sum().item()
        rss = ((y_hat - y) ** 2).sum().item()
        # 此时没有意义
        if tss == 0 or rss == 0:
            r2 = 1
        else:
            r2 = 1 - (rss / tss)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        #print(f'%%%%%%%{y.numel()}')
        metric.add(float(l.mean()), float(r2), 1)
    # 返回训练损失
    return metric[0] / metric[2], metric[1] / metric[2]

def train_decay(net, train_features, test_features, train_labels, test_labels, batch_size, loss,
          num_epochs=400, isbias = True, weight_decay = 0):
    train_iter = d2l.load_array((train_features, train_labels),
                                batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        train_loss, train_r2 = train_epoch_ch3(net, train_iter, loss, trainer)

    net.eval()
    pre = net(test_features)
    mse = loss(pre, test_labels).mean().item()

    y_mean = test_labels.mean()
    tss = ((test_labels - y_mean) ** 2).sum().item()
    rss = ((pre - test_labels) ** 2).sum().item()
    r2 = 1 - (rss / tss)
    print(f'神经网络模型【原数据】MSE:{mse:.5f},r2: {r2:.5f}')
    # print('weight:', net[0].weight.data.numpy())
    return mse, r2
