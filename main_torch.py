import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import emoji
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, cohen_kappa_score
from torch.autograd import Variable
from b_data import Dataseter
from c_module.EEGnet import EEGNet


def evaluate(model, X, Y, params=["acc"]):
    def props_to_onehot(props):
        if isinstance(props, list):
            props = np.array(props)
        a = np.argmax(props, axis=1)
        b = np.zeros((len(a), props.shape[1]))
        b[np.arange(len(a)), a] = 1
        return b

    results = []
    # batch_size = 128
    #
    # predicted = []
    #
    # for i in range(len(X) // batch_size):
    #     s = i * batch_size
    #     e = i * batch_size + batch_size
    #
    #     inputs = Variable(torch.from_numpy(X[s:e]).cuda(0))
    #     pred = model(inputs)
    #
    #     predicted.append(pred.b_data.cpu().numpy())

    inputs = Variable(torch.from_numpy(X).cuda(0))
    predicted = model(inputs)
    predicted = predicted.data.cpu().numpy()

    # print(type(predicted),predicted.shape, predicted)
    # print(type(Y),Y.shape, Y)
    # exit()

    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2 * precision * recall / (precision + recall))
        if param == "kappa":
            from sklearn import preprocessing
            enc = preprocessing.OneHotEncoder(categories='auto')
            enc.fit([[0], [1], [2]])
            results.append(cohen_kappa_score(enc.inverse_transform(Y),
                                             enc.inverse_transform(props_to_onehot(np.round(predicted)))))
    return results


batch_size = 16
print(emoji.emojize('修改好模型参数了吗? 🤷‍♂️🤷‍♂️🤦‍♂️🤦‍♂️🤦‍♂️🤦‍♂️🤦‍♂️'))
net = EEGNet(classes_num=2, num_channels=8, time_points=4096, drop_out=0.5).cuda(0)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3)

# 载入数据
dataset = Dataseter.Dataset_experiment_0408()
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# state_dict = torch.load('S1-10_500epo.pkl')
# net.load_state_dict(state_dict)
# params = ["acc", "auc", "kappa"]
# test_res = evaluate(net, test_data, test_labels, params)
# print("Test - acc", test_res[0])
# print("Test - auc", test_res[1])
# print("Test - kappa", test_res[2])
# exit()


def train(epoch):
    print("\nEpoch ", epoch)
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("Training Loss ", running_loss)


def test(max_test_res):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_dataloader:
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('accuracy on test set: %d %% ' % (100 * correct / total))
        max_test_res = max(max_test_res, (100 * correct / total))
        correct = 0
        total = 0
        for data in train_dataloader:
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('accuracy on train set: %d %% ' % (100 * correct / total))
    return max_test_res


log = 'EEGnet ' + 'exp0408' + '.pkl'
only_test = False
if __name__ == '__main__':

    if os.path.exists(log):
        checkpoint = torch.load(log)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0

    if only_test:
        test()
        exit()

    max_test = 0
    for epoch in range(start_epoch + 1, 1000):
        train(epoch)
        max_test = test(max_test)
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, log)
    print('best_test_result:', max_test)

    # 模型评价
    # params = ["acc", "auc"]
    # print(params)

    # train_res = evaluate(net, train_data, train_labels, params)
    # print("Train - acc", train_res[0])
    # print("Train - auc", train_res[1])
    # # print("Validation - ", evaluate(net, X_val, y_val, params))
    # test_res = evaluate(net, test_data, test_labels, params)
    # print("Test - acc", test_res[0])
    # print("Test - auc", test_res[1])
    # 保存参数
    # save_params = np.concatenate((save_params, [[running_loss], [train_res[0]], [test_res[0]]]), 1)

# fn = 'd_saved/bci_ii.npy'
# np.save(fn, save_params)
# 保存模型
# torch.save(net.state_dict(), 'S1-10_500epo.pkl')
