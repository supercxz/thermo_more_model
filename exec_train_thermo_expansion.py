#cuo wu

import numpy
import os.path
import torch
import itertools
import pandas
import util.autoencoderres as ae
import util.dopnet_thermo_expansion as dp
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from util.ml00 import get_k_folds_list
from datetime import datetime
# experiment settings
dataset_path = 'dataset/rt_df_thermo1.xlsx'
dataset_path_train = 'dataset/rt_df_thermo1_train.xlsx'
dataset_path_test='dataset/rt_df_thermo1_test.xlsx'
target_idx = 5
max_dops = 4
init_lr = 1e-2

# normalize input data
norm_target = True


# dataset loading


# list objects storing prediction results
list_test_mae = list()
list_test_rmse = list()
list_test_r2 = list()
list_preds = list()
list_embs = list()
dataset = dp.load_dataset(dataset_path, comp_idx=9, target_idx=4, max_dops=max_dops, cond_idx=None, norm_target=norm_target)
dataset_train = dp.load_dataset(dataset_path_train, comp_idx=9, target_idx=4, max_dops=max_dops, cond_idx=None, norm_target=norm_target
                          )
comps_train = [x.comp for x in dataset_train]
targets_train = numpy.array([x.target for x in dataset_train]).reshape(-1, 1)
dop_dataset_train = dp.get_dataset(dataset_train, max_dops)
data_loader_train = DataLoader(dop_dataset_train, batch_size=32, shuffle=True)
data_loader_calc = DataLoader(dop_dataset_train, batch_size=32)

# load test dataset
dataset_test = dp.load_dataset(dataset_path_test, comp_idx=9, target_idx=4, max_dops=max_dops, cond_idx=None, norm_target=norm_target
                          )
comps_test = [x.comp for x in dataset_test]
targets_test = numpy.array([x.target for x in dataset_test]).reshape(-1, 1)
dop_dataset_test = dp.get_dataset(dataset_test, max_dops)
data_loader_test = DataLoader(dop_dataset_test, batch_size=32)

# define host embedding network and its optimizer
emb_host = ae.Autoencoder(dataset[0].host_feat.shape[0], 64).cuda()
optimizer_emb = torch.optim.Adam(emb_host.parameters(), lr=1e-3, weight_decay=1e-5)

# train the host embedding network
for epoch in range(0, 300):
    train_loss = ae.train(emb_host, data_loader_train, optimizer_emb)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(epoch + 1, 300, train_loss))

now = datetime.now()
formatted_time = now.strftime("%Y%m%d%H%M%S")
model_save_folder = 'save/thermo_expansion/model/model_res' + formatted_time

if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)

    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d%H%M%S")
    # torch.save(emb_host.state_dict(), model_save_folder + '/host_embedding_res' + formatted_time + '.pt')

# calculate host embeddings
host_embs_train = ae.test(emb_host, data_loader_calc)
host_embs_test = ae.test(emb_host, data_loader_test)

# load dataset for DopNet
dop_dataset_train.host_feats = host_embs_train
dop_dataset_test.host_feats = host_embs_test
data_loader_train = DataLoader(dop_dataset_train, batch_size=32, shuffle=True)
data_loader_calc = DataLoader(dop_dataset_train, batch_size=32)
data_loader_test = DataLoader(dop_dataset_test, batch_size=32)

# define DopNet and its optimizer
pred_model = dp.DopNet(host_embs_train.shape[1], dataset[0].dop_feats.shape[1], dim_out=1, max_dops=max_dops).cuda()
optimizer = torch.optim.SGD(pred_model.parameters(), lr=init_lr, weight_decay=1e-7)
criterion = torch.nn.L1Loss()

# train DopNet
for epoch in range(0, 600):
    if (epoch + 1) % 200 == 0:
        for g in optimizer.param_groups:
            g['lr'] *= 0.5
    now = datetime.now()
    formatted_time = now.strftime("%Y%m%d%H%M%S")

    train_loss = dp.train(pred_model, data_loader_train, optimizer, criterion)
    preds_test = dp.test(pred_model, data_loader_test).cpu().numpy()
    test_loss = mean_absolute_error(targets_test, preds_test)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}\tTest loss: {:.4f}'.format(epoch + 1, 600, train_loss, test_loss))
# torch.save(pred_model.state_dict(), model_save_folder + '/DopNet_final' + formatted_time + '.pt')

# calculate predictions, embeddings, and evaluation metrics
preds_test = dp.test(pred_model, data_loader_test).cpu().numpy()
embs_test = dp.emb(pred_model, data_loader_test).cpu().numpy()
embs_train = dp.emb(pred_model, data_loader_train).cpu().numpy()
list_test_mae.append(mean_absolute_error(targets_test, preds_test))
list_test_rmse.append(numpy.sqrt(mean_squared_error(targets_test, preds_test)))
list_test_r2.append(r2_score(targets_test, preds_test))

# save prediction and embedding results to the list objects
idx_test = numpy.array([x.idx for x in dataset_test]).reshape(-1, 1)
list_preds.append(numpy.hstack([idx_test, targets_test, preds_test]))

idx_train = numpy.array([x.idx for x in dataset_train]).reshape(-1, 1)

list_embs.append(numpy.hstack([idx_test, targets_test, embs_test]))
list_embs.append(numpy.hstack([idx_train, targets_train, embs_train]))

pow = datetime.now()
formatted_time = now.strftime("%Y%m%d%H%M%S")
save_folder = 'save/thermo_expansion/result/result_res' + formatted_time
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
# pandas.DataFrame(numpy.vstack(list_unk)).to_csv(save_folder + '/preds_unk009_2.csv', header=None, index=None)
pandas.DataFrame(numpy.vstack(list_preds)).to_csv(save_folder + '/preds_id_target_predict.csv', header=None, index=None)
pandas.DataFrame(numpy.vstack(list_embs)).to_csv(save_folder + '/embs_id_target_predict.csv', header=None, index=None)

# pandas.DataFrame(numpy.vstack(list_embs_0)).to_csv(save_folder + '/embs_dopnet009_2_0.csv', header=None, index=None)
# pandas.DataFrame(numpy.vstack(list_embs_5)).to_csv(save_folder + '/embs_dopnet009_2_5.csv', header=None, index=None)
# pandas.DataFrame(numpy.vstack(list_embs_10)).to_csv(save_folder + '/embs_dopnet009_2_10.csv', header=None, index=None)
# pandas.DataFrame(numpy.vstack(list_embs_200)).to_csv(save_folder + '/embs_dopnet009_2_200.csv', header=None, index=None)
# pandas.DataFrame(numpy.vstack(list_embs_400)).to_csv(save_folder + '/embs_dopnet009_2_400.csv', header=None, index=None)

# pandas.DataFrame(numpy.vstack(preds_test)).to_csv(save_folder + '/test_pred009_2.csv', header=None, index=None)
# print evaluation results
print('Test MAE: ' + str(numpy.mean(list_test_mae)))
print('Test RMSE: ' + str(numpy.mean(list_test_rmse)))
print('Test R2: ' + str(numpy.mean(list_test_r2)))


