import numpy as np
import matplotlib.pyplot as plt
import torch
from math import ceil
from torchvision.utils import make_grid, save_image
import os
import properscoring as ps
import torchmetrics
import lpips
from torchmetrics_wrap import FrechetVideoDistance
# from timm.models import create_model
# import models_class
import tempfile
import logging
 
# from fvcore.common.checkpoint import Checkpointer
from signal import pause

save_image_name="encoder-distill-8-11-new-loss-"
# mcvd_preds=torch.from_numpy(np.load('/opt/data/private/mcvd-pytorch/TCGdiff_pred_v3.npy'))
# mcvd_gt=torch.from_numpy(np.load('/opt/data/private/mcvd-pytorch/TCGdiff_gt_v3.npy'))
# mcvd_cond=torch.from_numpy(np.load('/opt/data/private/mcvd-pytorch/TCGdiff_cond_v3.npy'))

#predrenn.shape  84,8,1,128,128
# predrnn_preds=torch.from_numpy(np.load('/opt/data/private/earth-forecasting-transformer/src/predrnn/predrnn_pred_traj-24-15.npy')).squeeze(-1).unsqueeze(2).to(torch.device("cuda")).float()[:,4:]

# predrnn_gt=torch.from_numpy(np.load('/opt/data/private/earth-forecasting-transformer/src/predrnn/predrnn_gt-3.npy'))
# predrnn_cond=torch.from_numpy(np.load('/opt/data/private/earth-forecasting-transformer/src/predrnn/predrnn_cond-3.npy'))

#phydnet.shape  84,8,1,128,128
# phydnet_preds=torch.from_numpy(np.load('/opt/data/private/earth-forecasting-transformer/src/phydnet/phydnet_pred_traj-24-15.npy')).squeeze(-1).unsqueeze(2).to(torch.device("cuda")).float()[:,4:]
# phydnet_gt=torch.from_numpy(np.load('/opt/data/private/earth-forecasting-transformer/src/phydnet/phydnet_gt-3.npy'))
# phydnet_cond=torch.from_numpy(np.load('/opt/data/private/earth-forecasting-transformer/src/phydnet/phydnet_cond-3.npy'))TCGdiff_preds=torch.from_numpy(np.load(f'/opt/data/private/mcvd-pytorch/TCGdiff_pred_traj-24-3-lx+era5-{save_image_name}.npy')).squeeze(-1).unsqueeze(2).to(torch.device("cuda")).float()[:,4:]# 

#gt.shape  84,8,1,128,128
#earthformer_preds.shape  84,8,1,128,128
# earthformer_preds=torch.from_numpy(np.load('/opt/data/private/earth-forecasting-transformer/src/earthformer/earthformer_pred_traj-24-15.npy')).squeeze(-1).unsqueeze(2).to(torch.device("cuda")).float()[:,4:]
# earthformer_gt=torch.from_numpy(np.load('/opt/data/private/earth-forecasting-transformer/src/earthformer/earthformer_gt.npy'))
# earthformer_cond=torch.from_numpy(np.load('/opt/data/private/earth-forecasting-transformer/src/earthformer/earthformer_cond.npy'))

#prediff_preds.shape  84,8,1,128,128
# prediff_preds=torch.from_numpy(np.load('/opt/data/private/PreDiff/Prediff_pred_traj-24-15.npy')).squeeze(-1).unsqueeze(2).to(torch.device("cuda")).float()[:,4:]
# prediff_gt=torch.from_numpy(np.load('/opt/data/private/PreDiff/prediff_gt.npy')).squeeze(-1)
# prediff_cond=torch.from_numpy(np.load('/opt/data/private/PreDiff/prediff_cond.npy')).squeeze(-1)

#TCGdiff_preds.shape  84,8,1,128,128
TCGdiff_preds=torch.from_numpy(np.load(f'/opt/data/private/mcvd-pytorch2/mcvd-pytorch/{save_image_name}.npy')).squeeze(-1).unsqueeze(2).to(torch.device("cuda")).float()[:,4:]# 

# #gt.shape  84,8,1,128,128          lx 的  TCGdiff_gt_traj-24-3-lx+era5-{save_image_name}  wyx的 
gt=torch.from_numpy(np.load(f'/opt/data/private/mcvd-pytorch2/mcvd-pytorch/{save_image_name}-gt.npy')).squeeze(-1).unsqueeze(2).to(torch.device("cuda")).float()[:,4:]
# TCGdiff_preds_nonauto=torch.from_numpy(np.load('/opt/data/private/mcvd-pytorch/TCGdiff_pred_nonauto.npy')).squeeze(-1).unsqueeze(2).to(torch.device("cuda")).float().expand(-1, -1,3, -1, -1)
# TCGdiff_gt=torch.from_numpy(np.load('/opt/data/private/mcvd-pytorch/TCGdiff_gt_1.npy'))
# TCGdiff_cond=torch.from_numpy(np.load('/opt/data/private/mcvd-pytorch/TCGdiff_cond_1.npy'))

# TCGdiff_preds_2=torch.from_numpy(np.load('/opt/data/private/mcvd-pytorch/TCGdiff_pred_2.npy'))
# TCGdiff_gt_2=torch.from_numpy(np.load('/opt/data/private/mcvd-pytorch/TCGdiff_gt_2.npy'))
# TCGdiff_cond_2=torch.from_numpy(np.load('/opt/data/private/mcvd-pytorch/TCGdiff_cond_2.npy'))
# TCGdiff_pred_traj=torch.from_numpy(np.load('/opt/data/private/mcvd-pytorch/TCGdiff_pred_traj.npy')).unsqueeze(2)
# TCGdiff_gt_traj=torch.from_numpy(np.load('/opt/data/private/mcvd-pytorch/TCGdiff_gt_traj.npy')).unsqueeze(2)

# TCGdiff_cond_traj=TCGdiff_gt_traj[:,:4]
# TCGdiff_gt_traj=TCGdiff_gt_traj[:,4:]
# TCGdiff_pred_traj=TCGdiff_pred_traj[:,4:]
# TCGdiff_pred_traj=torch.stack([TCGdiff_pred_traj[9],TCGdiff_pred_traj[70],TCGdiff_pred_traj[81],
#                              TCGdiff_pred_traj[94]],dim=0)
# TCGdiff_gt_traj=torch.stack([TCGdiff_gt_traj[9],TCGdiff_gt_traj[70],TCGdiff_gt_traj[81],
#                              TCGdiff_gt_traj[94]],dim=0)
# lpips_model = lpips.LPIPS(net='alex')
# mcvd_preds_lpips = mcvd_preds.expand(-1, -1, 3, -1, -1).float()
# predrnn_preds_lpips = predrnn_preds.expand(-1, -1, 3, -1, -1).float()
# phydnet_preds_lpips = phydnet_preds.expand(-1, -1, 3, -1, -1).float()
# earthformer_preds_lpips = earthformer_preds.expand(-1, -1, 3, -1, -1).float()
# earthformer_gt_lpips = earthformer_gt.expand(-1, -1, 3, -1, -1).float()
# print(TCGdiff_preds.size())
# print(prediff_preds.size())
# print(predrnn_preds.size())
# print(phydnet_preds.size())
# print(earthformer_preds.size())
print("=============LPIPS====================")
lpips_model = lpips.LPIPS(net='alex').to(torch.device("cuda"))
TCGdiff_preds_lpips = TCGdiff_preds.expand(-1, -1, 3, -1, -1).float()
# prediff_preds_lpips = prediff_preds.expand(-1, -1, 3, -1, -1).float()
# predrnn_preds_lpips = predrnn_preds.expand(-1, -1, 3, -1, -1).float()
# phydnet_preds_lpips = phydnet_preds.expand(-1, -1, 3, -1, -1).float()
# earthformer_preds_lpips = earthformer_preds.expand(-1, -1, 3, -1, -1).float()
gt_lpips = gt.expand(-1, -1, 3, -1, -1).float()

# lpips_scores = []
# for i in range(TCGdiff_preds_lpips.size(0)):  # Iterate over the batch
#     for j in range(TCGdiff_preds_lpips.size(1)):  # Iterate over the sequence length
#         lpips_value = lpips_model(prediff_preds_lpips[i, j], gt_lpips[i, j])
#         lpips_scores.append(lpips_value.item())
# mean_lpips_score = torch.tensor(lpips_scores).mean()
# print(f"prediff LPIPS Score: {mean_lpips_score}")


lpips_scores = []
for i in range(TCGdiff_preds_lpips.size(0)):  # Iterate over the batch
    for j in range(TCGdiff_preds_lpips.size(1)):  # Iterate over the sequence length
        lpips_value = lpips_model(TCGdiff_preds_lpips[i, j], gt_lpips[i, j])
        lpips_scores.append(lpips_value.item())
mean_lpips_score = torch.tensor(lpips_scores).mean()
print(f"TCGdiff LPIPS Score: {mean_lpips_score}")

# lpips_scores = []
# for i in range(TCGdiff_preds_lpips.size(0)):  # Iterate over the batch
#     for j in range(TCGdiff_preds_lpips.size(1)):  # Iterate over the sequence length
#         lpips_value = lpips_model(predrnn_preds_lpips[i, j], gt_lpips[i,j])
#         lpips_scores.append(lpips_value.item())
# mean_lpips_score = torch.tensor(lpips_scores).mean()
# print(f"predrnn LPIPS Score: {mean_lpips_score}")

# lpips_scores = []
# for i in range(TCGdiff_preds_lpips.size(0)):  # Iterate over the batch
#     for j in range(TCGdiff_preds_lpips.size(1)):  # Iterate over the sequence length
#         lpips_value = lpips_model(phydnet_preds_lpips[i, j], gt_lpips[i, j])
#         lpips_scores.append(lpips_value.item())
# mean_lpips_score = torch.tensor(lpips_scores).mean()
# print(f"phydnet LPIPS Score: {mean_lpips_score}")

# lpips_scores = []
# for i in range(TCGdiff_preds_lpips.size(0)):  # Iterate over the batch
#     for j in range(TCGdiff_preds_lpips.size(1)):  # Iterate over the sequence length
#         lpips_value = lpips_model(earthformer_preds_lpips[i,j], gt_lpips[i, j])
#         lpips_scores.append(lpips_value.item())
# mean_lpips_score = torch.tensor(lpips_scores).mean()
# print(f"earthformer LPIPS Score: {mean_lpips_score}")






print("=============SSIM====================")
ssim = torchmetrics.StructuralSimilarityIndexMeasure().to(torch.device("cuda"))

# ssim_score=0
# for i in range(TCGdiff_preds.size(1)):
#     ssim_score+=ssim(prediff_preds[:,i],gt[:,i])
# # ssim_score+=ssim(prediff_preds[:,3],gt[:,3])
# print("prediff_ssim_score",ssim_score/TCGdiff_preds.size(1))

ssim_score=0
for i in range(TCGdiff_preds.size(1)):
    ssim_score+=ssim(TCGdiff_preds[:,i],gt[:,i])
# ssim_score+=ssim(TCGdiff_preds[:,3],gt[:,3])
print("TCGdiff_ssim_score",ssim_score/TCGdiff_preds.size(1))
# ssim_score=0
# for i in range(predrnn_preds.size(1)):
#     ssim_score+=ssim(predrnn_preds[:,i],gt[:,i])
# # ssim_score+=ssim(predrnn_preds[:,3],gt[:,3])
# print("predrnn_ssim_score",ssim_score/TCGdiff_preds.size(1))
# ssim_score=0
# for i in range(phydnet_preds.size(1)):
#     ssim_score+=ssim(phydnet_preds[:,i],gt[:,i])
# # ssim_score+=ssim(phydnet_preds[:,3],gt[:,3])
# print("phydnet_ssim_score",ssim_score/TCGdiff_preds.size(1))
# ssim_score=0
# for i in range(earthformer_preds.size(1)):
#     ssim_score+=ssim(earthformer_preds[:,i],gt[:,i])
# # ssim_score+=ssim(earthformer_preds[:,3],gt[:,3])
# print("earthformer_ssim_score",ssim_score/TCGdiff_preds.size(1))

# print(prediff_preds.size())
# print(TCGdiff_preds.size())
# print(predrnn_preds.size())
# print(phydnet_preds.size())
# print(earthformer_preds.size())
# X.reshape(len(X), -1, ch, imsize, imsize).permute(0, 2, 1, 4, 3).reshape(len(X), ch, -1, imsize).permute(0, 1, 3, 2)
# prediff_preds=prediff_preds.reshape(len(prediff_preds), -1, 1, prediff_preds.size(3), prediff_preds.size(4)).permute(0, 2, 1, 4, 3).reshape(len(prediff_preds), 1, -1, prediff_preds.size(3)).permute(0, 1, 3, 2)
# prediff_gt=prediff_gt.reshape(len(prediff_gt), -1, 1, prediff_gt.size(2), prediff_gt.size(3)).permute(0, 2, 1, 4, 3).reshape(len(prediff_gt), 1, -1, prediff_gt.size(2)).permute(0, 1, 3, 2)
# prediff_cond=prediff_cond.reshape(len(TCGdiff_cond), -1, 1, TCGdiff_cond.size(2), TCGdiff_cond.size(3)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_cond), 1, -1, TCGdiff_cond.size(2)).permute(0, 1, 3, 2)

# TCGdiff_preds_2=TCGdiff_preds_2.reshape(len(TCGdiff_preds_2), -1, 1, TCGdiff_preds_2.size(2), TCGdiff_preds_2.size(3)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_preds_2), 1, -1, TCGdiff_preds_2.size(2)).permute(0, 1, 3, 2)
# TCGdiff_gt_2=TCGdiff_gt_2.reshape(len(TCGdiff_gt_2), -1, 1, TCGdiff_gt_2.size(2), TCGdiff_gt_2.size(3)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_gt_2), 1, -1, TCGdiff_gt_2.size(2)).permute(0, 1, 3, 2)
# TCGdiff_cond_2=TCGdiff_cond_2.reshape(len(TCGdiff_cond_2), -1, 1, TCGdiff_cond_2.size(2), TCGdiff_cond_2.size(3)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_cond_2), 1, -1, TCGdiff_cond_2.size(2)).permute(0, 1, 3, 2)

# TCGdiff_preds=TCGdiff_preds.reshape(len(TCGdiff_preds), -1, 1, TCGdiff_preds.size(3), TCGdiff_preds.size(4)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_preds), 1, -1, TCGdiff_preds.size(3)).permute(0, 1, 3, 2)
# TCGdiff_gt=TCGdiff_gt.reshape(len(TCGdiff_gt), -1, 1, TCGdiff_gt.size(2), TCGdiff_gt.size(3)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_gt), 1, -1, TCGdiff_gt.size(2)).permute(0, 1, 3, 2)
# TCGdiff_cond=TCGdiff_cond.reshape(len(TCGdiff_cond), -1, 1, TCGdiff_cond.size(2), TCGdiff_cond.size(3)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_cond), 1, -1, TCGdiff_cond.size(2)).permute(0, 1, 3, 2)

# TCGdiff_preds_2=TCGdiff_preds_2.reshape(len(TCGdiff_preds_2), -1, 1, TCGdiff_preds_2.size(2), TCGdiff_preds_2.size(3)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_preds_2), 1, -1, TCGdiff_preds_2.size(2)).permute(0, 1, 3, 2)
# TCGdiff_gt_2=TCGdiff_gt_2.reshape(len(TCGdiff_gt_2), -1, 1, TCGdiff_gt_2.size(2), TCGdiff_gt_2.size(3)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_gt_2), 1, -1, TCGdiff_gt_2.size(2)).permute(0, 1, 3, 2)
# TCGdiff_cond_2=TCGdiff_cond_2.reshape(len(TCGdiff_cond_2), -1, 1, TCGdiff_cond_2.size(2), TCGdiff_cond_2.size(3)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_cond_2), 1, -1, TCGdiff_cond_2.size(2)).permute(0, 1, 3, 2)

# TCGdiff_pred_traj=TCGdiff_pred_traj.reshape(len(TCGdiff_pred_traj), -1, 1, TCGdiff_pred_traj.size(2), TCGdiff_pred_traj.size(3)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_pred_traj), 1, -1, TCGdiff_pred_traj.size(2)).permute(0, 1, 3, 2)
# TCGdiff_gt_traj=TCGdiff_gt_traj.reshape(len(TCGdiff_gt_traj), -1, 1, TCGdiff_gt_traj.size(2), TCGdiff_gt_traj.size(3)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_gt_traj), 1, -1, TCGdiff_gt_traj.size(2)).permute(0, 1, 3, 2)

# mcvd_preds=mcvd_preds.reshape(len(mcvd_preds), -1, 1, mcvd_preds.size(2), mcvd_preds.size(3)).permute(0, 2, 1, 4, 3).reshape(len(mcvd_preds), 1, -1, mcvd_preds.size(2)).permute(0, 1, 3, 2)
# mcvd_gt=mcvd_gt.reshape(len(mcvd_gt), -1, 1, mcvd_gt.size(2), mcvd_gt.size(3)).permute(0, 2, 1, 4, 3).reshape(len(mcvd_gt), 1, -1, mcvd_gt.size(2)).permute(0, 1, 3, 2)
# mcvd_cond=mcvd_cond.reshape(len(mcvd_cond), -1, 1, mcvd_cond.size(2), mcvd_cond.size(3)).permute(0, 2, 1, 4, 3).reshape(len(mcvd_cond), 1, -1, mcvd_cond.size(2)).permute(0, 1, 3, 2)
# # print("mcvd_preds",mcvd_preds.size())
# predrnn_preds=predrnn_preds.reshape(len(predrnn_preds), -1, 1, predrnn_preds.size(3), predrnn_preds.size(4)).permute(0, 2, 1, 4, 3).reshape(len(predrnn_preds), 1, -1, predrnn_preds.size(3)).permute(0, 1, 3, 2)
# predrnn_gt=predrnn_gt.permute(0,1,4,2,3).permute(0, 2, 1, 4, 3).reshape(len(predrnn_gt), 1, -1, predrnn_gt.size(2)).permute(0, 1, 3, 2)
# predrnn_cond=predrnn_cond.permute(0,1,4,2,3).permute(0, 2, 1, 4, 3).reshape(len(predrnn_cond), 1, -1, predrnn_cond.size(2)).permute(0, 1, 3, 2)
# print("predrnn_preds",predrnn_preds.size())

# phydnet_preds=phydnet_preds.reshape(len(phydnet_preds), -1, 1, phydnet_preds.size(3), phydnet_preds.size(4)).permute(0, 2, 1, 4, 3).reshape(len(phydnet_preds), 1, -1, phydnet_preds.size(3)).permute(0, 1, 3, 2)
# phydnet_gt=phydnet_gt.permute(0,1,4,2,3).permute(0, 2, 1, 4, 3).reshape(len(phydnet_gt), 1, -1, phydnet_gt.size(2)).permute(0, 1, 3, 2)
# phydnet_cond=phydnet_cond.permute(0,1,4,2,3).permute(0, 2, 1, 4, 3).reshape(len(phydnet_cond), 1, -1, phydnet_cond.size(2)).permute(0, 1, 3, 2)
# print("phydnet_preds",phydnet_preds.size())

# earthformer_preds=earthformer_preds.reshape(len(earthformer_preds), -1, 1, earthformer_preds.size(3), earthformer_preds.size(4)).permute(0, 2, 1, 4, 3).reshape(len(earthformer_preds), 1, -1, earthformer_preds.size(3)).permute(0, 1, 3, 2)

# gt=gt.reshape(len(gt), -1, 1, gt.size(3), gt.size(4)).permute(0, 2, 1, 4, 3).reshape(len(gt), 1, -1, gt.size(3)).permute(0, 1, 3, 2)
# earthformer_gt=earthformer_gt.permute(0,1,4,2,3).permute(0, 2, 1, 4, 3).reshape(len(earthformer_gt), 1, -1, earthformer_gt.size(2)).permute(0, 1, 3, 2)
# earthformer_cond=earthformer_cond.permute(0,1,4,2,3).permute(0, 2, 1, 4, 3).reshape(len(earthformer_cond), 1, -1, earthformer_cond.size(2)).permute(0, 1, 3, 2)

# print("earthformer_preds",earthformer_preds.size())

padding = 0.5*torch.ones(len(TCGdiff_preds), 1, TCGdiff_preds.size(-2), 2)

nrow = ceil(np.sqrt((12)*84)/(12))
# print(nrow)
# print(padding.size())
# print(TCGdiff_preds_nonauto.size())
# prediff_preds=(prediff_preds-prediff_preds.min())/(prediff_preds.max()-prediff_preds.min())
# prediff_gt=(prediff_gt-prediff_gt.min())/(prediff_gt.max()-prediff_gt.min())
# prediff_cond=(prediff_cond-prediff_cond.min())/(prediff_cond.max()-prediff_cond.min())
# TCGdiff_preds_nonauto=(TCGdiff_preds_nonauto-TCGdiff_preds_nonauto.min())/(TCGdiff_preds_nonauto.max()-TCGdiff_preds_nonauto.min())
TCGdiff_preds=(TCGdiff_preds-TCGdiff_preds.min())/(TCGdiff_preds.max()-TCGdiff_preds.min())

gt = (gt - gt.min())/(gt.max() - gt.min())
# TCGdiff_gt_2=(TCGdiff_gt_2-TCGdiff_gt_2.min())/(TCGdiff_gt_2.max()-TCGdiff_gt_2.min())
# TCGdiff_cond_2=(TCGdiff_cond_2-TCGdiff_cond_2.min())/(TCGdiff_cond_2.max()-TCGdiff_cond_2.min())

# TCGdiff_preds=(TCGdiff_preds-TCGdiff_preds.min())/(TCGdiff_preds.max()-TCGdiff_preds.min())
# TCGdiff_gt=(TCGdiff_gt-TCGdiff_gt.min())/(TCGdiff_gt.max()-TCGdiff_gt.min())
# TCGdiff_cond=(TCGdiff_cond-TCGdiff_cond.min())/(TCGdiff_cond.max()-TCGdiff_cond.min())

# TCGdiff_pred_traj=(TCGdiff_pred_traj-TCGdiff_pred_traj.min())/(TCGdiff_pred_traj.max()-TCGdiff_pred_traj.min())
# TCGdiff_gt_traj=(TCGdiff_gt_traj-TCGdiff_gt_traj.min())/(TCGdiff_gt_traj.max()-TCGdiff_gt_traj.min())
# TCGdiff_cond_traj=(TCGdiff_cond_traj-TCGdiff_cond_traj.min())/(TCGdiff_cond_traj.max()-TCGdiff_cond_traj.min())
# mcvd_preds=(mcvd_preds-mcvd_preds.min())/(mcvd_preds.max()-mcvd_preds.min())
# mcvd_gt=(mcvd_gt-mcvd_gt.min())/(mcvd_gt.max()-mcvd_gt.min())
# mcvd_cond=(mcvd_cond-mcvd_cond.min())/(mcvd_cond.max()-mcvd_cond.min())

# for i in range(TCGdiff_pred_traj.size(0)):
#     for j in range(TCGdiff_pred_traj.size(1)):
#         # plt.imshow(TCGdiff_pred_traj[i,j], cmap='plasma', interpolation='nearest')
#         # plt.colorbar()
#         fig, ax = plt.subplots()
#         cmap = plt.get_cmap('jet')  # 获取'jet'色彩映射
#         cax = ax.imshow(TCGdiff_pred_traj[i, j], cmap=cmap)
#         fig.colorbar(cax)  # 添加颜色条
#         ax.set_title('Colorbar for TCGdiff_pred_traj[{},{}]'.format(i, j))

#         # 保存图像和颜色条
#         plt.savefig('/opt/data/private/mcvd-pytorch/TCGdiff_traj_{}_{}.png'.format(i, j), dpi=300)

        # plt.imsave('/opt/data/private/mcvd-pytorch/TCGdiff_pred_traj_{}_{}.png'.format(i,j),TCGdiff_pred_traj[i,j], cmap='jet',  dpi=300)
   
# noisy_image=mcvd_gt[27,2]
# def add_diffusion_noise(image, noise_level, beta):
#     noise = np.random.randn(*image.shape) * noise_level
#     noisy_image = np.sqrt(1 - beta) * image + np.sqrt(beta) * noise
#     noisy_image = np.clip(noisy_image, 0, 1)
#     return noisy_image
# beta_values = np.linspace(0, 0.1, 100) 
# noisy_images=[]
# for beta in beta_values:
#     noisy_image = add_diffusion_noise(noisy_image, noise_level=1.0, beta=beta)
#     noisy_images.append(noisy_image)

# for i in range(len(noisy_images)):
#     plt.imsave('/opt/data/private/mcvd-pytorch/gt_27_2_{}.png'.format(i),noisy_images[i], cmap='jet',  dpi=300)





# predrnn_preds=(predrnn_preds-predrnn_preds.min())/(predrnn_preds.max()-predrnn_preds.min())
# # predrnn_gt=(predrnn_gt-predrnn_gt.min())/(predrnn_gt.max()-predrnn_gt.min())
# # predrnn_cond=(predrnn_cond-predrnn_cond.min())/(predrnn_cond.max()-predrnn_cond.min())

# phydnet_preds=(phydnet_preds-phydnet_preds.min())/(phydnet_preds.max()-phydnet_preds.min())
# # phydnet_gt=(phydnet_gt-phydnet_gt.min())/(phydnet_gt.max()-phydnet_gt.min())
# # phydnet_cond=(phydnet_cond-phydnet_cond.min())/(phydnet_cond.max()-phydnet_cond.min())

# earthformer_preds=(earthformer_preds-earthformer_preds.min())/(earthformer_preds.max()-earthformer_preds.min())
# # earthformer_gt=(earthformer_gt-earthformer_gt.min())/(earthformer_gt.max()-earthformer_gt.min())
# # earthformer_cond=(earthformer_cond-earthformer_cond.min())/(earthformer_cond.max()-earthformer_cond.min())
# TCGdiff_preds_PSNR = 10 * torch.log10((1 ** 2) / torch.mean((TCGdiff_preds - gt) ** 2))

# prediff_preds_PSNR = 10 * torch.log10((1 ** 2) / torch.mean((prediff_preds - gt) ** 2))

# predrnn_preds_PSNR = 10 * torch.log10((1 ** 2) / torch.mean((predrnn_preds - gt) ** 2))

# phydnet_preds_PSNR = 10 * torch.log10((1 ** 2) / torch.mean((phydnet_preds - gt) ** 2))

# earthformer_preds_PSNR = 10 * torch.log10((1 ** 2) / torch.mean((earthformer_preds - gt) ** 2))


# print('TCGdiff_preds_PSNR',TCGdiff_preds_PSNR)
# print('prediff_preds_PSNR',prediff_preds_PSNR)
# print('predrnn_preds_PSNR',predrnn_preds_PSNR)
# print('phydnet_preds_PSNR',phydnet_preds_PSNR)
# print('earthformer_preds_PSNR',earthformer_preds_PSNR)


# def load_checkpoint(model, state_dict, mode=None):

#     # reuse Checkpointer in fvcore to support flexible loading
#     ckpt = Checkpointer(model, save_to_disk=False)
#     logging.basicConfig()
#     ckpt.logger.setLevel(logging.INFO)
#     # since Checkpointer requires the weight to be put under `model` field, we need to save it to disk
#     tmp_path = tempfile.NamedTemporaryFile('w+b')
#     torch.save({'model': state_dict}, tmp_path.name)
#     ckpt.load(tmp_path.name)

# classifier=create_model(
#     'crossvit_small_224',
#     pretrained=False,
#     num_classes=2,
#     drop_rate=0.0,
#     drop_path_rate=0.1,
#     drop_block_rate=None,)
# classifier.to(torch.device("cuda"))
# print("Loading pretrained classifier")
# checkpoint = torch.load('/opt/data/private/CrossViT-main/output_TC/model_best.pth', map_location='cpu')
# # load_checkpoint(classifier, checkpoint['model'])
# TCGdiff_preds=TCGdiff_preds.expand(-1, -1,3, -1, -1).to(torch.device("cuda")).float()
# prediff_preds=prediff_preds.expand(-1, -1,3, -1, -1).to(torch.device("cuda")).float()
# predrnn_preds=predrnn_preds.expand(-1, -1,3, -1, -1).to(torch.device("cuda")).float()
# phydnet_preds=phydnet_preds.expand(-1, -1,3, -1, -1).to(torch.device("cuda")).float()
# earthformer_preds=earthformer_preds.expand(-1, -1,3, -1, -1).to(torch.device("cuda")).float()
# TCGdiff_seqs=torch.cat([TCGdiff_cond,TCGdiff_preds],dim=1).expand(-1, -1,3, -1, -1).to(torch.device("cuda")).float()
# predrnn_seqs=torch.cat([TCGdiff_cond,predrnn_preds],dim=1).expand(-1,-1, 3, -1, -1).to(torch.device("cuda")).float()
# phydnet_seqs=torch.cat([TCGdiff_cond,phydnet_preds],dim=1).expand(-1,-1, 3, -1, -1).to(torch.device("cuda")).float()
# earthformer_seqs=torch.cat([TCGdiff_cond,earthformer_preds],dim=1).expand(-1,-1, 3, -1, -1).to(torch.device("cuda")).float()

# i=0

# for TCGdiff,prediff,predrnn,phydnet,earthformer in zip(TCGdiff_preds,prediff_preds,predrnn_preds,phydnet_preds,earthformer_preds):
#     output_TCGdiff = classifier(TCGdiff).argmax(dim=1)
#     output_prediff = classifier(prediff).argmax(dim=1)
#     output_predrnn = classifier(predrnn).argmax(dim=1)
#     output_phydnet = classifier(phydnet).argmax(dim=1)
#     output_earthformer = classifier(earthformer).argmax(dim=1)
#     print("第{}个台风的生成检测结果========>>>>>>>".format(i))
#     print("TCGdiff",output_TCGdiff)
#     print("prediff",output_prediff)
#     print("predrnn",output_predrnn)
#     print("phydnet",output_phydnet)
#     print("earthformer",output_earthformer)
#     print("===============================>>>>>>>")
#     i+=1

# for i in range(TCGdiff_preds.size(0)):
#     output_TCGdiff = classifier(TCGdiff_preds[i]).argmax(dim=1)
#     output_TCGdiff_nonauto = classifier(TCGdiff_preds_nonauto[i]).argmax(dim=1)
#     # output_prediff = classifier(prediff_preds[i]).argmax(dim=1)
#     # output_predrnn = classifier(predrnn_preds[i]).argmax(dim=1)
#     # output_phydnet = classifier(phydnet_preds[i]).argmax(dim=1)
#     # output_earthformer = classifier(earthformer_preds[i]).argmax(dim=1)
#     print("第{}个台风的生成检测结果========>>>>>>>".format(i))
#     print("TCGdiff",output_TCGdiff)
#     print("TCGdiff_nonauto",output_TCGdiff_nonauto)
#     # print("prediff",output_prediff)
#     # print("predrnn",output_predrnn)
#     # print("phydnet",output_phydnet)
#     # print("earthformer",output_earthformer)
#     print("===============================>>>>>>>")
#     i+=1
print("=============FVD====================")
TCGdiff_test_fvd_traj = FrechetVideoDistance(
                feature=400,
                layout="NTCHW",
                reset_real_features=False,
                normalize=False,
                auto_t=True, )

prediff_test_fvd = FrechetVideoDistance(
                feature=400,
                layout="NTCHW",
                reset_real_features=False,
                normalize=False,
                auto_t=True, ).to(torch.device("cuda"))

TCGdiff_test_fvd = FrechetVideoDistance(
                feature=400,
                layout="NTCHW",
                reset_real_features=False,
                normalize=False,
                auto_t=True, ).to(torch.device("cuda"))

predrnn_test_fvd = FrechetVideoDistance(
                feature=400,
                layout="NTCHW",
                reset_real_features=False,
                normalize=False,
                auto_t=True, ).to(torch.device("cuda"))

phydnet_test_fvd = FrechetVideoDistance(
                feature=400,
                layout="NTCHW",
                reset_real_features=False,
                normalize=False,
                auto_t=True, ).to(torch.device("cuda"))

earthformer_test_fvd = FrechetVideoDistance(
                feature=400,
                layout="NTCHW",
                reset_real_features=False,
                normalize=False,
                auto_t=True, ).to(torch.device("cuda"))

# prediff_test_fvd.update(prediff_preds.float(), real=False)
# prediff_test_fvd.update(gt.float(), real=True)
# prediff_test_fvd = prediff_test_fvd.compute()

TCGdiff_test_fvd.update(TCGdiff_preds.float(), real=False)
TCGdiff_test_fvd.update(gt.float(), real=True)
TCGdiff_test_fvd = TCGdiff_test_fvd.compute()

# predrnn_test_fvd.update(predrnn_preds.float(), real=False)
# predrnn_test_fvd.update(gt.float(), real=True)
# predrnn_test_fvd = predrnn_test_fvd.compute()

# phydnet_test_fvd.update(phydnet_preds.float(), real=False)
# phydnet_test_fvd.update(gt.float(), real=True)
# phydnet_test_fvd = phydnet_test_fvd.compute()

# earthformer_test_fvd.update(earthformer_preds.float(), real=False)
# earthformer_test_fvd.update(gt.float(), real=True)
# earthformer_test_fvd = earthformer_test_fvd.compute()
# print("prediff fvd",prediff_test_fvd)
print("TCGdiff fvd",TCGdiff_test_fvd)
# print("predrnn fvd",predrnn_test_fvd)
# print("phydnet fvd",phydnet_test_fvd)
# print("earthformer fvd",earthformer_test_fvd)

# TCGdiff_mse=((TCGdiff_preds-TCGdiff_gt)**2).mean()
# TCGdiff_mae=((TCGdiff_preds-TCGdiff_gt).abs()).mean()
# # # mcvd_mse=((mcvd_preds-mcvd_gt)**2).mean()
# # # mcvd_mae=(mcvd_preds-mcvd_gt).abs().mean()

# predrnn_mse=((predrnn_preds-predrnn_gt)**2).mean()
# predrnn_mae=(predrnn_preds-predrnn_gt).abs().mean()

# phydnet_mse=((phydnet_preds-phydnet_gt)**2).mean()
# phydnet_mae=(phydnet_preds-phydnet_gt).abs().mean()

# earthformer_mse=((earthformer_preds-earthformer_gt)**2).mean()
# earthformer_mae=(earthformer_preds-earthformer_gt).abs().mean()
# print("TCGdiff_mse",TCGdiff_mse.item())
# # # print("mcvd_mse",mcvd_mse.item())
# print("predrnn_mse",predrnn_mse.item())
# print("phydnet_mse",phydnet_mse.item())
# print("earthformer_mse",earthformer_mse.item())
# print("TCGdiff_mae",TCGdiff_mae.item())
# # # print("mcvd_mae",mcvd_mae.item())

# print("predrnn_mae",predrnn_mae.item())

# print("phydnet_mae",phydnet_mae.item())

# print("earthformer_mae",earthformer_mae.item())
print("=============CRPS====================")
# prediff_preds_crps=prediff_preds[:].reshape(-1)
gt_crps= gt[:].reshape(-1)
# prediff_crps_score = ps.crps_ensemble(gt_crps.cpu().numpy(), prediff_preds_crps.cpu().numpy())

TCGdiff_preds_crps=TCGdiff_preds[:].reshape(-1)
TCGdiff_crps_score = ps.crps_ensemble(gt_crps.cpu().numpy(), TCGdiff_preds_crps.cpu().numpy())

# predrnn_preds_crps=predrnn_preds[:].reshape(-1)
# predrnn_crps_score = ps.crps_ensemble(gt_crps.cpu().numpy(), predrnn_preds_crps.cpu().numpy())

# phydnet_preds_crps=phydnet_preds[:].reshape(-1)
# phydnet_crps_score = ps.crps_ensemble(gt_crps.cpu().numpy(), phydnet_preds_crps.cpu().numpy())

# earthformer_preds_crps=earthformer_preds[:].reshape(-1)
# earthformer_crps_score = ps.crps_ensemble(gt_crps.cpu().numpy(), earthformer_preds_crps.cpu().numpy())

# print("prediff_crps",prediff_crps_score.mean())
print("TCGdiff_crps",TCGdiff_crps_score.mean())
# print("predrnn_crps",predrnn_crps_score.mean())

# print("phydnet_crps",phydnet_crps_score.mean())

# print("earthformer_crps",earthformer_crps_score.mean())


# print(TCGdiff_preds.size())
# print(prediff_preds.size())
# print(predrnn_preds.size())
# print(phydnet_preds.size())
# print(earthformer_preds.size())



# prediff_preds=prediff_preds.reshape(len(prediff_preds), -1, 1, prediff_preds.size(3), prediff_preds.size(4)).permute(0, 2, 1, 4, 3).reshape(len(prediff_preds), 1, -1, prediff_preds.size(3)).permute(0, 1, 3, 2)
TCGdiff_preds=TCGdiff_preds.reshape(len(TCGdiff_preds), -1, 1, TCGdiff_preds.size(3), TCGdiff_preds.size(4)).permute(0, 2, 1, 4, 3).reshape(len(TCGdiff_preds), 1, -1, TCGdiff_preds.size(3)).permute(0, 1, 3, 2)
# phydnet_preds=phydnet_preds.reshape(len(phydnet_preds), -1, 1, phydnet_preds.size(3), phydnet_preds.size(4)).permute(0, 2, 1, 4, 3).reshape(len(phydnet_preds), 1, -1, phydnet_preds.size(3)).permute(0, 1, 3, 2)
# earthformer_preds=earthformer_preds.reshape(len(earthformer_preds), -1, 1, earthformer_preds.size(3), earthformer_preds.size(4)).permute(0, 2, 1, 4, 3).reshape(len(earthformer_preds), 1, -1, earthformer_preds.size(3)).permute(0, 1, 3, 2)
# predrnn_preds=predrnn_preds.reshape(len(predrnn_preds), -1, 1, predrnn_preds.size(3), predrnn_preds.size(4)).permute(0, 2, 1, 4, 3).reshape(len(predrnn_preds), 1, -1, predrnn_preds.size(3)).permute(0, 1, 3, 2)
gt=gt.reshape(len(gt), -1, 1, gt.size(3), gt.size(4)).permute(0, 2, 1, 4, 3).reshape(len(gt), 1, -1, gt.size(3)).permute(0, 1, 3, 2)

print("=============make_grid====================")

gt_grid = make_grid(gt, nrow=nrow, padding=6, pad_value=0.5)
save_image(gt_grid, os.path.join('/opt/data/private/mcvd-pytorch2/mcvd-pytorch/lx', f'dydiffusion-gt-3.png'))

TCGdiff_grid = make_grid(TCGdiff_preds, nrow=nrow, padding=6, pad_value=0.5)
save_image(TCGdiff_grid, os.path.join('/opt/data/private/mcvd-pytorch2/mcvd-pytorch/lx', f'dydiffusion-3.png'))

# prediff_grid = make_grid(prediff_preds, nrow=nrow, padding=6, pad_value=0.5)
# save_image(prediff_grid, os.path.join('/opt/data/private/mcvd-pytorch/lx', 'prediff_full_grid-24.png'))

# earthformer_grid = make_grid(earthformer_preds, nrow=nrow, padding=6, pad_value=0.5)
# save_image(earthformer_grid, os.path.join('/opt/data/private/mcvd-pytorch/lx', 'earthformer_full_grid-24.png'))

# phydnet_grid = make_grid(phydnet_preds, nrow=nrow, padding=6, pad_value=0.5)
# save_image(phydnet_grid, os.path.join('/opt/data/private/mcvd-pytorch/lx', 'phydnet_full_grid-24.png'))

# predrnn_grid = make_grid(predrnn_preds, nrow=nrow, padding=6, pad_value=0.5)
# save_image(predrnn_grid, os.path.join('/opt/data/private/mcvd-pytorch/lx', 'predrnn_full_grid-24.png'))

# TCGdiff_grid = make_grid(torch.cat([TCGdiff_cond, padding, TCGdiff_gt, padding, TCGdiff_preds],dim=-1), nrow=nrow, padding=6, pad_value=0.5)
# 将网格图像转换为 NumPy 数组
# TCGdiff_grid_np = TCGdiff_grid.permute(1, 2, 0).cpu().numpy()

# 将图像转换为灰度
# TCGdiff_grid_gray = np.mean(TCGdiff_grid_np, axis=-1)
# plt.imsave('/opt/data/private/mcvd-pytorch/TCGdiff_full_grid_jet.png',TCGdiff_grid_gray, cmap='jet',  dpi=300)
# save_image(TCGdiff_grid, os.path.join('/opt/data/private/mcvd-pytorch', 'TCGdiff_full_grid.png'))

# prediff_grid = make_grid(torch.cat([prediff_cond, padding, prediff_gt, padding, prediff_preds],dim=-1), nrow=nrow, padding=6, pad_value=0.5)
# prediff_grid_np = prediff_grid.permute(1, 2, 0).cpu().numpy()

# 将图像转换为灰度
# prediff_grid_gray = np.mean(prediff_grid_np, axis=-1)
# plt.imsave('/opt/data/private/mcvd-pytorch/prediff_full_grid_jet.png',prediff_grid_gray, cmap='jet',  dpi=300)

# save_image(prediff_grid, os.path.join('/opt/data/private/mcvd-pytorch', 'prediff_full_grid.png'))

# mcvd_grid = make_grid(torch.cat(
#                         [mcvd_cond, padding, mcvd_gt, padding, mcvd_preds] ,
#                             dim=-1), nrow=nrow, padding=6, pad_value=0.5)
# save_image(mcvd_grid, os.path.join('/opt/data/private/mcvd-pytorch', 'mcvd_full_grid.png'))

# predrnn_grid = make_grid(torch.cat(
#                         [predrnn_cond, padding, predrnn_gt, padding, predrnn_preds] ,
#                             dim=-1), nrow=nrow, padding=6, pad_value=0.5)
# predrnn_grid_np = predrnn_grid.permute(1, 2, 0).cpu().numpy()

# 将图像转换为灰度
# predrnn_grid_gray = np.mean(predrnn_grid_np, axis=-1)
# plt.imsave('/opt/data/private/mcvd-pytorch/predrnn_full_grid_jet.png',predrnn_grid_gray, cmap='jet',  dpi=300)
# save_image(predrnn_grid, os.path.join('/opt/data/private/mcvd-pytorch', 'predrnn_full_grid.png'))

# phydnet_grid = make_grid(torch.cat(
#                         [phydnet_cond, padding, phydnet_gt, padding, phydnet_preds] ,
#                             dim=-1), nrow=nrow, padding=6, pad_value=0.5)
# phydnet_grid_np = phydnet_grid.permute(1, 2, 0).cpu().numpy()

# 将图像转换为灰度
# phydnet_grid_gray = np.mean(phydnet_grid_np, axis=-1)
# plt.imsave('/opt/data/private/mcvd-pytorch/phydnet_full_grid_jet.png',phydnet_grid_gray, cmap='jet',  dpi=300)
# save_image(phydnet_grid, os.path.join('/opt/data/private/mcvd-pytorch', 'phydnet_full_grid.png'))

# earthformer_grid = make_grid(torch.cat(
#                         [earthformer_cond, padding, earthformer_gt, padding, earthformer_preds] ,
#                             dim=-1), nrow=nrow, padding=6, pad_value=0.5)
# earthformer_grid_np = earthformer_grid.permute(1, 2, 0).cpu().numpy()

# 将图像转换为灰度
# earthformer_grid_gray = np.mean(earthformer_grid_np, axis=-1)
# plt.imsave('/opt/data/private/mcvd-pytorch/earthformer_full_grid_jet.png',earthformer_grid_gray, cmap='jet',  dpi=300)
# save_image(earthformer_grid, os.path.join('/opt/data/private/mcvd-pytorch', 'earthformer_full_grid.png'))