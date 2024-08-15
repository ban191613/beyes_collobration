# %%
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
current_working_dir = os.getcwd()
sys.path.append(current_working_dir + "\src")  # 把src目录加入


# %%
import mechanistic_model as m_model
import json

with open("parameters.json", "r") as file:  # 导入参数
    parameters = json.load(file)
# print(parameters["Initial value of cloud parameters"])

# %%


filename = parameters["filename"]
# 机理模型参数

model_yf = m_model.yf
model_ym = m_model.ym
parameter_d = parameters["parameter dimension"]
variable_d = parameters["independent variable dimension"]
mechanistic_model_samples_num = parameters["number of mechanistic model samples"]

# 主动学习
device_num = parameters["Number of device-side data"]
active_learning = parameters["open Active Learning"]
active_learning_num = parameters["number of active learning samples"]

# 云端
cloud_calibration_num = parameters["number of parameter calibration data in the cloud"]
cloud_iter = parameters["number of cloud iterations"]
cloud_initial = parameters["Initial value of cloud parameters"]

# 边端
edge_iter = parameters["number of edge iterations"]
edge_initial = parameters["Initial value of edge parameters"]


# MetropolisHastings

Monte_Carlo_samples = parameters["Number of Monte Carlo samples"]
random_walks = parameters["random_walk"]
burn_ratio = parameters["Monte Carlo burn ratio"]

gp_iter = parameters["gaussian process iterations"]
gp_lr = parameters["gaussian process learning rate"]

prior_mean = parameters["prior mean"]
prior_cov = parameters["prior cov"]
prior_lower = parameters["prior lower range"]
prior_upper = parameters["prior upper range"]


# %%
from data_set import data_set

data_set = data_set(
    model_yf=model_yf,
    value_dim=variable_d,
    cloud_num=cloud_calibration_num,
    device_num=device_num,
    edge_iter=edge_iter,
)
data_set.generate_data()
# dataset.save_generate_data('output.csv')

# %%
from cloud_edge import cloud_edge

cloudEdge = cloud_edge(
    data_set=data_set,
    dim_x=variable_d,
    cloud_ym=model_ym,
    dim_u=parameter_d,
    cloud_m_num=mechanistic_model_samples_num,
    cloud_u=cloud_initial,
    cloud_iter=cloud_iter,
    edge_u=edge_initial,
    edge_iter=edge_iter,
    prior_mean=prior_mean,
    prior_cov=prior_cov,
    prior_lower=prior_lower,
    prior_upper=prior_upper,
    mcmc_sample_num=Monte_Carlo_samples,
    random_walk=random_walks,
    mcmc_br=burn_ratio,
    gp_iter=gp_iter,
    gp_lr=gp_lr,
    active_learning=active_learning,
    active_learning_num=active_learning_num,
    filename=filename,
)
cloudEdge.forward(cloud_iter)
# runs8
