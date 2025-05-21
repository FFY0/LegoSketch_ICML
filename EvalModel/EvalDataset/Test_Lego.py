import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EvalModel.EvalDataset.EvalModelUtil import eval_dataset_with_seed
from DatasetLoader import project_path

dataset_space_dic = {
    'synthetic_0_5': list(range(300, 2400, 300)),
    'synthetic_0_7': list(range(300, 2400, 300)),
    'synthetic_0_9': list(range(300, 2400, 300)),
    'synthetic_1_1': list(range(300, 2400, 300)),
    'synthetic_1_3': list(range(300, 2400, 300)),
    'synthetic_1_5': list(range(300, 2400, 300)),
}

if __name__ == '__main__':
    device = 'cuda:0'
    model_0_list = ["EvalModel/EvalDataset/ModelRepository/FinalAblation/Base_seed0_model",
                    "EvalModel/EvalDataset/ModelRepository/FinalAblation/Base_seed1_model"]
    model_path_list = [model_0_list,]
    model_name_list = ['Base',]
    for i in range(len(model_path_list)):
        for j in range(len(model_path_list[i])):
            model_path_list[i][j] = os.path.join(project_path , model_path_list[i][j])
    real_dataset_list = ['synthetic_0_5','synthetic_0_7','synthetic_0_9','synthetic_1_1','synthetic_1_3','synthetic_1_5']
    for dataset_name in real_dataset_list:
        space_budget_list = dataset_space_dic[dataset_name]
        eval_dataset_with_seed(model_path_list,model_name_list, device, dataset_name, space_budget_list)