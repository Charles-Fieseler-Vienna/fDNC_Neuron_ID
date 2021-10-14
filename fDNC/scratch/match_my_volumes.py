from DLC_for_WBFM.utils.projects.finished_project_data import finished_project_data
from DLC_for_WBFM.utils.xinwei_fdnc.formatting import zimmer2leifer
from fDNC.src.DNC_predict import predict_matches

# Load my data
project_path = '/scratch/zimmer/Charles/dlc_stacks/project-pytest/project_config.yaml'
project_dat = finished_project_data.load_final_project_data_from_config(project_path)

# Get point clouds
pts0 = zimmer2leifer(project_dat.get_centroids_as_numpy(0))
pts0_copy = zimmer2leifer(project_dat.get_centroids_as_numpy(0))
pts0_offset = zimmer2leifer(project_dat.get_centroids_as_numpy(0))[2:, :]

# Match and print
model_path = '../model/model.bin'

matches_self = predict_matches(pts0, pts0_copy, model_path=model_path)
print(f"Self matches: {matches_self[:10]}")

matches_offset = predict_matches(pts0, pts0_copy, model_path=model_path)
print(f"Offset matches: {matches_offset[:10]}")
