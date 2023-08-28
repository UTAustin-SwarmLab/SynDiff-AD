import os
import pandas
from tqdm import trange
from colorama import Fore
import pickle
import numpy as np
from IPython import embed

# Directory structure:
# /home/user_name/xplane/{condition}/{condition}_{train/test/val} ----original dataset
# /home/user_name/xplane_dataset/train/  ----final dataset

def orig_data_to_waypoint(frac, home_dir: str = "/home/pulkit/xplane/", main_dir: str = None, history_len: int = 4, data_dir: str = "/home/pulkit/diffusion-model-based-task-driven-training/xplane_training/"):

    main_dir = "/home/pulkit/datasets_xplane/"+frac+"/xplane_dataset/train"
    
    image_list = [x for x in os.listdir(main_dir) if x.endswith(".png")]
    label_file = {}
    for tod in ["morning","afternoon","night"]:    
        label_file[tod] = home_dir+tod+"/"+tod+"_train/" +"labels.csv"
    labels_df = {"morning": pandas.read_csv(label_file["morning"], sep=","), "afternoon": pandas.read_csv(label_file["afternoon"], sep=","), "night": pandas.read_csv(label_file["night"], sep=",")}

    waypoint_data_dict = {}

    def get_image_index_from_name(name: str):
        return int(name.split("_")[-1].split(".")[0])

    for index in trange(
        len(image_list),
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET),
    ):
        image_name = image_list[index]
        tod = ""
        if "afternoon" in image_name:
            tod = "afternoon"
        elif "morning" in image_name:
            tod = "morning"
        else:
            tod = "night"
        image_key = get_image_index_from_name(image_name)
        history_index = image_key - history_len
        # print(labels_df[tod])
        image_df_index = labels_df[tod].index[labels_df[tod]["image_filename"] == image_name].tolist()[0]

        try:
            history_image_name = labels_df[tod].iloc[[image_df_index - history_len]]["image_filename"][
                image_df_index - history_len
            ]
        except KeyError:
            continue

        history_image_idx = get_image_index_from_name(history_image_name)
        if history_image_idx == history_index:
            specific_rows = labels_df[tod].iloc[
                np.linspace(history_image_idx, image_df_index, history_len + 1).astype(int).tolist()
            ]

            # there are many states of interest, you can modify to access which ones you want
            dist_centerline = [val[1] for val in specific_rows["distance_to_centerline_meters"].items()]

            # normalized downtrack position
            # downtrack_position = [
            #     val[1] for val in specific_rows["downtrack_position_meters"].items()
            # ]

            heading_error = [val[1] for val in specific_rows["heading_error_degrees"].items()]

            target_tensor_list = [dist_centerline, heading_error]

            waypoint_data_dict[image_name] = target_tensor_list
    print("The length of the dictionary is {}".format(len(waypoint_data_dict)))
    data_dir = "/home/pulkit/diffusion-model-based-task-driven-training/xplane_training/"
    with open(os.path.join(data_dir, frac + "waypoint_data.pickle"), "wb") as handle:
        pickle.dump(waypoint_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data_dir = "/home/pulkit/diffusion-model-based-task-driven-training/xplane_training/"
    print("Generating waypoint.pickle at " + data_dir)
    for frac in ["0.1","0.2", "0.3", "0.4"]:
        orig_data_to_waypoint(frac)
