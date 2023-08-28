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

def orig_data_to_waypoint(home_dir: str = "/home/pulkit/xplane/", main_dir: str = "/home/pulkit/xplane_dataset/train", history_len: int = 0, data_dir: str = "/home/pulkit/diffusion-model-based-task-driven-training/xplane_training/"):

    
    

    for tod in ["morning","afternoon","night"]:
        image_list = [x for x in os.listdir(home_dir+tod+"/"+tod+"_test/") if x.endswith(".png")]
        label_file = home_dir+tod+"/"+tod+"_test/" +"labels.csv"
        labels_df = pandas.read_csv(label_file, sep=",")

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

            image_df_index = labels_df.index[labels_df["image_filename"] == image_name].tolist()[0]

            try:
                history_image_name = labels_df.iloc[[image_df_index - history_len]]["image_filename"][
                    image_df_index - history_len
                ]
            except KeyError:
                continue

            history_image_idx = get_image_index_from_name(history_image_name)
            if history_image_idx == history_index:
                specific_rows = labels_df.iloc[
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

        with open(os.path.join(data_dir, tod+"_waypoint_testdata.pickle"), "wb") as handle:
            pickle.dump(waypoint_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data_dir = "/home/pulkit/diffusion-model-based-task-driven-training/xplane_training/"
    print("Generating waypoint.pickle at " + data_dir)
    orig_data_to_waypoint()
