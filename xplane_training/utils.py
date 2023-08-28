import os


def remove_and_create_dir(path):
    """System call to rm -rf and then re-create a dir"""

    # dir = os.path.dirname(path)
    # print("attempting to delete ", dir, " path ", path)
    # if os.path.exists(path):
    #     os.system("rm -rf " + path)
    os.system("mkdir -p " + path)
    os.system("mkdir -p " + path + "output_images")

    return path


SEED = 0

OKRED = "\033[91m"
OKBLUE = "\033[94m"
ENDC = "\033[0m"
OKGREEN = "\033[92m"
OKYELLOW = "\033[93m"

pbar_cmap = [  # List of colors, same length as `dnames`
    "\x1b[38;5;231m",
    "\x1b[38;5;194m",
    "\x1b[38;5;151m",
    "\x1b[38;5;114m",
    "\x1b[38;5;71m",
    "\x1b[38;5;29m",
    "\x1b[38;5;22m",
    "\x1b[38;5;22m",
    "\x1b[38;5;22m",
    "\x1b[38;5;22m"
    # ...may include many more colors
]
