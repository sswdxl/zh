import pandas as pd

with open("csl_camera_readly.tsv", "rb") as fp:
    lines = fp.readlines()
    title_list = []
    abstract_list = []
    keywords_list = []
    class_list = []
    branch_list = []
    for line in lines:
        line_split = line.decode().strip().split("\t")
        if len(line_split) == 5:
            title_list.append(line_split[0])
            abstract_list.append(line_split[1])
            keywords_list.append(line_split[2])
            class_list.append(line_split[3])
            branch_list.append(line_split[4])
        else:
            continue
    df = pd.DataFrame({"title": title_list, "abstract": abstract_list, "keywords": keywords_list,
                       "class": class_list, "branch": branch_list})
