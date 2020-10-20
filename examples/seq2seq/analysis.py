import json
import argparse
import collections
from utils import calculate_rouge


def split_turns(source_line):
    turns = []
    current_turn_tokens = []
    for token in source_line.strip().split():
        current_turn_tokens.append(token)
        if token == "<eou>":
            turns.append(current_turn_tokens)
            current_turn_tokens = []
    return turns
            
def get_num_dialogue_turns(source_line):
    turns = split_turns(source_line)
    return len(turns)

def get_num_speakers(source_line):
    turns = split_turns(source_line)
    speakers = set([turn[0] for turn in turns])
    return len(speakers)

def filter_speaker(source_lines, filter_arg):
    indexes = []
    for idx, source_line in enumerate(source_lines):
        if "-" in filter_arg:
            min_value, max_value = filter_arg.split("-")
            if int(min_value) <= get_num_speakers(source_line) <= int(max_value):
                indexes.append(idx)
        else:
            if get_num_speakers(source_line) == int(filter_arg):
                indexes.append(idx)
    return indexes

def filter_turn(source_lines, filter_arg):
    indexes = []
    for idx, source_line in enumerate(source_lines):
        if "-" in filter_arg:
            min_value, max_value = filter_arg.split("-")
            if int(min_value) <= get_num_dialogue_turns(source_line) <= int(max_value):
                indexes.append(idx)
        else:
            if get_num_dialogue_turns(source_line) == int(filter_arg):
                indexes.append(idx)
    return indexes

def filter_ids(source_lines, filter_str):
    if filter_str == "all":
        return list(range(len(source_lines)))
    if filter_str.startswith("speaker"):
        filter_arg = filter_str.split(":")[1]
        return filter_speaker(source_lines, filter_arg)
    if filter_str.startswith("turn"):
        filter_arg = filter_str.split(":")[1]
        return filter_turn(source_lines, filter_arg)
    raise ValueError("Invalid filter string.")

def select_lines_by_ids(lines, ids):
    return [lines[idx] for idx in ids]

def main(args):
    with open(args.source_file, "r") as f:
        source_lines = [line.strip() for line in f if line.strip()]
    with open(args.target_file, "r") as f:
        target_lines = [line.strip() for line in f if line.strip()]
    assert len(source_lines) == len(target_lines)

    model_pred_lines = {}
    for model, pred_file in args.model_preds.items():
        with open(pred_file, "r") as f:
            model_pred_lines[model] = [line.strip() for line in f if line.strip()]
        assert len(model_pred_lines[model]) == len(source_lines), (len(model_pred_lines), len(source_lines), pred_file)

    line_objs = []
    for filter_str in args.filters:
        obj = collections.OrderedDict()
        obj["filter_str"] = filter_str

        ids = filter_ids(source_lines, filter_str)
        obj["count"] = len(ids)
        
        for model, pred_lines in model_pred_lines.items():
            filtered_tgt_lines = select_lines_by_ids(target_lines, ids)
            filtered_pred_lines = select_lines_by_ids(pred_lines, ids)
            rouge_scores = calculate_rouge(filtered_pred_lines, filtered_tgt_lines)
            for rouge_key, score in rouge_scores.items():
                obj[f"{model}_{rouge_key}"] = score
        line_objs.append(obj)

    json.dump(line_objs, open("./analysis.json", "w"))

    ordered_keys = line_objs[0].keys()
    with open(args.output_file, "w") as f:
        f.write("|".join(ordered_keys) + "\n")
        f.write("|---" * len(ordered_keys) + "|\n")
        for obj in line_objs:
            for key in ordered_keys:
                f.write("|")
                if key not in obj:
                    f.write("-")
                elif isinstance(obj[key], str):
                    f.write(obj[key])
                elif isinstance(obj[key], float):
                    f.write(f"{obj[key]:.4f}")
                elif isinstance(obj[key], int):
                    f.write(f"{obj[key]}")
                else:
                    print(type(obj[key]), obj[key])
                    raise ValueError("Unknown field")
            f.write("|\n")

def make_figure_turn():
    line_objs = json.load(open("./analysis.json", "r"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np


    labels = [str(item) for item in range(3, 31)]
    pgn_scores = [[obj["PGN_rougeL"] for obj in line_objs if obj["filter_str"] == "turn:" + label][0] for label in labels]
    bart_scores = [[obj["BART_rougeL"] for obj in line_objs if obj["filter_str"] == "turn:" + label][0] for label in labels]
    hsa_scores = [[obj["HSA_rougeL"] for obj in line_objs if obj["filter_str"] == "turn:" + label][0] for label in labels]

    x = list(range(len(labels)))  # the label locations

    fig = plt.figure(figsize=(6.4, 3.2))
    ax = plt.axes((.085, .15, .905, .84))

    line1 = ax.plot(np.array(x), np.array(pgn_scores), linewidth=2, marker='x', markersize=6, label='PGN')
    line2 = ax.plot(np.array(x), np.array(bart_scores), linewidth=2, marker='^', markersize=6, label='BART')
    line3 = ax.plot(np.array(x), np.array(hsa_scores), linewidth=2, marker='o', markersize=6, label='BART+HSA')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('RougeL', fontsize=12)
    plt.xlabel('Number of dialogue turns', fontsize=12)
    plt.xticks(x, fontsize=10)
    ax.set_xticklabels(labels)
    # plt.yticks([50, 60, 70, 80, 90], fontsize=10)
    # plt.ylim(50, 95)
    ax.legend(fontsize=9, markerscale=1.0, loc=0)


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)


    # autolabel(rects1)
    # autolabel(rects2)
    # autolabel(rects3)

    fig.tight_layout()
    plt.grid(axis="y")


    # save to files in both png and pdf format
    from matplotlib.backends.backend_pdf import PdfPages
    plt.savefig("./figures/turns.png", format="png")
    with PdfPages("./figures/turns.pdf") as pdf:
        plt.savefig(pdf, format="pdf")
    plt.show()

def make_figure_speaker():
    line_objs = json.load(open("./analysis.json", "r"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np


    labels = [str(item) for item in range(2, 6)]
    pgn_scores = [[obj["PGN_rougeL"] for obj in line_objs if obj["filter_str"] == "speaker:" + label][0] for label in labels]
    bart_scores = [[obj["BART_rougeL"] for obj in line_objs if obj["filter_str"] == "speaker:" + label][0] for label in labels]
    hsa_scores = [[obj["HSA_rougeL"] for obj in line_objs if obj["filter_str"] == "speaker:" + label][0] for label in labels]

    x = list(range(len(labels)))  # the label locations

    fig = plt.figure(figsize=(6.4, 3.2))
    ax = plt.axes((.085, .15, .905, .84))

    line1 = ax.plot(np.array(x), np.array(pgn_scores), linewidth=2, marker='x', markersize=6, label='PGN')
    line2 = ax.plot(np.array(x), np.array(bart_scores), linewidth=2, marker='^', markersize=6, label='BART')
    line3 = ax.plot(np.array(x), np.array(hsa_scores), linewidth=2, marker='o', markersize=6, label='BART+HSA')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('RougeL', fontsize=12)
    plt.xlabel('Number of dialogue speakers', fontsize=12)
    plt.xticks(x, fontsize=10)
    ax.set_xticklabels(labels)
    # plt.yticks([50, 60, 70, 80, 90], fontsize=10)
    # plt.ylim(50, 95)
    ax.legend(fontsize=9, markerscale=1.0, loc=0)


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11)


    # autolabel(rects1)
    # autolabel(rects2)
    # autolabel(rects3)

    fig.tight_layout()
    plt.grid(axis="y")


    # save to files in both png and pdf format
    from matplotlib.backends.backend_pdf import PdfPages
    plt.savefig("./figures/speakers.png", format="png")
    with PdfPages("./figures/speakers.pdf") as pdf:
        plt.savefig(pdf, format="pdf")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, default="/home/disk2/lyj2019/workspace/my_paper/dataset/SAMsum/finished_src_tgt_datas/test.source")
    parser.add_argument("--target_file", type=str, default="/home/disk2/lyj2019/workspace/my_paper/dataset/SAMsum/finished_src_tgt_datas/test.target")
    parser.add_argument("--output_file", type=str, default="./analysis.md")
    args = parser.parse_args()
    args.model_preds = {
        "PGN": "/home/disk2/lyj2019/workspace/my_paper/dataset/SAMsum/finished_csv_datas/HSA_result/trpgn_sasa.txt",
        "BART": "/home/disk2/lyj2019/workspace/my_paper/dataset/SAMsum/finished_csv_datas/HSA_result/dbart_test_generations1000.txt",
        "HSA": "/home/disk2/lyj2019/workspace/my_paper/dataset/SAMsum/finished_csv_datas/HSA_result/dbart_test_generations2530.txt"
    }
    args.filters = [
        "all",
        "speaker:2", "speaker:3", "speaker:4", "speaker:5", "speaker:6", "speaker:7", "speaker:9", "speaker:11",
        "turn:3", "turn:4", "turn:5", "turn:6", "turn:7", "turn:8", "turn:9", "turn:10", "turn:11", "turn:12",
        "turn:13", "turn:14", "turn:15", "turn:16", "turn:17", "turn:18", "turn:19", "turn:20", "turn:21", 
        "turn:22", "turn:23", "turn:24", "turn:25", "turn:26", "turn:27", "turn:28", "turn:29", "turn:30",
        "speaker:2", "speaker:3-4", "speaker:5-11", "speaker:2-3", "speaker:4-5", "speaker:6-100",
        "turn:3-10", "turn:11-20", "turn:20-100"
    ]
    # main(args)
    # make_figure_turn()
    make_figure_speaker()