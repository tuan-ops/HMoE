import json
import argparse


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_comparison_table(file1, file2):
    m1 = load_json(file1)
    m2 = load_json(file2)

    name1 = m1.get("model_type", "model_1")
    name2 = m2.get("model_type", "model_2")

    keys = [
        "final_lm_loss",
        "final_total_loss",
        "tokens_per_sec",
        "peak_vram_mb",
        "total_params",
        "train_time_sec",
    ]

    rows = []

    for key in keys:
        v1 = m1.get(key, "N/A")
        v2 = m2.get(key, "N/A")

        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            diff = v1 - v2

            # Với loss/time/vram/params: thấp hơn thường tốt hơn
            # Với tokens_per_sec: cao hơn tốt hơn
            if key == "tokens_per_sec":
                better = name1 if v1 > v2 else name2 if v2 > v1 else "equal"
            else:
                better = name1 if v1 < v2 else name2 if v2 < v1 else "equal"
        else:
            diff = "N/A"
            better = "N/A"

        rows.append([key, fmt(v1), fmt(v2), fmt(diff), better])

    headers = ["metric", name1, name2, "diff", "better"]

    col_widths = [
        max(len(str(row[i])) for row in [headers] + rows)
        for i in range(len(headers))
    ]

    def print_row(row):
        print(
            " | ".join(
                str(row[i]).ljust(col_widths[i])
                for i in range(len(row))
            )
        )

    print("\nMETRICS COMPARISON")
    print_row(headers)
    print("-+-".join("-" * w for w in col_widths))

    for row in rows:
        print_row(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", type=str)
    parser.add_argument("file2", type=str)

    args = parser.parse_args()

    print_comparison_table(args.file1, args.file2)