import torch
import pprint

def inspect_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")

    print("=" * 80)
    print("Checkpoint:", path)

    print("\n[1] Top-level keys")
    print(list(ckpt.keys()))

    print("\n[2] Step")
    print(ckpt.get("step"))

    print("\n[3] Config")
    pprint.pprint(ckpt.get("config", {}))

    print("\n[4] Model tensors")
    state = ckpt.get("model_state_dict", {})
    print("Number of tensors:", len(state))

    for name, tensor in state.items():
        print(f"{name:70s} {tuple(tensor.shape)}")

    print("\n[5] Expert tensors only")
    for name, tensor in state.items():
        if "expert" in name.lower() or "moe" in name.lower() or "router" in name.lower():
            print(f"{name:70s} {tuple(tensor.shape)}")

    print("\n[6] Optimizer param groups")
    opt = ckpt.get("optimizer_state_dict")
    if opt is not None:
        for group in opt.get("param_groups", []):
            small_group = {
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
                "betas": group.get("betas"),
                "eps": group.get("eps"),
            }
            pprint.pprint(small_group)

    print("\n[7] History")
    history = ckpt.get("history")
    if history is None:
        print("Không có history trong checkpoint.")
    else:
        for row in history[:20]:
            print(row)

if __name__ == "__main__":
    inspect_checkpoint("checkpoints/hmoe_top_k_step_300.pt")