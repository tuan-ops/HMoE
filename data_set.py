"""
Tạo tập dữ liệu 4 domain chuyên biệt với mức độ token dễ/khó khác nhau:
  Domain 0 (EASY)  : Assembly / CPU — cú pháp ngắn, từ lặp nhiều
  Domain 1 (MEDIUM): Transformer / AI — thuật ngữ dài, câu trung bình
  Domain 2 (HARD)  : Toán học — ký hiệu đặc biệt, cấu trúc logic chặt
  Domain 3 (HARD+) : Sinh học phân tử — từ chuyên môn rất dài, hiếm gặp
"""

import random
import os

random.seed(42)
# Domain 0: Assembly / CPU (token đều, ngắn, lặp nhiều -> EASY)
ASM_INSTRUCTIONS = [
    "mov rax, rbx", "push rbp", "pop rbx", "add rax, 1",
    "sub rsp, 8", "call printf", "ret", "xor rax, rax",
    "cmp rax, 0", "jne loop", "jmp exit", "lea rdi, [rsp]",
    "mov [rbp-8], rax", "nop", "int 0x80", "syscall",
    "imul rcx, rdx", "div rbx", "and rax, 0xff", "or rbx, rcx",
    "shl rax, 2", "shr rbx, 1", "test rax, rax", "not rbx",
]
ASM_COMMENTS = [
    "; save return address", "; zero out register", "; loop counter",
    "; align stack", "; syscall number", "; argument 1",
    "; check condition", "; function prologue", "; restore caller",
]
ASM_CONTEXT = [
    "section .text\nglobal _start\n_start:\n",
    "section .data\nmsg db 'hello', 0\n",
    "; x86-64 linux calling convention\n",
    "bits 64\ndefault rel\n",
]

def asm_block():
    lines = []
    lines.append(random.choice(ASM_CONTEXT))
    n = random.randint(4, 12)
    for _ in range(n):
        instr = random.choice(ASM_INSTRUCTIONS)
        if random.random() < 0.35:
            instr += "  " + random.choice(ASM_COMMENTS)
        lines.append(instr)
    return "\n".join(lines)

# Domain 1: Transformer / AI  (token medium — thuật ngữ quen, câu trung bình)

AI_TEMPLATES = [
    "The {comp} in a Transformer consists of {part1} and {part2}.",
    "Self-attention computes the dot product of {q} and {k} scaled by sqrt({d}).",
    "The {comp} applies layer normalization before the {op} operation.",
    "Feed-forward networks expand the {dim}-dimensional embedding to {ff_dim} dimensions.",
    "Positional encoding adds {enc_type} to each token embedding.",
    "The {comp} uses {heads} attention heads with dimension {head_dim} each.",
    "Mixture-of-Experts replaces the {ffn} with a set of {n} specialized experts.",
    "The router assigns each token a probability distribution over {n} experts.",
    "Top-k routing selects the {k} highest-probability experts for each token.",
    "Load balancing loss penalizes unequal utilization of the {n} expert modules.",
    "Residual connections add the input {x} to the output of each sub-layer.",
    "The {comp} stacks {layers} identical transformer blocks for depth.",
    "Cross-entropy loss measures divergence between logits and target distribution.",
    "Gradient clipping prevents exploding gradients by capping the norm at {clip}.",
    "The learning rate warms up over {steps} steps then decays with cosine schedule.",
]
AI_VARS = {
    "comp": ["encoder", "decoder", "transformer block", "attention layer", "MoE layer"],
    "part1": ["multi-head attention", "self-attention", "cross-attention", "linear projection"],
    "part2": ["feed-forward network", "layer norm", "residual connection", "dropout"],
    "q": ["query Q", "the query matrix", "query vectors"],
    "k": ["key K", "the key matrix", "key vectors"],
    "d": ["d_k", "the head dimension", "64"],
    "op": ["attention", "feed-forward", "projection", "normalization"],
    "dim": ["512", "768", "1024", "256"],
    "ff_dim": ["2048", "3072", "4096"],
    "enc_type": ["sinusoidal encodings", "learned embeddings", "rotary position encodings"],
    "heads": ["8", "12", "16", "4"],
    "head_dim": ["64", "128", "96"],
    "ffn": ["FFN layer", "dense feed-forward", "MLP block"],
    "n": ["4", "8", "16", "32"],
    "k": ["2", "3", "4"],
    "x": ["the input tensor", "the hidden state h", "the token embedding"],
    "layers": ["6", "12", "24", "4"],
    "clip": ["1.0", "0.5", "5.0"],
    "steps": ["1000", "4000", "10000"],
}

def ai_block():
    lines = []
    n = random.randint(3, 8)
    for _ in range(n):
        tmpl = random.choice(AI_TEMPLATES)
        for k, v in AI_VARS.items():
            if "{" + k + "}" in tmpl:
                tmpl = tmpl.replace("{" + k + "}", random.choice(v))
        lines.append(tmpl)
    return " ".join(lines)

# Domain 2: Toán học  (token KHÓ — ký hiệu, công thức, logic chặt)

MATH_TEMPLATES = [
    "Let f: R^{n} -> R^{m} be a {prop} function. Then {stmt}.",
    "The {op} of matrices A and B satisfies {prop_mat}.",
    "For all epsilon > 0, there exists delta > 0 such that |x - {a}| < delta implies |f(x) - {lim}| < epsilon.",
    "The gradient descent update rule is theta_{t+1} = theta_t - alpha * grad L(theta_t).",
    "By the chain rule, d/dx [f(g(x))] = f'(g(x)) * g'(x).",
    "The softmax function sigma(z)_i = exp(z_i) / sum_j exp(z_j) is {prop}.",
    "Eigenvalue decomposition: A = Q Lambda Q^T where {prop_eig}.",
    "The cross-entropy loss H(p, q) = - sum_x p(x) log q(x) is minimized when {cond}.",
    "KL divergence D_KL(P || Q) = sum_x P(x) log(P(x) / Q(x)) >= 0 by {reason}.",
    "For a convex function f, the {ineq} inequality holds: f(lambda*x + (1-lambda)*y) <= lambda*f(x) + (1-lambda)*f(y).",
    "The {norm} of a vector v in R^n is ||v||_{p} = (sum_i |v_i|^p)^(1/p).",
    "Matrix multiplication AB where A in R^{m x k} and B in R^{k x n} yields C in R^{m x n}.",
    "The sigmoid activation sigma(x) = 1/(1 + exp(-x)) has derivative sigma(x)(1 - sigma(x)).",
    "The Hessian H_ij = d^2 L / (d theta_i d theta_j) determines the {curv} of the loss surface.",
]
MATH_VARS = {
    "n": ["n", "d", "p"], "m": ["m", "k", "q"],
    "prop": ["continuous", "differentiable", "Lipschitz-continuous", "convex", "smooth"],
    "stmt": ["it is uniformly continuous", "the Jacobian exists", "local minima are global"],
    "op": ["product", "sum", "Hadamard product", "Kronecker product"],
    "prop_mat": ["(AB)^T = B^T A^T", "trace(AB) = trace(BA)", "rank(AB) <= min(rank A, rank B)"],
    "a": ["0", "c", "x_0"], "lim": ["L", "f(c)", "0"],
    "prop_eig": ["Q is orthogonal", "Lambda contains eigenvalues", "columns of Q are eigenvectors"],
    "cond": ["p = q", "the distributions match", "q approximates p perfectly"],
    "reason": ["Jensen's inequality", "log-sum inequality", "convexity of -log"],
    "ineq": ["Jensen", "Cauchy-Schwarz", "AM-GM"],
    "norm": ["L1 norm", "L2 norm", "Lp norm", "Frobenius norm"],
    "curv": ["curvature", "local geometry", "second-order structure"],
}

def math_block():
    lines = []
    n = random.randint(2, 6)
    for _ in range(n):
        tmpl = random.choice(MATH_TEMPLATES)
        for k, v in MATH_VARS.items():
            if "{" + k + "}" in tmpl:
                tmpl = tmpl.replace("{" + k + "}", random.choice(v))
        lines.append(tmpl)
    return " ".join(lines)

# Domain 3: Sinh học phân tử  (token rất khó — từ dài hiếm gặp)
BIO_TEMPLATES = [
    "The {proc} process involves {enzyme} catalyzing the {rxn} reaction at the {site} site.",
    "{prot} undergoes {mod} modification at {res} residues, altering its {func}.",
    "The {path} pathway activates {tf} transcription factor, upregulating {gene} expression.",
    "CRISPR-Cas9 introduces double-strand breaks at {locus}, enabling {edit} editing.",
    "The {struct} structure of {prot} is stabilized by {bond} bonds between {aa1} and {aa2}.",
    "Ribosomal translation of mRNA proceeds from 5' to 3', with {aa} amino acids incorporated per codon.",
    "Apoptosis is triggered by {signal} signaling through {complex} caspase activation.",
    "The {org} organelle performs {func} by maintaining {grad} gradient across its membrane.",
    "Epigenetic regulation via {mark} methylation silences {gene} gene loci in {tissue} tissue.",
    "Protein folding chaperones such as {chap} prevent {agg} aggregation under {stress} stress.",
    "The {mut} mutation in {gene} gene causes {disease} via {mech} mechanism.",
    "RNA splicing removes {n} introns and joins {m} exons to produce mature mRNA.",
    "Phosphorylation of {prot} at Ser-{pos} by {kin} kinase activates the {path} cascade.",
]
BIO_VARS = {
    "proc": ["transcription", "translation", "replication", "splicing", "glycolysis"],
    "enzyme": ["RNA polymerase II", "DNA helicase", "topoisomerase I", "ATP synthase", "ribonuclease"],
    "rxn": ["phosphorylation", "deamination", "methylation", "ubiquitination", "acetylation"],
    "site": ["active", "allosteric", "catalytic", "regulatory", "binding"],
    "prot": ["p53", "BRCA1", "mTOR", "EGFR", "ubiquitin", "calmodulin", "tubulin"],
    "mod": ["post-translational", "phosphorylation", "ubiquitination", "SUMOylation", "acetylation"],
    "res": ["serine", "threonine", "tyrosine", "lysine", "cysteine"],
    "func": ["transcriptional activity", "protein stability", "subcellular localization", "enzymatic activity"],
    "path": ["PI3K-AKT", "MAPK-ERK", "NF-kB", "Wnt-beta-catenin", "JAK-STAT", "mTORC1"],
    "tf": ["NF-kB", "STAT3", "HIF-1alpha", "AP-1", "Sp1"],
    "gene": ["BCL2", "MYC", "TP53", "VEGFA", "CDKN1A", "TERT"],
    "locus": ["intron 11", "exon 3", "the promoter region", "CpG island"],
    "edit": ["knock-in", "knockout", "base", "prime"],
    "struct": ["tertiary", "quaternary", "secondary", "alpha-helical"],
    "bond": ["disulfide", "hydrogen", "hydrophobic", "van der Waals", "ionic"],
    "aa1": ["Cys-34", "His-64", "Asp-102", "Arg-145"],
    "aa2": ["Cys-80", "Ser-195", "Glu-35", "Lys-41"],
    "aa": ["20", "standard 20", "3", "4"],
    "signal": ["TNF-alpha", "Fas ligand", "cytochrome c", "TRAIL"],
    "complex": ["apoptosome", "DISC", "inflammasome"],
    "org": ["mitochondrial", "lysosomal", "endoplasmic reticulum", "nuclear envelope"],
    "grad": ["proton", "electrochemical", "ion", "pH"],
    "mark": ["histone H3K27", "CpG", "H3K4me3", "H3K9me3"],
    "tissue": ["hepatic", "neural", "epithelial", "hematopoietic"],
    "chap": ["Hsp70", "GroEL/GroES", "Hsp90", "TRiC"],
    "agg": ["amyloid", "protein", "fibrillar"],
    "stress": ["heat shock", "oxidative", "ER", "hypoxic"],
    "mut": ["missense", "nonsense", "frameshift", "splice-site"],
    "disease": ["carcinogenesis", "neurodegeneration", "immune dysregulation"],
    "mech": ["loss-of-function", "dominant-negative", "gain-of-function"],
    "n": ["3", "5", "12", "8"], "m": ["4", "6", "9"],
    "pos": ["473", "308", "127", "9"],
    "kin": ["PKA", "CDK2", "AMPK", "mTORC2"],
}

def bio_block():
    lines = []
    n = random.randint(2, 5)
    for _ in range(n):
        tmpl = random.choice(BIO_TEMPLATES)
        for k, v in BIO_VARS.items():
            if "{" + k + "}" in tmpl:
                tmpl = tmpl.replace("{" + k + "}", random.choice(v))
        lines.append(tmpl)
    return " ".join(lines)

GENERATORS = [asm_block, ai_block, math_block, bio_block]
DOMAIN_NAMES = ["[ASM]", "[AI]", "[MATH]", "[BIO]"]

def build_dataset(n_blocks_per_domain: int = 600) -> str:
    """
    Tạo dataset với:
    - n_blocks_per_domain block mỗi domain
    - Mỗi block được đánh nhãn domain ở đầu
    - Interleave ngẫu nhiên để model phải switch context
    """
# Duyệt qua các hàm tạo và nhãn tương ứng
    blocks = []
    for domain_id, (gen_fn, marker) in enumerate(zip(GENERATORS, DOMAIN_NAMES)):
        for _ in range(n_blocks_per_domain):
            text = gen_fn()
            blocks.append((domain_id, f"{marker} {text}"))

    random.shuffle(blocks)
    return "\n\n".join(text for _, text in blocks), [d for d, _ in blocks]

def build_dataset_by_domain(n_blocks_per_domain: int = 600):
    """
    Tạo file riêng cho từng domain để đánh giá per-domain perplexity
    """
# Duyệt qua từng miền nhưng không trộn lại với nhau
    domain_texts = {}
    for domain_id, (gen_fn, marker) in enumerate(zip(GENERATORS, DOMAIN_NAMES)):
        blocks = []
        for _ in range(n_blocks_per_domain):
            blocks.append(f"{marker} {gen_fn()}")
        domain_texts[domain_id] = "\n\n".join(blocks)
    return domain_texts

if __name__ == "__main__":
    # Tạo thư mục data/ 
    project_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(project_dir, "data")
    os.makedirs(out_dir, exist_ok=True)

    # Main mixed dataset cho train.py
    print("Generating mixed train dataset...")
    text, domain_labels = build_dataset(n_blocks_per_domain=800)

    main_path = os.path.join(out_dir, "tiny_text.txt")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Per-domain test sets để sau này evaluate riêng từng domain
    print("Generating per-domain test sets...")
    domain_texts = build_dataset_by_domain(n_blocks_per_domain=200)

    for d_id, d_text in domain_texts.items():
        domain_name = DOMAIN_NAMES[d_id][1:-1].lower()
        path = os.path.join(out_dir, f"domain_{d_id}_{domain_name}.txt")

        with open(path, "w", encoding="utf-8") as f:
            f.write(d_text)

    # Stats
    chars = len(text)
    vocab = len(set(text))

    print(f"\nDataset stats:")
    print(f"  Train file    : {main_path}")
    print(f"  Total chars   : {chars:,}")
    print(f"  Vocab size    : {vocab}")
    print(f"  Domains       : {DOMAIN_NAMES}")
    print(f"  Output dir    : {out_dir}")

    print(f"\nDomain test files:")
    for d_id, d_text in domain_texts.items():
        domain_name = DOMAIN_NAMES[d_id][1:-1].lower()
        path = os.path.join(out_dir, f"domain_{d_id}_{domain_name}.txt")
        print(
            f"  {DOMAIN_NAMES[d_id]}: {path} | "
            f"{len(d_text):,} chars | vocab={len(set(d_text))}"
        )

    print("\nDone.")