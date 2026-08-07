**Paper:** *Attention Is All You Need* (Vaswani et al., 2017) – the seminal work that introduced the **Transformer** architecture.

---

## 1. Motivation
- **Sequence‑to‑sequence** tasks (e.g., machine translation) had relied on recurrent (RNN) or convolutional (CNN) models.
- RNNs suffer from **sequential bottlenecks** (slow training, limited parallelism) and difficulty capturing long‑range dependencies.
- The authors asked: *Can we build a model that forgoes recurrence entirely and still achieve state‑of‑the‑art performance?*

---

## 2. Core Idea
- Replace recurrence with **self‑attention** (also called “scaled dot‑product attention”) as the sole means of inter‑token communication.
- Stack multiple identical layers that consist of:
  1. **Multi‑Head Self‑Attention** – lets the model attend to information from different representation subspaces simultaneously.
  2. **Position‑wise Feed‑Forward Networks** – applied independently to each position.
- Use **residual connections** + **layer normalization** around each sub‑layer.

---

## 3. Architecture Details

| Component | Description |
|-----------|-------------|
| **Input Embedding** | Token embeddings + **positional encodings** (sinusoidal functions) to inject order information. |
| **Encoder** | N = 6 identical layers. Each layer: <br>• Multi‑head self‑attention (queries, keys, values all from the same source). <br>• Add‑&‑Norm. <br>• Position‑wise fully‑connected feed‑forward (two linear layers with ReLU). |
| **Decoder** | N = 6 identical layers. Each layer: <br>• Masked multi‑head self‑attention (prevents positions from attending to future tokens). <br>• Add‑&‑Norm. <br>• Multi‑head **encoder‑decoder attention** (queries from decoder, keys/values from encoder). <br>• Add‑&‑Norm.
