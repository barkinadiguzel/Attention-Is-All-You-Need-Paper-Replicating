# 🧠 Transformer From Scratch — *Replicating “Attention Is All You Need”*

Reimplementation of the **Transformer architecture** proposed in  
📄 [Vaswani et al., 2017 — *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)

This project faithfully reproduces the model described in the paper, built entirely **from scratch using PyTorch**.  
Every component — from **positional encoding** to **multi-head attention**, **feed-forward layers**, and the **Noam learning rate scheduler** — follows the original architecture and equations.

---

## 📸 Model Overview

### Figure 1: Transformer Architecture  
![Figure 1: The Transformer - model architecture](images/Figure%201:%20The%20Transformer%20-%20model%20architecture.png)  
*Overall encoder–decoder architecture consisting of stacked attention and feed-forward layers.*

---

### Figure 2: Scaled Dot-Product & Multi-Head Attention  
![Figure 2: Scaled Dot-Product & Multi-Head Attention](images/Figure%202:%20(left)%20Scaled%20Dot-Product%20Attention.%20(right)%20Multi-Head%20Attention%20consists%20of%20several%20attention%20layers%20running%20in%20parallel.png)  
*The left side shows how attention weights are computed using scaled dot-products of queries and keys.  
The right side demonstrates how multiple attention heads work in parallel to capture different dependencies.*

---

### Figure 3: Example of Attention Visualization  
![Figure 3: Example of the attention mechanism](images/Figure%203:%20An%20example%20of%20the%20attention%20mechanism%20following%20long-distance%20dependencies%20in%20the%20encoder%20self-attention%20in%20layer%205%20of%206.%20Many%20of%20the%20attention%20heads%20attend%20to%20a%20distant%20dependency%20of%20the%20verb%20‘making’,%20completing%20the%20phrase%20‘making...more%20difficult’.%20Attentions%20here%20shown%20only%20for%20the%20word%20‘making’.%20Different%20colors%20represent%20different%20heads.%20Best%20viewed%20in%20color.png)  
*Visualization from the paper: attention heads in layer 5 focusing on distant relationships like “making” → “difficult”.  
Different colors represent different heads.*

---
## 🧩 Project Structure
```bash

Attention-Is-All-You-Need-Paper-Replicating/
│
├── 📁 src/
│ ├── 1_input_embedding/
│ │ ├── embeddings.py → TokenEmbedding (makale 3.4)
│ │ └── positional_encoding.py → Sinusoidal encoding (makale 3.5)
│ │
│ ├── 2_attention/
│ │ ├── scaled_dot_product.py → softmax(QKᵀ / √dₖ)V (makale 3.2.1)
│ │ └──multi_head_attention.py → Concat(head₁,…,headₕ)W₀ (makale 3.2.2)
│ │ 
│ ├── 3_feed_forward/
│ │ └── positionwise_ffn.py → FFN(x)=max(0,xW₁+b₁)W₂+b₂ (makale 3.3)
│ │
│ ├── 4_encoder_decoder/
│ │ ├── encoder_layer.py → MultiHead + FFN + Residual (makale 3.1)
│ │ ├── decoder_layer.py → Masked + Encoder-Attention + FFN
│ │ ├── encoder.py → 6-layer Encoder stack
│ │ └── decoder.py → 6-layer Decoder stack
│ │
│ ├── 5_transformer/
│ │ └── transformer.py → Encoder + Decoder + Linear + Softmax (makale 3.1 genel mimari)
│ │
│ ├── 6_training/
│ │ ├── optimizer.py → Noam LR schedule (makale 5.2)
│ │ ├── loss_fn.py → CrossEntropy + Label Smoothing (makale 5.3)
│ │ ├── train_utils.py → train_step(), create_masks()
│ │ └── regularization.py → Dropout (makale 5.3)
│
├── 📁images/
│ ├── Figure 1: The Transformer - model architecture.png
│ ├── Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.png
│ └── Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6..png
│
│└──requirements.txt
```
---
## ⚡Feedback

For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)




