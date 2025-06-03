# Non Verbis, Sed GRPO repository

First tests on a GRPO-trained rebus solver - work in progress.

Which models and where to find them on HF:
| Model                 | SFT (5070)                                | SFT (500)                                     | SFT (5070) + GRPO                                 | SFT (500) + GRPO |
|-----------------------|-------------------------------------------|-----------------------------------------------|---------------------------------------------------|----------------------|
| Phi-3 Mini 3.8B       |`gsarti/phi3-mini-rebus-solver-adapters`   |`saracandu/phi3-mini-rebus-solver-coldstart`   |`saracandu/phi3-mini-rebus-solver-adapters-grpo`   |`saracandu/phi3-mini-rebus-solver-coldstart-grpo`|
| LLaMA-3.1 8B Instruct |`gsarti/llama-3.1-8b-rebus-solver-adapters`|`saracandu/llama-3.1-8b-rebus-solver-coldstart`|`saracandu/llama-3.1-8b-rebus-solver-adapters-grpo`| `saracandu/llama-3.1-8b-rebus-solver-coldstart-grpo` |

---

SFT+GRPO (old testing phase):
| Model                 | SFT (5070)   | Tested? | SFT (500)   | Tested? | SFT (5070) + GRPO   | Tested? | SFT (500) + GRPO   | Tested? |
|-----------------------|-------------|---------|--------------|---------|---------------------|---------|--------------------|---------|
| Phi-3 Mini 3.8B       | ✅​ (gab)    | ✅       | ✅​           | ✅      | ✅​                  | ✅      | ✅​                   | ✅      |
| LLaMA-3.1 8B Instruct | ✅​ (gab)    | ✅       | ✅​           | ✅      | ✅​                  | ✅      | ✅                   | ✅      |

---

Mounting GRPO (300 steps) on SFT checkpoints:
| Model / Config        | Phi-3 Mini 3.8B | Tested? | LLaMA-3.1 8B Instruct | Tested? |
|-----------------------|------------------|----------|------------------------|----------|
| SFT (5070) + GRPO     | ✅               | ✅       | ✅                      | ✅       |
| SFT (4000) + GRPO     | ⏱️                | -        | ⏱️                      | -        |
| SFT (3000) + GRPO     | ⏱️                | -        | ⏱️                      | -        |
| SFT (2000) + GRPO     | 🏃🏼​                | -        | ⏱️                      | -        |
| SFT (500) + GRPO      | ✅                 | ✅         | ✅                      | ✅         |

---

Re-run SFT testing phases:
| Model / Config        | Phi-3 Mini 3.8B | LLaMA-3.1 8B Instruct |
|-----------------------|-----------------|-----------------------|
| SFT (5070)            | -               | ⏱️                    |
| SFT (4000)            | -               | ⏱️                    |
| SFT (3000)            | -               | ⏱️                    |
| SFT (2000)            | -​               | ⏱️                    |
| SFT (1000)            | -​               | ⏱️                    |
| SFT (500)             | -               | -                     |

---

TO-DO: run SFT+GRPO testing phases


