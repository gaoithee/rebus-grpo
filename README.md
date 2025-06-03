# Non Verbis, Sed GRPO repository

First tests on a GRPO-trained rebus solver - work in progress.

Which models and where to find them on HF:
| Model                 | SFT (5070)                                | SFT (500)                                     | SFT (5070) + GRPO                                 | SFT (500) + GRPO |
|-----------------------|-------------------------------------------|-----------------------------------------------|---------------------------------------------------|----------------------|
| Phi-3 Mini 3.8B       |`gsarti/phi3-mini-rebus-solver-adapters`   |`saracandu/phi3-mini-rebus-solver-coldstart`   |`saracandu/phi3-mini-rebus-solver-adapters-grpo`   |`saracandu/phi3-mini-rebus-solver-coldstart-grpo`|
| LLaMA-3.1 8B Instruct |`gsarti/llama-3.1-8b-rebus-solver-adapters`|`saracandu/llama-3.1-8b-rebus-solver-coldstart`|`saracandu/llama-3.1-8b-rebus-solver-adapters-grpo`| 🏃 |

| Model                 | SFT (5070)   | Tested? | SFT (500)   | Tested? | SFT (5070) + GRPO   | Tested? | SFT (500) + GRPO   | Tested? |
|-----------------------|-------------|---------|--------------|---------|---------------------|---------|--------------------|---------|
| Phi-3 Mini 3.8B       | ✅​ (gab)    | ✅       | ✅​           | ✅      | ✅​                  | ✅      | ✅​                   | ✅      |
| LLaMA-3.1 8B Instruct | ✅​ (gab)    | ✅       | ✅​           | ✅      | ✅​                  | ✅      | 🏃​                   | ❌      |
| Qwen-2.5 7B Instruct  | 🏃           | ❌      | 🏃           | ❌      | ❌                  | ❌      | ❌                   | ❌      |
| Qwen-2.5 3B Instruct  | 🏃           | ❌      | 🏃           | ❌      | ❌                  | ❌      | ❌                   | ❌      |
| Qwen-3 4B             | 🏃 (gab)     | ❌      | 🏃 (gab)     | ❌      | ❌                  | ❌      | ❌                   | ❌      |
| Phi-4 Mini            | 🏃 (queue)   | ❌      | 🏃 (queue)   | ❌      | ❌                  | ❌      | ❌                   | ❌      |


