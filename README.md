# Action Classifier — Fine-tuned Qwen 0.5B

A local, zero-API-cost action extraction model for chat/tool-calling assistants.

Instead of sending every user message to a frontier model to decide *"is this a task, an email, or chit-chat?"*, this project fine-tunes **Qwen2.5-0.5B-Instruct** with a small LoRA adapter that does one narrow job: read one user message and return JSON.

Everything else — conversation memory, missing-field tracking, follow-up questions, validation, confirmation, execution — is plain Python.

> Small model for perception. Plain code for control.

## Why

Most assistants route simple requests through a large model even though the task is pattern matching, not reasoning:

```
"create a task titled login bug"  ->  {"action": "create_task", "parameters": {"title": "login bug"}}
```

That is overkill for GPT-4-class models. Here it runs on a laptop:

- **34 MB** LoRA adapter (~1 GB base model), runs offline
- **Milliseconds** per inference on Apple Silicon (MPS), CUDA, or CPU
- **$0** API cost, no user data leaves the machine

## Supported actions

| Action | Required parameters |
|---|---|
| `create_task` | title |
| `send_message` | recipient, message |
| `schedule_meeting` | title, participants, date, start_time, duration |
| `create_calendar_event` | title, date, start_time, end_time |
| `search_documents` | query |
| `send_email` | recipient, subject, body |
| `update_task` | task_id |
| `delete_task` | task_id |
| `get_calendar_events` | date |
| `create_support_ticket` | title, description |

Schemas live in `action_list.json` and `ACTION_SCHEMAS` in `app.py`.

## Architecture

```
              TRAINING (Colab, once)                    RUNTIME (app.py, per message)
    --------------------------------------      --------------------------------------
    500 labelled utterances                     user sends a message
            |                                            |
            v                                            v
    clean + split 80/20                         small model returns JSON
    augment (synonyms, bare requests,           {"action": ..., "parameters": {...}}
             non-actions -> null)                        |
            |                                            v
            v                                   Python guards:
    prompt + JSON answer + <|im_end|>             - action must be believable
            |                                     - values must come from the user
            v                                            |
    LoRA fine-tune Qwen2.5-0.5B                          v
            |                                   state machine: merge, ask for
            v                                   missing fields, confirm, execute
    save adapter -> working/action-model-v2
```

The model never remembers, never decides, never executes. It classifies and extracts; Python owns the conversation state. The prompt layout used at training time and at inference time is identical — see `flow.md` for the full walkthrough.

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit chat app: loads base model + adapter, runs the query flow |
| `finetune.ipynb` | Colab notebook: data prep, LoRA training, evaluation, export |
| `flow.md` | Detailed training + query flow documentation |
| `action_list.json` | Action schemas (required/optional parameters) |
| `actions_training_500.json` | Labelled training data |
| `working/action-model-v2/` | Trained LoRA adapter + tokenizer |
| `working/action-model-v2/checkpoint-*/` | Intermediate training checkpoints |

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers peft streamlit
streamlit run app.py
```

The app loads `Qwen/Qwen2.5-0.5B-Instruct` from Hugging Face (cached after first run) and merges the adapter from `working/action-model-v2`. Set the adapter path and prompt format in the sidebar.

## Example

```
User : create a support ticket
Bot  : {"action":"create_support_ticket","parameters":{}}  ->  "What should the title be?"

User : login is broken
Bot  : "How would you describe it?"

User : users cannot sign in to the portal
Bot  : "Here's what I have: ... Shall I go ahead?"  [Confirm] [Cancel]
```

## Training

Open `finetune.ipynb` in Colab (GPU) and run all cells. It:

1. Loads and cleans `actions_training_500.json`
2. Splits 80/20 **before** augmentation (validation stays untouched)
3. Adds synonym variants, bare requests, and non-action examples
4. Applies LoRA to Qwen2.5-0.5B-Instruct and trains
5. Evaluates on the held-out split (action accuracy, exact-parameter accuracy)
6. Saves the adapter to `working/action-model-v2`

## Limitations

- Trained on a small domain-specific dataset (500 seed examples); expect weak generalization to unseen actions
- CPU-only inference is possible but slow
- `execute_action()` in `app.py` is a placeholder — wire it to your real dispatcher

## License

See the repository for license details. Base model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
