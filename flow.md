# Action Agent: Training and Query Flow

## 1. Big picture

```
 PHASE A: TRAINING (Colab, once)                 PHASE B: QUERYING (app.py, every message)
 ───────────────────────────────                 ─────────────────────────────────────────
 your 500 examples                               user types a message
        │                                               │
        ▼                                               ▼
 clean + split + add examples                    wrap it in the SAME prompt layout
        │                                               │
        ▼                                               ▼
 build "prompt + JSON answer + stop token"       Qwen + your adapter writes JSON
        │                                               │
        ▼                                               ▼
 train a small LoRA adapter                      Python checks, remembers, asks, confirms
        │                                               │
        ▼                                               ▼
 save adapter folder  ───────────────────────►   execute_action(...)
```

Rule that ties both phases together: **the prompt layout in training and in the app must be identical.**

---

## 2. Training flow (Colab notebook)

```
Cell 0-1   install libraries
   │
   ▼
Cell 2     load actions_training_500.json
           └─ clean(): keep utterance, action, parameters; drop empty values
   │       {utterance: "Create a task ...", action: "create_task", parameters: {...}}
   ▼
Cell 4     SPLIT FIRST (80% train / 20% validation)
   │
   ├─ validation  ────────────────► kept untouched (the exam)
   │
   └─ training  ──► add three kinds of extra examples
                    ├─ synonym variants   "support ticket" -> "service request"
                    ├─ bare requests      "create a support ticket"  -> parameters {}
                    └─ non-actions        "hi", "what is a service request" -> action null
   │
   ▼
Cell 5     load tokenizer + base model (Qwen2.5-0.5B-Instruct)
   │
   ▼
Cell 6     turn each example into numbers
           ┌────────────────────────────────────────────────────────────────┐
           │ text = PROMPT + JSON ANSWER + <|im_end|>                       │
           │                                                                │
           │ input_ids: [ prompt tokens ][ answer tokens ][ end token ]     │
           │ labels:    [ -100 ... -100  ][ answer tokens ][ end token ]     │
           │              (ignored)         (graded)         (graded)       │
           └────────────────────────────────────────────────────────────────┘
   │
   ▼
Cell 7     attach LoRA (freeze Qwen, add small trainable matrices A and B)
   │
   ▼
Cell 8     train loop (repeated for 4 passes over the data)
           ┌──────────────────────────────────────────────┐
           │ take 4 examples (pad to same length)         │
           │        ▼                                     │
           │ model predicts next token at every position  │
           │        ▼                                     │
           │ loss = error on graded positions only        │
           │        ▼                                     │
           │ backprop -> update only LoRA weights         │
           │        ▼                                     │
           │ after 4 batches (16 examples): optimizer step│
           └──────────────────────────────────────────────┘
           end of each pass: compute validation loss, keep the best
   │
   ▼
Cell 9     save adapter + tokenizer  ->  /working/action-model-v2
   │
   ▼
Cell 10    evaluate on validation
           prompt -> generate -> parse JSON -> compare with expected
           prints: action accuracy, exact-parameter accuracy, per-action table, failures
   │
   ▼
Cell 11    zip -> download -> unzip on your Mac -> set as adapter path in app.py
```

### The prompt (identical in training and in the app)

```
### Instruction:
Determine the action and parameters required for this user request.

### User:
<user message>

### Response:
```

The model completes it with `{"action": "...", "parameters": {...}}`.

---

## 3. Query flow (app.py, one chat message)

```
user message
   │
   ▼
┌─────────────────────────────────────────────┐
│ 1. Control words?                           │
│    "yes / confirm"  (when ready)  ──► execute_action ──► reset state ──► reply
│    "cancel / no"                  ──► reset state ──► reply
└──────────────────────┬──────────────────────┘
                       │ neither
                       ▼
┌─────────────────────────────────────────────┐
│ 2. Ask the model about THIS one message     │
│    build_prompt(message) ──► generate       │
│    stops when one JSON object is complete   │
│    parse JSON (if invalid -> treated as {}) │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ 3. Guard 1: is the model's action believable?
│    action word appears in the message?  ──► accept
│    no, and exactly one other action's words match ──► reroute to that one
│    no, and none / several match         ──► reject
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ 4. Guard 2: are the values the user's own?  │
│    keep a value only if it shares words with the message
│    (date/time fields: only need something date/time-like)
│    drop anything not in the action's schema │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ 5. Also read explicit phrases (no model)    │
│    "title is login broken"                  │
│    "set the priority to high"               │
└──────────────────────┬──────────────────────┘
                       ▼
              decide what this message was
```

### Deciding what the message was (`apply_turn`)

```
model gave a valid action?
├── YES
│    ├── no action in progress, or a different action ──► NEW ACTION (start fresh)
│    └── same action as in progress
│         ├── new values found ──► UPDATE (merge into what we have)
│         └── nothing new      ──► fall through to "no new action" below
│
└── NO / nothing new
     ├── action in progress
     │    ├── explicit "field is value" found  ──► UPDATE
     │    ├── still collecting AND message is not a question
     │    │        ──► treat the message as the answer to the field we just asked
     │    └── otherwise ──► OFF-TOPIC (keep the action, remind the user)
     │
     └── nothing in progress
          ├── model action was rejected ──► UNCLEAR (show what I can do)
          └── otherwise                 ──► NO ACTION (chit-chat, show what I can do)
```

### State and reply

```
Python computes:  missing = required fields that are empty

missing is not empty ──► status "collecting" ──► bot asks for the FIRST missing field
                                                 e.g. "What should the title be?"
missing is empty     ──► status "ready"      ──► bot shows summary + [Confirm] [Cancel]
```

```
[Confirm]  ──► execute_action(action, parameters)   (placeholder for your real dispatcher)
[Cancel]   ──► drop the action
```

---

## 4. Example conversation

```
User : create a support ticket
Model: {"action":"create_support_ticket","parameters":{}}
State: collecting, missing = [title, description]
Bot  : Sure, let's raise a support ticket. What should the title be?

User : login is broken
Model: (no valid action)
Python: no new action + still collecting -> answer to "title"
State: collecting, missing = [description]
Bot  : Got it. How would you describe it?

User : users cannot sign in to the portal
Python: answer to "description"
State: ready
Bot  : Here's what I have: Title: login is broken / Description: users cannot ... Shall I go ahead?
       [Confirm] [Cancel]

User : (clicks Confirm)
Bot  : Done! I've submitted your request to raise a support ticket.
```

---

## 5. Who does what

| The model does | Python does |
|---|---|
| Reads ONE message | Remembers the conversation |
| Picks an action, extracts values | Checks the action and values are real |
| Writes JSON | Tracks missing fields, asks questions, writes replies |
| | Confirms and executes |

---

## 6. Files

| File | Purpose |
|---|---|
| Colab notebook (cells 0-11) | Trains and evaluates the adapter |
| `/working/action-model-v2/` | Saved adapter: `adapter_config.json`, adapter weights, tokenizer files |
| `app.py` | Streamlit chat: loads base Qwen + adapter, runs the query flow above |
