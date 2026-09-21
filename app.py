import json
import os
import re

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_ADAPTER_PATH = (
    "/Users/sankar/Documents/mock_interview/"
    "actionclassifier-llm/working/action-model-v2"
)

MAX_NEW_TOKENS = 300

# Two ways of prompting the model. Use the one your adapter was trained on.
PROMPT_MODE_HISTORY = "Full conversation (original prompt)"
PROMPT_MODE_SEQUENTIAL = "State + latest message"

# Full-conversation mode: also send the bot's own replies to the model?
INCLUDE_ASSISTANT_TURNS = True


# ============================================================
# ACTION SCHEMAS
# ============================================================

ACTION_SCHEMAS = {
    "create_task": {
        "required": ["title"],
        "optional": ["description", "priority", "due_date", "assignee"],
    },
    "send_message": {
        "required": ["recipient", "message"],
        "optional": ["channel"],
    },
    "schedule_meeting": {
        "required": ["title", "participants", "date", "start_time", "duration"],
        "optional": ["description", "location", "meeting_link"],
    },
    "create_calendar_event": {
        "required": ["title", "date", "start_time", "end_time"],
        "optional": ["location", "description", "reminder"],
    },
    "search_documents": {
        "required": ["query"],
        "optional": ["folder", "file_type", "date_range"],
    },
    "send_email": {
        "required": ["recipient", "subject", "body"],
        "optional": ["cc", "bcc", "attachments"],
    },
    "update_task": {
        "required": ["task_id"],
        "optional": ["title", "description", "priority", "due_date", "assignee", "status"],
    },
    "delete_task": {
        "required": ["task_id"],
        "optional": [],
    },
    "get_calendar_events": {
        "required": ["date"],
        "optional": ["start_time", "end_time", "calendar"],
    },
    "create_support_ticket": {
        "required": ["title", "description"],
        "optional": ["priority", "category", "attachments"],
    },
}

# Natural-language phrase for each action (used in bot replies)
ACTION_LABELS = {
    "create_task": "create a task",
    "send_message": "send a message",
    "schedule_meeting": "schedule a meeting",
    "create_calendar_event": "create a calendar event",
    "search_documents": "search your documents",
    "send_email": "send an email",
    "update_task": "update a task",
    "delete_task": "delete a task",
    "get_calendar_events": "check your calendar",
    "create_support_ticket": "raise a support ticket",
}

HELP_MESSAGE = (
    "I can help with things like creating tasks, sending emails or messages, "
    "scheduling meetings, managing calendar events, searching documents and "
    "raising support tickets. What would you like to do?"
)

# Typed replies that confirm / cancel without clicking the buttons
CONFIRM_WORDS = {
    "yes", "y", "yep", "yeah", "sure", "ok", "okay", "confirm", "confirmed",
    "go ahead", "do it", "proceed", "yes please", "looks good",
}
CANCEL_WORDS = {
    "cancel", "stop", "abort", "discard", "never mind", "nevermind",
    "forget it", "drop it", "cancel it",
}
DECLINE_WORDS = {"no", "nope", "nah"}  # only treated as cancel when awaiting confirmation


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading model...")
def load_model(adapter_path):

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")

    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        raise FileNotFoundError(f"adapter_config.json was not found in: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {}
    if DEVICE in ("mps", "cuda"):
        kwargs["torch_dtype"] = torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ============================================================
# PROMPT
# ============================================================
# NOTE: if your LoRA adapter was trained on a different prompt layout,
# adjust this function to match your training format.

def build_prompt_sequential(state, user_message):
    """
    Sequential prompt: the action state built from all PREVIOUS turns
    + the single NEW user message. The model no longer has to re-read
    (and re-interpret) the whole chat every time.
    """

    schemas = json.dumps(ACTION_SCHEMAS)

    if state:
        current = json.dumps(
            {
                "action": state["action"],
                "parameters": state["parameters"],
                "missing_params": state["missing_params"],
            }
        )
    else:
        current = "none (no action in progress)"

    return f"""You are an action extraction system inside a chat assistant.

Read the LATEST USER MESSAGE and decide:
1. Is the user asking to perform an action, or just chatting / asking something unrelated?
2. If an action is already in progress, is the message giving more details for it,
   changing a detail, or starting a different action?

RULES:
- Use the CURRENT STATE as the memory of everything said so far.
- Keep all parameters already in the CURRENT STATE.
- If the user changes a parameter, use the newest value.
- Only use parameters defined for the action in AVAILABLE ACTIONS.
- NEVER invent or assume values.
- If no action is requested, return action null.
- Return ONLY valid JSON. No explanations. No markdown.

AVAILABLE ACTIONS:
{schemas}

CURRENT STATE:
{current}

LATEST USER MESSAGE:
{user_message}

REQUIRED OUTPUT FORMAT:
{{"action": "action_name or null", "parameters": {{}}, "missing_params": [], "status": "idle | collecting | ready"}}

JSON:
"""


def build_prompt_history(messages):
    """
    Original prompt format: the conversation, in order, from the message that
    started the current action up to the newest message.
    """

    conversation = ""

    for message in messages:

        if message["role"] == "assistant" and not INCLUDE_ASSISTANT_TURNS:
            continue

        conversation += f"{message['role'].upper()}: {message['content']}\n"

    schemas = json.dumps(ACTION_SCHEMAS, indent=2)

    return f"""
You are a conversational action extraction system.

You must analyze the COMPLETE conversation history.

Your job is to determine:

1. The current user action.
2. All parameters that have already been provided.
3. Any parameters that are still missing.
4. Whether the action is ready for execution.

IMPORTANT RULES:

- Analyze the entire conversation.
- Remember information from previous messages.
- Preserve previously provided parameters.
- If the user changes a parameter, use the newest value.
- NEVER invent information.
- NEVER assume missing values.
- NEVER execute an action.
- Return ONLY valid JSON.
- Do not write explanations.
- Do not use markdown.

AVAILABLE ACTIONS:

{schemas}

REQUIRED OUTPUT FORMAT:

{{
    "action": "action_name",
    "parameters": {{
    }},
    "missing_params": [],
    "status": "collecting"
}}

STATUS RULES:

"idle"
= no actionable request exists.

"collecting"
= an action exists but one or more required parameters are missing.

"ready"
= ALL required parameters for the action are present.

Example:

If the conversation is:

USER:
Create a meeting with Rahul.

Then return something like:

{{
    "action": "schedule_meeting",
    "parameters": {{
        "participants": ["Rahul"]
    }},
    "missing_params": [
        "title",
        "date",
        "start_time",
        "duration"
    ],
    "status": "collecting"
}}

CONVERSATION:

{conversation}

JSON:
"""


def build_prompt(mode, messages, state, segment_start):
    """
    messages[-1] is always the newest user message.
    segment_start = index of the user message that started the action in progress
    (None when nothing is in progress).
    """

    if mode == PROMPT_MODE_SEQUENTIAL:
        return build_prompt_sequential(state, messages[-1]["content"])

    # No action in progress -> the model only sees the new message, so earlier
    # chit-chat / failed attempts / finished actions can't leak into it.
    if state is None or segment_start is None:
        return build_prompt_history(messages[-1:])

    return build_prompt_history(messages[segment_start:])


# ============================================================
# MODEL INFERENCE
# ============================================================

def generate(prompt, adapter_path):

    tokenizer, model = load_model(adapter_path)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def extract_json(text):

    text = text.replace("```json", "").replace("```", "")

    start = text.find("{")

    if start == -1:
        raise ValueError("The model did not return a JSON object.")

    # raw_decode stops at the end of the first complete JSON object,
    # so trailing text or a second object won't break parsing.
    obj, _ = json.JSONDecoder().raw_decode(text[start:])

    if not isinstance(obj, dict):
        raise ValueError("The model returned JSON, but not an object.")

    return obj


# ============================================================
# STATE HANDLING (deterministic — the model only extracts)
# ============================================================

def is_empty(value):
    if value is None or value == [] or value == {}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "null"}:
        return True
    return False


def clean_parameters(action, params):
    """Keep only parameters that exist in the schema and have a real value."""

    if not isinstance(params, dict):
        return {}

    schema = ACTION_SCHEMAS[action]
    allowed = set(schema["required"]) | set(schema["optional"])

    return {
        key: value
        for key, value in params.items()
        if key in allowed and not is_empty(value)
    }


def build_state(action, parameters):
    """Compute missing params / status ourselves instead of trusting the model."""

    missing = [
        field
        for field in ACTION_SCHEMAS[action]["required"]
        if is_empty(parameters.get(field))
    ]

    return {
        "action": action,
        "parameters": parameters,
        "missing_params": missing,
        "status": "ready" if not missing else "collecting",
    }


def apply_model_result(prev_state, result):
    """
    Fold ONE new model result into the running state.
    Returns (new_state, intent) where intent is one of:

      new_action          user started an action (or switched to a different one)
      update_action       user added / changed details of the action in progress
      no_change           same action, but nothing new was extracted
      off_topic           chatting while an action is in progress (state kept)
      no_action           chatting, nothing in progress
      unsupported_action  model named an action we don't have (state kept)
    """

    action = result.get("action")

    if isinstance(action, str) and action.strip().lower() in {"", "null", "none"}:
        action = None

    if action is None:
        return prev_state, ("off_topic" if prev_state else "no_action")

    if action not in ACTION_SCHEMAS:
        return prev_state, "unsupported_action"

    new_params = clean_parameters(action, result.get("parameters"))

    # Same action -> merge into what we already collected
    if prev_state and prev_state["action"] == action:
        merged = {**prev_state["parameters"], **new_params}
        intent = "update_action" if merged != prev_state["parameters"] else "no_change"
        return build_state(action, merged), intent

    # Different (or first) action -> start fresh
    return build_state(action, new_params), "new_action"


# ============================================================
# BOT MESSAGES
# ============================================================

def human_list(items):
    items = [item.replace("_", " ") for item in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def format_parameters(parameters):
    lines = []
    for key, value in parameters.items():
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
    return "\n".join(lines)


def next_step(state):
    if state["status"] == "ready":
        return (
            "Here's what I have:\n\n"
            + format_parameters(state["parameters"])
            + "\n\nShall I go ahead?"
        )
    return "I still need the " + human_list(state["missing_params"]) + "."


def compose_reply(prev_state, state, intent):

    if intent == "no_action":
        return HELP_MESSAGE

    if intent == "unsupported_action":
        return "I can't do that yet. " + HELP_MESSAGE

    label = ACTION_LABELS[state["action"]]

    if intent == "off_topic":
        return (
            f"I'm not sure how that relates to your request to {label}. "
            + next_step(state)
            + " (Say \"cancel\" to drop it.)"
        )

    if intent == "no_change":
        return "I didn't catch any new details. " + next_step(state)

    if intent == "update_action":
        return "Got it. " + next_step(state)

    # new_action
    if prev_state and prev_state["action"] != state["action"]:
        prefix = f"Okay, switching over to {label}."
    else:
        prefix = f"Sure, let's {label}."

    return prefix + " " + next_step(state)


# ============================================================
# EXECUTION
# ============================================================

def execute_action(action, parameters):
    # =====================================================
    # REPLACE THIS WITH YOUR REAL ACTION DISPATCHER
    # result = my_dispatcher[action](**parameters)
    # =====================================================
    return f"✅ Done! I've submitted your request to {ACTION_LABELS[action]}."


def confirm_and_execute():
    ss = st.session_state
    state = ss.action_state
    message = execute_action(state["action"], state["parameters"])
    ss.action_state = None
    ss.segment_start = None
    ss.messages.append({"role": "assistant", "content": message})


def cancel_action():
    ss = st.session_state
    ss.action_state = None
    ss.segment_start = None
    ss.messages.append(
        {
            "role": "assistant",
            "content": "No problem, I've dropped that. What would you like to do next?",
        }
    )


def detect_control(text, state):
    """Typed confirm / cancel while an action is in progress."""

    if not state:
        return None

    normalized = re.sub(r"[^a-z ]", "", text.lower()).strip()
    ready = state["status"] == "ready"

    if ready and normalized in CONFIRM_WORDS:
        return "confirm"

    if normalized in CANCEL_WORDS:
        return "cancel"

    if ready and normalized in DECLINE_WORDS:
        return "cancel"

    return None


# ============================================================
# ONE CHAT TURN
# ============================================================

def handle_turn(user_text, adapter_path, prompt_mode):

    ss = st.session_state

    ss.messages.append({"role": "user", "content": user_text})

    user_index = len(ss.messages) - 1
    prev_state = ss.action_state

    # 1) Simple control replies (confirm / cancel)
    control = detect_control(user_text, prev_state)

    if control == "confirm":
        ss.debug = {"intent": "confirm"}
        confirm_and_execute()
        return

    if control == "cancel":
        ss.debug = {"intent": "cancel"}
        cancel_action()
        return

    # 2) Ask the model about the NEW message only, given the running state
    debug = {"prev_state": prev_state}

    try:
        prompt = build_prompt(prompt_mode, ss.messages, prev_state, ss.segment_start)
        raw = generate(prompt, adapter_path)
        debug["prompt_mode"] = prompt_mode
        debug["raw_output"] = raw

        result = extract_json(raw)
        debug["model_result"] = result

        # 3) Decide the intent + fold into the running state
        state, intent = apply_model_result(prev_state, result)
        debug["intent"] = intent
        debug["new_state"] = state

        ss.action_state = state

        # Track where the current action's conversation begins
        if intent == "new_action":
            ss.segment_start = user_index
        elif state is None:
            ss.segment_start = None

        reply = compose_reply(prev_state, state, intent)

    except FileNotFoundError as error:
        debug["error"] = str(error)
        reply = f"⚠️ {error}"

    except Exception as error:
        debug["error"] = repr(error)
        reply = "Sorry, I couldn't understand that. Could you rephrase it?"

    ss.debug = debug
    ss.messages.append({"role": "assistant", "content": reply})


# ============================================================
# UI
# ============================================================

def render_history():

    ss = st.session_state
    state = ss.action_state
    last = len(ss.messages) - 1

    for i, message in enumerate(ss.messages):

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            # Confirmation buttons live inside the bot's last message
            awaiting_confirmation = (
                i == last
                and message["role"] == "assistant"
                and state is not None
                and state["status"] == "ready"
            )

            if awaiting_confirmation:

                col_confirm, col_cancel = st.columns(2)

                if col_confirm.button(
                    "✅ Confirm",
                    type="primary",
                    use_container_width=True,
                    key="confirm_btn",
                ):
                    confirm_and_execute()
                    st.rerun()

                if col_cancel.button(
                    "✖ Cancel",
                    use_container_width=True,
                    key="cancel_btn",
                ):
                    cancel_action()
                    st.rerun()


def main():

    st.set_page_config(page_title="Action Agent", page_icon="🤖", layout="wide")

    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("action_state", None)
    ss.setdefault("debug", None)
    ss.setdefault("segment_start", None)

    # ---------------- Sidebar ----------------

    st.sidebar.title("⚙️ Configuration")

    adapter_path = st.sidebar.text_input("LoRA adapter path", value=DEFAULT_ADAPTER_PATH)

    st.sidebar.write(f"**Device:** `{DEVICE}`")
    st.sidebar.write(f"**Base model:** `{BASE_MODEL}`")

    prompt_mode = st.sidebar.selectbox(
        "Prompt format",
        [PROMPT_MODE_HISTORY, PROMPT_MODE_SEQUENTIAL],
        help="Use the format your adapter was trained on.",
    )

    show_debug = st.sidebar.toggle("Show debug info", value=False)

    st.sidebar.divider()

    if st.sidebar.button("🗑️ Clear conversation", use_container_width=True):
        ss.messages = []
        ss.action_state = None
        ss.debug = None
        ss.segment_start = None
        st.rerun()

    if show_debug and ss.debug:
        with st.sidebar.expander("Last turn", expanded=True):
            st.json(ss.debug)
        if ss.action_state:
            with st.sidebar.expander("Current action state"):
                st.json(ss.action_state)

    # ---------------- Chat ----------------

    st.title("🤖 Action Agent")

    render_history()

    user_input = st.chat_input("What would you like me to do?")

    if user_input:

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                handle_turn(user_input, adapter_path, prompt_mode)

        # Redraw so the new messages (and confirm buttons) render in order
        st.rerun()


if __name__ == "__main__":
    main()