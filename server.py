# server.py
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from tokenizers import Tokenizer

from models.GPT import GPT
model_checkpoint_path = '/Users/josh/Documents/GitHub/diogenes-server/checkpoints/diogenes-v1.pth'
tokenizer_path = '/Users/josh/Documents/GitHub/diogenes-server/data/tokenizer.json'

# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

# Load model once at startup
model = None
tokenizer = None
sessions: dict[str, list] = {}  # session_id -> message history

@app.on_event("startup")
def load_model():
    global model, tokenizer
    # Load your model and tokenizer here
    
    chkpt = torch.load(model_checkpoint_path, weights_only=False, map_location=device)
    model_config = chkpt['model_config']

    tok_path = tokenizer_path or chkpt.get('tokenizer_path', 'data/tokenizer.json')
    tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = tokenizer.get_vocab_size()

    model = GPT(
        vocab_size=vocab_size,
        embed_dim=model_config['emb_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        is_causal=True,
    ).to(device)
    model.load_state_dict(chkpt['model_state_dict'])
    model.eval()


def sample(logits, temperature, top_k, top_p):
    logits = logits / temperature

    if top_k is not None:
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = float('-inf')

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_logits[cumulative_probs - sorted_probs >= top_p] = float('-inf')
        logits = torch.scatter(logits, 0, sorted_indices, sorted_logits)

    # If all logits were filtered out, fall back to greedy
    if not torch.isfinite(logits).any():
        return logits.argmax().item()

    probs = torch.softmax(logits, dim=-1)
    probs = probs.clamp(min=0)  # guard against tiny negatives from floating-point error
    return torch.multinomial(probs, num_samples=1).item()


@torch.no_grad()
def generate_stream(
    model,
    tokenizer,
    prompt,
    max_new_tokens=128,
    max_context=256,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    system_token_len=0,
):
    """Yields decoded text for each token as it is generated."""
    encoded = tokenizer.encode(prompt)
    context = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Truncate to max_context, pinning the system-prompt prefix so it is
        # never evicted from the context window.
        if system_token_len > 0 and context.shape[1] > max_context:
            sys_part  = context[:, :system_token_len]
            conv_part = context[:, system_token_len:]
            conv_part = conv_part[:, -(max_context - system_token_len):]
            ctx = torch.cat([sys_part, conv_part], dim=1)
        else:
            ctx = context[:, -max_context:]

        logits = model(ctx)
        next_token = sample(logits[0, -1], temperature=temperature, top_k=top_k, top_p=top_p)

        if next_token == 3:  # <|im_end|>
            break

        yield tokenizer.decode([next_token])
        context = torch.cat([context, torch.tensor([[next_token]], device=device)], dim=1)


def generate(model, tokenizer, prompt, **kwargs):
    return "".join(generate_stream(model, tokenizer, prompt, **kwargs))

def build_prompt(messages, system=None):
    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    max_context: int = 256
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    default_system = "You are a helpful and friendly AI assistant named Diogenes that is knowledgeable in every field. You are tasked with answering questions and requests from a user with very short, concise and informative answers."
    
    system_token_len = len(
        tokenizer.encode(f"<|im_start|>system\n{default_system}<|im_end|>\n").ids
    )

    messages = sessions.setdefault(req.session_id, [])
    messages.append({"role": "user", "content": req.message})

    prompt = build_prompt(messages, system=default_system)
    reply = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=req.max_new_tokens,
        max_context=req.max_context,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        system_token_len=system_token_len,
    )

    messages.append({"role": "assistant", "content": reply})
    return ChatResponse(reply=reply)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    default_system = "You are a helpful and friendly AI assistant named Diogenes that is knowledgeable in every field. You are tasked with answering questions and requests from a user with concise and informative answers. Please think deeply about your answers."

    system_token_len = len(
        tokenizer.encode(f"<|im_start|>system\n{default_system}<|im_end|>\n").ids
    )

    messages = sessions.setdefault(req.session_id, [])
    messages.append({"role": "user", "content": req.message})
    prompt = build_prompt(messages, system=default_system)

    reply_parts: list[str] = []

    def token_stream():
        for token in generate_stream(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=req.max_new_tokens,
            max_context=req.max_context,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            system_token_len=system_token_len,
        ):
            reply_parts.append(token)
            yield f"data: {json.dumps(token)}\n\n"
        # Persist the completed reply to history once generation finishes
        messages.append({"role": "assistant", "content": "".join(reply_parts)})

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disables buffering in nginx / Cloudflare tunnels
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"cleared": session_id}