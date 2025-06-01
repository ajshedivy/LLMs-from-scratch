# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from rich.progress import track
from rich.layout import Layout
from rich.align import Align
import os
import torch.nn.functional as F
import math
import numpy as np

#####################################
# Chapter 2
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# Chapter 3
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size, interactive=True, is_trained=False):
    console = Console(width=100)
    
    # Show model training status
    model_status = "[bold green]TRAINED MODEL[/bold green]" if is_trained else "[bold red]UNTRAINED MODEL[/bold red]"
    
    console.print(Panel.fit(
        f"🤖 [bold blue]GPT Token Generation Process[/bold blue] - {model_status}",
        subtitle="Watch how tokens are selected one by one",
        width=95,
        border_style="cyan"
    ))
    
    # idx is (B, T) array of indices in the current context
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Display initial context
    console.print("\n[bold green]Initial Context:[/bold green]")
    initial_text = tokenizer.decode(idx.squeeze(0).tolist())
    console.print(Panel(f"[yellow]\"{initial_text}\"[/yellow]", width=95))
    
    if interactive:
        console.print("[bold magenta]Interactive Mode:[/bold magenta] Press Enter after each token, type 'auto' to switch to automatic")
    
    auto_mode = not interactive
    
    # Create a table to show token generation history
    history_table = Table(title="[bold]Token Generation History[/bold]", width=95)
    history_table.add_column("#", style="cyan", width=4)
    history_table.add_column("Token", style="green", width=15)
    history_table.add_column("Probability", style="magenta", width=15)
    history_table.add_column("Current Text", style="yellow", width=60)
    
    for step in range(max_new_tokens):
        if not auto_mode:
            console.print(f"\n[bold cyan]Step {step+1}/{max_new_tokens}[/bold cyan] - Press Enter to generate next token...")
            user_input = input()
            if user_input.lower() == 'auto':
                auto_mode = True
                console.print("[magenta]Switching to automatic mode...[/magenta]")
                
        # Show progress only in auto mode
        if auto_mode and step == 0:
            console.print("\n[cyan]Generating tokens...[/cyan]")
        
        # Crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        logits = logits[:, -1, :]
        
        # Convert logits to probabilities with softmax
        probs = F.softmax(logits, dim=-1)

        # Calculate entropy as a measure of model uncertainty
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
        
        # Get the top 5 tokens
        top_tokens = torch.topk(logits, 5)
        top_token_ids = top_tokens.indices[0].tolist()
        top_token_texts = [tokenizer.decode([tid]) for tid in top_token_ids]
        top_token_probs = [probs[0, tid].item() for tid in top_token_ids]
        
        # Get the selected token (highest probability)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        selected_prob = probs[0, idx_next.item()].item()
        next_token_text = tokenizer.decode([idx_next.item()])
        
        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Get current full text
        current_text = tokenizer.decode(idx.squeeze(0).tolist())
        
        # Add token to history table
        history_table.add_row(
            f"{step+1}",
            f"\"{next_token_text}\"",
            f"{selected_prob:.4%}",
            f"\"{current_text}\""
        )
        
        # Clear screen between tokens in interactive mode
        if not auto_mode and step > 0:
            os.system('cls' if os.name=='nt' else 'clear')
            console.print(Panel.fit(
                f"🤖 [bold blue]GPT Token Generation Process[/bold blue] - {model_status}",
                subtitle=f"Step {step+1}/{max_new_tokens}",
                width=95,
                border_style="cyan"
            ))
            console.print("\n[bold green]Current Context:[/bold green]")
            console.print(Panel(f"[yellow]\"{current_text}\"[/yellow]", width=95))
            console.print(history_table)
        
        # Create token selection visualization
        console.print("\n[bold white on blue]Token Selection[/bold white on blue]")
        
        # Show confidence indicator
        if is_trained:
            confidence_desc = "[green]HIGH[/green]" if selected_prob > 0.5 else "[yellow]MEDIUM[/yellow]" if selected_prob > 0.1 else "[red]LOW[/red]"
        else:
            confidence_desc = "[red]RANDOM[/red]" if entropy > 8 else "[yellow]UNCERTAIN[/yellow]"
            
        console.print(f"Model confidence: {confidence_desc} (Entropy: {entropy:.2f} bits)")

        # Show token options
        token_table = Table(show_header=True, width=95)
        token_table.add_column("", style="cyan", width=4)
        token_table.add_column("Token", style="green", width=15)
        token_table.add_column("Probability", style="magenta", width=15)
        token_table.add_column("Relative Likelihood", style="yellow", width=60)
        
        # Calculate maximum bar width
        max_bar_width = 50
        
        for i, (tid, ttext, tprob) in enumerate(zip(top_token_ids, top_token_texts, top_token_probs)):
            # Create a simple visual bar representing relative probability
            bar_width = int(max_bar_width * (tprob / top_token_probs[0]))
            bar = "█" * bar_width
            
            # Mark the selected token - now with a proper check mark for the top token
            if i == 0:
                marker = "✓"
                token_text = f"[bold green]\"{ttext}\"[/bold green]"
            else:
                marker = " "
                token_text = f"\"{ttext}\""
            
            token_table.add_row(
                marker,
                token_text,
                f"{tprob:.4%}",
                bar
            )
        
        console.print(token_table)
        
        # Show current progress
        if not auto_mode:
            console.print(f"\nProgress: {step+1}/{max_new_tokens} tokens generated")
            if step < max_new_tokens - 1:
                input("\nPress Enter to continue to the next token...")
    
    console.print("\n[bold blue]Generation Complete![/bold blue]")
    
    # Make sure to display the final state of the history table
    if not auto_mode:
        os.system('cls' if os.name=='nt' else 'clear')
        console.print(Panel.fit(
            f"🤖 [bold blue]GPT Token Generation Process[/bold blue] - {model_status}",
            subtitle="Generation Complete",
            width=95,
            border_style="cyan"
        ))
        console.print("\n[bold green]Current Context:[/bold green]")
        current_text = tokenizer.decode(idx.squeeze(0).tolist())
        console.print(Panel(f"[yellow]\"{current_text}\"[/yellow]", width=95))
        console.print(history_table)
    
    console.print(Panel(
        f"[bold green]Final Generated Text:[/bold green]\n[yellow]\"{current_text}\"[/yellow]",
        title="🎉 Result",
        border_style="green",
        width=95
    ))
    
    return idx

def main():
    console = Console(width=100)
    console.print("[bold blue]GPT Model Text Generation Demo[/bold blue]", justify="center")
    console.print("=" * 95, justify="center")
    
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }
    
    # Display model configuration
    config_table = Table(title="Model Configuration", width=95)
    config_table.add_column("Parameter", style="cyan", width=20)
    config_table.add_column("Value", style="magenta", width=20)
    
    for key, value in GPT_CONFIG_124M.items():
        config_table.add_row(key, str(value))
    
    console.print(config_table)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()  # disable dropout
    
    # Display training status
    is_trained = False  # Set to True after training the model
    training_status = "[bold green]TRAINED MODEL[/bold green]" if is_trained else "[bold red]UNTRAINED MODEL[/bold red]"
    console.print(f"\nModel Status: {training_status}")
    
    if not is_trained:
        console.print("[yellow]Note: The untrained model will select tokens randomly.[/yellow]")
    
    start_context = "Hello, I am"
    
    # Show input context
    console.print(Panel.fit(
        f"Starting with: [yellow]\"{start_context}\"[/yellow]",
        title="Context",
        border_style="blue",
        width=95
    ))

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # Ask if the user wants interactive mode
    interactive = input("\nStep through token generation one by one? (y/n) ").lower().startswith('y')
    
    # If we're starting in interactive mode, clear the screen first
    if interactive:
        os.system('cls' if os.name=='nt' else 'clear')
    
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"],
        interactive=interactive,
        is_trained=is_trained
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)


if __name__ == "__main__":
    main()
