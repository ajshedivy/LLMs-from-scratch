import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich import print as rprint
from rich.measure import Measurement
from rich.highlighter import ReprHighlighter
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, ProgressColumn
from rich.syntax import Syntax
from rich.live import Live
from rich.box import ROUNDED, SIMPLE
from rich.text import Text
from rich.align import Align
import time
import math
from llms_from_scratch import MultiHeadAttention


class MultiheadAttentionViz(MultiHeadAttention):
    """Extended version of MultiHeadAttention that stores attention weights for visualization"""
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__(d_in, d_out, context_length, dropout, num_heads, qkv_bias)
        # Will store attention weights and intermediate values for visualization
        self.attn_weights = None
        self.queries = None
        self.keys = None
        self.values = None
        self.attn_scores = None
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Store input for visualization
        self.input = x.detach()

        # Linear projections
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape for multi-head attention
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Store projected values for visualization
        self.keys = keys.detach()
        self.queries = queries.detach()
        self.values = values.detach()

        # Transpose for attention calculation
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)
        
        # Store attention scores before masking
        self.attn_scores_raw = attn_scores.detach()

        # Apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # Store masked attention scores
        self.attn_scores = attn_scores.detach()

        # Calculate and store attention weights for visualization
        self.attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        # Apply dropout
        attn_weights_dropout = self.dropout(self.attn_weights)

        # Compute context vectors
        context_vec = (attn_weights_dropout @ values).transpose(1, 2)

        # Combine heads
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        # Final projection
        output = self.out_proj(context_vec)
        
        # Store final output
        self.output = output.detach()

        return output


class TransformerHooks:
    """Hook manager for transformer model to collect activations during forward pass"""
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.attention_weights = {}
        self.layer_inputs = {}
        self.layer_outputs = {}
        self.norm_stats = {}
        self.embeddings = {}
        self.head_outputs = {}
        self.setup_hooks()
        
    def setup_hooks(self):
        # Input embeddings hooks
        def token_emb_hook(module, inp, out):
            self.embeddings['token_emb'] = out.detach()
        
        def pos_emb_hook(module, inp, out):
            self.embeddings['pos_emb'] = out.detach()
        
        # Register embedding hooks
        self.hooks.append(self.model.tok_emb.register_forward_hook(token_emb_hook))
        self.hooks.append(self.model.pos_emb.register_forward_hook(pos_emb_hook))
        
        # Output head hook
        def out_head_hook(module, inp, out):
            self.head_outputs['logits'] = out.detach()
            
        self.hooks.append(self.model.out_head.register_forward_hook(out_head_hook))
        
        # Transformer block hooks
        for i, block in enumerate(self.model.trf_blocks):
            # Attention hooks
            def make_attn_hook(layer_idx):
                def hook(module, inp, out):
                    self.activations[f'attn_out_{layer_idx}'] = out.detach()
                return hook
            
            # Attention weight hooks - works with our MultiheadAttentionViz
            def make_attn_weights_hook(layer_idx):
                def hook(module, inp, out):
                    if hasattr(module, 'attn_weights'):
                        self.attention_weights[f'layer_{layer_idx}'] = module.attn_weights.detach()
                        
                        # Also store query, key, value projections if available
                        if hasattr(module, 'queries'):
                            self.activations[f'queries_{layer_idx}'] = module.queries.detach()
                        if hasattr(module, 'keys'):
                            self.activations[f'keys_{layer_idx}'] = module.keys.detach()
                        if hasattr(module, 'values'):
                            self.activations[f'values_{layer_idx}'] = module.values.detach()
                        if hasattr(module, 'attn_scores'):
                            self.activations[f'attn_scores_{layer_idx}'] = module.attn_scores.detach()
                return hook
            
            # Layer norm hooks
            def make_norm1_hook(layer_idx):
                def hook(module, inp, out):
                    if inp and len(inp) > 0:
                        self.norm_stats[f'norm1_layer_{layer_idx}_input'] = inp[0].detach()
                        self.norm_stats[f'norm1_layer_{layer_idx}_mean'] = inp[0].mean(dim=-1).detach()
                        self.norm_stats[f'norm1_layer_{layer_idx}_std'] = inp[0].std(dim=-1).detach()
                        self.norm_stats[f'norm1_layer_{layer_idx}_out'] = out.detach()
                return hook
                
            def make_norm2_hook(layer_idx):
                def hook(module, inp, out):
                    if inp and len(inp) > 0:
                        self.norm_stats[f'norm2_layer_{layer_idx}_input'] = inp[0].detach()
                        self.norm_stats[f'norm2_layer_{layer_idx}_mean'] = inp[0].mean(dim=-1).detach()
                        self.norm_stats[f'norm2_layer_{layer_idx}_std'] = inp[0].std(dim=-1).detach()
                        self.norm_stats[f'norm2_layer_{layer_idx}_out'] = out.detach()
                return hook
            
            # FFN hooks
            def make_ffn_hook(layer_idx):
                def hook(module, inp, out):
                    self.activations[f'ffn_out_{layer_idx}'] = out.detach()
                return hook
            
            # Layer in/out hooks
            def make_layer_in_hook(layer_idx):
                def hook(module, inp, out):
                    if inp and len(inp) > 0:
                        self.layer_inputs[f'layer_{layer_idx}'] = inp[0].detach()
                return hook
                
            def make_layer_out_hook(layer_idx):
                def hook(module, inp, out):
                    self.layer_outputs[f'layer_{layer_idx}'] = out.detach()
                return hook
                
            # Register hooks for this layer
            self.hooks.append(block.register_forward_hook(make_layer_in_hook(i)))
            self.hooks.append(block.register_forward_hook(make_layer_out_hook(i)))
            self.hooks.append(block.att.register_forward_hook(make_attn_hook(i)))
            self.hooks.append(block.att.register_forward_hook(make_attn_weights_hook(i)))
            self.hooks.append(block.norm1.register_forward_hook(make_norm1_hook(i)))
            self.hooks.append(block.norm2.register_forward_hook(make_norm2_hook(i)))
            self.hooks.append(block.ff.register_forward_hook(make_ffn_hook(i)))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def clear_activations(self):
        self.activations = {}
        self.attention_weights = {}
        self.layer_inputs = {}
        self.layer_outputs = {}
        self.norm_stats = {}
        self.embeddings = {}
        self.head_outputs = {}


def generate_text_extended(model, idx, max_new_tokens, context_size, interactive=True):
    """
    Generate text with detailed visualization of the transformer's inner workings
    
    Args:
        model: GPT model instance
        idx: tensor of token ids (shape: [batch_size, seq_len])
        max_new_tokens: number of new tokens to generate
        context_size: maximum context size the model can handle
        interactive: whether to pause after each token generation
        
    Returns:
        tensor of token ids including generated tokens
    """
    console = Console(width=120, highlight=False)
    
    # Title display
    console.print(Panel.fit(
        "🔍 [bold blue]GPT Model Architecture Visualization[/bold blue]",
        subtitle="A step-by-step walkthrough of token generation",
        width=110,
        border_style="cyan",
        box=ROUNDED
    ))
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Setup hooks to monitor model internals
    hooks = TransformerHooks(model)
    
    # Display initial context
    console.print("\n[bold green]Starting Context:[/bold green]")
    initial_text = tokenizer.decode(idx.squeeze(0).tolist())
    console.print(Panel(f"[yellow]\"{initial_text}\"[/yellow]", width=110))
    
    # Show model architecture overview
    console.print("\n[bold white on blue]Model Architecture Overview[/bold white on blue]")
    
    arch_table = Table(title="GPT Model Components", width=110)
    arch_table.add_column("Component", style="cyan", width=30)
    arch_table.add_column("Description", style="white", width=80)
    
    arch_table.add_row("Token Embeddings", "Converts token IDs to vectors in embedding space")
    arch_table.add_row("+ Position Embeddings", "Adds positional information to token embeddings")
    arch_table.add_row("Transformer Blocks (x12)", "Process token sequences through attention and feed-forward layers")
    arch_table.add_row("Layer Normalization", "Stabilizes activations throughout the network")
    arch_table.add_row("Output Head", "Projects from embedding dimension to vocabulary probabilities")
    
    console.print(arch_table)
    
    if interactive:
        console.print("\n[bold magenta]Interactive Mode:[/bold magenta] Press Enter to step through each component")
        input("\nPress Enter to begin the visual walkthrough...")
    
    # Process token by token
    for step in range(max_new_tokens):
        # Clear the screen at the start of each token generation
        if interactive:
            os.system('cls' if os.name=='nt' else 'clear')
            console.print(Panel.fit(
                f"🔍 [bold blue]GPT Model Architecture - Token Generation Step {step+1}/{max_new_tokens}[/bold blue]",
                subtitle="Visualizing the transformer processing pipeline",
                width=110,
                border_style="cyan",
                box=ROUNDED
            ))
        
        # Show current context
        current_text = tokenizer.decode(idx.squeeze(0).tolist())
        console.print("\n[bold green]Current Context:[/bold green]")
        console.print(Panel(f"[yellow]\"{current_text}\"[/yellow]", width=110))
        
        # Clear previous activations
        hooks.clear_activations()
        
        # Crop context if needed
        idx_cond = idx[:, -context_size:]
        
        # Forward pass with hooks collecting data
        with torch.no_grad():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Processing through transformer...", total=100)
                progress.update(task, advance=50)
                logits = model(idx_cond)
                progress.update(task, advance=50)
        
        # 1. Visualize the Token Embedding Process
        console.print("\n[bold white on dark_blue]1. Token Embedding Layer[/bold white on dark_blue]")
        
        if 'token_emb' in hooks.embeddings and 'pos_emb' in hooks.embeddings:
            # Get embedding data
            tok_emb = hooks.embeddings['token_emb']
            pos_emb = hooks.embeddings['pos_emb']
            
            # Create a visual diagram of the embedding process
            console.print(f"[cyan]◉ Input Token IDs:[/cyan] {idx_cond.squeeze(0).tolist()}")
            
            # Display the token embedding visually
            token_table = Table(title="Token Embedding Process", width=110, box=SIMPLE)
            token_table.add_column("Step", style="cyan", width=25)
            token_table.add_column("Visualization", style="white", width=85)
            
            # Show token embedding lookup
            token_ids = idx_cond.squeeze(0).tolist()
            token_texts = [tokenizer.decode([tid]) for tid in token_ids]
            
            token_row = ""
            for i, (tid, txt) in enumerate(zip(token_ids, token_texts)):
                token_row += f"[bold magenta]{tid}[/bold magenta]:[blue]\"{txt}\"[/blue] → "
            token_row = token_row[:-3]  # Remove the last arrow
            
            token_table.add_row("Token ID → Embedding Lookup", token_row)
            
            # Show token embedding shapes
            shape_info = f"Token Embeddings: {list(tok_emb.shape)} | Position Embeddings: {list(pos_emb.shape)}"
            token_table.add_row("Embedding Shapes", shape_info)
            
            # Visual representation of embedding
            emb_visual = "[bright_cyan]"
            for i in range(min(10, tok_emb.size(2))):  # Show first few dimensions
                emb_visual += "⬚ "
            emb_visual += "... " + str(tok_emb.size(2)) + " dimensions[/bright_cyan]"
            token_table.add_row("Token Embedding Vector", emb_visual)
            
            # Show position embedding
            pos_visual = "[bright_green]"
            for i in range(min(10, pos_emb.size(1))):  # Show first few dimensions
                pos_visual += "⬚ "
            pos_visual += "... " + str(pos_emb.size(1)) + " dimensions[/bright_green]"
            token_table.add_row("Position Embedding Vector", pos_visual)
            
            # Show combined embedding
            combined_visual = "[bright_yellow]"
            for i in range(min(10, tok_emb.size(2))):  # Show first few dimensions
                combined_visual += "⬚ "
            combined_visual += "... " + str(tok_emb.size(2)) + " dimensions[/bright_yellow]"
            token_table.add_row("Combined Embedding (token + position)", combined_visual)
            
            console.print(token_table)
            
            # Token embedding stats table
            emb_stats = Table(width=110)
            emb_stats.add_column("Embedding Type", style="cyan")
            emb_stats.add_column("Mean", style="green")
            emb_stats.add_column("Std Dev", style="yellow")
            emb_stats.add_column("Min", style="red")
            emb_stats.add_column("Max", style="blue")
            
            # Add token embedding stats
            emb_stats.add_row(
                "Token Embeddings",
                f"{tok_emb.mean().item():.4f}",
                f"{tok_emb.std().item():.4f}",
                f"{tok_emb.min().item():.4f}",
                f"{tok_emb.max().item():.4f}"
            )
            
            # Add positional embedding stats
            emb_stats.add_row(
                "Position Embeddings",
                f"{pos_emb.mean().item():.4f}",
                f"{pos_emb.std().item():.4f}",
                f"{pos_emb.min().item():.4f}",
                f"{pos_emb.max().item():.4f}"
            )
            
            console.print(emb_stats)
            
            # Explanation of embedding process
            console.print(Panel(
                "[white]The input token IDs are first converted to embedding vectors (768 dimensions each). "
                "Position embeddings are added to provide sequential information. The combined embeddings "
                "are then passed through dropout and fed to the transformer blocks.[/white]",
                title="Embedding Process Explained",
                width=110
            ))
            
            if interactive:
                input("\nPress Enter to continue to transformer blocks...")

        # 2. Visualize Selected Transformer Blocks
        console.print("\n[bold white on dark_blue]2. Transformer Block Processing[/bold white on dark_blue]")
        
        # Get number of layers
        num_layers = len([k for k in hooks.layer_inputs.keys() if k.startswith('layer_')])
        
        # Select which layers to show in detail
        layers_to_show = [0, num_layers//2, num_layers-1]  # First, middle, last
        
        # Create overview diagram of transformer blocks
        console.print("Full model contains", num_layers, "transformer blocks. Showing selected blocks:")
        
        for layer_idx in layers_to_show:
            console.print(f"\n[bold]Transformer Block {layer_idx+1}/{num_layers}[/bold]")
            
            # Create a visual representation of the transformer block
            block_table = Table(width=110, box=SIMPLE)
            block_table.add_column("Component", style="cyan", width=25)
            block_table.add_column("Processing Flow", style="white", width=85)
            
            # Show layer normalization input
            if f'norm1_layer_{layer_idx}_input' in hooks.norm_stats:
                norm1_in = hooks.norm_stats[f'norm1_layer_{layer_idx}_input']
                block_table.add_row("Input", f"Tensor shape: {list(norm1_in.shape)}")
            
                # Show LayerNorm1 operation
                if f'norm1_layer_{layer_idx}_out' in hooks.norm_stats:
                    norm1_out = hooks.norm_stats[f'norm1_layer_{layer_idx}_out']
                    
                    block_table.add_row(
                        "Layer Normalization 1", 
                        "[bright_cyan]Normalizing input → μ=0, σ=1[/bright_cyan]"
                    )
                    
                    # Show attention operation
                    block_table.add_row(
                        "Self-Attention", 
                        "[yellow]Q, K, V projections → Attention scores → Context vectors[/yellow]"
                    )
                    
                    block_table.add_row(
                        "Residual Connection", 
                        "[green]Output = LayerNorm(Self-Attention(Input)) + Input[/green]"
                    )
                    
                    # Show LayerNorm2 operation
                    block_table.add_row(
                        "Layer Normalization 2", 
                        "[bright_cyan]Normalizing attention output → μ=0, σ=1[/bright_cyan]"
                    )
                    
                    # Show FFN operation
                    block_table.add_row(
                        "Feed-Forward Network", 
                        "[magenta]Linear¹(768→3072) → GELU → Linear²(3072→768)[/magenta]"
                    )
                    
                    # Show residual connection
                    block_table.add_row(
                        "Residual Connection", 
                        "[green]Output = LayerNorm(FFN(Attention Output)) + Attention Output[/green]"
                    )
                    
                    # Completed block output
                    if f'layer_{layer_idx}' in hooks.layer_outputs:
                        layer_out = hooks.layer_outputs[f'layer_{layer_idx}']
                        block_table.add_row(
                            "Block Output",
                            f"Tensor shape: {list(layer_out.shape)}"
                        )
            
            console.print(block_table)
            
            # 3. Visualize attention mechanism in detail for this layer
            console.print(f"\n[bold]3. Masked Multi-Head Attention (Block {layer_idx+1})[/bold]")
            
            # Show attention weights as a visual pattern
            if f'layer_{layer_idx}' in hooks.attention_weights:
                attn_weights = hooks.attention_weights[f'layer_{layer_idx}']
                
                # Create a more detailed visualization of attention mechanism
                attention_table = Table(title=f"Self-Attention Mechanism (Block {layer_idx+1})", width=110)
                attention_table.add_column("Component", style="cyan", width=25)
                attention_table.add_column("Description", style="white", width=85)
                
                # Explain the attention process
                attention_table.add_row(
                    "Mechanism", 
                    "Multi-head attention allows the model to focus on different parts of the input sequence"
                )
                
                attention_table.add_row(
                    "Number of heads", 
                    f"{attn_weights.shape[1]} independent attention mechanisms in parallel"
                )
                
                attention_table.add_row(
                    "Causal masking",
                    "Future tokens are masked to ensure the model only attends to previous tokens"
                )
                
                console.print(attention_table)
                
                # Show attention heatmap visualization for selected heads
                if attn_weights.shape[1] > 0:  # If we have attention heads
                    num_heads = attn_weights.shape[1]
                    seq_len = attn_weights.shape[2]
                    
                    # Select heads to visualize
                    heads_to_show = min(3, num_heads)
                    console.print(f"\n[bold]Attention Patterns (Showing {heads_to_show} of {num_heads} heads):[/bold]")
                    
                    for head_idx in range(heads_to_show):
                        # Get attention weights for this head
                        head_weights = attn_weights[0, head_idx, :, :].cpu().numpy()
                        
                        # Create a simple text-based heatmap (last few tokens only for readability)
                        display_tokens = min(10, seq_len)
                        if seq_len > display_tokens:
                            head_weights = head_weights[-display_tokens:, -display_tokens:]
                            
                        console.print(f"[bold]Head {head_idx+1} Attention Pattern:[/bold]")
                        
                        # Create a colored heatmap with Unicode blocks
                        heatmap_lines = []
                        for row_idx in range(head_weights.shape[0]):
                            row_chars = []
                            for col_idx in range(head_weights.shape[1]):
                                weight = head_weights[row_idx, col_idx]
                                # Use different colors and intensities
                                if weight > 0.75:    char = f"[bright_yellow on yellow]█[/]"
                                elif weight > 0.5:   char = f"[yellow]█[/]"
                                elif weight > 0.25:  char = f"[yellow]▓[/]"
                                elif weight > 0.1:   char = f"[yellow]▒[/]"
                                elif weight > 0.01:  char = f"[dark_yellow]░[/]"
                                else:               char = " "
                                row_chars.append(char)
                            heatmap_lines.append("".join(row_chars))
                            
                        # Show the heatmap
                        for line in heatmap_lines:
                            console.print(line)
                            
                        # Add a key to interpret the heatmap
                        console.print("[bright_yellow on yellow]█[/]>75% [yellow]█[/]>50% [yellow]▓[/]>25% [yellow]▒[/]>10% [dark_yellow]░[/]>1%")
                        console.print()
            
            if interactive and layer_idx != layers_to_show[-1]:
                input("\nPress Enter to view the next transformer block...")
                
        # 4. After all transformer blocks, show the final layer norm and output projection
        console.print("\n[bold white on dark_blue]4. Final Layer Normalization & Output Projection[/bold white on dark_blue]")
        
        # Create a table to show the final processing steps
        final_table = Table(width=110)
        final_table.add_column("Component", style="cyan", width=25)
        final_table.add_column("Processing", style="white", width=85)
        
        # Last layer output becomes input to final layer norm
        if f'layer_{num_layers-1}' in hooks.layer_outputs:
            last_layer_out = hooks.layer_outputs[f'layer_{num_layers-1}']
            
            final_table.add_row(
                "Final State",
                f"Output from last transformer block: {list(last_layer_out.shape)}"
            )
            
            final_table.add_row(
                "Final Layer Norm",
                "Normalizing the final transformer block output"
            )
            
            # Output projection
            if 'logits' in hooks.head_outputs:
                logits_tensor = hooks.head_outputs['logits']
                
                final_table.add_row(
                    "Output Projection",
                    f"Linear projection from embedding dimension ({last_layer_out.shape[-1]}) to vocabulary size ({logits_tensor.shape[-1]})"
                )
        
        console.print(final_table)
        
        # 5. Visualize the final embedding vector (truncated) and token selection process
        console.print("\n[bold white on dark_blue]5. Token Selection Process[/bold white on dark_blue]")
        
        # Focus on the last token position only for next token prediction
        logits = logits[:, -1, :]
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get the top 5 tokens
        top_n = 5
        top_tokens = torch.topk(logits, top_n)
        top_token_ids = top_tokens.indices[0].tolist()
        top_token_texts = [tokenizer.decode([tid]) for tid in top_token_ids]
        top_token_probs = [probs[0, tid].item() for tid in top_token_ids]
        
        # Select the next token (argmax)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        next_token_text = tokenizer.decode([idx_next.item()])
        
        # Table for token selection
        token_selection = Table(title="[bold]Token Selection Process[/bold]", width=110)
        token_selection.add_column("Rank", style="cyan", width=8)
        token_selection.add_column("Token", style="green", width=20)
        token_selection.add_column("Logit", style="magenta", width=12)
        token_selection.add_column("Probability", style="yellow", width=15)
        token_selection.add_column("Relative Likelihood", style="blue", width=50)
        
        # Add top token candidates
        max_bar_width = 40
        
        for i, (tid, ttext, tprob) in enumerate(zip(top_token_ids, top_token_texts, top_token_probs)):
            # Create visual bar
            bar_width = int(max_bar_width * (tprob / top_token_probs[0]))
            bar = "█" * bar_width
            
            # Format logit value
            logit_val = logits[0, tid].item()
            
            # Show the selected token
            if i == 0:
                marker = "✓"
                token_text = f"[bold green]\"{ttext}\"[/bold green]"
                token_selection.add_row(f"{marker} {i+1}", token_text, f"{logit_val:.4f}", f"{tprob:.4%}", bar)
            else:
                token_selection.add_row(f"  {i+1}", f"\"{ttext}\"", f"{logit_val:.4f}", f"{tprob:.4%}", bar)
        
        console.print(token_selection)
        
        # Add the selected token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Show updated text with the new token
        updated_text = tokenizer.decode(idx.squeeze(0).tolist())
        console.print(f"\n[bold green]Text with new token:[/bold green]")
        console.print(Panel(f"[yellow]\"{updated_text}\"[/yellow]", width=110))
        
        # Wait for user input in interactive mode
        if interactive and step < max_new_tokens - 1:
            input("\nPress Enter to generate the next token...")
    
    # Remove hooks when done
    hooks.remove_hooks()
    
    # Show final output
    console.print("\n[bold white on blue]Generation Complete![/bold white on blue]")
    final_text = tokenizer.decode(idx.squeeze(0).tolist())
    console.print(Panel(
        f"[bold green]Final Generated Text:[/bold green]\n\n[yellow]\"{final_text}\"[/yellow]",
        title="🎉 Result",
        border_style="green",
        width=110
    ))
    
    return idx


def create_visualizable_model(model):
    """Replace attention modules with visualization-enabled versions"""
    for block in model.trf_blocks:
        # Create new attention module with same parameters
        old_att = block.att
        new_att = MultiheadAttentionViz(
            d_in=old_att.W_query.in_features,
            d_out=old_att.W_query.out_features,
            context_length=old_att.mask.shape[0],
            dropout=old_att.dropout.p,
            num_heads=old_att.num_heads,
            qkv_bias=hasattr(old_att.W_query, 'bias') and old_att.W_query.bias is not None
        )
        
        # Copy weights
        with torch.no_grad():
            new_att.W_query.weight.copy_(old_att.W_query.weight)
            new_att.W_key.weight.copy_(old_att.W_key.weight)
            new_att.W_value.weight.copy_(old_att.W_value.weight)
            new_att.out_proj.weight.copy_(old_att.out_proj.weight)
            
            if hasattr(old_att.out_proj, 'bias') and old_att.out_proj.bias is not None:
                new_att.out_proj.bias.copy_(old_att.out_proj.bias)
        
        # Replace module
        block.att = new_att
    
    return model


def main_extended_visualization(model, start_text="Hello, I am", max_tokens=5, interactive=True):
    """Main function to demonstrate the extended visualization"""
    console = Console(width=120)
    console.print("[bold blue]GPT Model Internals Visualization Demo[/bold blue]", justify="center")
    console.print("=" * 110, justify="center")
    
    # Prepare the model for visualization
    model = create_visualizable_model(model)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    context_size = model.pos_emb.weight.shape[0]  # Extract context size from position embeddings
    
    model.eval()  # Disable dropout for consistent results
    
    console.print(f"\n[bold green]Starting text:[/bold green] \"{start_text}\"")
    console.print(f"[bold]Will generate {max_tokens} new tokens[/bold]")
    
    if interactive:
        input("\nPress Enter to begin the visualization...")
    
    out = generate_text_extended(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=max_tokens,
        context_size=context_size,
        interactive=interactive
    )
    
    return out


if __name__ == "__main__":
    # Example standalone usage
    from llms_from_scratch.ch04 import GPTModel  # Import your GPT model
    
    # Configuration for a small GPT model
    GPT_CONFIG_TINY = {
        "vocab_size": 50257,     # GPT-2 vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }
    
    # Create a small model for demonstration
    torch.manual_seed(123)  # For reproducibility
    model = GPTModel(GPT_CONFIG_TINY)
    
    # Run the visualization
    main_extended_visualization(
        model=model,
        start_text="Hello, I am",
        max_tokens=5,
        interactive=True
    )