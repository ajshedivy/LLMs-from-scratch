# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import matplotlib.pyplot as plt
import os
import torch
import urllib.request
import tiktoken
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich.columns import Columns
from rich.align import Align


# Import from local files
from llms_from_scratch.ch04 import GPTModel, generate_text_simple
from llms_from_scratch.ch02 import create_dataloader_v1

# Initialize Rich console
console = Console()


def print_separator(char="=", length=80):
    """Print a visual separator line for better output organization."""
    console.print(char * length, style="bright_blue")


def print_header(title):
    """Print a formatted header with visual emphasis."""
    header_panel = Panel(
        Text(title.upper(), style="bold bright_white", justify="center"),
        style="bright_blue",
        padding=(1, 2)
    )
    console.print(header_panel)


def display_generated_text(text, epoch, start_context):
    """Display generated text with Rich formatting."""
    # Create the main content
    content = Text()
    content.append(f"Starting context: ", style="bold cyan")
    content.append(f"'{start_context}'\n\n", style="italic bright_white")
    
    # Wrap and style the generated text
    wrapped_text = text.replace("\n", " ")
    content.append("Generated text:\n", style="bold yellow")
    content.append(wrapped_text, style="bright_white")
    
    # Create panel with generated text
    panel = Panel(
        content,
        title=f"🤖 Model Output After Epoch {epoch}",
        title_align="center",
        border_style="bright_green",
        padding=(1, 2)
    )
    console.print(panel)


def display_training_insights(train_losses, val_losses, epoch, train_loss, val_loss):
    """Display educational insights with Rich formatting including loss trends."""
    insights = []
    
    # Analyze trends if we have enough history
    if len(train_losses) >= 2:
        prev_train_loss = train_losses[-2]
        prev_val_loss = val_losses[-2]
        
        # Training loss trend
        train_change = prev_train_loss - train_loss
        if train_change > 0.01:
            train_trend = "📉 decreasing"
            train_emoji = "✅"
        elif train_change < -0.01:
            train_trend = "📈 increasing"
            train_emoji = "⚠️"
        else:
            train_trend = "➡️ stable"
            train_emoji = "📊"
        
        # Validation loss trend
        val_change = prev_val_loss - val_loss
        if val_change > 0.01:
            val_trend = "📉 decreasing"
            val_emoji = "✅"
        elif val_change < -0.01:
            val_trend = "📈 increasing"
            val_emoji = "⚠️"
        else:
            val_trend = "➡️ stable"
            val_emoji = "📊"
        
        # Add trend insights
        insights.append(f"{train_emoji} Training loss: {train_trend} ({train_change:+.4f})")
        insights.append(f"{val_emoji} Validation loss: {val_trend} ({val_change:+.4f})")
        
        # Overall assessment
        if train_change > 0 and val_change > 0:
            insights.append("🎯 Both losses improving - excellent progress!")
        elif train_change > 0 and val_change <= 0:
            insights.append("⚡ Training improving but validation not - watch for overfitting")
        elif train_change <= 0 and val_change > 0:
            insights.append("🤔 Validation improving but training not - unusual pattern")
        elif train_change < 0 and val_change < 0:
            insights.append("🔄 Both losses increasing - may need learning rate adjustment")
        else:
            insights.append("📊 Losses are stabilizing")
    else:
        # First epoch - just show current values
        insights.append(f"📊 Initial training loss: {train_loss:.4f}")
        insights.append(f"📊 Initial validation loss: {val_loss:.4f}")
    
    # Additional insights about overfitting/underfitting
    # if train_loss < val_loss:
    #     gap = val_loss - train_loss
    #     if gap > 0.5:
    #         insights.append("⚡ Large train/val gap - significant overfitting")
    #     elif gap > 0.1:
    #         insights.append("📚 Moderate train/val gap - some overfitting")
    #     else:
    #         insights.append("📚 Model fits training data well")
    # elif train_loss > val_loss * 1.05:
    #     insights.append("🔍 Training loss higher than validation - check data")
    
    # Create insight panels
    insight_text = "\n".join(f"• {insight}" for insight in insights)
    panel = Panel(
        insight_text,
        title=f"📊 Training Insights - Epoch {epoch}",
        title_align="center",
        border_style="bright_yellow",
        padding=(0, 1)
    )
    console.print(panel)


def wait_for_user_input(epoch, total_epochs):
    """Pause training and wait for user input with Rich formatting."""
    if epoch < total_epochs:
        # Create instruction panel
        instruction_text = Text()
        instruction_text.append("Take time to observe the loss reduction and text quality improvement.\n", style="italic")
        instruction_text.append("Press ", style="white")
        instruction_text.append("Enter", style="bold green")
        instruction_text.append(" to continue to the next epoch, or ", style="white")
        instruction_text.append("'q' + Enter", style="bold red")
        instruction_text.append(" to quit training.", style="white")
        
        panel = Panel(
            instruction_text,
            title=f"📚 Training Paused After Epoch {epoch}",
            title_align="center",
            border_style="bright_cyan",
            padding=(1, 2)
        )
        console.print(panel)
        
        console.print("→ ", style="bold bright_cyan", end="")
        user_input = input().strip().lower()
        if user_input == 'q':
            console.print("🛑 Training stopped by user.", style="bold red")
            return False
        return True
    return True


def show_progress_bar(current, total, width=50):
    """Display a simple ASCII progress bar."""
    filled = int(width * current // total)
    bar = '█' * filled + '░' * (width - filled)
    percent = 100 * current / total
    return f"[{bar}] {percent:.1f}%"


def display_training_config(num_epochs, device, start_context, eval_freq):
    """Display training configuration with Rich formatting."""
    config_data = [
        ["Epochs", str(num_epochs)],
        ["Device", str(device)],
        ["Starting Context", f"'{start_context}'"],
        ["Eval Frequency", f"every {eval_freq} steps"]
    ]
    
    config_table = Table(show_header=False, box=None, padding=(0, 1))
    config_table.add_column("Setting", style="bold cyan")
    config_table.add_column("Value", style="bright_white")
    
    for setting, value in config_data:
        config_table.add_row(f"• {setting}:", value)
    
    panel = Panel(
        config_table,
        title="🎯 Training Configuration",
        title_align="center",
        border_style="bright_blue",
        padding=(0, 1)
    )
    console.print(panel)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context, epoch=None):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        
        if epoch is not None:
            display_generated_text(decoded_text, epoch, start_context)
        else:
            print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()
    return decoded_text


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    
    # Store all training steps for final table
    training_history = []

    # Print welcome header and training table
    print_header("Interactive GPT Training Experience")
    display_training_config(num_epochs, device, start_context, eval_freq)
    
    console.print("\n📊 Training will evaluate every {} steps within each epoch".format(eval_freq), style="bold cyan")
    console.print("🎯 After each epoch: text generation → pause → continue", style="bold cyan")

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_start_time = time.time()
        
        # Show progress indicator with Rich styling
        progress = show_progress_bar(epoch, num_epochs)
        console.print(f"\n🔄 Starting Epoch {epoch+1}/{num_epochs} {progress}", style="bold bright_blue")
        
        # Track steps within this epoch
        epoch_steps = []

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Evaluation step based on eval_freq
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                # Print step metrics
                console.print(
                    f"  Step {global_step:06d}: Train loss {train_loss:.4f}, Val loss {val_loss:.4f}, Tokens: {tokens_seen:,}",
                    style="dim white"
                )
                
                # Store step data
                step_data = {
                    'epoch': epoch + 1,
                    'step': global_step,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'tokens': tokens_seen,
                    'type': 'step'
                }
                training_history.append(step_data)
                epoch_steps.append(step_data)

        # Calculate final losses for this epoch
        train_loss, val_loss = evaluate_model(
            model, train_loader, val_loader, device, eval_iter)
        
        # Only append if we didn't already add it in the eval step
        if global_step % eval_freq != 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)

        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Store epoch summary
        epoch_data = {
            'epoch': epoch + 1,
            'step': global_step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'tokens': tokens_seen,
            'duration': epoch_duration,
            'type': 'epoch_end'
        }
        training_history.append(epoch_data)
        
        # Print epoch summary
        console.print(f"\n✅ Epoch {epoch+1} Complete:", style="bold green")
        console.print(f"   Final - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Duration: {epoch_duration:.2f}s", style="green")

        # Generate and display sample text
        generate_and_print_sample(
            model, tokenizer, device, start_context, epoch=epoch+1
        )

        # Add educational insights
        display_training_insights(train_losses, val_losses, epoch+1, train_loss, val_loss)

        # Wait for user input before next epoch (except for last epoch)
        if epoch < num_epochs - 1:
            if not wait_for_user_input(epoch+1, num_epochs):
                console.print(f"\n📈 Training stopped early after {epoch+1} epochs.", style="bold yellow")
                break
        else:
            console.print(f"\n🎉 Training completed! All {num_epochs} epochs finished.", style="bold green")

    # Display final comprehensive training table
    display_training_summary_table(training_history)

    return train_losses, val_losses, track_tokens_seen


def display_training_summary_table(training_history):
    """Display a comprehensive table of all training steps and epochs."""
    console.print("\n")
    
    # Create final summary table
    summary_table = Table(
        title="📈 Complete Training Summary", 
        show_header=True, 
        header_style="bold bright_magenta",
        title_style="bold bright_white"
    )
    
    summary_table.add_column("Type", style="cyan", justify="center", width=12)
    summary_table.add_column("Epoch", style="blue", justify="center", width=8)
    summary_table.add_column("Step", style="white", justify="center", width=8)
    summary_table.add_column("Train Loss", style="green", justify="center", width=12)
    summary_table.add_column("Val Loss", style="yellow", justify="center", width=12)
    summary_table.add_column("Tokens", style="magenta", justify="center", width=12)
    summary_table.add_column("Duration", style="red", justify="center", width=10)
    
    for entry in training_history:
        entry_type = "🔄 Step" if entry['type'] == 'step' else "✅ Epoch"
        duration = f"{entry.get('duration', 0):.2f}s" if entry['type'] == 'epoch_end' else "-"
        
        summary_table.add_row(
            entry_type,
            str(entry['epoch']),
            str(entry['step']),
            f"{entry['train_loss']:.4f}",
            f"{entry['val_loss']:.4f}",
            f"{entry['tokens']:,}",
            duration
        )
    
    # Display the table in a panel
    final_panel = Panel(
        summary_table,
        title="🎯 Training Complete - Full History",
        title_align="center",
        border_style="bright_green",
        padding=(1, 1)
    )
    console.print(final_panel)
    
    # Display summary statistics
    if training_history:
        epoch_entries = [entry for entry in training_history if entry['type'] == 'epoch_end']
        if epoch_entries:
            initial_loss = epoch_entries[0]['train_loss']
            final_loss = epoch_entries[-1]['train_loss']
            total_improvement = initial_loss - final_loss
            total_tokens = epoch_entries[-1]['tokens']
            total_time = sum(entry.get('duration', 0) for entry in epoch_entries)
            
            stats_text = f"""
• Total Loss Improvement: {total_improvement:.4f}
• Initial Train Loss: {initial_loss:.4f} → Final: {final_loss:.4f}
• Total Tokens Processed: {total_tokens:,}
• Total Training Time: {total_time:.1f} seconds
• Average Time per Epoch: {total_time/len(epoch_entries):.1f} seconds
            """
            
            stats_panel = Panel(
                stats_text.strip(),
                title="📊 Training Statistics",
                title_align="center",
                border_style="bright_blue",
                padding=(0, 1)
            )
            console.print(stats_panel)


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(gpt_config, settings):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    ##############################
    # Download data if necessary
    ##############################

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=2, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

    ###########################
    # After training
    ###########################

    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
