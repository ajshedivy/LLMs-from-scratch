import gensim.downloader
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  – triggers Matplotlib 3-D support
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def load_glove(dim: int = 50):
    """Load GloVe embeddings of the requested dimensionality."""
    name = f"glove-wiki-gigaword-{dim}"
    print(f"➡  Loading pre-trained model: '{name}' …")
    return gensim.downloader.load(name)

def most_similar(model, token: str, topn: int = 10):
    """Return a list of (word, vector) pairs for *token* and its *topn* neighbours."""
    sims = model.most_similar(token, topn=topn)
    words = [token] + [w for w, _ in sims]
    vectors = np.vstack([model[w] for w in words])
    return words, vectors

def reduce_to_3d(vectors: np.ndarray, random_state: int = 42):
    """Project *vectors* to 3-D with PCA."""
    pca = PCA(n_components=3, random_state=random_state)
    reduced = pca.fit_transform(vectors)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"➡  PCA explains {explained:.1f}% of the variance in 3 components.")
    return reduced

def plot_3d_interactive(words, coords):
    """Render an interactive 3-D scatter plot using Plotly."""
    
    # Create the figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1], 
        z=coords[:, 2],
        mode='markers+text',
        marker=dict(
            size=8,
            color=np.arange(len(words)),  # Color by index
            colorscale='Viridis',
            opacity=0.8,
            line=dict(width=2, color='white')
        ),
        text=words,
        textposition="middle center",
        textfont=dict(size=12, color='black'),
        hovertemplate='<b>%{text}</b><br>' +
                     'PC-1: %{x:.3f}<br>' +
                     'PC-2: %{y:.3f}<br>' +
                     'PC-3: %{z:.3f}<br>' +
                     '<extra></extra>',
        name='Word Embeddings'
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Interactive GloVe embeddings for 'tower' and nearest neighbours",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title='PC-1',
            yaxis_title='PC-2',
            zaxis_title='PC-3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    fig.show()

def plot_directions_interactive(words, coords):
    """
    Show interactive directions (unit vectors) from the origin.
    The closer the arrows, the higher the average cosine similarity.
    """
    # 1️⃣  normalise: keep only direction, discard length
    unit = coords / np.linalg.norm(coords, axis=1, keepdims=True)

    # 2️⃣  compute the mean direction (red arrow)
    mean_dir = unit.mean(axis=0)
    mean_dir /= np.linalg.norm(mean_dir)

    # Create the figure
    fig = go.Figure()

    # 3️⃣  translucent unit sphere for context
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.1,
        colorscale='Blues',
        showscale=False,
        hoverinfo='skip',
        name='Unit Sphere'
    ))

    # 4️⃣  arrows for each word (using cones for arrow heads)
    colors = px.colors.qualitative.Set3
    for i, (word, vec) in enumerate(zip(words, unit)):
        color = colors[i % len(colors)]
        
        # Arrow shaft (line from origin to point)
        fig.add_trace(go.Scatter3d(
            x=[0, vec[0]],
            y=[0, vec[1]],
            z=[0, vec[2]],
            mode='lines',
            line=dict(color=color, width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Arrow head (cone)
        fig.add_trace(go.Cone(
            x=[vec[0]], y=[vec[1]], z=[vec[2]],
            u=[vec[0]*0.1], v=[vec[1]*0.1], w=[vec[2]*0.1],
            colorscale=[[0, color], [1, color]],
            showscale=False,
            sizemode='absolute',
            sizeref=0.1,
            anchor='tail',
            hovertemplate=f'<b>{word}</b><br>' +
                         'Direction: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>' +
                         '<extra></extra>',
            name=word
        ))
        
        # Text labels
        fig.add_trace(go.Scatter3d(
            x=[vec[0] * 1.1],
            y=[vec[1] * 1.1], 
            z=[vec[2] * 1.1],
            mode='text',
            text=[word],
            textfont=dict(size=10, color=color),
            showlegend=False,
            hoverinfo='skip'
        ))

    # 5️⃣  mean direction (thicker, red)
    fig.add_trace(go.Scatter3d(
        x=[0, mean_dir[0]],
        y=[0, mean_dir[1]],
        z=[0, mean_dir[2]],
        mode='lines',
        line=dict(color='red', width=8),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Cone(
        x=[mean_dir[0]], y=[mean_dir[1]], z=[mean_dir[2]],
        u=[mean_dir[0]*0.15], v=[mean_dir[1]*0.15], w=[mean_dir[2]*0.15],
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        sizemode='absolute',
        sizeref=0.15,
        anchor='tail',
        hovertemplate='<b>Mean Direction</b><br>' +
                     'Direction: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>' +
                     '<extra></extra>',
        name='Mean Direction'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[mean_dir[0] * 1.1],
        y=[mean_dir[1] * 1.1],
        z=[mean_dir[2] * 1.1],
        mode='text',
        text=['mean'],
        textfont=dict(size=12, color='red'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 6️⃣  Update layout
    fig.update_layout(
        title={
            'text': "Interactive Direction vectors in 3-D (unit sphere)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title='PC-1',
            yaxis_title='PC-2', 
            zaxis_title='PC-3',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    fig.show()

def plot_3d_comparison(words, coords):
    """Create a comparison view with both scatter and direction plots."""
    
    # Normalize vectors for direction plot
    unit = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    mean_dir = unit.mean(axis=0)
    mean_dir /= np.linalg.norm(mean_dir)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Embedding Positions', 'Direction Vectors'),
        horizontal_spacing=0.05
    )
    
    # Left plot: Scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color=np.arange(len(words)),
                colorscale='Viridis',
                opacity=0.8
            ),
            text=words,
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>' +
                         'PC-1: %{x:.3f}<br>' +
                         'PC-2: %{y:.3f}<br>' +
                         'PC-3: %{z:.3f}<br>' +
                         '<extra></extra>',
            name='Positions'
        ),
        row=1, col=1
    )
    
    # Right plot: Direction vectors
    # Unit sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:15j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    
    fig.add_trace(
        go.Surface(
            x=xs, y=ys, z=zs,
            opacity=0.1,
            colorscale='Blues',
            showscale=False,
            hoverinfo='skip'
        ),
        row=1, col=2
    )
    
    # Direction arrows
    colors = px.colors.qualitative.Set3
    for i, (word, vec) in enumerate(zip(words, unit)):
        color = colors[i % len(colors)]
        
        # Arrow lines
        fig.add_trace(
            go.Scatter3d(
                x=[0, vec[0]],
                y=[0, vec[1]],
                z=[0, vec[2]],
                mode='lines+text',
                line=dict(color=color, width=4),
                text=['', word],
                textposition='middle center',
                showlegend=False,
                hovertemplate=f'<b>{word}</b><extra></extra>'
            ),
            row=1, col=2
        )
    
    # Mean direction
    fig.add_trace(
        go.Scatter3d(
            x=[0, mean_dir[0]],
            y=[0, mean_dir[1]],
            z=[0, mean_dir[2]],
            mode='lines+text',
            line=dict(color='red', width=8),
            text=['', 'mean'],
            textfont=dict(color='red', size=12),
            showlegend=False,
            hovertemplate='<b>Mean Direction</b><extra></extra>'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Interactive Word Embedding Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        width=1400,
        height=600,
        margin=dict(l=0, r=0, b=0, t=80)
    )
    
    # Update scene properties for both subplots
    for i in [1, 2]:
        fig.update_scenes(
            xaxis_title='PC-1',
            yaxis_title='PC-2',
            zaxis_title='PC-3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            row=1, col=i
        )
    
    fig.show()
    
    
    
def mathematical_attention_example():
    """Work through attention with real numbers and visualizations"""
    
    import torch
    import numpy as np
    
    print("🔢 Mathematical Attention Example: 'red car'")
    print("=" * 60)
    input("Press Enter to see initial embeddings...")
    
    # Initial embeddings (2D for simplicity)
    red_embedding = torch.tensor([0.8, 0.2])  # High color, low vehicle
    car_embedding = torch.tensor([0.1, 0.9])  # Low color, high vehicle
    
    print(f"\n📊 Initial Embeddings:")
    print(f"red = {red_embedding.tolist()} (high color dimension)")
    print(f"car = {car_embedding.tolist()} (high vehicle dimension)")
    
    input("\nPress Enter to create Query, Key, Value vectors...")
    
    # Weight matrices (2x2 for simplicity)
    W_q = torch.tensor([[0.5, 0.3], [0.2, 0.7]])
    W_k = torch.tensor([[0.4, 0.6], [0.8, 0.1]]) 
    W_v = torch.tensor([[1.0, 0.2], [0.3, 0.8]])
    
    # Generate Q, K, V
    red_query = red_embedding @ W_q
    red_key = red_embedding @ W_k
    red_value = red_embedding @ W_v
    
    car_query = car_embedding @ W_q  
    car_key = car_embedding @ W_k
    car_value = car_embedding @ W_v
    
    print(f"\n🎯 Generated Vectors:")
    print(f"red_query = {red_query.round(decimals=2).tolist()}")
    print(f"red_key = {red_key.round(decimals=2).tolist()}")
    print(f"red_value = {red_value.round(decimals=2).tolist()}")
    print(f"car_query = {car_query.round(decimals=2).tolist()}")
    print(f"car_key = {car_key.round(decimals=2).tolist()}")
    print(f"car_value = {car_value.round(decimals=2).tolist()}")
    
    
    input("\nPress Enter to calculate attention scores...")
    
    # Calculate attention scores
    car_to_red_score = torch.dot(car_query, red_key)
    car_to_car_score = torch.dot(car_query, car_key)
    
    print(f"\n🔍 Attention Scores (Raw Dot Products):")
    print(f"car attending to red: torch.dot(car_query, red_key) = {car_to_red_score:.3f}")
    print(f"car attending to itself: torch.dot(car_query, car_key) = {car_to_car_score:.3f}")
    print(f"Raw scores vector: [{car_to_red_score:.3f}, {car_to_car_score:.3f}]")
    
    input("\nPress Enter to apply softmax normalization...")
    
    # Apply softmax - show the step-by-step process
    scores = torch.tensor([car_to_red_score, car_to_car_score])
    
    print(f"\n⚖️ Softmax Transformation:")
    print(f"Step 1 - Raw scores: {scores.round(decimals=3).tolist()}")
    
    # Show exponential step
    exp_scores = torch.exp(scores)
    print(f"Step 2 - Apply exp(): {exp_scores.round(decimals=3).tolist()}")
    
    # Show sum for normalization
    sum_exp = torch.sum(exp_scores)
    print(f"Step 3 - Sum of exp(): {sum_exp:.3f}")
    
    # Final probabilities
    attention_weights = exp_scores / sum_exp
    print(f"Step 4 - Normalize: {attention_weights.round(decimals=3).tolist()}")
    
    # Verify it's a probability distribution
    print(f"✅ Sum check: {attention_weights.sum():.3f} (should be 1.0)")
    print(f"→ car pays {attention_weights[0]:.1%} attention to red!")
    print(f"→ car pays {attention_weights[1]:.1%} attention to itself!")
    
    input("\nPress Enter to see the final transformation...")
    
    # Final weighted combination
    new_car = attention_weights[0] * red_value + attention_weights[1] * car_value
    
    print(f"\n🎉 Final Result:")
    print(f"Original car = {car_embedding.tolist()}")
    print(f"Updated car = {new_car.round(decimals=2).tolist()}")
    print(f"→ car now has significant color information! 🎨")
    
    print("\n✨ Mathematics made the magic happen! ✨")
    print(f"\n🔄 Summary: Scores → Weights")
    print(f"Raw scores: {scores.round(decimals=3).tolist()} (can be negative/any range)")
    print(f"Final weights: {attention_weights.round(decimals=3).tolist()} (always positive, sum to 1)")
    
    return red_embedding, car_embedding, new_car, attention_weights


def mathematical_attention_example_full():
    """Work through attention with real numbers and visualizations"""
    
    import torch
    import numpy as np
    
    print("🔢 Mathematical Attention Example: 'the red car'")
    print("=" * 60)
    input("Press Enter to see initial embeddings...")
    
    # Initial embeddings (2D for simplicity) - NOW WITH 3 TOKENS
    the_embedding = torch.tensor([0.1, 0.1])   # Low color, low vehicle (function word)
    red_embedding = torch.tensor([0.8, 0.2])  # High color, low vehicle
    car_embedding = torch.tensor([0.1, 0.9])  # Low color, high vehicle
    
    print(f"\n📊 Initial Embeddings:")
    print(f"the = {the_embedding.tolist()} (function word - low semantic content)")
    print(f"red = {red_embedding.tolist()} (high color dimension)")
    print(f"car = {car_embedding.tolist()} (high vehicle dimension)")
    
    input("\nPress Enter to create Query, Key, Value vectors...")
    
    # Weight matrices (2x2 for simplicity)
    W_q = torch.tensor([[0.5, 0.3], [0.2, 0.7]])
    W_k = torch.tensor([[0.4, 0.6], [0.8, 0.1]]) 
    W_v = torch.tensor([[1.0, 0.2], [0.3, 0.8]])
    
    # Generate Q, K, V for ALL tokens
    the_query = the_embedding @ W_q
    the_key = the_embedding @ W_k
    the_value = the_embedding @ W_v
    
    red_query = red_embedding @ W_q
    red_key = red_embedding @ W_k
    red_value = red_embedding @ W_v
    
    car_query = car_embedding @ W_q  
    car_key = car_embedding @ W_k
    car_value = car_embedding @ W_v
    
    print(f"\n🎯 Generated Vectors:")
    print(f"the_query = {the_query.round(decimals=2).tolist()}")
    print(f"red_query = {red_query.round(decimals=2).tolist()}")
    print(f"car_query = {car_query.round(decimals=2).tolist()}")
    
    input("\nPress Enter to calculate attention scores...")
    
    # Calculate attention scores for car attending to all tokens
    car_to_the_score = torch.dot(car_query, the_key)
    car_to_red_score = torch.dot(car_query, red_key)
    car_to_car_score = torch.dot(car_query, car_key)
    
    print(f"\n🔍 Attention Scores (car attending to all tokens):")
    print(f"car attending to 'the': {car_to_the_score:.3f}")
    print(f"car attending to 'red': {car_to_red_score:.3f}")
    print(f"car attending to itself: {car_to_car_score:.3f}")
    
    # Apply softmax across all 3 scores
    scores = torch.tensor([car_to_the_score, car_to_red_score, car_to_car_score])
    attention_weights = torch.softmax(scores, dim=0)
    
    print(f"\n⚖️ Attention Weights (after softmax):")
    print(f"car pays {attention_weights[0]:.1%} attention to 'the'")
    print(f"car pays {attention_weights[1]:.1%} attention to 'red'") 
    print(f"car pays {attention_weights[2]:.1%} attention to itself")
    
    # Final weighted combination includes ALL tokens
    new_car = (attention_weights[0] * the_value + 
               attention_weights[1] * red_value + 
               attention_weights[2] * car_value)
    
    print(f"\n🎉 Final Result:")
    print(f"new_car = {attention_weights[0]:.3f} × the_value + {attention_weights[1]:.3f} × red_value + {attention_weights[2]:.3f} × car_value")
    print(f"→ 'the' typically gets low attention (function word)")
    print(f"→ 'red' gets high attention (describes car)")
    print(f"→ 'car' pays some attention to itself")
    
    
def mathematical_attention_example_full_masked():
    """Work through attention with real numbers and visualizations"""
    
    import torch
    import numpy as np
    
    print("🔢 Mathematical Attention Example: 'the red car'")
    print("=" * 60)
    input("Press Enter to see initial embeddings...")
    
    # Initial embeddings (2D for simplicity) - NOW WITH 3 TOKENS
    the_embedding = torch.tensor([0.1, 0.1])   # Low color, low vehicle (function word)
    red_embedding = torch.tensor([0.8, 0.2])  # High color, low vehicle
    car_embedding = torch.tensor([0.1, 0.9])  # Low color, high vehicle
    
    print(f"\n📊 Initial Embeddings:")
    print(f"the = {the_embedding.tolist()} (function word - low semantic content)")
    print(f"red = {red_embedding.tolist()} (high color dimension)")
    print(f"car = {car_embedding.tolist()} (high vehicle dimension)")
    
    input("\nPress Enter to create Query, Key, Value vectors...")
    
    # Weight matrices (2x2 for simplicity)
    W_q = torch.tensor([[0.5, 0.3], [0.2, 0.7]])
    W_k = torch.tensor([[0.4, 0.6], [0.8, 0.1]]) 
    W_v = torch.tensor([[1.0, 0.2], [0.3, 0.8]])
    
    # Generate Q, K, V for ALL tokens
    the_query = the_embedding @ W_q
    the_key = the_embedding @ W_k
    the_value = the_embedding @ W_v
    
    red_query = red_embedding @ W_q
    red_key = red_embedding @ W_k
    red_value = red_embedding @ W_v
    
    car_query = car_embedding @ W_q  
    car_key = car_embedding @ W_k
    car_value = car_embedding @ W_v
    
    print(f"\n🎯 Generated Vectors:")
    print(f"the_query = {the_query.round(decimals=2).tolist()}")
    print(f"red_query = {red_query.round(decimals=2).tolist()}")
    print(f"car_query = {car_query.round(decimals=2).tolist()}")
    
    input("\nPress Enter to calculate attention scores...")
    
    # Calculate COMPLETE attention matrix (all tokens attending to all tokens)
    print(f"\n🔍 Complete Attention Scores (Raw Dot Products):")
    
    # All pairwise dot products
    the_to_the_score = torch.dot(the_query, the_key)
    the_to_red_score = torch.dot(the_query, red_key)
    the_to_car_score = torch.dot(the_query, car_key)
    
    red_to_the_score = torch.dot(red_query, the_key)
    red_to_red_score = torch.dot(red_query, red_key)
    red_to_car_score = torch.dot(red_query, car_key)
    
    car_to_the_score = torch.dot(car_query, the_key)
    car_to_red_score = torch.dot(car_query, red_key)
    car_to_car_score = torch.dot(car_query, car_key)
    
    # Create attention matrix
    attention_matrix = torch.tensor([
        [the_to_the_score, the_to_red_score, the_to_car_score],
        [red_to_the_score, red_to_red_score, red_to_car_score],
        [car_to_the_score, car_to_red_score, car_to_car_score]
    ])
    
    print(f"Attention Matrix (rows=queries, cols=keys):")
    print(f"       the     red     car")
    print(f"the  {attention_matrix[0,0]:.3f}  {attention_matrix[0,1]:.3f}  {attention_matrix[0,2]:.3f}")
    print(f"red  {attention_matrix[1,0]:.3f}  {attention_matrix[1,1]:.3f}  {attention_matrix[1,2]:.3f}")
    print(f"car  {attention_matrix[2,0]:.3f}  {attention_matrix[2,1]:.3f}  {attention_matrix[2,2]:.3f}")
    
    input("\nPress Enter to apply causal masking...")
    
    # Apply causal mask (autoregressive - can only attend to previous tokens + self)
    print(f"\n🚫 Applying Causal Mask (GPT-style):")
    print("Tokens can only attend to previous tokens + themselves!")
    
    # Create causal mask (upper triangular filled with -inf)
    mask = torch.triu(torch.full((3, 3), float('-inf')), diagonal=1)
    print(f"Causal mask:")
    print(f"       the     red     car")
    print(f"the   0.0    -inf    -inf")
    print(f"red   0.0     0.0    -inf") 
    print(f"car   0.0     0.0     0.0")
    
    # Apply mask
    masked_scores = attention_matrix + mask
    
    print(f"\nMasked scores:")
    print(f"       the     red     car")
    for i, token in enumerate(['the', 'red', 'car']):
        row_str = f"{token:3s}"
        for j in range(3):
            if masked_scores[i,j] == float('-inf'):
                row_str += "    -inf"
            else:
                row_str += f"  {masked_scores[i,j]:6.3f}"
        print(row_str)
    
    input("\nPress Enter to apply softmax with masking...")
    
    # Apply softmax row-wise (each token's attention distribution)
    print(f"\n⚖️ Softmax with Causal Masking:")
    
    the_attention_weights = torch.softmax(masked_scores[0], dim=0)
    red_attention_weights = torch.softmax(masked_scores[1], dim=0)
    car_attention_weights = torch.softmax(masked_scores[2], dim=0)
    
    print(f"'the' attention weights: {the_attention_weights.round(decimals=3).tolist()}")
    print(f"'red' attention weights: {red_attention_weights.round(decimals=3).tolist()}")
    print(f"'car' attention weights: {car_attention_weights.round(decimals=3).tolist()}")
    
    print(f"\n🔍 What this means:")
    print(f"→ 'the' can only attend to itself (100% self-attention)")
    print(f"→ 'red' can attend to 'the' ({red_attention_weights[0]:.1%}) and itself ({red_attention_weights[1]:.1%})")
    print(f"→ 'car' can attend to all previous tokens: 'the' ({car_attention_weights[0]:.1%}), 'red' ({car_attention_weights[1]:.1%}), itself ({car_attention_weights[2]:.1%})")
    
    input("\nPress Enter to see the final transformations...")
    
    # Final weighted combinations for ALL tokens
    new_the = the_attention_weights[0] * the_value  # Only attends to itself
    
    new_red = (red_attention_weights[0] * the_value + 
               red_attention_weights[1] * red_value)
               
    new_car = (car_attention_weights[0] * the_value + 
               car_attention_weights[1] * red_value + 
               car_attention_weights[2] * car_value)
    
    print(f"\n🎉 Final Results with Causal Masking:")
    print(f"Original 'the' = {the_embedding.tolist()}")
    print(f"Updated 'the'  = {new_the.round(decimals=2).tolist()}")
    print(f"Original 'red' = {red_embedding.tolist()}")
    print(f"Updated 'red'  = {new_red.round(decimals=2).tolist()}")
    print(f"Original 'car' = {car_embedding.tolist()}")
    print(f"Updated 'car'  = {new_car.round(decimals=2).tolist()}")
    
    print(f"\n💡 Key Insight:")
    print(f"• 'the' unchanged (no previous context)")
    print(f"• 'red' slightly influenced by 'the'")
    print(f"• 'car' combines info from ALL previous tokens")
    print(f"• Apply mask: the → red → car    # Information flows left-to-right only")
    print(f"• This prevents the model from 'cheating' by seeing future words!")
    
    return the_embedding, red_embedding, car_embedding, new_the, new_red, new_car



def demonstrate_qkv_process():
    """Step-by-step demonstration of Query-Key-Value attention"""
    
    print("🎬 Scene: Processing 'The red car'")
    print("=" * 50)
    input("Press Enter to continue...")
    
    # Step 1: Show what each word contributes
    words_info = {
        "red": {
            "query": "What nouns do I describe?",
            "key": "I'm a color adjective", 
            "value": "[color info, visual properties]"
        },
        "car": {
            "query": "What words describe me?",
            "key": "I'm a vehicle noun",
            "value": "[vehicle info, transportation]"
        }
    }
    
    print("\n📋 Step 1: Each word creates Query, Key, Value")
    for word, info in words_info.items():
        print(f"\n'{word}':")
        print(f"  Query: {info['query']}")
        print(f"  Key: {info['key']}")  
        print(f"  Value: {info['value']}")
        
    input("\nPress Enter to see attention scoring...")
    
    # Step 2: Show attention scoring
    print("\n🎯 Step 2: Calculate Attention Scores")
    print("car_query × red_key = HIGH SCORE ✅ (adjective-noun match)")
    print("red_query × car_key = LOW SCORE ❌ (red doesn't need car info)")
    
    input("\nPress Enter to see probability conversion...")
    
    # Step 3: Show softmax
    print("\n📊 Step 3: Convert to Probabilities")
    print("car pays: 80% attention to red, 20% to itself")
    print("red pays: 10% attention to car, 90% to itself")
    
    input("\nPress Enter to see the final result...")
    
    # Step 4: Show final result
    print("\n🎉 Step 4: Final Result")
    print("car_embedding = original_car + 0.8 × red_information")
    print("→ car now knows it's red! 🔴🚗")
    
    print("\n✨ That's the magic of attention! ✨")
    
    
def visualize_embedding_transformation(red_emb, car_emb, new_car_emb, weights):
    """Visualize how attention transforms embeddings"""
    
    # Data from previous calculation
    embeddings = {
        'red': red_emb.numpy(),
        'car (original)': car_emb.numpy(), 
        'car (updated)': new_car_emb.detach().numpy()
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Before and after embeddings
    words = list(embeddings.keys())
    colors = ['red', 'blue', 'green']
    
    for i, (word, embedding) in enumerate(embeddings.items()):
        ax1.scatter(embedding[0], embedding[1], s=200, c=colors[i], label=word, alpha=0.8)
        ax1.annotate(word, (embedding[0], embedding[1]), xytext=(10, 10), 
                    textcoords='offset points', fontsize=10)
    
    # Draw arrow showing transformation
    ax1.annotate('', xy=new_car_emb.numpy(), xytext=car_emb.numpy(),
                arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    ax1.text(0.4, 0.7, 'Attention Transformation', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.6))
    
    ax1.set_xlabel('Color Dimension')
    ax1.set_ylabel('Vehicle Dimension')
    ax1.set_title('Embedding Space Transformation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Attention weights
    ax2.bar(['red', 'car (self)'], weights.numpy(), color=['red', 'blue'], alpha=0.7)
    ax2.set_ylabel('Attention Weight')
    ax2.set_title('How Much "car" Attends to Each Word')
    ax2.set_ylim(0, 1)
    
    # Add percentage labels
    for i, weight in enumerate(weights):
        ax2.text(i, weight + 0.02, f'{weight:.1%}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    print("🔍 Key Insight:")
    print(f"• car's color dimension increased from {car_emb[0]:.1f} to {new_car_emb[0]:.1f}")
    print(f"• This happened because car paid {weights[0]:.1%} attention to red!")
    print("• Attention literally moves embeddings in semantic space! 🚀")
    
    
def detective_attention_demo():
    """Show how attention connects long-range dependencies in detective story"""
    
    # Simplified detective story
    story = [
        "The", "butler", "entered", "the", "study", "at", "9", "PM", ".",
        "Lady", "Margaret", "was", "found", "dead", "at", "10", "PM", ".",
        "The", "gardener", "had", "an", "alibi", ".",
        "Therefore", "the", "murderer", "was"
    ]
    
    # When processing "murderer", attention focuses on key evidence
    # Simulated attention weights (higher = more attention)
    murderer_attention = [
        0.05, 0.3,  0.2,  0.05, 0.25, 0.05, 0.1,  0.05, 0.02,  # butler, entered, study
        0.05, 0.05, 0.05, 0.05, 0.1,  0.05, 0.1,  0.05, 0.02,  # Lady Margaret, dead, 10 PM
        0.05, 0.02, 0.05, 0.05, 0.02, 0.02,                     # gardener, alibi
        0.05, 0.05, 0.1,  0.0                                    # therefore, the, murderer
    ]
    
    # Create attention heatmap
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create heatmap data
    attention_matrix = np.array(murderer_attention).reshape(1, -1)
    
    im = ax.imshow(attention_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=0.3)
    
    # Set ticks and labels
    ax.set_xticks(range(len(story)))
    ax.set_xticklabels(story, rotation=45, ha='right')
    ax.set_yticks([0])
    ax.set_yticklabels(['murderer'])
    
    # Highlight key evidence
    high_attention_indices = [i for i, att in enumerate(murderer_attention) if att > 0.15]
    for idx in high_attention_indices:
        ax.text(idx, 0, '!!!', ha='center', va='center', fontsize=16)
    
    ax.set_title('Attention Pattern: "murderer" attending to evidence', fontsize=14, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')
    
    plt.tight_layout()
    plt.show()
    
    print("!!! Detective's Attention Pattern:")
    print("🟥 High attention (red) → Key evidence: butler, entered, study, dead, 10 PM")
    print("🟨 Medium attention → Contextual info: Therefore, the")
    print("⬜ Low attention → Less relevant: articles, punctuation")
    print("💡 Result: Model can predict 'butler' based on accumulated evidence!")
    
    
def interactive_tower_attention():
    """Interactive demonstration of how 'tower' meaning changes with context"""
    
    scenarios = [
        {
            "sentence": "Look at that tall tower",
            "words": ["Look", "at", "that", "tall", "tower"],
            "tower_attention": [0.1, 0.1, 0.15, 0.4, 0.25],  # Focuses on 'tall'
            "meaning": "Generic tall structure",
            "embedding_shift": "Neutral (no specific context)"
        },
        {
            "sentence": "Look at that tall Eiffel tower", 
            "words": ["Look", "at", "that", "tall", "Eiffel", "tower"],
            "tower_attention": [0.05, 0.05, 0.1, 0.2, 0.5, 0.1],  # High attention to 'Eiffel'
            "meaning": "Iconic Paris landmark",
            "embedding_shift": "Updated with French landmark info"
        },
        {
            "sentence": "Look at that tall miniature tower",
            "words": ["Look", "at", "that", "tall", "miniature", "tower"], 
            "tower_attention": [0.05, 0.05, 0.1, 0.2, 0.5, 0.1],  # High attention to 'miniature'
            "meaning": "Small scale model",
            "embedding_shift": "Updated with size/scale info"
        }
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Top row: attention weights
    for i, scenario in enumerate(scenarios):
        words = scenario["words"]
        weights = scenario["tower_attention"]
        
        bars = axes[0, i].bar(words, weights, 
                             color=['orange' if w > 0.3 else 'lightblue' for w in weights])
        
        # Highlight 'tower'
        tower_idx = words.index('tower')
        bars[tower_idx].set_color('red')
        bars[tower_idx].set_alpha(0.8)
        
        axes[0, i].set_title(f'Attention Pattern → "{scenario["sentence"]}"', fontsize=10)
        axes[0, i].set_ylabel('Attention Weight')
        axes[0, i].tick_params(axis='x', rotation=45)
        axes[0, i].set_ylim(0, 0.6)
    
    # Bottom row: embedding transformations (conceptual)
    transformations = [
        "Tower[generic structure]",
        "Tower + Eiffel[Paris landmark]", 
        "Tower + miniature[small model]"
    ]
    
    colors = ['gray', 'gold', 'purple']
    for i, (transform, color) in enumerate(zip(transformations, colors)):
        axes[1, i].text(0.5, 0.5, transform, ha='center', va='center', 
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.6))
        axes[1, i].set_xlim(0, 1)
        axes[1, i].set_ylim(0, 1) 
        axes[1, i].axis('off')
        axes[1, i].set_title(scenarios[i]["meaning"], fontsize=10, pad=10)
    
    plt.suptitle('How Attention Transforms "Tower" Meaning', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("🔄 The Attention Process:")
    print("1. 'tower' asks: 'What describes me?'")
    print("2. Other words respond with relevance scores")
    print("3. High-scoring words update tower's meaning")
    print("4. Result: Context-specific representation!")
    
    
    
def visualize_mole_disambiguation():
    """Show how attention disambiguates the word 'mole' in different contexts"""
    
    examples = [
        {
            "sentence": "American shrew mole",
            "context": "animal",
            "attention_to_mole": ["American: 0.4", "shrew: 0.5", "mole: 0.1"],
            "meaning": "small mammal",
            "color": "green"
        },
        {
            "sentence": "One mole of carbon dioxide", 
            "context": "chemistry",
            "attention_to_mole": ["One: 0.3", "mole: 0.1", "carbon: 0.3", "dioxide: 0.3"],
            "meaning": "measurement unit",
            "color": "blue"
        },
        {
            "sentence": "Take a biopsy of the mole",
            "context": "medical", 
            "attention_to_mole": ["biopsy: 0.6", "of: 0.1", "the: 0.1", "mole: 0.2"],
            "meaning": "skin growth",
            "color": "red"
        }
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for i, example in enumerate(examples):
        # Extract attention data
        words = [item.split(": ")[0] for item in example["attention_to_mole"]]
        weights = [float(item.split(": ")[1]) for item in example["attention_to_mole"]]
        
        # Create attention plot
        bars = axes[i].bar(words, weights, color=example["color"], alpha=0.7)
        
        # Highlight 'mole' if present
        if 'mole' in words:
            mole_idx = words.index('mole')
            bars[mole_idx].set_color('black')
            bars[mole_idx].set_alpha(0.8)
        
        axes[i].set_title(f'"{example["sentence"]}"→ {example["meaning"]}', 
                         fontsize=11, pad=15)
        axes[i].set_ylabel('Attention Weight')
        axes[i].set_ylim(0, 0.7)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.suptitle('How Attention Disambiguates "Mole"', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    print("💡 The Magic of Context:")
    print("• Same word 'mole' gets different meanings based on what it pays attention to")
    print("• Animal context: focuses on 'American' and 'shrew'") 
    print("• Chemistry context: focuses on 'One', 'carbon', 'dioxide'")
    print("• Medical context: focuses on 'biopsy'")
    
    


# Simple attention demonstration
def simple_attention_demo():
    """Demonstrate how attention weights change based on context"""
    
    # Example sentences with different contexts for 'bank'
    sentences = [
        "The river bank was muddy",
        "The bank approved my loan", 
        "I deposited money at the bank"
    ]
    
    # Simulated attention weights for word 'bank' attending to other words
    attention_weights = {
        "Sentence 1 (river context)": {
            "words": ["The", "river", "bank", "was", "muddy"],
            "weights": [0.1, 0.6, 0.1, 0.1, 0.1]  # High attention to 'river'
        },
        "Sentence 2 (financial context)": {
            "words": ["The", "bank", "approved", "my", "loan"], 
            "weights": [0.1, 0.1, 0.2, 0.1, 0.5]  # High attention to 'loan'
        },
        "Sentence 3 (financial context)": {
            "words": ["I", "deposited", "money", "at", "the", "bank"],
            "weights": [0.1, 0.4, 0.3, 0.1, 0.05, 0.05]  # High attention to 'deposited', 'money'
        }
    }
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, (title, data) in enumerate(attention_weights.items()):
        words = data["words"]
        weights = data["weights"]
        
        # Create bar plot
        bars = axes[i].bar(words, weights, color=['lightblue' if w < 0.3 else 'orange' for w in weights])
        axes[i].set_title(title, fontsize=12, pad=20)
        axes[i].set_ylabel('Attention Weight')
        axes[i].set_ylim(0, 0.7)
        
        # Highlight the word 'bank' 
        bank_idx = words.index('bank') if 'bank' in words else None
        if bank_idx is not None:
            bars[bank_idx].set_color('red')
            bars[bank_idx].set_alpha(0.7)
        
        # Rotate x-axis labels
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("🎯 Key Insight: The word 'bank' attends to different words based on context!")
    print("• River context → focuses on 'river' and 'muddy'")
    print("• Financial context → focuses on 'loan', 'deposited', 'money'")
    print("• Same word, different attention patterns!")