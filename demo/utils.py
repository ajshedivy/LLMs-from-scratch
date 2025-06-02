import gensim.downloader
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  – triggers Matplotlib 3-D support
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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