# Urban Flood Water Level Forecasting using Graph Neural Networks and LSTM

## Abstract
Urban flooding is a critical challenge requiring accurate and timely prediction systems. This project develops a deep learning framework to forecast water levels in complex urban catchments modeled as coupled 1D (sewer) and 2D (surface) hydraulic systems. By integrating Graph Neural Networks (GNNs) to capture spatial topology with Long Short-Term Memory (LSTM) networks for temporal dynamics, we achieved highly accurate predictions. The final hybrid model demonstrated outstanding performance with a normalized Mean Squared Error (MSE) of 0.0127 and an $R^2$ score of 0.9873, significantly outperforming baseline methods.

## 1. Introduction
Flooding in urban environments is complex because water moves through two distinct but interacting systems:
1.  **1D Sewer Networks**: The underground system of pipes and manholes that drains water away.
2.  **2D Surface Mesh**: The overland terrain (streets, buildings) where rain falls and runoff flows.

Traditional hydraulic simulations are accurate but computationally expensive and slow. This project aims to build a data-driven "surrogate model" that can emulate these physics-based simulations in real-time. The core challenge is effectively modeling the interactions between the sparse underground graph and the dense surface mesh under varying rainfall conditions.

## 2. Dataset & Exploratory Data Analysis
The study utilizes data from two distinct urban catchment models:
*   **Model 1**: A smaller, high-fidelity model (~3,700 surface nodes) with spatially varying rainfall data.
*   **Model 2**: A larger model (~4,300 surface nodes) assuming uniform rainfall.

Extensive Exploratory Data Analysis (EDA) revealed critical insights:
*   **Scale Imbalance**: The surface mesh (2D) contains nearly 220 times more nodes than the sewer network (1D). This necessitates a modeling approach that treats these layers differently.
*   **Temporal Dynamics**: Analysis of water level autocorrelation confirmed that past states are strong predictors of future levels, supporting the use of Recurrent Neural Networks (RNNs) like LSTM.
*   **Rainfall Response**: Surface nodes respond rapidly to rainfall peaks, while underground nodes show a lagged, damped response.
*   **Feature Importance**: Rainfall intensity, previous water levels, and static topographic features (elevation) were identified as the most predictive variables.

## 3. Methodology
To address the spatiotemporal complexity, we developed a **Hybrid GNN + LSTM** architecture (Method 2), which stands out as the most robust approach.

### 3.1. Architecture Overview
The model consists of three main components:
1.  **1D Encoder (GAT)**: A Graph Attention Network processes the underground pipe network. It uses attention mechanisms to weigh the influence of upstream and downstream manholes differently, reflecting the directional flow of water in pipes.
2.  **2D Encoder (GCN)**: A Graph Convolutional Network handles the surface mesh. Since surface flow is diffusive, a GCN efficiently aggregates information from neighboring terrain nodes.
3.  **Temporal Decoder (LSTM)**: The spatial features extracted by the GNNs are fed into an LSTM sequence model to predict the future water level based on the history of the event.

### 3.2. Code Implementation
The core PyTorch module defining this hybrid architecture is shown below:

```python
class HybridGNN_LSTM(nn.Module):
    def __init__(self, n_1d_nodes, n_2d_nodes,
                 hidden_1d=64, hidden_2d=32):
        super().__init__()

        # 1D Encoder: Graph Attention Network
        self.gat1d = GATConv(1, hidden_1d, heads=4, concat=False)

        # 2D Encoder: Graph Convolutional Network
        self.gcn2d = GCNConv(2, hidden_2d) # Input: water level + rainfall

        # LSTM Temporal Processors
        self.lstm_1d = nn.LSTM(hidden_1d, hidden_1d, batch_first=True)
        self.lstm_2d = nn.LSTM(hidden_2d, hidden_2d, batch_first=True)

        # Output Decoders
        self.dec_1d = nn.Linear(hidden_1d, 1)
        self.dec_2d = nn.Linear(hidden_2d, 1)

    def forward(self, x_1d, x_2d, rain, edge_index_1d, edge_index_2d, coupling_idx):
        # ... (Forward pass logic involving feature extraction and sequence modeling)
        # 1. Spatial encoding via GNNs
        # 2. Sequence processing via LSTMs
        # 3. Final prediction via Linear layers
        return pred_1d, pred_2d
```

### 3.3. Advanced Refinement (Ensemble Strategy)
In addition to the core GNN model, we explored an **Ensemble Approach (Method 3)**. This strategy categorizes rainfall events into "Regimes" (e.g., Short/Intense storms vs. Long/Light rain) using clustering algorithms. Different models (Random Forest, XGBoost, GNN) are then weighted dynamically based on the identified regime, improving robustness across diverse weather scenarios.

## 4. Experimental Results
The Hybrid GNN + LSTM model was trained and evaluated on historical flood events. The performance metrics on the standardized test set are excellent:

*   **Mean Squared Error (MSE)**: 0.0127
*   **Root Mean Squared Error (RMSE)**: 0.1126
*   **Mean Absolute Error (MAE)**: 0.0780
*   **Coefficient of Determination ($R^2$)**: 0.9873

**Visual Analysis**:
Plots comparing predicted water levels against actual simulation data show a very high degree of alignment. The model successfully captures:
*   The timing of the peak flood (when the water is highest).
*   The recession curve (how quickly the water drains away).
*   The interaction between the two layers (1D and 2D), validating the effectiveness of the dual-encoder design.

## 5. Conclusion
This project demonstrates that deep learning, specifically the combination of Graph Neural Networks and LSTMs, offers a powerful solution for real-time urban flood forecasting. By respecting the graph-based topology of sewer systems and surface terrain, the model achieves near-perfect accuracy ($R^2 > 0.98$) while being significantly faster than traditional physical simulations. Future work involves fully integrating the regime-based ensemble to further harden the system against extreme, unseen weather events.
