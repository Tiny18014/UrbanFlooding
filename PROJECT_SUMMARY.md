# Urban Flood Water Level Forecasting Project

## 🌊 Project Overview
This project tackles the **Urban Flood Water Level Forecasting** competition, focusing on predicting water levels in complex, coupled hydraulic systems. The core challenge lies in modeling the interaction between **1D underground drainage networks** and **2D surface overland flow meshes** under varying rainfall conditions.

By leveraging Graph Neural Networks (GNNs) and autoregressive sequence modeling, the goal is to accurately forecast water levels across thousands of spatial nodes, enabling better early warning systems for urban flooding.

---

## 🎯 Competition & Objective
*   **Goal:** Build an autoregressive model to predict water levels for coupled 1D (underground pipes/manholes) and 2D (surface terrain) systems.
*   **Target:** Water level (in meters) at each node for future timesteps.
*   **Metric:** **Standardized RMSE**. Errors are normalized by the standard deviation of water levels for each node type, ensuring that the sparse but critical 1D nodes contribute equally to the vast 2D mesh.
*   **Input Data:**
    *   **Forcing:** Rainfall intensity (temporal, spatially varying or uniform).
    *   **States:** Historical water levels, inlet flows.
    *   **Static Properties:** Node elevation, area, pipe connections, surface roughness.
    *   **Graph Structure:** 1D drainage graph, 2D surface mesh, and 1D-2D coupling edges.

---

## 📊 The Dataset Explanation
The dataset represents two distinct urban catchment areas ("Model 1" and "Model 2"), physically modeled using high-fidelity hydraulic software.

### 1. The Coupled 1D-2D System
The physical system is modeled as a **dual-layer graph**:
*   **1D Drainage Network (Underground):**
    *   Represents manholes, pipes, and pumps.
    *   **Structure:** A directed graph (flow direction matters).
    *   **Scale:** Sparse. Model 1 has ~17 nodes; Model 2 has ~198 nodes.
    *   **Dynamics:** Driven by inflow from the surface and pipe physics.
*   **2D Surface Mesh (Overland):**
    *   Represents the city terrain, streets, and buildings.
    *   **Structure:** An undirected mesh/grid.
    *   **Scale:** Massive. Model 1 has ~3,700 nodes; Model 2 has ~4,300 nodes.
    *   **Dynamics:** Driven directly by rainfall and gravity-based flow.
*   **Coupling (1D-2D):**
    *   Specific nodes in the 2D mesh are connected to 1D manholes (inlets).
    *   Water flows between these layers based on hydraulic head differences (e.g., surface runoff entering a drain, or a surcharging pipe flooding the street).

### 2. Rainfall Forcing
Rainfall is the primary driver ("forcing") of the system.
*   **Model 1:** Features **spatially varying rainfall**, meaning rain intensity differs across the map at the same timestep.
*   **Model 2:** Features **uniform rainfall**, where the entire catchment receives the same intensity at any given time.
*   **Data:** Provided as time-series data mapped to 2D nodes.

---

## 🔍 EDA & Analysis Summary
I have performed a comprehensive Exploratory Data Analysis across 5 notebooks to inform the modeling strategy.

### **Notebook 01: Data Loading & Statistics**
*   **Objective:** Establish data pipelines and baseline statistics.
*   **Work Done:**
    *   Loaded static graph files (nodes, edges) and dynamic event files.
    *   Quantified the **scale imbalance**: Validated that 2D nodes outnumber 1D nodes by ~220x in Model 1.
    *   Analyzed sequence lengths (variable, 94–445 timesteps).
    *   **Critical Step:** Calculated global means and standard deviations for water levels to implement the Standardized RMSE loss function.

### **Notebook 02: Target & Rainfall Analysis**
*   **Objective:** Understand the physics of the target variable.
*   **Work Done:**
    *   Analyzed water level distributions (non-normal, physically bounded).
    *   Confirmed strong **temporal autocorrelation** (water level at $t$ is highly dependent on $t-1$), identifying autoregressive models (LSTM/GRU) as strong candidates.
    *   Correlated rainfall peaks with flood peaks, confirming that 2D surface nodes respond faster and more strongly to rainfall than deep drainage nodes.

### **Notebook 03: Spatial & Network Analysis**
*   **Objective:** Visualize and map the graph topology.
*   **Work Done:**
    *   Visualized node positions to understand the catchment layout.
    *   Mapped **1D-2D coupling connections**, showing that drainage interaction is highly localized.
    *   Analyzed topographic features (Elevation, Flow Accumulation) which are critical for predicting surface flow direction.

### **Notebook 04: Feature Importance**
*   **Objective:** Select the best predictors.
*   **Work Done:**
    *   Performed correlation analysis between static/dynamic features and target water levels.
    *   **Top Predictors:**
        1.  **Temporal Lags:** Water levels at $t-1, t-2$.
        2.  **Rainfall:** Current and lagged precipitation.
        3.  **Static:** Surface elevation and flow accumulation.
    *   **PCA Analysis:** Showed minimal redundancy among the top features, suggesting all 5-6 key features should be kept.

### **Notebook 05: Event Analysis & Modeling Insights**
*   **Objective:** Strategy formulation.
*   **Work Done:**
    *   Clustered training events into 4 distinct groups (e.g., "Short/Intense" vs "Long/Drizzle") to ensure robust cross-validation.
    *   Established **Persistence Baselines** (predicting $t+1 = t$).
        *   Model 1 Baseline Score: 0.1645
        *   Model 2 Baseline Score: 0.0612
    *   Developed the final modeling strategy (Separated GNNs for 1D/2D).

---

## 🚀 Modeling Strategy & Future Work
Based on the analysis, the following approach is recommended for the competition submission:

1.  **Architecture:**
    *   **Separate Models:** Train distinct GNNs for the 1D network (high capacity, directed) and 2D mesh (efficient, undirected).
    *   **Message Passing:** Implement a specialized "Coupling Layer" to exchange hidden states between 1D and 2D graphs at each timestep.
    *   **Temporal Decoder:** Use an LSTM or GRU to handle the strong temporal dependencies identified in Notebook 02.

2.  **Training:**
    *   **Loss Function:** Standardized RMSE (weighted to balance the 1D/2D node count imbalance).
    *   **Strategy:** Autoregressive training with **Teacher Forcing** (slowly decaying ground-truth input) to stabilize long-horizon predictions.

3.  **Next Steps:**
    *   Implement the `GraphDataset` loader using PyTorch Geometric.
    *   Train the baseline LSTM (temporal only) to quantify the gain from adding spatial GNN layers.
    *   Execute the stratified cross-validation split defined in Notebook 05.
