graph TB
    %% Input Layer
    A[Input Layer<br>batch_size, 1, window_size, num_features] --> B[Conv2D Layer 1<br>batch_size, 32, window_size, num_features]
    B --> C[BatchNorm + ReLU]
    C --> D[MaxPool2D<br>batch_size, 32, window_size/2, num_features/2]
    
    %% Conv Layer 2
    D --> E[Conv2D Layer 2<br>batch_size, 64, window_size/2, num_features/2]
    E --> F[BatchNorm + ReLU]
    F --> G[MaxPool2D<br>batch_size, 64, window_size/4, num_features/4]
    
    %% Conv Layer 3
    G --> H[Conv2D Layer 3<br>batch_size, 128, window_size/4, num_features/4]
    H --> I[BatchNorm + ReLU]
    I --> J[MaxPool2D<br>batch_size, 128, window_size/8, num_features/8]
    
    %% Flatten Layer
    J --> K[Flatten<br>batch_size, flattened_size]
    
    %% Fully Connected Layers
    K --> L[Fully Connected Layer 1<br>batch_size, 256]
    L --> M[BatchNorm + ReLU]
    M --> N[Dropout]
    N --> O[Fully Connected Layer 2<br>batch_size, 64]
    O --> P[BatchNorm + ReLU]
    P --> Q[Dropout]
    Q --> R[Output Layer<br>batch_size, 1]
    
    %% Connections for Fully Connected Layers
    L -.-> M
    M -.-> N
    O -.-> P
    P -.-> Q
