from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Neural Network Architecture')

# Input Layer
dot.node('A', 'Input Layer\n(batch_size, 1, window_size, num_features)')
dot.node('B', 'Conv2D Layer 1\n(batch_size, 32, window_size, num_features)')
dot.node('C', 'BatchNorm + ReLU')
dot.node('D', 'MaxPool2D\n(batch_size, 32, window_size/2, num_features/2)')

# Conv Layer 2
dot.node('E', 'Conv2D Layer 2\n(batch_size, 64, window_size/2, num_features/2)')
dot.node('F', 'BatchNorm + ReLU')
dot.node('G', 'MaxPool2D\n(batch_size, 64, window_size/4, num_features/4)')

# Conv Layer 3
dot.node('H', 'Conv2D Layer 3\n(batch_size, 128, window_size/4, num_features/4)')
dot.node('I', 'BatchNorm + ReLU')
dot.node('J', 'MaxPool2D\n(batch_size, 128, window_size/8, num_features/8)')

# Flatten Layer
dot.node('K', 'Flatten\n(batch_size, flattened_size)')

# Fully Connected Layers
dot.node('L', 'Fully Connected Layer 1\n(batch_size, 256)')
dot.node('M', 'BatchNorm + ReLU')
dot.node('N', 'Dropout')
dot.node('O', 'Fully Connected Layer 2\n(batch_size, 64)')
dot.node('P', 'BatchNorm + ReLU')
dot.node('Q', 'Dropout')
dot.node('R', 'Output Layer\n(batch_size, 1)')

# Connect the nodes
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')

dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G')

dot.edge('G', 'H')
dot.edge('H', 'I')
dot.edge('I', 'J')

dot.edge('J', 'K')

dot.edge('K', 'L')
dot.edge('L', 'M')
dot.edge('M', 'N')
dot.edge('N', 'O')
dot.edge('O', 'P')
dot.edge('P', 'Q')
dot.edge('Q', 'R')

# Add dashed edges for fully connected layers
dot.edge('L', 'M', style='dashed')
dot.edge('M', 'N', style='dashed')
dot.edge('O', 'P', style='dashed')
dot.edge('P', 'Q', style='dashed')

# Render the graph to a PNG file
dot.render('neural_network_architecture', format='png', cleanup=True)

# Output the file
print("Graph generated as 'neural_network_architecture.png'")
