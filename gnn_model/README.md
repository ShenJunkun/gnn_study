# GNN经典模型
GNN有非常多的经典模型，包括GraphSAGE、GCN、GAT、GIN等。、

# GNN模型
## GraphSAGE(Graph Sample and Agregated)
GraphSAGE通过采样图中的邻居节点，并将这些邻居节点的信息聚合到中心节点，从而学习节点的表示。它具有采用和聚合两个阶段，使其适用于大型图。

## GCN(Graph Convolutional Network)
GCN是一种基于卷积操作的GNN模型，通过聚合节点的邻居信息来学习节点的表示。它在图分类和节点分类任务上表现良好，被认为是图神经网络的里程碑之一。

## GAT(Graph Attention Network)
GAT使用注意力机制来为每个邻居节点分配不同的权重，从而更灵活地聚合邻居信息。这使得GAT能够捕捉到节点表示中的更复杂的关系。

## GIN (Graph Isomorphism Network)
GIN通过迭代图同构网络层，逐渐改进节点的表示。它的设计使得模型能够处理不同节点之间的关系，而不仅仅是邻居节点。