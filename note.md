#### Spectral graph coarsening for GCNNs

Let $A\in\mathbb{R}^{n\times n}$​ be an adjacency matrix, $X\in\mathbb{R}^{n\times d}$​ be the input node features. The spectral graph coarsening layer is defined as the following
$$
\begin{align}
X &= \textrm{GNN}(X, A; \Theta_\textrm{GNN}) \\
S &= \textrm{MLP}(X; \Theta_\textrm{MLP})
\end{align}
$$

The network parameters $\Theta_{\cdot}$ are optimized via gradient descent to minimize the following loss.
$$
\begin{align}
\mathcal{L}_u &= \mathcal{L}_c + \mathcal{L}_o \\
&= -\frac{Tr(S^TAS)}{Tr(S^TDS)} + \big\lvert\big\lvert\frac{S^TS}{||S^TS||_F} - \frac{I_K}{\sqrt{K}}\big\rvert\big\rvert_F
\end{align}
$$

Note that in the above formulation, the orthogonality term $\mathcal{L}_o$​ serves to encourage balanced clusters, while the cut term  $\mathcal{L}_c$​ serves to encourage clusters that satisfy certain cut properties.

The coarsened adjacency matrix and pooled vertex features are then computed:
$$
\hat{A} = S^TAS;\quad \hat{X}=S^TX
$$

The rows of $S$ can be interpreted as cluster assignments, and the columns can be interpreted as cluster centroid representations using the original observations as a dictionary. 

#### Semi-supervised graph coarsening

Consider if we have some labels $y$​​ associated with rows of $X$​ (the nodes). It makes sense that we may want clusters to satisfy label constraints- i.e. that nodes with the same label lie in the same partition. In the context of Spectral clustering, this corresponds to the inclusion of new constraints, where we want the constrain the signs of certain entries of $z$​ to be positive or negative:
$$
\begin{align}
&\min_z &&z^TLz \\
&\textrm{s.t.} &&z^T1 = 0 \textrm{, }z^Tz = n \\
& &&z^TD \leq 0
\end{align}
$$
where the rows of $D$​ are $1$​ or $-1$​​-hot vectors denoting the sign (intended cluster) of the associated entry of $z$​. 

Alternatively, a quadratic term where $L$​​ is the graph Laplacian, $C$​​ is a diagonal matrix (e.g. characterizes supervised/unsupervised examples), and $\gamma$​​ denotes the margin can also be used: 
$$
\begin{align}
&\min_z &&z^TLz + c(z - \gamma)^TC(z-\gamma) \\
&\textrm{s.t.} &&z^T1 = 0 \textrm{, }z^Tz = n
\end{align}
$$
One issue with the above is that $\gamma$​ needs to be set appropriately somehow.

Although we are pretty much done here, we can simplify things further. Consider the eigenvalue decomposition of $L=U\Sigma U^T$​. Let $V$​ ($D$​) be the matrix with all eigenvectors $U$​ (eigenvalues $\Sigma$​​), excluding the trivial ones. Then, the above optimization problem can be re-written:
$$
\begin{align}
&\min_w &&w^TDw + c(Vw - \gamma)^TC(Vw-\gamma) \\
&\textrm{s.t.} &&w^Tw = n
\end{align}
$$
with $G = (D + cV^TCV)$ and $b=cV^TC\gamma$​​, we can again re-write:
$$
\begin{align}
&\min_w &&w^TGw + -2b^Tw + c\gamma^TC\gamma \\
&\textrm{s.t.} &&w^Tw = n
\end{align}
$$

#### Adapting to neural networks

We want something like the above, but we don't want constraints so we can do gradient descent natively.

Here is the same formulation as sec. 1, but now we add a supervised loss term $\mathcal{L}_s$:
$$
\begin{align}
\mathcal{L}_u &= \mathcal{L}_c+ \mathcal{L}_o + \mathcal{L}_s  \\
&= -\frac{Tr(S^TAS)}{Tr(S^TDS)} + \big\lvert\big\lvert\frac{S^TS}{||S^TS||_F} - \frac{I_K}{\sqrt{K}}\big\rvert\big\rvert_F + \mathcal{L}_s  
\end{align}
$$

The question is how to set $\mathcal{L}_s$​​​​​? What kind of inductive bias do we want? One idea is to induce a sparse structure on the matrix $S$​​​​​. As mentioned previously, the rows of $S$​​ can be interpreted as cluster assignments, and the columns can be interpreted as cluster centroid representations using the original observations as a dictionary. In particular, labels on nodes induce a group structure on the rows of $S$​​ (each row of $S$​​​​​ corresponds to a node + label- nodes with the same labels belong to the same group..one question is how to set these groups for an intermediate coarsened graph- e.g. via prediction?)

<img src="/home/orange3xchicken/Downloads/sparsity.png" style="zoom:50%;" />

What kind of sparse structure do we want? (1) We want the rows of $S$​​​​​ (cluster assignments) to be sparse (mainly want individual samples to be members of a single cluster). This is accomplished via a softmax. (2) We want the columns to exhibit some kind of group structure (group sparsity, neighboring nodes sharing the same label should belong to the same cluster). (3) We want the columns of $S$​​ to be orthogonal.

In the fully supervised setting, when all nodes are labeled, i.e. all nodes are assigned to non-overlapping groups, the matrix group $\ell_{2,1}$​​​-norm satisfies what we want:
$$
R(S) = \lambda \sum_{i}\sum_{g}||S^{(i)}_g||_g \\
$$
Where $\lambda$ is a regularization weight, $||S^{(i)}_g||_g$ is the group $\ell_2$ norm on the $i$-th column of $S$: $||S^{(i)}_g||_g = \sqrt{\sum_{j=1}^{|G_g|}(S^{(i)}_{g,j})^2}$​.​​​

Things get harder if we consider semi-supervision (we know the groups of only a few nodes). We can apply techniques used to deal with soft or overlapping group assignments. Note that the labels are only used implicitly during training, so node labels are not needed for prediction.

##### Computational complexity

The space complexity of this algorithm is $O(NK)$​​, as it depends on the dimension of the cluster assignment matrix $S \in \mathbb{R}^{N\times k}$​​. The number of clusters $k$​ is a hyperparameter​

##### Initial experiments

Initial experiments are promising. For example, the regularizer results in nearly a 10% improvement in test set accuracy on the fully supervised Enzymes dataset for a simple gcnn architecture: CONV -> POOL -> CONV -> FC -> Dense 

The format of the plot titles is $\lambda:$​​training accuracy/valid accuracy/test accuracy (left regularized, right no regularizer)

<img src="/home/orange3xchicken/Downloads/enzymes_reg.png" alt="enzymes_reg" style="zoom: 67%;" /><img src="/home/orange3xchicken/Downloads/enzymes_noreg.png" alt="enzymes_noreg" style="zoom:67%;" />

##### WIP Abstract

Numerous graph machine learning architectures are available for supervised learning using either node labels or graph labels. However, an under-explored area is learning on graphs from hierarchical labels. For example, to encode traffic patterns in a road network, one may want to take into account both per-road (node) as well as per-town (subgraph) usage metrics. When encoding the graph structure of molecules, it may be useful to include both local measurements (e.g., bond lengths) as well as per-molecule (e.g., solubility). In this work, we propose a novel label-aware hierarchical graph-coarsening algorithm which exploits the topology of the graph, node representations, and node labels to augment graph and subgraph-level prediction.

##### WIP Intro

Learning on graph-structured data is a well-studied problem, and recent advances in graph neural networks have produced promising results. A wide variety of techniques for learning representations of graphs have been applied to a diverse set of problems including social network analysis, drug discovery, and molecular structure analysis~\cite{} in both fully supervised and semi-supervised settings~\cite{}.

A typical prerequisite for hierarchical or multi-scale-based methods is a scale-hierarchy which is either pre-defined or learned from the data. In this paper, we propose to induce multi-scale representation learning on graphs with neural networks through a multi-task learning framework which simultaneously optimizes local and global losses and backpropogates error signals from multiple levels of the scale-hierarchy. The levels of the scale-hierarchy are learned through graph pooling layers which progressively coarsen, or cluster, nodes of the input graph while classification layers can either be learned for each level of the hierarchy or coupled over all layers. By propagating local and global losses, we hypothesize two advantages: (i) by increasing the number and diversity of error signals observed by the neural network, we facilitate convergence to better solutions (ii) by encouraging the network to learn both global and local properties of the data, we facilitate representations that generalize better to unseen data.