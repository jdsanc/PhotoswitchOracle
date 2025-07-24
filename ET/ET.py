"""
Wrapper class for Equivariant Transformer that outputs both invariant and equivariant node embeddings.
"""

import torch
from torch import nn, Tensor
from typing import Optional, Tuple
from .equivariant_transformer import EquivariantTransformer
from .output_modules import OutputHead
from .utils import scatter


class ET_backbone(nn.Module):
    """
    Single wrapper class for Equivariant Transformer that outputs both invariant and equivariant embeddings
    for both nodes and molecules in a dictionary format.
    
    Args:
        representation_model: The Equivariant Transformer model
        invariant_dim: Dimension of the invariant (scalar) embeddings
        equivariant_dim: Dimension of the equivariant (vector) embeddings
        hidden_channels: Hidden dimension from the representation model
        activation: Activation function for the projection layers
        dtype: Data type for the model
        num_hidden_layers: Number of hidden layers in projection networks
        aggregation: Aggregation method for molecular embeddings ("sum", "mean", "max")
    """
    
    def __init__(
        self,
        emb_dim: int,
        num_layers: int,
        num_heads: int,
        n_node_classes = 120,
        cutoff_upper : float = 6.0,
        activation: str = "silu",
        aggregation: str = "sum",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.dtype = dtype
        self.emb_dim = emb_dim
        self.n_node_classes = n_node_classes
        self.cutoff_upper = cutoff_upper
        self.aggregation = aggregation

        self.rep_model = EquivariantTransformer(
            hidden_channels=emb_dim,
            num_layers=num_layers,
            num_rbf=64,
            rbf_type="expnorm",
            trainable_rbf=True,
            activation=activation,
            attn_activation=activation,
            neighbor_embedding=True,
            num_heads=num_heads,
            distance_influence="both",
            cutoff_lower=0.0,
            cutoff_upper=cutoff_upper,
            max_z=n_node_classes,
            max_num_neighbors=32,
            check_errors=True,
            box_vecs=None,
            vector_cutoff=True,
            dtype=dtype,
            )

        # Create projection networks for invariant embeddings
        self.node_head = OutputHead(
            input_dim = emb_dim,
            hidden_dim = emb_dim,
            output_dim = emb_dim,
            aggregate = False,
            agg_op=aggregation,
            activation="silu",
            dtype=dtype,
            )
        
        self.mol_head = OutputHead(
            input_dim = emb_dim,
            hidden_dim = emb_dim,
            output_dim = emb_dim,
            aggregate = True,
            agg_op=aggregation,
            activation="silu",
            dtype=dtype,
            )
    def get_layer_groups(self):

        '''
        returns a list of tuples, where each tuple contains the name and parameter of a layer group.
        layer groups are defined by depth in the model, e.g. embedding layers, main layers, head layers.
        '''

        ### parse out layer groups so that a layer wise learning rate can be applied
        self.embedding_layer =[
            'rep_model.embedding',
            'rep_model.distance',
            'rep_model.neighbor',
            ]
        
        self.main_layers = [
            'rep_model.attention_'
            ]
        
        self.head_layer = [
            'rep_model.out',
            'node_head',
            'mol_head',
            ]
        
        emb_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if name.startswith(tuple(self.embedding_layer)):
                emb_params.append((name, param))
            elif name.startswith(tuple(self.main_layers)):
                continue
            elif name.startswith(tuple(self.head_layer)):
                head_params.append((name, param))
            else:
                raise ValueError(f"Parameter {name} not assigned to any layer group!")

        layer_groups = [emb_params]
        for layer in self.rep_model.attention_layers:
            layer_groups.append([(n,p) for n, p in layer.named_parameters() if p.requires_grad])
        layer_groups.append(head_params)

        return layer_groups
    
    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
    ) -> dict:
    # def forward(self, inputs):
        """
        Forward pass that returns both node and molecular embeddings in a dictionary.
        
        Args:
            z (Tensor): Atomic numbers of shape (N,)
            pos (Tensor): Atomic positions of shape (N, 3)
            batch (Tensor, optional): Batch indices of shape (N,)
            box (Tensor, optional): Box vectors for periodic boundary conditions
            q (Tensor, optional): Atomic charges of shape (N,)
            s (Tensor, optional): Atomic spins of shape (N,)
            
        Returns:
            dict: Dictionary containing:
                - 'node_inv': Node invariant embeddings of shape (N, invariant_dim)
                - 'node_eqv': Node equivariant embeddings of shape (N, 3, equivariant_dim)
                - 'mol_inv': Molecular invariant embeddings of shape (num_graphs, invariant_dim)
                - 'mol_eqv': Molecular equivariant embeddings of shape (num_graphs, 3, equivariant_dim)
        """
        # z = inputs.z
        # pos = inputs.pos
        # batch = inputs.batch
        # box = inputs.box if hasattr(inputs, 'box') else None
        # q = inputs.q if hasattr(inputs, 'q') else None
        # s = inputs.s if hasattr(inputs, 's') else None  

        # Get representation from the Equivariant Transformer
        x, vec, z, pos, batch = self.rep_model(z, pos, batch, box, q, s)
        
        # Project to node embeddings
        x_node, v_node = self.node_head(x, vec)

        # Project to molecular embeddings
        x_mol, v_mol = self.mol_head(x, vec, batch)

        # Return as dictionary
        return {
            'node_inv': x_node,
            'node_eqv': v_node,
            'mol_inv': x_mol,
            'mol_eqv': v_mol,
        }
