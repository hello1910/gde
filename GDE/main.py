import dgl
import torch as th

g = dgl.DGLGraph()
g.add_nodes(4)
# A couple edges one-by-one
g.add_edge(3,0)
g.add_edge(3,1)
g.add_edge(3,2)
g.add_edge(0,1)
g.add_edge(0,2)
g.add_edge(1,2)
g.add_edges(g.nodes(), g.nodes())
edges = g.edges()
n_edges = g.number_of_edges()
x = th.randn(4, 70) . #edges size 80, #nodes size 60
g.ndata['x'] = x
g.edata['w'] = th.randn(10,80)

gdefunc = GCDEFunc(input_dim=64, hidden_dim=64, graph=g, activation=torch.tanh, dropout=0.9).to(device)

# dopri5 is an adaptive step solver and will call `gdefunc` several times to ensure correctness up to pre-specified 
# tolerance levels. As suggested in the original Neural ODE paper and as observed during internal tests, lower tolerances 
# are sufficient for classification tasks 
gde = ODEBlock(odefunc=gdefunc, method='rk4', atol=1e-3, rtol=1e-3, adjoint=True).to(device)


m = nn.Sequential(GCNLayer(g=g, in_feats=num_feats, out_feats=64, activation=F.relu, dropout=0.5),
                  gde,
                  GCNLayer(g=g, in_feats=64, out_feats=n_classes, activation=None, dropout=0.)
                  ).to(device)

