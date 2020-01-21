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

opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=0.001)
criterion = torch.nn.MSELoss()
logger = PerformanceContainer(data={'train_loss':[], 'train_accuracy':[],
                                   'test_loss':[], 'test_accuracy':[],
                                   'forward_time':[], 'backward_time':[],
                                   'nfe': []})
steps = 400
verbose_step = 50
num_grad_steps = 0

for i in range(steps): # looping over epochs
    m.train()
    start_time = time.time()
    outputs = m(X)
    f_time = time.time() - start_time

    nfe = m._modules['1'].odefunc.nfe
    y_pred = outputs

    loss = criterion(y_pred, Y)
    opt.zero_grad()
    
    start_time = time.time()
    loss.backward()
    b_time = time.time() - start_time
    
    opt.step()
    num_grad_steps += 1

    with torch.no_grad():
        m.eval()

        # calculating outputs again with zeroed dropout
        y_pred = m(X)
        m._modules['1'].odefunc.nfe = 0

        train_loss = loss.item()
        train_acc = accuracy(y_pred, Y).item()
        test_acc = accuracy(y_pred, Y).item()
        test_loss = criterion(y_pred, Y).item()
        logger.deep_update(logger.data, dict(train_loss=[train_loss], train_accuracy=[train_acc],
                           test_loss=[test_loss], test_accuracy=[test_acc], nfe=[nfe], forward_time=[f_time],
                           backward_time=[b_time]))

    if num_grad_steps % verbose_step == 0:
        print('[{}], Loss: {:3.3f}, Train Accuracy: {:3.3f}, Test Accuracy: {:3.3f}, NFE: {}'.format(num_grad_steps,
                                                                                                    train_loss,
                                                                                                    train_acc,
                                                                                                    test_acc,
                                                                                                    nfe))
