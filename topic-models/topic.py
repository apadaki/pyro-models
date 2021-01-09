import pyro
from pyro.infer import SVI,TraceEnum_ELBO, config_enumerate
import pyro.distributions as dist
import filter
import torch
from pyro.ops.indexing import Vindex
from  torch.distributions import constraints

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

torch.cuda.empty_cache()

device = torch.device(dev)

documentText = filter.parseText('test.csv')

data = torch.tensor(documentText[0][0:60])
#data = torch.tensor(documentText[0][0:20])
data = torch.transpose(data,0,1)
data = data.to(device)
num_words = len(documentText[1])
#num_documents = len(documentText[0])
words_per_doc = documentText[2]


num_topics = 5
#num_words = 20000
num_documents = 60
#words_per_doc = 200

@config_enumerate()
def model(data):
    alpha = torch.ones(num_topics)
    alpha = alpha.to(device)
    eta = torch.ones(num_words)
    eta = eta.to(device)

    with pyro.plate("topic_loop", num_topics):
        beta = pyro.sample("beta",dist.Dirichlet(eta))
        beta = beta.to(device)
        #print(beta)
        #print(beta.shape)

    with pyro.plate("document_loop", num_documents):
        theta = pyro.sample("theta",dist.Dirichlet(alpha))
        theta = theta.to(device)
        #print(theta.shape)
        #print(theta)
        with pyro.plate("word_loop",words_per_doc):
            z = pyro.sample("z",dist.Categorical(theta))
            z= z.to(device)
            #print(z.shape)
            #print(z)
            results = pyro.sample("obs", dist.Categorical(Vindex(beta)[z]), obs=data)
            #print(results.shape)
            return results

def guide(data):
    alpha = pyro.param("alpha", torch.ones(num_documents,num_topics),constraint = constraints.positive)
    alpha = alpha.to(device)
    eta = pyro.param("eta", torch.ones(num_topics,num_words),constraint = constraints.positive)
    eta = eta.to(device)
    #theta1 = pyro.param("theta1",torch.ones(num_topics),constraint = constraints.positive)

    with pyro.plate("topic_loop",num_topics):
        beta = pyro.sample("beta",dist.Dirichlet(eta))
        beta = beta.to(device)

    with pyro.plate("document_loop",num_documents):
        theta = pyro.sample("theta",dist.Dirichlet(alpha))
        theta = theta.to(device)

    #with pyro.plate("second_document_loop",num_documents):
    #    with pyro.plate("word_loop", words_per_doc):
    #        z = pyro.sample("z",dist.Categorical(theta1))

adam_params = {"lr": 0.001, "betas": (0.90, 0.999)}
optimizer = pyro.optim.Adam(adam_params)
elbo = TraceEnum_ELBO(max_plate_nesting=2)

# setup the inference algorithm
pyro.clear_param_store()
svi = SVI(model, guide, optimizer, loss=elbo)

losses = []
n_steps = 100000

#print(model())
#print(data.shape)
#model()

# do gradient steps

for step in range(n_steps):
    loss = svi.step(data)
    losses.append(loss)
    if step % 100 == 0:
       print("Step " + str(step) + ":" + str(loss))
