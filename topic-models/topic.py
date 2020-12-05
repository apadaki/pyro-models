import pyro
from pyro.infer import SVI,TraceEnum_ELBO
import pyro.distributions as dist
import filter
import torch
from  torch.distributions import constraints

num_topics = 5
num_documents = 100
num_words = 200

documentText = filter.parseText('test.csv')

#data = torch.tensor(documentText[0])
data = torch.tensor(documentText[0][0:10])
num_words = len(documentText[1])
num_documents = len(documentText[0])
words_per_doc = documentText[2]

def model(data):
    alpha = torch.ones(num_topics)
    eta = torch.ones(num_words)
    with pyro.plate("topic_loop", num_topics):
        beta = pyro.sample("beta",dist.Dirichlet(eta))

    with pyro.plate("document_loop", num_documents):
        theta = pyro.sample("theta",dist.Dirichlet(alpha))
        with pyro.plate("word_loop", words_per_doc):
            z = pyro.sample("z",dist.Categorical(theta))
            pyro.sample("obs", dist.Categorical(beta[z]), obs=data)

def guide(data):
    alpha = pyro.param("alpha", torch.ones(num_documents,num_topics),constraint = constraints.positive)
    eta = pyro.param("eta", torch.ones(num_topics,num_words),constraint = constraints.positive)
    theta1 = pyro.param("theta1",torch.ones(num_topics),constraint = constraints.positive)

    with pyro.plate("topic_loop",num_topics):
        beta = pyro.sample("beta",dist.Dirichlet(eta))

    with pyro.plate("document_loop",num_documents):
        theta = pyro.sample("theta",dist.Dirichlet(alpha))

    with pyro.plate("second_document_loop",num_documents):
        with pyro.plate("word_loop", words_per_doc):
            z = pyro.sample("z",dist.Categorical(theta1))

adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = pyro.optim.Adam(adam_params)
elbo = TraceEnum_ELBO(max_plate_nesting=2)

# setup the inference algorithm
pyro.clear_param_store()
svi = SVI(model, guide, optimizer, loss=elbo)

losses = []
n_steps = 5000
# do gradient steps
for step in range(n_steps):
    loss = svi.step(data)
    losses.append(loss)
    if step % 100 == 0:
        print(loss)
        print(i)
