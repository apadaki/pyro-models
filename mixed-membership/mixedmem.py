import torch
import torch.distributions.constraints as constraints
import pyro
import os
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import matplotlib.pyplot as plt

def data_generation(K, N, J, n):
    var_param_1 = 2
    var_param_2 = 3
    var_param_3 = [[None] * K for _ in range(J)]
    for j in range(J):
        for k in range(K):
            var_param_3[j][k] = torch.ones(n[j])

    alpha0 = pyro.sample('alpha0', pyro.distributions.Gamma(var_param_1, var_param_2))
    xi = pyro.sample('xi', pyro.distributions.Dirichlet(torch.ones(K)));

    alpha = alpha0 * xi

    g = [None] * N
    z = [[None] * J for _ in range(N)]
    y = [[None] * J for _ in range(N)]
    lam = [[None] * K for _ in range(J)]

    lambda_defined = False
    for i in pyro.plate("membership_loop", N):
        # sample g_i from dirichlet distribution
        g[i] = pyro.sample("gdata_{}".format(i), pyro.distributions.Dirichlet(alpha))

        for j in pyro.plate("question_loop_{}".format(i), J):
            if not lambda_defined:
                for k in pyro.plate("profile_loop_{}".format(j), K):
                    lam[j][k] = pyro.sample("lambda_data_j{}k{}".format(j, k), pyro.distributions.Dirichlet(var_param_3[j][k]))
            # sample z_ij from discrete distribution
            z[i][j] = pyro.sample("zdata_{}{}".format(i, j), pyro.distributions.Categorical(g[i]))
            y[i][j] = pyro.sample("ydata_{}{}".format(i, j), pyro.distributions.Categorical(lam[j][z[i][j]]))
        lambda_defined=True
    return g, z, y, lam

CLASSES = 2
INDIVIDUALS = 5
QUESTIONS = 4
NUM_CHOICES = [3, 2, 3, 4] # jth element = number of answer choices for question j
g, z, y, lam = data_generation(CLASSES, INDIVIDUALS, QUESTIONS, NUM_CHOICES)
data = y # data is an NxJ matrix corresponding to an answer for a given question for a given individual

print(data)
    

def model(K, N, J, n, data):
    var_param_1 = 2
    var_param_2 = 3
    var_param_3 = [[None] * K for _ in range(J)]
    for j in range(J):
        for k in range(K):
            var_param_3[j][k] = torch.ones(n[j])

    alpha0 = pyro.sample('alpha0', pyro.distributions.Gamma(var_param_1, var_param_2))
    xi = pyro.sample('xi', pyro.distributions.Dirichlet(torch.ones(K)))

    alpha = alpha0 * xi

    g = [None] * N
    z = [[None] * J for _ in range(N)]
    y = [[None] * J for _ in range(N)]
    lam = [[None] * K for _ in range(J)]

    lambda_defined = False
    for i in pyro.plate("membership_loop", N):
        # sample g_i from dirichlet distribution
        g[i] = pyro.sample("gdata_{}".format(i), pyro.distributions.Dirichlet(alpha))

        for j in pyro.plate("question_loop_{}".format(i), J):
            if not lambda_defined:
                for k in pyro.plate("profile_loop_{}".format(j), K):
                    lam[j][k] = pyro.sample("lambda_data_j{}k{}".format(j, k), pyro.distributions.Dirichlet(var_param_3[j][k]))
            # sample z_ij from discrete distribution
            z[i][j] = pyro.sample("zdata_{}{}".format(i, j), pyro.distributions.Categorical(g[i]))
            y[i][j] = pyro.sample("ydata_{}{}".format(i, j), pyro.distributions.Categorical(lam[j][z[i][j]]), obs=data[i][j])
        lambda_defined=True
    return y

def guide(K, N, J, n, data):
    var_param_1 = pyro.param("var_param_1", torch.tensor(2.5), constraint = constraints.positive) # Finalize initial guesses later 
    var_param_2 = pyro.param("var_param_2", torch.tensor(3.5), constraint = constraints.positive) 
    var_param_3 = [[None] * K for _ in range(J)]
    var_param_4 = pyro.param("var_param_4", torch.ones(K), constraint = constraints.positive)
    for j in range(J):
        for k in range(K):
            var_param_3[j][k] = pyro.param("var_param_3_j{}k{}".format(j, k), torch.ones(n[j]), constraints.positive)
    
    alpha0 = pyro.sample("alpha0", pyro.distributions.Gamma(var_param_1, var_param_2))
    xi = pyro.sample("xi", pyro.distributions.Dirichlet(var_param_4))

    for j in range(J):
        for k in range(K):
            pyro.sample("lambda_data_j{}k{}".format(j, k), pyro.distributions.Dirichlet(var_param_3[j][k]))

    for i in pyro.plate("membership_loop", N):
        # sample g_i from dirichlet distribution
        g[i] = pyro.sample("gdata_{}".format(i), pyro.distributions.Dirichlet(alpha0*xi))

        for j in pyro.plate("question_loop_{}".format(i), J):
            # sample z_ij from discrete distribution
            z[i][j] = pyro.sample("zdata_{}{}".format(i, j), pyro.distributions.Categorical(g[i]))
                
# print("\n"+str(model(2, 5, 4, [3, 2, 1, 4], data)))

# set up the optimizer
adam_params = {"lr": 0.01, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

n_steps = 2500

ELBO_CHANGES = []

# do gradient steps
for step in range(n_steps):
    ELBO = str(svi.step(CLASSES, INDIVIDUALS, QUESTIONS, NUM_CHOICES, data))
    if step % 10 == 0:
        print('n =', step)
        ELBO_CHANGES.append(ELBO)
        # print(ELBO)
        print(pyro.param('var_param_1').item())
        print(pyro.param('var_param_2').item())


print("XI: " + str(pyro.param('var_param_4')))

# plt.plot(ELBO_CHANGES)
# plt.title("ELBO")
# plt.ylabel("step")
# plt.xlabel("n");



var_param_1 = pyro.param("var_param_1").item()
var_param_2 = pyro.param("var_param_2").item()
var_param_3 = [[None] * CLASSES for _ in range(QUESTIONS)]
var_param_4 = pyro.param("var_param_4")

for j in range(QUESTIONS):
    for k in range(CLASSES):
        var_param_3[j][k] = pyro.param("var_param_3_j{}k{}".format(j, k))

print(str(var_param_1) + '\n')

print(str(var_param_2) + '\n')

print(str(var_param_3) + '\n')