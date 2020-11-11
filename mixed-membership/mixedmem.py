import torch
import pyro
import os

def data_generation(K, N, J, n):
    var_param_1 = 1
    var_param_2 = 2
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

        for j in pyro.plate("question_loop", J):
            if not lambda_defined:
                for k in pyro.plate("profile_loop", K):
                    lam[j][k] = pyro.sample("lambda_data_j{}k{}".format(j, k), pyro.distributions.Dirichlet(torch.ones(n[j])))
            # sample z_ij from discrete distribution
            z[i][j] = pyro.sample("zdata_{}{}".format(i, j), pyro.distributions.Categorical(g[i]))
            y[i][j] = pyro.sample("ydata_{}{}".format(i, j), pyro.distributions.Categorical(lam[j][z[i][j]]))
        lambda_defined=True
    return g, z, y, lam

g, z, y, lam = data_generation(2, 5, 4, [3, 2, 1, 4])
data = y # data is an NxJ matrix corresponding to an answer for a given question for a given individual

print(data)
    

def model(K, N, J, n, data):
    var_param_1 = 1
    var_param_2 = 2
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

        for j in pyro.plate("question_loop", J):
            if not lambda_defined:
                for k in pyro.plate("profile_loop", K):
                    lam[j][k] = pyro.sample("lambda_data_j{}k{}".format(j, k), pyro.distributions.Dirichlet(torch.ones(n[j])))
            # sample z_ij from discrete distribution
            z[i][j] = pyro.sample("zdata_{}{}".format(i, j), pyro.distributions.Categorical(g[i]))
            y[i][j] = pyro.sample("ydata_{}{}".format(i, j), pyro.distributions.Categorical(lam[j][z[i][j]]), obs=data[i][j])
        lambda_defined=True
    return y
    
print("\n"+str(model(2, 5, 4, [3, 2, 1, 4], data)))