import torch 
from pyro import poutine
import numpy as np
from pyro.infer.autoguide.initialization import init_to_sample
from collections import defaultdict
import pyro as pyro
from pyro import plate
from torch.distributions.dirichlet import Dirichlet as dirichlet
import pyro.distributions as dist
from torch.distributions import constraints
from pyro.distributions import Dirichlet, Gamma

from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete

import seaborn as sbn

from csvparser import CSV_Parser

cp = CSV_Parser()

keys, data = cp.get_keys_and_data('data/betterdata.csv')

desired_keys = list(range(3,11))

refined_keys = []
refined_data = []

for key_index in desired_keys:
    refined_keys.append(keys[key_index])

answer_dict = {}
answer_dict['Nothing at all'] = 0
answer_dict['A little'] = 1
answer_dict['A lot'] = 2
answer_dict['Refused'] = -1

for i in range(len(data)):
    refined_data.append([])
    for key_index in desired_keys:
        refined_data[i].append(answer_dict[data[i][key_index]])

valid_data = []
for i in range(len(refined_data)):
    if -1 not in refined_data[i]:
        valid_data.append(refined_data[i])


data_params = {}
data_params['individuals'] = len(valid_data)
data_params['questions'] = len(valid_data[0]) 
data_params['levels'] = 3
data_params['profiles'] = 2


# Here I will wefine all of the constants
# num_people = 10000
num_people = data_params['individuals']
num_questions = data_params['questions']
num_levels = data_params['levels']
num_profiles = data_params['profiles']

# data generation process 
def generate_data(): 
    params = {}
    data_tensor = torch.IntTensor(valid_data)
    params['answers'] = data_tensor
    return params

    
    # # This will contain the "true values"
    # params = {}
    
    # params['alpha_0'] = pyro.sample('alpha_0',dist.Gamma(1,1))
    # xi_dist = Dirichlet(torch.ones(num_profiles))
    # params['xi'] = pyro.sample('xi', xi_dist)
    # alpha = params['xi'] * params['alpha_0']
    
    # #Shape [num_people, num_profiles]
    # #Determines membership of each person to the profiles
    # with plate("g_vals", size = num_people):
    #     params['g'] = pyro.sample('g',Dirichlet(alpha))
    
    # #Here I determine the values for the lambda
    # #Shape [num_profiles, num_questions, num_levels]
    # #First index determines extreme profile 
    # #Second index determines question analized
    # #Third index determines probabilities of levels
    # with plate('question_probabilities', size=num_questions):
    #     with plate('pure_profiles', size=num_profiles):
    #         params['lambdas'] = pyro.sample('lambda', Dirichlet(torch.ones(num_profiles,num_questions,num_levels) * 0.4))

    # #Here I determine the values for the Z
    # #Shape [num_people, num_questions]
    # #First index determines the person to  analyze
    # #Second index determines the profile assumed in question
    # with plate('profile_in_question', size=num_questions):
    #     with plate('profile_assumed_p', size=num_people):
    #         g_altered = torch.ones(num_questions,num_people,num_profiles) * params['g'] #trick to duplicate data
    #         g_altered = g_altered.permute(1,0,2) # Change axis to match what i find inutitive
    #         params['profile_assumed'] = pyro.sample('profiles_assumed',dist.Categorical(g_altered))
            
                                             
    # #There are the survey answers
    # #Shape [num_people,num_questions]
    # #First index represents person who answered the survey
    # #Second index represents number of the question
    # with plate('answers_questions', size=num_questions) as questions: 
    #     with plate('answers_person', size=num_people): 
    #         #Clever indexing and batching tricks 
    #         # I am sorry about this I struggled finding a simpler way to do this
    #         # This is certainly bad style and should be corrected I am just not quite sure how
    #         range_questions = torch.arange(num_questions, dtype=torch.long)
    #         range_questions = torch.ones(num_people,num_questions,dtype=torch.long) * range_questions
    #         effective_lambdas = params['lambdas'][params['profile_assumed'],range_questions]
    #         params['answers'] = pyro.sample('answers', dist.Categorical(effective_lambdas))
        
    return params

trace = poutine.trace(generate_data).get_trace()
print(trace.format_shapes())

params = generate_data()

print("The true parameters are: ")
for param in params: 
    print("-"*20)
    print(param + ":")
    print(params[param].shape)
    print(params[param])


@config_enumerate
def model(data): 
    alpha_0 = pyro.sample('alpha_0',dist.Gamma(2,1))
    xi_dist = Dirichlet(torch.ones(num_profiles))
    xi = pyro.sample('xi', xi_dist)
    alpha = xi * alpha_0
    
    
    
    #Shape [num_people, num_profiles]
    #Determines membership of each person to the profiles
    with plate("g_vals", size = num_people):
        g = pyro.sample('g',Dirichlet(alpha))
    
    #Here I determine the values for the lambda
    #Shape [num_profiles, num_questions, num_levels]
    #First index determines extreme profile 
    #Second index determines question analized
    #Third index determines probabilities of levels
    with plate('question_probabilities', size=num_questions):
        with plate('pure_profiles', size=num_profiles):
            lambdas = pyro.sample('lambda', Dirichlet(torch.ones(num_profiles,num_questions,num_levels)))

    #Here I determine the values for the Z
    #Shape [num_people, num_questions]
    #First index determines the person to  analyze
    #Second index determines the profile assumed in question
    with plate('profile_in_question', size=num_questions):
        with plate('profile_assumed_p', size=num_people):
            g_altered = torch.ones(num_questions,num_people,num_profiles) * g #trick to duplicate data
            g_altered = g_altered.permute(1,0,2) # Change axis to match what i find inutitive
            profile_assumed = pyro.sample('profiles_assumed',dist.Categorical(g_altered))
                                             
    #There are the survey answers
    #Shape [num_people,num_questions]
    #First index represents person who answered the survey
    #Second index represents number of the question
    with plate('answers_questions', size=num_questions) as questions: 
        with plate('answers_person', size=num_people): 
            #Clever indexing and batching tricks 
            # I am sorry about this I struggled finding a simpler way to do this
            # This is certainly bad style and should be corrected I am just not quite sure how
            range_questions = torch.arange(num_questions, dtype=torch.long)
            range_questions = torch.ones(num_people,num_questions,dtype=torch.long) * range_questions
            effective_lambdas = lambdas[profile_assumed,range_questions]
            answers = pyro.sample('answers', dist.Categorical(effective_lambdas), obs=data)

def guide(data): 
    k_gamma = pyro.param('k_gamma', torch.tensor(1.0), constraint=constraints.positive)
    theta_gamma = pyro.param('k_gamma', torch.tensor(1.0), constraint=constraints.positive)
    xi_means = pyro.param('xi_means', torch.ones(num_profiles), constraint=constraints.positive)
    alpha = pyro.param('alpha', torch.ones((num_people,num_profiles)), constraint=constraints.positive) 
    lambdas_val = pyro.param('lambda_vals', torch.ones((num_profiles,num_questions,num_levels)), constraint=constraints.positive)
    
    alpha_0 = pyro.sample('alpha_0', Gamma(k_gamma,theta_gamma))
    xi = pyro.sample('xi', Dirichlet(xi_means))
    with plate("g_vals", size = num_people):
        g = pyro.sample('g',Dirichlet(alpha))

    with plate('question_probabilities', size=num_questions):
        with plate('pure_profiles', size=num_profiles):
            lambdas = pyro.sample('lambda', Dirichlet(lambdas_val))
    
    
# Register hooks to monitor gradient norms.
optim = pyro.optim.Adam({'lr': 0.05, 'betas':[0.09,0.99]})
elbo = TraceEnum_ELBO(max_plate_nesting=2)

pyro.clear_param_store()

svi = SVI(model,guide,optim,loss=elbo)
gradient_norms = defaultdict(list)
for name, value in pyro.get_param_store().named_parameters():
    value.register_hook(lambda gu, name=name: gradient_norms[name].append(gu.norm().item()))

losses = []
for i in range(1200):
    loss = svi.step(params['answers'])
    losses.append(loss)
    if i % 200 == 0: 
        print(loss)
        print(i)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,3), dpi=100).set_facecolor('white')
plt.plot(losses)
plt.xlabel('iters')
plt.ylabel('loss')
plt.title('Convergence of SVI')

plt.savefig('images/losses.png')
print('done!')


from pyro.infer import Predictive 

num_samples = 20000
predictive = Predictive(model, guide=guide,num_samples=num_samples)
samples = {k: v.detach().cpu().numpy()
               for k, v in predictive(params['answers']).items()
               if k != "obs"}

g_inf = samples['g'].mean(0)
g_real = params['g'].detach().numpy()

g_inf2 = g_inf[0]

svm = sbn.heatmap(g_inf2, cmap='vlag')
fig = svm.get_figure()
fig.savefig('images/map1.png')
print('saved!')

svm = sbn.heatmap(params['g'].detach().numpy(),cmap='vlag')
fig = svm.get_figure()
fig.savefig('images/map2.png')

svm = sbn.heatmap(g_inf2 - g_real, cmap='vlag')
fig = svm.get_figure()
fig.savefig('images/map3.png')

g_inf[0] - g_real