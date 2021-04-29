'''
This code creates a very Simple Nested Sampler (SNS), and applies to the examples of a unimodal and a bimodal Gaussian distribution in two dimensions. The nested sampler works, but is extremely inefficient, and to be used for pedagogic purposes. For an efficient nested sampler, I strongly recomment PolyChord (https://github.com/PolyChord/PolyChordLite)

This code was written by Pablo Lemos (UCL)

pablo.lemos.18@ucl.ac.uk

March, 2020
'''

import numpy as np
import pandas as pd
import time
import sys

from getdist import MCSamples
from scipy.special import logsumexp

class Param:
  ''' 
  A class to represent parameters to be sampled

  Attributes
  ----------
  name : str
    Name of the parameter
  label : str
    LaTeX name of the parameter for plotting
  prior_type: str
    Type of prior used for the parameter. Must be 'Uniform' or 'Gaussian'
  prior: np.float(2)
    A tuple with the prior values. If prior_type is Uniform, the numbers 
    represent the minimum and maximum value of the prior respectively. If 
    prior_type is Gaussian, they represent the mean and standard deviation 
    respectively
  '''

  def __init__(self, name, prior_type, prior, label = ''):
    '''
    Parameters
    ----------
    name : str
      Name of the parameter
    prior_type : str
      Type of prior used for the parameter. Must be 'Uniform' or 'Gaussian'
    prior : np.float(2)
      A tuple with the prior values. If prior_type is Uniform, the numbers 
      represent the minimum and maximum value of the prior respectively. If 
      prior_type is Gaussian, they represent the mean and standard deviation 
      respectively
    label : str
      LaTeX name of the parameter for plotting. Defaults to '', in which case it 
      is just the name of the parameter
    '''

    self.name = name
    self.prior_type = prior_type
    self.prior = prior

    if label == '':
      self.label = name
    else: 
      self.label = label

    if (prior_type not in ['Uniform', 'Gaussian']):
      print(
          "ERROR, prior type unknown. Only 'Uniform' or 'Gaussian' can be used")
      sys.exit()

class NestedSampler:
  '''
  The nested sampler class

  Attributes
  ----------
  loglike: function
    The logarithm of the likelihood function
  params: ls[params]
    A list contatining all parameters, the elements belong to the parameter 
    class
  nlive : int
    The number of live points. Should be set to ~25*nparams
  tol: float
    The tolerance, which decides the stopping of the algorithm
  max_nsteps: int
    The maximum number of steps to be used before stopping if the target 
    tolerance is not achieved
  nparams: int
    The number of parameters
  paramnames : ls[str]
    A list containing the names of each parameter
  paramlabels : ls[str]
    A list containing the labels of each parameter
  dead_points: pd.DataFrame
    A pandas dataframe containing all the dead points
  live_points: pd.DataFrame
    A pandas dataframe containing all the live points
  logZ: float
    The logarithm of the evidence
  err_logZ: float
    The estimated error in the logZ calculation
  like_evals: int
    The number of likelihood evaluations

  Methods
  -------
  sample_prior(npoints, initial_step)
    Produce samples from the prior distribution
  '''

  def __init__(
      self, loglike, params, nlive = 50, tol = 0.1, max_nsteps = 10000):

    '''
    Parameters
    ----------
    loglike: function
      The logarithm of the likelihood function
    params: ls[params]
      A list contatining all parameters, the elements belong to the parameter 
      class
    nlive : int
      The number of live points. Should be set to ~25*nparams. Defaults to 50
    tol: float
      The tolerance, which decides the stopping of the algorithm. Defaults to 
      0.1
    max_nsteps: int
      The maximum number of steps to be used before stopping if the target 
      tolerance is not achieved. Defaults to 10,000
    '''

    self.loglike = loglike
    self.params = params
    self.nlive = nlive
    self.tol = tol
    self.max_nsteps = max_nsteps

    self.nparams = len(params)
    self.paramnames = []
    self.paramlabels = []
    self.dead_points = pd.DataFrame()
    self.live_points = pd.DataFrame()
    self.logZ = -1e300 # This is equivalent to starting with Z = 0
    self.err_logZ = -1e300 # This is equivalent to starting with Z = 0
    self.like_evals = 0

  def sample_prior(self, npoints, initial_step = False):
    ''' Produce samples from the prior distributions

    Parameters:
    -----------
    npoints : int
      The number of samples to be produced
    initial_step : bool
      A boolean indicating if this is the initial sampling step. Defaults to 
      False

    Returns: 
    samples: pd.Dataframe
      A pandas dataframe containing the values of the parameters and the 
      corresponding log likelihood, prior probability, and log posterior. The 
      weights are set to unity, as they are calculated in a separate function
    '''
    
    # Create an empty list for the samples
    prior_samples = []

    # Iterate over all parameters
    for param in self.params:
      if initial_step == True: 
        self.paramnames.append(param.name)
        self.paramlabels.append(param.label)
      if param.prior_type == 'Uniform':
        values = np.random.uniform(
            low = param.prior[0], 
            high = param.prior[1], 
            size = npoints)
        
      elif param.prior_type == 'Gaussian':
        values = np.random.normal(
            loc = param.prior[0], 
            scale = param.prior[1], 
            size = npoints)
      
      # Append to the list
      prior_samples.append(values)

    # Transpose and create pandas dataframe
    samples = pd.DataFrame(
        np.asarray(prior_samples).T, columns = self.paramnames)
    
    # Calculate log likelihood
    logL = []
    for index, row in samples.iterrows():
      logL.append(self.loglike(row))

    # Log likelihood
    samples['logL'] = logL 

    # Weights
    samples['weights'] = np.ones(len(logL))

    # Count likelihood evaluations
    self.like_evals += npoints

    return samples

  def get_prior_volume(self, i):
    ''' Calculate the prior volume for a given sample

    Parameters
    ----------
    i : int
      The current iteration
    
    Returns
    -------
    Xi : float
      The corresponding prior volume
    '''

    Xi = np.exp(-i/float(self.nlive))
    return Xi

  def get_weight(self):
    ''' Calculate the weight at a given iteration, calculated as the number of 
    dead points
    

    Returns
    -------
    weight : float
      The sample weight
    '''
    
    iteration = len(self.dead_points.index)
    X_plus = self.get_prior_volume(iteration+2)
    X_minus = self.get_prior_volume(iteration)

    weight = 0.5*(X_minus - X_plus)
    return weight

  def add_logZ_live(self):
    ''' Add the contribution from the live points to the Evidence calculation
    '''
    
    # Find current iteration
    iteration = len(self.dead_points.index)

    # Find the weight as the prior volume
    Xm =  self.get_prior_volume(iteration)*np.ones(self.nlive)

    # get the evidence contribution
    logZ_live = logsumexp(
        self.live_points['logL'] + np.log(Xm)) - np.log(self.nlive)

    # Add to the current evidence
    self.logZ = logsumexp([self.logZ, logZ_live])

  def get_delta_logZ(self):
    ''' Find the maximum contribution to the evidence from the livepoints, 
    used for the stopping criterion

    Returns
    -------
    delta_logZ : float
      The maximum contribution to the evidence from the live points
    '''

    # Find index with minimun likelihood
    max_logL = max(self.live_points['logL'])

    # Find current iteration
    iteration = len(self.dead_points.index)

    # Get prior volume
    Xi =  self.get_prior_volume(iteration)

    # Get delta_logZ as log(Xi*L)
    delta_logZ = np.log(Xi) + max_logL 

    return delta_logZ

  def find_new_sample(self, min_like):
    ''' Sample the prior until finding a sample with higher likelihood than a 
    given value

    Parameters
    ----------
      min_like : float
        The threshold log-likelihood

    Returns
    -------
      newsample : pd.DataFrame
        A new sample
    '''

    newlike = -np.infty
    while newlike < min_like:
      newsample = self.sample_prior(npoints = 1)
      newlike = newsample['logL'][0]

    return newsample

  def move_one_step(self):
    ''' Find highest log like, get rid of that point, and sample a new one '''

    # Find index with minimun likelihood
    min_index = self.live_points['logL'].idxmin()

    # Store value of likelihood at that point
    min_like = self.live_points['logL'][min_index]   

    # "Kill" that sample
    min_sample = self.live_points.iloc[min_index]
    min_sample['weights'] = self.get_weight()
    self.dead_points = self.dead_points.append(
        min_sample, ignore_index=True, sort=True)
    self.live_points = self.live_points.drop(min_index)

    # Add to the log evidence
    self.logZ = logsumexp(
        [self.logZ, min_sample['logL'] + np.log(min_sample['weights'])])

    # Add a new sample   
    newsample = self.find_new_sample(min_like)
    self.live_points = self.live_points.append(
        newsample, ignore_index=True, sort=True)

  def terminate(self, run_time):
    ''' Terminates the algorithm by adding the final live points to the dead
    points, calculating the final log evidence and acceptance rate, and printing
    a message

    Parameters
    ----------
    run_time: float
      The time taken for the algorithm to finish, in seconds
    '''
    
    weights_live = self.get_weight()
    self.live_points['weights'] = weights_live

    self.dead_points = self.dead_points.append(
        self.live_points, ignore_index=True, sort=True)
    
    # Add the contribution from the live points to the evidence
    self.add_logZ_live()

    # Convert the prior weights to posterior weights
    self.dead_points['weights'] *= np.exp(self.dead_points['logL']-self.logZ)

    acc_rate = len(self.dead_points)/float(self.like_evals)

    print('---------------------------------------------')
    print('Nested Sampling completed')
    print('Run time =', run_time, 'seconds')
    print('Acceptance rate =', acc_rate)
    print('logZ =', self.logZ, '+/-', self.err_logZ)
    print('---------------------------------------------')


  def run(self):
    ''' The main function of the algorithm. Runs the Nested sampler'''

    start_time = time.time()

    # Generate live points
    self.live_points = self.sample_prior(npoints=self.nlive, initial_step=True)

    # Run the algorithm
    nsteps = 0
    epsilon = np.infty
    while (nsteps < self.max_nsteps and epsilon > self.tol): 
      self.move_one_step()
      delta_logZ = self.get_delta_logZ()
      epsilon = np.exp(delta_logZ -self.logZ)

      if nsteps%100 == 0:
        print(nsteps, 'completed, logZ =', self.logZ, ', epsilon =', epsilon)
      nsteps +=1 

    if nsteps == self.max_nsteps:
      print('WARNING: Target tolerance was not achieved after', nsteps, 
            'steps. Increase max_nsteps')
    
    # For now I am using this simplified calculation of the error in logZ
    self.err_logZ = epsilon * abs(self.logZ)
      
    run_time = time.time() - start_time
      
    self.terminate(run_time)

  def convert_to_getdist(self):
    ''' Converts the output of the algorithm to a Getdist samples object, for
    plotting the posterior. 

    Returns
    -------
    getdist_samples
      A getdist samples objects with the samples
    '''

    samples = np.asarray(self.dead_points[self.paramnames])
    weights = np.asarray(self.dead_points['weights'])
    loglikes = np.asarray(self.dead_points['logL'])

    getdist_samples = MCSamples(
        samples=samples ,
        weights=weights, 
        loglikes = loglikes,
        names = self.paramnames, 
        labels = self.paramlabels)
    
    return getdist_samples
