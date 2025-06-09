import numpy as np
import time
from itertools import permutations

numSpikes = 100
numEvents = numSpikes
similarity_threshold = 0.93
numNeurons = np.random.randint(7,15)
timing_order = np.zeros(((numNeurons,4))) ## Order, in integers 1-4, which tetrodes this neuron is closest to. Allows us to pattern their responses.

## Set timing orders
orders = list(permutations([1,2,3,4])) ## Generate all permutations & then pick random ones for orders.
for i in range(numNeurons):
    rand_order = np.random.randint(0,len(orders))
    timing_order[i] = orders[rand_order]
    orders.pop(rand_order)
    
## Generate spikes
BASE_AMP = 1000 ## pretty arbitrary base value for neuronal peak amplitude
data = np.zeros((numSpikes,4))
neuron_spikes = {} ## contains which neurons caused which spikes to test for accuracy later
for spike in range(numSpikes):
    dim = 0
    neuron = np.random.randint(0,numNeurons)
    if neuron+1 not in neuron_spikes.keys(): ## index dictionary if not present
        neuron_spikes[neuron+1] = [spike]
    else:
        neuron_spikes[neuron+1].append(spike)
    order = timing_order[neuron]
    for tetrode in order:
        data[spike,dim] = tetrode * BASE_AMP + np.random.randint(-750,750)
        dim+=1
    time.sleep(0.1)

def calculateNeuronalVariance(spikes, mean): ## take in all spikes of a given neuron and calculate the variance
    summed_variance = 0
    for spike in spikes:
        summed_variance += np.linalg.norm(spike-mean)

    return summed_variance/len(spikes)

## Testing dot product stuff.
unit_data = np.zeros((numSpikes,4))
## Convert to unit vectors
for i in range(len(data)):
    vector = data[i]/np.linalg.norm(data[i])
    unit_data[i] = vector

## To keep track of if multiple non-overlapping neurons claim the same spike
## i.e. if two neurons say "this was my spike," pick the most similar.
similarities = np.zeros(numSpikes)
results = {'Neuron 1': [0]} ## formatted as shown, each neuron is assigned spikes based on similarity to already existing ones
unassigned_spikes = []

## Checking dot products to see if they are largest when matched with each other (or highly similar vectors)
## This accounts for directionality in the 4D space.
for curr_spike in range(numEvents):
    #if np.linalg.norm(data[curr_spike]) < 100000000: ## Only handle low amplitude spikes first -- avoid groupings of neurons extremely far apart.
    closest_match = similarity_threshold ## anything greater than meets threshold
    likeliest_neuron = ''
    for neuron in results.keys(): ## assign spikes to one of the neurons in results (or add a new one)

        first_spike = results[neuron][0] ## to compare directions
        avg_spike = np.sum(data[results[neuron]], axis=0)/len(results[neuron]) ## to calculate variance
        variance = calculateNeuronalVariance(results[neuron], avg_spike)
        dot_product = np.dot(unit_data[first_spike], unit_data[curr_spike])

        if dot_product > closest_match: ## if this vector is more similar to this tetrode amplitude pattern than the previous closest
            dist = np.linalg.norm(avg_spike-curr_spike)
            if dist < variance:
                closest_match = dot_product
                likeliest_neuron = neuron

    if closest_match == similarity_threshold: ## i.e. this spike has not been assigned to a neuron -- none met the threshold
        ## add the next neuron to the dictionary
        num = len(results) + 1 
        results[f'Neuron {num}'] = [curr_spike]

    else:
        results[likeliest_neuron].append(curr_spike)

print(f'Actual Neuron #: {numNeurons}')
print(f'# Accuracy: {float(np.min([numNeurons,len(results)]/np.max([numNeurons,len(results)]))) * 100}%')
for i in range(len(results.keys())):
    num_correct = 0
    for spike in results[f'Neuron {i+1}']:
        if spike in neuron_spikes[i+1]:
            num_correct += 1
    print(f'Spike Accuracy: {num_correct/len(results[f'Neuron {i+1}'])*100}%')
    print(f'Neuron {i+1}: {results[f'Neuron {i+1}']}')


