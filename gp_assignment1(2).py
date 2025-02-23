import numpy as np
import random 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, dimensions, learningRate=0.01):
        self.weights = [random.uniform(-1, 1) for _ in range(dimensions)]
        self.threshold = random.uniform(0, 1)
        self.learningRate = learningRate

    def classify(self, inputs):
        weightedSum = sum(w*x for w, x in zip(self.weights, inputs))
        return 1 if weightedSum > self.threshold else 0

    def fitness(self, trainingSet):
        errors = []
        for inputs, label in trainingSet:
            output = self.classify(inputs)
            error = (output - label) ** 2  
            errors.append(error)
        return 1 / (1 + np.mean(errors))

def initiliasePopulation(populationSize, dimensions):
    return [Perceptron(dimensions) for _ in range(populationSize)]
            

def evaluatePopulation(population, trainingSet):
    fitnessValues = []
    for perceptron in population:
        fitness = perceptron.fitness(trainingSet)
        fitnessValues.append((perceptron, fitness))
    return fitnessValues



def elitismSelection(fitnessValues, elitismRate):
    fitnessValues.sort(key=lambda x: x[1], reverse=True)
    numElites = int(elitismRate * len(fitnessValues))
    elites = [model for model, fitness in fitnessValues[:numElites]]
    return elites


def crossoverFunction(parent1, parent2):
    crossoverPoint = random.randint(0, len(parent1.weights) - 1)
    childWeights = parent1.weights[:crossoverPoint] + parent2.weights[crossoverPoint:]
    childPerceptron = Perceptron(len(childWeights))
    childPerceptron.weights = childWeights
    childPerceptron.threshold = parent1.threshold
    return childPerceptron

def mutationFunction(perceptron, mutationRate):
    mutatedWeights = [weight + random.uniform(-0.1, 0.1) if random.random() < mutationRate else weight
                      for weight in perceptron.weights]
    mutatedThreshold = perceptron.threshold + random.uniform(-0.1, 0.1) if random.random() < mutationRate else perceptron.threshold
    mutatedPerceptron = Perceptron(len(mutatedWeights))
    mutatedPerceptron.weights = mutatedWeights
    mutatedPerceptron.threshold = mutatedThreshold
    return mutatedPerceptron


def createNewPopulation(currentPopulation, populationSize, mutationRate, crossoverRate, copyRate, elitismRate, trainingSet):
    fitnessValues = evaluatePopulation(currentPopulation, trainingSet)
    
    elites = elitismSelection(fitnessValues, elitismRate)
    newPopulation = elites.copy()
    
    numCopies = int(copyRate * populationSize) - len(elites)
    newPopulation.extend(random.choices(elites, k=numCopies))
    
    numCrossovers = int(crossoverRate * populationSize)
    for _ in range(numCrossovers):
        parents = random.sample(elites, 2)
        childPerceptron = crossoverFunction(parents[0], parents[1])
        newPopulation.append(childPerceptron)
    
    numMutants = int(mutationRate * populationSize)
    mutants = random.sample(elites, numMutants)
    for mutant in mutants:
        mutatedPerceptron = mutationFunction(mutant, mutationRate)
        newPopulation.append(mutatedPerceptron)
    
    return newPopulation

def testPerceptron(perceptron, testSet):
    predictions = []
    
    for inputs, label in testSet:
        output = perceptron.classify(inputs)
        predictions.append((inputs, label, output))
    
    return predictions

populationSize = 500
mutationRate = 0.1
crossoverRate = 0.2
copyRate = 0.7
generations = 100
desiredAccuracy = 0.95
elitismRate = 0.1

data = pd.read_csv('gp-training-set.csv')
dimensions = data.shape[1] - 1  
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values
dataset = list(zip(features, labels))

trainingSet, testSet = train_test_split(dataset, test_size=0.2, random_state=42)

initialPopulation = initiliasePopulation(populationSize, dimensions)

for generation in range(generations):
    initialPopulation = createNewPopulation(initialPopulation, populationSize, mutationRate, crossoverRate, copyRate, elitismRate, trainingSet)
    fitnessValues = evaluatePopulation(initialPopulation, trainingSet)
    maxFitness = max(fitnessValues, key=lambda x: x[1])[1]
    print(f"Generation {generation}, Max Fitness: {maxFitness}")

    if maxFitness >= desiredAccuracy:
        break

bestPerceptron = fitnessValues[0][0]

testResults = testPerceptron(bestPerceptron, testSet)
for inputs, actualLabel, predictedLabel in testResults:
    print(f"Inputs: {inputs}, Actual Label: {actualLabel}, Predicted Label: {predictedLabel}")

accuracy = sum(1 for _, actual, predicted in testResults if actual == predicted) / len(testResults)
print(f"Test Set Accuracy: {accuracy:.2f}")
print(f"Best Perceptron Weights: {bestPerceptron.weights}")
print(f"Best Perceptron Threshold: {bestPerceptron.threshold}")
print(f"Best Perceptron Accuracy: {accuracy:.2f}")