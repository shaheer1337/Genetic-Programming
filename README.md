# Perceptron Training using Genetic Programming

This project demonstrates the training of a Perceptron model using Genetic Programming to label a dataset. The code utilizes various genetic programming techniques such as selection, crossover, and mutation to evolve a population of perceptrons over several generations to achieve high classification accuracy.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

You can install the required packages using the following command:

```bash
pip install numpy pandas matplotlib scikit-learn
```

No skeleton code was used in this implementation

# Code Explanation
## Perceptron Class

The Perceptron class represents a simple perceptron model with the following methods:

```python
def __init__(self, dimensions, learningRate=0.01):
        self.weights = [random.uniform(-1, 1) for _ in range(dimensions)]
        self.threshold = random.uniform(0, 1)
        self.learningRate = learningRate 
```
Initializes the perceptron with random weights, a random threshold, and a learning rate.

```python
def classify(self, inputs):
        weightedSum = sum(w*x for w, x in zip(self.weights, inputs))
        return 1 if weightedSum > self.threshold else 0
```
Classifies the input data based on the weighted sum of inputs and the threshold.

```python
def fitness(self, trainingSet):
        errors = []
        for inputs, label in trainingSet:
            output = self.classify(inputs)
            error = (output - label) ** 2  
            errors.append(error)
        return 1 / (1 + np.mean(errors))
```

Calculates the fitness of the perceptron based on the mean squared error over the training set.

# Genetic Programming Functions

```python
def initiliasePopulation(populationSize, dimensions):
    return [Perceptron(dimensions) for _ in range(populationSize)]

```
Initializes a population of perceptrons with random weights.

```python
def evaluatePopulation(population, trainingSet):
    fitnessValues = []
    for perceptron in population:
        fitness = perceptron.fitness(trainingSet)
        fitnessValues.append((perceptron, fitness))
    return fitnessValues
```

Evaluates the fitness of each perceptron in the population based on their performance on the training set.

```python
def elitismSelection(fitnessValues, elitismRate):
    fitnessValues.sort(key=lambda x: x[1], reverse=True)
    numElites = int(elitismRate * len(fitnessValues))
    elites = [model for model, fitness in fitnessValues[:numElites]]
    return elites
```

Selects the top-performing perceptrons based on their fitness values, ensuring that the best models are carried over to the next generation.

```python
def crossoverFunction(parent1, parent2):
    crossoverPoint = random.randint(0, len(parent1.weights) - 1)
    childWeights = parent1.weights[:crossoverPoint] + parent2.weights[crossoverPoint:]
    childPerceptron = Perceptron(len(childWeights))
    childPerceptron.weights = childWeights
    childPerceptron.threshold = parent1.threshold
    return childPerceptron
```
Performs crossover between two parent perceptrons to create a child perceptron by combining their weights.

```python
def mutationFunction(perceptron, mutationRate):
    mutatedWeights = [weight + random.uniform(-0.1, 0.1) if random.random() < mutationRate else weight
                      for weight in perceptron.weights]
    mutatedThreshold = perceptron.threshold + random.uniform(-0.1, 0.1) if random.random() < mutationRate else perceptron.threshold
    mutatedPerceptron = Perceptron(len(mutatedWeights))
    mutatedPerceptron.weights = mutatedWeights
    mutatedPerceptron.threshold = mutatedThreshold
    return mutatedPerceptron
```

Mutates the weights and threshold of a perceptron based on a specified mutation rate, introducing randomness and variation.

```python
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
```
Creates a new population of perceptrons using elitism, copying, crossover, and mutation. This function evolves the population over generations to improve overall fitness.

```python
def testPerceptron(perceptron, testSet):
    predictions = []
    
    for inputs, label in testSet:
        output = perceptron.classify(inputs)
        predictions.append((inputs, label, output))
    
    return predictions
```
Tests a perceptron on the test set and returns the classification results.

# Training and Testing

The dataset is loaded from gp-training-set.csv and split into training and testing sets.

The initial population of perceptrons is generated and evolved over several generations.

The best perceptron is selected based on fitness and evaluated on the test set.

# Output
The weights, threshold, and accuracy of the best perceptron are printed.

The classification results of the best perceptron on the test set are displayed.

# How to Run
Ensure you have the required packages installed.

Place the dataset file (gp-training-set.csv) in the same directory as the code.

Run the script using Python:

```bash
python gp-assignment1(2).py
```
