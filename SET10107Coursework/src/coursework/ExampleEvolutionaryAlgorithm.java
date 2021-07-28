package coursework;

import java.util.ArrayList;
import java.util.Collections;

import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * 
 * You Can Use This Class to implement your EA or implement your own class that
 * extends {@link NeuralNetwork}
 * 
 */

public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {

	/**
	 * The Main Evolutionary Loop
	 */

	@Override
	public void run() {
		// Initialise a population of Individuals with random weights
		population = initialise();

		// Record a copy of the best Individual in the population
		best = getBest();
		System.out.println("Best From Initialisation " + best);

		/**
		 * main EA processing loop
		 */

		while (evaluations < Parameters.maxEvaluations) {
			Individual parent1;
			Individual parent2;
			ArrayList<Individual> children = new ArrayList<>();

			// Choosing a selection operator using the parameters class
			if (Parameters.selection == "tournament") {
				parent1 = tournamentSelection();
				parent2 = tournamentSelection();
			} else if (Parameters.selection == "roulette") {
				parent1 = rouletteSelection();
				parent2 = rouletteSelection();
			} else {
				parent1 = randomSelection();
				parent2 = randomSelection();
			}

			// Choosing a reproduction operator using the parameters class
			if (Parameters.reproduction == "onePointCrossover") {
				children = onePointCrossover(parent1, parent2);
				children = onePointCrossover(parent1, parent2);
			} else if (Parameters.reproduction == "twoPointCrossover") {
				children = twoPointCrossover(parent1, parent2);
				children = twoPointCrossover(parent1, parent2);
			} else if (Parameters.reproduction == "uniform") {
				children = uniform(parent1, parent2);
				children = uniform(parent1, parent2);
			} else {
				System.out.println("Please specify a reproduction parameter.");
			}

			// Choosing a mutation operator using the parameters class
			if (Parameters.mutation == "mutate") {
				mutate(children);
			} else if (Parameters.mutation == "mutateSwap") {
				mutateSwap(children);
			} else if (Parameters.mutation == "mutateRandom") {
				mutateRandom(children);
			} else if (Parameters.mutation == "mutateInversion") {
				mutateInversion(children);
			} else {
				System.out.println("Please specify a mutation parameter.");
			}

			// Evaluate the children
			evaluateIndividuals(children);

			// Replace children in population
			replace(children);

			// check to see if the best has improved
			best = getBest();

			// Implemented in NN class.
			outputStats();

			// Increment number of completed generations
		}

		// save the trained network to disk
		saveNeuralNetwork();
	}

	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */

	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}

	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */

	private Individual getBest() {
		best = null;
		;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}

	/**
	 * Generates a randomly initialised population
	 * 
	 */

	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			// chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		evaluateIndividuals(population);
		return population;
	}

	/**
	 * Selection -- Random: Chooses a random member of the population
	 */

	private Individual randomSelection() {
		Individual parent = population.get(Parameters.random.nextInt(Parameters.popSize));
		return parent.copy();
	}

	/**
	 * Selection -- Tournament: Chooses two random members of population as
	 * candidates and selects the candidate with the best fitness as the parent.
	 */

	private Individual tournamentSelection() { // Pick a specified number of tournament members and compete to find the
												// best

		ArrayList<Individual> candidates = new ArrayList<Individual>();

		for (int i = 0; i < Parameters.tournamentSize; i++) { //
			candidates.add(population.get(Parameters.random.nextInt(Parameters.popSize)));
		}

		Individual candidate = candidates.get(0);
		for (int i = 0; i < Parameters.tournamentSize; i++) {
			if (candidate.fitness > candidates.get(i).fitness) { // candidate is worse than opposition, so replace
				candidate = candidates.get(i);
			}
		}

		return candidate.copy();
	}

	/**
	 * Selection -- Roulette: A 'wheel' where fitter individuals have more chance of
	 * being selected. The algorithm will calculate the sum of all fitness values in
	 * the population.
	 */

	private Individual rouletteSelection() {
		double cumulativeFitness = 0.0;
		for (Individual temp : population) { // Calculate sum of fitness values in population
			cumulativeFitness += temp.fitness;
		}
		// Get random value
		double value = Parameters.random.nextDouble();
		double random = cumulativeFitness * value;
		
		for (int i = 0; i < population.size(); i++) { // Loop until candidate is found
			random = 1/population.get(i).fitness; 
			if (random <  0) { // Could be a good individual if results in a negative number
				return population.get(i);
			}
		}
		// If any errors return the last item in population
		return population.get(population.size() - 1);
	}
	
	/**
	 * Crossover / Reproduction -- One point crossover: A point is randomly selected in the chromosome
	 * as the cut point. Then, combine the genetic material from parents based on the cut point.
	 */

	private ArrayList<Individual> onePointCrossover(Individual parent1, Individual parent2) {
		Individual offspring1 = new Individual();
		Individual offspring2 = new Individual();
		// Select a random cut point
		int cutPoint = Parameters.random.nextInt(parent1.chromosome.length);

		// Generate empty list of children
		ArrayList<Individual> children = new ArrayList<>();
		for (int i = 0; i < parent1.chromosome.length; i++) {
			if (i < cutPoint) { // Give genetic material up to the cut point
				offspring1.chromosome[i] = parent1.chromosome[i];
				offspring2.chromosome[i] = parent2.chromosome[i];
			} else { // Give genetic material when at the cut point from different parent than in
						// first cut point
				offspring1.chromosome[i] = parent2.chromosome[i];
				offspring2.chromosome[i] = parent1.chromosome[i];
			}
		}
		if (offspring1.fitness < offspring2.fitness) {// Tournament to keep the best offspring
			children.add(offspring1);
		} else {
			children.add(offspring2);
		}
		return children;
	}

	/**
	 * Crossover / Reproduction -- Two point crossover: Two cut points are randomly
	 * selected; the genetic material which lands inside of these points is swapped
	 * to produce an offspring
	 */

	private ArrayList<Individual> twoPointCrossover(Individual parent1, Individual parent2) {
		Individual offspring1 = new Individual();
		Individual offspring2 = new Individual();
		// Select two random cut points
		int cutPoint1 = Parameters.random.nextInt(parent1.chromosome.length);
		int cutPoint2 = Parameters.random.nextInt(parent1.chromosome.length);

		// Generate empty list of children
		ArrayList<Individual> children = new ArrayList<>();
		for (int i = 0; i < parent1.chromosome.length; i++) {
			if (i >= cutPoint1 && i <= cutPoint2) { // Exchange genetic material from between the two cut points
				offspring1.chromosome[i] = parent1.chromosome[i];
				offspring2.chromosome[i] = parent2.chromosome[i];
			} else {
				offspring1.chromosome[i] = parent2.chromosome[i];
				offspring2.chromosome[i] = parent1.chromosome[i];
			}
		}
		if (offspring1.fitness < offspring2.fitness) {// Tournament to keep the best offspring
			children.add(offspring1);
		} else {
			children.add(offspring2);
		}
		return children;
	}

	/**
	 * Crossover / Reproduction -- Uniform: Randomly decides which parent the child
	 * will inherit a gene from
	 */

	private ArrayList<Individual> uniform(Individual parent1, Individual parent2) {
		Individual offspring1 = new Individual();
		Individual offspring2 = new Individual();

		// Generate empty list of children
		ArrayList<Individual> children = new ArrayList<>();
		for (int i = 0; i < parent1.chromosome.length; i++) {
			if (Parameters.random.nextBoolean()) { // Use 'next boolean' to replicate a 50% chance of which parent the
													// offspring will inherit each gene from
				offspring1.chromosome[i] = parent1.chromosome[i];
				offspring2.chromosome[i] = parent2.chromosome[i];
			} else {
				offspring1.chromosome[i] = parent2.chromosome[i];
				offspring2.chromosome[i] = parent1.chromosome[i];
			}
		}
		if (offspring1.fitness < offspring2.fitness) {// Tournament to keep the best offspring
			children.add(offspring1);
		} else {
			children.add(offspring2);
		}
		return children;
	}

	/**
	 * Mutation -- mutate:
	 */

	private void mutate(ArrayList<Individual> children) {
		for (Individual child : children) {
			for (int i = 0; i < child.chromosome.length; i++) {
				if (Parameters.random.nextDouble() < Parameters.mutateRate) {
					if (Parameters.random.nextBoolean()) {
						child.chromosome[i] += (Parameters.mutateChange);
					} else {
						child.chromosome[i] -= (Parameters.mutateChange);
					}
				}
			}
		}
	}

	/**
	 * Mutation -- mutateSwap: Picks two random positions in chromosome and swap
	 * them around
	 */

	private void mutateSwap(ArrayList<Individual> children) {
		for (Individual child : children) {
			if (Parameters.random.nextDouble() < Parameters.mutateRate) { // Random chance of a mutation occurring at
																			// random points in chromosome
				int position1 = Parameters.random.nextInt(child.chromosome.length);
				int position2 = Parameters.random.nextInt(child.chromosome.length);

				while (position1 == position2) {// To avoid having the same genes (on, for four days now)
					position2 = Parameters.random.nextInt(child.chromosome.length);
				}

				// Store the first chromosome position for swapping at the end
				double tempPosition1 = child.chromosome[position1];
				child.chromosome[position1] = child.chromosome[position2];
				child.chromosome[position2] = tempPosition1;
			}
		}
	}

	/**
	 * Mutation -- mutateInversion: Picks a random start and end point, and inverse
	 * everything which lies in between these points
	 */

	private void mutateInversion(ArrayList<Individual> children) {
		for (Individual child : children) {
			if (Parameters.random.nextDouble() < Parameters.mutateRate) {
				// Randomly choose a start and end point in the chromosome
				int startPoint = Parameters.random.nextInt(child.chromosome.length);
				int endPoint = Parameters.random.nextInt(child.chromosome.length);

				while (startPoint >= endPoint) { // Ensure start point is never greater than or equal to end point
					startPoint = Parameters.random.nextInt(child.chromosome.length);
					endPoint = Parameters.random.nextInt(child.chromosome.length);
				}
				// The amount of genes selected
				int geneNum = (endPoint - startPoint) + 1;

				double[] chromosomeTemp = new double[geneNum];
				for (int i = 0; i < geneNum; i++) { // Give the temporary array some genes to be swapped
					chromosomeTemp[i] = child.chromosome[i + startPoint];
				}
				// Inverse the selected genes into the child chromosome
				for (int i = 0; i < geneNum; i++) {
					child.chromosome[endPoint - i] = chromosomeTemp[i];
				}
			}
		}
	}

	/**
	 * Function to choose (randomly) between multiple mutation methods on the same
	 * run
	 */

	private void mutateRandom(ArrayList<Individual> children) {
		if (Parameters.random.nextDouble() < Parameters.mutateRate) {
			if (Parameters.random.nextBoolean()) { // Use the swapping algorithm 50% of the time
				mutateSwap(children);
			} else { // Use inversion other half of the time
				mutateInversion(children);
			}
		}
	}

	/**
	 * Replaces the worst member of the population if and only if they have less
	 * fitness than the worst member of the population
	 */

	private void replace(ArrayList<Individual> children) {
		for (Individual child : children) { // For each child
			int worstIndex = getWorstIndex();
			if (child.fitness < population.get(worstIndex).fitness) { // Replace if better than worst
				population.set(worstIndex, child);
			}
		}
	}

	/**
	 * Returns the index of the worst member of the population
	 */

	private int getWorstIndex() {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++) {
			Individual individual = population.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i;
			}
		}
		return idx;
	}

	@Override
	public double activationFunction(double x) {
		if (x < -20.0) {
			return -1.0;
		} else if (x > 20.0) {
			return 1.0;
		}
		return Math.tanh(x);
	}
}