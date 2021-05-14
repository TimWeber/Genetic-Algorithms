%% Some spring cleaning
clear
clc
close all

%% %% Define variables for the algorithm
N = 50; % gene size
P = 50; % population size


numberOfLoops = 10;
rateGenerator = 6; % Finds the best solution for mutation rate and crossover

% Gene limits
geneLimitMax = 32;
geneLimitMin = -32;


% Define crossover rate
crossRateCapHigh = 0.8; % the maximum % chance of a crossover occuring
crossRateCapLow = 0.02; % lowest % chance of a crossover occuring


% Define mutation rate 
mutationRateCapHigh = 0.08; % maximum % chance of a mutation occuring
mutationRateCapLow = 0.02; % lowest % chance of a mutation occuring

% Mustation alter ammount
muteStepMax = 1;                 % used to calculate how large of a mutation 
muteStepMin = -7.88888888888889; % lower end of calcutation how large of an alteration to make


% Number of generations
generations = 500;

%% Declaration of arrays needed later 
% Array to populate with best and mean fitness
meanFitnessOfPopulation = zeros(1,P);
bestFitnessOfPopulation = zeros(1,P);
averageBest = zeros(1, numberOfLoops);
mute = linspace(mutationRateCapLow, mutationRateCapHigh,rateGenerator);
cross = linspace(crossRateCapLow, crossRateCapHigh, rateGenerator);
mutationRate3Dplot = zeros(rateGenerator, rateGenerator);
crossRate3Dplot = zeros(rateGenerator, rateGenerator);
bestMutation3Dplot = zeros(rateGenerator,rateGenerator);
bestFitnessGraph = zeros(10, generations); % values of best fitnesses
graphMean = zeros(10, generations); % values mean fitness
surfGraph = zeros(10, rateGenerator);
bestMutationRate = zeros(1, rateGenerator);

muteStepRates = linspace(muteStepMin, muteStepMax, 10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate two arrays of random floats for the mutation rate and crossover rate values 
for i = 1:1: rateGenerator
    mutationRate3Dplot(i,:) = mute(i);
    crossRate3Dplot(i,:) = cross; 
end

% use the limits set to fill the mutation and crossover arrays before
% creating a population with those values
for k = 1:1:rateGenerator
    mutationRate = mute(k);
    for bestMutation3Dplot = 1:1:rateGenerator
        crossRate = cross(bestMutation3Dplot);
        population = population_generator(P, N, geneLimitMin, geneLimitMax);

%% Generate offspring,crossover and mutation over number of generations
    for i=1:1:generations
        % population = roulette_generator(P, population);
        population = tournament_generator(P, population);
        population = crossover(P,N,population,crossRate);
        population = mutation(P,N,population,mutationRate,muteStepMin,muteStepMax,geneLimitMin,geneLimitMax);
        
%% Mean fitness of current population
        meanFitnessOfPopulation(i) = sum([population(:).fitness]) / P;
        
%% Overrite best fitness found
        if i == 1
            bestFitness = min([population(:).fitness]) ;
        end
        bestFitnessOfPopulation(i) = min([population(:).fitness]) ;
        if bestFitnessOfPopulation(i) < bestFitness
            bestFitness = bestFitnessOfPopulation(i);
        end
        bestFitnessOfPopulation(i) = bestFitness;
    end
    bestMutationRate(k,bestMutation3Dplot) = bestFitnessOfPopulation(generations);
    end
end

bestMutation = min(bestMutationRate(:));
[row, col] = find(bestMutationRate == bestMutation);
crossRate = cross(col);
mutationRate = mute(row);

% Create a surface plot to visualise the effects of the different crossover
% and mutation rates
figure
s = surf(mutationRate3Dplot, crossRate3Dplot, bestMutationRate, 'FaceAlpha', 0.3, 'FaceLighting', 'gouraud');
s.EdgeColor = 'flat';
colorbar
title('A visulisation of how a change in mutation and corssover rates affect fitness')
xlabel('Mutation Rate')
ylabel('Crossover Rate')
zlabel('Fitness')
hold on
scatter3(mutationRate, crossRate, bestMutation, 'r', 'LineWidth', 1)
hold off

%% Use the best mutation and crossover rate to find the best fitness after a number of rounds
for j=1:1:numberOfLoops
    population = population_generator(P, N, geneLimitMin, geneLimitMax);
    
%% Generate offspring,crossover and mutation over number of generations
    for i=1:1:generations
        % population = roulette_generator(P,population);
        population = tournament_generator(P, population);
        population = crossover(P,N,population,crossRate);
        population = mutation(P,N,population,mutationRate,muteStepMin,muteStepMax,geneLimitMin,geneLimitMax);
        
%% Mean fitness of current population
        meanFitnessOfPopulation(i) = sum([population(:).fitness])/P;
        
%% Overrite best fitness found
        if i == 1
            best = min([population(:).fitness]);
        end
        bestFitnessOfPopulation(i) = min([population(:).fitness]);
        
        if bestFitnessOfPopulation(i) < best
            best = bestFitnessOfPopulation(i);
        end
        
        bestFitnessOfPopulation(i) = best;
    end

    averageBest(j) = bestFitnessOfPopulation(generations);
    bestFitnessGraph(j,:) = bestFitnessOfPopulation;
    graphMean(j,:) = meanFitnessOfPopulation;
end

[bestValue, index] = min(averageBest);

%% Plot graph
X = 1:1:generations;
figure
plot(X, bestFitnessGraph(index,:), X, graphMean(index,:))
title('Graph Showing the best and mean fitness')
legend('Best fitness','Mean fitness')
xlabel('GENERATIONS')
ylabel('FITNESS')
totalAverage = sum(averageBest)/numberOfLoops;
disp(['Average of the ' num2str(numberOfLoops) ' top fitness = ' num2str(totalAverage)])
disp(' ');
disp(['Best fitness after ' num2str(numberOfLoops) ' runs = ' num2str(bestValue)])


%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Population generator
function population = population_generator(P, N, geneLimitMin, geneLimitMax)
    population(P).gene = zeros(1,N);
    population(P).fitness = 0;
    for i=1:1:P
        for j=1:1:N
            population(i).gene(j) = geneLimitMin+rand(1) * (geneLimitMax-geneLimitMin);
        end
        population(i).fitness = 0;
    end
   population = minimisation(P, N, population);
end

%% Tournament_selection
function population = tournament_generator(P,population)
    for i=1:1:P
        parent1 = randi([1 P]);
        parent2 = randi([1 P]);
        if population(parent1).fitness < population(parent2).fitness
            population(i) = population(parent1);
        else
            population(i) = population(parent2);
        end
    end
end
%% Function for the roulette wheel
% function population = roulette_generator(P, population)
% fitnessRoulette = zeros(1, P);
%     for x = 1:1:P
%         fitnessRoulette(x) = 1/population(x).fitness;
%     end
%         totalFitness = sum(fitnessRoulette);
%     for i = 1:1:P
%         selection = 0 + rand(1) * (totalFitness - 0);
%         runningTotal = 0;
%         e = 1;
%         while runningTotal <= selection
%             runningTotal = runningTotal + (1.0 / population(e).fitness);
%             e = e + 1;
%             if e > 50
%                 e = 1;
%             end
%         end
%         if (e-1) == 0
%             e = 51;
%         end
%         population(i) = population(e-1);
%     end
% end

%% Cross over function
function population = crossover(P, N, population, crossRate)
    if randi([1 100])<(crossRate * 100)
        for i=1:2:P
        temp = population(i);
        %crossPoint = randi([1 N]);
        crossPoint1 = randi([1 N/2]);
        crossPoint2 = randi([crossPoint1 N]);
            for j=crossPoint1:1:crossPoint2
             %for j = crossPoint:1:N    
                population(i).gene(j) = population(i+1).gene(j);
                population(i+1).gene(j) = temp.gene(j);
            end
        end
    end
end

%% Mutation function 
function population = mutation(P, N, population, mutationRate, muteStepMax , muteStepMin, geneValueMin, geneValueMax)
    for i=1:1:P
        for j=1:1:N
            alter = muteStepMin + rand(1) * (muteStepMax - muteStepMin);
            if randi([1 100]) < (mutationRate * 100)
                if randi([0 1]) == 1
                    population(i).gene(j) = population(i).gene(j) + alter;
                if population(i).gene(j) > geneValueMax
                  population(i).gene(j) = geneValueMax;
                end
               else
                population(i).gene(j) = population(i).gene(j) - alter;
                    if population(i).gene(j) < geneValueMin
                       population(i).gene(j) = geneValueMin;
                    end
                       
                end
            end
        end
    end
    population = minimisation(P, N, population);
end

%% Minimisation function
function population = minimisation(P, N, population)
    f1 = zeros(1,N);
    f2 = zeros(1,N); 
    for t=1:1:P
        for i =1:1:N
            f1(i) = (population(t).gene(i)^2);
            f2(i) = (cos(2*pi*population(t).gene(i)));
        end
        f3 = sum(f1);
        f4 = sum(f2);
        population(t).fitness = -20*exp(-0.2*sqrt((1/N)*f3))-exp((1/N)*f4);
    end
end
