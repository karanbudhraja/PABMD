function evaluation(abm)
% get abm as argument

% constants
EVALUATION_PREFIX = 'evaluation_distance_';
EVALUATION_SUFFIX = '_cross_validation.txt';
EVALUATION_SUFFIX_RANDOM = '_cross_validation_random_baseline.txt';

% get data
randomData = importdata(strcat('random/', EVALUATION_PREFIX, abm, EVALUATION_SUFFIX_RANDOM));
withSamplingDatasetSelectionPruningData = importdata(strcat('with_sampling_datasetselection_pruning/', EVALUATION_PREFIX, abm, EVALUATION_SUFFIX));
withoutSamplingData = importdata(strcat('without_sampling/', EVALUATION_PREFIX, abm, EVALUATION_SUFFIX));
withoutDatasetSelectionData = importdata(strcat('without_datasetselection/', EVALUATION_PREFIX, abm, EVALUATION_SUFFIX));
withoutPruningData = importdata(strcat('without_pruning/', EVALUATION_PREFIX, abm, EVALUATION_SUFFIX));
allPlusOutlierDetectionData = importdata(strcat('../all_with_od/', EVALUATION_PREFIX, abm, EVALUATION_SUFFIX));

% compile data
% all experiments have the same size
allData = [randomData; withSamplingDatasetSelectionPruningData; withoutSamplingData; withoutDatasetSelectionData; withoutPruningData; allPlusOutlierDetectionData];
experimentSize = size(randomData, 1);
samplingPresent = [zeros(experimentSize, 1); ones(experimentSize, 1); zeros(experimentSize, 1); ones(experimentSize, 1); ones(experimentSize, 1); ones(experimentSize, 1)];
datasetSelectionPresent = [zeros(experimentSize, 1); ones(experimentSize, 1); ones(experimentSize, 1); zeros(experimentSize, 1); ones(experimentSize, 1); ones(experimentSize, 1)];
pruningPresent = [zeros(experimentSize, 1); ones(experimentSize, 1); ones(experimentSize, 1); ones(experimentSize, 1); zeros(experimentSize, 1); ones(experimentSize, 1)];
outlierDetectionPresent = [zeros(experimentSize, 1); zeros(experimentSize, 1); zeros(experimentSize, 1); zeros(experimentSize, 1); zeros(experimentSize, 1); ones(experimentSize, 1)];

% perform n-way anova
p = anovan(allData, {samplingPresent, datasetSelectionPresent, pruningPresent, outlierDetectionPresent});

% clean
clear all;

end