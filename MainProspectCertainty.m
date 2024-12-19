%% Main Prospect Certainty

clear
clc
close all

% Define the group masks and the logit
masks = [2, 2.8, 3, 1, 5]; % Example group masks
assumedBehaviours_m=[0, 1.2, 0.7, -1, -1.9];
logit = 2.76; % Example logit
assumedBehaviours_l=1.25;

assumedBehaviours_m_l=[assumedBehaviours_m,assumedBehaviours_l];

Pr_w_MaksLogits = weightedProbability(masks,logit);
prospectCertainty(assumedBehaviours_m_l, Pr_w_MaksLogits);



