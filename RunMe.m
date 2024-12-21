% Kindly follow the latest Release... This code was developed from scratch
% to illustrate the Prospect Certainty method for data-driven models. It
% features a simple Multi-Layer Perceptron (MLP) with a randomly generated
% dataset. The final results reflect the model's simplicity and the
% dataset's lack of coherence. However, this code is intended solely to
% facilitate the reproducibility of the method.
% 
% If you utilize this code, please cite the following paper:
% 
% Qais Yousef, Pu Li. Prospect certainty for data-driven models, 29 March
% 2024, PREPRINT (Version 1) available at Research Square
% [https://doi.org/10.21203/rs.3.rs-4114659/v1]
% 
% Additionally, please note that a comprehensive, tested package will be
% released soon.
% 
% Qais Yousef 
% 21.12.2024

clc
clear
close all
%% Initialize prospect certainty parameters

% select hidden_size for the hidden layer
hidden_size = 9;

% select output_masks_size for the output logit
output_masks_size = 3;

% select masks_ratio for the connection between the masks and the nodes of
% the hidden layer. Between 0 and 1.
masks_ratio = 0.5;

%% Call the MPL function that is supplied with the prospect certainty layer

MLP_ProspectCertainty(hidden_size, output_masks_size, masks_ratio)