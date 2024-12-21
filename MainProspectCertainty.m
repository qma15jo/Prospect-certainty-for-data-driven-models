function [best_alternative_idx_vector] = MainProspectCertainty(Ologit, Omasks)

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


%% Main Prospect Certainty

best_alternative_idx_vector = []; % to store the best alternatives.

% to calculate the PC for each sample
for i= 1:size(Ologit,2)

% Define the group masks and the logit
masks = Omasks(:,i); % Example group masks
masks = masks';

assumedBehaviours_m=ones(size(masks)); % this value is assumed here, while in
% real examples should be taken from Wasserstein Distance function to test
% the previous output training dataset distribution with the previous
% outputs distribution, with that after adding the node value.

logit = Ologit(1,i); % for one sample
assumedBehaviours_l=1;

assumedBehaviours_l_m=[assumedBehaviours_l,assumedBehaviours_m];

Pr_w_LogitsMaks = weightedProbability(logit,masks);
best_alternative_idx_vector = [best_alternative_idx_vector, prospectCertainty(assumedBehaviours_l_m, Pr_w_LogitsMaks)];

end

end