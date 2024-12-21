function [best_alternative_idx] = prospectCertainty(Logitmasks, Pr_w_LogitsMasks)

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


%% Input the decision alternatives and their probabilities from the arguments


% Define parameters for the value function
alpha = 0.88;

% Define parameter for the probability weighting function
gamma = 0.61;

% Define value function
value_function = @(x) (x >= 0) .* (x .^ alpha);

% Define probability weighting function
probability_weighting = @(p) (p.^gamma) ./ ((p.^gamma + (1 - p).^gamma) .^ (1/gamma));

% Calculate Prospect Certainty Index for each alternative
num_alternatives = length(Logitmasks);
PCI = zeros(1, num_alternatives);

for i = 1:num_alternatives
    behaviour = Logitmasks(1,i);
    probability = Pr_w_LogitsMasks(1,i);
    
    % Calculate weighted probability and behaviours
    weighted_probability = probability_weighting(probability);
    behaviours = value_function(behaviour);
    
    % Calculate Prospect Certainty Index for the current alternative
    PCI(i) = weighted_probability * behaviours;
end

% Find the alternative with the highest Prospect Certainty Index
[~, best_alternative_idx] = max(PCI);

% Display the results
for i = 1:num_alternatives
    fprintf('Alternative %c: PCI = %.2f\n', char('A' + (i - 1)), PCI(i));
end

fprintf('Best Alternative: %c\n', char('A' + (best_alternative_idx - 1)));

end