function prospectCertainty(masksLogit, Pr_w_MaksLogits)
% Input the decision alternatives and their probabilities from the arguments


% Define parameters for the value function
alpha = 0.88;

% Define parameter for the probability weighting function
gamma = 0.61;

% Define value function
value_function = @(x) (x >= 0) .* (x .^ alpha);

% Define probability weighting function
probability_weighting = @(p) (p.^gamma) ./ ((p.^gamma + (1 - p).^gamma) .^ (1/gamma));

% Calculate Prospect Certainty Index for each alternative
num_alternatives = length(masksLogit);
PCI = zeros(1, num_alternatives);

for i = 1:num_alternatives
    behaviour = masksLogit(1,i);
    probability = Pr_w_MaksLogits(1,i);
    
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