function Pr_w_MaksLogits = weightedProbability(logit,masks)

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



%% Data preparation

logitMasks=[logit,masks];
occurances=zeros(1,size(logitMasks,2));
for i=1:size(logitMasks,2)
    for ii=1:size(logitMasks,2)
        if logitMasks(1,i)==logitMasks(1,ii)
            occurances(1,i)=occurances(1,i)+1;
        end
    end
end
k = exp(1) + 1e-5; % Combined constant
bias = 1e-3; % Small bias term to ensure the logit's weight is slightly greater

% Calculate the mean and standard deviation of the group masks
mu = mean(masks);
sigma = std(masks) + 1e-5; % Add a small constant to avoid NaN

% Initialize arrays to store weights
w_masks = zeros(size(masks));
w_logit = 0;

% Calculate the initial weights for the masks
for i = 1:length(masks)
    d_i = abs(masks(i) - mu);
    w_masks(i) = 1 / (log(d_i + k + bias));
end

% Calculate the initial weight for the logit
d_u = abs(logit - mu);
w_logit = 1 / (log(d_u + k)); % Add bias directly

% Ensure logit weight is slightly greater than sample at mean
if logit == mu && any(masks == mu)
    w_logit = w_logit + 0.01; % Slightly increase logit weight
end

% Normalize the weights
S = sum(w_masks) + w_logit;
w_i = w_masks / S;
w_u = w_logit / S;

% % Display the weights
% disp('Weights for each group sample:');
% disp(w_i);
% disp('Weight for the logit:');
% disp(w_u);

%% Weighted probability
Pr_w_masks=w_i.*occurances(1,1:end-1);
Pr_w_logit=w_u*occurances(end);

% Normalize the weighted probability
S = sum(Pr_w_masks) + Pr_w_logit;
Pr_w_masks = Pr_w_masks / S;
Pr_w_logit = Pr_w_logit / S;

% Display the results
disp('Weighted probability for the masks:');
disp(Pr_w_masks);
disp('Weight for the logit:');
disp(Pr_w_logit);
Pr_w_MaksLogits=[Pr_w_masks,Pr_w_logit];

end