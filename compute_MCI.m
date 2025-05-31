function p=compute_MCI(X_scaled)
X = X_scaled;   
i=1;
% Perform K-Means Clustering
K = 3;  % Number of clusters
b=1:0.2:5;
for j=b
[idx, ~, ~] = weighted_Kmeans(X, K, j, 100);
% Compute Silhouette Scores
sil_scores = silhouette(X, idx, 'Euclidean');  % Using Euclidean distance
% Display Average Silhouette Score
avg_silhouette(i) = mean(sil_scores);
[~,id]=max(avg_silhouette);
i=i+1;
end
p=b(id);
end