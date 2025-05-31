function [idx, centroids, weights] = weighted_Kmeans(X, K, b, max_iters)


    % integrated_iKM_Minkowski: Intelligent K-Means with Minkowski distance and weight updates
    % X: Data matrix (n x d), where n is the number of samples, d is the number of dimensions
    % K: Number of clusters
    % b: Minkowski metric parameter
    % max_iters: Maximum number of iterations for the entire process
    % idx: Cluster labels for each data point
    % centroids: Final cluster centroids
    % weights: Final weights for each feature

    % Step 1: Intelligent K-Means initialization
    ana_centroids = iKMeans(X, K);
    centroids=ana_centroids;
    % Initialize weights (equal weights at the start)
    [~, d] = size(X);  
    weights = ones(1, d) / d;  % Start with equal weights for all features
   
    % Step 2: Perform iterative clustering with Minkowski distance and weight updates
    for iter = 1:max_iters
        % Step 2.1: Perform K-Means using Minkowski distance and current weights
        [idx, centroids,weights,dist] = minkowski_kmeans(X, K, weights, b, max_iters,centroids);
        
        % Step 2.2: Update the weights based on current centroids and clusters
       
        
        % Display progress
        
        
        % Optional: Break if weights or centroids converge
    end
end

% --- Helper Functions ---

function [anomalous_centroids] = iKMeans(X, K)
    % iKMeans: Intelligent K-Means clustering initialization
    [~, d] = size(X);  % Number of data points and dimensions
    reference_point = mean(X, 1);  % Step 1: Grand mean (gravity center)
    anomalous_centroids = [];
    remaining_points = X;  % Step 3: List of points not yet clustered
    anomalous_size=[];
    % Step 4: Find anomalous clusters iteratively
    while size(remaining_points, 1) > 0
        dist_to_ref = pdist2(remaining_points, reference_point, 'euclidean');
        [~, farthest_idx] = max(dist_to_ref);
        initial_centroid = remaining_points(farthest_idx, :);
        centroid = initial_centroid;
        prev_centroid = inf(1, d);
        
        % Step 6: Form anomalous cluster iteratively
        while norm(centroid - prev_centroid) > 1e-5
            prev_centroid = centroid;
            dist_to_centroid = pdist2(remaining_points, centroid, 'euclidean');
            dist_to_ref = pdist2(remaining_points, reference_point, 'euclidean');
            anomalous_cluster_idx = find(dist_to_centroid < dist_to_ref);
            anomalous_cluster = remaining_points(anomalous_cluster_idx, :);
            centroid = mean(anomalous_cluster, 1);
        end
        
        anomalous_centroids = [anomalous_centroids; centroid];
        anomalous_size=[anomalous_size;size(anomalous_cluster_idx)];
        remaining_points(anomalous_cluster_idx, :) = [];
    end
    
    % Step 7: Select largest K anomalous clusters if needed
    if size(anomalous_centroids, 1) > K
        [~, sorted_idx] = sort(anomalous_size, 'descend');
        anomalous_centroids = anomalous_centroids(sorted_idx(1:K), :);
    end

end

function [idx, centroids,weights,dist] = minkowski_kmeans(X, K, weights, b, ~,init_centroids)
    % Minkowski K-Means clustering with weighted features
    [n, ~] = size(X);
    idx = zeros(n, 1);
    centroids = init_centroids;  % Random initialization
    
   
        % Step 1: Assign each point to nearest centroid (Minkowski b-distance)
        for i = 1:n
            for k = 1:K
                dist(k) = sum(weights.^b .* abs(X(i, :) - centroids(k, :)).^b);
            end
            [~,m1]=min(dist);
             idx(i) = m1;  % Assign to nearest centroid
        end
       new_centroids=update_centroid(X,idx,K);
       weights=update_weights(X, new_centroids, idx, b);
       centroids = new_centroids;
    end


function weights = update_weights(X, centroids, idx, b)
    % Update feature weights based on Minkowski distance
    
    [~, d] = size(X);
    K = size(centroids, 1);
    weights = zeros(1, d);  % Initialize weights
    
    % Step 1: Compute Dv^b for each feature
    Dv_b = zeros(1, d);
    for v = 1:d
        for k = 1:K
            cluster_points = X(idx == k, :);
            for i = 1:size(cluster_points, 1)
                Dv_b(v) = Dv_b(v) + abs(cluster_points(i, v) - centroids(k, v))^b;
            end
        end
    end
    
    % Step 2: Update weights according to formula
    for v = 1:d
        sum_term = 0;
        for u = 1:d
            sum_term = sum_term + (Dv_b(v) / Dv_b(u))^(1 / (b - 1));
        end
        weights(v) = 1 / sum_term;
    end
end


function cent = update_centroid(g,idx,K)
for i=1:K
    Y=g(idx==i,:);
        cent(i,:)=sum(Y)/size(Y,1);
end

end
        
      

    


