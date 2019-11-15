function [V] = compute_V_frames(q)
    v2 = [-q(2); q(1); -q(4); q(3)];
    v3 = [-q(3); q(4); q(1); -q(2)];
    v4 = [-q(4); -q(3); q(2); q(1)];
    V = [v2 v3 v4];
end