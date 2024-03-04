;;Instance with 2x2x2 points
(define (problem droneprob_2_2_2) (:domain drone)
(:objects 
x0y0z0 - location
x0y0z1 - location
x0y1z0 - location
x0y1z1 - location
x1y0z0 - location
x1y0z1 - location
x1y1z0 - location
x1y1z1 - location
) 
(:init (= (x) 0) (= (y) 0) (= (z) 0)
 (= (min_x) 0)  (= (max_x) 2) 
 (= (min_y) 0)  (= (max_y) 2) 
 (= (min_z) 0)  (= (max_z) 2) 
(= (xl x0y0z0) 0)
(= (yl x0y0z0) 0)
(= (zl x0y0z0) 0)
(= (xl x0y0z1) 0)
(= (yl x0y0z1) 0)
(= (zl x0y0z1) 1)
(= (xl x0y1z0) 0)
(= (yl x0y1z0) 1)
(= (zl x0y1z0) 0)
(= (xl x0y1z1) 0)
(= (yl x0y1z1) 1)
(= (zl x0y1z1) 1)
(= (xl x1y0z0) 1)
(= (yl x1y0z0) 0)
(= (zl x1y0z0) 0)
(= (xl x1y0z1) 1)
(= (yl x1y0z1) 0)
(= (zl x1y0z1) 1)
(= (xl x1y1z0) 1)
(= (yl x1y1z0) 1)
(= (zl x1y1z0) 0)
(= (xl x1y1z1) 1)
(= (yl x1y1z1) 1)
(= (zl x1y1z1) 1)
(= (battery-level) 13)
(= (battery-level-full) 13)
)
(:goal (and 
(visited x0y0z0)
(visited x0y0z1)
(visited x0y1z0)
(visited x0y1z1)
(visited x1y0z0)
(visited x1y0z1)
(visited x1y1z0)
(visited x1y1z1)
(= (x) 0) (= (y) 0) (= (z) 0) ))
);; end of the problem instance
