(define (problem instance_4_3_2_1)
(:domain ext-plant-watering)
(:objects
	plant1 - plant
	plant2 - plant
	plant3 - plant
	tap1 - tap
	agent1 - agent
	agent2 - agent
)
(:init
	(= (maxx) 4)
	(= (minx) 1)
	(= (maxy) 4)
	(= (miny) 1)
	(= (total_poured) 0)
	(= (total_loaded) 0)
	(= (water_reserve) 5)
	(= (carrying agent1) 0)
	(= (max_carry agent1) 5)
	(= (carrying agent2) 0)
	(= (max_carry agent2) 5)
	(= (poured plant1) 0)
	(= (poured plant2) 0)
	(= (poured plant3) 0)
	(= (x plant1) 3)
	(= (y plant1) 1)
	(= (x plant2) 3)
	(= (y plant2) 3)
	(= (x plant3) 2)
	(= (y plant3) 1)
	(= (x tap1) 3)
	(= (y tap1) 2)
	(= (x agent1) 1)
	(= (y agent1) 2)
	(= (x agent2) 2)
	(= (y agent2) 2)
)
(:goal
(and
	(= (poured plant1) 3)
	(= (poured plant2) 1)
	(= (poured plant3) 1)
	(= (total_poured) (total_loaded))
)))
