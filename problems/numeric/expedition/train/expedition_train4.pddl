(define (problem expedition_train4)

    (:domain expedition)

    (:objects
        s0 s1 - sled
        wa0 wa1 wa2 wa3 wa4 - waypoint
    )

    (:init
        (at s0 wa0)
        (= (sled_capacity s0) 4)
        (= (sled_supplies s0) 1)
        (at s1 wa0)
        (= (sled_capacity s1) 4)
        (= (sled_supplies s1) 1)
        (= (waypoint_supplies wa0) 1000)
        (= (waypoint_supplies wa1) 0)
        (= (waypoint_supplies wa2) 0)
        (= (waypoint_supplies wa3) 0)
        (= (waypoint_supplies wa4) 0)
        (is_next wa0 wa1)
        (is_next wa1 wa2)
        (is_next wa2 wa3)
        (is_next wa3 wa4)
    )

    (:goal
        (and
            (at s0 wa4)
            (at s1 wa4)
        )
    )
)


