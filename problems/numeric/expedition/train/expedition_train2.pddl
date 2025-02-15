(define (problem expedition_train2)

    (:domain expedition)

    (:objects
        s0 s1 - sled
        wa0 wa1 wa2 wa3 wa4 wa5 wb0 wb1 wb2 wb3 wb4 wb5 - waypoint
    )

    (:init
        (at s0 wa0)
        (= (sled_capacity s0) 5)
        (= (sled_supplies s0) 1)
        (= (waypoint_supplies wa0) 1000)
        (= (waypoint_supplies wa1) 0)
        (= (waypoint_supplies wa2) 0)
        (= (waypoint_supplies wa3) 0)
        (= (waypoint_supplies wa4) 0)
        (= (waypoint_supplies wa5) 0)
        (is_next wa0 wa1)
        (is_next wa1 wa2)
        (is_next wa2 wa3)
        (is_next wa3 wa4)
        (is_next wa4 wa5)
        (at s1 wb0)
        (= (sled_capacity s1) 5)
        (= (sled_supplies s1) 1)
        (= (waypoint_supplies wb0) 1000)
        (= (waypoint_supplies wb1) 0)
        (= (waypoint_supplies wb2) 0)
        (= (waypoint_supplies wb3) 0)
        (= (waypoint_supplies wb4) 0)
        (= (waypoint_supplies wb5) 0)
        (is_next wb0 wb1)
        (is_next wb1 wb2)
        (is_next wb2 wb3)
        (is_next wb3 wb4)
        (is_next wb4 wb5)
    )

    (:goal
        (and
            (at s0 wa5)
            (at s1 wb5)
        )
    )
)


