(define (problem hydropower_train1)
    (:domain hydropower)
    (:objects
        n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 n16 n17 n18 n19 n20 n21 n22 n23 n24 n25 n26 - turnvalue
        t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 - time
    )

    (:init
        (= (value n0) 0)
        (= (value n1) 1)
        (= (value n2) 2)
        (= (value n3) 3)
        (= (value n4) 4)
        (= (value n5) 5)
        (= (value n6) 6)
        (= (value n7) 7)
        (= (value n8) 8)
        (= (value n9) 9)
        (= (value n10) 10)
        (= (value n11) 11)
        (= (value n12) 12)
        (= (value n13) 13)
        (= (value n14) 14)
        (= (value n15) 15)
        (= (value n16) 16)
        (= (value n17) 17)
        (= (value n18) 18)
        (= (value n19) 19)
        (= (value n20) 20)
        (= (value n21) 21)
        (= (value n22) 22)
        (= (value n23) 23)
        (= (value n24) 24)
        (= (value n25) 25)
        (= (value n26) 26)
        
        (demand t0 n3)
        (demand t1 n4)
        (demand t2 n5)
        (demand t3 n9)
        (demand t4 n13)
        (demand t5 n18)
        (demand t6 n19)
        (demand t7 n19)
        (demand t8 n19)
        (demand t9 n19)
        
        (timenow t0)
        (before t0 t1)
        (before t1 t2)
        (before t2 t3)
        (before t3 t4)
        (before t4 t5)
        (before t5 t6)
        (before t6 t7)
        (before t7 t8)
        (before t8 t9)
        
        (= (stored_units) 0)
        (= (stored_capacity) 1)
        (= (funds) 1000)
    )

    (:goal (and
        (>= (funds) 1010)
    ))
)