(define (problem market_train1)
    (:domain Trader)
    (:objects
        Canberra Sydney - market
        camel0 - camel
        Vegemite TimTams MeatPie - goods
    )

    (:init
        (= (price Vegemite Canberra) 50)
        (= (on-sale Vegemite Canberra) 100)
        (= (price TimTams Canberra) 20)
        (= (on-sale TimTams Canberra) 100)
        (= (price MeatPie Canberra) 30)
        (= (on-sale MeatPie Canberra) 100)
        (= (price Vegemite Sydney) 20)
        (= (on-sale Vegemite Sydney) 100)
        (= (price TimTams Sydney) 70)
        (= (on-sale TimTams Sydney) 100)
        (= (price MeatPie Sydney) 100)
        (= (on-sale MeatPie Sydney) 100)
        (= (bought Vegemite) 0)
        (= (bought TimTams) 0)
        (= (bought MeatPie) 0)
        (= (drive-cost Canberra Sydney) 10)
        (= (drive-cost Sydney Canberra) 10)
        (can-drive Canberra Sydney)
        (can-drive Sydney Canberra)
        (= (drive-cost Canberra Canberra) 0)
        (= (drive-cost Sydney Sydney) 0)
        (= (cash) 500)
        (= (capacity) 20)
        (at camel0 Canberra)
    )

    (:goal (and
        (>= (cash) 800)
    ))
)
