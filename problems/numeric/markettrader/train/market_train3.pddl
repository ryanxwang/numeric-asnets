(define (problem market_train3)
    (:domain Trader)
    (:objects
        Canberra Sydney - market
        camel0 - camel
        Vegemite TimTams MeatPie Roo - goods
    )

    (:init
        (= (price Vegemite Canberra) 50)
        (= (on-sale Vegemite Canberra) 94)
        (= (price TimTams Canberra) 20)
        (= (on-sale TimTams Canberra) 5)
        (= (price MeatPie Canberra) 30)
        (= (on-sale MeatPie Canberra) 0)
        (= (price Roo Canberra) 5)
        (= (on-sale Roo Canberra) 1)
        (= (price Vegemite Sydney) 20)
        (= (on-sale Vegemite Sydney) 2)
        (= (price TimTams Sydney) 70)
        (= (on-sale TimTams Sydney) 0)
        (= (price MeatPie Sydney) 100)
        (= (on-sale MeatPie Sydney) 3)
        (= (price Roo Sydney) 10)
        (= (on-sale Roo Sydney) 5)
        (= (bought Vegemite) 0)
        (= (bought TimTams) 0)
        (= (bought MeatPie) 0)
        (= (bought Roo) 0)
        (= (drive-cost Canberra Sydney) 2)
        (= (drive-cost Sydney Canberra) 2)
        (can-drive Canberra Sydney)
        (can-drive Sydney Canberra)
        (= (drive-cost Canberra Canberra) 0)
        (= (drive-cost Sydney Sydney) 0)
        (= (cash) 20)
        (= (capacity) 4)
        (at camel0 Canberra)
    )

    (:goal (and
        (>= (cash) 100)
    ))
)
